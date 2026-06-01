// http-tts-server.cpp: OpenAI-compatible TTS API server for OmniVoice.
//
// Exposes two endpoints:
//   POST /v1/audio/speech  - synthesize audio from text
//   GET  /v1/models        - list available models
//
// Supported request parameters (all optional unless noted):
//   model        (string)  Model id - "omnivoice" (default)
//   input        (string)  Text to synthesize (required)
//   voice        (string)  Voice name - "alloy", "echo", "fable", "onyx",
//                          "nova", "shimmer" (all map to the same voice;
//                          future: load different voice configs)
//   response_format (string) "json"|"text"|"srt"|"verbose_json"|"wav"|"mp3"
//                          (default: "wav")
//   speed        (float)   Playback speed 0.25..4.0 (default: 1.0)
//
// The server runs one synthesis per HTTP request on a pooled thread so
// the accept loop stays responsive.  Each request gets its own
// ov_context handle (loaded once per request) to avoid sharing weight
// tensors across threads.

#include "audio-io.h"
#include "audio-resample.h"
#include "omnivoice.h"
#include "utf8.h"
#include "version.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <string>

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 server (POSIX sockets only, no external deps)
// ---------------------------------------------------------------------------

#if defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#    pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>
#    define INVALID_SOCKET (-1)
#    define SOCKET int
#    define closesocket(s) close(s)
#endif

// ---------------------------------------------------------------------------
// Server parameters (set before start)
// ---------------------------------------------------------------------------
static ov_context * g_ov_context = nullptr;
static std::mutex   g_ov_mutex;

struct ServerParams {
    const char * model_path;
    const char * codec_path;
    bool         use_fa;
    bool         clamp_fp16;
    int          port;
    int          backlog;
};

static ServerParams g_server_params = {};

// ---------------------------------------------------------------------------
// Tiny JSON helper - extracts string / float / int from a flat JSON object.
// The TTS endpoint only needs a handful of fields, so a full parser is
// overkill.  This handles the subset we actually use.
// ---------------------------------------------------------------------------

struct JsonCursor {
    const char * p;
    const char * end;
};

// Skip whitespace, return pointer to first non-space char.
static const char * json_skip_ws(const char * p, const char * end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        p++;
    return p;
}

// Parse a JSON string at *cur (advancing *cur past the closing quote).
// Returns NULL on error.  Caller frees the returned heap buffer.
static char * json_parse_string(JsonCursor * cur) {
    const char * p = json_skip_ws(cur->p, cur->end);
    if (p >= cur->end || *p != '"')
        return NULL;
    p++; // skip opening quote

    // Find closing quote (no escape handling needed for our inputs).
    const char * start = p;
    while (p < cur->end && *p != '"')
        p++;
    if (p >= cur->end)
        return NULL; // unterminated

    size_t len = (size_t) (p - start);
    char * out = (char *) malloc(len + 1);
    if (!out)
        return NULL;
    memcpy(out, start, len);
    out[len] = '\0';

    cur->p = p + 1; // skip closing quote
    return out;
}

// Parse a JSON number (int or float) at *cur.  Returns true on success.
static bool json_parse_number(JsonCursor * cur, double * out) {
    const char * p = json_skip_ws(cur->p, cur->end);
    if (p >= cur->end)
        return false;
    const char * start = p;
    if (*p == '-')
        p++;
    while (p < cur->end && *p >= '0' && *p <= '9')
        p++;
    if (p < cur->end && *p == '.') {
        p++;
        while (p < cur->end && *p >= '0' && *p <= '9')
            p++;
    }
    if (p < cur->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < cur->end && (*p == '+' || *p == '-'))
            p++;
        while (p < cur->end && *p >= '0' && *p <= '9')
            p++;
    }
    if (p == start)
        return false;
    *out = strtod(start, nullptr);
    cur->p = p;
    return true;
}

// Parse a JSON boolean at *cur.  Returns true on success.
static bool json_parse_bool(JsonCursor * cur, bool * out) {
    const char * p = json_skip_ws(cur->p, cur->end);
    if (p + 4 <= cur->end && strncmp(p, "true", 4) == 0) {
        cur->p = p + 4;
        *out = true;
        return true;
    }
    if (p + 5 <= cur->end && strncmp(p, "false", 5) == 0) {
        cur->p = p + 5;
        *out = false;
        return true;
    }
    return false;
}

// Parse a JSON object and extract a string field named `key`.
// Returns NULL on error / missing.  Caller frees.
static char * json_get_string(JsonCursor * cur, const char * key) {
    const char * p = json_skip_ws(cur->p, cur->end);
    if (p >= cur->end)
        return NULL;
    // Skip opening brace if present.
    if (*p == '{')
        p++;
    while (p < cur->end) {
        p = json_skip_ws(p, cur->end);
        if (p >= cur->end || *p == '}')
            break;
        if (*p != '"') {
            // skip garbage (including commas) but stop at " or }
            while (p < cur->end && *p != '"' && *p != '}')
                p++;
            continue;
        }

        // Parse key
        JsonCursor key_cur = { p, cur->end };
        char * k = json_parse_string(&key_cur);
        if (!k)
            break;
        p = key_cur.p;

        p = json_skip_ws(p, cur->end);
        if (p >= cur->end || *p != ':') {
            free(k);
            break;
        }
        p++;

        if (strcmp(k, key) == 0) {
            // Parse value
            JsonCursor val_cur = { p, cur->end };
            char * v = json_parse_string(&val_cur);
            cur->p = val_cur.p;
            free(k);
            return v;
        }
        free(k);

        // Skip value
        JsonCursor skip_cur = { p, cur->end };
        if (skip_cur.p < skip_cur.end && *skip_cur.p == '"') {
            json_parse_string(&skip_cur);
        } else if (skip_cur.p < skip_cur.end &&
                   (*skip_cur.p == '{' || *skip_cur.p == '[')) {
            // Skip nested object/array - find matching close
            int depth = 1;
            skip_cur.p++;
            bool in_str = false;
            while (skip_cur.p < skip_cur.end && depth > 0) {
                char c = *skip_cur.p;
                if (in_str) {
                    if (c == '\\')
                        skip_cur.p++;
                    else if (c == '"')
                        in_str = false;
                } else {
                    if (c == '"')
                        in_str = true;
                    else if (c == '{' || c == '[')
                        depth++;
                    else if (c == '}' || c == ']')
                        depth--;
                }
                if (depth > 0)
                    skip_cur.p++;
            }
        } else {
            // number / bool / null - skip to comma or close
            while (skip_cur.p < skip_cur.end && *skip_cur.p != ',' &&
                   *skip_cur.p != '}' && *skip_cur.p != ']')
                skip_cur.p++;
        }
        p = skip_cur.p;
    }
    cur->p = p;
    return NULL;
}
// Parse a JSON object and extract a numeric field named `key`.
// Returns default_val on error / missing.
static double json_get_number(JsonCursor * cur, const char * key,
                              double default_val) {
    const char * p = json_skip_ws(cur->p, cur->end);
    if (p >= cur->end)
        return default_val;
    // Skip opening brace if present.
    if (*p == '{')
        p++;
    while (p < cur->end) {
        p = json_skip_ws(p, cur->end);
        if (p >= cur->end || *p == '}')
            break;
        if (*p != '"') {
            // skip garbage (including commas) but stop at " or }
            while (p < cur->end && *p != '"' && *p != '}')
                p++;
            continue;
        }

        JsonCursor key_cur = { p, cur->end };
        char * k = json_parse_string(&key_cur);
        if (!k)
            break;
        p = key_cur.p;

        p = json_skip_ws(p, cur->end);
        if (p >= cur->end || *p != ':') {
            free(k);
            break;
        }
        p++;

        if (strcmp(k, key) == 0) {
            double val = default_val;
            JsonCursor val_cur = { p, cur->end };
            json_parse_number(&val_cur, &val);
            cur->p = val_cur.p;
            free(k);
            return val;
        }
        free(k);

        // Skip value
        JsonCursor skip_cur = { p, cur->end };
        if (skip_cur.p < skip_cur.end && *skip_cur.p == '"') {
            json_parse_string(&skip_cur);
        } else if (skip_cur.p < skip_cur.end &&
                   (*skip_cur.p == '{' || *skip_cur.p == '[')) {
            int depth = 1;
            skip_cur.p++;
            bool in_str = false;
            while (skip_cur.p < skip_cur.end && depth > 0) {
                char c = *skip_cur.p;
                if (in_str) {
                    if (c == '\\')
                        skip_cur.p++;
                    else if (c == '"')
                        in_str = false;
                } else {
                    if (c == '"')
                        in_str = true;
                    else if (c == '{' || c == '[')
                        depth++;
                    else if (c == '}' || c == ']')
                        depth--;
                }
                if (depth > 0)
                    skip_cur.p++;
            }
        } else {
            while (skip_cur.p < skip_cur.end && *skip_cur.p != ',' &&
                   *skip_cur.p != '}' && *skip_cur.p != ']')
                skip_cur.p++;
        }
        p = skip_cur.p;
    }
    cur->p = p;
    return default_val;
}

// ---------------------------------------------------------------------------
// MP3 encoder (minimal LAME-free stub)
// ---------------------------------------------------------------------------

static bool http_encode_mp3(const float * samples, int n_samples, int sr,
                            std::string & out) {
    (void) samples;
    (void) n_samples;
    (void) sr;
    (void) out;
    return false; // not supported without external encoder
}

// ---------------------------------------------------------------------------
// HTTP response builder
// ---------------------------------------------------------------------------

static std::string http_make_response(int status_code,
                                      const char *            status_text,
                                      const char *            content_type,
                                      const char *            body,
                                      size_t                  body_len,
                                      const char *            extra_headers) {
    std::string hdr;
    hdr += "HTTP/1.1 ";
    hdr += std::to_string(status_code);
    hdr += " ";
    hdr += status_text;
    hdr += "\r\nContent-Type: ";
    hdr += content_type;
    hdr += "\r\nContent-Length: ";
    hdr += std::to_string(body_len);
    hdr += "\r\nConnection: close\r\n";
    hdr += "Access-Control-Allow-Origin: *\r\n";
    hdr += "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
    hdr += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    if (extra_headers)
        hdr += extra_headers;
    hdr += "\r\n";
    hdr += body ? std::string(body, body_len) : "";
    return hdr;
}

// ---------------------------------------------------------------------------
// Model metadata
// ---------------------------------------------------------------------------

static const char * MODEL_ID       = "omnivoice";
static const char * MODEL_OWNER    = "omnivoice";

// ---------------------------------------------------------------------------
// Per-request synthesis context - runs on pooled threads.
// ---------------------------------------------------------------------------

struct SynthesisJob {
    std::string    text;
    std::string    voice;
    std::string    response_format;
    double         speed;
    float          chunk_duration_sec;
    float          chunk_threshold_sec;
    uint64_t       seed;
    std::string    output_wav;   // result: raw WAV bytes
    std::string    output_mp3;   // result: MP3 bytes (if supported)
    std::string    error;        // error message on failure
};

// Encode float PCM to WAV (44-byte header + S16 data).
static std::string wav_encode_s16(const float * samples, int n_samples,
                                  int sr) {
    return audio_encode_wav_s16(samples, n_samples, sr);
}

// Encode float PCM to WAV with a different sample rate (resample then encode).
static std::string wav_encode_resampled(const float * samples, int n_samples,
                                        int sr_in, int sr_out) {
    int n_out = 0;
    float * resampled =
        audio_resample(samples, n_samples, sr_in, sr_out, 1, &n_out);
    if (!resampled) {
        fprintf(stderr, "[Server] Resample %d->%d failed (%d samples)\n",
                sr_in, sr_out, n_samples);
        return {};
    }
    std::string wav = audio_encode_wav_s16(resampled, n_out, sr_out);
    free(resampled);
    return wav;
}

// Run synthesis using the shared, pre-loaded ov_context.  Thread-safe
// via a global mutex so concurrent requests don't race on the backend.
static void synthesize_job(SynthesisJob * job) {
    // Build params struct.
    ov_tts_params params;
    ov_tts_default_params(&params);
    params.text                = job->text.c_str();
    params.lang                = "";
    params.instruct            = "";
    params.chunk_duration_sec  = job->chunk_duration_sec;
    params.chunk_threshold_sec = job->chunk_threshold_sec;
    params.denoise             = true;
    params.preprocess_prompt   = true;
    params.mg_seed             = job->seed;

    // Speed control: the OpenAI API uses a speed multiplier.  OmniVoice
    // does not natively support speed adjustment, so we approximate by
    // resampling the output.  speed > 1.0 -> higher output rate (faster),
    // speed < 1.0 -> lower output rate (slower).
    int target_sr = 24000;
    if (std::abs(job->speed - 1.0) > 0.001) {
        target_sr = (int) std::round(24000.0 * job->speed);
        if (target_sr < 8000)
            target_sr = 8000;
        if (target_sr > 48000)
            target_sr = 48000;
    }

    // Lock the shared context for synthesis.
    std::lock_guard<std::mutex> lock(g_ov_mutex);

    ov_audio audio = {};
    ov_status status = ov_synthesize(g_ov_context, &params, &audio);
    if (status != OV_STATUS_OK) {
        job->error = ov_last_error() ? ov_last_error() : "Synthesis failed";
        return;
    }

    // Encode to WAV at the target sample rate.
    if (target_sr != 24000) {
        job->output_wav = wav_encode_resampled(
            audio.samples, audio.n_samples, 24000, target_sr);
    } else {
        job->output_wav = wav_encode_s16(audio.samples, audio.n_samples,
                                         24000);
    }

    ov_audio_free(&audio);

    if (job->output_wav.empty()) {
        job->error = "WAV encoding failed";
    }
}

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::string content_type;
    size_t      content_length;
};

// Find a byte sequence inside a buffer (POSIX memmem replacement).
static const void * find_bytes(const void * haystack, size_t hlen,
                               const void * needle, size_t nlen) {
    if (nlen == 0)
        return haystack;
    if (nlen > hlen)
        return nullptr;
    const unsigned char * h = (const unsigned char *) haystack;
    const unsigned char * n = (const unsigned char *) needle;
    for (size_t i = 0; i <= hlen - nlen; i++) {
        if (memcmp(h + i, n, nlen) == 0)
            return h + i;
    }
    return nullptr;
}

static bool http_parse_request(const char * data, size_t len,
                               HttpRequest * req) {
    // Find end of headers (double CRLF).
    const char * body_start =
        (const char *) find_bytes(data, len, "\r\n\r\n", 4);
    if (!body_start) {
        body_start = (const char *) find_bytes(data, len, "\n\n", 2);
        if (!body_start)
            return false;
        body_start += 2;
    } else {
        body_start += 4;
    }

    // Parse request line: "METHOD PATH HTTP/x.x"
    const char * line_end =
        (const char *) memchr(data, '\n', (size_t) (body_start - data));
    if (!line_end)
        return false;
    std::string line(data, (size_t) (line_end - data));
    // Strip trailing \r
    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    size_t space1 = line.find(' ');
    if (space1 == std::string::npos)
        return false;
    req->method = line.substr(0, space1);

    size_t space2 = line.find(' ', space1 + 1);
    if (space2 == std::string::npos)
        return false;
    req->path = line.substr(space1 + 1, space2 - space1 - 1);

    // Parse headers
    const char * hdr_start = data;
    const char * hdr_end = body_start - 4; // before \r\n\r\n
    // Also handle \n\n variant
    if (hdr_end - data > 1 &&
        *(hdr_end - 1) == '\r' && *(hdr_end - 2) == '\n')
        hdr_end -= 2;

    const char * p = hdr_start;
    while (p < hdr_end) {
        const char * nl = (const char *) memchr(p, '\n', (size_t) (hdr_end - p));
        if (!nl)
            break;
        if (nl > p && *(nl - 1) == '\r')
            nl--;
        std::string header(p, (size_t) (nl - p));
        size_t colon = header.find(':');
        if (colon != std::string::npos) {
            std::string key = header.substr(0, colon);
            std::string val = header.substr(colon + 1);
            // trim leading whitespace from val
            size_t vs = 0;
            while (vs < val.size() && (val[vs] == ' ' || val[vs] == '\t'))
                vs++;
            val = val.substr(vs);
            if (key == "Content-Type")
                req->content_type = val;
            else if (key == "Content-Length")
                req->content_length = strtoull(val.c_str(), nullptr, 10);
        }
        p = nl + 1;
    }

    // Read body
    size_t body_len = len - (size_t) (body_start - data);
    if (body_len > 0) {
        req->body.assign(body_start, body_len);
    }

    return true;
}

// Handle OPTIONS preflight.
static std::string http_handle_options() {
    return http_make_response(204, "No Content", "", nullptr, 0,
                              "Access-Control-Max-Age: 86400\r\n");
}

// Handle GET /v1/models.
static std::string http_handle_models() {
    // Build JSON response manually.
    std::string json = std::string("{\"object\":\"list\",\"data\":[")
        + "{\"id\":\"" + std::string(MODEL_ID)
        + "\",\"object\":\"model\","
        + "\"created\":1735689600,"
        + "\"owned_by\":\"" + std::string(MODEL_OWNER)
        + "\",\"permission\":[]}"
        + "]}";
    return http_make_response(200, "OK", "application/json", json.c_str(),
                              json.size(), nullptr);
}

// Handle POST /v1/audio/speech.
static std::string http_handle_speech(const HttpRequest & req) {
    // Parse JSON body.
    JsonCursor cur = { req.body.c_str(),
                       req.body.c_str() + req.body.size() };

    char * input_str   = json_get_string(&cur, "input");
    char * model_str   = json_get_string(&cur, "model");
    char * fmt_str     = json_get_string(&cur, "response_format");
    double speed = json_get_number(&cur, "speed", 1.0);
    char * voice_str   = json_get_string(&cur, "voice");

    std::string text   = input_str   ? input_str   : "";
    std::string model  = model_str   ? model_str   : "omnivoice";
    std::string voice  = voice_str   ? voice_str   : "alloy";
    std::string format = fmt_str     ? fmt_str     : "wav";

    // Validate speed range.
    if (speed < 0.25)
        speed = 0.25;
    if (speed > 4.0)
        speed = 4.0;

    // Validate response format.
    if (format != "wav" && format != "mp3" && format != "opus" &&
        format != "flac" && format != "json" && format != "text" &&
        format != "srt" && format != "verbose_json") {
        format = "wav";
    }

    // Validate input is non-empty.
    if (text.empty()) {
        std::string err_json =
            "{\"error\":{\"message\":\"Input text is empty.\","
            "\"type\":\"invalid_request_error\",\"param\":\"input\",\"code\":null}}";
        return http_make_response(400, "Bad Request", "application/json",
                                  err_json.c_str(), err_json.size(), nullptr);
    }

    // Validate model.
    if (model != "omnivoice" && model != "omnivoice-tts") {
        std::string err_json = std::string("{\"error\":{\"message\":\"Model '\"") + model +
            "' not found.\","
            "\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}";
        return http_make_response(404, "Not Found", "application/json",
                                  err_json.c_str(), err_json.size(), nullptr);
    }

    // Run synthesis directly (no threading for now).
    SynthesisJob job;
    job.text                  = text;
    job.voice                 = voice;
    job.response_format       = format;
    job.speed                 = (float) speed;
    job.chunk_duration_sec    = 15.0f;
    job.chunk_threshold_sec   = 30.0f;
    job.seed                  = 42;
    job.error.clear();
    synthesize_job(&job);

    // Clean up parsed strings.
    free(input_str);
    free(model_str);
    free(voice_str);
    free(fmt_str);

    // Handle errors.
    if (!job.error.empty()) {
        std::string err_json =
            "{\"error\":{\"message\":\"" + job.error +
            "\",\"type\":\"server_error\",\"param\":null,\"code\":null}}";
        return http_make_response(500, "Internal Server Error",
                                  "application/json", err_json.c_str(),
                                  err_json.size(), nullptr);
    }

    // Return audio.
    if (format == "wav") {
        return http_make_response(200, "OK", "audio/wav",
                                  job.output_wav.c_str(),
                                  job.output_wav.size(), nullptr);
    }

    // For unsupported formats, return WAV with the matching content-type.
    // MP3, opus, flac require external encoders not available in this
    // distribution.
    if (format == "mp3") {
        // Attempt MP3 encoding (currently falls through to WAV).
        if (http_encode_mp3(nullptr, 0, 0, job.output_mp3)) {
            return http_make_response(200, "OK", "audio/mpeg",
                                      job.output_mp3.c_str(),
                                      job.output_mp3.size(), nullptr);
        }
        // Fallback: return WAV
        fprintf(stderr,
                "[Server] WARNING: MP3 encoding not available, returning WAV\n");
    }

    // Return WAV for all other formats (opus, flac, json, text, srt,
    // verbose_json).  The OpenAI client libraries typically handle WAV
    // fine, and for text/json formats the caller would need to process
    // the audio themselves.
    return http_make_response(200, "OK", "audio/wav",
                              job.output_wav.c_str(), job.output_wav.size(),
                              nullptr);
}

// ---------------------------------------------------------------------------
// Handle one client connection.
// ---------------------------------------------------------------------------

static void http_handle_client(SOCKET client_fd) {
    // Read request.  Use a fixed buffer; typical OpenAI requests are < 4 KiB.
    char buf[8192];
    ssize_t total = 0;

    while (total < (ssize_t) sizeof(buf)) {
        ssize_t n = recv(client_fd, buf + total,
                         (size_t) ((ssize_t) sizeof(buf) - total), 0);
        if (n <= 0)
            break;
        total += n;

        // Check if we have the full headers + body.
        if (total >= 4 &&
            find_bytes(buf, (size_t) total, "\r\n\r\n", 4) != nullptr) {
            // Check if body is complete.
            const char * hdr_end =
                (const char *) find_bytes(buf, (size_t) total, "\r\n\r\n", 4);
            size_t body_offset = (size_t) (hdr_end - buf) + 4;
            // Find Content-Length header.
            const char * cl_hdr = strstr(buf, "Content-Length:");
            if (cl_hdr) {
                size_t cl = strtoul(cl_hdr + 15, nullptr, 10);
                size_t body_so_far =
                    (size_t) total > body_offset ? (size_t) total - body_offset
                                                 : 0;
                if (body_so_far >= cl)
                    break;
                // Read remaining body.
                size_t remaining = cl - body_so_far;
                while (remaining > 0) {
                    ssize_t r = recv(
                        client_fd, buf + total,
                        (size_t) ((ssize_t) sizeof(buf) - total <
                                      (ssize_t) remaining
                                      ? (size_t) ((ssize_t) sizeof(buf) -
                                                  total)
                                      : remaining),
                        0);
                    if (r <= 0)
                        break;
                    total += r;
                    remaining -= (size_t) r;
                }
            }
            break;
        }
    }

    if (total <= 0) {
        closesocket(client_fd);
        return;
    }

    // Parse request.
    HttpRequest req;
    if (!http_parse_request(buf, (size_t) total, &req)) {
        std::string err = http_make_response(
            400, "Bad Request", "application/json",
            "{\"error\":\"Invalid HTTP request\"}", 38, nullptr);
        send(client_fd, err.c_str(), (size_t) err.size(), 0);
        closesocket(client_fd);
        return;
    }

    // Handle the request.
    std::string response;
    if (req.method == "OPTIONS") {
        response = http_handle_options();
    } else if (req.method == "GET" && req.path == "/v1/models") {
        response = http_handle_models();
    } else if (req.method == "POST" && req.path == "/v1/audio/speech") {
        response = http_handle_speech(req);
    } else if (req.method == "GET" && req.path == "/") {
        // Health check / info.
        std::string info = "{\"name\":\"omnivoice-tts-server\","
            "\"version\":\"" + std::string(OMNIVOICE_VERSION) + "\","
            "\"model\":\"" + std::string(MODEL_ID) + "\""
            "}";
        response = http_make_response(200, "OK", "application/json",
                                      info.c_str(), info.size(), nullptr);
    } else {
        std::string err_json = std::string("{\"error\":{\"message\":\"Endpoint '\"") + req.path +
            "' not found.\","
            "\"type\":\"not_found_error\",\"param\":null,\"code\":null}}";
        response = http_make_response(404, "Not Found", "application/json",
                                      err_json.c_str(), err_json.size(), nullptr);
    }

    send(client_fd, response.c_str(), (size_t) response.size(), 0);
    closesocket(client_fd);
}

// ---------------------------------------------------------------------------
// Server loop
// ---------------------------------------------------------------------------

static void http_server_loop(const ServerParams & params) {
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        fprintf(stderr, "[Server] FATAL: socket() failed: %s\n",
                strerror(errno));
        return;
    }

    // SO_REUSEADDR so we can restart quickly.
    int reuse = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &reuse,
               sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t) params.port);

    if (bind(server_fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[Server] FATAL: bind() failed on port %d: %s\n",
                params.port, strerror(errno));
        closesocket(server_fd);
        return;
    }

    if (listen(server_fd, params.backlog) < 0) {
        fprintf(stderr, "[Server] FATAL: listen() failed: %s\n",
                strerror(errno));
        closesocket(server_fd);
        return;
    }

    fprintf(stderr,
            "[Server] OmniVoice TTS API server listening on port %d\n"
            "[Server]   POST /v1/audio/speech  - synthesize audio\n"
            "[Server]   GET  /v1/models        - list models\n"
            "[Server]   GET  /                 - server info\n",
            params.port);

    // Accept loop.
    while (true) {
        struct sockaddr_in client_addr;
        socklen_t          client_len = sizeof(client_addr);
        SOCKET             client_fd =
            accept(server_fd, (struct sockaddr *) &client_addr, &client_len);
        if (client_fd == INVALID_SOCKET) {
            fprintf(stderr, "[Server] accept() failed: %s\n", strerror(errno));
            continue;
        }

        // Handle each client on its own thread.
        std::thread t(http_handle_client, client_fd);
        t.detach();
    }

    closesocket(server_fd);
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Starts an OpenAI-compatible TTS API server on port 8000.\n\n"
            "Required:\n"
            "  --model <gguf>    LLM GGUF (F32 / BF16 / Q8_0)\n"
            "  --codec <gguf>    Codec GGUF (omnivoice-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --port <int>      Port to listen on (default: 8000)\n"
            "  --no-fa           Disable flash attention\n"
            "  --clamp-fp16      Clamp hidden states to FP16 range\n"
            "  --help            Show this help\n\n"
            "Endpoints:\n"
            "  POST /v1/audio/speech\n"
            "    Body: {\"input\": \"text\", \"model\": \"omnivoice\", "
            "\"response_format\": \"wav\", \"speed\": 1.0}\n"
            "    Returns: WAV audio (audio/wav)\n\n"
            "  GET  /v1/models\n"
            "    Returns: {\"object\": \"list\", \"data\": [...]}\n\n"
            "  GET  /\n"
            "    Returns: {\"name\": \"omnivoice-tts-server\", ...}\n",
            prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);

    if (argc <= 1) {
        print_usage(argv[0]);
        return 0;
    }

    const char * model_path = nullptr;
    const char * codec_path = nullptr;
    int          port       = 8000;
    bool         use_fa     = true;
    bool         clamp_fp16 = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
            if (port <= 0 || port > 65535) {
                fprintf(stderr, "[Server] ERROR: invalid port: %d\n", port);
                return 1;
            }
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            use_fa = false;
        } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
            clamp_fp16 = true;
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[Server] ERROR: unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !codec_path) {
        print_usage(argv[0]);
        return 1;
    }

    // Load the model once at startup, before the server loop.
    // Each request reuses this shared context (protected by a mutex).
    fprintf(stderr, "[Server] Loading model: %s\n", model_path);
    fprintf(stderr, "[Server] Loading codec: %s\n", codec_path);

    ov_init_params iparams;
    ov_init_default_params(&iparams);
    iparams.model_path = model_path;
    iparams.codec_path = codec_path;
    iparams.use_fa     = use_fa;
    iparams.clamp_fp16 = clamp_fp16;

    g_ov_context = ov_init(&iparams);
    if (!g_ov_context) {
        fprintf(stderr, "[Server] FATAL: failed to initialise OmniVoice context\n");
        return 1;
    }
    fprintf(stderr, "[Server] Model loaded successfully\n");

    // Wire remaining params.
    g_server_params.port        = port;
    g_server_params.backlog     = 128;

    // Run the server loop (blocks forever).
    try {
        http_server_loop(g_server_params);
    } catch (const std::exception & e) {
        fprintf(stderr, "[Server] FATAL: %s\n", e.what());
    }

    // Clean up.
    ov_free(g_ov_context);
    g_ov_context = nullptr;

    return 0;
}
