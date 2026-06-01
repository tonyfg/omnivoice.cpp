# omnivoice.cpp

Local AI text-to-speech with voice cloning and voice design, powered
by GGML. C++17 port of OmniVoice (k2-fsa/OmniVoice). 646 languages,
24 kHz mono output, runs on CPU, CUDA, ROCm, Metal, Vulkan.

## Features

- Voice cloning from a reference WAV plus its transcript
- Voice design via attribute keywords (gender, age, pitch, style,
  volume, emotion)
- Auto voice with consistent speaker identity across long inputs
- Long-form synthesis with punctuation-aware text chunking, voice
  prompt promotion, cross-fade and pydub-strict silence removal
- Bit deterministic generation in greedy mode, seedable Philox PRNG
  for stochastic sampling
- Q8_0 quantisation of the 612 M parameter Qwen3 backbone
- Two CLI tools : `omnivoice-tts` (text -> WAV) and `omnivoice-codec`
  (WAV <-> RVQ codes)
- An OpenAI-compatible TTS API server (`omnivoice-tts-server`)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

## Model conversion

Pre-converted GGUFs are available on Hugging Face :

  https://huggingface.co/Serveurperso/OmniVoice-GGUF

Drop them in `models/` and skip to the quick start. To convert from
the original checkpoint :

```
./checkpoints.sh      # hf download k2-fsa/OmniVoice -> checkpoints/
./convert.py          # 2 GGUFs in BF16 -> models/
./quantize.sh         # base LM Q8_0 (tokenizer stays at native dtype)
```

## Quick start

```
echo "Hello world." | ./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --lang English -o hello.wav
```

Voice cloning :

```
./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-wav ref.wav --ref-text ref.txt \
    --lang English -o out.wav < prompt.txt
```

## API server

The `omnivoice-tts-server` binary ships with the default build (built alongside
the CLI tools). It exposes an OpenAI-compatible TTS API on port 8000:

```
./build/omnivoice-tts-server \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf
```

Optional flags: `--port <int>` (default 8000), `--no-fa` (disable flash
attention), `--clamp-fp16` (clamp hidden states to FP16 range).

### Endpoints

**`POST /v1/audio/speech`** — synthesize audio from text.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello world.", "response_format": "wav", "speed": 1.0}' \
    -o speech.wav
```

Request body parameters (all optional unless noted):

- `input` *(string, required)* — Text to synthesize
- `model` *(string)* — Model id, `"omnivoice"` (default)
- `voice` *(string)* — Voice name (`"alloy"`, `"echo"`, `"fable"`, `"onyx"`,
  `"nova"`, `"shimmer"`); all map to the same voice today
- `response_format` *(string)* — `"json"`, `"text"`, `"srt"`, `"verbose_json"`,
  `"wav"` (default), or `"mp3"`
- `speed` *(float)* — Playback speed 0.25..4.0 (default 1.0)

Returns the requested format as the response body.

**`GET /v1/models`** — list available models.

```bash
curl http://localhost:8000/v1/models
# → {"object":"list","data":[{"id":"omnivoice","object":"model",...}]}
```

**`GET /`** — server info.

Compatible with the OpenAI TTS API contract, so existing OpenAI SDK clients
work out of the box:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="omnivoice",   # ignored by the server
)

with open("speech.wav", "wb") as f:
    response = client.audio.speech.create(
        model="omnivoice",
        voice="alloy",
        input="Hello from OmniVoice!",
    )
    f.write(response.content)
```

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

```c
#include "omnivoice.h"

struct ov_init_params iparams;
ov_init_default_params(&iparams);
iparams.model_path = "models/omnivoice-base-Q8_0.gguf";
iparams.codec_path = "models/omnivoice-tokenizer-F32.gguf";

struct ov_context * ov = ov_init(&iparams);

struct ov_tts_params params;
ov_tts_default_params(&params);
params.text = "Hello world.";
params.lang = "English";

struct ov_audio audio = { 0 };
ov_synthesize(ov, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
ov_audio_free(&audio);
ov_free(ov);
```

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build, so any regression that breaks plain C consumability fails
the build, not just an opt-in target.

For a binding-friendly shared library (libomnivoice.so / .dll / .dylib),
configure with `cmake -DOMNIVOICE_SHARED=ON ...`. The shared target
exports only the `ov_*` symbols ; every internal `pipeline_*` and
`backend_*` stays hidden inside the .so.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline, every CLI flag, the public API
reference and the validation results.

## License

MIT. See [LICENSE](LICENSE).

Upstream model : OmniVoice by Xiaomi / k2-fsa, Apache 2.0.
Audio codec : Higgs Audio v2 (`bosonai/higgs-audio-v2-tokenizer`),
Apache 2.0.
