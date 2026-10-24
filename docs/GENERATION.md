# Images, video and voice on the NPU

OpenNPU runs generation models the same way it runs everything else: the heavy network (the
UNet or diffusion transformer of an image/video model, or the TTS model) goes through
`opennpu.offload`, so its linear and convolution layers run on the Ryzen AI NPU and the rest
stays on the CPU. Any Hugging Face **diffusers** text-to-image / text-to-video pipeline and any
**transformers** text-to-speech model works; the catalog below is what has been run end to end.

## Install

```bash
./install.sh --gen          # adds diffusers, ffmpeg (bundled) and soundfile, about 250 MB
```

## Use it

From a terminal:

```bash
opennpu gen list                                   # curated models, RAM estimate, notes
opennpu image "a red fox in fresh snow" -o fox.png              # SD-Turbo, 1 step
opennpu image "a lighthouse at dusk, oil painting" -m sd-1.5 --steps 25 --seed 7
opennpu image "..." -m image:<any/diffusers-repo>               # any other model
opennpu video "a fox walking through snow" -m wan-1.3b -o fox.mp4
opennpu speak "Hello from the NPU." -o hello.wav                 # MMS-TTS English
opennpu speak "..." -m speech:facebook/mms-tts-deu               # 1100+ languages
```

From the web UI: open `http://127.0.0.1:11540` and pick **Create**.

From any OpenAI client (the server is at `http://127.0.0.1:11540/v1`):

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:11540/v1", api_key="local")
img = c.images.generate(model="sd-turbo", prompt="a red fox in snow", response_format="b64_json")
c.audio.speech.create(model="tts-1", voice="alloy", input="Hello").write_to_file("hello.mp3")
```

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/images/generations` | `model, prompt, n, size ("512x512"), steps, guidance, seed, negative_prompt` | `data[].b64_json` (PNG) |
| `POST /v1/audio/speech` | `model, input, voice, response_format (wav, mp3, opus, aac, flac, pcm)` | audio bytes |
| `POST /v1/videos/generations` | `model, prompt, size, frames, fps, steps, seed` | `data[0].b64_json` (MP4) |
| `POST /v1/audio/transcriptions` | multipart `file` (webm, ogg, wav, mp3, m4a, flac), `model`, `language`, `prompt`, `response_format` (json, verbose_json, text); or JSON `{audio_b64, format, ...}` | `{text}` (+ `language`, `duration`) |
| `GET /v1/gen/progress` | | `{task, model, phase, step, steps, started, elapsed_s}` |
| `POST /v1/gen/cancel` | | `{cancelled}`; the running request answers 499 |
| `GET /v1/gen/models` | | catalog with task, RAM estimate, downloaded |

OpenAI model and voice names (`tts-1`, `alloy`, ...) map to the default local voice, so
existing clients work unchanged.

## Memory

Laptops share RAM between the desktop, the chat model and generation, and running out freezes
the machine. So before anything loads, OpenNPU reads the parameter count from the weight file
headers and compares it with the RAM that is free right now:

- if the model fits with a CPU copy next to the NPU buffers, layers are placed per layer by
  timing both (`placement=auto`);
- if it only fits without the CPU copy, every offloaded layer stays on the NPU (`placement=npu`);
- otherwise the load is refused with how much RAM it needs and what to close.

Only the denoiser gets NPU buffers; text encoders and the VAE load once in bf16. A generation
model and the chat model never hold RAM at the same time in the server: loading one unloads
the other. Downloads are checked against free disk space first and fetch only the files the
pipeline uses (fp16 variants where the repo has them).

## Speed

The first images of a model are slower while OpenNPU times each layer on the CPU and the NPU
and keeps the faster one; the decisions are saved (under `~/.cache/opennpu/gen-placement`), so
this happens once per model and resolution, not every start.

`tests/gen_sweep.py` runs every catalog model end to end (one process per model, under a
memory cap) and writes `reports/gen/results.md`. Measured on a Ryzen AI 7 PRO 350 (XDNA2,
45 GB RAM), steady state after the warm-up, every output checked by eye (images also against a
CPU render, voices by transcribing them with Whisper):

| Model | Task | Setting | Time | Share of params on the NPU |
|---|---|---|---|---|
| sd-turbo | image | 512 px, 1 step | 3.9 s (CPU only: 4.2 s) | 99.9% |
| sdxl-turbo | image | 512 px, 2 steps | 6.3 s | 99.9% |
| sana-600m | image | 512 px, 20 steps | 20 s | 99.6% |
| sd-1.5 | image | 512 px, 25 steps | 65 s (CPU only: 105 s) | 99.9% |
| sdxl | image | 1024 px, 25 steps | 5.3 min | 99.9% |
| ltx-video | video | 512x320, 25 frames, 30 steps | 2.1 min | ~100% of the transformer |
| wan-1.3b | video | 480x320, 9 frames, 12 steps | 6 min | ~100% of the transformer |
| speecht5 | voice | one sentence | 1.5 s | 98.9% |
| mms-tts-eng | voice | one sentence | 0.5 s | 0% (small convs run on the CPU) |
| bark-small | voice | 8 s of audio | 46 s | 68.6% |

Speech-to-text (Whisper, encoder and decoder layers placed per layer; 2.6 s clip, steady state):
whisper-small 2.5 s on the NPU vs 4.0 s CPU only (default; transcribes correctly), whisper-base
0.9 s vs 1.7 s (faster, more mistakes). Uploads are limited to 25 MB and 30 minutes.

`flux-schnell` (needs ~42 GB free) and `sd3.5-medium` (gated) were not run here.

## Limits

- Attention, the VAE decoder and the samplers run on the CPU today, so the NPU speed-up for
  diffusion is modest (SD-Turbo 512 px: 3.75 s per image on the NPU vs 4.2 s on the CPU).
- Small TTS models (MMS/VITS) are made of small 1-D convolutions that are faster on the CPU, so
  they run there; transformer TTS models (Bark) put their linear layers on the NPU.
- Video models need 15 to 25 GB of free RAM and several minutes per short clip.
- Gated models (SD 3.5, FLUX dev) need their licence accepted on Hugging Face and
  `hf auth login`.
