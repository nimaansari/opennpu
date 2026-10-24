# OpenNPU

**Local AI on the NPU of AMD Ryzen AI laptops, on Linux.**

macOS has Core ML and Apple Intelligence. Windows has Copilot+ PCs, Foundry Local and the Ryzen AI
software. On Linux, the NPU in millions of Ryzen AI laptops mostly sits idle: there has been no app
for running your own models on it. OpenNPU is that app, open source, with its own NPU engines
(no closed runtime underneath).

- **Chat** with local LLMs in a web UI, a native GTK desktop app, or the terminal
- **Vision**: attach or paste an image and ask about it (any Hugging Face vision-language model)
- **Voice**: talk to it (Whisper on the NPU) and have replies read aloud
- **Images, video and speech** generation (diffusers, text-to-speech)
- **Saved conversations**, each remembering its model; several models loaded at once
- **Dashboard** of your usage and your hardware, kept only on your computer
- **OpenAI- and Ollama-compatible API**, so Open WebUI, Continue, OpenCode, the `openai` SDK and
  other apps can use the NPU as a drop-in local backend

> **Status: beta (v0.3).** Tested on one machine: a Ryzen AI 7 PRO 350 laptop (XDNA2 "Krackan"),
> Ubuntu 26.04, kernel 6.17, XRT 2.25. Other XDNA2 chips should work but are untested; reports
> welcome. See [Hardware](#hardware) and [Limits](#limits).

## Install

```bash
git clone https://github.com/nimaansari/opennpu && cd opennpu
./install.sh --check        # checks kernel, NPU driver, XRT, permissions; prints the fix for anything missing
./install.sh --native       # recommended: every NPU engine, incl. GGUF / LM Studio models via llama.cpp (~5 GB)
./install.sh --gen          # optional: + image, video and speech generation
```

`--native` adds the MLIR-AIR toolchain (NPU kernels are compiled for each model's matrix shapes on
first use, a few minutes once, then cached) and builds llama.cpp with OpenNPU's NPU backend; it
needs `sudo apt install git cmake ninja-build g++ llvm-18`. A plain `./install.sh` installs only
the chat server, apps and the universal engine (much slower for LLMs; no GGUF models).

The installer never uses sudo. It puts everything in `~/.local/share/opennpu` with a private
Python 3.12 (so distro Python upgrades can't break it), and runs the server as a user systemd
service on `127.0.0.1:11540` (local only). System prerequisites it can't install itself are printed
with the exact command, e.g. on Ubuntu:

```bash
sudo add-apt-repository ppa:amd-team/xrt && sudo apt install libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu
sudo usermod -aG render $USER        # then log out and in
```

## Use it

```bash
xdg-open http://127.0.0.1:11540              # web UI: chat, create, dashboard, models, discover, system
opennpu-desktop                              # the desktop app (also in your app launcher)
opennpu run Qwen/Qwen3-8B                    # chat in the terminal (any Hugging Face model id)
opennpu pull Qwen/Qwen3-30B-A3B-Instruct-2507
opennpu list                                 # installed models, by engine
opennpu image "a red fox in fresh snow" -o fox.png
opennpu doctor                               # what is wrong and the exact command to fix it
```

Discover (web UI and desktop app) searches Hugging Face, shows which engine would run each model and
whether it fits in RAM, and downloads it with progress and cancel; then "Chat now" opens it.

From other apps: OpenAI base URL `http://127.0.0.1:11540/v1`, or Ollama at `http://127.0.0.1:11540`.

## Engines and speed

OpenNPU picks the engine by model. All numbers measured on the Ryzen AI 7 PRO 350.

| Engine | Models | Decode speed |
|---|---|---|
| **Fused** (own kernels, MLIR-AIR) | dense Qwen3 8B / 14B, Qwen2.5 14B (and DeepSeek-R1 distills), Mistral-Nemo 12B, and their fine-tunes | 8B 11-12 tok/s, 12B 8.6, 14B 6.9 |
| **MoE** (own kernels) | Qwen3-30B-A3B family (Instruct, Coder, fine-tunes; GGUF read directly) | 10-11 tok/s |
| **4-bit NPU engine** | any other Hugging Face LLM, dense or mixture-of-experts (gpt-oss, Qwen3.5/3.6-MoE, Granite, Phi, OLMo, ...): every large matmul on the NPU in 4-bit, the model's own code for the rest | gpt-oss-20b 17-17.5 tok/s (prompts ~110 tok/s), Qwen3.6-35B-A3B 14-16 tok/s (prompts ~65-78 tok/s) |
| **llama.cpp on the NPU** | any GGUF model llama.cpp supports, including your LM Studio folder (~/.lmstudio/models): llama.cpp built with OpenNPU's ggml backend, its matrix products (dense and mixture-of-experts) on the NPU | gpt-oss-20b 15-18 tok/s (prompts 85-100 tok/s), Qwen3.6-35B-A3B 13.5-14.5 tok/s (prompts 70 tok/s), Nemotron-3-Nano 13, Gemma-4-26B-A4B 10; dense Llama-3.1-8B 11 (15 with a draft model), Qwen2.5-14B 6 (7-8.5 text, 11-14 code with a draft); small models Qwen3-0.6B ~100, Gemma-3-1B 56, Qwen3-1.7B 40 |
| **Universal** | vision-language, ONNX, timm and other PyTorch models: linear layers on the NPU | e.g. LFM2-350M 36 tok/s, Qwen3-0.6B 19-26 tok/s |

Prompts on the fused engines go through a batched NPU prefill (MLIR-AIR's fused prefill: int4
GEMMs and attention on the NPU, 128-token chunks): a 1,500-token prompt on Qwen3-8B starts
answering in about 3.6 s (one token at a time it took over two minutes); about 235 tokens take
0.8 s on 8B and 1.1-1.3 s on 12B/14B. It is built for each model on first use (under a minute)
and runs whenever it is faster than feeding the new tokens one by one.

The 4-bit engine quantizes a model straight from its checkpoint into NPU memory when it loads
(no full-size copy in RAM; gpt-oss-20b takes ~14 GB, Qwen3.6-35B-A3B ~20 GB) and runs it through a
shape-generic int4 GEMV device that streams weights at up to ~60 GB/s; prompts go through an
8-token variant of it on the same weights (~3-4x faster than one token at a time). Packed weights
are cached on disk, so a second load takes seconds. Models with linear-attention layers (Qwen3.6)
keep a checkpoint at the end of each prompt, so follow-up turns don't re-read the conversation.
The engine drives the NPU through libonpu, a small C++ library shared with the llama.cpp backend
(built by `llama/build.sh`), which also runs the per-token CPU work PyTorch would spread over many
tiny ops: linear-attention (Gated DeltaNet) steps, decode attention, the expert FFN glue; a MoE's
shared expert runs on the CPU while its routed experts run on the NPU. Weights are quantized per
32 values with a scale/offset search and least-squares refit (as llama.cpp's K-quants), the same
quantizer the llama.cpp backend uses: against plain min/max rounding it cut chat-text perplexity
of gpt-oss-20b from 111 to 33 and of Qwen3.6-35B-A3B from 7.87 to 7.16. `OPENNPU_NPU4_FORMAT=q4s`
trades a little of that (~6% perplexity) for 10% fewer bytes per token (~5-10% faster decode).

The llama.cpp backend adapts to the model. Small models (under ~1.5 GB of weights per token) keep
their many small matrices on the CPU, where they finish faster than one NPU dispatch (~75 µs), and
send only the large output layer to the NPU. Dense models of 4 GB and more with plain attention use
speculative decoding when a smaller local model of the same family is installed (for example
Llama-3.2-1B for Llama-3.1-8B, Qwen2.5-0.5B for Qwen2.5-14B/32B): the small model drafts up to six
tokens and the big one checks them in one NPU pass, which costs about 1.4x a single token.
`OPENNPU_LLAMA_SPEC=off` turns that off; `OPENNPU_LLAMA_DRAFT=<file.gguf>` picks the draft.

Outputs are checked against a CPU reference. While the NPU generates, the server's CPU use stays
near zero, so your laptop stays responsive. The universal path was verified on 46 architectures
(text, vision, audio, CLIP, ONNX): [docs/sweep-results.md](docs/sweep-results.md).

## Hardware

| NPU | Chips | Status |
|---|---|---|
| XDNA2 (aie2p) | Ryzen AI 300 "Strix Point", "Krackan Point" (e.g. Ryzen AI 7 PRO 350), "Strix Halo" | supported; tested on Krackan only |
| XDNA1 (aie2) | Ryzen 7040 / 8040 "Phoenix", "Hawk Point" | not yet (roadmap) |

Linux kernel 6.14 or newer (in-kernel `amdxdna` driver) and AMD's XRT runtime.

## Limits

- The MoE engine (Qwen3-30B-A3B) still reads your prompt one token at a time (about 8 tok/s), so
  long prompts are slow to start there. Follow-up turns reuse the conversation's cache.
- The fused engines' context is 2048 tokens today.
- **Speed is set by memory bandwidth, not by the NPU's compute.** Each generated token reads all
  of a dense model's weights from system RAM, which the NPU shares with the CPU and GPU (about
  90 GB/s on a DDR5-5600 laptop, of which the NPU reaches ~50-60 GB/s). A 4-bit 8B model (4.6 GB)
  therefore tops out near 11 tok/s, a 14B near 6 and a 27B near 3, whatever engine runs it; the
  CPU alone gets within ~20% of the NPU on these. Mixture-of-experts models (gpt-oss-20b,
  Qwen3-30B-A3B, Qwen3.6-35B-A3B) read only their active experts (~2 GB per token) and are the
  fastest way to big-model quality here (14-18 tok/s). Machines with wider memory (e.g. Strix Halo)
  should be proportionally faster. What the NPU adds is efficiency: the CPU stays free and cool.
- Models outside the fused families run on the universal path: prompt on the NPU, token-by-token
  generation on the CPU. That works for any model, but is slower for big ones.

## Privacy

Everything runs on your machine. Conversations (`sessions.db`) and the usage dashboard
(`usage.db`) are SQLite files under `~/.local/share/opennpu`. The dashboard stores metadata only
(model, tokens, speed, time), never prompts or replies; recording can be paused or cleared. The
server listens on 127.0.0.1 and refuses browser requests from other websites.

## How it works

- **Fused engines** run a whole transformer layer per NPU dispatch, with 4-bit weights resident in
  NPU-shared memory, built on AMD/Xilinx MLIR-AIR (`opennpu/native/`, patch in
  `opennpu/native/patches/`). Prompts are prefilled in batches by MLIR-AIR's fused prefill device
  on the same 4-bit weights (`opennpu/native/prefill.py`), which seeds the decoder's KV cache.
- **Universal path** swaps a model's linear layers for NPU layers backed by AMD IRON's GEMM:
  one hardware context, weights uploaded once, any matrix shape (`opennpu/engine.py`,
  `opennpu/offload.py`). In Python: `opennpu.offload(model)` on any `torch.nn.Module`.
- **Server** (`opennpu/native/server.py`) keeps a pool of loaded models (the NPU time-shares them),
  caches each conversation's prompt state, and serves the web UI, the desktop app and the APIs.

## Development

```bash
~/.local/share/opennpu/env/bin/python -m pytest tests/unit tests/qa/test_api.py   # server + API, no NPU needed
cd tests/e2e && npm ci && npx playwright test                                      # web UI in a browser
xvfb-run -a python3 -m pytest tests/qa/test_desktop_app.py                         # desktop app, headless
python3 tests/qa/live_smoke.py                                                     # real models on your NPU
```

The test plan and results are in [docs/QA.md](docs/QA.md). Contributions are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Third-party components and their licenses: [NOTICE](NOTICE).
