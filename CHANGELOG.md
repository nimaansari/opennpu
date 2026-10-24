# Changelog

## 0.3.0 (unreleased)

First public release.

### Engines
- **4-bit NPU engine (`npu4`)** for Hugging Face LLMs, dense and mixture-of-experts, with a C++
  core (libonpu): fused expert FFN in one NPU dispatch, an 8-token device for prompts, a growing
  bf16 KV cache read directly by decode attention, and a packed-weight disk cache. Better 4-bit
  quantizers (`q4nx` default; `q4s` speed mode), optional two-pass ~8-bit matrices.
- **llama.cpp on the NPU** (`llama/ggml-opennpu`): GGUF models from LM Studio, Hugging Face or
  `~/models` run their matrix products on the NPU; Q4_K transcodes exactly, 5-6 bit layers of
  mixture-of-experts files stay on the CPU at full precision.
  - Small models: matrices too small to be worth an NPU dispatch run on the CPU (Qwen3-0.6B
    30 -> ~100 tok/s).
  - Speculative decoding for 8B+ plain-attention models with a smaller local model of the same
    family as draft (Llama-3.1-8B 11 -> 15 tok/s; Qwen2.5-14B 6 -> 7-8.5 text, 11-14 code).
- Fused MLIR-AIR engines for dense and MoE models (`./install.sh --native`), with batched prefill.
- Universal engine for other Hugging Face models; image, video and speech generation (`--gen`).

### Apps
- Web UI, GTK desktop app and terminal CLI; saved conversations, several models loaded at once,
  vision, voice input and read-aloud, usage and hardware dashboard (local only).
- Discover: search, download and run Hugging Face models from the apps.
- OpenAI- and Ollama-compatible API on `127.0.0.1:11540`.

### Fixes before release
- The model list no longer hangs on machines with many GGUF files (fast GGUF metadata reader,
  one catalog scan at a time); GGUF model loads are ~20 s faster.
