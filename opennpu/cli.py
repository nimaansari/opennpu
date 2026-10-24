"""opennpu command line. Every model runs on OpenNPU's own NPU engines.

  opennpu run MODEL [-p PROMPT] [--think]  chat (REPL without -p) through the local server
  opennpu list                             installed models, by engine
  opennpu pull HF_REPO                     download a model (checks NPU support first)
  opennpu start | stop | status | logs     the background server (127.0.0.1:11540, web UI at /)
  opennpu serve [--host H] [--port P]      run the server in the foreground
  opennpu doctor [--json]                  check driver / XRT / NPU / environment, with fixes
  opennpu embed MODEL TEXT...              sentence embeddings (universal NPU path)
  opennpu transcribe MODEL AUDIO           speech to text with a Whisper checkpoint
  opennpu bench MODEL                      NPU vs CPU speed and output parity for any model
  opennpu image|video|speak ...            image, video and speech generation
  opennpu native MODEL [-p PROMPT]         the fused NPU engine directly, without the server

Engines (picked automatically by model): OpenNPU fused (dense Qwen3 / Qwen2.5 / Mistral-Nemo,
own MLIR-AIR kernels), OpenNPU MoE (Qwen3-30B-A3B family), OpenNPU universal (any Hugging Face /
GGUF / ONNX / timm / torch model: linear layers on the NPU), and diffusers / Whisper / TTS models
for generation.
"""
import argparse
import copy
import os
import sys
import time


def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get("OPENNPU_CPU_THREADS", "4")))
    return torch


def load_causal_lm(model_id, cpu_only=False):
    _torch()
    from .loaders import load
    from .offload import offload
    model, tok, _ = load(model_id, task="generate", dtype="auto")
    if not cpu_only:
        print(offload(model), file=sys.stderr)
    return tok, model


def load_encoder(model_id, cpu_only=False):
    _torch()
    from .loaders import load
    from .offload import offload
    model, tok, _ = load(model_id, task="embed")
    if not cpu_only:
        print(offload(model), file=sys.stderr)
    return tok, model


def embed(tok, model, texts):
    torch = _torch()
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(v, dim=-1)


def chat(tok, model, messages, max_new_tokens=256, temperature=0.0):
    torch = _torch()
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        kw["temperature"] = temperature
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    dt = time.perf_counter() - t
    n_in = enc["input_ids"].shape[1]
    text = tok.decode(out[0, n_in:], skip_special_tokens=True)
    return text, n_in, out.shape[1] - n_in, dt


TIER2_QUANT = {"int8": "int8", "bf16": "accurate", "bfp16": "fast"}


def cmd_run(a):
    if not a.local:
        from .native import npu_cli            # through the server: the right engine, sessions, cache
        return npu_cli.cmd_chat([a.model] + (["-p", a.prompt] if a.prompt else []) + (["--think"] if a.think else []))
    if a.quant in TIER2_QUANT:
        os.environ["OPENNPU_PRECISION"] = TIER2_QUANT[a.quant]
    print(f"[opennpu] universal NPU path in this process ({os.environ.get('OPENNPU_PRECISION', 'flm')})", file=sys.stderr)
    tok, model = load_causal_lm(a.model, a.cpu)
    history = []
    prompts = [a.prompt] if a.prompt else None
    while True:
        p = prompts.pop(0) if prompts else (None if a.prompt else input(">>> "))
        if not p:
            break
        history.append({"role": "user", "content": p})
        text, n_in, n_out, dt = chat(tok, model, history, a.max_tokens)
        history.append({"role": "assistant", "content": text})
        print(text)
        print(f"[{n_in} prompt tok, {n_out} new tok, {dt:.2f}s, {n_out/dt:.1f} tok/s]", file=sys.stderr)


def cmd_embed(a):
    tok, model = load_encoder(a.model, a.cpu)
    v = embed(tok, model, a.text)
    for t, row in zip(a.text, v):
        print(f"{t[:40]!r:44s} dim={row.numel()} [{', '.join(f'{x:.4f}' for x in row[:6].tolist())}, ...]")
    if len(a.text) > 1:
        print("cosine sim matrix:\n", (v @ v.T).numpy().round(3))


def cmd_transcribe(a):
    torch = _torch()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from .offload import offload
    import librosa
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).eval()
    if not a.cpu:
        print(offload(model), file=sys.stderr)
    audio, _ = librosa.load(a.audio, sr=16000)
    t = time.perf_counter()
    texts = []
    for s in range(0, len(audio), 30 * 16000):
        feats = proc(audio[s:s + 30 * 16000], sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats)
        texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    print(" ".join(texts))
    print(f"[{len(audio)/16000:.1f}s audio in {time.perf_counter()-t:.1f}s]", file=sys.stderr)


def cmd_bench(a):
    if a.quant in TIER2_QUANT:
        os.environ["OPENNPU_PRECISION"] = TIER2_QUANT[a.quant]
    torch = _torch()
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    from .offload import offload, parity
    cfg = AutoConfig.from_pretrained(a.model)
    is_lm = any(x.endswith("ForCausalLM") or x.endswith("LMHeadModel") for x in (cfg.architectures or []))
    cls = AutoModelForCausalLM if is_lm else AutoModel
    tok = AutoTokenizer.from_pretrained(a.model)
    ref = cls.from_pretrained(a.model, dtype=torch.float32).eval()
    npu = copy.deepcopy(ref)
    print(offload(npu))
    text = ("The quick brown fox jumps over the lazy dog while the NPU does the heavy lifting. " * 64)
    for n in a.tokens:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=n)
        with torch.no_grad():
            rel, cos, rt, yt = parity(lambda: ref(**ids), lambda: npu(**ids))
        print(f"  {ids['input_ids'].shape[1]:5d} tokens: cos={cos:.4f} rel={rel:.4f}  CPU {rt:8.1f} ms  NPU {yt:8.1f} ms  ({rt/yt:.2f}x)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opennpu", description="Run local AI models on AMD Ryzen AI NPUs under Linux.")
    ap.add_argument("--precision", choices=["flm", "fast", "accurate", "int8"],
                    help="universal path GEMM backend (default: IRON flm.GEMM when available)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").add_argument("--json", action="store_true")
    sub.add_parser("list")
    pl = sub.add_parser("pull"); pl.add_argument("repo")
    for c in ("start", "stop", "restart", "status", "logs"):
        sub.add_parser(c)
    r = sub.add_parser("run"); r.add_argument("model"); r.add_argument("-p", "--prompt")
    r.add_argument("--think", action="store_true", help="let reasoning models think first")
    r.add_argument("--local", action="store_true", help="run in this process on the universal path, without the server")
    r.add_argument("--max-tokens", type=int, default=256); r.add_argument("--cpu", action="store_true")
    r.add_argument("--quant", choices=sorted(TIER2_QUANT), help="universal path with --local: int8 / bf16 / bfp16")
    e = sub.add_parser("embed"); e.add_argument("model"); e.add_argument("text", nargs="+"); e.add_argument("--cpu", action="store_true")
    t = sub.add_parser("transcribe"); t.add_argument("model"); t.add_argument("audio"); t.add_argument("--cpu", action="store_true")
    b = sub.add_parser("bench"); b.add_argument("model"); b.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 512])
    b.add_argument("--quant", choices=sorted(TIER2_QUANT))
    nv = sub.add_parser("native"); nv.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=11540)
    from .gen.cli import add_parsers as _gen_parsers
    _gen_parsers(sub)
    a = ap.parse_args(argv)
    if a.precision:
        os.environ["OPENNPU_PRECISION"] = a.precision
    if a.cmd == "doctor":
        from .doctor import main as doctor
        return doctor(["--json"] if a.json else [])
    if a.cmd in ("list", "pull", "start", "stop", "restart", "status", "logs"):
        from .native import npu_cli
        sys.argv = ["npu", a.cmd] + ([a.repo] if a.cmd == "pull" else [])
        return npu_cli.main()
    if a.cmd == "native":
        # The fused engine lives in the engine environment (pyxrt + mlir_air); hand over to it.
        import shlex
        import subprocess
        from . import config
        env_sh = str(config.AIRENV)
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = (f"set +u; source {shlex.quote(env_sh)} >/dev/null; ulimit -c 0; cd {shlex.quote(here)}; "
               f"exec python3 -m opennpu.native.chat {' '.join(shlex.quote(x) for x in a.rest)}")
        return subprocess.call(["bash", "-c", cmd])
    if a.cmd == "serve":
        from .native import server
        sys.argv = ["opennpu-server", "--host", a.host, "--port", str(a.port)]
        return server.main()
    if a.cmd in ("image", "video", "speak", "gen"):
        from .gen.cli import run as gen_run
        return gen_run(a)
    return {"run": cmd_run, "embed": cmd_embed, "transcribe": cmd_transcribe, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
