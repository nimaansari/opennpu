"""Chat with a dense HF checkpoint on the native (own-engine) NPU path.

    python -m opennpu.native.chat <hf_dir_or_repo> [-p PROMPT] [--max-tokens N]

Resolves the checkpoint to a fused-decode geometry, builds the decode templates on first
use (~1 min, once per architecture), quantizes the weights on first use (~2-4 min,
once per checkpoint), then streams tokens. Without -p it is an interactive REPL.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from opennpu.native import specgen

from opennpu import config

TEMPLATES = Path(os.environ.get("OPENNPU_NATIVE_TEMPLATES", config.NATIVE_TEMPLATES))
HERE = Path(__file__).resolve().parent

# Geometries validated end-to-end on this NPU (name in fused_decode's _MODELS, kernel macro).
# A checkpoint matches when specgen derives exactly this engine entry from its config.
NATIVE = {
    "qwen3-8b": "QWEN3_8B",
    "qwen3-14b": "QWEN3_14B",
    "qwen2.5-14b": "QWEN2_5_14B",
    "mistral-nemo-12b": "MISTRAL_NEMO_12B",
}


def resolve(ckpt_dir):
    """config.json -> (decode model name, MODEL_TYPE macro), or raise specgen.Unsupported."""
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    want = specgen.from_hf_config(cfg, "x").driver_entry()
    import ast
    from opennpu.native.runtime import DEC
    # read the engine's model table without importing the (heavy) builder
    tree = ast.parse((DEC / "fused_decode.py").read_text())
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.Assign) and
                                  getattr(n.targets[0], "id", "") in ("_MODEL_DEFAULTS", "_MODELS")],
                            type_ignores=[]), "fd_table", "exec"), ns)
    keys = ("K", "M", "DH_A", "KV_PER_CU", "N_ATTN_CU", "I2P", "J2P", "GQA_SEG", "PAIR_ROWS", "UNI_DEC")
    pad = lambda v, k: -(-v // k) * k  # noqa: E731
    for name, macro in NATIVE.items():
        e = ns["_MODELS"][name]
        # fine-tunes often append a few tokens (e.g. ChatML); fine while the padded grid is the same
        same_vocab = pad(e["VOCAB_SIZE"], e["K"]) == pad(want["VOCAB_SIZE"], want["K"])
        if same_vocab and all(e.get(k) == want.get(k) for k in keys) and \
                bool(e.get("HAS_QK_NORM")) == bool(want.get("HAS_QK_NORM")) and \
                bool(e.get("HAS_QKV_BIAS")) == bool(want.get("HAS_QKV_BIAS")):
            return name, macro
    raise specgen.Unsupported(
        f"{cfg.get('model_type')} {cfg.get('hidden_size')}x{cfg.get('num_hidden_layers')}: no validated "
        f"native geometry (have {', '.join(NATIVE)})")


def ensure_templates(name, macro, lbuild=2048):
    d = TEMPLATES / name
    if not (d / f"decode_L{lbuild}.xclbin").exists():
        print(f"[opennpu] building {name} NPU decode kernels (one time, ~1 min) ...", file=sys.stderr)
        subprocess.run([str(HERE / "build_decode.sh"), name, macro, str(d), str(lbuild)], check=True,
                       stdout=subprocess.DEVNULL)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opennpu native")
    ap.add_argument("model")
    ap.add_argument("-p", "--prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="Qwen3: keep the thinking block")
    a = ap.parse_args(argv)

    ck = a.model
    if not os.path.isdir(ck):
        from huggingface_hub import snapshot_download
        ck = snapshot_download(ck, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    name, macro = resolve(ck)
    tdir = ensure_templates(name, macro)

    from transformers import AutoTokenizer
    from opennpu.native.runtime import NativeDecoder
    tok = AutoTokenizer.from_pretrained(ck)
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    gc = Path(ck) / "generation_config.json"
    if gc.exists():
        e = json.loads(gc.read_text()).get("eos_token_id")
        eos |= set(e if isinstance(e, list) else [e] if e is not None else [])
    dec = NativeDecoder(ck, name, tdir)

    def turn(history):
        kw = {} if a.think else {"enable_thinking": False}
        try:
            ids = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=True, **kw)
        except TypeError:
            ids = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        dec.reset()
        shown = [""]
        out = []

        def emit(t):
            out.append(t)
            s = tok.decode(out, skip_special_tokens=True)
            sys.stdout.write(s[len(shown[0]):])
            sys.stdout.flush()
            shown[0] = s

        n = min(a.max_tokens, dec.ATTN_MAXL - len(ids))
        if n <= 0:
            raise SystemExit(f"prompt is {len(ids)} tokens; context is {dec.ATTN_MAXL}")
        new, tp, td = dec.generate(ids, n, eos=eos, on_token=emit, temperature=a.temperature)
        print(f"\n[{len(ids)} prompt tok {tp:.1f}s | {len(new)} tok {len(new) / max(td, 1e-9):.2f} tok/s]",
              file=sys.stderr)
        return tok.decode(new, skip_special_tokens=True)

    if a.prompt:
        turn([{"role": "user", "content": a.prompt}])
        return
    hist = []
    while True:
        try:
            q = input("\n> ")
        except EOFError:
            break
        if q.strip() in ("/exit", "/quit"):
            break
        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": turn(hist)})


if __name__ == "__main__":
    main()
