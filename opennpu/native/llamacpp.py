"""GGUF models through llama.cpp with OpenNPU's NPU backend (llama/ggml-opennpu).

llama.cpp runs the model (every architecture it supports, GGUF read as-is); its matrix products,
dense and mixture-of-experts, run on the NPU through OpenNPU's int4 GEMV device. The rest
(attention, norms, sampling) stays on the CPU. One llama-server process per loaded model, on a
private localhost port; this module starts it, builds the GEMV shapes its matrices need first,
and talks to it.

Models: *.gguf in ~/models, GGUF repos in the Hugging Face cache, and LM Studio's model folder
(~/.lmstudio/models). A sibling mmproj-*.gguf adds image input.
"""
import base64
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from opennpu import config

LMSTUDIO = Path(os.environ.get("OPENNPU_LMSTUDIO_MODELS", Path.home() / ".lmstudio/models"))
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
THREADS = int(os.environ.get("OPENNPU_LLAMA_THREADS", "4"))
CTX = int(os.environ.get("OPENNPU_LLAMA_CTX", "8192"))
SLOTS = 4


def server_bin():
    for p in (os.environ.get("OPENNPU_LLAMA_SERVER"), config.HOME / "llama.cpp/build/bin/llama-server"):
        if p and Path(p).exists():
            return Path(p)
    return None


def available():
    return server_bin() is not None


# ------------------------------------------------------------------ discovery
def _pick(d):
    """The model file in a folder (largest non-mmproj GGUF; the first part of a split) + mmproj."""
    files = sorted(d.glob("*.gguf"))
    w = [f for f in files if not f.name.lower().startswith("mmproj")
         and ("-of-" not in f.name.lower() or "-00001-of-" in f.name.lower())]
    if not w:
        return None, None
    mm = [f for f in files if f.name.lower().startswith("mmproj")]
    return max(w, key=lambda f: f.stat().st_size), (mm[0] if mm else None)


# GGUF architectures that are not language models llama.cpp can chat with (diffusion / video
# / audio models also ship as GGUF)
NOT_LLM = {"wan", "flux", "sd3", "sdxl", "sd1", "ltxv", "hyvid", "cosmos", "hidream", "chroma", "lumina2",
           "qwen_image", "aura", "mochi", "clip", "t5", "t5encoder", "umt5", "whisper", "bark", "kokoro", "outetts"}


def gguf_models(claimed=()):
    """name -> (gguf path, mmproj path or "")."""
    out = _gguf_models(claimed)
    return {k: v for k, v in out.items() if (_arch_cached(v[0]) or "") not in NOT_LLM}


_ARCH = {}


def _arch_cached(path):
    if path not in _ARCH:
        _ARCH[path] = gguf_arch(path)
    return _ARCH[path]


def _gguf_models(claimed=()):
    out = {}
    for d in sorted(LMSTUDIO.glob("*/*")) if LMSTUDIO.is_dir() else []:
        if d.is_dir():
            m, mm = _pick(d)
            if m:
                out[f"{d.parent.name}/{d.name}"] = (str(m), str(mm or ""))
    for d in sorted(config.MODELS.iterdir()) if config.MODELS.is_dir() else []:
        if d.is_dir() and d.name not in claimed and not (d / "config.json").exists():
            m, mm = _pick(d)
            if m:
                out[d.name] = (str(m), str(mm or ""))
    for d in sorted(HF_CACHE.glob("models--*")) if HF_CACHE.is_dir() else []:
        snaps = sorted((d / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
        if snaps:
            m, mm = _pick(snaps[-1])
            if m and not any(snaps[-1].glob("*.safetensors")):
                out[d.name[len("models--"):].replace("--", "/")] = (str(m), str(mm or ""))
    return out


_GG_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # GGUF value type -> bytes
_GG_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_header(path, want=None, token_prefix=0):
    """Metadata of a GGUF without reading its tensors: {key: value} for the keys in `want` (all
    scalars and strings if None; arrays are skipped). Stops as soon as every wanted key is found.
    token_prefix > 0 adds "_tokens": (vocabulary size, sha1 of its first token_prefix tokens).
    gguf's GGUFReader parses every array (a 150k-token vocabulary) and took ~7 s per file."""
    import hashlib
    import struct
    out = {}
    with open(path, "rb", buffering=1 << 20) as f:
        rd = f.read
        if rd(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _ver, _nt, nkv = struct.unpack("<IQQ", rd(20))

        def string():
            (n,) = struct.unpack("<Q", rd(8))
            return rd(n)

        for _ in range(nkv):
            key = string().decode(errors="replace")
            (t,) = struct.unpack("<I", rd(4))
            if t == 9:                                              # array
                et, n = struct.unpack("<IQ", rd(12))
                if key == "tokenizer.ggml.tokens" and token_prefix and et == 8:
                    h = hashlib.sha1()
                    for i in range(n):
                        b = string()
                        if i < token_prefix:
                            h.update(b)
                    out["_tokens"] = (n, h.hexdigest())
                elif et == 8:
                    for _ in range(n):
                        f.seek(struct.unpack("<Q", rd(8))[0], 1)
                elif et in _GG_FIXED:
                    f.seek(_GG_FIXED[et] * n, 1)
                else:
                    raise ValueError(f"{path}: nested GGUF array in {key}")
            elif t == 8:
                v = string().decode(errors="replace")
                if want is None or key in want:
                    out[key] = v
            elif t in _GG_FIXED:
                (v,) = struct.unpack(_GG_FMT[t], rd(_GG_FIXED[t]))
                if want is None or key in want:
                    out[key] = v
            else:
                raise ValueError(f"{path}: unknown GGUF value type {t}")
            if want is not None and all(k in out for k in want) and (not token_prefix or "_tokens" in out):
                break
    return out


class GTensor:
    __slots__ = ("name", "shape", "type_name", "n_bytes")

    def __init__(self, name, shape, type_name, n_bytes):
        self.name, self.shape, self.type_name, self.n_bytes = name, shape, type_name, n_bytes


_INFO = {}


def gguf_info(path):
    """(metadata {key: scalar, string or short numeric list}, [GTensor]) of a GGUF, without reading
    tensor data or vocabulary arrays (cached per file)."""
    import struct
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    if key in _INFO:
        return _INFO[key]
    kv = {}
    tensors = []
    with open(path, "rb", buffering=1 << 20) as f:
        rd = f.read
        if rd(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _ver, nt, nkv = struct.unpack("<IQQ", rd(20))

        def string():
            (n,) = struct.unpack("<Q", rd(8))
            return rd(n)

        for _ in range(nkv):
            k = string().decode(errors="replace")
            (t,) = struct.unpack("<I", rd(4))
            if t == 9:
                et, n = struct.unpack("<IQ", rd(12))
                if et == 8:
                    for _ in range(n):
                        f.seek(struct.unpack("<Q", rd(8))[0], 1)
                elif et in _GG_FIXED and n <= 4096 and et != 7:      # small numeric lists (per-layer values)
                    kv[k] = list(struct.unpack(f"<{n}{_GG_FMT[et][1]}", rd(_GG_FIXED[et] * n)))
                elif et in _GG_FIXED:
                    f.seek(_GG_FIXED[et] * n, 1)
                else:
                    raise ValueError(f"{path}: nested GGUF array in {k}")
            elif t == 8:
                kv[k] = string().decode(errors="replace")
            elif t in _GG_FIXED:
                kv[k] = struct.unpack(_GG_FMT[t], rd(_GG_FIXED[t]))[0]
            else:
                raise ValueError(f"{path}: unknown GGUF value type {t}")
        for _ in range(nt):
            name = string().decode(errors="replace")
            (nd,) = struct.unpack("<I", rd(4))
            shape = list(struct.unpack(f"<{nd}Q", rd(8 * nd)))
            tt, _off = struct.unpack("<IQ", rd(12))
            qt = GGMLQuantizationType(tt)
            bs, ts = GGML_QUANT_SIZES[qt]
            n = 1
            for d in shape:
                n *= d
            tensors.append(GTensor(name, shape, qt.name, n // bs * ts))
    _INFO[key] = (kv, tensors)
    return _INFO[key]


def _kv_suffix(kv, suffix, default=0):
    """A per-architecture key (e.g. "qwen3moe.expert_count") by its suffix."""
    return next((v for k, v in kv.items() if k.endswith(suffix)), default)


def kv_cache_bytes(path, tokens, slots=1):
    """RAM llama.cpp's KV cache takes for `tokens` positions over `slots` slots, f16, from the GGUF
    header. Hybrid models (Qwen3.5/3.6) count every full_attention_interval-th layer. With sliding-
    window attention, full layers are those with fewer KV heads when the per-layer counts differ
    (Gemma 4: every 6th), else half of them (gpt-oss; conservative for Gemma 3); the window layers
    hold `sliding_window` positions per slot."""
    kv, _ = gguf_info(path)
    arch = kv.get("general.architecture", "")
    g = lambda k, d=0: kv.get(f"{arch}.{k}", d)     # noqa: E731
    n_layer = int(g("block_count"))
    heads = g("attention.head_count", 1)
    hq = max(heads) if isinstance(heads, list) else int(heads)
    kvh = g("attention.head_count_kv", heads)
    per = [int(x) for x in (kvh if isinstance(kvh, list) else [kvh] * n_layer)][:n_layer]
    k_len = int(g("attention.key_length", int(g("embedding_length")) // max(hq, 1)))
    v_len = int(g("attention.value_length", k_len))
    win = int(g("attention.sliding_window"))
    full, swa = list(range(len(per))), []
    if g("full_attention_interval"):
        n = int(g("full_attention_interval"))
        full = full[n - 1::n]
    elif win:
        if len(set(per)) > 1:
            full = [i for i in full if per[i] == min(per)]
        else:
            full = full[1::2]
        swa = [i for i in range(len(per)) if i not in full]
    ks, vs = int(g("attention.key_length_swa", k_len)), int(g("attention.value_length_swa", v_len))
    b = sum(per[i] for i in full) * (k_len + v_len) * 2 * tokens
    b += sum(per[i] for i in swa) * (ks + vs) * 2 * min(tokens, win * slots)
    return b


def gguf_arch(path):
    try:
        return gguf_header(path, {"general.architecture"}).get("general.architecture")
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ GEMV shapes
# tensors llama.cpp multiplies by the same input, which the NPU backend runs as one dispatch
FUSE = [("attn_q", "attn_k", "attn_v"), ("attn_qkv", "attn_gate"), ("ffn_gate", "ffn_up"),
        ("ffn_gate_shexp", "ffn_up_shexp"), ("ffn_gate_exps", "ffn_up_exps")]


def shapes(path):
    """(tiles, kc) shapes the model's matrices need (same rules as the backend), fused groups too."""
    from opennpu.native.npu4.model import Mat, expert_cols
    kv, tensors = gguf_info(path)
    n_used = int(_kv_suffix(kv, ".expert_used_count", 1)) or 1
    need = set()
    blocks = {}

    def add(N, K, experts):
        try:
            if experts:
                if N * K >= (1 << 20) // 8:
                    m = Mat(N, K, expert_cols(n_used, N, K))
                    need.add((m.tiles, m.kc))
            elif N * K >= 1 << 20:
                m = Mat(N, K, 8)
                need.add((m.tiles, m.kc))
        except ValueError:                          # no NPU geometry: that matrix stays on the CPU
            pass

    for t in tensors:
        sh = [int(x) for x in t.shape]
        if len(sh) < 2 or sh[0] % 32:
            continue
        experts = len(sh) == 3 or (len(sh) == 4 and sh[3] == 1 and sh[2] > 1)
        add(sh[1], sh[0], experts)
        parts = t.name.split(".")
        if parts[0] == "blk" and len(parts) >= 3:
            blocks.setdefault(parts[1], {})[parts[2]] = (sh, experts)
    for b in blocks.values():
        for names in FUSE:
            have = [b[n] for n in names if n in b]
            for n in range(2, len(have) + 1):       # the backend fuses the largest built prefix
                grp = have[:n]
                if len({g[0][0] for g in grp}) == 1 and len({g[1] for g in grp}) == 1:
                    add(sum(g[0][1] for g in grp), grp[0][0][0], grp[0][1])
    return need


# ------------------------------------------------------------------ precision placement
# 5-6 bit GGUF weights lose precision when squeezed into the NPU's 4-bit format, but the CPU reads
# them about as fast as the NPU would; wider types (Q8/F16/BF16) would make the CPU much slower.
CPU_TYPES = {"Q5_0", "Q5_1", "Q5_K", "Q6_K"}
CPU_SHARE = float(os.environ.get("OPENNPU_LLAMA_CPU_SHARE", "0.15"))


def cpu_tensors(path):
    """Mixture-of-experts GGUFs: the 5-6 bit matrices (e.g. the Q6_K layers of a Q4_K_M file) stay on
    the CPU at full precision when they are at most CPU_SHARE of the bytes a decoded token reads
    (experts count for the share a token uses). Dense models keep everything on the NPU. Qwen3-30B-A3B Q4_K_M: perplexity 6.36 -> 6.25 (CPU alone
    6.23), same decode speed. Q4_K transcodes to the NPU exactly, so only these lose anything.
    OPENNPU_LLAMA_CPU_SHARE=0 turns it off."""
    if CPU_SHARE <= 0:
        return []
    kv, tensors = gguf_info(path)
    used = int(_kv_suffix(kv, ".expert_used_count"))
    n_exp = int(_kv_suffix(kv, ".expert_count"))
    if not (used and n_exp):
        return []                    # dense: every such weight is read every token (14B: 4.7 -> 2.1 tok/s)
    frac = used / n_exp
    wide = total = 0.0
    keep = []
    for t in tensors:
        if len(t.shape) < 2 or "embd" in t.name:
            continue
        nb = float(t.n_bytes) * (frac if "_exps" in t.name else 1.0)   # bytes a token reads
        total += nb
        if t.type_name in CPU_TYPES and t.name != "output.weight":   # the LM head: big, read every token
            wide += nb
            keep.append(t.name)
    share = wide / total if total else 0.0
    return keep if keep and share <= CPU_SHARE else []


def min_params(path):
    """Small dense models: matrices below this many weights run on the CPU. One NPU dispatch costs a
    fixed ~75 us, more than the CPU needs for a small matrix, and a small model is all small matrices
    (Qwen3-0.6B: 42 -> 85 tok/s keeping everything but the LM head on the CPU). From ~9B up the NPU's
    faster weight streaming wins (Qwen3.5-9B 9.8 -> 6.6 tok/s with the same rule), so they keep the
    backend default (1M); so do mixture-of-experts models. None = backend default."""
    if os.environ.get("OPENNPU_MIN_PARAMS"):
        return None
    kv, tensors = gguf_info(path)
    if int(_kv_suffix(kv, ".expert_count")) > 1:
        return None
    per_token = sum(float(t.n_bytes) for t in tensors if len(t.shape) >= 2 and "token_embd" not in t.name)
    return 32_000_000 if per_token < 1.5e9 else None    # 3-4B: no gain (measured 16M: -10% to +0%)


# ------------------------------------------------------------------ engine
# ------------------------------------------------------------------ speculative decoding
# A small model of the same family drafts up to SPEC_N tokens; the big one checks them in one pass.
# On the NPU a pass of 2-8 tokens costs ~1.4x one token (the 8-token device reads the weights once),
# so accepted drafts are nearly free: Llama-3.1-8B + Llama-3.2-1B 11 -> 15 tok/s, Qwen2.5-14B + 0.5B
# 6.1 -> 8 (text) / 14 (code), Qwen2.5-Coder-32B 2.5 -> 4.3. More than 7
# drafts overflow the 8-token pass (10.7 tok/s at 8). Only for plain-attention dense models:
# llama.cpp checkpoints recurrent state (Qwen3.5/3.8) and re-runs the accepted tokens, which lost
# speed (Qwen3.5-9B 9.4 -> 8.5, Qwen3.8-27B MTP 3.0 -> 2.9); mixture-of-experts verify passes read
# many more experts. OPENNPU_LLAMA_SPEC=off disables it; OPENNPU_LLAMA_DRAFT=<path> picks the draft.
SPEC = os.environ.get("OPENNPU_LLAMA_SPEC", "auto")
SPEC_N = int(os.environ.get("OPENNPU_LLAMA_SPEC_N", "6"))
SPEC_ARCHS = {"llama", "qwen2", "qwen3", "mistral", "phi3", "gemma2", "gemma3", "granite", "olmo2", "smollm3"}
SPEC_RATIO = 2.5                    # the draft file must be at least this many times smaller
SPEC_MIN_BYTES = 4e9                # smaller models step fast enough that drafting costs more than it saves
                                    # (Llama-3.2-3B text 21 -> 16 tok/s, Qwen3-1.7B 39 -> 21; Llama-3.1-8B 11 -> 15)
SPEC_VOCAB_PREFIX = 100_000         # tokens compared (Qwen2.5 0.5B vs 14B differ only in padding at the end)
_META = {}


def gguf_meta(path):
    """(arch, tokenizer signature, has experts) of a GGUF, cached on disk by path/size/mtime."""
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if not _META:
        try:
            _META.update(json.loads((config.CACHE / "gguf_meta.json").read_text()))
        except Exception:  # noqa: BLE001 - no cache yet
            pass
    if key not in _META:
        h = gguf_header(path, None, token_prefix=SPEC_VOCAB_PREFIX)
        n, digest = h.get("_tokens", (0, ""))
        experts = any(k.endswith(".expert_count") and int(v) > 1 for k, v in h.items())
        _META[key] = [h.get("general.architecture", ""), f"{h.get('tokenizer.ggml.model', '')}|{n}|{digest}", experts]
        try:
            (config.CACHE / "gguf_meta.json").write_text(json.dumps(_META))
        except Exception:  # noqa: BLE001
            pass
    return tuple(_META[key])


def _all_ggufs():
    """Every model GGUF in the model folders (not only the one each folder is listed by)."""
    dirs = [Path(m).parent for m, _ in _gguf_models().values()]
    out = set()
    for d in set(dirs):
        for f in d.glob("*.gguf"):
            n = f.name.lower()
            if not n.startswith("mmproj") and ("-of-" not in n or "-00001-of-" in n):
                out.add(str(f))
    return sorted(out)


def _same_vocab(a, b):
    """Same tokenizer: same model, same first SPEC_VOCAB_PREFIX tokens, sizes within llama.cpp's
    128-token allowance (a vocabulary shorter than the prefix must match in full)."""
    ma, na, ha = a.split("|")
    mb, nb, hb = b.split("|")
    na, nb = int(na), int(nb)
    if ma != mb or ha != hb or abs(na - nb) > 128:
        return False
    return na == nb or min(na, nb) >= SPEC_VOCAB_PREFIX


def draft_for(path):
    """The smallest local GGUF that can draft for `path` (same tokenizer, >= SPEC_RATIO x smaller), or None."""
    if SPEC == "off":
        return None
    if os.environ.get("OPENNPU_LLAMA_DRAFT"):
        return os.environ["OPENNPU_LLAMA_DRAFT"]
    try:
        arch, sig, experts = gguf_meta(path)
        if arch not in SPEC_ARCHS or experts:
            return None
        size = os.path.getsize(path)
        if size < SPEC_MIN_BYTES:
            return None
        ok = []
        for p in _all_ggufs():
            if p == path or os.path.getsize(p) * SPEC_RATIO > size:
                continue
            a, sg, ex = gguf_meta(p)
            if _same_vocab(sg, sig) and not ex:
                ok.append((a != arch, os.path.getsize(p), p))   # same family first (guesses more alike), then smallest
        return min(ok)[2] if ok else None
    except Exception:  # noqa: BLE001 - no draft: plain decoding
        return None


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class LlamaEngine:
    sessions = True

    def __init__(self, path, mmproj="", log=print):
        from opennpu.native.npu4.model import BUILD, ensure_shapes
        b = server_bin()
        if b is None:
            raise FileNotFoundError("llama.cpp with the NPU backend is not built: run llama/build.sh "
                                    "(or ./install.sh --native)")
        self.path, self.mmproj = path, mmproj or ""
        ensure_shapes(shapes(path), log)
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [str(b), "-m", path, "--host", "127.0.0.1", "--port", str(self.port), "-t", str(THREADS),
               "-c", str(CTX * SLOTS), "-np", str(SLOTS), "--jinja", "--no-webui"]
        mp = min_params(path)
        if not mp:
            # no CPU weight repacking: it would take expert banks away from the NPU (gpt-oss-20b
            # 17.8 -> 11 tok/s). Small models keep it: the NPU's buffer type claims its matrices first
            # and the CPU's share gets the faster interleaved layouts.
            cmd.append("-nr")
        # the draft runs on the NPU too: keeping its layers on the CPU gained 5% on Qwen2.5-14B but
        # lost 23% on Qwen2.5-Coder-32B (4.4 -> 3.4 tok/s)
        self.draft = draft_for(path)
        if self.draft:
            cmd += ["-md", self.draft, "--spec-type", "draft-simple", "--spec-draft-n-max", str(min(SPEC_N, 7))]
            log(f"llama.cpp: speculative decoding with draft {Path(self.draft).name}")
        cmd += os.environ.get("OPENNPU_LLAMA_ARGS", "").split()          # extra llama-server flags
        if self.mmproj:
            cmd += ["--mmproj", self.mmproj]
        env = {**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "OPENNPU_GEMV_DIR": str(BUILD)}
        if mp:
            env["OPENNPU_MIN_PARAMS"] = str(mp)
            # most of a small model's work is then on the CPU, handed back and forth with the NPU many
            # times per token: OpenMP's default wait (spin ~1 ms, then sleep) instead of PASSIVE, where
            # waking the sleeping threads each time cost 20-38% (Qwen3-1.7B 26 -> 36 tok/s). Bigger
            # models mostly wait on the NPU and do as well or better with PASSIVE (Qwen3.5-4B 14.3 vs 13.7).
            del env["OMP_WAIT_POLICY"]
            log(f"llama.cpp: small model, matrices under {mp // 1_000_000}M weights run on the CPU")
        if os.environ.get("OPENNPU_LLAMA_OMP"):
            env["OMP_WAIT_POLICY"] = os.environ["OPENNPU_LLAMA_OMP"]
        keep = cpu_tensors(path)
        if keep and "OPENNPU_CPU_TENSORS" not in os.environ:
            env["OPENNPU_CPU_TENSORS"] = ",".join(keep)
            log(f"llama.cpp: {len(keep)} 5-6 bit matrices stay on the CPU at full precision")
        self.logf = open(config.CACHE / f"llama-{self.port}.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t0 = time.time()
        while True:                                   # load + warm-up (packs every matrix for the NPU)
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see {self.logf.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if json.loads(r.read() or b"{}").get("status") == "ok":
                        break
            except Exception:  # noqa: BLE001 - still loading
                pass
            if time.time() - t0 > 1800:
                self.close()
                raise TimeoutError("llama-server did not become ready in 30 min")
            time.sleep(1)
        log(f"llama.cpp (NPU) ready in {time.time() - t0:.0f}s: {Path(path).name}")
        self.slot_of, self.last, self.lock = {}, {}, threading.Lock()
        self.vision = bool(self.mmproj)

    def _slot(self, session):
        """A llama-server slot per conversation (its prompt cache), least recently used reassigned."""
        s = session or "_anon"
        if s not in self.slot_of:
            used = set(self.slot_of.values())
            free = [i for i in range(SLOTS) if i not in used]
            if not free:
                old = min(self.slot_of, key=self.last.get)
                free = [self.slot_of.pop(old)]
                self.last.pop(old, None)
            self.slot_of[s] = free[0]
        self.last[s] = time.time()
        return self.slot_of[s]

    def chat(self, messages, max_tokens, temperature, think, on_text, session=None):
        msgs = []
        for m in messages:
            imgs = m.get("images") or []
            if imgs and not self.vision:
                raise ValueError("this GGUF model has no vision projector (mmproj); pick a vision model for images")
            if imgs:
                parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
                         for b in imgs]
                parts.append({"type": "text", "text": m["content"]})
                msgs.append({"role": m["role"], "content": parts})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {"messages": msgs, "max_tokens": max_tokens, "stream": True, "temperature": temperature,
                "cache_prompt": True, "id_slot": self._slot(session),
                "chat_template_kwargs": {"enable_thinking": bool(think)},
                "reasoning_format": "deepseek", "timings_per_token": False}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        timings, n_out, in_think = {}, 0, False
        with self.lock, urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                d = json.loads(line[5:])
                timings = d.get("timings") or timings
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    rc, c = delta.get("reasoning_content"), delta.get("content")
                    if rc:
                        if not in_think:
                            on_text("<think>")
                            in_think = True
                        on_text(rc)
                    if c:
                        if in_think:
                            on_text("</think>\n\n")
                            in_think = False
                        on_text(c)
                        n_out += 1
            if in_think:
                on_text("</think>\n\n")
        return {"prompt_tokens": timings.get("prompt_n", 0) + timings.get("cache_n", 0),
                "cached_tokens": timings.get("cache_n", 0), "completion_tokens": timings.get("predicted_n", n_out),
                "prefill_s": round(timings.get("prompt_ms", 0) / 1000, 2),
                "decode_tok_s": round(timings.get("predicted_per_second", 0), 2)}

    def embed(self, texts):
        raise ValueError("this is a chat model; load an embedding model for /v1/embeddings")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()
