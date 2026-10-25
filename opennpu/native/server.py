"""OpenNPU model server: every model runs on the NPU; nothing is computed on the CPU/iGPU.

OpenAI-compatible:  GET /v1/models   POST /v1/chat/completions (stream or not)
Ollama-compatible:  GET /api/tags /api/ps /api/version   POST /api/chat /api/generate
Extras:             GET /status /api/stats   POST /unload      GET /  web UI

Models
  native:<dir>   any dense HF checkpoint under ~/models that the own fused NPU engine
                 runs (Qwen3 8B/14B, Qwen2.5-14B / DeepSeek-R1-Distill-14B, Mistral-Nemo
                 12B and their fine-tunes). The bare dir
                 name also works.
  universal:<id> any other Hugging Face causal LM (dir under ~/models or already in the HF
                 cache): OpenNPU's universal path, linear layers on the NPU (prefill on the
                 NPU, short decode steps on the CPU). Also serves /v1/embeddings.

One model is resident at a time (the NPU and RAM are shared); a request for another
model swaps it. A model idle for OPENNPU_IDLE_S (default 600 s) is unloaded, so an idle
server holds ~100 MB of RAM and no CPU.

    python -m opennpu.native.server [--port 11540]
"""
import argparse
import gc
import json
import os
# Idle OpenMP threads must sleep, not spin: spinning CPU threads take the power/memory budget the
# NPU shares with them (measured: the 4-bit engine 7.3 vs 6.1 tok/s). Set before torch loads.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from opennpu import config

MODELS = config.MODELS
IDLE_S = int(os.environ.get("OPENNPU_IDLE_S", "600"))
MAX_BODY = int(os.environ.get("OPENNPU_MAX_BODY", 32 * 2**20))     # bytes per request
ORIGINS = {o.strip() for o in os.environ.get("OPENNPU_ORIGINS", "").split(",") if o.strip()}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ telemetry
class Telemetry:
    """Numbers the UI shows. All measured here, none estimated:
    busy %   share of the last 10 s the NPU engine spent generating
    cpu %    this server's own CPU use (the point of the NPU: the CPU stays free)
    power    whole-system draw from the battery sensor, only while on battery (the
             driver exposes no NPU-only power), giving joules per token"""
    def __init__(self):
        self.busy = []                       # (start, end) generation intervals
        self.last = {}                       # stats of the last generation
        self._cpu = (time.time(), self._cpu_s())
        self.cpu_pct = 0.0
        self.npu = {}
        self._watts = []                     # power samples during the current generation
        self.generating_now = False
        threading.Thread(target=self._probe_npu, daemon=True).start()
        threading.Thread(target=self._sample_power, daemon=True).start()

    @staticmethod
    def power_w():
        bat = Path("/sys/class/power_supply/BAT0")
        try:
            if (bat / "status").read_text().strip() == "Discharging":
                return int((bat / "power_now").read_text()) / 1e6
        except (OSError, ValueError):
            pass
        return None

    def _sample_power(self):
        n = 0
        while True:
            time.sleep(0.5)
            n += 1
            if n % 4 == 0:                          # server CPU over the last 2 s, one sampler for all clients
                t, c = time.time(), self._cpu_s()
                self.cpu_pct = 100.0 * (c - self._cpu[1]) / max(t - self._cpu[0], 1e-3)
                self._cpu = (t, c)
            if self.generating_now:
                w = self.power_w()
                if w is not None:
                    self._watts.append(w)

    @staticmethod
    def _cpu_s():
        f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")

    def _probe_npu(self):
        try:
            out = subprocess.run(["xrt-smi", "examine"], capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return
        for line in out.splitlines():
            if "|" in line and "npu" in line.lower() and "BDF" not in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    self.npu.update(device=cells[1], arch=cells[2])
            elif "NPU Firmware" in line:
                self.npu["firmware"] = line.split(":")[-1].strip()
            elif line.strip().startswith("Version") and "xrt" not in self.npu:
                self.npu["xrt"] = line.split(":")[-1].strip()

    def generating(self, t0, t1):
        self.busy.append((t0, t1))
        self.busy = [b for b in self.busy if b[1] > t1 - 60]

    def snapshot(self, current_start=None):
        now = time.time()
        win = 10.0
        spans = list(self.busy) + ([(current_start, now)] if current_start else [])
        busy = sum(max(0.0, min(b, now) - max(a, now - win)) for a, b in spans)
        cpu = self.cpu_pct
        rss = 0
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) * 1024
        power = self.power_w()
        return {"npu_busy_pct": round(100 * busy / win, 1), "cpu_pct": round(cpu, 1), "rss_bytes": rss,
                "system_power_w": power, "npu": self.npu, "last": self.last}


TEL = Telemetry()

from opennpu import usage as _usage  # noqa: E402
USAGE = _usage.Store()
from opennpu import sessions as _sessions  # noqa: E402
SESSIONS = _sessions.Store()


def _rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) * 1024
    return 0


def _npu_busy_now(win=5.0):
    now = time.time()
    spans = list(TEL.busy) + ([(MGR.gen_start, now)] if MGR.gen_start else [])
    return 100.0 * sum(max(0.0, min(b, now) - max(a, now - win)) for a, b in spans) / win


SAMPLER = None                          # started in main(): tests and imports don't sample


def _hub_err(e):
    name = type(e).__name__
    if name == "RepositoryNotFoundError":
        return "no such model (or it is private)"
    if name == "GatedRepoError":
        return "this model is gated: accept its license on huggingface.co and run `hf auth login`"
    if name in ("ConnectionError", "ConnectTimeout", "ReadTimeout", "OfflineModeIsEnabled") or "Connection" in name:
        return "can't reach huggingface.co (offline?)"
    return f"{name}: {str(e)[:200]}"


# ------------------------------------------------------------------ model catalog
def native_models():
    from opennpu.native.chat import resolve
    out = {}
    for d in sorted(MODELS.iterdir()) if MODELS.is_dir() else []:
        if (d / "config.json").exists() and any(d.glob("*.safetensors")):
            try:
                out[d.name] = (d, *resolve(d))
            except Exception:
                pass
    return out


def moe_models():
    """~/models dirs holding a qwen3_moe checkpoint the own MoE engine runs."""
    import json as _j
    out = {}
    for d in sorted(MODELS.iterdir()) if MODELS.is_dir() else []:
        cfgp = d / "config.json"
        if not cfgp.exists():
            continue
        if not ((d / "model.gguf").exists() or any(d.glob("*.safetensors"))):
            continue
        try:
            if _j.loads(cfgp.read_text()).get("model_type") == "qwen3_moe":
                out[d.name] = (str(d),)
        except Exception:
            pass
    return out


def _hf_cache_models():
    """Causal LMs already in the Hugging Face cache (repo id -> snapshot dir)."""
    root = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    out = {}
    for d in sorted(root.glob("models--*")) if root.is_dir() else []:
        snaps = sorted((d / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
        if not snaps or not (snaps[-1] / "config.json").exists():
            continue
        if not (any(snaps[-1].glob("*.safetensors")) or any(snaps[-1].glob("*.bin"))):
            continue
        try:
            archs = json.loads((snaps[-1] / "config.json").read_text()).get("architectures") or []
        except ValueError:
            continue
        cfg = json.loads((snaps[-1] / "config.json").read_text())
        if any(a.endswith("ForCausalLM") for a in archs) or _is_vlm(cfg):
            out[d.name[len("models--"):].replace("--", "/")] = str(snaps[-1])
    return out


def _is_vlm(cfg):
    """Image-text-to-text models (Qwen-VL, SmolVLM, Gemma 3, LLaVA, ...): a vision tower + an LM."""
    archs = cfg.get("architectures") or []
    return "vision_config" in cfg and any(a.endswith(("ForConditionalGeneration", "ForImageTextToText", "ForCausalLM")) for a in archs)


_VLM = {}


def model_config(ref):
    """config.json of a local dir or a cached / hub repo (None if unavailable)."""
    try:
        p = Path(ref) / "config.json"
        if p.exists():
            return json.loads(p.read_text())
        from huggingface_hub import try_to_load_from_cache
        c = try_to_load_from_cache(ref, "config.json")
        return json.loads(Path(c).read_text()) if isinstance(c, str) else None
    except (OSError, ValueError):
        return None


def is_vision(ref):
    if ref not in _VLM:
        cfg = model_config(ref)
        _VLM[ref] = bool(cfg and _is_vlm(cfg))
    return _VLM[ref]


def universal_models(claimed):
    """Everything the fused engines don't take: ~/models dirs + HF-cache causal LMs."""
    out = {}
    for d in sorted(MODELS.iterdir()) if MODELS.is_dir() else []:
        if d.name in claimed or not (d / "config.json").exists():
            continue
        if any(d.glob("*.safetensors")) or any(d.glob("*.gguf")):
            out[d.name] = (str(d),)
    for repo, snap in _hf_cache_models().items():
        out.setdefault(repo, (repo,))
    return out


_CAT = [0.0, None]
_CAT_LOCK = threading.Lock()


def catalog():
    """Installed models (rescanned at most every 5 s: it walks ~/models and the HF cache). One scan
    at a time: requests that arrive during a scan wait for it instead of each starting their own."""
    if _CAT[1] is not None and time.time() - _CAT[0] <= 5:
        return _CAT[1]
    with _CAT_LOCK:
        if _CAT[1] is None or time.time() - _CAT[0] > 5:
            _CAT[:] = [time.time(), _scan()]
        return _CAT[1]


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:      # noqa: BLE001 - a broken source hides its models, not the server
        log(f"model scan: {fn.__name__} failed: {e!r}")
        return {}


def _scan():
    c = {f"native:{k}": ("native", v) for k, v in _safe(native_models).items()}
    c.update({f"moe:{k}": ("moe", v) for k, v in _safe(moe_models).items()})
    claimed = {k.split(":", 1)[1] for k in c}
    for k, v in _safe(universal_models, claimed).items():
        kind = "npu4" if _safe_bool(_npu4_ok, v[0]) else "universal"
        c[f"{kind}:{k}"] = (kind, v)
    c.update({f"llama:{k}": ("llama", v) for k, v in _safe(llama_models, claimed).items()})
    return c


def llama_models(claimed=()):
    """GGUF models for llama.cpp on the NPU (~/models, HF cache GGUF repos, LM Studio's folder)."""
    from opennpu.native import llamacpp
    return llamacpp.gguf_models(claimed) if llamacpp.available() else {}


def _safe_bool(fn, *a):
    try:
        return bool(fn(*a))
    except Exception:  # noqa: BLE001
        return False


_NPU4 = {}


def _npu4_ok(ref):
    """Causal LMs (dense or MoE, incl. MoE vision-language models, with images) run on the universal
    4-bit NPU engine: every large matmul on the NPU. Small vision-language models keep the
    universal path; GGUF-only checkpoints too."""
    if os.environ.get("OPENNPU_NPU4", "1") == "0":
        return False
    if ref in _NPU4:
        return _NPU4[ref]
    cfg = model_config(ref) or {}
    tc = cfg.get("text_config") or cfg
    archs = cfg.get("architectures") or []
    moe = any(tc.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    path = Path(ref) if Path(ref).exists() else Path(_hf_cache_models().get(ref, "/nonexistent"))
    ok = (bool(cfg) and (moe or (any(a.endswith("ForCausalLM") for a in archs) and not _is_vlm(cfg)))
          and any(path.glob("*.safetensors")) and config.PREFILL_ENV.exists())
    _NPU4[ref] = ok
    return ok


def _npu4_vision():
    """npu4 builds vision-language checkpoints with their vision tower (npu4/model.py)."""
    return os.environ.get("OPENNPU_NPU4_VISION", "cpu").lower() != "off"


ENGINE_LABEL = {"native": "OpenNPU fused engine", "moe": "OpenNPU MoE engine",
                "universal": "OpenNPU universal (NPU prefill, CPU decode)",
                "npu4": "OpenNPU 4-bit engine (any LLM, matmuls on the NPU)",
                "llama": "llama.cpp on the NPU (GGUF)"}


def _dir_bytes(p):
    try:
        p = Path(p)
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) + \
            sum(f.resolve().stat().st_size for f in p.rglob("*") if f.is_symlink() and f.resolve().is_file())
    except OSError:
        return 0


_SIZES = {}
_GGUF_ARCH = {}


def describe(key, kind, info):
    """Model card for the UI / API: engine, on-disk size, family."""
    if key not in _SIZES:
        if kind in ("universal", "npu4"):
            path = info[0] if Path(info[0]).exists() else _hf_cache_models().get(info[0], "")
        elif kind == "llama":
            path = info[0]
        else:
            path = info[0] if kind in ("native", "moe") else Path(info[1]) / "models" / info[2]
        _SIZES[key] = _dir_bytes(path)
    d = {"id": key, "object": "model", "owned_by": "local", "engine": kind,
         "engine_label": ENGINE_LABEL[kind], "size_bytes": _SIZES[key], "device": "npu"}
    if kind == "native":
        d["family"] = info[1]
    elif kind == "moe":
        d["family"] = "qwen3-moe"
    elif kind == "npu4":
        d["family"] = (model_config(info[0]) or {}).get("model_type") or "any architecture"
    elif kind == "llama":
        if info[0] not in _GGUF_ARCH:
            from opennpu.native.llamacpp import gguf_arch
            _GGUF_ARCH[info[0]] = gguf_arch(info[0]) or "gguf"
        d["family"] = _GGUF_ARCH[info[0]]
    else:
        d["family"] = "vision + language" if kind == "universal" and is_vision(info[0]) else "any architecture"
        d["device"] = "npu+cpu"
    d["vision"] = bool(info[1]) if kind == "llama" else \
        (kind == "universal" or (kind == "npu4" and _npu4_vision())) and is_vision(info[0])
    d["name"] = key.split(":", 1)[1]
    return d


_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _repo_id(ref):
    """Only Hugging Face repo ids ("org/name") reach the loaders from HTTP: no file paths (a
    pickled .pt would run code on load), no "..", no single-file formats."""
    if not _REPO.match(ref) or ".." in ref or ref.lower().endswith((".pt", ".pth", ".bin", ".onnx", ".gguf", ".pkl")):
        raise KeyError(f"{ref!r} is not a Hugging Face repo id (org/name)")
    return ref


def _hf_cached(repo):
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


def fetch(repo):
    """Make sure an HF repo is on disk BEFORE the resident model is unloaded: a typo is a 404 (not
    an evicted model), a too-big repo is refused up front, and the download runs without the
    model lock so other requests keep being served."""
    if _hf_cached(repo):
        return
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except RepositoryNotFoundError:
        raise KeyError(f"no model {repo!r} on Hugging Face (check the spelling) and none installed")
    except GatedRepoError:
        raise PermissionError(f"{repo} is gated: accept its license on huggingface.co and log in (hf auth login)")
    pats = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*")
    import fnmatch
    need = sum(f.size or 0 for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in pats))
    free = shutil.disk_usage(os.environ.get("HF_HOME", Path.home() / ".cache")).free
    if need * 1.1 > free:
        raise MemoryError(f"{repo} needs {need / 2**30:.1f} GiB of disk, {free / 2**30:.1f} GiB free")
    log(f"downloading {repo} ({need / 2**30:.1f} GiB) ...")
    snapshot_download(repo, allow_patterns=list(pats))


def _prepare(model):
    """Resolve the model and fetch it if needed (outside the lock: may download for minutes).
    Raises KeyError / PermissionError / MemoryError before any response has been started."""
    key, (kind, info) = find(model)
    if kind in ("universal", "embed") and "/" in info[0] and not Path(info[0]).exists():
        fetch(info[0])
        _CAT[1] = None                          # now installed: its engine may be npu4
        key = find(model)[0]
    return key


def find(model):
    if isinstance(model, str) and model.startswith("flm:"):
        raise KeyError("FastFlowLM support was removed: OpenNPU runs every model on its own engines. "
                       "Pick one from /v1/models (e.g. a Qwen3 or Qwen3-30B-A3B model)")
    c = catalog()
    if model in c:
        return model, c[model]
    for k in c:                                   # bare names: "Qwen3-14B", "Qwen3-30B-A3B-Instruct-2507"
        if k.split(":", 1)[1] == model:
            return k, c[k]
    if ":" in model and model.split(":", 1)[0] in ("universal", "npu4"):   # saved under the other engine
        other = {"universal": "npu4", "npu4": "universal"}[model.split(":", 1)[0]] + ":" + model.split(":", 1)[1]
        if other in c:
            return other, c[other]
    if model.startswith(("universal:", "npu4:")) or (model.count("/") == 1 and ":" not in model):
        repo = _repo_id(model.split(":", 1)[-1])      # any HF repo id: downloaded on first use
        return f"universal:{repo}", ("universal", (repo,))
    if model.startswith("embed:"):
        repo = _repo_id(model.split(":", 1)[1])
        return f"embed:{repo}", ("embed", (repo,))
    raise KeyError(f"unknown model {model!r}; GET /v1/models lists what is installed")


# ------------------------------------------------------------------ engines
KV_SNAPSHOTS = int(os.environ.get("OPENNPU_KV_SNAPSHOTS", "4"))   # sessions kept warm per model


class KVSessions:
    """Per-session prompt caching for the fused NPU decoders.

    Their KV cache is indexed by position and attention only reads positions < pos, so rewinding
    pos to the longest common token prefix of the new prompt is exact: a follow-up turn feeds
    only its new tokens instead of the whole conversation again. Switching sessions snapshots
    the KV buffer to RAM (a few hundred MB, LRU, at most KV_SNAPSHOTS) and restores it on return."""
    sessions = True

    def _kv_init(self, dec, bo):
        import numpy as np
        self._np, self._kv_bo = np, bo
        self._kv_sess, self._fed, self._snaps = None, [], OrderedDict()
        orig = dec.step

        def step(tok, *a, **k):                  # track exactly which tokens are in the KV cache
            r = orig(tok, *a, **k)
            self._fed.append(tok)
            return r
        dec.step = step

    def _kv_switch(self, session):
        import pyxrt as xrt
        s = session or "_anon"
        if s == self._kv_sess:
            return
        size = self._kv_bo.size()
        if self._kv_sess is not None and self._fed and mem_available() - size > RAM_RESERVE:
            self._kv_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            self._snaps[self._kv_sess] = (self._np.frombuffer(self._kv_bo.map(), self._np.uint8, count=size).copy(),
                                          list(self._fed))
            while len(self._snaps) > KV_SNAPSHOTS:
                self._snaps.popitem(last=False)
        snap = self._snaps.pop(s, None)
        if snap is not None:
            self._np.frombuffer(self._kv_bo.map(), self._np.uint8, count=size)[:] = snap[0]
            self._kv_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._fed = snap[1]
        else:
            self._fed = []
        self.dec.pos = len(self._fed)
        self._kv_sess = s

    def _kv_prefix(self, ids):
        """Rewind to the longest common prefix (keeping >= 1 token to feed); returns its length."""
        fed, n, k = self._fed, min(len(self._fed), len(ids) - 1), 0
        while k < n and fed[k] == ids[k]:
            k += 1
        self._fed = fed[:k]
        self.dec.pos = k
        return k


class NativeEngine(KVSessions):
    def __init__(self, ckpt, name, macro):
        from transformers import AutoTokenizer
        from opennpu.native.chat import ensure_templates
        from opennpu.native.runtime import NativeDecoder
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.dec = NativeDecoder(ckpt, name, ensure_templates(name, macro), verbose=False)
        self._kv_init(self.dec, self.dec.kvc)
        self.eos = {self.tok.eos_token_id} if self.tok.eos_token_id is not None else set()
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "<｜end▁of▁sentence｜>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                self.eos.add(i)
        gc_path = Path(ckpt) / "generation_config.json"
        if gc_path.exists():
            e = json.loads(gc_path.read_text()).get("eos_token_id")
            self.eos |= set(e if isinstance(e, list) else [e] if e is not None else [])
        self.pf, self._closed = None, False
        threading.Thread(target=self._init_prefill, name="prefill-init", daemon=True).start()

    def _init_prefill(self):
        """Batched NPU prefill, prepared in the background (one-time device build and weight
        packing can take minutes); prompts feed per token until it is ready."""
        from opennpu.native import prefill
        try:
            pf = prefill.for_decoder(self.dec, build_if_missing=True, log=log)
            if pf is None or self._closed:
                return
            if mem_available() - pf.nbytes() < RAM_RESERVE:
                log(f"fused prefill skipped: needs {pf.nbytes() / 2**30:.1f} GiB more RAM than is free")
                return
            t = time.time()
            pf.load()
            if self._closed:
                pf.unload()
                return
            self.pf = pf
            log(f"fused prefill ready ({time.time() - t:.0f}s)")
        except Exception as e:  # noqa: BLE001
            log(f"fused prefill unavailable: {type(e).__name__}: {e}")

    def chat(self, messages, max_tokens, temperature, think, on_text, session=None):
        kw = {} if think else {"enable_thinking": False}
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
        except TypeError:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        room = self.dec.ATTN_MAXL - len(ids)
        if room <= 1:
            raise BadRequest(f"this conversation is {len(ids)} tokens; the model's context is {self.dec.ATTN_MAXL}. "
                             "Start a new conversation (or shorten the message).")
        self._kv_switch(session)
        cached = self._kv_prefix(ids)
        out, shown = [], [""]

        def emit(t):
            out.append(t)
            s = self.tok.decode(out, skip_special_tokens=True)
            if not s.endswith("�"):            # wait for complete UTF-8 characters
                on_text(s[len(shown[0]):])
                shown[0] = s

        pf, lg, t0 = self.pf, None, time.perf_counter()
        if pf is not None and pf.worth_it(len(ids), len(ids) - cached):
            lg = pf.run(ids)                       # whole prompt on the NPU, KV seeded, pos set
            self._fed, cached = list(ids), 0
        new, tp, td = self.dec.generate(ids[cached:], min(max_tokens, room), eos=self.eos, on_token=emit,
                                        temperature=temperature, seed=int(time.time()), logits=lg)
        if lg is not None:
            tp = time.perf_counter() - t0 - td
        s = self.tok.decode(out, skip_special_tokens=True)
        if len(s) > len(shown[0]):
            on_text(s[len(shown[0]):])
        return {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": len(new),
                "prefill_s": round(tp, 2), "decode_tok_s": round(len(new) / max(td, 1e-9), 2),
                "batched_prefill": lg is not None}

    def close(self):
        self._closed = True
        if self.pf is not None:
            self.pf.unload()
            self.pf = None
        del self.dec


class MoeEngine(KVSessions):
    """Own fused MoE NPU engine (Qwen3-30B-A3B-class). Builds its templates on first use."""

    def __init__(self, ckpt_dir):
        import os as _os
        import subprocess as _sp
        from pathlib import Path as _P
        from transformers import AutoTokenizer
        from opennpu.native.moe_runtime import MoeDecoder, DECODE_MODEL
        tdir = config.MOE_TEMPLATES
        here = _P(__file__).resolve().parent
        if not (tdir / "layer" / "decode_L2048.xclbin").exists():
            log(f"building {DECODE_MODEL} NPU templates (one time) ...")
            _sp.run(["bash", "-c",
                     f"DECODE_WAVE_HI=1 {here}/build_decode.sh {DECODE_MODEL} QWEN3_30B_A3B {tdir}/layer 2048"],
                    check=True, stdout=_sp.DEVNULL)
            _sp.run(["bash", "-c",
                     f"DECODE_WAVE_LO=1 DECODE_WAVE_HI=6 {here}/build_decode.sh {DECODE_MODEL} QWEN3_30B_A3B {tdir}/lm 2048"],
                    check=True, stdout=_sp.DEVNULL)
        _os.environ["OPENNPU_MOE_TEMPLATES"] = str(tdir)
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        self.dec = MoeDecoder(ckpt_dir, verbose=False)
        self._kv_init(self.dec, self.dec.kv_parent)
        self.eos = {self.tok.eos_token_id} if self.tok.eos_token_id is not None else set()
        for s in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(s)
            if isinstance(i, int) and i >= 0:
                self.eos.add(i)

    def chat(self, messages, max_tokens, temperature, think, on_text, session=None):
        try:
            text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=think)
        except TypeError:
            text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = [int(x) for x in self.tok(text, add_special_tokens=False)["input_ids"]]
        room = self.dec.ATTN_MAXL - len(ids)
        if room <= 1:
            raise BadRequest(f"this conversation is {len(ids)} tokens; the model's context is {self.dec.ATTN_MAXL}. "
                             "Start a new conversation (or shorten the message).")
        self._kv_switch(session)
        cached = self._kv_prefix(ids)
        out, shown = [], [""]

        def emit(tk):
            out.append(tk)
            s = self.tok.decode(out, skip_special_tokens=True)
            if not s.endswith("\ufffd"):
                on_text(s[len(shown[0]):]); shown[0] = s

        new, tp, td = self.dec.generate(ids[cached:], min(max_tokens, room), eos=self.eos, on_token=emit,
                                        temperature=temperature)
        s = self.tok.decode(out, skip_special_tokens=True)
        if len(s) > len(shown[0]):
            on_text(s[len(shown[0]):])
        return {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": len(new),
                "prefill_s": round(tp, 2), "decode_tok_s": round(len(new) / max(td, 1e-9), 2)}

    def close(self):
        self._snaps.clear()
        del self.dec


class UniversalEngine:
    """Any HF causal LM through OpenNPU's universal offload (linear layers on the NPU)."""

    def __init__(self, ref):
        import torch
        from opennpu.offload import offload
        torch.set_num_threads(int(os.environ.get("OPENNPU_CPU_THREADS", "4")))
        self.torch = torch
        self.vision = is_vision(ref)
        if self.vision:
            # image-text-to-text: the vision tower and the LM's linear layers both go to the NPU
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from opennpu.loaders import auto_dtype
            self.proc = AutoProcessor.from_pretrained(ref)
            self.model = AutoModelForImageTextToText.from_pretrained(ref, dtype=auto_dtype(ref)).eval()
            self.tok = self.proc.tokenizer
        else:
            from opennpu.loaders import load
            self.model, self.tok, _ = load(ref, task="generate", dtype="auto")
        log(offload(self.model))

    def _vision_inputs(self, messages, think):
        import io
        from PIL import Image
        conv = []
        for m in messages:
            parts = [{"type": "image", "image": Image.open(io.BytesIO(b)).convert("RGB")} for b in m.get("images", [])]
            parts.append({"type": "text", "text": m["content"]})
            conv.append({"role": m["role"], "content": parts})
        return self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                             return_tensors="pt")

    def chat(self, messages, max_tokens, temperature, think, on_text):
        from transformers import TextIteratorStreamer
        if self.vision:
            enc = self._vision_inputs(messages, think)
        else:
            try:
                enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                   return_dict=True, enable_thinking=think)
            except (TypeError, ValueError):
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
                enc = self.tok(text, return_tensors="pt")
        from transformers import StoppingCriteria, StoppingCriteriaList
        n_in = enc["input_ids"].shape[1]
        cfg = self.model.config
        ctx = (getattr(cfg, "max_position_embeddings", None) or getattr(getattr(cfg, "text_config", None), "max_position_embeddings", None)
               or 4096)
        if n_in >= ctx:
            raise BadRequest(f"this conversation is {n_in} tokens; the model's context is {ctx}. "
                             "Start a new conversation (or shorten the message).")
        stop = threading.Event()

        class _Stop(StoppingCriteria):
            def __call__(self, *a, **k):
                return stop.is_set()
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
        kw = dict(**enc, max_new_tokens=min(max_tokens, ctx - n_in), do_sample=temperature > 0, streamer=streamer,
                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([_Stop()]))
        if temperature > 0:
            kw["temperature"] = temperature
        res, t0 = {}, time.time()

        def run():
            try:
                with self.torch.no_grad():
                    res["out"] = self.model.generate(**kw)
            except BaseException as e:      # noqa: BLE001 - re-raised in the request thread
                res["err"] = e
                streamer.end()              # unblock the reader below, or the lock is held forever
        th = threading.Thread(target=run, daemon=True)
        th.start()
        first = None
        try:
            for piece in streamer:
                if piece:
                    first = first or time.time()
                    on_text(piece)
        except BaseException:
            stop.set()                      # client gone / error: stop generate() before releasing the lock
            raise
        finally:
            th.join()
        if "err" in res:
            raise res["err"]
        n_out = int(res["out"].shape[1] - n_in) if "out" in res else 0
        dt = time.time() - (first or t0)
        return {"prompt_tokens": int(n_in), "completion_tokens": n_out,
                "decode_tok_s": round(max(n_out - 1, 0) / max(dt, 1e-9), 2)}

    def embed(self, texts):
        raise ValueError("this is a chat model; load an embedding model for /v1/embeddings")

    def close(self):
        del self.model


class EmbedEngine:
    def __init__(self, ref):
        import torch
        from opennpu.loaders import load
        from opennpu.offload import offload
        self.torch = torch
        self.model, self.tok, _ = load(ref, task="embed")
        log(offload(self.model))

    def embed(self, texts):
        torch = self.torch
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        return v.tolist(), int(enc["attention_mask"].sum())

    def chat(self, *a, **k):
        raise ValueError("this is an embedding model; it has no chat")

    def close(self):
        del self.model


def _status(e):
    return 404 if isinstance(e, KeyError) else 403 if isinstance(e, PermissionError) else \
        507 if isinstance(e, MemoryError) else 400 if isinstance(e, BadRequest) else 500


def _msg(e):
    return e.args[0] if isinstance(e, KeyError) and e.args else str(e)


def _err(msg, code=500):
    return {"message": str(msg), "type": "invalid_request_error" if code < 500 else "server_error", "code": code}


GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)    # the client went away


class BadRequest(ValueError):
    """A client error: reported as HTTP 400 with the message, never as a 500."""


def _int(v, name, lo=None, hi=None):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise BadRequest(f"{name} must be a number")
    try:
        i = int(v)
    except ValueError:
        raise BadRequest(f"{name} must be an integer, got {v!r}")
    if (lo is not None and i < lo) or (hi is not None and i > hi):
        raise BadRequest(f"{name} must be between {lo} and {hi}")
    return i


def _messages(msgs):
    """OpenAI / Ollama chat messages: a non-empty list of {role, content}; content is a string or
    a list of parts: {type: text, text} and {type: image_url, image_url: {url: data:...}}.
    Returns [{role, content, images?: [bytes]}] (Ollama's "images": [base64] is accepted too)."""
    from opennpu import sessions as S
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages must be a non-empty list of {role, content}")
    out, n_img = [], 0
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise BadRequest(f"messages[{i}] must be an object with a string role")
        c, imgs = m.get("content"), []
        if isinstance(c, list):
            texts = []
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type", "text") == "text":
                    texts.append(str(p.get("text", "")))
                elif p.get("type") == "image_url":
                    iu = p.get("image_url")
                    imgs.append(iu.get("url") if isinstance(iu, dict) else iu)
            c = "".join(texts)
        for u in m.get("images") or []:          # Ollama: raw base64; sessions: media URLs
            imgs.append(u if isinstance(u, str) and (u.startswith("data:") or S.MEDIA_URL in u) else f"data:image/png;base64,{u}")
        if c is None and m["role"] == "assistant":
            c = ""                                  # assistant turns that only carried tool calls
        if not isinstance(c, str):
            raise BadRequest(f"messages[{i}].content must be a string")
        d = {"role": m["role"], "content": c}
        if imgs:
            n_img += len(imgs)
            try:
                d["images"] = [S.image_bytes(u) for u in imgs]
            except S.SessionError as e:
                raise BadRequest(f"messages[{i}]: {e}")
        out.append(d)
    if n_img > S.MAX_IMAGES * 4:
        raise BadRequest(f"too many images ({n_img}); at most {S.MAX_IMAGES} per message")
    for m in out:
        if len(m.get("images", [])) > S.MAX_IMAGES:
            raise BadRequest(f"at most {S.MAX_IMAGES} images per message")
    return out


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024
    return 0


RAM_RESERVE = float(os.environ.get("OPENNPU_RAM_RESERVE_GB", "4")) * 2**30   # always left free for the system
MAX_MODELS = int(os.environ.get("OPENNPU_MAX_MODELS", "4"))


def _glob_bytes(pattern_dir, pattern):
    try:
        return sum(f.stat().st_size for f in Path(pattern_dir).glob(pattern))
    except OSError:
        return 0


def estimate_bytes(kind, info):
    """RAM a model will take once resident (weights in NPU-shared host memory + caches)."""
    from opennpu.native.runtime import CACHE as NCACHE
    if kind == "native":
        w = _glob_bytes(NCACHE, f"{Path(info[0]).name}-*.W.bin")
        return (w or _dir_bytes(info[0]) * 0.3) + 0.6 * 2**30
    if kind == "moe":
        w = _glob_bytes(NCACHE, f"{Path(info[0]).name}-moe.*")
        return (w or _dir_bytes(info[0]) * 1.05) + 1.5 * 2**30
    if kind == "npu4":
        from opennpu.loaders import param_count
        return (param_count(info[0]) or 4e9) * 0.6 + 2 * 2**30
    if kind == "llama":                   # NPU-packed weights ~ the GGUF's size; the mmap is page cache
        from opennpu.native import llamacpp as lc
        tokens = lc.CTX * lc.SLOTS
        est = Path(info[0]).stat().st_size * 1.1 + 1.5 * 2**30
        try:                              # + KV cache of every slot (Llama-3.1-8B: 4 GiB at 4 x 8k)
            est += lc.kv_cache_bytes(info[0], tokens, lc.SLOTS)
            d = lc.draft_for(info[0])     # + the speculative draft model and its cache
            if d:
                est += Path(d).stat().st_size * 1.1 + lc.kv_cache_bytes(d, tokens, lc.SLOTS)
        except Exception as e:            # noqa: BLE001 - an odd header: weights-only estimate
            log(f"memory estimate for {info[0]}: {e!r}")
        return est
    if kind in ("universal", "embed"):
        from opennpu.loaders import param_count
        n = param_count(info[0]) or 1e9
        return n * (2 if n > 1.5e9 else 4) * 2.2 + 0.5 * 2**30
    return 2**30


class Manager:
    """A pool of resident models. Several fit at once (the NPU time-shares their hardware
    contexts): switching between sessions that use different models, or reading a reply aloud,
    does not reload anything. The least recently used model is evicted when RAM runs short,
    after OPENNPU_MAX_MODELS, or after IDLE_S without use. All model work is serialised by
    `lock` (one NPU)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pool = OrderedDict()            # key -> {"engine", "kind", "bytes", "last"}, LRU first
        self.gen_start = self.loading = None
        self.last = time.time()
        self._pending = None
        threading.Thread(target=self._idle, daemon=True).start()

    # ---- compatibility with the single-slot API (gen/http.py until it moves to use())
    @property
    def key(self):
        return next(reversed(self.pool)) if self.pool else None

    @key.setter
    def key(self, k):
        if k is not None and self._pending is not None:
            self.pool[k] = {"engine": self._pending, "kind": "gen", "bytes": 0, "last": time.time()}
            self.pool.move_to_end(k)
            self._pending = None

    @property
    def engine(self):
        return self.pool[self.key]["engine"] if self.pool else None

    @engine.setter
    def engine(self, e):
        self._pending = e

    def _unload(self):
        self.evict_all()

    # ---- pool
    def resident(self):
        return list(self.pool)

    def evict(self, key):
        ent = self.pool.pop(key, None)
        if ent is None:
            return
        log(f"unloading {key}")
        try:
            ent["engine"].close()
        except Exception as e:      # noqa: BLE001 - the memory is released either way
            log("unload error:", repr(e))
        del ent
        gc.collect()

    def evict_all(self):
        for k in list(self.pool):
            self.evict(k)

    def _make_room(self, need, kind):
        while self.pool and (len(self.pool) >= MAX_MODELS or mem_available() - need < RAM_RESERVE):
            self.evict(next(iter(self.pool)))

    def use(self, key, factory, est_bytes=0, kind="other"):
        """The resident engine for key, loading it (and evicting LRU models) if needed."""
        if key in self.pool:
            self.pool.move_to_end(key)
            self.pool[key]["last"] = time.time()
            return key, self.pool[key]["engine"]
        self._make_room(est_bytes, kind)
        if mem_available() - est_bytes < RAM_RESERVE / 2:
            raise MemoryError(f"{key} needs about {est_bytes / 2**30:.1f} GiB; only "
                              f"{mem_available() / 2**30:.1f} GiB of RAM is available")
        self.loading = key
        t0 = time.time()
        log(f"loading {key} ({est_bytes / 2**30:.1f} GiB est.) ...")
        try:
            eng = factory()
        finally:
            self.loading = None
        self.pool[key] = {"engine": eng, "kind": kind, "bytes": est_bytes, "last": time.time()}
        log(f"loaded {key} in {time.time() - t0:.1f}s; resident: {', '.join(self.pool)}")
        return key, eng

    def get(self, model):
        key, (kind, info) = find(model)
        if key in self.pool:
            return self.use(key, None)
        if kind == "llama":
            from opennpu.native.llamacpp import LlamaEngine
            return self.use(key, lambda: LlamaEngine(info[0], info[1], log=log), estimate_bytes(kind, info), kind)
        if kind == "npu4":
            from opennpu.native.npu4.engine import Npu4Engine
            return self.use(key, lambda: Npu4Engine(info[0], log=log), estimate_bytes(kind, info), kind)
        cls = {"native": NativeEngine, "moe": MoeEngine, "universal": UniversalEngine, "embed": EmbedEngine}.get(kind)
        factory = (lambda: cls(*info)) if kind == "native" else (lambda: cls(info[0]))
        return self.use(key, factory, estimate_bytes(kind, info), kind)

    def _idle(self):
        while True:
            time.sleep(15)
            try:
                if self.pool and self.lock.acquire(blocking=False):
                    try:
                        for k, e in list(self.pool.items()):
                            if time.time() - e["last"] > IDLE_S:
                                log(f"idle {IDLE_S}s: {k}")
                                self.evict(k)
                    finally:
                        self.lock.release()
            except Exception as e:      # noqa: BLE001 - the idle thread must survive
                log("idle unload failed:", repr(e))


MGR = Manager()


# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120                   # idle / slow-sending connections can't hold a thread forever

    def handle(self):
        try:
            super().handle()
        except GONE:                # client went away mid-response: nothing to report
            pass

    def log_message(self, *a):
        pass

    def _save_turn(self, sid, raw_msgs, reply, model, st=None, error=None, stopped=False):
        from opennpu import sessions as S
        try:
            hist = [S.clean_message(m) for m in raw_msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")]
            r = {"role": "assistant", "content": reply}
            if st:
                r["stats"] = {k: v for k, v in st.items() if k != "session"}
            if error:
                r["error"] = error
            if stopped:
                r["stopped"] = True
            cur = SESSIONS.get(sid)
            title = cur["title"]
            if title in (None, "", "New conversation"):
                first = next((m["content"] for m in hist if m["role"] == "user" and m["content"]), "")
                title = first[:60] or title
            SESSIONS.update(sid, title=title, model=model, messages=hist + [r])
        except Exception as e:      # noqa: BLE001 - saving must never break the reply
            log("session save failed:", repr(e))

    def _client(self):
        """Which app asked: our own clients say so; others are guessed from the User-Agent."""
        c = (self.headers.get("X-OpenNPU-Client") or "").lower()
        if c in ("web", "desktop", "cli"):
            return c
        ua = (self.headers.get("User-Agent") or "").lower()
        for k, name in (("openai", "openai sdk"), ("ollama", "ollama client"), ("curl", "curl"),
                        ("python-urllib", "script"), ("python-requests", "script"), ("mozilla", "browser")):
            if k in ua:
                return name
        return "api"

    def _origin_ok(self):
        """Browsers send Origin on cross-site requests. Only local pages (and any listed in
        OPENNPU_ORIGINS) may use the API; otherwise every website the user opens could."""
        o = self.headers.get("Origin")
        if not o:
            return True
        host = urllib.parse.urlsplit(o).hostname or ""
        return host in ("127.0.0.1", "localhost", "::1") or o in ORIGINS or "*" in ORIGINS

    def _cors(self):
        o = self.headers.get("Origin")
        if o and self._origin_ok():
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")

    def _json(self, code, obj, ollama=False):
        if not ollama and isinstance(obj, dict) and isinstance(obj.get("error"), str):   # OpenAI's error shape
            obj = {"error": _err(obj["error"], code)}
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204 if self._origin_ok() else 403)
        self._cors()
        for h, v in (("Access-Control-Allow-Headers", "*"), ("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS"),
                     ("Content-Length", "0")):
            self.send_header(h, v)
        self.end_headers()

    def _q(self, name, default):
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        return (qs.get(name) or [default])[0]

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            raise BadRequest(f"request body must be 0..{MAX_BODY} bytes")
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise BadRequest("bad json")
        if not isinstance(req, dict):
            raise BadRequest("request body must be a JSON object")
        return req

    def do_PATCH(self):
        if not self._origin_ok():
            return self._json(403, {"error": "origin not allowed"})
        path = self.path.split("?", 1)[0]
        if not path.startswith("/v1/sessions/"):
            return self._json(404, {"error": "not found"})
        from opennpu import sessions as S
        try:
            req = self._body()
            msgs = req.get("messages")
            if msgs is not None:
                if not isinstance(msgs, list):
                    raise BadRequest("messages must be a list")
                msgs = [S.clean_message(m) for m in msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")]
            for k in ("title", "model"):
                if req.get(k) is not None and not isinstance(req[k], str):
                    raise BadRequest(f"{k} must be a string")
            return self._json(200, SESSIONS.update(path.split("/")[3], req.get("title"), req.get("model"), msgs))
        except KeyError as e:
            return self._json(404, {"error": _msg(e)})
        except (BadRequest, S.SessionError) as e:
            return self._json(400, {"error": str(e)})

    def do_DELETE(self):
        if not self._origin_ok():
            return self._json(403, {"error": "origin not allowed"})
        if self.path.split("?", 1)[0].startswith("/v1/sessions/"):
            try:
                SESSIONS.delete(self.path.split("?", 1)[0].split("/")[3])
                return self._json(200, {"deleted": True})
            except KeyError as e:
                return self._json(404, {"error": _msg(e)})
        if self.path.split("?", 1)[0] == "/api/hub/model":
            from opennpu import hub
            rid = self._q("id", "")
            try:
                with MGR.lock:
                    for k in [k for k in MGR.pool if k.split(":", 1)[1] in (rid, rid.split("/")[-1])]:
                        MGR.evict(k)
                hub.remove(rid)
                with hub.PULLS.lock:
                    hub.PULLS.jobs.pop(rid, None)
                _CAT[1] = None
                _NPU4.pop(rid, None)
                return self._json(200, {"deleted": rid})
            except hub.HubError as e:
                return self._json(400, {"error": str(e)})
            except KeyError as e:
                return self._json(404, {"error": _msg(e)})
        if self.path.split("?", 1)[0] == "/api/usage":
            what = self._q("what", "all")
            if what not in ("all", "usage", "hardware"):
                return self._json(400, {"error": "what must be all, usage or hardware"})
            USAGE.clear(what)
            return self._json(200, {"cleared": what})
        return self._json(404, {"error": "not found"})

    def _hub_get(self, path):
        from opennpu import hub
        try:
            if path == "/api/hub/search":
                return self._json(200, {"data": hub.search(self._q("q", ""), self._q("task", "chat"),
                                                           self._q("sort", "downloads"), self._q("limit", "30"))})
            if path == "/api/hub/model":
                return self._json(200, hub.details(self._q("id", "")))
            if path == "/api/hub/pulls":
                return self._json(200, {"data": hub.PULLS.list(), "free_disk": hub.free_disk()})
        except hub.HubError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - offline, unknown repo, rate limit, ...
            code = 404 if type(e).__name__ == "RepositoryNotFoundError" else 502
            return self._json(code, {"error": f"Hugging Face: {_hub_err(e)}"})
        return self._json(404, {"error": "not found"})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/v1/models/"):              # OpenAI retrieve-model
            mid = urllib.parse.unquote(path[len("/v1/models/"):])
            c = catalog()
            key = mid if mid in c else next((k for k in c if k.split(":", 1)[1] == mid), None)
            if key is None:
                return self._json(404, {"error": f"model {mid!r} not found"})
            return self._json(200, dict(describe(key, *c[key]), created=0))
        if path in ("/v1/models", "/v1/models/"):
            return self._json(200, {"object": "list", "data": [dict(describe(k, *v), created=0) for k, v in catalog().items()]})
        if path in ("/status", "/health"):
            return self._json(200, self._status())
        if path == "/api/stats":
            st = self._status()
            st.update(TEL.snapshot(MGR.gen_start))
            return self._json(200, st)
        if path == "/v1/sessions":
            return self._json(200, {"object": "list", "data": SESSIONS.list()})
        if path.startswith("/v1/sessions/media/"):
            from opennpu import sessions as S
            f = S.media_path(urllib.parse.unquote(path[len("/v1/sessions/media/"):]))
            if not f:
                return self._json(404, {"error": "not found"})
            b = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}[f.suffix[1:]])
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(b)
            return
        if path.startswith("/v1/sessions/"):
            try:
                return self._json(200, SESSIONS.get(path.split("/")[3]))
            except KeyError as e:
                return self._json(404, {"error": _msg(e)})
        if path == "/api/usage":
            return self._json(200, USAGE.summary(self._q("range", "24h")))
        if path == "/api/hardware":
            rng = self._q("range", "24h")
            return self._json(200, dict(USAGE.hardware(rng), now=SAMPLER.now if SAMPLER else {},
                                        info=_usage.static_info(TEL.npu), enabled=USAGE.enabled))
        if path == "/api/usage.csv":
            b = USAGE.csv().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="opennpu-usage.csv"')
            self.send_header("Content-Length", str(len(b)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(b)
            return
        if path.startswith("/api/hub/"):
            return self._hub_get(path)
        if path == "/api/version":
            from opennpu import __version__
            return self._json(200, {"version": __version__})
        if path == "/api/tags":                    # Ollama: installed models
            return self._json(200, {"models": [
                {"name": k, "model": k, "size": describe(k, *v)["size_bytes"],
                 "details": {"family": describe(k, *v)["family"], "format": v[0]}} for k, v in catalog().items()]})
        if path == "/v1/gen/models":               # image / video / speech models
            from opennpu.gen import http as gen
            return self._json(200, gen.models())
        if path == "/v1/gen/progress":             # no MGR.lock: a running generation holds it
            from opennpu.gen import http as gen
            return self._json(200, gen.progress())
        if path == "/api/ps":                      # Ollama: loaded models
            return self._json(200, {"models": [{"name": k, "model": k, "size_vram": int(e["bytes"])}
                                              for k, e in reversed(MGR.pool.items())]})
        if self._static(path):
            return
        self._json(404, {"error": "not found"})

    def _status(self):
        idle_left = max(0, int(IDLE_S - (time.time() - MGR.pool[MGR.key]["last"]))) if MGR.key else None
        return {"status": "ok", "loaded": MGR.key, "resident": list(reversed(MGR.pool)), "loading": MGR.loading, "generating": MGR.gen_start is not None,
                "idle_unload_s": IDLE_S, "unload_in_s": idle_left}

    def _static(self, path):
        web = Path(__file__).resolve().parent.parent / "web"
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (web / rel).resolve()
        if web not in f.parents or not f.is_file():
            return False
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript",
                 ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}.get(f.suffix, "application/octet-stream")
        b = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(b)
        return True

    def do_POST(self):
        if not self._origin_ok():
            return self._json(403, {"error": "origin not allowed; set OPENNPU_ORIGINS to allow it"})
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            return self._json(411, {"error": "send the request body with a Content-Length (chunked bodies are not supported)"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True            # don't read (or wait for) a body we refuse
            return self._json(413 if n > MAX_BODY else 400,
                              {"error": f"request body must be 0..{MAX_BODY} bytes (Content-Length {n})"})
        if self.path.split("?", 1)[0] == "/v1/audio/transcriptions":       # multipart or JSON, raw bytes
            return self._transcribe(self.rfile.read(n), self.headers.get("Content-Type") or "")
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad json"})
        if not isinstance(req, dict):
            return self._json(400, {"error": "request body must be a JSON object"})
        try:
            return self._post(req)
        except BadRequest as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})

    def _post(self, req):
        path = self.path.split("?", 1)[0]
        if path == "/v1/sessions":
            for k in ("title", "model"):
                if req.get(k) is not None and not isinstance(req[k], str):
                    raise BadRequest(f"{k} must be a string")
            return self._json(200, SESSIONS.create(req.get("title"), req.get("model")))
        if path == "/v1/sessions/import":
            if not isinstance(req.get("sessions"), list):
                raise BadRequest("sessions must be a list")
            from opennpu import sessions as S
            try:
                return self._json(200, {"imported": SESSIONS.import_many(req["sessions"])})
            except S.SessionError as e:
                raise BadRequest(str(e))
        if path == "/api/usage/settings":
            if not isinstance(req.get("enabled"), bool):
                raise BadRequest("enabled must be true or false")
            USAGE.set_enabled(req["enabled"])
            return self._json(200, {"enabled": USAGE.enabled})
        if path in ("/api/hub/pull", "/api/hub/cancel"):
            from opennpu import hub
            if not isinstance(req.get("id"), str):
                raise BadRequest("id must be a model id like org/name")
            try:
                if path == "/api/hub/cancel":
                    return self._json(200, hub.PULLS.cancel(req["id"]))
                files = req.get("files")
                if files is not None and not (isinstance(files, list) and all(isinstance(f, str) for f in files)):
                    raise BadRequest("files must be a list of file names")
                gguf = req.get("gguf") if isinstance(req.get("gguf"), str) else None
                return self._json(200, hub.PULLS.start(req["id"], files, gguf))
            except BadRequest:
                raise
            except hub.HubError as e:
                raise BadRequest(str(e))
            except KeyError as e:
                return self._json(404, {"error": _msg(e)})
            except Exception as e:  # noqa: BLE001 - offline, unknown repo, ...
                code = 404 if type(e).__name__ == "RepositoryNotFoundError" else 502
                return self._json(code, {"error": f"Hugging Face: {_hub_err(e)}"})
        if path.startswith("/unload"):
            with MGR.lock:
                if req.get("model"):
                    MGR.evict(find(req["model"])[0])
                else:
                    MGR.evict_all()
            return self._json(200, {"unloaded": True, "resident": list(MGR.pool)})
        if path.startswith("/v1/chat/completions"):
            return self._openai_chat(req)
        if path in ("/api/chat", "/api/generate"):
            return self._ollama(req, path == "/api/generate")
        if path in ("/v1/embeddings", "/api/embed"):
            return self._embeddings(req, ollama=path == "/api/embed")
        if path == "/v1/gen/cancel":                # no MGR.lock: a running generation holds it
            from opennpu.gen import http as gen
            return self._json(200, gen.cancel())
        if path in ("/v1/images/generations", "/v1/videos/generations", "/v1/audio/speech"):
            return self._gen(path, req)
        return self._json(404, {"error": "not found"})

    def _transcribe(self, raw, ctype):
        from opennpu.gen import http as gen
        t0, status = time.time(), "error"
        req = None
        if "multipart/" not in ctype.lower():
            try:
                req = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            if not isinstance(req, dict):
                return self._json(400, {"error": "request body must be a JSON object"})
            raw = b""
        ran = True
        try:
            out_type, body = gen.transcriptions(MGR, TEL, log, req, raw, ctype)
            status = "ok"
        except gen.GenError as e:
            status = "cancelled" if e.code == 499 else "error"
            ran = e.code in (499, 500, 503)
            return self._json(e.code, {"error": {"message": str(e), "type": "cancelled" if e.code == 499 else "invalid_request_error"}})
        except Exception as e:  # noqa: BLE001
            log(f"transcription failed: {type(e).__name__}: {e}")
            return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
        finally:
            if ran:
                USAGE.record("transcribe", (req or {}).get("model") or "whisper-small", "Whisper on the NPU", self._client(),
                             status, None, None, None, round(time.time() - t0, 3))
        self.send_response(200)
        self.send_header("Content-Type", out_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # -- image / video / speech generation (opennpu.gen); shares MGR's single model slot
    def _gen(self, path, req):
        t0 = time.time()
        kind = {"/v1/images/generations": "image", "/v1/videos/generations": "video"}.get(path, "speech")
        status, model, detail = "error", req.get("model"), {}
        try:
            r = self._gen_run(path, req, kind, detail)
            status = detail.pop("_status", "ok")
            model = detail.pop("_model", model)
            return r
        finally:
            if detail.get("_ran", True):
                detail.pop("_ran", None)
                USAGE.record(kind, model or {"image": "sd-turbo", "video": "wan-1.3b"}.get(kind, "mms-tts-eng"), "diffusers/transformers on the NPU",
                             self._client(), status, None, None, None, round(time.time() - t0, 3), None, None,
                             size=req.get("size"), steps=detail.get("steps"), frames=detail.get("frames"),
                             chars=len(req.get("input") or "") if kind == "speech" else None)

    def _gen_run(self, path, req, kind, detail):
        from opennpu.gen import http as gen
        try:
            if path == "/v1/audio/speech":
                ctype, body = gen.speech(MGR, TEL, log, req)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)
                return
            fn = gen.images if path == "/v1/images/generations" else gen.videos
            res = fn(MGR, TEL, log, req)
            t = res.get("timings")
            t = t[0] if isinstance(t, list) and t else t if isinstance(t, dict) else {}
            detail.update(_model=res.get("model"), steps=t.get("steps"), frames=t.get("frames"))
            return self._json(200, res)
        except gen.GenError as e:
            detail["_status"] = "cancelled" if e.code == 499 else "error"
            detail["_ran"] = e.code in (499, 500, 503)          # bad requests (4xx) never ran
            return self._json(e.code, {"error": {"message": str(e),
                                                 "type": "cancelled" if e.code == 499 else "invalid_request_error"}})
        except GONE:                         # the client left while the result was being sent
            self.close_connection = True
            detail["_status"] = "cancelled"
            return None
        except Exception as e:  # noqa: BLE001
            log(f"generation failed: {type(e).__name__}: {e}")
            return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})

    # -- shared generation path: lock, (re)load, telemetry, time-to-first-token
    def _generate(self, model, msgs, max_tokens, temperature, think, effort, on_text, on_ready=None, session=None):
        """Returns (key, stats) or raises. on_ready() fires once the model is loaded."""
        _prepare(model)
        with MGR.lock:
            key, eng = MGR.get(model)
            if on_ready:
                on_ready(key)
            t0 = time.time()
            MGR.gen_start, MGR.last = t0, t0
            TEL._watts, TEL.generating_now = [], True
            first = [None]

            def emit(t):
                if first[0] is None:
                    first[0] = time.time() - t0
                on_text(t)
            status, st = "error", {}
            vision = getattr(eng, "vision", False)
            if not vision and any(m.get("images") for m in msgs):
                raise BadRequest(f"{key} can't see images; pick a vision model (marked 'vision' in /v1/models)")
            emsgs = msgs if vision else [{"role": m["role"], "content": m["content"]} for m in msgs]
            kw = {"session": session} if getattr(eng, "sessions", False) else {}
            try:
                st = eng.chat(emsgs, max_tokens, temperature, think, emit, **kw)
                status = "ok"
            except GONE:
                status = "cancelled"
                raise
            finally:
                t1 = time.time()
                TEL.generating(t0, t1)
                TEL.generating_now = False
                MGR.gen_start, MGR.last = None, t1
                if key in MGR.pool:
                    MGR.pool[key]["last"] = t1          # idle time counts from the end of use
                watts = sum(TEL._watts) / len(TEL._watts) if TEL._watts else None
                USAGE.record("chat", key, ENGINE_LABEL.get(find(key)[1][0]), self._client(), status,
                             st.get("prompt_tokens"), st.get("completion_tokens"),
                             round(first[0], 3) if first[0] is not None else None, round(t1 - t0, 3),
                             st.get("decode_tok_s"), round(watts * (t1 - t0), 1) if watts else None,
                             api=self.path.split("?")[0])
            st = dict(st, ttft_s=round(first[0], 2) if first[0] is not None else None,
                      total_s=round(t1 - t0, 2), model=key, engine=ENGINE_LABEL[find(key)[1][0]])
            if TEL._watts and st.get("completion_tokens"):
                avg = sum(TEL._watts) / len(TEL._watts)
                st.update(system_power_w=round(avg, 1),
                          joules_per_token=round(avg * (t1 - t0) / st["completion_tokens"], 2))
            TEL.last = st
            return key, st

    @staticmethod
    def _params(req):
        opts = req.get("options") or {}                 # Ollama puts sampling knobs here
        if not isinstance(opts, dict):
            raise BadRequest("options must be an object")
        max_tokens = _int(req.get("max_tokens") or req.get("max_completion_tokens") or opts.get("num_predict") or 1024,
                          "max_tokens", -1, 1 << 20)
        if max_tokens <= 0:                         # Ollama: -1 = until done
            max_tokens = 4096
        t = req.get("temperature", opts.get("temperature", 0.0))
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float))):
            raise BadRequest("temperature must be a number")
        temperature = float(t or 0.0)
        if not 0.0 <= temperature <= 2.0:
            raise BadRequest("temperature must be between 0 and 2")
        think = req.get("think")
        if think is None:
            think = (req.get("chat_template_kwargs") or {}).get("enable_thinking", False)
        return max_tokens, temperature, bool(think), req.get("reasoning_effort")

    def _chunked_start(self, ctype, session=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if session:
            self.send_header("X-OpenNPU-Session", session)
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _openai_chat(self, req):
        model = req.get("model") or MGR.key
        if not model:
            return self._json(400, {"error": "no model given and none loaded"})
        if not isinstance(model, str):
            raise BadRequest("model must be a string")
        raw_msgs = req.get("messages")
        msgs = _messages(raw_msgs)                             # validate before anything is fetched
        max_tokens, temperature, think, effort = self._params(req)
        sid = req.get("session")
        if sid is not None and (not isinstance(sid, str) or (sid != "new" and not SESSIONS.exists(sid))):
            return self._json(404, {"error": f"no session {sid!r}"})
        try:
            _prepare(model)
        except (KeyError, PermissionError, MemoryError) as e:
            return self._json(_status(e), {"error": _msg(e)})
        if sid == "new":
            sid = SESSIONS.create(model=model)["id"]
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        acc = []                                               # the reply, for the session record

        def save(key, st=None, error=None, stopped=False):
            if sid:
                self._save_turn(sid, raw_msgs, "".join(acc), key, st, error, stopped)
        finish = lambda st: "length" if st.get("completion_tokens", 0) >= max_tokens else "stop"  # noqa: E731
        if req.get("stream"):
            self._chunked_start("text/event-stream", sid)
            key = [model]

            def send(obj):
                self._chunk(f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode())

            def chunk(delta, reason=None, **extra):
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": key[0],
                        "choices": [{"index": 0, "delta": delta, "finish_reason": reason}], **extra}

            def ready(k):
                key[0] = k
                send(chunk({"role": "assistant", "content": ""}))      # OpenAI: the first delta names the role

            def text(t):
                acc.append(t)
                send(chunk({"content": t}))
            try:
                try:
                    k, st = self._generate(model, msgs, max_tokens, temperature, think, effort, text,
                                           on_ready=ready, session=sid)
                    if sid:
                        st = dict(st, session=sid)
                    save(k, st)
                    send(chunk({}, finish(st), npu=st))
                except GONE:
                    raise
                except Exception as e:
                    log("generation failed:", repr(e))
                    save(key[0], error=_msg(e))
                    send({"error": dict(_err(_msg(e), _status(e)), **({"session": sid} if sid else {}))})
                send("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
            except GONE:
                save(key[0], stopped=True)          # stop pressed / tab closed: keep what arrived
                self.close_connection = True
            return
        try:
            key, st = self._generate(model, msgs, max_tokens, temperature, think, effort, acc.append, session=sid)
        except Exception as e:
            log("generation failed:", repr(e))
            save(model, error=_msg(e))
            return self._json(_status(e), {"error": _msg(e)})
        save(key, st)
        return self._json(200, {
            **({"session": sid} if sid else {}),
            "id": rid, "object": "chat.completion", "created": created, "model": key,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(acc)}, "finish_reason": finish(st)}],
            "usage": {"prompt_tokens": st.get("prompt_tokens", 0), "completion_tokens": st["completion_tokens"],
                      "total_tokens": st.get("prompt_tokens", 0) + st["completion_tokens"]},
            "npu": st})

    def _embeddings(self, req, ollama=False):
        model = req.get("model") or "BAAI/bge-small-en-v1.5"
        texts = req.get("input")
        texts = [texts] if isinstance(texts, str) else texts
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            raise BadRequest("input must be a string or a non-empty list of strings")
        if not isinstance(model, str):
            raise BadRequest("model must be a string")
        try:
            key, (_, (repo,)) = find(model if model.startswith("embed:") else "embed:" + model.split(":", 1)[-1])
            fetch(repo)
        except (KeyError, PermissionError, MemoryError) as e:
            return self._json(_status(e), {"error": _msg(e)})
        try:
            with MGR.lock:
                _, eng = MGR.get(key)
                t0 = time.time()
                MGR.gen_start = t0
                ok = False
                try:
                    vecs, ntok = eng.embed(texts)
                    ok = True
                finally:
                    TEL.generating(t0, time.time())
                    MGR.gen_start, MGR.last = None, time.time()
                    USAGE.record("embed", key, "OpenNPU universal (NPU)", self._client(), "ok" if ok else "error",
                                 ntok if ok else None, None, None, round(time.time() - t0, 3), inputs=len(texts))
        except Exception as e:
            log("embedding failed:", repr(e))
            return self._json(500, {"error": str(e)})
        if ollama:
            return self._json(200, {"model": model, "embeddings": vecs, "prompt_eval_count": ntok})
        return self._json(200, {"object": "list", "model": model, "usage": {"prompt_tokens": ntok, "total_tokens": ntok},
                                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]})

    def _ollama(self, req, generate):
        model = req.get("model") or MGR.key
        if not model or not isinstance(model, str):
            return self._json(400, {"error": "model is required"}, ollama=True)
        try:
            _prepare(model)
        except (KeyError, PermissionError, MemoryError) as e:
            return self._json(_status(e), {"error": _msg(e)}, ollama=True)
        if generate and not req.get("prompt") and not req.get("system"):
            # Ollama clients preload with an empty prompt, and unload with keep_alive: 0
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                if req.get("keep_alive") in (0, "0", "0s", "0m"):
                    with MGR.lock:
                        MGR.evict(find(model)[0])
                    return self._json(200, {"model": model, "created_at": ts, "response": "", "done": True, "done_reason": "unload"}, ollama=True)
                with MGR.lock:
                    MGR.get(model)
                    MGR.last = time.time()
            except Exception as e:      # noqa: BLE001
                return self._json(_status(e), {"error": _msg(e)}, ollama=True)
            return self._json(200, {"model": model, "created_at": ts, "response": "", "done": True, "done_reason": "load"}, ollama=True)
        try:
            if generate:
                if not isinstance(req.get("prompt", ""), str) or not isinstance(req.get("system") or "", str):
                    raise BadRequest("prompt and system must be strings")
                msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + \
                    [{"role": "user", "content": req.get("prompt", "")}]
            else:
                msgs = _messages(req.get("messages"))
            max_tokens, temperature, think, effort = self._params(req)
        except BadRequest as e:
            return self._json(400, {"error": str(e)}, ollama=True)
        stream = req.get("stream", True)               # Ollama streams unless told not to

        def piece(text, done=False, st=None):
            d = {"model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "done": done}
            if generate:
                d["response"] = text
            else:
                d["message"] = {"role": "assistant", "content": text}
            if st:
                d.update(done_reason="length" if st.get("completion_tokens", 0) >= max_tokens else "stop", eval_count=st.get("completion_tokens", 0),
                         prompt_eval_count=st.get("prompt_tokens", 0),
                         total_duration=int(st.get("total_s", 0) * 1e9),
                         eval_duration=int(st.get("completion_tokens", 0) / max(st.get("decode_tok_s") or 1e-9, 1e-9) * 1e9))
            return d
        if stream:
            self._chunked_start("application/x-ndjson")
            try:
                try:
                    _, st = self._generate(model, msgs, max_tokens, temperature, think, effort,
                                           lambda t: self._chunk((json.dumps(piece(t)) + "\n").encode()))
                    self._chunk((json.dumps(piece("", True, st)) + "\n").encode())
                except GONE:
                    raise
                except Exception as e:
                    log("generation failed:", repr(e))
                    self._chunk((json.dumps({"error": str(e)}) + "\n").encode())
                self.wfile.write(b"0\r\n\r\n")
            except GONE:
                self.close_connection = True
            return
        parts = []
        try:
            _, st = self._generate(model, msgs, max_tokens, temperature, think, effort, parts.append)
        except Exception as e:
            log("generation failed:", repr(e))
            return self._json(_status(e), {"error": _msg(e)}, ollama=True)
        return self._json(200, piece("".join(parts), True, st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=config.PORT)
    a = ap.parse_args()
    global SAMPLER
    SAMPLER = _usage.Sampler(USAGE, _npu_busy_now, _rss)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    log(f"OpenNPU server on http://{a.host}:{a.port}  (models: {', '.join(catalog()) or 'none'})")
    from opennpu import hub
    hub.PULLS.on_done = lambda job: _CAT.__setitem__(1, None)     # list it in /v1/models at once
    threading.Thread(target=hub.PULLS.resume, daemon=True, name="resume-downloads").start()

    def bye(*_):
        with MGR.lock:
            MGR._unload()
        os._exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    srv.serve_forever()


if __name__ == "__main__":
    main()
