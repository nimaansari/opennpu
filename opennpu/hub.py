"""Hugging Face Hub inside OpenNPU: search, inspect, download (with progress and cancel).

Everything goes through the local server (`/api/hub/*`), so the web UI, the desktop app and the
CLI share one download queue. Searches and downloads talk to huggingface.co; nothing else does.

Where downloads go:
  fused engines (dense Qwen3/Qwen2.5/Llama/Mistral, Qwen3-MoE)  ~/models/<name>   (OPENNPU_MODELS)
  everything else (universal engine, vision, speech, image gen)  the Hugging Face cache
"""
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import config

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))

# task chips in the UI -> Hub pipeline tags
TASKS = {
    "chat": ["text-generation"],
    "vision": ["image-text-to-text"],
    "embed": ["feature-extraction", "sentence-similarity"],
    "speech": ["automatic-speech-recognition", "text-to-speech", "text-to-audio"],
    "image": ["text-to-image", "image-to-image"],
    "video": ["text-to-video", "image-to-video"],
}
DENSE = {"qwen3", "qwen2", "llama", "mistral"}
SORTS = {"downloads", "likes", "trending_score", "last_modified", "created_at"}


class HubError(ValueError):
    pass


def valid_repo(repo):
    if not isinstance(repo, str) or not _REPO.match(repo) or ".." in repo:
        raise HubError("model id must look like org/name")
    return repo


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _mem_gb():
    try:
        with open("/proc/meminfo") as f:
            kv = {ln.split(":")[0]: int(ln.split()[1]) for ln in f}
        return kv["MemTotal"] / 2**20
    except (OSError, KeyError, ValueError):
        return 0.0


def free_disk(path=None):
    p = Path(path or config.MODELS)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


# ------------------------------------------------------------------ which engine runs it
def plan(model_type=None, pipeline=None, params=None, gguf=False, cfg=None, library=None):
    """How OpenNPU would run a Hub model: engine, label, whether it is the fast path, RAM need.

    params: parameter count (safetensors total or GGUF), may be None."""
    mt = model_type or (cfg or {}).get("model_type")
    gb = (params or 0) / 1e9
    ram = _mem_gb()
    out = {"engine": "universal", "fast": False, "ram_gb": None, "fits": None, "note": ""}
    if pipeline in ("text-to-image", "image-to-image", "text-to-video", "image-to-video") or library == "diffusers":
        out.update(engine="gen", label="Create (image / video on the NPU)")
    elif pipeline in ("text-to-speech", "text-to-audio"):
        out.update(engine="gen", label="Speech generation")
    elif pipeline == "automatic-speech-recognition":
        out.update(engine="gen", label="Speech to text (Whisper-class)")
    elif pipeline in ("feature-extraction", "sentence-similarity"):
        out.update(engine="embed", label="Embeddings (NPU)")
    elif mt in DENSE and pipeline in (None, "text-generation") and not gguf:
        out.update(engine="native", fast=True, label="Fused NPU engine (fastest)")
        out["ram_gb"] = round(gb * 0.63 * 2 + 1, 1) if gb else None      # decode + prefill weights
    elif mt == "qwen3_moe" and not gguf:
        out.update(engine="moe", fast=True, label="NPU MoE engine")
        out["ram_gb"] = round(gb * 0.63 + 3, 1) if gb else None
    elif pipeline in (None, "text-generation") or (cfg or {}).get("num_experts") or (cfg or {}).get("num_local_experts") \
            or ((cfg or {}).get("text_config") or {}).get("num_experts"):
        out.update(engine="npu4", label="4-bit NPU engine (any LLM)")
        if gguf:
            from opennpu.native import llamacpp
            out.update(engine="llama", label="llama.cpp on the NPU (GGUF)") if llamacpp.available() else \
                out.update(engine="universal", label="Universal NPU engine (GGUF)")
        out["ram_gb"] = round(gb * 0.6 + 2, 1) if gb else None
    else:
        out.update(label="Universal NPU engine")
        out["ram_gb"] = round(gb * 0.7 + 2, 1) if gb else None
    if out["ram_gb"] and ram:
        out["fits"] = out["ram_gb"] < ram - 4
        if not out["fits"]:
            out["note"] = f"needs ~{out['ram_gb']:.0f} GB RAM, this machine has {ram:.0f} GB"
    return out


def _card(m):
    params = None
    if getattr(m, "safetensors", None):
        params = m.safetensors.total
    gg = getattr(m, "gguf", None) or None
    if not params and gg:
        params = gg.get("total")
    cfg = getattr(m, "config", None) or {}
    p = plan(cfg.get("model_type"), m.pipeline_tag, params, bool(gg) and not getattr(m, "safetensors", None),
             cfg, getattr(m, "library_name", None))
    return {"id": m.id, "downloads": m.downloads or 0, "likes": m.likes or 0, "task": m.pipeline_tag,
            "params": params, "gguf": bool(gg), "model_type": cfg.get("model_type"),
            "gated": bool(getattr(m, "gated", False)), "updated": str(getattr(m, "last_modified", "") or "")[:10],
            "library": getattr(m, "library_name", None), "plan": p, "installed": installed(m.id)}


def search(q="", task="chat", sort="downloads", limit=30):
    """Hub search -> cards (most downloaded first by default)."""
    q = (q or "").strip()[:100]
    sort = sort if sort in SORTS else "downloads"
    try:
        limit = max(1, min(int(limit or 30), 60))
    except (TypeError, ValueError):
        raise HubError("limit must be a number")
    tags = TASKS.get(task) if task and task != "all" else None
    expand = ["downloads", "likes", "pipeline_tag", "safetensors", "gguf", "config", "lastModified", "gated",
              "library_name"]
    api, seen, out = _api(), set(), []
    for tag in tags or [None]:
        for m in api.list_models(search=q or None, pipeline_tag=tag, sort=sort, limit=limit, expand=expand):
            if m.id not in seen:
                seen.add(m.id)
                out.append(_card(m))
    out.sort(key=lambda c: -(c["likes"] if sort == "likes" else c["downloads"]))
    return out[:limit]


def details(repo):
    """Files (with sizes), GGUF variants, the engine plan and what a download would fetch."""
    valid_repo(repo)
    api = _api()
    m = api.model_info(repo, files_metadata=True)
    files = [{"name": s.rfilename, "size": s.size or 0} for s in m.siblings or []]
    cfg = m.config or {}
    try:
        from huggingface_hub import hf_hub_download
        if any(f["name"] == "config.json" for f in files):
            cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:  # noqa: BLE001 - gated or offline: keep the summary config
        pass
    params = m.safetensors.total if getattr(m, "safetensors", None) else None
    if not params and getattr(m, "gguf", None):
        params = m.gguf.get("total")
    ggufs = [f for f in files if f["name"].endswith(".gguf")]
    p = plan(cfg.get("model_type"), m.pipeline_tag, params, bool(ggufs) and not getattr(m, "safetensors", None), cfg,
             getattr(m, "library_name", None))
    if p["engine"] == "native":
        p.update(_native_check(cfg))
    sel = default_files(files, p["engine"])
    return {"id": repo, "task": m.pipeline_tag, "params": params, "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"), "gated": bool(getattr(m, "gated", False)),
            "license": (getattr(m, "card_data", None) or {}).get("license") if getattr(m, "card_data", None) else None,
            "files": files, "gguf": [{"name": f["name"], "size": f["size"]} for f in ggufs], "plan": p,
            "download": {"files": sel, "bytes": sum(f["size"] for f in files if f["name"] in sel)},
            "free_disk": free_disk(), "installed": installed(repo)}


def _native_check(cfg):
    """The fused engine needs one of its built geometries (chat.resolve); else universal."""
    import tempfile
    try:
        from opennpu.native.chat import resolve
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps(cfg))
            resolve(d)
        return {}
    except ImportError:
        return {}                               # no native toolchain here: trust the family
    except Exception as e:  # noqa: BLE001
        return {"engine": "npu4", "fast": False, "label": "4-bit NPU engine (any LLM)",
                "note": f"no fused-engine geometry for this shape ({str(e)[:120]})"}


_WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".msgpack", ".h5", ".ckpt", ".npz")
_SKIP_DIRS = ("original/", "metal/", "onnx/", "openvino/", "coreml/", "flax_model", "tf_model", "rust_model")


def default_files(files, engine, gguf=None):
    """What to fetch: configs/tokenizers + one copy of the weights (safetensors over .bin; one
    GGUF file, Q4_K_M by default, when the repo only has GGUFs)."""
    names = [f["name"] for f in files]
    small = [n for n in names if not n.endswith(_WEIGHTS) and not n.startswith(_SKIP_DIRS)
             and not n.endswith((".md", ".png", ".jpg", ".gif", ".gitattributes", ".ipynb"))]
    st = [n for n in names if n.endswith(".safetensors") and not n.startswith(_SKIP_DIRS)]
    if st:
        # diffusers repos keep fp16 variants beside fp32: prefer the fp16 set when there is one
        fp16 = [n for n in st if ".fp16." in n]
        if fp16 and engine == "gen":
            st = fp16
        else:
            st = [n for n in st if ".fp16." not in n] or st
        return small + st
    gg = [n for n in names if n.endswith(".gguf") and "mmproj" not in n]
    if gg:
        pick = gguf if gguf in gg else None
        for pref in ("Q4_K_M", "Q4_K", "MXFP4", "Q4_0", "Q5_K_M", "Q8_0", "F16", "BF16"):
            pick = pick or next((n for n in gg if pref.lower() in n.lower() and "-of-" not in n), None)
        pick = pick or gg[0]
        return small + [pick]
    other = [n for n in names if n.endswith((".bin", ".pt", ".pth", ".onnx")) and not n.startswith(_SKIP_DIRS)]
    return small + other


# ------------------------------------------------------------------ installed models
def local_dir(repo):
    return Path(config.MODELS) / repo.split("/")[-1]


def installed(repo):
    """Downloaded? (a complete ~/models/<name> or HF-cache snapshot with weights)."""
    d = local_dir(repo)
    if (d / ".opennpu-source").exists() and (d / ".opennpu-source").read_text().strip() == repo:
        return True
    c = HF_CACHE / ("models--" + repo.replace("/", "--")) / "snapshots"
    if c.is_dir():
        for s in c.iterdir():
            if any(s.glob("*.safetensors")) or any(s.glob("*.gguf")) or any(s.glob("*.bin")) or any(s.glob("**/*.safetensors")):
                return True
    return False


def remove(repo):
    """Delete a downloaded model (its ~/models dir and/or HF cache entry)."""
    valid_repo(repo)
    gone = False
    d = local_dir(repo)
    if (d / ".opennpu-source").exists() and (d / ".opennpu-source").read_text().strip() == repo:
        shutil.rmtree(d)
        gone = True
    c = HF_CACHE / ("models--" + repo.replace("/", "--"))
    if c.is_dir():
        shutil.rmtree(c)
        gone = True
    if not gone:
        raise KeyError(f"{repo} is not downloaded by OpenNPU")


# ------------------------------------------------------------------ downloads
class Pulls:
    """Background downloads, one `hf download` subprocess each (killable), progress by bytes on disk."""

    STATE = Path(os.environ.get("OPENNPU_PULLS", config.HOME / "pulls.json"))

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.on_done = None                   # callback(job) when a download completes

    def _save(self):
        """Unfinished downloads, so a server restart resumes them (hf download continues partial files)."""
        todo = [{"id": j["id"], "files": j["files"]} for j in self.jobs.values() if j["state"] == "downloading"]
        try:
            self.STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(todo))
            tmp.replace(self.STATE)
        except OSError:
            pass

    def resume(self):
        try:
            todo = json.loads(self.STATE.read_text())
        except (OSError, ValueError):
            return
        for t in todo if isinstance(todo, list) else []:
            try:
                self.start(t["id"], t.get("files"))
            except Exception:  # noqa: BLE001 - offline now: it stays listed as failed, the user can retry
                pass

    def list(self):
        with self.lock:
            return [self._view(j) for j in self.jobs.values()]

    def _view(self, j):
        done = j["state"] == "done"
        got = j["total"] if done else min(_tree_bytes(j["watch"]) - j["base"], j["total"])
        el = max(time.time() - j["t0"], 1e-3)
        rate = max(got, 0) / el
        return {"id": j["id"], "state": j["state"], "bytes": max(got, 0), "total": j["total"], "error": j.get("error"),
                "engine": j["engine"], "rate": rate, "eta_s": (j["total"] - got) / rate if rate > 1e5 and not done else None,
                "started": j["t0"], "model": j.get("model")}

    def start(self, repo, files=None, gguf=None):
        valid_repo(repo)
        with self.lock:
            j = self.jobs.get(repo)
            if j and j["state"] == "downloading":
                return self._view(j)
        info = details(repo)
        if info["gated"]:
            tok = os.environ.get("HF_TOKEN") or _hf_token()
            if not tok:
                raise HubError(f"{repo} is gated: accept its license on huggingface.co and log in "
                               "(`hf auth login`), then try again")
        engine = info["plan"]["engine"]
        sel = default_files(info["files"], engine, gguf) if gguf or not files else [f for f in files if isinstance(f, str)]
        sizes = {f["name"]: f["size"] for f in info["files"]}
        unknown = [f for f in sel if f not in sizes]
        if unknown:
            raise HubError(f"not in {repo}: {', '.join(unknown[:3])}")
        total = sum(sizes[f] for f in sel)
        to_local = engine in ("native", "moe")
        dest = local_dir(repo) if to_local else None
        need = total + 2 * 2**30
        if free_disk(dest or HF_CACHE) < need:
            raise HubError(f"not enough disk space: {repo} needs {total / 1e9:.1f} GB, "
                           f"{free_disk(dest or HF_CACHE) / 1e9:.1f} GB free")
        watch = dest if to_local else HF_CACHE / ("models--" + repo.replace("/", "--"))
        hf = Path(os.sys.executable).parent / "hf"
        cmd = [str(hf) if hf.exists() else "hf", "download", repo, *sel]
        if dest:
            dest.mkdir(parents=True, exist_ok=True)
            cmd += ["--local-dir", str(dest)]
        j = {"id": repo, "state": "downloading", "total": total, "base": 0, "watch": watch,
             "t0": time.time(), "engine": engine, "dest": str(dest) if dest else None, "files": sel}
        # progress counts the selected files' bytes already on disk (resumed downloads start part-way)
        j["base"] = max(0, _tree_bytes(watch) - _have(watch, sel))
        j["proc"] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                     start_new_session=True, env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        with self.lock:
            self.jobs[repo] = j
            self._save()
        threading.Thread(target=self._wait, args=(j,), daemon=True, name=f"pull {repo}").start()
        return self._view(j)

    def _wait(self, j):
        err = j["proc"].stderr.read()
        rc = j["proc"].wait()
        with self.lock:
            if j["state"] == "cancelled":
                return
            if rc == 0:
                j["state"] = "done"
                if j["dest"]:
                    Path(j["dest"], ".opennpu-source").write_text(j["id"])
                j["model"] = (f"{j['engine']}:{Path(j['dest']).name}" if j["dest"]
                              else f"{j['engine'] if j['engine'] in ('npu4', 'embed', 'llama') else 'universal'}:{j['id']}")
                if self.on_done:
                    try:
                        self.on_done(j)
                    except Exception:  # noqa: BLE001
                        pass
            else:
                j["state"] = "failed"
                lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
                j["error"] = lines[-1][:300] if lines else f"download exited with {rc}"
            self._save()

    def cancel(self, repo):
        with self.lock:
            j = self.jobs.get(repo)
            if not j:
                raise KeyError(f"no download of {repo}")
            if j["state"] == "downloading":
                j["state"] = "cancelled"
                try:
                    os.killpg(j["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.jobs.pop(repo)
            self._save()
        return {"id": repo, "state": "cancelled"}


def _hf_token():
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # noqa: BLE001
        return None


def _have(watch, files):
    """Bytes of `files` already complete under watch (a local dir, or an HF cache repo dir)."""
    watch = Path(watch) if watch else None
    if not watch or not watch.exists():
        return 0
    n = 0
    roots = [watch] if not (watch / "snapshots").is_dir() else list((watch / "snapshots").iterdir())
    for r in roots:
        for f in files:
            p = r / f
            try:
                if p.exists():
                    n += p.resolve().stat().st_size
            except OSError:
                pass
    return n


def _tree_bytes(p):
    p = Path(p) if p else None
    if not p or not p.exists():
        return 0
    n = 0
    for f in p.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                n += f.stat().st_size
        except OSError:
            pass
    return n


PULLS = Pulls()
