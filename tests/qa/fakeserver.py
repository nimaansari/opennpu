"""The real OpenNPU HTTP server with deterministic fake engines: no NPU, no model downloads.

    python tests/qa/fakeserver.py [PORT]        prints "READY <port>" once listening

Everything above the engines is the production code: routing, validation, OpenAI/Ollama
shapes, streaming, the model lock, telemetry, CORS, static web UI, generation request
handling. Engines are scripted by the last user message:

    __error__       the engine raises mid-generation (after one token)
    __error0__      the engine raises before the first token
    __slow__        200 tokens, 50 ms apart (for stop / disconnect / concurrency tests)
    __xss__         markdown with HTML/script payloads
    __think__       a <think>...</think> block then an answer
    __md__          headings, lists, code fences, inline code, bold
    __long__        2000 tokens
    __unicode__     multi-byte characters (CJK, emoji, RTL)
    anything else   "Echo: <message>" streamed word by word

The Hugging Face Hub (/api/hub/*) is faked one level down too: opennpu.hub's HfApi is a canned
catalog (FakeHub below) and `hf download` is this file re-run as a fake downloader that writes the
files into a temp dir over a few seconds, so search, details, pull progress, cancel (a real
SIGTERM to a real process group), installed state and remove all run the production code.
Repo names steer it: "Slow" downloads take a minute (cancel tests), "Broken" downloads fail,
"offline" in a search or id raises a connection error.
"""
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENNPU_IDLE_S", "600")

import tempfile  # noqa: E402

_FAKE_HF = "--fake-hf" in sys.argv            # this file re-run as the fake `hf download` (see below)
# analytics go to a throwaway database, never the user's real one
_TMP = None if _FAKE_HF else tempfile.mkdtemp(prefix="opennpu-qa-")
if not _FAKE_HF:
    os.environ.setdefault("OPENNPU_USAGE_DB", os.path.join(_TMP, "usage.db"))
    os.environ.setdefault("OPENNPU_SESSIONS_DB", os.path.join(_TMP, "sessions.db"))
    # Hub downloads land in throwaway dirs, never ~/models or the real HF cache; no real HF login
    os.environ["OPENNPU_MODELS"] = os.path.join(_TMP, "models")
    os.environ.pop("HF_TOKEN", None)
    import opennpu.native.server as srv  # noqa: E402
    # results must not depend on the developer's machine: llama.cpp counts as not built (as in CI)
    from opennpu.native import llamacpp as _llamacpp  # noqa: E402
    _llamacpp.available = lambda: False

LOADS = []                     # (key, t) every engine load, for tests that check reuse/unload


class FakeEngine:
    sessions = True                     # like the native engines: per-session prefix (KV) cache

    def __init__(self, key):
        self.key = key
        self.vision = "vlm" in key
        self.closed = False
        self.kv = {}                    # session -> "tokens" (words) already processed
        LOADS.append((key, time.time()))

    def _script(self, text, think):
        if "__xss__" in text:
            return ['<script>window.__pwned=1</script>', ' <img src=x onerror="window.__pwned=2">',
                    ' **bold** `code <b>` ', '[link](javascript:alert(1))', '\n\n```html\n<script>x</script>\n```']
        if "__think__" in text:
            return ["<think>", "Let me ", "reason about it.", "</think>", "The answer ", "is 42."]
        if "__md__" in text:
            return ["# Title\n\n", "Some **bold** and *it* and `code`.\n\n", "- one\n- two\n\n",
                    "1. first\n2. second\n\n", "```python\nprint('hi')\n```\n\n", "Done."]
        if "__long__" in text:
            return [f"w{i} " for i in range(2000)]
        if "__slow__" in text:
            return [f"s{i} " for i in range(200)]
        if "__unicode__" in text:
            return ["日本語 ", "テスト ", "🚀✨ ", "مرحبا ", "naïve café"]
        return ["Echo: "] + [w + " " for w in text.split()]

    def chat(self, messages, max_tokens, temperature, think, on_text, session=None):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        prompt = " ".join(f"{m['role']}: {m['content']}" for m in messages).split()
        old = self.kv.get(session or "_anon", [])
        cached = 0
        while cached < min(len(old), len(prompt) - 1) and old[cached] == prompt[cached]:
            cached += 1
        if "images" in messages[-1]:
            n = len(messages[-1]["images"])
            on_text(f"I see {n} image{'s' if n > 1 else ''} ({sum(len(b) for b in messages[-1]['images'])} bytes). ")
        last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        if not isinstance(last, str):
            last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
        if "__error0__" in last:
            raise RuntimeError("engine failed before the first token")
        if "__toolong__" in last:
            raise srv.BadRequest("this conversation is 2400 tokens; the model's context is 2048. "
                                 "Start a new conversation (or shorten the message).")
        toks = self._script(last, think)[:max_tokens]
        delay = 0.05 if "__slow__" in last else float(os.environ.get("FAKE_DELAY", "0.002"))
        t0 = time.time()
        for i, t in enumerate(toks):
            if "__error__" in last and i == 1:
                raise RuntimeError("engine failed mid-generation")
            time.sleep(delay)
            on_text(t)
        dt = max(time.time() - t0, 1e-6)
        self.kv[session or "_anon"] = prompt + [f"assistant: {''.join(toks)}"]
        return {"cached_tokens": cached, "prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in messages),
                "completion_tokens": len(toks), "decode_tok_s": round(len(toks) / dt, 2)}

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = [float((hash(t) >> (i * 3)) % 7) + 1.0 for i in range(8)]
            n = sum(x * x for x in v) ** 0.5
            vecs.append([x / n for x in v])
        return vecs, sum(len(t.split()) for t in texts)

    def close(self):
        self.closed = True


CATALOG = {
    "native:Fake-Qwen3-8B": ("native", ("/nonexistent/Fake-Qwen3-8B", "qwen3-8b", "QWEN3_8B")),
    "moe:Fake-Qwen3-30B-A3B": ("moe", ("/nonexistent/Fake-Qwen3-30B-A3B",)),
    "native:Fake-Mistral-Nemo-12B": ("native", ("/nonexistent/Fake-Mistral-Nemo-12B", "mistral-nemo-12b", "MISTRAL_NEMO_12B")),
    "universal:org/fake-small": ("universal", ("org/fake-small",)),
    "universal:org/fake-vlm": ("universal", ("org/fake-vlm",)),
}


def _get(model):
    """The real model pool (LRU, eviction) with fake engines in it."""
    key, (kind, info) = srv.find(model)
    return srv.MGR.use(key, lambda: FakeEngine(key), 0, kind)


# ---- generation: real request handling (opennpu.gen.http), fake engines under it
class FakeImageEngine:
    def generate(self, prompt, **kw):
        from PIL import Image
        import opennpu.gen.runtime as rt
        if "__slow__" in prompt:                       # 20 steps of 0.25 s through the real cancel/progress hooks
            rt.set_progress(phase="generating", step=0, steps=20)
            for i in range(20):
                if rt.CANCEL.is_set():
                    raise rt.GenCancelled("generation cancelled")
                time.sleep(0.25)
                rt.set_progress(step=i + 1)
        w, h = kw.get("width") or 64, kw.get("height") or 64
        return [Image.new("RGB", (w, h), (15, 138, 108))], {"steps": kw.get("steps") or 1, "seconds": 0.01}

    def close(self):
        pass


class FakeVideoEngine:
    def generate(self, prompt, **kw):
        return [b"frame"] * 3, {"fps": 8, "seconds": 0.01}

    def close(self):
        pass


class FakeSpeechEngine:
    def generate(self, text, **kw):
        import numpy as np
        return (np.zeros(1600, dtype=np.float32), 16000), {"seconds": 0.01}

    def close(self):
        pass


class FakeTranscribeEngine:
    def transcribe(self, audio, **kw):
        return {"text": "hello from the microphone", "language": "en", "duration": len(audio) / 16000}, {"seconds": 0.01}

    def close(self):
        pass


def _install_gen_fakes():
    import opennpu.gen.runtime as rt

    def load(model, task=None, log=None, **kw):
        if "gated" in model:
            raise rt.ModelUnavailable(f"{model} is gated: accept its license on huggingface.co")
        if "missing" in model:
            raise rt.ModelUnavailable(f"{model} not found")
        if "huge" in model:
            raise rt.NotEnoughMemory(f"{model} needs 42 GB, 12 GB available")
        LOADS.append((f"{task}:{model}", time.time()))
        return {"image": FakeImageEngine, "video": FakeVideoEngine, "speech": FakeSpeechEngine,
                "transcribe": FakeTranscribeEngine}[task]()
    rt.load = load

    def repo_files(repo):            # the pre-unload availability check; stays offline in tests
        if "gated" in repo:
            raise rt.ModelUnavailable(f"{repo} is gated: accept its license on huggingface.co")
        if "missing" in repo or repo.count("/") != 1:
            raise rt.ModelUnavailable(f"{repo} not found")
        return None
    rt._repo_files = repo_files
    import opennpu.gen.http as gh                  # fake engines use no RAM; don't let the pool
    gh._estimate_bytes = lambda *a, **k: 1 << 20   # refuse them based on this machine's real RAM
    rt.mp4_bytes = lambda frames, fps: b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _fetch(repo):
    """Stand-in for the Hub lookup: names containing "typo" don't exist, "gated" are gated."""
    if "typo" in repo:
        raise KeyError(f"no model {repo!r} on Hugging Face (check the spelling) and none installed")
    if "gated" in repo:
        raise PermissionError(f"{repo} is gated")
    if "enormous" in repo:
        raise MemoryError(f"{repo} needs 900 GiB of disk")


# ---- Hugging Face Hub: a canned catalog under the real opennpu.hub code
MB = 1 << 20


class RepositoryNotFoundError(Exception):       # matched by name, like huggingface_hub's own
    pass


def _qwen3_cfg(hidden, layers, inter, heads=16, kv=8):
    return {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": hidden,
            "num_hidden_layers": layers, "intermediate_size": inter, "num_attention_heads": heads,
            "num_key_value_heads": kv, "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-6,
            "rope_theta": 1000000, "tie_word_embeddings": True, "max_position_embeddings": 40960}


def _repo(id, task, params=None, model_type=None, files=(), downloads=0, likes=0, gated=False, gguf=None,
          library="transformers", license="apache-2.0", cfg=None):
    small = [("config.json", 1024), ("tokenizer.json", 2 * MB), ("tokenizer_config.json", 8192), ("README.md", 9000)]
    return {"id": id, "pipeline_tag": task, "params": params, "cfg": cfg or ({"model_type": model_type} if model_type else {}),
            "files": ([] if gguf else small) + list(files), "downloads": downloads, "likes": likes, "gated": gated,
            "gguf": gguf, "library_name": library, "license": license, "updated": "2026-09-30"}


HUB = [
    _repo("Qwen/Qwen3-0.6B", "text-generation", 751_632_384, cfg=_qwen3_cfg(1024, 28, 3072),
          files=[("model.safetensors", 24 * MB), ("generation_config.json", 239)], downloads=9_100_000, likes=812),
    _repo("Qwen/Qwen3-30B-A3B", "text-generation", 30_532_122_624, "qwen3_moe",
          files=[("model-00001-of-00002.safetensors", 16 * MB), ("model-00002-of-00002.safetensors", 16 * MB)],
          downloads=2_400_000, likes=790),
    _repo("meta-llama/Llama-3.2-1B-Instruct", "text-generation", 1_235_814_400, "llama",
          files=[("model.safetensors", 12 * MB)], downloads=3_300_000, likes=1100, gated="manual", license="llama3.2"),
    _repo("unsloth/gpt-oss-20b-GGUF", "text-generation", 20_914_757_184, library="gguf",
          gguf={"total": 20_914_757_184, "architecture": "gpt-oss"},
          files=[("README.md", 9000), ("gpt-oss-20b-Q4_K_M.gguf", 12 * MB), ("gpt-oss-20b-Q8_0.gguf", 16 * MB),
                 ("gpt-oss-20b-F16.gguf", 20 * MB)], downloads=480_000, likes=410),
    _repo("deepseek-ai/DeepSeek-V3", "text-generation", 671_026_419_200, "deepseek_v3",
          files=[("model-00001-of-00163.safetensors", 4 * MB)], downloads=600_000, likes=3900, license="mit"),
    _repo("opennpu-test/Slow-Qwen3-1.7B", "text-generation", 2_031_739_904, cfg=_qwen3_cfg(2048, 28, 6144),
          files=[("model.safetensors", 40 * MB)], downloads=1200, likes=3),
    _repo("opennpu-test/Broken-Qwen3-4B", "text-generation", 4_022_468_096, cfg=_qwen3_cfg(2560, 36, 9728, 32),
          files=[("model.safetensors", 8 * MB)], downloads=900, likes=1),
    _repo("Qwen/Qwen3-VL-2B-Instruct", "image-text-to-text", 2_127_532_032, "qwen3_vl",
          files=[("model.safetensors", 10 * MB)], downloads=1_700_000, likes=260),
    _repo("BAAI/bge-small-en-v1.5", "feature-extraction", 33_360_000, "bert",
          files=[("model.safetensors", 4 * MB)], downloads=5_200_000, likes=380, library="sentence-transformers", license="mit"),
    _repo("openai/whisper-tiny", "automatic-speech-recognition", 37_760_640, "whisper",
          files=[("model.safetensors", 4 * MB)], downloads=1_100_000, likes=340),
    _repo("facebook/mms-tts-eng", "text-to-speech", 36_300_000, "vits",
          files=[("model.safetensors", 4 * MB)], downloads=310_000, likes=210, license="cc-by-nc-4.0"),
    _repo("stabilityai/sd-turbo", "text-to-image", None, files=[("unet/diffusion_pytorch_model.fp16.safetensors", 6 * MB)],
          downloads=900_000, likes=420, library="diffusers", license="other"),
    _repo("Lightricks/LTX-Video", "text-to-video", None, files=[("transformer/diffusion_pytorch_model.safetensors", 6 * MB)],
          downloads=150_000, likes=1600, library="diffusers", license="other"),
]
HUB_BY_ID = {r["id"]: r for r in HUB}


def _info(r):
    from types import SimpleNamespace as NS
    return NS(id=r["id"], pipeline_tag=r["pipeline_tag"], downloads=r["downloads"], likes=r["likes"], gated=r["gated"],
              library_name=r["library_name"], last_modified=r["updated"] + "T12:00:00.000Z", config=dict(r["cfg"]),
              safetensors=NS(total=r["params"]) if r["params"] and not r["gguf"] else None, gguf=r["gguf"],
              card_data={"license": r["license"]}, siblings=[NS(rfilename=n, size=sz) for n, sz in r["files"]])


class FakeHub:
    """The two HfApi calls opennpu.hub makes."""

    def list_models(self, search=None, pipeline_tag=None, sort=None, limit=30, expand=None):
        if search and "offline" in search.lower():
            raise ConnectionError("Max retries exceeded with url: /api/models")
        hits = [r for r in HUB if (not pipeline_tag or r["pipeline_tag"] == pipeline_tag) and
                (not search or all(w in r["id"].lower() for w in search.lower().split()))]
        return [_info(r) for r in hits[:limit]]

    def model_info(self, repo, files_metadata=False):
        if "offline" in repo.lower():
            raise ConnectionError("Max retries exceeded with url: /api/models/" + repo)
        if repo not in HUB_BY_ID:
            raise RepositoryNotFoundError(f"404 Client Error: Repository Not Found for url: /api/models/{repo}")
        return _info(HUB_BY_ID[repo])


def _fake_hf_hub_download(repo, filename, **kw):
    """details() reads the full config.json: serve the canned one from a temp file."""
    r = HUB_BY_ID.get(repo)
    if r is None or filename != "config.json":
        raise OSError(f"{repo}/{filename} is not in the fake hub")
    p = Path(_TMP, "hub-configs", repo.replace("/", "--") + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(r["cfg"]))
    return str(p)


class _FakeSubprocess:
    """opennpu.hub runs `hf download REPO FILES... [--local-dir D]`: run this file as that command."""
    PIPE, DEVNULL = __import__("subprocess").PIPE, __import__("subprocess").DEVNULL

    @staticmethod
    def Popen(cmd, **kw):
        import json
        import subprocess
        from opennpu import hub
        repo = cmd[2]
        env = dict(kw.pop("env", None) or os.environ, FAKE_HF_SIZES=json.dumps(dict(HUB_BY_ID[repo]["files"])),
                   FAKE_HF_CACHE=str(hub.HF_CACHE))
        return subprocess.Popen([sys.executable, __file__, "--fake-hf", *cmd[1:]], env=env, **kw)


def _fake_hf_main(argv):
    """The fake `hf download`: writes each file in 20 slices over FAKE_PULL_S seconds."""
    import json
    args = argv[argv.index("download") + 1:]
    dest = None
    if "--local-dir" in args:
        i = args.index("--local-dir")
        dest, args = Path(args[i + 1]), args[:i] + args[i + 2:]
    repo, files = args[0], args[1:]
    sizes = json.loads(os.environ["FAKE_HF_SIZES"])
    if dest is None:
        dest = Path(os.environ["FAKE_HF_CACHE"]) / ("models--" + repo.replace("/", "--")) / "snapshots" / ("0" * 40)
    secs = 60.0 if "Slow" in repo else float(os.environ.get("FAKE_PULL_S", "2.5"))
    slices = 20
    for k in range(slices):
        for f in files:
            p = dest / f
            p.parent.mkdir(parents=True, exist_ok=True)
            want = sizes[f] * (k + 1) // slices
            have = p.stat().st_size if p.exists() else 0
            with open(p, "ab") as fh:
                fh.write(b"\0" * (want - have))
        if "Broken" in repo and k == slices // 2:
            print("Traceback (most recent call last):\nhuggingface_hub.errors.HfHubHTTPError: 500 Server Error: "
                  "Internal Server Error for url: https://huggingface.co/" + repo, file=sys.stderr)
            sys.exit(1)
        time.sleep(secs / slices)


def _install_hub_fakes():
    import huggingface_hub
    from opennpu import hub
    hub.HF_CACHE = Path(_TMP, "hf-cache")
    hub._api = FakeHub
    hub.subprocess = _FakeSubprocess
    hub._hf_token = lambda: None                 # gated stays gated, whatever this machine is logged in as
    hub._native_check = lambda cfg: {}           # the geometry table needs the NPU toolchain: trust the family
    huggingface_hub.hf_hub_download = _fake_hf_hub_download


def _hub_models():
    """Models downloaded through /api/hub/pull show up in /v1/models like real installs do."""
    from opennpu import hub
    out = {}
    for r in HUB:
        repo = r["id"]
        src = hub.local_dir(repo) / ".opennpu-source"
        if src.exists() and src.read_text().strip() == repo:
            kind = "moe" if r["cfg"].get("model_type") == "qwen3_moe" else "native"
            out[f"{kind}:{hub.local_dir(repo).name}"] = (kind, (str(hub.local_dir(repo)),))
        elif hub.installed(repo):
            out[f"universal:{repo}"] = ("universal", (repo,))
    return out


def main():
    if "--fake-hf" in sys.argv:
        return _fake_hf_main(sys.argv)
    srv.fetch = _fetch
    srv._scan = lambda: {**CATALOG, **_hub_models()}
    _install_hub_fakes()
    srv._CAT[:] = [0.0, None]
    srv.MGR.get = _get
    srv.describe = lambda key, kind, info: {
        "id": key, "object": "model", "owned_by": "local", "engine": kind, "engine_label": srv.ENGINE_LABEL[kind],
        "size_bytes": 4_400_000_000, "device": "npu", "family": "fake", "name": key.split(":", 1)[1], "vision": "vlm" in key}
    srv.TEL.npu.update(device="RyzenAI-npu6", arch="aie2p", firmware="1.1.2.64", xrt="2.25.00")
    _install_gen_fakes()
    # hardware history: two hours of synthetic per-minute rows (fake server only) + live sampling
    import math
    now = int(time.time() // 60 * 60)
    for i in range(120, 0, -1):
        w = math.sin(i / 9)
        srv.USAGE.hw_row((now - i * 60, 20 + 10 * w, 35 + 10 * w, 14e9 + 2e9 * w, 46e9, 1e9, 30 + 25 * w, 60 + 30 * w,
                          55 + 8 * w, 70 + 5 * w, 48 + 4 * w, 41 + 2 * w, None, 5e8))
    srv.USAGE.flush()
    srv.SAMPLER = srv._usage.Sampler(srv.USAGE, srv._npu_busy_now, srv._rss, interval=1)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", port), srv.H)
    httpd.daemon_threads = True
    print(f"READY {httpd.server_address[1]}", flush=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
