"""HTTP handlers for generation, mounted by opennpu.native.server.

OpenAI-compatible where OpenAI has an endpoint:
  POST /v1/images/generations   {model, prompt, n, size: "512x512", seed, steps, guidance, negative_prompt}
  POST /v1/audio/speech         {model, input, voice, response_format: wav|mp3|opus|aac|flac|pcm}
Extras:
  POST /v1/videos/generations   {model, prompt, size, frames, fps, steps, seed} -> mp4 (b64_json)
  GET  /v1/gen/models           curated generation models, with RAM estimates and download state

Generation models share the server's single model slot (Manager): loading one unloads the
chat model and vice versa, so RAM is never held by two big models at once.
"""
import base64
import subprocess
import time
from pathlib import Path

from .catalog import CATALOG, resolve

_AUDIO_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac",
                "flac": "audio/flac", "pcm": "audio/L16"}


class GenError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


_CACHED = [0.0, set()]


def models():
    # scanning the HF cache takes ~0.5-1 s on a big cache: rescan at most every 30 s
    if time.time() - _CACHED[0] > 30:
        from huggingface_hub import scan_cache_dir
        try:
            _CACHED[1] = {r.repo_id for r in scan_cache_dir().repos}
        except Exception:   # noqa: BLE001 - no cache yet
            _CACHED[1] = set()
        _CACHED[0] = time.time()
    cached = _CACHED[1]
    return {"object": "list", "data": [
        {"id": name, "object": "model", "task": e["task"], "repo": e["repo"], "ram_gb": e["ram_gb"],
         "downloaded": e["repo"] in cached, "note": e["note"]} for name, e in CATALOG.items()]}


def progress():
    """GET /v1/gen/progress: what is generating now (poll it; no lock taken)."""
    try:
        from .runtime import PROGRESS
    except ImportError:
        return {"task": None, "phase": "idle", "step": 0, "steps": 0, "started": None}
    p = dict(PROGRESS)
    if p.get("started"):
        p["elapsed_s"] = round(time.time() - p["started"], 1)
    return p


def cancel():
    """POST /v1/gen/cancel: stop the running generation at its next step (no lock taken)."""
    try:
        from .runtime import CANCEL, PROGRESS
    except ImportError:
        return {"cancelled": False, "reason": "nothing is generating"}
    if PROGRESS.get("phase") in (None, "idle"):
        return {"cancelled": False, "reason": "nothing is generating"}
    CANCEL.set()
    return {"cancelled": True, "task": PROGRESS.get("task"), "model": PROGRESS.get("model")}


def _key(model, task):
    t, _, _, name = resolve(model, task)
    return t, (name if name.startswith(f"{t}:") else f"{t}:{name}")


def _estimate_bytes(model, task, repo_files):
    """RAM to reserve in the model pool before loading: the catalog estimate, or for any other
    repo its weight files (bf16 on the CPU plus NPU buffers, about 2.3x the bf16 bytes)."""
    from .catalog import CATALOG
    name = resolve(model, task)[3].split(":", 1)[-1]
    if name in CATALOG:
        return int(CATALOG[name]["ram_gb"] * 2**30)
    if not repo_files:
        return 4 << 30
    w = sum(v for f, v in repo_files.items() if f.endswith((".safetensors", ".bin")) and ".fp16." not in f)
    fp16 = sum(v for f, v in repo_files.items() if ".fp16." in f)
    return int((fp16 or w / 2) * 2.3)


def _with_engine(mgr, tel, log, model, task, fn):
    """Load (or reuse) the generation engine from the server's model pool and run fn(engine) under the lock.

    Generation models live in the pool next to the chat model: small ones (TTS, Whisper) stay
    resident beside it, big ones (diffusion, video) make the pool evict what they need."""
    try:
        t, key = _key(model, task)
    except KeyError as e:
        raise GenError(400, e.args[0])
    if t != task:
        raise GenError(400, f"{model} is a {t} model; use the {t} endpoint")
    try:
        from .runtime import CANCEL, GenCancelled, ModelUnavailable, NotEnoughMemory, _repo_files, load, set_progress
    except ImportError as e:
        raise GenError(501, f"generation support is not installed ({e.name} missing): "
                            f"run ./install.sh --gen")
    resident = key in _resident(mgr)
    files = None
    if not resident:
        # Check the repo exists (and is not gated) before anything is evicted for it.
        repo = resolve(model, task)[1]
        if not repo or not repo.strip():
            raise GenError(400, "model is empty")
        if not Path(repo).is_dir():
            try:
                files = _repo_files(repo)
            except ModelUnavailable as e:
                raise GenError(403 if "gated" in str(e) else 404, str(e))
    with mgr.lock:
        CANCEL.clear()
        set_progress(task=t, model=key, phase="starting" if resident else "loading", step=0, steps=0,
                     started=time.time())
        try:
            try:
                if hasattr(mgr, "use"):
                    key, eng = mgr.use(key, lambda: load(model, task=t, log=log),
                                       _estimate_bytes(model, task, files), kind=t)
                else:                                   # older single-slot manager
                    if mgr.key != key:
                        mgr._unload()
                        mgr.engine, mgr.key = load(model, task=t, log=log), key
                    eng = mgr.engine
            except (NotEnoughMemory, MemoryError) as e:
                raise GenError(503, str(e))
            except ModelUnavailable as e:
                raise GenError(403 if "gated" in str(e) else 404, str(e))
            t0 = time.time()
            set_progress(phase="starting")
            mgr.gen_start, mgr.last = t0, t0
            tel.generating_now = True
            try:
                return key, fn(eng)
            except GenCancelled:
                raise GenError(499, "generation cancelled")
            finally:
                t1 = time.time()
                tel.generating(t0, t1)
                tel.generating_now = False
                mgr.gen_start, mgr.last = None, t1
        finally:
            CANCEL.clear()
            set_progress(task=None, model=None, phase="idle", step=0, steps=0, started=None)


def _resident(mgr):
    if hasattr(mgr, "resident"):
        return mgr.resident()
    return [mgr.key] if getattr(mgr, "key", None) else []


def _num(req, name, lo, hi, cast=int):
    v = req.get(name)
    if v is None:
        return None
    try:
        x = cast(v)
    except (TypeError, ValueError):
        raise GenError(400, f"{name} must be a number, got {v!r}")
    if not lo <= x <= hi:
        raise GenError(400, f"{name} must be between {lo} and {hi}")
    return x


def _size(req, default=None):
    s = req.get("size")
    if not s or s == "auto":
        return default or (None, None)
    try:
        w, h = (int(x) for x in str(s).lower().split("x"))
    except ValueError:
        raise GenError(400, f"bad size {s!r}; use WIDTHxHEIGHT, e.g. 512x512")
    if not (64 <= w <= 2048 and 64 <= h <= 2048):
        raise GenError(400, "size must be between 64 and 2048 pixels per side")
    return w, h


# Limits that keep one request from OOMing the laptop or holding the model slot for hours.
MAX_PROMPT = 2000          # characters
MAX_SPEECH = 4096          # characters, OpenAI's limit for /v1/audio/speech
MAX_VIDEO_FRAMES = 121
MAX_VIDEO_PIXELS = 1280 * 720


def _text(req, name, limit, required=True):
    v = req.get(name)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise GenError(400, f"{name} is required")
        return None
    if not isinstance(v, str):
        raise GenError(400, f"{name} must be a string")
    if len(v) > limit:
        raise GenError(400, f"{name} is too long ({len(v)} characters; the limit is {limit})")
    return v


def images(mgr, tel, log, req):
    from .runtime import png_bytes
    prompt = _text(req, "prompt", MAX_PROMPT)
    req["negative_prompt"] = _text(req, "negative_prompt", MAX_PROMPT, required=False)
    n = _num(req, "n", 1, 8) or 1
    if req.get("response_format") == "url":
        raise GenError(400, "response_format=url is not supported; use b64_json")
    w, h = _size(req)
    seed = _num(req, "seed", 0, 2**32 - 1)
    for k, hi in (("steps", 200), ("guidance", 50)):
        req[k] = _num(req, k, 0, hi, float if k == "guidance" else int)

    def run(eng):
        out, stats = [], []
        for k in range(n):
            imgs, st = eng.generate(prompt, negative=req.get("negative_prompt"), steps=req.get("steps"),
                                    guidance=req.get("guidance"), width=w, height=h,
                                    seed=None if seed is None else int(seed) + k)
            out.append(base64.b64encode(png_bytes(imgs[0])).decode())
            stats.append(st)
        return out, stats
    key, (b64s, stats) = _with_engine(mgr, tel, log, req.get("model") or "sd-turbo", "image", run)
    return {"created": int(time.time()), "model": key, "data": [{"b64_json": b} for b in b64s],
            "timings": stats}


def videos(mgr, tel, log, req):
    from .runtime import mp4_bytes
    prompt = _text(req, "prompt", MAX_PROMPT)
    req["negative_prompt"] = _text(req, "negative_prompt", MAX_PROMPT, required=False)
    w, h = _size(req)
    if w and h and w * h > MAX_VIDEO_PIXELS:
        raise GenError(400, f"video size is limited to {MAX_VIDEO_PIXELS} pixels per frame (1280x720)")
    req["frames"] = _num(req, "frames", 1, MAX_VIDEO_FRAMES)
    req["fps"] = _num(req, "fps", 1, 60)
    req["seed"] = _num(req, "seed", 0, 2**32 - 1)
    for k, hi in (("steps", 200), ("guidance", 50)):
        req[k] = _num(req, k, 0, hi, float if k == "guidance" else int)

    def run(eng):
        frames, st = eng.generate(prompt, negative=req.get("negative_prompt"), steps=req.get("steps"),
                                  guidance=req.get("guidance"), width=w, height=h, seed=req.get("seed"),
                                  frames=req.get("frames"), fps=req.get("fps"))
        return base64.b64encode(mp4_bytes(frames, st["fps"])).decode(), dict(st, frames=len(frames))
    key, (b64, st) = _with_engine(mgr, tel, log, req.get("model") or "wan-1.3b", "video", run)
    return {"created": int(time.time()), "model": key, "data": [{"b64_json": b64, "mime_type": "video/mp4"}],
            "timings": st}


def speech(mgr, tel, log, req):
    """Returns (content_type, bytes)."""
    from .runtime import wav_bytes
    text = _text(req, "input", MAX_SPEECH)
    req["seed"] = _num(req, "seed", 0, 2**32 - 1)
    if req.get("voice") is not None and not isinstance(req.get("voice"), str):
        raise GenError(400, "voice must be a string")
    fmt = req.get("response_format") or "mp3"
    if not isinstance(fmt, str):
        raise GenError(400, "response_format must be a string")
    fmt = fmt.lower()
    if fmt not in _AUDIO_TYPES:
        raise GenError(400, f"response_format must be one of {', '.join(_AUDIO_TYPES)}")
    model = req.get("model") or "mms-tts-eng"
    if model in ("tts-1", "tts-1-hd", "gpt-4o-mini-tts"):      # OpenAI names from existing clients
        model = "mms-tts-eng"
    voice = req.get("voice")
    if voice in ("alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"):
        voice = None                                          # OpenAI voice names: use the model default

    def run(eng):
        (audio, sr), st = eng.generate(text, voice=voice, seed=req.get("seed"))
        return wav_bytes(audio, sr), sr, st
    _, (wav, sr, st) = _with_engine(mgr, tel, log, model, "speech", run)
    if fmt == "wav":
        return _AUDIO_TYPES[fmt], wav
    if fmt == "pcm":
        return _AUDIO_TYPES[fmt], wav[44:]
    return _AUDIO_TYPES[fmt], _transcode(wav, fmt)


def _transcode(wav, fmt):
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:   # noqa: BLE001
        raise GenError(501, f"{fmt} needs ffmpeg; ask for response_format=wav")
    codec = {"mp3": ["-f", "mp3", "-b:a", "128k"], "opus": ["-f", "ogg", "-c:a", "libopus"],
             "aac": ["-f", "adts", "-c:a", "aac"], "flac": ["-f", "flac"]}[fmt]
    r = subprocess.run([ffmpeg, "-loglevel", "error", "-i", "pipe:0", *codec, "pipe:1"],
                       input=wav, capture_output=True, timeout=120)
    if r.returncode != 0:
        raise GenError(500, f"encoding {fmt} failed: {r.stderr.decode(errors='replace')[:200]}")
    return r.stdout


# ------------------------------------------------------------------ speech-to-text
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # OpenAI's upload limit
MAX_AUDIO_SECONDS = 30 * 60          # keeps one request from holding the model slot for hours


def _multipart(raw_body, content_type):
    """{field: str | (filename, bytes)} from a multipart/form-data body (stdlib only)."""
    from email.parser import BytesParser
    from email.policy import HTTP
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + raw_body)
    if not msg.is_multipart():
        raise GenError(400, "expected multipart/form-data with a file field")
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        fn = part.get_filename()
        out[name] = (fn, data) if fn is not None else data.decode(errors="replace")
    return out


def transcriptions(mgr, tel, log, req, raw_body=b"", content_type=""):
    """POST /v1/audio/transcriptions. Returns (content_type, bytes).

    multipart/form-data: file, model, language, prompt, response_format (json|verbose_json|text)
    application/json:    {audio_b64, format, model, language, prompt, response_format}
    """
    ctype = (content_type or "").lower()
    if ctype.startswith("multipart/form-data"):
        if len(raw_body) > MAX_AUDIO_BYTES + 1024 * 1024:
            raise GenError(413, "audio is larger than 25 MB")
        form = _multipart(raw_body, content_type)
        f = form.get("file")
        if not isinstance(f, tuple) or not f[1]:
            raise GenError(400, "file is required (multipart field 'file')")
        audio_bytes = f[1]
        fields = {k: v for k, v in form.items() if isinstance(v, str)}
    else:
        fields = req if isinstance(req, dict) else {}
        b64 = fields.get("audio_b64")
        if not isinstance(b64, str) or not b64:
            raise GenError(400, "send multipart/form-data with 'file', or JSON with 'audio_b64'")
        if len(b64) > MAX_AUDIO_BYTES * 4 // 3 + 4:
            raise GenError(413, "audio is larger than 25 MB")
        try:
            audio_bytes = base64.b64decode(b64, validate=True)
        except ValueError:
            raise GenError(400, "audio_b64 is not valid base64")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise GenError(413, "audio is larger than 25 MB")
    fmt = fields.get("response_format") or "json"
    if fmt not in ("json", "verbose_json", "text"):
        raise GenError(400, "response_format must be json, verbose_json or text")
    language = _text(fields, "language", 16, required=False)
    prompt = _text(fields, "prompt", 1000, required=False)
    model = fields.get("model") or "whisper-small"
    if not isinstance(model, str):
        raise GenError(400, "model must be a string")
    if model in ("whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"):   # OpenAI names
        model = "whisper-small"
    try:
        from .runtime import decode_audio
        audio = decode_audio(audio_bytes, max_seconds=MAX_AUDIO_SECONDS + 1)
    except ImportError as e:
        raise GenError(501, f"generation support is not installed ({e.name} missing): run ./install.sh --gen")
    except ValueError as e:
        raise GenError(400, str(e))
    if len(audio) > MAX_AUDIO_SECONDS * 16000:
        raise GenError(400, f"audio is longer than {MAX_AUDIO_SECONDS // 60} minutes")
    if len(audio) < 1600:
        raise GenError(400, "audio is shorter than 0.1 s")

    def run(eng):
        return eng.transcribe(audio, language=language, prompt=prompt, want_language=fmt == "verbose_json")
    key, (res, st) = _with_engine(mgr, tel, log, model, "transcribe", run)
    if fmt == "text":
        return "text/plain; charset=utf-8", res["text"].encode()
    body = {"text": res["text"]}
    if fmt == "verbose_json":
        body.update(task="transcribe", language=res["language"], duration=res["duration"], model=key, timings=st)
    import json
    return "application/json", json.dumps(body).encode()
