"""Thin client for the local OpenNPU server. Standard library only; every call is blocking and
is run off the GTK main thread by the caller (see app.run_bg)."""
import base64
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from opennpu import config

URL = config.URL


class ApiError(Exception):
    def __init__(self, msg, code=None, kind=None):
        super().__init__(msg)
        self.code, self.kind = code, kind

    @property
    def cancelled(self):
        return self.code == 499 or self.kind == "cancelled"


def _req(path, body=None, timeout=30, method=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data, {"Content-Type": "application/json", "X-OpenNPU-Client": "desktop"},
                               method=method)
    try:
        return urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read() or b"{}").get("error")
        except ValueError:
            err = None
        kind = err.get("type") if isinstance(err, dict) else None
        if isinstance(err, dict):
            err = err.get("message")
        raise ApiError(err or f"HTTP {e.code}", e.code, kind) from None
    except (urllib.error.URLError, OSError) as e:
        raise ApiError(f"server not reachable ({getattr(e, 'reason', e)})") from None


def get(path, timeout=10):
    return json.load(_req(path, timeout=timeout))


def post(path, body, timeout=3600):
    return json.load(_req(path, body, timeout=timeout))


def health():
    try:
        get("/health", timeout=2)
        return True
    except ApiError:
        return False


def chat_models():
    return get("/v1/models")["data"]


def gen_models():
    try:
        return list(get("/v1/gen/models").get("data") or [])
    except Exception:  # noqa: BLE001 - generation is optional; never take chat models down with it
        return []


def gen_progress():
    return get("/v1/gen/progress", timeout=3)


def gen_cancel():
    return post("/v1/gen/cancel", {}, timeout=10)


def usage(rng="7d"):
    return get(f"/api/usage?range={rng}", timeout=15)


def hardware(rng="24h"):
    return get(f"/api/hardware?range={rng}", timeout=15)


def usage_csv():
    return _req("/api/usage.csv", timeout=60).read()


def usage_settings(enabled):
    return post("/api/usage/settings", {"enabled": bool(enabled)}, timeout=15)


def usage_clear(what="usage"):
    return json.load(_req(f"/api/usage?what={what}", timeout=60, method="DELETE"))


def stats():
    return get("/api/stats", timeout=3)


def unload():
    return post("/unload", {}, timeout=60)


def abort(holder):
    """Shut the streaming connection from another thread: unblocks the reader at once (even while
    the server is still loading the model) and tells the server to stop generating."""
    import socket
    for r in list(holder):
        try:
            r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except Exception:  # noqa: BLE001 - already closed
            pass


def chat_stream(model, messages, think, cancelled, on_text, on_done, holder=None, session=None):
    """Streams a chat completion. cancelled() is polled between chunks; abort(holder) cuts the
    connection, and the server stops generating as soon as it next writes. With `session` the
    server stores the turn itself and keeps that session's KV cache warm."""
    body = {"model": model, "messages": messages, "stream": True, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": think}, "reasoning_effort": "medium" if think else "low"}
    if session:
        body["session"] = session
    r = _req("/v1/chat/completions", body, timeout=3600)
    if holder is not None:
        holder.append(r)
        if cancelled():
            abort(holder)
    stats = None
    try:
        for raw in r:
            if cancelled():
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            if "error" in d:
                raise ApiError(d["error"] if isinstance(d["error"], str) else d["error"].get("message"))
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                on_text(delta)
            stats = d.get("npu", stats)
    finally:
        r.close()
    on_done(stats)


# ---- server-side sessions (shared by every client on this machine)
def sessions_supported():
    try:
        get("/v1/sessions", timeout=5)
        return True
    except ApiError as e:
        if e.code in (404, 405, 501):
            return False
        raise


def sessions():
    return list(get("/v1/sessions", timeout=15).get("data") or [])


def session(sid):
    return get(f"/v1/sessions/{urllib.parse.quote(sid, safe='')}", timeout=30)


def session_create(title=None, model=None):
    return post("/v1/sessions", {k: v for k, v in (("title", title), ("model", model)) if v}, timeout=30)


def session_update(sid, **fields):
    return json.load(_req(f"/v1/sessions/{urllib.parse.quote(sid, safe='')}", fields, timeout=30, method="PATCH"))


def session_delete(sid):
    return json.load(_req(f"/v1/sessions/{urllib.parse.quote(sid, safe='')}", timeout=30, method="DELETE"))


def sessions_import(items):
    return post("/v1/sessions/import", {"sessions": items}, timeout=300)


def media(url):
    """Bytes of a session image (a server-relative URL such as /v1/sessions/media/<file>)."""
    if not url.startswith("/"):
        raise ApiError("only server-relative media URLs are fetched")
    return _req(url, timeout=60).read()


# ---- voice
def transcribe(wav_bytes, model=None):
    body = {"audio_b64": base64.b64encode(wav_bytes).decode(), "format": "wav"}
    if model:
        body["model"] = model
    return post("/v1/audio/transcriptions", body, timeout=600).get("text", "").strip()


def image(model, prompt, size, steps=None, seed=None, negative=None):
    body = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json"}
    for k, v in (("steps", steps), ("seed", seed), ("negative_prompt", negative)):
        if v not in (None, ""):
            body[k] = v
    d = post("/v1/images/generations", body)
    return base64.b64decode(d["data"][0]["b64_json"]), d.get("timings")


def video(model, prompt, size=None, frames=None, steps=None, seed=None):
    body = {"model": model, "prompt": prompt}
    for k, v in (("size", size), ("frames", frames), ("steps", steps), ("seed", seed)):
        if v not in (None, ""):
            body[k] = v
    d = post("/v1/videos/generations", body)
    return base64.b64decode(d["data"][0]["b64_json"]), d.get("timings")


def speech(model, text):
    r = _req("/v1/audio/speech", {"model": model, "input": text, "response_format": "wav"}, timeout=3600)
    return r.read()


# ---- Hugging Face Hub (search / details / downloads run in the server: one queue for every app)
def _rid(rid):
    return urllib.parse.quote(rid, safe="/")


def hub_search(q="", task="chat", sort="downloads", limit=30):
    qs = urllib.parse.urlencode({"q": q, "task": task, "sort": sort, "limit": limit})
    return list(get(f"/api/hub/search?{qs}", timeout=60).get("data") or [])


def hub_model(rid):
    return get(f"/api/hub/model?id={_rid(rid)}", timeout=60)


def hub_pull(rid, gguf=None, files=None):
    body = {"id": rid}
    if gguf:
        body["gguf"] = gguf
    if files:
        body["files"] = list(files)
    return post("/api/hub/pull", body, timeout=120)


def hub_pulls():
    return get("/api/hub/pulls", timeout=10)


def hub_cancel(rid):
    return post("/api/hub/cancel", {"id": rid}, timeout=30)


def hub_remove(rid):
    return json.load(_req(f"/api/hub/model?id={_rid(rid)}", timeout=120, method="DELETE"))


def service(action):
    """start | stop | restart the user service. Returns (ok, message)."""
    try:
        p = subprocess.run(["systemctl", "--user", action, config.SERVICE], capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return False, "systemctl not found: start the server with `npu start` or `python -m opennpu.native.server`"
    except subprocess.TimeoutExpired:
        return False, f"systemctl {action} timed out"
    return p.returncode == 0, (p.stderr or p.stdout).strip()


def doctor():
    """Runs `opennpu doctor --json` in the engine env. Returns a list of check dicts."""
    exe = config.HOME / "env" / "bin" / "opennpu"
    cmd = [str(exe), "doctor", "--json"] if exe.exists() else ["opennpu", "doctor", "--json"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise ApiError(f"could not run opennpu doctor: {e}")
    out = p.stdout[p.stdout.find("["):] if "[" in p.stdout else p.stdout      # skip any log noise before the JSON
    try:
        checks = json.loads(out)
    except ValueError:
        checks = None
    if not isinstance(checks, list) or not all(isinstance(c, dict) for c in checks):
        tail = (p.stderr or p.stdout or "doctor failed").strip().splitlines()
        raise ApiError(tail[-1] if tail else "doctor failed")
    return checks
