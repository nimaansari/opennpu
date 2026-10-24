"""API contract tests: the real server, fake engines (tests/qa/fakeserver.py). No NPU needed.

    python -m pytest tests/qa/test_api.py
"""
import base64
import http.client
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
M = "native:Fake-Qwen3-8B"


@pytest.fixture(scope="module")
def port():
    p = subprocess.Popen([sys.executable, str(HERE / "fakeserver.py"), "0"], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("READY"), line
    yield int(line.split()[1])
    p.kill()


def req(port, method, path, body=None, headers=None, raw=None, timeout=30):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json", **(headers or {})}
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, dict(r.getheaders()), out


def jreq(port, method, path, body=None, **kw):
    st, h, out = req(port, method, path, body, **kw)
    try:
        return st, json.loads(out)
    except ValueError:
        pytest.fail(f"{method} {path} -> {st}, not JSON: {out[:200]!r}")


def chat(port, content, **extra):
    return jreq(port, "POST", "/v1/chat/completions",
                {"model": M, "messages": [{"role": "user", "content": content}], **extra})


def sse(port, content, **extra):
    st, h, out = req(port, "POST", "/v1/chat/completions",
                     {"model": M, "messages": [{"role": "user", "content": content}], "stream": True, **extra})
    assert st == 200
    lines = [ln for ln in out.decode().split("\n") if ln.strip()]
    assert all(ln.startswith("data: ") for ln in lines), lines[:3]
    assert lines[-1] == "data: [DONE]"
    return h, [json.loads(ln[6:]) for ln in lines[:-1]]


# ------------------------------------------------------------------ discovery
def test_health_and_stats(port):
    assert jreq(port, "GET", "/health")[1]["status"] == "ok"
    st = jreq(port, "GET", "/api/stats")[1]
    for k in ("npu_busy_pct", "cpu_pct", "rss_bytes", "npu", "last", "loaded", "generating"):
        assert k in st


def test_models_lists_every_engine(port):
    data = jreq(port, "GET", "/v1/models")[1]
    assert data["object"] == "list"
    ids = {m["id"] for m in data["data"]}
    assert {"native:Fake-Qwen3-8B", "moe:Fake-Qwen3-30B-A3B", "native:Fake-Mistral-Nemo-12B", "universal:org/fake-small"} <= ids
    for m in data["data"]:
        assert m["object"] == "model" and m["engine_label"] and "size_bytes" in m


def test_ollama_tags_and_version(port):
    tags = jreq(port, "GET", "/api/tags")[1]["models"]
    assert tags and all({"name", "model", "size", "details"} <= set(t) for t in tags)
    assert jreq(port, "GET", "/api/version")[1]["version"]


def test_unknown_routes_are_json_404(port):
    assert jreq(port, "GET", "/v1/nope")[0] == 404
    assert jreq(port, "POST", "/v1/nope", {})[0] == 404


# ------------------------------------------------------------------ chat, OpenAI shape
def test_chat_non_stream(port):
    st, d = chat(port, "hello world")
    assert st == 200
    assert d["object"] == "chat.completion" and d["model"] == M
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "Echo: hello world "}
    assert d["choices"][0]["finish_reason"] == "stop"
    u = d["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["completion_tokens"] == 3
    assert d["npu"]["engine"] and d["npu"]["ttft_s"] is not None


def test_bare_model_name_resolves(port):
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": "Fake-Qwen3-8B", "messages": [{"role": "user", "content": "x"}]})
    assert st == 200 and d["model"] == M


def test_stream_matches_non_stream(port):
    h, chunks = sse(port, "stream me please")
    assert h.get("Content-Type", "").startswith("text/event-stream")
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
    assert text == chat(port, "stream me please")[1]["choices"][0]["message"]["content"]
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"     # OpenAI: first delta carries the role
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and "npu" in chunks[-1]


def test_finish_reason_length_when_truncated(port):
    st, d = chat(port, "one two three four five six", max_tokens=2)
    assert d["usage"]["completion_tokens"] == 2
    assert d["choices"][0]["finish_reason"] == "length"


def test_unicode_survives(port):
    assert "日本語" in chat(port, "__unicode__")[1]["choices"][0]["message"]["content"]


def test_think_block_passes_through(port):
    c = chat(port, "__think__")[1]["choices"][0]["message"]["content"]
    assert c.startswith("<think>") and "42" in c


@pytest.mark.parametrize("body,code", [
    ({"model": M}, 400),                                           # no messages
    ({"model": M, "messages": "hi"}, 400),                         # messages not a list
    ({"model": M, "messages": [{"role": "user"}]}, 400),           # no content
    ({"model": M, "messages": [{"role": "user", "content": "x"}], "max_tokens": "abc"}, 400),
    ({"model": M, "messages": [{"role": "user", "content": "x"}], "temperature": "hot"}, 400),
    ({"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]}, 404),
])
def test_bad_chat_requests_are_4xx_json(port, body, code):
    st, d = jreq(port, "POST", "/v1/chat/completions", body)
    assert st == code, d
    assert "error" in d


@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2, 3]", b'"just a string"', b"null"])
def test_malformed_bodies(port, raw):
    st, d = jreq(port, "POST", "/v1/chat/completions", raw=raw)
    assert st == 400, d


def test_oversized_body_rejected(port):
    st, _, _ = req(port, "POST", "/v1/chat/completions", raw=b"x" * 10, headers={"Content-Length": str(200 * 2**20)})
    assert st == 413


def test_engine_error_non_stream_is_500_json(port):
    st, d = chat(port, "__error0__")
    assert st == 500 and "error" in d
    assert chat(port, "still alive")[0] == 200


def test_engine_error_mid_stream_is_reported_and_server_recovers(port):
    st, h, out = req(port, "POST", "/v1/chat/completions",
                     {"model": M, "messages": [{"role": "user", "content": "__error__ x"}], "stream": True})
    body = out.decode()
    assert '"error"' in body and body.rstrip().endswith("data: [DONE]")
    assert chat(port, "after error")[0] == 200


def test_client_disconnect_mid_stream_frees_the_model_quickly(port):
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps({"model": M, "messages": [{"role": "user", "content": "__slow__"}], "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    s.recv(4096)
    time.sleep(0.3)
    s.close()                                       # user hit stop / closed the tab
    t = time.time()
    assert chat(port, "next one", max_tokens=3)[0] == 200
    assert time.time() - t < 3, "generation for a gone client kept the model busy"


def test_concurrent_requests_are_serialised_not_mixed(port):
    out = {}

    def go(i):
        out[i] = chat(port, f"client{i} a b c")[1]["choices"][0]["message"]["content"]
    ts = [threading.Thread(target=go, args=(i,)) for i in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(out[i] == f"Echo: client{i} a b c " for i in range(5))


# ------------------------------------------------------------------ Ollama shape
def test_ollama_chat_stream_ndjson(port):
    st, h, out = req(port, "POST", "/api/chat", {"model": M, "messages": [{"role": "user", "content": "hi ollama"}]})
    assert st == 200
    rows = [json.loads(x) for x in out.decode().splitlines() if x.strip()]
    assert all(not r["done"] for r in rows[:-1]) and rows[-1]["done"] is True
    assert "".join(r["message"]["content"] for r in rows) == "Echo: hi ollama "
    assert rows[-1]["eval_count"] == 3 and rows[-1]["done_reason"] == "stop"


def test_ollama_generate_non_stream(port):
    st, d = jreq(port, "POST", "/api/generate", {"model": M, "prompt": "hey", "stream": False})
    assert st == 200 and d["done"] and d["response"] == "Echo: hey "


def test_ollama_ps(port):
    chat(port, "load it")
    assert jreq(port, "GET", "/api/ps")[1]["models"][0]["name"] == M


# ------------------------------------------------------------------ embeddings
def test_embeddings_openai_and_ollama(port):
    st, d = jreq(port, "POST", "/v1/embeddings", {"model": "fake/embed", "input": ["a", "b c"]})
    assert st == 200 and len(d["data"]) == 2 and d["data"][1]["index"] == 1
    assert abs(sum(x * x for x in d["data"][0]["embedding"]) - 1) < 1e-6
    st, d = jreq(port, "POST", "/api/embed", {"model": "fake/embed", "input": "single"})
    assert st == 200 and len(d["embeddings"]) == 1


@pytest.mark.parametrize("inp", [None, [], 5, {"a": 1}, [1, 2]])
def test_embeddings_bad_input(port, inp):
    st, d = jreq(port, "POST", "/v1/embeddings", {"model": "fake/embed", "input": inp})
    assert st == 400, d


# ------------------------------------------------------------------ security
def test_foreign_origin_blocked_local_allowed(port):
    for o in ("https://evil.example", "null", "http://127.0.0.1.evil.example", "http://localhost.evil.example"):
        assert req(port, "POST", "/v1/chat/completions", {}, headers={"Origin": o})[0] == 403, o
    st, h, _ = req(port, "POST", "/v1/chat/completions",
                   {"model": M, "messages": [{"role": "user", "content": "x"}]}, headers={"Origin": "http://localhost:3000"})
    assert st == 200 and h.get("Access-Control-Allow-Origin") == "http://localhost:3000"


def test_foreign_origin_cannot_read_gets(port):
    st, h, _ = req(port, "GET", "/v1/models", headers={"Origin": "https://evil.example"})
    assert h.get("Access-Control-Allow-Origin") is None


def test_preflight(port):
    assert req(port, "OPTIONS", "/v1/chat/completions", headers={"Origin": "http://localhost:5173"})[0] == 204
    assert req(port, "OPTIONS", "/v1/chat/completions", headers={"Origin": "https://evil.example"})[0] == 403


@pytest.mark.parametrize("path", ["/../native/server.py", "/icons/../../config.py", "/%2e%2e/config.py",
                                  "/..%2f..%2fconfig.py", "//etc/passwd", "/icons/%2e%2e/%2e%2e/config.py"])
def test_static_path_traversal(port, path):
    st, _, out = req(port, "GET", path)
    assert st == 404 and b"import" not in out


def test_web_ui_served(port):
    st, h, out = req(port, "GET", "/")
    assert st == 200 and h["Content-Type"].startswith("text/html") and b"app.js" in out
    for f, ct in (("/app.js", "text/javascript"), ("/app.css", "text/css"), ("/gen.js", "text/javascript"),
                  ("/icons/cpu.svg", "image/svg+xml")):
        st, h, _ = req(port, "GET", f)
        assert st == 200 and h["Content-Type"].startswith(ct), f
    assert "nosniff" in req(port, "GET", "/")[1].get("X-Content-Type-Options", "")


def test_unload(port):
    chat(port, "x")
    assert jreq(port, "POST", "/unload", {})[1]["unloaded"]
    assert jreq(port, "GET", "/health")[1]["loaded"] is None


# ------------------------------------------------------------------ generation (opennpu.gen.http)
def test_gen_models(port):
    d = jreq(port, "GET", "/v1/gen/models")[1]
    assert d["data"] and all({"id", "task", "repo", "ram_gb", "downloaded", "note"} <= set(m) for m in d["data"])


def test_image_generation(port):
    st, d = jreq(port, "POST", "/v1/images/generations", {"prompt": "fox", "size": "64x64", "n": 2})
    assert st == 200 and len(d["data"]) == 2
    assert base64.b64decode(d["data"][0]["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("body,code", [
    ({}, 400), ({"prompt": "x", "size": "big"}, 400), ({"prompt": "x", "size": "10x10"}, 400),
    ({"prompt": "x", "response_format": "url"}, 400), ({"prompt": "x", "n": "abc"}, 400),
    ({"prompt": "x", "seed": "abc"}, 400), ({"prompt": "x", "model": "image:org/gated-thing"}, 403),
    ({"prompt": "x", "model": "image:org/missing-thing"}, 404), ({"prompt": "x", "model": "image:org/huge-thing"}, 503),
    ({"prompt": "x", "model": "mms-tts-eng"}, 400),                     # speech model on the image endpoint
])
def test_image_errors(port, body, code):
    st, d = jreq(port, "POST", "/v1/images/generations", body)
    assert st == code, d
    assert d["error"]["message"]


def test_video_generation(port):
    st, d = jreq(port, "POST", "/v1/videos/generations", {"prompt": "waves", "size": "64x64"})
    assert st == 200 and d["data"][0]["mime_type"] == "video/mp4"


def test_speech(port):
    st, h, out = req(port, "POST", "/v1/audio/speech", {"input": "hello", "response_format": "wav"})
    assert st == 200 and h["Content-Type"] == "audio/wav" and out[:4] == b"RIFF"
    st, h, out = req(port, "POST", "/v1/audio/speech", {"model": "tts-1", "voice": "alloy", "input": "hi", "response_format": "pcm"})
    assert st == 200 and out[:4] != b"RIFF"
    for bad in ({"response_format": "wav"}, {"input": "x", "response_format": "midi"}):
        st, d = jreq(port, "POST", "/v1/audio/speech", bad)
        assert st == 400


def test_chat_after_generation_reloads_chat_model(port):
    jreq(port, "POST", "/v1/images/generations", {"prompt": "fox", "size": "64x64"})
    assert chat(port, "back to chat")[0] == 200


# ------------------------------------------------------------------ model refs (security)
@pytest.mark.parametrize("ref", ["universal:/tmp/evil.pt", "universal:../../etc", "universal:org/x.pt",
                                 "universal:/home/user/model", "universal:a/b/c", "embed:/etc/passwd",
                                 "universal:org/../secret", "universal:-rf/x"])
def test_only_repo_ids_reach_the_loaders(port, ref):
    st, d = chat(port, "x") if False else jreq(port, "POST", "/v1/chat/completions",
                                                {"model": ref, "messages": [{"role": "user", "content": "x"}]})
    assert st == 404, d
    st, d = jreq(port, "POST", "/v1/embeddings", {"model": ref, "input": "x"})
    assert st == 404, d


def test_typo_repo_is_404_and_keeps_the_loaded_model(port):
    assert chat(port, "load the real one")[0] == 200
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": "Qwen/Qwen3-8b-typo", "messages": [{"role": "user", "content": "x"}]})
    assert st == 404 and "spelling" in d["error"]["message"]
    assert jreq(port, "GET", "/health")[1]["loaded"] == M        # not evicted by a typo


@pytest.mark.parametrize("ref,code", [("org/gated-model", 403), ("org/enormous-model", 507)])
def test_hub_refusals_before_unloading(port, ref, code):
    chat(port, "x")
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": ref, "messages": [{"role": "user", "content": "x"}]})
    assert st == code, d
    assert jreq(port, "GET", "/health")[1]["loaded"] == M


def test_stream_error_status_for_typo(port):
    st, h, out = req(port, "POST", "/v1/chat/completions",
                     {"model": "org/also-typo", "messages": [{"role": "user", "content": "x"}], "stream": True})
    assert st == 404


def test_retrieve_model(port):
    st, d = jreq(port, "GET", "/v1/models/native:Fake-Qwen3-8B")
    assert st == 200 and d["id"] == M and d["object"] == "model" and "created" in d
    assert jreq(port, "GET", "/v1/models/Fake-Qwen3-8B")[1]["id"] == M
    assert jreq(port, "GET", "/v1/models/nope")[0] == 404
    assert jreq(port, "GET", "/statusXYZ")[0] == 404


def test_chunked_body_is_411(port):
    st, _, _ = req(port, "POST", "/v1/chat/completions", raw=b"5\r\n{}   \r\n0\r\n\r\n",
                   headers={"Transfer-Encoding": "chunked"})
    assert st == 411


def test_ollama_preload_and_keep_alive_unload(port):
    jreq(port, "POST", "/unload", {})
    st, d = jreq(port, "POST", "/api/generate", {"model": M})
    assert st == 200 and d["done_reason"] == "load"
    assert jreq(port, "GET", "/health")[1]["loaded"] == M
    st, d = jreq(port, "POST", "/api/generate", {"model": M, "keep_alive": 0})
    assert d["done_reason"] == "unload" and jreq(port, "GET", "/health")[1]["loaded"] is None


def test_stats_cpu_is_stable_across_concurrent_polls(port):
    vals = []
    ts = [threading.Thread(target=lambda: vals.append(jreq(port, "GET", "/api/stats")[1]["cpu_pct"])) for _ in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(vals)) == 1


def test_context_overflow_is_400_with_advice(port):
    st, d = chat(port, "__toolong__")
    assert st == 400 and "Start a new conversation" in d["error"]["message"]
    st, h, out = req(port, "POST", "/v1/chat/completions",
                     {"model": M, "messages": [{"role": "user", "content": "__toolong__"}], "stream": True})
    ev = [json.loads(x[6:]) for x in out.decode().split("\n") if x.startswith("data: {")]
    assert ev[-1]["error"]["code"] == 400


# ------------------------------------------------------------------ local analytics
def test_usage_records_requests_without_content(port):
    jreq(port, "DELETE", "/api/usage?what=usage")
    chat(port, "secret words here", max_tokens=8)
    req(port, "POST", "/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": "web call"}]},
        headers={"X-OpenNPU-Client": "web"})
    chat(port, "__error0__")
    jreq(port, "POST", "/v1/embeddings", {"model": "fake/embed", "input": ["a b", "c"]})
    jreq(port, "POST", "/v1/images/generations", {"prompt": "fox", "size": "64x64"})
    jreq(port, "POST", "/v1/images/generations", {"prompt": "x", "size": "bad"})          # 400: never ran, not counted
    time.sleep(0.5)
    d = jreq(port, "GET", "/api/usage?range=24h")[1]
    t = d["totals"]
    assert t["requests"] == 5 and t["errors"] == 1 and t["completion_tokens"] >= 4
    assert d["by_kind"] == {"chat": 3, "embed": 1, "image": 1}
    assert d["by_client"].get("web") == 1
    assert {m["model"] for m in d["by_model"]} >= {M, "embed:fake/embed"}
    assert sum(d["hour_of_day"]) == 5 and len(d["timeline"]) >= 24
    raw = json.dumps(d)
    assert "secret words" not in raw and "web call" not in raw                # metadata only, never text
    st, h, csv = req(port, "GET", "/api/usage.csv")
    assert st == 200 and h["Content-Type"].startswith("text/csv") and csv.decode().count("\n") == 6
    assert b"secret" not in csv


def test_usage_toggle_and_clear(port):
    assert jreq(port, "POST", "/api/usage/settings", {"enabled": False})[1]["enabled"] is False
    n = jreq(port, "GET", "/api/usage")[1]["totals"]["requests"]
    chat(port, "not recorded")
    time.sleep(0.3)
    assert jreq(port, "GET", "/api/usage")[1]["totals"]["requests"] == n
    assert jreq(port, "POST", "/api/usage/settings", {"enabled": True})[1]["enabled"] is True
    assert jreq(port, "POST", "/api/usage/settings", {"enabled": "yes"})[0] == 400
    assert jreq(port, "DELETE", "/api/usage?what=usage")[1]["cleared"] == "usage"
    assert jreq(port, "GET", "/api/usage")[1]["totals"]["requests"] == 0
    assert jreq(port, "DELETE", "/api/usage?what=everything")[0] == 400


def test_hardware_history_and_info(port):
    d = jreq(port, "GET", "/api/hardware?range=24h")[1]
    assert d["series"] and {"cpu", "ram_used", "npu_busy", "cpu_temp"} <= set(d["series"][0])
    assert d["info"]["cpu"] and d["info"]["ram_total"] > 0 and d["info"]["npu"]["device"] == "RyzenAI-npu6"
    assert d["step_s"] == 900
    assert len(jreq(port, "GET", "/api/hardware?range=1h")[1]["series"]) >= 50


def test_analytics_are_local_origin_only(port):
    for method, path in (("DELETE", "/api/usage"), ("POST", "/api/usage/settings")):
        assert req(port, method, path, {"enabled": True}, headers={"Origin": "https://evil.example"})[0] == 403
    st, h, _ = req(port, "GET", "/api/usage", headers={"Origin": "https://evil.example"})
    assert h.get("Access-Control-Allow-Origin") is None


# ------------------------------------------------------------------ sessions (saved on this computer)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_session_lifecycle_and_server_side_save(port):
    s = jreq(port, "POST", "/v1/sessions", {"model": M})[1]
    sid = s["id"]
    msgs = [{"role": "user", "content": "first question here"}]
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": M, "messages": msgs, "session": sid})
    assert st == 200 and d["session"] == sid
    got = jreq(port, "GET", f"/v1/sessions/{sid}")[1]
    assert got["title"] == "first question here" and got["model"] == M
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]
    assert got["messages"][1]["content"] == "Echo: first question here " and got["messages"][1]["stats"]["engine"]
    # follow-up: the server reuses what it already processed for this session
    msgs2 = got["messages"] + [{"role": "user", "content": "and a second one"}]
    d2 = jreq(port, "POST", "/v1/chat/completions", {"model": M, "messages": msgs2, "session": sid})[1]
    assert d2["npu"]["cached_tokens"] > 3
    assert len(jreq(port, "GET", f"/v1/sessions/{sid}")[1]["messages"]) == 4
    # an edited history replaces the stored one (no append)
    jreq(port, "POST", "/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": "edited"}], "session": sid})
    assert len(jreq(port, "GET", f"/v1/sessions/{sid}")[1]["messages"]) == 2
    assert sid in [x["id"] for x in jreq(port, "GET", "/v1/sessions")[1]["data"]]
    assert jreq(port, "PATCH", f"/v1/sessions/{sid}", {"title": "Renamed"})[1]["title"] == "Renamed"
    assert jreq(port, "DELETE", f"/v1/sessions/{sid}")[1]["deleted"]
    assert jreq(port, "GET", f"/v1/sessions/{sid}")[0] == 404


def test_session_new_and_stream_reports_id(port):
    st, h, out = req(port, "POST", "/v1/chat/completions",
                     {"model": M, "stream": True, "session": "new", "messages": [{"role": "user", "content": "hello stream"}]})
    sid = h["X-OpenNPU-Session"]
    last = [json.loads(x[6:]) for x in out.decode().split("\n") if x.startswith("data: {")][-1]
    assert last["npu"]["session"] == sid
    assert jreq(port, "GET", f"/v1/sessions/{sid}")[1]["messages"][1]["content"] == "Echo: hello stream "


def test_session_keeps_partial_reply_when_client_disconnects(port):
    sid = jreq(port, "POST", "/v1/sessions", {})[1]["id"]
    s = socket.create_connection(("127.0.0.1", port))
    b = json.dumps({"model": M, "stream": True, "session": sid, "messages": [{"role": "user", "content": "__slow__"}]}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    s.recv(4096)
    time.sleep(0.4)
    s.close()
    time.sleep(0.6)
    m = jreq(port, "GET", f"/v1/sessions/{sid}")[1]["messages"][-1]
    assert m["stopped"] is True and m["content"].startswith("s0")


def test_unknown_session_is_404_before_anything_runs(port):
    assert jreq(port, "POST", "/v1/chat/completions", {"model": M, "session": "nope",
                                                       "messages": [{"role": "user", "content": "x"}]})[0] == 404


def test_session_import(port):
    n = jreq(port, "POST", "/v1/sessions/import", {"sessions": [
        {"id": "old-1", "title": "Old chat", "model": M, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"id": "old-1", "messages": []}, "garbage"]})[1]["imported"]
    assert n == 1 and jreq(port, "GET", "/v1/sessions/old-1")[1]["title"] == "Old chat"


# ------------------------------------------------------------------ vision
def test_images_reach_vision_models_and_are_stored_as_files(port):
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    msg = [{"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": url}}]}]
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": "universal:org/fake-vlm", "messages": msg, "session": "new"})
    assert st == 200 and d["choices"][0]["message"]["content"].startswith(f"I see 1 image ({len(PNG)} bytes)")
    saved = jreq(port, "GET", f"/v1/sessions/{d['session']}")[1]["messages"][0]
    assert saved["content"] == "what is this" and saved["images"][0].startswith("/v1/sessions/media/")
    st, h, b = req(port, "GET", saved["images"][0])
    assert st == 200 and h["Content-Type"] == "image/png" and b == PNG
    # the stored message can be sent back as-is (history with images)
    st, d2 = jreq(port, "POST", "/v1/chat/completions", {"model": "universal:org/fake-vlm", "messages": [saved]})
    assert st == 200 and "I see 1 image" in d2["choices"][0]["message"]["content"]
    assert any(m.get("vision") for m in jreq(port, "GET", "/v1/models")[1]["data"])


@pytest.mark.parametrize("url,why", [("https://example.com/x.png", "remote"), ("data:image/svg+xml;base64,PHN2Zz4=", "PNG, JPEG"),
                                     ("data:image/png;base64," + base64.b64encode(b"x" * (11 * 2**20)).decode(), "limit"),
                                     ("/v1/sessions/media/../../sessions.db", "unknown image")])
def test_bad_images_are_400(port, url, why):
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": "universal:org/fake-vlm", "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]})
    assert st in (400, 413) and (st == 413 or why in d["error"]["message"]), d


def test_text_model_refuses_images(port):
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]})
    assert st == 400 and "can't see images" in d["error"]["message"]


# ------------------------------------------------------------------ model pool
def test_several_models_stay_resident(port):
    jreq(port, "POST", "/unload", {})
    for m in (M, "native:Fake-Mistral-Nemo-12B", "universal:org/fake-small"):
        chat(port, "x", model=m) if False else jreq(port, "POST", "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": "x"}]})
    res = jreq(port, "GET", "/health")[1]["resident"]
    assert set(res) == {M, "native:Fake-Mistral-Nemo-12B", "universal:org/fake-small"} and res[0] == "universal:org/fake-small"
    assert len(jreq(port, "GET", "/api/ps")[1]["models"]) == 3
    jreq(port, "POST", "/unload", {"model": "native:Fake-Mistral-Nemo-12B"})
    assert "native:Fake-Mistral-Nemo-12B" not in jreq(port, "GET", "/health")[1]["resident"]


def test_voice_models_do_not_evict_the_chat_model(port):
    chat(port, "load the chat model")
    assert M in jreq(port, "GET", "/health")[1]["resident"]
    st, h, wav = req(port, "POST", "/v1/audio/speech", {"model": "mms-tts-eng", "input": "hello there", "response_format": "wav"})
    assert st == 200
    st, h, out = req(port, "POST", "/v1/audio/transcriptions",
                     {"audio_b64": base64.b64encode(wav).decode(), "format": "wav"})
    res = jreq(port, "GET", "/health")[1]["resident"]
    assert M in res, res                                   # read-aloud and voice input never evict chat
    assert any(k.startswith("speech:") for k in res)


def test_fastflowlm_models_are_a_clear_404(port):
    st, d = jreq(port, "POST", "/v1/chat/completions", {"model": "flm:gpt-oss:20b", "messages": [{"role": "user", "content": "x"}]})
    assert st == 404 and "FastFlowLM support was removed" in d["error"]["message"]
    assert not any(m["engine"] == "flm" for m in jreq(port, "GET", "/v1/models")[1]["data"])


# ------------------------------------------------------------------ Hugging Face Hub (fake HfApi + fake `hf download`)
def _err_text(d):
    e = d.get("error")
    return e.get("message", "") if isinstance(e, dict) else str(e or "")


def _wait_pull(port, rid, states=("done", "failed", "cancelled"), timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = next((j for j in jreq(port, "GET", "/api/hub/pulls")[1]["data"] if j["id"] == rid), None)
        if job and job["state"] in states:
            return job
        time.sleep(0.2)
    pytest.fail(f"{rid} did not reach {states}")


def test_hub_search_cards_and_plans(port):
    st, d = jreq(port, "GET", "/api/hub/search?q=qwen3&task=chat&sort=downloads&limit=30")
    assert st == 200
    cards = {c["id"]: c for c in d["data"]}
    assert "Qwen/Qwen3-0.6B" in cards and "Qwen/Qwen3-VL-2B-Instruct" not in cards       # task filter
    c = cards["Qwen/Qwen3-0.6B"]
    for k in ("downloads", "likes", "task", "params", "gguf", "model_type", "gated", "updated", "library", "installed", "plan"):
        assert k in c, k
    assert c["plan"]["engine"] == "native" and c["plan"]["fast"] is True and c["updated"] == "2026-09-30"
    assert cards["Qwen/Qwen3-30B-A3B"]["plan"]["engine"] == "moe"
    downloads = [c["downloads"] for c in d["data"]]
    assert downloads == sorted(downloads, reverse=True)
    allc = {c["id"]: c for c in jreq(port, "GET", "/api/hub/search?task=all&limit=60")[1]["data"]}
    assert allc["meta-llama/Llama-3.2-1B-Instruct"]["gated"] is True
    assert allc["unsloth/gpt-oss-20b-GGUF"]["gguf"] is True and allc["unsloth/gpt-oss-20b-GGUF"]["plan"]["engine"] in ("llama", "universal")
    big = allc["deepseek-ai/DeepSeek-V3"]["plan"]
    assert big["fits"] is False and "RAM" in big["note"]
    assert allc["stabilityai/sd-turbo"]["plan"]["engine"] == "gen" and allc["BAAI/bge-small-en-v1.5"]["plan"]["engine"] == "embed"
    likes = [c["likes"] for c in jreq(port, "GET", "/api/hub/search?task=all&sort=likes")[1]["data"]]
    assert likes == sorted(likes, reverse=True)


@pytest.mark.parametrize("qs", ["limit=abc", "limit=1e9x"])
def test_hub_search_bad_limit_is_400(port, qs):
    st, d = jreq(port, "GET", f"/api/hub/search?{qs}")
    assert st == 400 and "limit" in _err_text(d)


def test_hub_search_offline_is_502(port):
    st, d = jreq(port, "GET", "/api/hub/search?q=offline")
    assert st == 502 and "offline" in _err_text(d)


def test_hub_details(port):
    st, d = jreq(port, "GET", "/api/hub/model?id=unsloth/gpt-oss-20b-GGUF")
    assert st == 200
    assert [g["name"] for g in d["gguf"]] == ["gpt-oss-20b-Q4_K_M.gguf", "gpt-oss-20b-Q8_0.gguf", "gpt-oss-20b-F16.gguf"]
    assert d["download"]["files"] == ["gpt-oss-20b-Q4_K_M.gguf"]                # one GGUF, Q4_K_M by default
    assert d["download"]["bytes"] == 12 * 2**20 and d["free_disk"] > 0 and d["license"] == "apache-2.0"
    st, d = jreq(port, "GET", "/api/hub/model?id=Qwen/Qwen3-0.6B")
    assert d["plan"]["engine"] == "native" and d["architectures"] == ["Qwen3ForCausalLM"]       # full config.json read
    assert "README.md" not in d["download"]["files"] and "model.safetensors" in d["download"]["files"]


@pytest.mark.parametrize("rid", ["", "noslash", "../etc/passwd", "a/../b", "org/name/extra", "-x/y"])
def test_hub_bad_ids_are_400(port, rid):
    import urllib.parse
    st, d = jreq(port, "GET", "/api/hub/model?id=" + urllib.parse.quote(rid))
    assert st == 400 and "org/name" in _err_text(d)
    st, d = jreq(port, "POST", "/api/hub/pull", {"id": rid})
    assert st == 400


def test_hub_unknown_model_is_404_offline_502(port):
    st, d = jreq(port, "GET", "/api/hub/model?id=nobody/no-such-model")
    assert st == 404 and "no such model" in _err_text(d)
    assert jreq(port, "GET", "/api/hub/model?id=org/offline-model")[0] == 502
    st, d = jreq(port, "POST", "/api/hub/pull", {"id": "nobody/no-such-model"})
    assert st == 404 and _err_text(d)


@pytest.mark.parametrize("body", [{}, {"id": 5}, {"id": "Qwen/Qwen3-0.6B", "files": "model.safetensors"},
                                  {"id": "Qwen/Qwen3-0.6B", "files": [1, 2]}])
def test_hub_pull_validation(port, body):
    assert jreq(port, "POST", "/api/hub/pull", body)[0] == 400


def test_hub_pull_gated_and_unknown_file_are_400(port):
    st, d = jreq(port, "POST", "/api/hub/pull", {"id": "meta-llama/Llama-3.2-1B-Instruct"})
    assert st == 400 and "gated" in _err_text(d) and "hf auth login" in _err_text(d)
    st, d = jreq(port, "POST", "/api/hub/pull", {"id": "Qwen/Qwen3-0.6B", "files": ["nope.bin"]})
    assert st == 400 and "nope.bin" in _err_text(d)


def test_hub_pull_progress_done_installed_listed_and_removed(port):
    rid = "Qwen/Qwen3-0.6B"
    jreq(port, "DELETE", "/api/hub/model?id=" + rid)                          # from an earlier run, if any
    st, job = jreq(port, "POST", "/api/hub/pull", {"id": rid})
    assert st == 200 and job["state"] == "downloading" and job["engine"] == "native" and job["total"] > 0
    assert jreq(port, "POST", "/api/hub/pull", {"id": rid})[1]["state"] == "downloading"     # same job, not a second one
    seen = []
    t0 = time.time()
    while time.time() - t0 < 20:
        d = jreq(port, "GET", "/api/hub/pulls")[1]
        assert d["free_disk"] > 0
        j = next(x for x in d["data"] if x["id"] == rid)
        for k in ("id", "state", "bytes", "total", "rate", "eta_s", "error", "engine", "model"):
            assert k in j, k
        seen.append(j["bytes"])
        if j["state"] == "done":
            break
        time.sleep(0.2)
    assert j["state"] == "done" and j["bytes"] == j["total"] and j["model"] == "native:Qwen3-0.6B"
    assert any(0 < b < j["total"] for b in seen)                               # progress was visible on the way
    card = next(c for c in jreq(port, "GET", "/api/hub/search?q=qwen3-0.6b")[1]["data"] if c["id"] == rid)
    assert card["installed"] is True
    time.sleep(5.2)                                                            # the model catalog rescans every 5 s
    assert "native:Qwen3-0.6B" in {m["id"] for m in jreq(port, "GET", "/v1/models")[1]["data"]}
    assert chat(port, "hi", model="native:Qwen3-0.6B")[0] == 200               # and it chats
    st, d = jreq(port, "DELETE", "/api/hub/model?id=" + rid)
    assert st == 200 and d["deleted"] == rid
    assert "native:Qwen3-0.6B" not in {m["id"] for m in jreq(port, "GET", "/v1/models")[1]["data"]}
    assert jreq(port, "GET", "/api/hub/model?id=" + rid)[1]["installed"] is False
    assert jreq(port, "DELETE", "/api/hub/model?id=" + rid)[0] == 404          # nothing left to remove


def test_hub_universal_gguf_variant_goes_to_the_hf_cache(port):
    rid = "unsloth/gpt-oss-20b-GGUF"
    st, job = jreq(port, "POST", "/api/hub/pull", {"id": rid, "gguf": "gpt-oss-20b-Q8_0.gguf"})
    assert st == 200 and job["total"] == 16 * 2**20                             # just that GGUF (README skipped)
    job = _wait_pull(port, rid)
    assert job["state"] == "done" and job["model"] in ("universal:" + rid, "llama:" + rid)
    assert jreq(port, "DELETE", "/api/hub/model?id=" + rid)[0] == 200


def test_hub_cancel_and_failed_download(port):
    rid = "opennpu-test/Slow-Qwen3-1.7B"
    assert jreq(port, "POST", "/api/hub/pull", {"id": rid})[0] == 200
    time.sleep(1.0)
    st, d = jreq(port, "POST", "/api/hub/cancel", {"id": rid})
    assert st == 200 and d == {"id": rid, "state": "cancelled"}
    job = next(j for j in jreq(port, "GET", "/api/hub/pulls")[1]["data"] if j["id"] == rid)
    assert job["state"] == "cancelled" and job["bytes"] < job["total"]
    assert jreq(port, "POST", "/api/hub/cancel", {"id": rid})[0] == 200        # a finished job: cleared from the list
    assert rid not in [j["id"] for j in jreq(port, "GET", "/api/hub/pulls")[1]["data"]]
    assert jreq(port, "POST", "/api/hub/cancel", {"id": rid})[0] == 404
    assert jreq(port, "POST", "/api/hub/cancel", {})[0] == 400
    job = (jreq(port, "POST", "/api/hub/pull", {"id": "opennpu-test/Broken-Qwen3-4B"}), _wait_pull(port, "opennpu-test/Broken-Qwen3-4B"))[1]
    assert job["state"] == "failed" and "500 Server Error" in job["error"] and job["model"] is None
