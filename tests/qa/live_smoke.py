"""Live smoke test: the running OpenNPU server with REAL models on the NPU. One model at a time.

    python tests/qa/live_smoke.py [--quick]        (server at $OPENNPU_URL, default :11540)

Prints one line per check and a summary; exit code 1 if anything failed. Skips engines whose
models are not installed. Heavy checks (30B MoE, image) are skipped with --quick.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("OPENNPU_URL", "http://127.0.0.1:11540")
QUICK = "--quick" in sys.argv
RESULTS = []


def call(method, path, body=None, timeout=1800, raw=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data or b"{}"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def stream(model, content, max_tokens=64):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(
        {"model": model, "stream": True, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": content}]}).encode(), {"Content-Type": "application/json"})
    text, last, role = "", None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if "error" in d:
                raise RuntimeError(d["error"])
            ch = d["choices"][0]
            role = role or ch["delta"].get("role")
            text += ch["delta"].get("content") or ""
            last = d
    return text, last, role


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        RESULTS.append((True, name))
        print(f"PASS  {name:52s} {time.time() - t:6.1f}s  {detail or ''}", flush=True)
    except SkipTest as e:
        print(f"SKIP  {name:52s} {e}", flush=True)
    except Exception as e:  # noqa: BLE001
        RESULTS.append((False, name))
        print(f"FAIL  {name:52s} {time.time() - t:6.1f}s  {type(e).__name__}: {e}", flush=True)


class SkipTest(Exception):
    pass


def first(models, engine, hint=""):
    ms = [m["id"] for m in models if m["engine"] == engine and hint.lower() in m["id"].lower()]
    if not ms:
        raise SkipTest(f"no {engine} model{' like ' + hint if hint else ''} installed")
    return ms[0]


def main():
    st, d = call("GET", "/v1/models")
    assert st == 200, "server not reachable"
    models = d["data"]
    print(f"{len(models)} models installed at {URL}\n", flush=True)

    def native():
        m = first(models, "native", "8B")
        text, last, role = stream(m, "What is 17*23? Answer with just the number.", 16)
        assert "391" in text, text
        assert role == "assistant" and last["npu"]["decode_tok_s"] > 2
        return f"{last['npu']['decode_tok_s']} tok/s  {text.strip()[:40]!r}"
    check("native dense engine (stream, OpenAI)", native)

    def ollama():
        m = first(models, "native", "8B")
        st, d = call("POST", "/api/chat", {"model": m, "stream": False, "messages": [{"role": "user", "content": "Capital of France? One word."}]})
        assert st == 200 and "paris" in d["message"]["content"].lower(), d
        return repr(d["message"]["content"].strip()[:30])
    check("native engine via Ollama /api/chat", ollama)

    def typo_keeps_model():
        before = call("GET", "/health")[1]["loaded"]
        st, d = call("POST", "/v1/chat/completions", {"model": "Qwen/Qwen3-8b-definitely-not-real-zz",
                                                       "messages": [{"role": "user", "content": "x"}]})
        assert st == 404, (st, d)
        assert call("GET", "/health")[1]["loaded"] == before
        st, d = call("POST", "/v1/chat/completions", {"model": "universal:/tmp/x.pt", "messages": [{"role": "user", "content": "x"}]})
        assert st == 404
        return f"404 + {before} still loaded"
    check("typo / path model refs rejected, model kept", typo_keeps_model)

    def universal():
        text, last, _ = stream("Qwen/Qwen3-0.6B", "What is 2+3? Answer with just the number.", 24)
        assert "5" in text, text
        return f"{last['npu']['decode_tok_s']} tok/s {text.strip()[:30]!r}"
    check("universal engine (HF repo by name)", universal)

    def llama_gguf(hint="", max_gb=8.0):
        ms = [m for m in models if m["engine"] == "llama" and hint.lower() in m["id"].lower()
              and m.get("size_bytes", 0) <= max_gb * 1e9]
        if not ms:
            raise SkipTest(f"no GGUF model{' like ' + hint if hint else ''} under {max_gb:g} GB installed")
        m = min(ms, key=lambda m: m.get("size_bytes", 0))["id"]
        text, last, _ = stream(m, "What is the capital of France? Answer with one word.", 48)
        assert "paris" in text.lower(), text
        return f"{m.split(':', 1)[1]}: {last['npu']['decode_tok_s']} tok/s {text.strip()[:20]!r}"
    check("llama.cpp engine (GGUF, smallest installed)", llama_gguf)
    if not QUICK:
        check("llama.cpp engine (GGUF mixture-of-experts)", lambda: llama_gguf("A3B", 30.0))

    def embeddings():
        st, d = call("POST", "/v1/embeddings", {"model": "BAAI/bge-small-en-v1.5", "input": ["the cat sat", "a cat was sitting", "stock market crash"]})
        assert st == 200, d
        v = [e["embedding"] for e in d["data"]]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
        assert dot(v[0], v[1]) > dot(v[0], v[2]) + 0.2
        return f"sim {dot(v[0], v[1]):.2f} vs {dot(v[0], v[2]):.2f}"
    check("embeddings (OpenAI)", embeddings)


    if not QUICK:
        def moe():
            m = first(models, "moe", "Instruct")
            text, last, _ = stream(m, "Capital of Japan? One word.", 12)
            assert "tokyo" in text.lower(), text
            return f"{last['npu']['decode_tok_s']} tok/s"
        check("own MoE engine (Qwen3-30B-A3B)", moe)

        def image():
            st, d = call("POST", "/v1/images/generations", {"model": "sd-turbo", "prompt": "a red fox in snow", "size": "512x512"})
            assert st == 200, d
            png = base64.b64decode(d["data"][0]["b64_json"])
            assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 50_000
            return f"{len(png) // 1024} KB, {d['timings'][0].get('seconds', 0):.1f}s"
        check("image generation (sd-turbo)", image)

        def speech():
            st, wav = call("POST", "/v1/audio/speech", {"model": "mms-tts-eng", "input": "Hello from the NPU.", "response_format": "wav"}, raw=True)
            assert st == 200 and wav[:4] == b"RIFF" and len(wav) > 20_000, st
            return f"{len(wav) // 1024} KB"
        check("speech (mms-tts-eng)", speech)

    call("POST", "/unload", {})
    ok = sum(r[0] for r in RESULTS)
    print(f"\n{ok}/{len(RESULTS)} passed", flush=True)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
