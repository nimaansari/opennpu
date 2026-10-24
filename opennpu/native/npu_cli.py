#!/usr/bin/env python3
"""npu -- run models on the NPU (talks to the OpenNPU server on localhost:11540).

  npu list                       installed models, by engine (all on OpenNPU's own NPU engines)
  npu chat MODEL [-p PROMPT]     chat (REPL without -p); --think keeps Qwen3 reasoning
  npu pull HF_REPO               download a dense HF checkpoint into ~/models (checks NPU support first)
  npu status | unload            what is loaded / free the NPU and its RAM now
  npu start | stop | restart | logs    control the background service
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if __package__ in (None, ""):          # run as a file (e.g. symlinked onto PATH): use this source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opennpu import config  # noqa: E402

URL = config.URL
AIRENV = str(config.AIRENV)
OPENNPU = str(Path(__file__).resolve().parents[2])
MODELS = str(config.MODELS)


def api(path, body=None, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json", "X-OpenNPU-Client": "cli"})
    return urllib.request.urlopen(req, timeout=timeout)


def up():
    try:
        api("/status", timeout=3)
        return True
    except Exception:
        return False


def ensure_up():
    if up():
        return
    print("[npu] starting the service ...", file=sys.stderr)
    subprocess.run(["systemctl", "--user", "start", config.SERVICE], check=False)
    for _ in range(60):
        if up():
            return
        time.sleep(1)
    sys.exit("[npu] server not reachable; see `npu logs`")


def airpy(code):
    return subprocess.run(["bash", "-c", f"set +u; source {AIRENV} >/dev/null; cd {OPENNPU}; python3 -c {json.dumps(code)}"],
                          capture_output=True, text=True)


def cmd_list():
    ensure_up()
    data = json.load(api("/v1/models"))["data"]
    st = json.load(api("/status"))
    for m in data:
        print(("* " if m["id"] == st.get("loaded") else "  ") + m["id"])
    if not data:
        print("no models yet: npu pull <hf repo>")


def cmd_chat(args):
    think = "--think" in args
    args = [a for a in args if a != "--think"]
    if not args:
        sys.exit("usage: npu chat MODEL [-p PROMPT]")
    model, prompt = args[0], None
    if "-p" in args:
        prompt = " ".join(args[args.index("-p") + 1:])
    ensure_up()
    hist = []

    def turn(q):
        hist.append({"role": "user", "content": q})
        body = {"model": model, "messages": hist, "stream": True, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": think}}
        t0, out, st = time.time(), [], {}
        for line in api("/v1/chat/completions", body):
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if "error" in d:
                e = d["error"]
                print(f"\n[npu] error: {e.get('message') if isinstance(e, dict) else e}", file=sys.stderr)
                return
            c = d["choices"][0]["delta"].get("content")
            if c:
                if not out:
                    print(f"[first token after {time.time() - t0:.1f}s]", file=sys.stderr)
                out.append(c)
                print(c, end="", flush=True)
            st = d.get("npu", st)
        print(flush=True)
        if st:
            print(f"[{st.get('completion_tokens')} tok, {st.get('decode_tok_s')} tok/s on the NPU]", file=sys.stderr)
        hist.append({"role": "assistant", "content": "".join(out)})

    if prompt:
        return turn(prompt)
    print(f"[npu] chatting with {model}; /exit to quit, /reset to clear history", file=sys.stderr)
    while True:
        try:
            q = input("\n> ")
        except EOFError:
            break
        if q.strip() in ("/exit", "/quit"):
            break
        if q.strip() == "/reset":
            hist.clear()
            continue
        turn(q)


def cmd_pull(args):
    if not args:
        sys.exit("usage: npu pull HF_REPO")
    repo = args[0]
    if repo.startswith("flm:"):
        sys.exit("[npu] FastFlowLM support was removed: OpenNPU runs every model on its own engines. "
                 "Pull a Hugging Face model instead, e.g. npu pull Qwen/Qwen3-8B")
    # check NPU support from config.json before downloading tens of GB
    chk = airpy(f"""
import json
from huggingface_hub import hf_hub_download
from opennpu.native import specgen
from opennpu.native.chat import resolve
import tempfile, shutil, os
d = tempfile.mkdtemp(); shutil.copy(hf_hub_download({repo!r}, 'config.json'), d)
open(os.path.join(d, 'x.safetensors'), 'w').close()
print('OK', resolve(d)[0])""")
    if "OK" not in chk.stdout:
        sys.exit(f"[npu] {repo} can't run on the native NPU engine yet:\n{(chk.stderr or chk.stdout).strip().splitlines()[-1]}")
    name = repo.split("/")[-1]
    print(f"[npu] {repo} -> {chk.stdout.split()[-1]} geometry; downloading to {MODELS}/{name} ...")
    r = subprocess.call(["bash", "-c", f"set +u; source {AIRENV} >/dev/null; hf download {repo} "
                         f"--include '*.json' --include '*.safetensors' --include '*.txt' --include 'tokenizer*' "
                         f"--include '*.jinja' --local-dir {MODELS}/{name}"])
    if r == 0:
        print(f"[npu] done. First load quantizes it for the NPU (a few minutes, once):  npu chat {name}")
    sys.exit(r)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        return print(__doc__)
    c, rest = a[0], a[1:]
    if c == "list":
        cmd_list()
    elif c == "chat":
        cmd_chat(rest)
    elif c == "pull":
        cmd_pull(rest)
    elif c == "status":
        ensure_up()
        print(json.load(api("/status")))
    elif c == "unload":
        ensure_up()
        api("/unload", {})
        print("[npu] unloaded")
    elif c in ("start", "stop", "restart"):
        subprocess.call(["systemctl", "--user", c, config.SERVICE])
    elif c == "logs":
        subprocess.call(["journalctl", "--user", "-u", config.SERVICE, "-n", "50", "--no-pager"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
