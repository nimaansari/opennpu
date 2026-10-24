# OpenNPU QA: test plan and results (v0.3, 2026-10-12)

Scope: the model server (OpenAI + Ollama APIs, generation endpoints), the web UI, the GTK desktop
app, and the installer. Machine: Ryzen AI 7 PRO 350 (XDNA2), Ubuntu 26.04, kernel 6.17, XRT 2.25.

## How it is tested

| Suite | What it drives | Engines | Runs in CI | Count |
|---|---|---|---|---|
| `tests/unit` | config, model resolution, origin checks, parameter parsing, markdown, generation validation, icon/asset integrity | none | yes | 51 |
| `tests/qa/test_api.py` | the real HTTP server over sockets: every route, error shapes, streaming, disconnects, concurrency, security, local analytics (no content stored), Hugging Face search / details / pull / cancel / remove | fake (scripted) | yes | 115 |
| `tests/e2e` (Playwright) | the web UI in Chromium against the real server: chat, stop, errors, conversations, markdown/XSS, themes, Create with progress and cancel, Dashboard (usage, hardware, CSV, clear, recording toggle), Discover (search, details, GGUF choice, download progress, Chat now, cancel, remove, offline / gated errors), mobile, tablet widths, axe accessibility (WCAG 2.1 AA) in both themes, keyboard, copy rules | fake | yes | 44 |
| `tests/qa/test_desktop_app.py` | the real GTK app under `xvfb-run` against the real server: chat, Esc/stop, switching mid-stream, errors, delete/undo, settings and theme persistence, corrupt files, markdown links, all pages, Create progress/cancel, Dashboard, Discover (search, download dialog with GGUF choice, progress, cancel, Chat now, remove), light and dark snapshots | fake | yes (libadwaita 1.5) | 21 |
| `tests/qa/live_smoke.py` | the running server with real models on the NPU | all real | no (needs the NPU) | 9 |
| `tests/sweep.py` | 46 model architectures through the universal path, output vs CPU | real | no | 46 |

The fake engines (`tests/qa/fakeserver.py`) replace only the model engines: routing, validation,
streaming, locking, telemetry, CORS, static files and generation request handling are the
production code. Replies are scripted by keyword (`__slow__`, `__error__`, `__xss__`,
`__think__`, `__md__`, ...), so every scenario is deterministic and needs no hardware. The
Hugging Face Hub is a canned catalog (fake `HfApi`) and `hf download` is a fake process that
writes files into a temp dir, so Discover's whole path (opennpu.hub, the server routes, progress,
a real SIGTERM on cancel) runs offline.

```bash
~/.local/share/opennpu/env/bin/python -m pytest tests/unit tests/qa/test_api.py
cd tests/e2e && npm ci && npx playwright test
xvfb-run -a /usr/bin/python3 -m pytest tests/qa/test_desktop_app.py
python3 tests/qa/live_smoke.py            # real models, one at a time
```

## Results (all green on 2026-10-12)

- Unit + API: 161 passed, 3 skipped (need PyGObject in the engine env), 2 xfailed (strict: a bad
  `files` list on `/api/hub/pull` is answered 502 instead of 400). Web e2e: 44 passed. Desktop: 21 passed.
- Live, real models on the NPU (9/9): native Qwen3-8B 10.2 tok/s (correct), Ollama route, typo
  and path model refs rejected with the loaded model kept, universal Qwen3-0.6B 15.7 tok/s,
  embeddings (similar 0.93 vs unrelated 0.41), gpt-oss-20b 15.8 tok/s, own MoE Qwen3-30B-A3B
  8.1 tok/s, SD-Turbo image in 5.4 s, MMS speech.
- Sweep: 46/46 architectures match CPU (cosine >= 0.994).

## Bugs found and fixed in this pass

Found by three independent code reviews (server 23, web UI 22, desktop 20) and by the test
suites while they were being written. Highlights:

**Security**
- Model names from HTTP could be file paths: `universal:/tmp/x.pt` reached
  `torch.load(weights_only=False)` (code execution). Now only catalog entries or `org/name`
  repo ids; no paths, `..`, or pickle-style files.
- A typo in a model name unloaded the resident model and started an unbounded download inside
  the model lock. Now a Hub lookup and disk check happen first (404 / 403 / 507), downloads run
  outside the lock, and the request body is validated before anything is fetched.
- Bodies of any size were read into memory, negative or huge `Content-Length` hung a thread,
  chunked bodies were mis-parsed as the next request. Now 413 / 400 / 411, plus a socket timeout.
- `nosniff` on all responses; CORS stays local-origin only.

**Server robustness**
- Non-object JSON, bad `messages` / `max_tokens` / `temperature` / embedding input crashed the
  connection or returned 500: now 400 with OpenAI's error shape (Ollama routes keep Ollama's).
- The universal engine deadlocked with the model lock held if generation raised, and kept
  generating after a client disconnect (two generations at once on the NPU). Fixed.
- FastFlowLM kept generating after the client left; failed starts left an orphan process.
- Streaming requests for a missing model answered 200 and failed inside the stream: now the
  status comes first. Context overflow is a 400 that says to start a new conversation.
- `finish_reason: "length"`, `delta.role` on the first chunk, `/v1/models/<id>`, Ollama
  preload and `keep_alive: 0`, a stable `cpu_pct`, a catalog one broken file can't take down.

**Web UI**
- Create was broken after a reload (scripts ran in the wrong order).
- On phones only the Chat page had a menu button: other pages were dead ends.
- Enter stopped a running reply; switching or deleting a conversation mid-stream froze the
  reply or kept the NPU busy; a failed turn broke every later turn on strict chat templates.
- Messages sent in the first second after load were dropped (model list not loaded yet).
- Markdown: italics inside code and math, raw `##`, broken numbered lists, reasoning shown as the
  answer for R1-style models. XSS: none found, verified by test.
- Accessibility: contrast below AA in four places, the conversation re-read on every token by
  screen readers, invisible delete buttons, dropdowns unreadable in a forced theme.
- Create: Generate did nothing before the model list loaded; results restarted playing media,
  leaked memory; no progress or cancel.

**Desktop app**
- Crashed at start on libadwaita < 1.6 (Ubuntu 24.04, Fedora 40) and the installer didn't notice.
- Any reply with a link showed raw markup; the chosen theme reverted on navigation.
- A corrupt conversations file silently wiped all history on the next save.
- Stop didn't work while a model loaded; Enter stopped replies; switching conversations hid the
  live reply; errors and closing the window lost partial replies.
- Service buttons failed silently; the launcher's "New conversation" didn't open one.

## Known limits (open)

- Generation shares the single model slot with chat: a long video blocks chat for minutes
  (by design for RAM, but there is no queue or priority yet).
- Speech generation can only be cancelled before it starts (the TTS models have no step hook).
- Native engines have a 2048-token context; long chats need a new conversation (the error says so).
- The desktop app's libadwaita 1.4/1.5 fallbacks are exercised by CI on Ubuntu 24.04, not on this
  machine (it has 1.9).
- Power and joules per token are whole-laptop figures from the battery sensor; the driver reports
  no NPU-only power.
