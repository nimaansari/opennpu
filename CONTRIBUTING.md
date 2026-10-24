# Contributing

Thanks for helping make the NPU useful on Linux.

## Most useful right now

- **Hardware reports.** Run `./install.sh --check`, `opennpu doctor --json` and
  `python3 tests/qa/live_smoke.py --quick` on your Ryzen AI laptop and open an issue with the
  output and your chip, distro, kernel and XRT version. Only Krackan (Ryzen AI 7 PRO 350) is
  tested so far.
- **Models.** Try a model with `opennpu run <hf repo>` and report what works, how fast, and what
  breaks.
- See the roadmap items in the README's Limits section and `docs/design/`.

## Development setup

```bash
OPENNPU_HOME=/tmp/opennpu-dev ./install.sh --native --no-service   # an isolated install
/tmp/opennpu-dev/env/bin/python -m pytest tests/unit tests/qa/test_api.py
```

- Tests that need no NPU (unit, API, web, desktop) use fake engines (`tests/qa/fakeserver.py`)
  and a throwaway data directory; they run in CI.
- Tests on real hardware (`tests/qa/live_smoke.py`, `tests/sweep.py`) are run by hand.
  Point them at a scratch `OPENNPU_HOME` so they don't add to your own conversations.
- Run NPU experiments with `opennpu/native/safe_run.sh` (no core dumps, hard timeout).

## Pull requests

- Keep changes focused; include tests for behaviour changes.
- `python -m pytest tests/unit tests/qa/test_api.py` and `ruff check` must pass.
- No telemetry, no network calls except explicit model downloads, local-only by default.
