"""The GTK desktop app, driven for real: a headless X server (xvfb-run), the real app code, the real
HTTP server with fake engines (fakeserver.py). Each scenario runs the app in its own process.

    xvfb-run -a python3 -m pytest tests/qa/test_desktop_app.py     (system Python with PyGObject)
"""
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
pytestmark = pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                                reason="needs a display: run under xvfb-run")
ENGINE_PY = os.environ.get("OPENNPU_PY", str(Path.home() / ".local/share/opennpu/env/bin/python"))


@pytest.fixture(scope="module")
def server():
    p = subprocess.Popen([ENGINE_PY, str(HERE / "fakeserver.py"), "0"], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("READY"), line
    yield f"http://127.0.0.1:{line.split()[1]}"
    p.kill()


# The driver: builds the real Window, then runs `steps` (generator) on the GTK loop; each
# `yield <seconds>` lets the app run that long. Results go to stdout as one JSON line.
DRIVER = textwrap.dedent('''
    import json, os, sys, time, traceback
    sys.path.insert(0, os.environ["ROOT"])
    from gi.repository import GLib, Gtk
    import opennpu.desktop.app as A
    out = {}
    def body(text):
        b = W.chat.input.get_buffer(); b.set_text(text)
    def texts():
        res, c = [], W.chat.thread.get_first_child()
        while c is not None:
            if isinstance(c, A.MessageView):
                res.append((c.role, c.text))
            c = c.get_next_sibling()
        return res
    def labels(w):
        res, stack = [], [w]
        while stack:
            x = stack.pop()
            if isinstance(x, Gtk.Label):
                res.append(x.get_text())
            c = x.get_first_child()
            while c is not None:
                stack.append(c); c = c.get_next_sibling()
        return res
    SCENARIO
    def run():
        gen = steps()
        def tick():
            try:
                d = next(gen)
            except StopIteration:
                print("RESULT " + json.dumps(out), flush=True); app.quit(); return False
            except Exception:
                out["error"] = traceback.format_exc(); print("RESULT " + json.dumps(out), flush=True); app.quit(); return False
            GLib.timeout_add(int(d * 1000), tick); return False
        GLib.timeout_add(1500, tick)
    app = A.App()
    W = None
    def act(*_):
        global W
        W = app.get_active_window()
        run()
    app.connect_after("activate", act)
    app.run([])
''')


def drive(server, home, scenario, env=None):
    code = DRIVER.replace("SCENARIO", textwrap.dedent(scenario))
    e = dict(os.environ, ROOT=str(ROOT), OPENNPU_URL=server, OPENNPU_HOME=str(home), GSK_RENDERER="cairo",
             DBUS_SESSION_BUS_ADDRESS="disabled:", **(env or {}))     # never attach to a running instance
    p = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=90, env=e)
    line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line, f"app produced no result\nstdout:{p.stdout[-2000:]}\nstderr:{p.stderr[-3000:]}"
    res = json.loads(line[7:])
    assert "error" not in res, res["error"]
    crit = [x for x in p.stderr.splitlines() if "CRITICAL" in x or "Traceback" in x]
    assert not crit, "\n".join(p.stderr.splitlines()[-40:])
    return res


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def test_chat_streams_stats_and_persists(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("hello desktop"); W.chat.send()
            yield 1.5
            out["texts"] = texts(); out["live"] = W.chat.live is not None
            import opennpu.desktop.api as api               # turns are saved by the server (sessions)
            out["saved"] = api.get(f"/v1/sessions/{W.chat.cur['id']}")["messages"]
    """)
    assert r["texts"] == [["user", "hello desktop"], ["assistant", "Echo: hello desktop "]]
    assert not r["live"]
    assert r["saved"][1]["content"] == "Echo: hello desktop " and r["saved"][1]["stats"]["engine"]


def test_enter_does_not_stop_escape_does_and_partial_is_kept(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("__slow__"); W.chat.send()
            yield 0.8
            body("next question"); W.chat.send()            # Enter while streaming: must not stop it
            yield 0.3
            out["still_live"] = W.chat.live is not None
            W.chat.stop()                                     # Stop / Esc
            yield 0.3
            m = W.chat.cur["messages"][-1]
            out["stopped"] = m.get("stopped"); out["partial"] = m["content"]
            out["draft"] = W.chat.input.get_buffer().get_property("text")
    """)
    assert r["still_live"] and r["stopped"] and r["partial"].startswith("s0 s1")
    assert r["draft"] == "next question"                     # the draft was not sent or lost


def test_switching_conversations_mid_stream(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("__slow__"); W.chat.send()
            first = W.chat.cur
            yield 0.6
            W.chat.new_chat()                                 # always allowed
            out["send_sensitive_elsewhere"] = W.chat.send_btn.get_sensitive()
            yield 0.3
            W.chat.show_convo(first["id"])                    # back: the live reply is re-attached
            yield 0.8
            out["live_text_shown"] = texts()[-1][1]
            W.chat.stop(); yield 0.3
            out["n_msgs"] = len(first["messages"])
    """)
    assert r["send_sensitive_elsewhere"] is False
    assert r["live_text_shown"].startswith("s0 s1 s2")
    assert r["n_msgs"] == 2


def test_engine_error_shows_message_and_history_skips_failed_turn(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("__error0__"); W.chat.send()
            yield 1.0
            out["err"] = W.chat.cur["messages"][-1].get("error")
            out["hist"] = W.chat.history(W.chat.cur)
            body("second"); W.chat.send()
            yield 1.0
            out["last"] = W.chat.cur["messages"][-1]["content"]
    """)
    assert "Could not get a reply" in r["err"]
    assert r["hist"] == []
    assert r["last"] == "Echo: second "


def test_delete_with_undo_and_theme_persists(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("keep me"); W.chat.send()
            yield 1.0
            cid = W.chat.cur["id"]
            W._delete(cid)
            out["after_delete"] = cid in [c["id"] for c in W.chat.convos.items]
            W.chat.convos.items.insert(0, {"id": cid, "title": "keep me", "messages": [], "model": None})  # what Undo does
            app.activate_action("theme", GLib.Variant.new_string("dark"))
            W.show("models"); W.show("system")                  # navigation saves settings too
            yield 0.3
            out["settings"] = json.load(open(A.DATA / "settings.json"))
    """)
    assert r["after_delete"] is False
    assert r["settings"]["theme"] == "dark" and r["settings"]["page"] == "system"


def test_corrupt_files_do_not_crash_or_wipe(server, home):
    d = home / "desktop"
    d.mkdir(parents=True)
    (d / "conversations.json").write_text("{truncated")
    (d / "settings.json").write_text('{"page": "nonexistent", "theme": "purple"}')
    r = drive(server, home, """
        def steps():
            yield 0.5
            out["convos_ok"] = isinstance(W.chat.convos.items, list)
            out["page"] = W.stack.get_visible_child_name()
            import glob
            out["backup"] = bool(glob.glob(str(A.DATA / "conversations.json.corrupt-*")))
    """)
    assert r == {"convos_ok": True, "page": "chat", "backup": True}


def test_markdown_links_render_as_markup_not_raw(server, home):
    r = drive(server, home, """
        def steps():
            v = A.MessageView("assistant", "See [docs](https://a.b/?q=1&r=2) for **more**")
            out["labels"] = labels(v)
            yield 0.1
    """)
    assert any(t == "See docs for more" for t in r["labels"]), r["labels"]


def test_models_create_system_pages_render(server, home):
    r = drive(server, home, """
        def steps():
            yield 1.5
            out["chat_models"] = [m["id"] for m in W.chat.models]
            W.show("create"); yield 0.3
            out["image_models"] = len(W.create.forms["image"].get("items") or [])
            W.show("system"); yield 2.5
            out["device"] = W.system.dev_rows["device"].get_text()
    """)
    assert "native:Fake-Qwen3-8B" in r["chat_models"]
    assert r["image_models"] > 0
    assert r["device"] == "RyzenAI-npu6"


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_headless_snapshot_both_themes(server, home, tmp_path, scheme):
    (home / "desktop").mkdir(parents=True)
    (home / "desktop" / "settings.json").write_text(json.dumps({"theme": scheme}))
    png = tmp_path / f"snap-{scheme}.png"
    e = dict(os.environ, ROOT=str(ROOT), OPENNPU_URL=server, OPENNPU_HOME=str(home), GSK_RENDERER="cairo", DBUS_SESSION_BUS_ADDRESS="disabled:",
             OPENNPU_DESKTOP_SNAPSHOT=str(png), OPENNPU_DESKTOP_SNAPSHOT_DELAY="3")
    p = subprocess.run(["/usr/bin/python3", "-m", "opennpu.desktop"], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert png.exists() and png.stat().st_size > 10_000, p.stderr[-2000:]
    if os.environ.get("SNAP_DIR"):
        shutil.copy(png, Path(os.environ["SNAP_DIR"]) / png.name)


def test_create_progress_cancel_then_generate(server, home):
    r = drive(server, home, """
        def steps():
            yield 1.5
            W.show("create")
            f = W.create.forms["image"]
            f["prompt"].get_buffer().set_text("__slow__ fox")
            W.create._generate(f)
            yield 2.5
            out["label_busy"] = f["go"].get_label()
            out["status"] = f["status_label"].get_text()
            out["fraction"] = f["progress"].get_fraction()
            W.create._cancel(f)
            yield 1.5
            out["after_cancel"] = f["status_label"].get_text()
            out["label_after"] = f["go"].get_label()
            f["prompt"].get_buffer().set_text("quick fox")
            W.create._generate(f)
            yield 1.5
            out["done"] = f["status_label"].get_text()
            out["files"] = len(list((A.DATA / "creations").glob("*.png")))
            f["prompt"].get_buffer().set_text("x" * 2001)
            W.create._generate(f)
            out["too_long_busy"] = bool(f.get("busy"))
    """)
    assert r["label_busy"] == "Cancel" and r["status"].startswith("Step ") and 0 < r["fraction"] <= 1
    assert r["after_cancel"] == "Cancelled" and r["label_after"] == "Generate image"
    assert r["done"].startswith("Done in") and r["files"] == 1
    assert r["too_long_busy"] is False


def test_dashboard_usage_hardware_record_toggle_and_clear(server, home):
    r = drive(server, home, """
        def steps():
            W.chat.select_model("native:Fake-Qwen3-8B")
            body("count this one"); W.chat.send()
            yield 1.2
            W.show("dashboard"); yield 1.5
            out["usage_labels"] = labels(W.dashboard.body)
            W.dashboard._set_tab("hardware"); yield 1.5
            out["hw_labels"] = labels(W.dashboard.body)
            out["export_hidden"] = not W.dashboard.export_btn.get_visible()
            W.dashboard._set_tab("usage"); yield 1.0
            W.dashboard.record.set_active(False); yield 1.0
            import opennpu.desktop.api as api
            out["enabled_after_off"] = api.usage("24h")["enabled"]
            W.dashboard.record.set_active(True); yield 1.0
            api.usage_clear("usage"); W.dashboard.refresh(); yield 1.0
            out["after_clear"] = labels(W.dashboard.body)
    """)
    assert "Requests" in r["usage_labels"] and "native:Fake-Qwen3-8B" in r["usage_labels"]
    assert "Desktop app" in r["usage_labels"]                       # the app that asked
    assert "count this one" not in " ".join(r["usage_labels"])
    assert "This machine" in r["hw_labels"] and "NPU busy" in r["hw_labels"] and r["export_hidden"]
    assert r["enabled_after_off"] is False
    assert "No usage in this range yet" in r["after_clear"]


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("tab", ["usage", "hardware"])
def test_dashboard_snapshots(server, home, tmp_path, scheme, tab):
    import urllib.request
    for i in range(4):                                              # some traffic to show
        urllib.request.urlopen(urllib.request.Request(server + "/v1/chat/completions", json.dumps(
            {"model": "native:Fake-Qwen3-8B", "messages": [{"role": "user", "content": f"q {i} a b c"}]}).encode(),
            {"Content-Type": "application/json", "X-OpenNPU-Client": "desktop"})).read()
    (home / "desktop").mkdir(parents=True, exist_ok=True)
    (home / "desktop" / "settings.json").write_text(json.dumps({"theme": scheme, "page": "dashboard", "dash_tab": tab, "dash_ur": "24h"}))
    png = tmp_path / f"dash-{tab}-{scheme}.png"
    e = dict(os.environ, ROOT=str(ROOT), OPENNPU_URL=server, OPENNPU_HOME=str(home), GSK_RENDERER="cairo",
             DBUS_SESSION_BUS_ADDRESS="disabled:", OPENNPU_DESKTOP_SNAPSHOT=str(png), OPENNPU_DESKTOP_SNAPSHOT_DELAY="4")
    p = subprocess.run(["/usr/bin/python3", "-m", "opennpu.desktop"], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert png.exists() and png.stat().st_size > 10_000, p.stderr[-2000:]
    assert "Traceback" not in p.stderr, p.stderr[-2000:]
    if os.environ.get("SNAP_DIR"):
        shutil.copy(png, Path(os.environ["SNAP_DIR"]) / png.name)


# ---------------------------------------------------------------- Discover (fake Hugging Face Hub in fakeserver.py)
HUB_RESET = """
    import opennpu.desktop.api as api
    def reset(rid):                                   # not downloaded, no job left over from another test
        for f in (lambda: api.hub_remove(rid), lambda: api.hub_cancel(rid), lambda: api.hub_cancel(rid)):
            try:
                f()
            except api.ApiError:
                pass
    def row_titles():
        res, r, i = [], None, 0
        while (r := W.discover.list.get_row_at_index(i)) is not None:
            res.append(r.card_id); i += 1
        return res
"""


def hub(scenario):
    return textwrap.dedent(HUB_RESET) + textwrap.dedent(scenario)


def test_discover_page_search_task_and_errors(server, home):
    r = drive(server, home, hub("""
        def steps():
            W.show("discover"); yield 2.0
            out["page"] = W.stack.get_visible_child_name()
            out["rows"] = row_titles()
            q = W.discover.rows["Qwen/Qwen3-0.6B"]
            out["q_sub"] = q.get_subtitle(); out["q_labels"] = labels(q)
            out["gated_labels"] = labels(W.discover.rows["meta-llama/Llama-3.2-1B-Instruct"])
            out["big_sub"] = W.discover.rows["deepseek-ai/DeepSeek-V3"].get_subtitle()
            W.discover.task.set_selected(6); yield 1.5               # All
            out["all_rows"] = row_titles()
            out["saved_task"] = W.settings.get("hub_task")
            W.discover.entry.set_text("sd-turbo"); W.discover.search(); yield 1.5
            out["turbo"] = row_titles(); out["turbo_labels"] = labels(W.discover.rows["stabilityai/sd-turbo"])
            W.discover.entry.set_text("offline"); W.discover.search(); yield 1.5
            out["err"] = W.discover.error_label.get_text(); out["error_shown"] = W.discover.error_label.get_visible()
            W.discover.task.set_selected(0)
    """))
    assert r["page"] == "discover"
    assert "Qwen/Qwen3-0.6B" in r["rows"] and "Qwen/Qwen3-VL-2B-Instruct" not in r["rows"]
    assert "752M" in r["q_sub"] and "9.1M downloads" in r["q_sub"] and "Fused NPU engine (fastest)" in r["q_sub"]
    assert "Fastest" in r["q_labels"] and "Download" in r["q_labels"]
    assert "Gated" in r["gated_labels"]
    assert "Warning: needs ~" in r["big_sub"] and "GB RAM" in r["big_sub"]
    assert "stabilityai/sd-turbo" in r["all_rows"] and r["saved_task"] == "all"
    assert r["turbo"] == ["stabilityai/sd-turbo"] and "Create (image / video on the NPU)" not in r["turbo_labels"]
    assert r["error_shown"] and "offline" in r["err"] and "installed models keep working offline" in r["err"]


def test_discover_download_progress_chat_now_and_remove(server, home):
    r = drive(server, home, hub("""
        def steps():
            rid = "Qwen/Qwen3-0.6B"
            reset(rid)
            W.show("discover"); yield 2.0
            W.discover.open_details(rid); yield 1.5
            d = W.discover.dialog
            out["size"] = d["download"].get_label(); out["dlg_labels"] = labels(d["page"])
            W.discover.start_pull(d); yield 1.2
            out["dialog_closed"] = W.discover.dialog is None
            pr = W.discover.pull_rows[rid]
            out["fraction"] = pr["bar"].get_fraction(); out["progress_sub"] = pr["row"].get_subtitle()
            out["row_while"] = labels(W.discover.rows[rid])
            yield 3.5
            pr = W.discover.pull_rows[rid]
            out["done_labels"] = labels(pr["row"]); out["installed_labels"] = labels(W.discover.rows[rid])
            pr["chat"].emit("clicked"); yield 7.0
            out["page_after"] = W.stack.get_visible_child_name(); out["model"] = W.chat.selected_model()
            W.show("discover"); yield 0.5
            W.discover._set_confirm(rid, rid); yield 0.3
            out["confirm_labels"] = labels(W.discover.rows[rid])
            W.discover.remove(rid); yield 1.5
            out["after_remove"] = labels(W.discover.rows[rid])
            out["installed_api"] = api.hub_model(rid)["installed"]
    """))
    assert r["size"] == "Download 27 MB"
    assert "apache-2.0" in r["dlg_labels"] and "Free disk" in r["dlg_labels"]
    assert r["dialog_closed"] and 0 < r["fraction"] < 1 and " of 27 MB" in r["progress_sub"]
    assert any(t.startswith("Downloading ") for t in r["row_while"])
    assert "Downloaded" in r["done_labels"] and "Chat now" in r["done_labels"] and "Installed" in r["installed_labels"]
    assert r["page_after"] == "chat" and r["model"] == "native:Qwen3-0.6B"
    assert "Remove its files?" in r["confirm_labels"] and "Keep" in r["confirm_labels"]
    assert "Installed" not in r["after_remove"] and "Download" in r["after_remove"] and r["installed_api"] is False


def test_discover_gguf_choice_cancel_and_gated(server, home):
    r = drive(server, home, hub("""
        def steps():
            for x in ("unsloth/gpt-oss-20b-GGUF", "opennpu-test/Slow-Qwen3-1.7B"):
                reset(x)
            W.show("discover"); yield 2.0
            W.discover.open_details("unsloth/gpt-oss-20b-GGUF"); yield 1.5
            d = W.discover.dialog
            out["default"] = W.discover._chosen_gguf(d); out["size0"] = d["download"].get_label()
            d["gguf"].set_selected(1); yield 0.2
            out["chosen"] = W.discover._chosen_gguf(d); out["size1"] = d["download"].get_label()
            d["win"].close(); yield 0.3
            W.discover.start_pull(rid="opennpu-test/Slow-Qwen3-1.7B"); yield 1.5
            out["slow_state"] = W.discover.pulls["opennpu-test/Slow-Qwen3-1.7B"]["state"]
            W.discover.cancel("opennpu-test/Slow-Qwen3-1.7B"); yield 1.5
            out["after_cancel"] = W.discover.pulls["opennpu-test/Slow-Qwen3-1.7B"]["state"]
            out["pull_row_gone"] = "opennpu-test/Slow-Qwen3-1.7B" not in W.discover.pull_rows
            W.discover.open_details("meta-llama/Llama-3.2-1B-Instruct"); yield 1.5
            d = W.discover.dialog
            out["gated_note"] = labels(d["page"])
            W.discover.start_pull(d); yield 1.5
            out["gated_error"] = d["error"].get_text(); out["still_open"] = W.discover.dialog is d
            d["win"].close(); yield 0.3
            reset("opennpu-test/Slow-Qwen3-1.7B")
    """))
    assert r["default"] == "gpt-oss-20b-Q4_K_M.gguf" and r["size0"] == "Download 13 MB"
    assert r["chosen"] == "gpt-oss-20b-Q8_0.gguf" and r["size1"] == "Download 17 MB"
    assert r["slow_state"] == "downloading" and r["after_cancel"] == "cancelled" and r["pull_row_gone"]
    assert any("Gated: accept the license" in t for t in r["gated_note"])
    assert "hf auth login" in r["gated_error"] and r["still_open"]


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_discover_snapshots(server, home, tmp_path, scheme):
    (home / "desktop").mkdir(parents=True)
    (home / "desktop" / "settings.json").write_text(json.dumps({"theme": scheme, "page": "discover"}))
    png = tmp_path / f"discover-{scheme}.png"
    e = dict(os.environ, ROOT=str(ROOT), OPENNPU_URL=server, OPENNPU_HOME=str(home), GSK_RENDERER="cairo",
             DBUS_SESSION_BUS_ADDRESS="disabled:", OPENNPU_DESKTOP_SNAPSHOT=str(png), OPENNPU_DESKTOP_SNAPSHOT_DELAY="4")
    p = subprocess.run(["/usr/bin/python3", "-m", "opennpu.desktop"], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert png.exists() and png.stat().st_size > 10_000, p.stderr[-2000:]
    assert "Traceback" not in p.stderr and "CRITICAL" not in p.stderr, p.stderr[-2000:]
    if os.environ.get("SNAP_DIR"):
        shutil.copy(png, Path(os.environ["SNAP_DIR"]) / png.name)
