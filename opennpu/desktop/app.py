"""OpenNPU desktop app (GTK 4 + libadwaita). A native front end for the local OpenNPU server:
chat with any installed model, generate images / video / speech, manage models and the server.

Runs on the system Python (needs only PyGObject); all inference stays in the server process.
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from opennpu import __version__, config  # noqa: E402
from opennpu.desktop import api, markdown, media  # noqa: E402
from opennpu.desktop.dashboard import DashboardPage  # noqa: E402
from opennpu.desktop.discover import DiscoverPage  # noqa: E402

APP_ID = "io.github.opennpu.OpenNPU"
HERE = Path(__file__).resolve().parent
DATA = config.HOME / "desktop"
STARTERS = ["Explain how a CPU cache works, briefly.", "Write a Python function that dedupes a list, keeping order.",
            "What are the trade-offs of running LLMs on an NPU?", "Draft a polite email declining a meeting."]
ENGINE_ORDER = ["moe", "native", "npu4", "llama", "universal"]


def run_bg(fn, ok=None, err=None):
    """Run fn() on a worker thread; deliver its result (or exception) on the GTK main loop."""
    def work():
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            if err:
                GLib.idle_add(err, e)
            return
        if ok:
            GLib.idle_add(ok, r)
    threading.Thread(target=work, daemon=True).start()


def spinner(**kw):
    """Adw.Spinner needs libadwaita 1.6; Ubuntu 24.04 / Fedora 40 ship 1.5."""
    if hasattr(Adw, "Spinner"):
        return Adw.Spinner(**kw)
    return Gtk.Spinner(spinning=True, **kw)


def write_atomic(path, text):
    """Write via a temp file + fsync + rename: a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path, kind, default):
    """Read JSON of the expected type; a corrupt file is kept aside (never silently overwritten)."""
    try:
        v = json.loads(path.read_text())
        if isinstance(v, kind):
            return v
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        pass
    try:
        path.replace(path.with_name(f"{path.name}.corrupt-{int(time.time())}"))
    except OSError:
        pass
    return default


class Settings(dict):
    """One settings store for the whole app (window state + theme), saved atomically."""
    path = DATA / "settings.json"

    def __init__(self):
        super().__init__(read_json(self.path, dict, {}))

    def save(self):
        try:
            write_atomic(self.path, json.dumps(self))
        except OSError:
            pass


SETTINGS = None


def settings():
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = Settings()
    return SETTINGS


def gb(n):
    return f"{n / 1e9:.0f} GB" if n >= 1e10 else f"{n / 1e9:.1f} GB"


def markup_ok(m):
    # <a href> is GtkLabel-only markup: Pango's parser rejects it, so check the rest without it
    try:
        Pango.parse_markup(re.sub(r"</?a\b[^>]*>", "", m), -1, "\0")
        return True
    except GLib.Error:
        return False


def icon_button(icon, tooltip, cb=None, flat=True):
    b = Gtk.Button(icon_name=icon, tooltip_text=tooltip)
    b.update_property([Gtk.AccessibleProperty.LABEL], [tooltip])
    if flat:
        b.add_css_class("flat")
    if cb:
        b.connect("clicked", cb)
    return b


def copy_text(widget, text):
    widget.get_clipboard().set(text)


# ====================================================================== chat
class Conversations:
    """Conversations. On a server with /v1/sessions they live there (shared by every client on
    this machine, and the server keeps each session's KV cache warm); otherwise in
    $OPENNPU_HOME/desktop/conversations.json. Messages: {role, content, stats?, error?, stopped?,
    images?: [server URL], image_files?: [local path], _images?: [(mime, bytes)] in memory}."""

    def __init__(self):
        self.path = DATA / "conversations.json"
        self.remote = False
        self.items = []
        for c in read_json(self.path, list, []):
            if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("messages"), list):
                continue
            msgs = [m for m in c["messages"] if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content", ""), str)]
            self.items.append({**c, "title": str(c.get("title") or "Conversation"), "messages": msgs})

    # ---- server sessions
    def go_remote(self, settings, done):
        """Switch to server sessions if the server has them; import local history once."""
        local = [c for c in self.items if c.get("messages")]

        def work():
            if not api.sessions_supported():
                return None
            if local and not settings.get("sessions_imported"):
                api.sessions_import([{"title": c["title"], "model": c.get("model"), "created": c.get("t"),
                                      "messages": [{k: v for k, v in m.items() if not k.startswith("_") and k != "image_files"}
                                                   for m in c["messages"]]} for c in local])
            return api.sessions()

        def ok(remote):
            if remote is None:
                return done(False)
            if local and not settings.get("sessions_imported"):
                settings["sessions_imported"] = True
                settings.save()
                try:                                    # the server has them now; keep the file as a backup
                    self.path.replace(self.path.with_suffix(".imported.json"))
                except OSError:
                    pass
            self.remote = True
            self.items = [{"id": r["id"], "title": r.get("title") or "Conversation", "model": r.get("model"),
                           "t": r.get("updated"), "messages": None, "count": r.get("messages", 0)} for r in remote]
            done(True)
        run_bg(work, ok, lambda e: done(False))

    def load(self, convo, done):
        """Fetch a server session's messages (and its images) once; done(convo) on the main loop."""
        if not self.remote or convo.get("messages") is not None or convo.get("_new"):
            return done(convo)

        def work():
            d = api.session(convo["id"])
            msgs = []
            for m in d.get("messages") or []:
                if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                    continue
                m = dict(m)
                if isinstance(m.get("content"), list):          # stored as OpenAI parts
                    m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
                imgs = []
                for u in m.get("images") or []:
                    try:
                        imgs.append((media.mimetypes.guess_type(u)[0] or "image/png", api.media(u)))
                    except api.ApiError:
                        pass
                if imgs:
                    m["_images"] = imgs
                msgs.append(m)
            return d, msgs

        def ok(res):
            d, msgs = res
            convo["messages"] = msgs
            convo["model"] = d.get("model") or convo.get("model")
            done(convo)

        def fail(e):
            convo["messages"] = []
            convo["_load_error"] = str(e)
            done(convo)
        run_bg(work, ok, fail)

    # ---- both stores
    def save(self):
        if self.remote:
            return                                      # the server stores each turn itself
        try:
            write_atomic(self.path, json.dumps([{**c, "messages": [{k: v for k, v in m.items() if not k.startswith("_")}
                                                                   for m in c["messages"]]}
                                                for c in self.items[:200]]))
        except OSError as e:
            print(f"opennpu-desktop: could not save conversations: {e}", file=sys.stderr)

    def new(self, model):
        c = {"id": uuid.uuid4().hex, "title": "New conversation", "model": model, "messages": [], "t": time.time()}
        if self.remote:
            c["_new"] = True                            # becomes a server session on the first send
        self.items.insert(0, c)
        return c

    def get(self, cid):
        return next((c for c in self.items if c["id"] == cid), None)

    def delete(self, cid, now=True):
        convo = self.get(cid)
        self.items = [c for c in self.items if c["id"] != cid]
        if self.remote:
            if now and convo and not convo.get("_new"):
                run_bg(lambda: api.session_delete(cid), None,
                       lambda e: print(f"opennpu-desktop: delete failed: {e}", file=sys.stderr))
        else:
            self.save()


def save_local_images(images):
    """Local store: attachments go next to conversations.json; the JSON keeps their paths."""
    out = []
    d = DATA / "media"
    d.mkdir(parents=True, exist_ok=True)
    for mime, data in images:
        p = d / f"{uuid.uuid4().hex}{media.mimetypes.guess_extension(mime) or '.img'}"
        p.write_bytes(data)
        out.append(str(p))
    return out


def load_local_images(m):
    if "_images" in m or not m.get("image_files"):
        return
    imgs = []
    for f in m["image_files"]:
        try:
            imgs.append((media.mimetypes.guess_type(f)[0] or "image/png", Path(f).read_bytes()))
        except OSError:
            pass
    m["_images"] = imgs


class MessageView(Gtk.Box):
    """One message. Assistant replies render markdown natively (prose, lists, code blocks)."""

    def __init__(self, role, text="", stats=None, error=None, stopped=False, images=None, speak=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.role = role
        self.speak = speak              # callable(text, button) for read-aloud, or None
        self.images = images or []
        self.add_css_class("msg")
        self.add_css_class(role)
        self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        if role == "user":
            self.set_halign(Gtk.Align.END)
            self.body.add_css_class("user-bubble")
        self.append(self.body)
        self.footer = Gtk.Box(spacing=12)
        self.footer.add_css_class("msg-footer")
        self.append(self.footer)
        self.text = text
        self.render(text, error=error, stopped=stopped)
        if stats or (role == "assistant" and text and not error):
            self.set_stats(stats or {})

    def _label(self, markup=None, text=None, css=None):
        lab = Gtk.Label(wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, xalign=0, selectable=True)
        lab.set_max_width_chars(80)
        if markup is not None and markup_ok(markup):
            lab.set_markup(markup)
            lab.connect("activate-link", lambda _l, uri: (Gtk.UriLauncher.new(uri).launch(lab.get_root(), None, None, None), True)[1])
        else:
            lab.set_text(text if text is not None else "")      # the plain source, never raw markup
        if css:
            lab.add_css_class(css)
        return lab

    def render(self, text, streaming=False, error=None, stopped=False):
        self.text = text
        while (c := self.body.get_first_child()) is not None:
            self.body.remove(c)
        if self.role != "user" and not streaming and (error or stopped or not text):
            if text:
                self.render(text)                       # keep the partial reply, note below it
            self.body.append(self._label(text=error or ("Stopped." if stopped else "The model returned an empty reply."),
                                         css="error" if error else "dim-label"))
            return
        if self.role == "user":
            if self.images:
                strip = Gtk.Box(spacing=6, halign=Gtk.Align.END)
                for _mime, data in self.images:
                    try:
                        tex = Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))
                    except GLib.Error:
                        continue
                    pic = Gtk.Image.new_from_paintable(tex)      # fixed size: Gtk.Picture grows to the image
                    pic.set_pixel_size(140)
                    pic.add_css_class("attach-thumb")
                    pic.update_property([Gtk.AccessibleProperty.LABEL], ["Attached image"])
                    strip.append(pic)
                self.body.append(strip)
            if text:
                self.body.append(self._label(text=text))
            return
        if not text and streaming:
            sp = spinner()
            sp.set_halign(Gtk.Align.START)
            self.body.append(sp)
            return
        reasoning, answer, thinking = markdown.split_reasoning(text)
        if reasoning:
            ex = Gtk.Expander(label="Thinking…" if thinking else "Reasoning", expanded=thinking)
            ex.add_css_class("reasoning")
            ex.set_child(self._label(text=reasoning, css="dim-label"))
            self.body.append(ex)
        for b in markdown.blocks(answer):
            if b[0] == "text":
                self.body.append(self._label(markup=b[1], text=b[2]))
            else:
                self.body.append(self._code(b[1], b[2]))

    def _code(self, lang, src):
        frame = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        frame.add_css_class("code-block")
        head = Gtk.Box()
        head.add_css_class("code-head")
        head.append(Gtk.Label(label=lang or "code", hexpand=True, xalign=0, css_classes=["caption", "dim-label"]))
        head.append(icon_button("edit-copy-symbolic", "Copy code", lambda b: copy_text(b, src)))
        frame.append(head)
        sw = Gtk.ScrolledWindow(vscrollbar_policy=Gtk.PolicyType.NEVER, propagate_natural_height=True)
        lab = Gtk.Label(label=src, xalign=0, selectable=True, css_classes=["monospace", "code-text"])
        sw.set_child(lab)
        frame.append(sw)
        return frame

    def set_stats(self, st):
        while (c := self.footer.get_first_child()) is not None:
            self.footer.remove(c)
        bits = []
        if st.get("decode_tok_s"):
            bits.append(f"{st['decode_tok_s']} tok/s")
        if st.get("completion_tokens") is not None:
            bits.append(f"{st['completion_tokens']} tokens")
        if st.get("ttft_s") is not None:
            bits.append(f"first token {st['ttft_s']} s")
        if st.get("cached_tokens"):
            bits.append(f"reused {st['cached_tokens']} tokens")
        if st.get("joules_per_token"):
            bits.append(f"{st['joules_per_token']} J/token")
        if st.get("engine"):
            bits.append(st["engine"])
        for t in bits:
            self.footer.append(Gtk.Label(label=t, css_classes=["caption", "dim-label"]))
        if self.text:
            self.footer.append(icon_button("edit-copy-symbolic", "Copy reply",
                                           lambda b: copy_text(b, markdown.split_reasoning(self.text)[1].strip())))
            if self.speak:
                self.footer.append(icon_button("audio-volume-high-symbolic", "Read aloud",
                                               lambda b: self.speak(markdown.split_reasoning(self.text)[1], b)))


class ChatPage(Gtk.Box):
    def __init__(self, win):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self.convos = Conversations()
        self.cur = None
        self.models = []
        self.live = None            # the one reply being generated: {convo, view, text, cancel, resp, done}

        self.scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.thread = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        self.thread.add_css_class("thread")
        clamp = Adw.Clamp(maximum_size=820, tightening_threshold=600, child=self.thread)
        self.scroll.set_child(clamp)
        self.append(self.scroll)

        # composer
        comp = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        comp.add_css_class("composer")
        self.input = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, accepts_tab=False,
                                  top_margin=10, bottom_margin=6, left_margin=12, right_margin=12)
        self.input.update_property([Gtk.AccessibleProperty.LABEL], ["Message"])
        isw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, propagate_natural_height=True,
                                 max_content_height=220, child=self.input)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._key)
        self.input.add_controller(keys)
        self.attached = []                       # [(mime, bytes)] for the next message
        self.attach_strip = Gtk.Box(spacing=6, margin_start=10, margin_end=10, margin_top=8, visible=False)
        self.recorder = media.Recorder()
        self.player = media.Player()
        row = Gtk.Box(spacing=8)
        row.add_css_class("composer-row")
        self.attach_btn = icon_button("mail-attachment-symbolic", "Attach images (vision models)", lambda *_: self._pick_images())
        row.append(self.attach_btn)
        self.mic_btn = Gtk.ToggleButton(icon_name="audio-input-microphone-symbolic", tooltip_text="Speak your message")
        self.mic_btn.update_property([Gtk.AccessibleProperty.LABEL], ["Record voice message"])
        self.mic_btn.add_css_class("flat")
        self.mic_btn.connect("toggled", self._mic_toggled)
        self.mic_btn.set_visible(media.Recorder.available())
        row.append(self.mic_btn)
        self.think = Gtk.ToggleButton(tooltip_text="Let reasoning models think before answering (slower)")
        tb = Gtk.Box(spacing=6)
        tb.append(Gtk.Image(icon_name="opennpu-brain-symbolic"))
        tb.append(Gtk.Label(label="Think"))
        self.think.set_child(tb)
        self.think.add_css_class("flat")
        row.append(self.think)
        row.append(Gtk.Label(label="Enter to send, Shift+Enter for a new line", hexpand=True, xalign=0,
                             css_classes=["caption", "dim-label"]))
        self.hint = row.get_last_child()
        self.send_btn = Gtk.Button(icon_name="mail-send-symbolic", tooltip_text="Send (Enter)")
        self.send_btn.update_property([Gtk.AccessibleProperty.LABEL], ["Send"])
        self.send_btn.add_css_class("suggested-action")
        self.send_btn.add_css_class("circular")
        self.send_btn.connect("clicked", lambda *_: self.stop() if self._streaming_here() else self.send())
        row.append(self.send_btn)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, css_classes=["composer-box"])
        box.append(self.attach_strip)
        box.append(isw)
        box.append(row)
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", lambda _t, files, _x, _y: (self._add_images([f.get_path() for f in files.get_files()]), True)[1])
        box.add_controller(drop)
        comp.append(Adw.Clamp(maximum_size=820, tightening_threshold=600, child=box))
        self.append(comp)

        # model picker lives in the header bar
        self.model_list = Gtk.StringList()
        self.picker = Gtk.DropDown(model=self.model_list, tooltip_text="Model")
        self.picker.set_enable_search(True)
        self.picker.set_expression(Gtk.PropertyExpression.new(Gtk.StringObject, None, "string"))
        fac = Gtk.SignalListItemFactory()
        fac.connect("setup", lambda _f, it: it.set_child(Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.MIDDLE,
                                                                     max_width_chars=28)))
        fac.connect("bind", lambda _f, it: it.get_child().set_text(it.get_item().get_string()))
        lfac = Gtk.SignalListItemFactory()
        lfac.connect("setup", lambda _f, it: it.set_child(Gtk.Label(xalign=0)))
        lfac.connect("bind", lambda _f, it: it.get_child().set_text(it.get_item().get_string()))
        self.picker.set_factory(fac)
        self.picker.set_list_factory(lfac)
        self.picker.connect("notify::selected", self._picked)

    # ---- models
    def set_models(self, models):
        keep = self.selected_model()
        self.models = sorted(models, key=lambda m: (ENGINE_ORDER.index(m["engine"]) if m["engine"] in ENGINE_ORDER else 9, m["id"]))
        self.model_list.splice(0, self.model_list.get_n_items(), [m["id"].split(":", 1)[1] for m in self.models])
        want = keep or (self.cur or {}).get("model") or self.win.settings.get("model")
        ids = [m["id"] for m in self.models]
        if want in ids:
            self.picker.set_selected(ids.index(want))
        if self.cur is None and not self.live:
            last = self.win.settings.get("convo")          # reopen the last conversation
            self.show_convo(last if self.convos.get(last) else None)

    def selected_model(self):
        i = self.picker.get_selected()
        return self.models[i]["id"] if self.models and i != Gtk.INVALID_LIST_POSITION and i < len(self.models) else None

    def select_model(self, mid):
        ids = [m["id"] for m in self.models]
        if mid in ids:
            self.picker.set_selected(ids.index(mid))

    def _picked(self, *_):
        m = self.selected_model()
        if m:
            self.win.settings["model"] = m
            self.win.save_settings()

    # ---- conversations
    def show_convo(self, cid):
        self.cur = self.convos.get(cid) if cid else None
        self.win.settings["convo"] = cid if self.cur else None
        self.win.save_settings()
        while (c := self.thread.get_first_child()) is not None:
            self.thread.remove(c)
        if self.cur is not None and self.cur.get("messages") is None:     # server session not fetched yet
            sp = spinner()
            sp.set_halign(Gtk.Align.CENTER)
            self.thread.append(sp)
            want = self.cur
            self.convos.load(want, lambda c: self.cur is want and self.show_convo(c["id"]))
            self.win.refresh_convos()
            return
        if self.cur is not None and self.cur.pop("_load_error", None):
            self.win.toast("Could not load this conversation from the server")
        if not self.cur or not self.cur["messages"]:
            self.thread.append(self._empty())
        else:
            for m in self.cur["messages"]:
                load_local_images(m)
                self.thread.append(MessageView(m["role"], m.get("content", ""), m.get("stats"), m.get("error"),
                                               m.get("stopped"), m.get("_images"), self.speak))
            if self.live and self.live["convo"] is self.cur:       # re-attach the reply still streaming here
                v = MessageView("assistant", speak=self.speak)
                v.render(self.live["text"], streaming=True)
                self.thread.append(v)
                self.live["view"] = v
            if self.cur.get("model"):
                self.select_model(self.cur["model"])
            self._scroll_end(force=True)
        self._busy()
        self.win.refresh_convos()
        self.input.grab_focus()

    def _empty(self):
        has = bool(self.models)
        sp = Adw.StatusPage(icon_name="opennpu-circuitry-symbolic",
                            title="What should the NPU work on?" if has else "No models installed yet",
                            description=("Everything runs on your Ryzen AI NPU, locally. "
                                         "The CPU stays free and nothing leaves this machine.") if has else
                            "Open Models to download one, or check that the server is running.")
        if has:
            grid = Gtk.FlowBox(max_children_per_line=2, min_children_per_line=1, selection_mode=Gtk.SelectionMode.NONE,
                               column_spacing=10, row_spacing=10, homogeneous=True)
            for q in STARTERS:
                b = Gtk.Button(child=Gtk.Label(label=q, wrap=True, xalign=0), css_classes=["card", "starter"])
                b.connect("clicked", lambda _b, q=q: (self.input.get_buffer().set_text(q), self.send()))
                grid.append(b)
            sp.set_child(Adw.Clamp(maximum_size=620, child=grid))
        return sp

    def new_chat(self):
        self.cur = None
        self.show_convo(None)

    # ---- attachments
    def _pick_images(self):
        flt = Gtk.FileFilter(name="Images")
        for mt in media.IMAGE_TYPES:
            flt.add_mime_type(mt)
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(flt)
        dlg = Gtk.FileDialog(title="Attach images", filters=store, default_filter=flt)

        def done(d, res):
            try:
                files = d.open_multiple_finish(res)
            except GLib.Error:
                return                                       # cancelled
            self._add_images([files.get_item(i).get_path() for i in range(files.get_n_items())])
        dlg.open_multiple(self.win, None, done)

    def _add_images(self, paths):
        for p in paths:
            if not p:
                continue
            if len(self.attached) >= media.MAX_IMAGES:
                self.win.toast(f"Up to {media.MAX_IMAGES} images per message")
                break
            try:
                self.attached.append(media.read_image(p))
            except (ValueError, OSError) as e:
                self.win.toast(str(e))
        self._render_attached()

    def _render_attached(self):
        while (c := self.attach_strip.get_first_child()) is not None:
            self.attach_strip.remove(c)
        for i, (_mime, data) in enumerate(self.attached):
            o = Gtk.Overlay()
            try:
                tex = Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))
            except GLib.Error:
                continue
            thumb = Gtk.Image.new_from_paintable(tex)
            thumb.set_pixel_size(64)
            thumb.add_css_class("attach-thumb")
            o.set_child(thumb)
            x = Gtk.Button(icon_name="window-close-symbolic", tooltip_text="Remove image",
                           halign=Gtk.Align.END, valign=Gtk.Align.START, css_classes=["circular", "osd", "attach-x"])
            x.update_property([Gtk.AccessibleProperty.LABEL], ["Remove image"])
            x.connect("clicked", lambda _b, i=i: (self.attached.pop(i), self._render_attached()))
            o.add_overlay(x)
            self.attach_strip.append(o)
        self.attach_strip.set_visible(bool(self.attached))

    def model_info(self, mid=None):
        mid = mid or self.selected_model()
        return next((m for m in self.models if m["id"] == mid), {})

    # ---- voice in: record -> /v1/audio/transcriptions -> into the message box
    def _mic_toggled(self, btn):
        if btn.get_active():
            try:
                self.recorder.start()
            except Exception as e:  # noqa: BLE001
                btn.set_active(False)
                return self.win.toast(f"Microphone: {e}")
            btn.add_css_class("destructive-action")
            btn.set_tooltip_text("Stop recording")

            def tick():
                if not self.recorder.recording:
                    return False
                if self.hint is not None:
                    self.hint.set_text(f"Recording {self.recorder.elapsed():.0f} s, click the mic to finish")
                return True
            GLib.timeout_add(250, tick)
            return
        btn.remove_css_class("destructive-action")
        btn.set_tooltip_text("Speak your message")
        if not self.recorder.recording:
            return
        btn.set_sensitive(False)
        if self.hint is not None:
            self.hint.set_text("Transcribing on the NPU…")

        def recorded(path, err):
            if err:
                btn.set_sensitive(True)
                self._busy()
                return self.win.toast(f"Recording: {err}")
            data = Path(path).read_bytes()
            try:
                os.unlink(path)
            except OSError:
                pass

            def ok(text):
                btn.set_sensitive(True)
                self._busy()
                if not text:
                    return self.win.toast("Didn't catch anything; try again closer to the mic")
                buf = self.input.get_buffer()
                cur = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
                buf.set_text((cur.rstrip() + " " if cur.strip() else "") + text)
                self.input.grab_focus()

            def fail(e):
                btn.set_sensitive(True)
                self._busy()
                self.win.toast("This server can't transcribe speech yet" if getattr(e, "code", None) == 404
                               else f"Transcription failed: {e}")
            run_bg(lambda: api.transcribe(data), ok, fail)
        self.recorder.stop(recorded)

    # ---- voice out: read a reply aloud with the speech model
    def speak(self, text, button=None):
        if self.player.playing:
            self.player.stop()
            return
        text = media.speakable(text)
        if not text:
            return
        model = self.win.speech_model()
        if not model:
            return self.win.toast("No speech model: install generation support (./install.sh --gen)")
        if button is not None:
            button.set_sensitive(False)

        def ok(wav):
            if button is not None:
                button.set_sensitive(True)
            self.player.play(wav)

        def fail(e):
            if button is not None:
                button.set_sensitive(True)
            self.win.toast(f"Read aloud failed: {e}")
        run_bg(lambda: api.speech(model, text), ok, fail)

    # ---- sending
    def _key(self, ctrl, keyval, code, state):
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and not state & Gdk.ModifierType.SHIFT_MASK:
            self.send()
            return True
        if keyval == Gdk.KEY_Escape and self._streaming_here():
            self.stop()
            return True
        return False

    def _streaming_here(self):
        return bool(self.live and self.cur is not None and self.live["convo"] is self.cur)

    @staticmethod
    def history(convo, max_chars=24000):
        """What the model sees: no failed turns (nor the question that failed), no empty replies,
        no reasoning, roles strictly alternating; oldest turns dropped beyond max_chars."""
        msgs, ms = [], convo["messages"]
        i = 0
        while i < len(ms):
            m = ms[i]
            if m["role"] == "user" and i + 1 < len(ms) and ms[i + 1].get("error"):
                i += 2
                continue
            i += 1
            if m.get("error") or (m["role"] == "assistant" and not m.get("content")):
                continue
            c = m["content"] if m["role"] == "user" else (markdown.split_reasoning(m["content"])[1].strip() or m["content"])
            if m["role"] == "user":
                load_local_images(m)
                imgs = m.get("_images") or []
            else:
                imgs = []
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1]["content"] += "\n\n" + c
                msgs[-1]["_images"] = msgs[-1].get("_images", []) + imgs
            else:
                msgs.append({"role": m["role"], "content": c, "_images": imgs})
        while len(msgs) > 1 and sum(len(x["content"]) for x in msgs) > max_chars:
            msgs.pop(0)
        if msgs and msgs[0]["role"] == "assistant":
            msgs.pop(0)
        # images only on the latest user turn that has them: older pictures cost a lot of prefill
        # and the model has already answered about them
        last_img = max((i for i, x in enumerate(msgs) if x["_images"]), default=None)
        return [{"role": x["role"], "content": media.content_parts(x["content"], x["_images"] if i == last_img else [])}
                for i, x in enumerate(msgs)]

    def send(self):
        if self.live:
            self.win.toast("Wait for this reply, or press Stop (Esc)" if self._streaming_here()
                           else "Another conversation is still generating")
            return
        buf = self.input.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        model = self.selected_model()
        images = list(self.attached)
        if not text and not images:
            return
        if not model:
            self.win.toast("No model selected" if self.models else "No models yet: start the server or add one on Models")
            return
        if images and not self.model_info(model).get("vision"):
            vis = [m["id"].split(":", 1)[1] for m in self.models if m.get("vision")]
            self.win.toast(f"{model.split(':', 1)[1]} can't see images. " +
                           (f"Pick a vision model, e.g. {vis[0]}" if vis else "No vision model is installed yet"))
            return
        if not text:
            text = "What is in this image?" if len(images) == 1 else "What is in these images?"
        if self.cur is None:
            self.cur = self.convos.new(model)
            self.cur["title"] = text[:60]
            self.win.settings["convo"] = self.cur["id"]
            self.win.save_settings()
            while (c := self.thread.get_first_child()) is not None:
                self.thread.remove(c)
        self.cur["model"] = model
        um = {"role": "user", "content": text}
        if images:
            um["_images"] = images
            if not self.convos.remote:
                um["image_files"] = save_local_images(images)
        self.cur["messages"].append(um)
        self.convos.items.remove(self.cur)
        self.convos.items.insert(0, self.cur)                  # most recent first
        buf.set_text("")
        self.attached = []
        self._render_attached()
        self.thread.append(MessageView("user", text, images=images))
        reply = MessageView("assistant", speak=self.speak)
        reply.render("", streaming=True)
        self.thread.append(reply)
        self.convos.save()
        self.win.refresh_convos()
        self._scroll_end(force=True)
        self._stream(model, self.history(self.cur), reply, self.think.get_active())

    def _stream(self, model, msgs, view, think):
        live = self.live = {"convo": self.cur, "view": view, "text": "", "dirty": False, "done": False,
                            "cancel": threading.Event(), "resp": [], "stats": None}
        self._busy()

        def tick():                              # repaint at most ~16 fps while streaming
            if live["done"]:
                return False
            if live["dirty"]:
                live["dirty"] = False
                v = live["view"]
                stick = self._at_bottom()
                v.render(live["text"], streaming=True)
                if stick and live["convo"] is self.cur:
                    self._scroll_end(force=True)
            return True
        GLib.timeout_add(60, tick)

        def on_text(t):
            live["text"] += t
            live["dirty"] = True

        convo = self.cur
        sid = ("new" if convo.get("_new") else convo["id"]) if self.convos.remote else None

        def work():
            try:
                api.chat_stream(model, msgs, think, live["cancel"].is_set, on_text,
                                lambda st: GLib.idle_add(self._finish, live, st, None), live["resp"], session=sid)
            except Exception as e:  # noqa: BLE001
                GLib.idle_add(self._finish, live, None, None if live["cancel"].is_set() else e)
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        """Stop now, even while the model is still loading: the UI is final at once and the
        connection is shut so the server frees the NPU; the worker's late result is ignored."""
        live = self.live
        if not live:
            return
        live["cancel"].set()
        api.abort(live["resp"])
        self._finish(live, live.get("stats"), None)

    def _finish(self, live, stats=None, error=None):
        if live["done"]:
            return False
        live["done"] = True
        text = live["text"]
        stopped = live["cancel"].is_set()
        m = {"role": "assistant", "content": text}
        if stats:
            m["stats"] = stats
        if error:
            m["error"] = ("The reply was cut off: " if text else "Could not get a reply: ") + str(error)
        if stopped:
            m["stopped"] = True
        live["convo"]["messages"].append(m)
        convo = live["convo"]
        sid = (stats or {}).get("session")
        if not sid and live["resp"]:                                  # stopped early: the header has it
            try:
                sid = live["resp"][0].headers.get("X-OpenNPU-Session")
            except Exception:  # noqa: BLE001
                sid = None
        if convo.get("_new") and sid:                                 # the server created it
            old = convo["id"]
            convo["id"] = sid
            title = convo["title"]
            run_bg(lambda: api.session_update(sid, title=title))
            convo.pop("_new", None)
            if self.win.settings.get("convo") == old:
                self.win.settings["convo"] = convo["id"]
                self.win.save_settings()
            self.win.refresh_convos()
        self.convos.save()
        if self.live is live:
            self.live = None
        v = live["view"]
        stick = self._at_bottom()
        v.render(text, error=m.get("error"), stopped=stopped)
        if stats:
            v.set_stats(stats)
        self._busy()
        if stick and live["convo"] is self.cur:
            self._scroll_end(force=True)
        self.win.poll_now()
        return False

    def save_partial(self):
        """Window closing mid-reply: keep what arrived, marked as stopped."""
        if self.live:
            self.stop()

    def _busy(self, *_):
        here, elsewhere = self._streaming_here(), bool(self.live) and not self._streaming_here()
        self.send_btn.set_icon_name("media-playback-stop-symbolic" if here else "mail-send-symbolic")
        label = "Stop (Esc)" if here else "Another conversation is generating" if elsewhere else "Send (Enter)"
        self.send_btn.set_tooltip_text(label)
        self.send_btn.update_property([Gtk.AccessibleProperty.LABEL], ["Stop" if here else "Send"])
        self.send_btn.set_sensitive(not elsewhere)
        self.send_btn.remove_css_class("suggested-action" if here else "destructive-action")
        self.send_btn.add_css_class("destructive-action" if here else "suggested-action")
        if self.hint is not None:
            self.hint.set_text("Another conversation is still generating" if elsewhere else
                               "Esc to stop" if here else "Enter to send, Shift+Enter for a new line")

    def _at_bottom(self):
        adj = self.scroll.get_vadjustment()
        return adj.get_upper() - adj.get_page_size() - adj.get_value() < 80

    def _scroll_end(self, force=False):
        if not force and not self._at_bottom():
            return

        def go():
            adj = self.scroll.get_vadjustment()
            adj.set_value(adj.get_upper() - adj.get_page_size())
        GLib.idle_add(go)


# ====================================================================== create
class CreatePage(Gtk.Box):
    TASKS = (("image", "Image", "opennpu-image-symbolic"), ("video", "Video", "opennpu-film-strip-symbolic"),
             ("speech", "Speech", "opennpu-waveform-symbolic"))
    SIZES = {"image": ["512x512", "768x768", "1024x1024", "768x512", "512x768"],
             "video": ["480x320", "512x320", "320x320", "256x256"]}

    def __init__(self, win):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self.models = []
        self.stack = Adw.ViewStack(vexpand=True)
        self.forms = {}
        for task, title, icon in self.TASKS:
            self.forms[task] = self._form(task)
            self.stack.add_titled_with_icon(self.forms[task]["root"], task, title, icon)
        self.stack.set_visible_child_name("image")
        self.switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        self.append(self.stack)

    def _form(self, task):
        f = {"task": task}
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14, css_classes=["create-form"])
        f["models"] = Gtk.StringList()
        f["picker"] = Gtk.DropDown(model=f["models"], hexpand=True)
        f["note"] = Gtk.Label(xalign=0, wrap=True, css_classes=["caption", "dim-label"])
        f["picker"].connect("notify::selected", lambda *_: self._note(f))
        mrow = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mrow.append(Gtk.Label(label="Model", xalign=0, css_classes=["heading"]))
        mrow.append(f["picker"])
        mrow.append(f["note"])
        page.append(mrow)
        page.append(Gtk.Label(label="What to say" if task == "speech" else "Prompt", xalign=0, css_classes=["heading"]))
        f["prompt"] = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=8, bottom_margin=8, left_margin=10,
                                   right_margin=10, css_classes=["prompt-box"])
        f["prompt"].update_property([Gtk.AccessibleProperty.LABEL], ["Prompt"])
        psw = Gtk.ScrolledWindow(min_content_height=90, max_content_height=180, propagate_natural_height=True,
                                 hscrollbar_policy=Gtk.PolicyType.NEVER, child=f["prompt"], css_classes=["prompt-frame"])
        page.append(psw)
        opts = Gtk.Box(spacing=12)
        if task in self.SIZES:
            f["size"] = Gtk.DropDown.new_from_strings(self.SIZES[task])
            opts.append(self._opt("Size", f["size"]))
            f["steps"] = Gtk.SpinButton.new_with_range(0, 100, 1)
            f["steps"].set_tooltip_text("0 = the model's default")
            opts.append(self._opt("Steps", f["steps"]))
            f["seed"] = Gtk.Entry(placeholder_text="random", width_chars=10)
            opts.append(self._opt("Seed", f["seed"]))
        if task == "video":
            f["frames"] = Gtk.SpinButton.new_with_range(0, 121, 4)
            f["frames"].set_tooltip_text("0 = the model's default")
            opts.append(self._opt("Frames", f["frames"]))
        f["go"] = Gtk.Button(label={"image": "Generate image", "video": "Generate video", "speech": "Speak"}[task],
                             css_classes=["suggested-action", "pill"], valign=Gtk.Align.END)
        f["go_label"] = f["go"].get_label()
        f["go"].connect("clicked", lambda *_: self._cancel(f) if f.get("busy") else self._generate(f))
        spacer = Gtk.Box(hexpand=True)
        opts.append(spacer)
        opts.append(f["go"])
        page.append(opts)
        f["progress"] = Gtk.ProgressBar(visible=False, show_text=False)
        page.append(f["progress"])
        f["status"] = Gtk.Box(spacing=8)
        f["spinner"] = spinner(visible=False)
        f["status_label"] = Gtk.Label(xalign=0, wrap=True, css_classes=["dim-label"])
        f["status"].append(f["spinner"])
        f["status"].append(f["status_label"])
        page.append(f["status"])
        f["result"] = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        page.append(f["result"])
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        scroll.set_child(Adw.Clamp(maximum_size=860, child=page))
        f["root"] = scroll
        return f

    @staticmethod
    def _opt(label, w):
        b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        b.append(Gtk.Label(label=label, xalign=0, css_classes=["caption-heading"]))
        b.append(w)
        return b

    def set_models(self, models):
        self.models = models
        for task, f in self.forms.items():
            ms = [m for m in models if m["task"] == task]
            f["items"] = ms
            f["models"].splice(0, f["models"].get_n_items(), [m["id"] for m in ms])
            self._note(f)

    def _note(self, f):
        i = f["picker"].get_selected()
        items = f.get("items") or []
        if i == Gtk.INVALID_LIST_POSITION or i >= len(items):
            f["note"].set_text("Generation models appear here once the server is running with ./install.sh --gen."
                               if not items else "")
            return
        m = items[i]
        dl = "downloaded" if m.get("downloaded") else "downloads on first use"
        f["note"].set_text(f"{m.get('note', '')}  ·  ~{m.get('ram_gb', '?')} GB RAM  ·  {dl}")

    def _generate(self, f):
        task = f["task"]
        items = f.get("items") or []
        i = f["picker"].get_selected()
        if i == Gtk.INVALID_LIST_POSITION or i >= len(items):
            self.win.toast("No model selected")
            return
        model = items[i]["id"]
        buf = f["prompt"].get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not prompt:
            self.win.toast("Write a prompt first")
            return
        limit = 4096 if task == "speech" else 2000                # the server's limits
        if len(prompt) > limit:
            self.win.toast(f"Too long: {len(prompt)} characters, the limit is {limit}")
            return
        size = f["size"].get_selected_item().get_string() if "size" in f else None
        steps = int(f["steps"].get_value()) or None if "steps" in f else None
        seed = f["seed"].get_text().strip() if "seed" in f else ""
        seed = int(seed) if seed.isdigit() else None
        frames = int(f["frames"].get_value()) or None if "frames" in f else None
        self._set_busy(f, True)
        f["spinner"].set_visible(True)
        dl = "" if items[i].get("downloaded") else " First use downloads the model, which can take a while."
        f["status_label"].set_text({"image": "Generating on the NPU…", "video": "Generating video on the NPU (minutes)…",
                                    "speech": "Synthesizing…"}[task] + dl)
        t0 = time.time()

        def work():
            if task == "image":
                return api.image(model, prompt, size, steps, seed)
            if task == "video":
                return api.video(model, prompt, size, frames, steps, seed)
            return api.speech(model, prompt), None

        def done(res):
            self._set_busy(f, False)
            data, timings = res
            self._show(f, data, timings, time.time() - t0)

        def fail(e):
            self._set_busy(f, False)
            f["status_label"].set_text("Cancelled" if getattr(e, "cancelled", False) else f"Failed: {e}")
            f["spinner"].set_visible(False)
        run_bg(work, done, fail)
        self._watch(f, t0)

    def _set_busy(self, f, busy):
        f["busy"] = busy
        f["go"].set_sensitive(True)
        f["go"].set_label("Cancel" if busy else f["go_label"])
        f["go"].remove_css_class("suggested-action" if busy else "destructive-action")
        f["go"].add_css_class("destructive-action" if busy else "suggested-action")
        f["progress"].set_visible(busy)
        f["progress"].set_fraction(0)

    def _cancel(self, f):
        f["go"].set_sensitive(False)
        f["go"].set_label("Cancelling…")
        run_bg(api.gen_cancel, None, lambda e: self.win.toast(f"Could not cancel: {e}"))

    def _watch(self, f, t0):
        """Poll /v1/gen/progress (it takes no lock) while this form's job runs."""
        def show(p):
            if not f.get("busy"):
                return
            el = int(time.time() - t0)
            steps, step = p.get("steps") or 0, p.get("step") or 0
            if p.get("phase") == "loading":
                f["progress"].pulse()
                txt = "Loading the model"
            elif steps > 1:
                f["progress"].set_fraction(min(1.0, step / steps))
                txt = f"Step {min(step, steps)} of {steps}"
            else:
                f["progress"].pulse()
                txt = "Generating on the NPU"
            f["status_label"].set_text(f"{txt}, {el // 60}:{el % 60:02d}")

        def tick():
            if not f.get("busy"):
                return False
            run_bg(api.gen_progress, show)
            return True
        GLib.timeout_add(1000, tick)

    def _show(self, f, data, timings, wall):
        task = f["task"]
        f["spinner"].set_visible(False)
        t = timings[0] if isinstance(timings, list) and timings else (timings or {})
        secs = t.get("seconds") if isinstance(t, dict) else None
        ext = {"image": ".png", "video": ".mp4", "speech": ".wav"}[task]
        out_dir = DATA / "creations"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            while (path := out_dir / f"{stamp}-{task}{f'-{n}' if n else ''}{ext}").exists():
                n += 1
            path.write_bytes(data)
            tex = Gdk.Texture.new_from_bytes(GLib.Bytes.new(data)) if task == "image" else None
        except (OSError, GLib.Error) as e:
            f["status_label"].set_text(f"Generated, but could not be saved or shown: {e}")
            return
        f["status_label"].set_text(f"Done in {wall:.1f} s" + (f" ({secs:.1f} s on the NPU)" if secs else ""))
        while (c := f["result"].get_first_child()) is not None:
            f["result"].remove(c)
        if task == "image":
            h = min(380, tex.get_height())                # fit the page; full size is in the saved file
            pic = Gtk.Picture(paintable=tex, can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN,
                              height_request=h, vexpand=False, css_classes=["result-media"])
            f["result"].append(pic)
        elif task == "video":
            vid = Gtk.Video(autoplay=True, loop=True, height_request=380, css_classes=["result-media"])
            vid.set_filename(str(path))
            f["result"].append(vid)
        else:
            media = Gtk.MediaFile.new_for_filename(str(path))
            f["result"].append(Gtk.MediaControls(media_stream=media))
            media.play()
        row = Gtk.Box(spacing=8)
        row.append(Gtk.Label(label=f"Saved to {path}", xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE,
                             css_classes=["caption", "dim-label"]))
        save = Gtk.Button(label="Save as…", css_classes=["pill"])
        save.connect("clicked", lambda *_: self._save_as(path))
        row.append(save)
        f["result"].append(row)

    def _save_as(self, path):
        dlg = Gtk.FileDialog(initial_name=path.name)

        def done(d, res):
            try:
                dest = d.save_finish(res)
            except GLib.Error:
                return                                   # cancelled
            try:
                if dest.get_path():
                    Path(dest.get_path()).write_bytes(path.read_bytes())
                else:                                     # a non-local destination (e.g. a phone over MTP)
                    dest.replace_contents(path.read_bytes(), None, False, Gio.FileCreateFlags.REPLACE_DESTINATION, None)
            except (OSError, GLib.Error) as e:
                self.win.toast(f"Could not save: {e}")
                return
            self.win.toast(f"Saved {dest.get_basename()}")
        dlg.save(self.win, None, done)


# ====================================================================== models
class ModelsPage(Gtk.ScrolledWindow):
    NOTES = {"moe": "Mixture-of-experts on OpenNPU's own open kernels",
             "native": "Dense models on OpenNPU's own open fused kernels",
             "npu4": "Any other LLM, dense or mixture-of-experts: 4-bit weights, every large matmul on the NPU",
             "llama": "GGUF models (incl. your LM Studio folder) run by llama.cpp, its matrix products on the NPU",
             "universal": "Vision and other Hugging Face models: linear layers on the NPU, decode on the CPU"}

    def __init__(self, win):
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.win = win
        self.page = Adw.PreferencesPage()
        self.set_child(self.page)
        self.groups = []
        self.loaded = None
        # download
        g = Adw.PreferencesGroup(title="Add a model",
                                 description="A Hugging Face repo id (e.g. Qwen/Qwen3-8B). "
                                             "Any Hugging Face chat model also works directly in Chat by its repo id.")
        self.pull = Adw.EntryRow(title="Hugging Face repo id", show_apply_button=True)
        self.pull.connect("apply", self._pull)
        g.add(self.pull)
        self.pull_status = Adw.ActionRow(title="", visible=False)
        self.pull_spin = spinner()
        self.pull_status.add_prefix(self.pull_spin)
        g.add(self.pull_status)
        browse = Adw.ActionRow(title="Browse Hugging Face", subtitle="Search, see how a model runs on the NPU, and download it",
                               activatable=True)
        browse.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
        browse.connect("activated", lambda *_: self.win.show("discover"))
        g.add(browse)
        self.page.add(g)
        self.pull_group = g

    def set_models(self, chat, gen, loaded):
        self.loaded = loaded
        for g in self.groups:
            self.page.remove(g)
        self.groups = []
        if not chat and not gen:
            g = Adw.PreferencesGroup(title="No models found",
                                     description="Start the server (System page) or add a model above.")
            self.page.add(g)
            self.groups.append(g)
        for eng in ENGINE_ORDER:
            ms = [m for m in chat if m["engine"] == eng]
            if not ms:
                continue
            g = Adw.PreferencesGroup(title=ms[0]["engine_label"], description=self.NOTES.get(eng, ""))
            for m in ms:
                sub = "  ·  ".join(x for x in (m.get("family", ""), gb(m["size_bytes"]) if m.get("size_bytes") else "",
                                               "loaded on the NPU" if m["id"] == loaded else "") if x)
                row = Adw.ActionRow(title=GLib.markup_escape_text(m["name"]), subtitle=GLib.markup_escape_text(sub))
                b = Gtk.Button(label="Chat", valign=Gtk.Align.CENTER, css_classes=["pill"])
                b.connect("clicked", lambda _b, mid=m["id"]: self.win.chat_with(mid))
                row.add_suffix(b)
                g.add(row)
            self.page.add(g)
            self.groups.append(g)
        if gen:
            g = Adw.PreferencesGroup(title="Image, video and speech",
                                     description="Generation models (downloaded on first use)")
            for m in gen:
                sub = f"{m['task']}  ·  ~{m.get('ram_gb', '?')} GB RAM  ·  " + \
                      ("downloaded" if m.get("downloaded") else "not downloaded")
                row = Adw.ActionRow(title=GLib.markup_escape_text(m["id"]),
                                    subtitle=GLib.markup_escape_text(f"{sub}\n{m.get('note', '')}"))
                b = Gtk.Button(label="Create", valign=Gtk.Align.CENTER, css_classes=["pill"])
                b.connect("clicked", lambda _b, mm=m: self.win.create_with(mm))
                row.add_suffix(b)
                g.add(row)
            self.page.add(g)
            self.groups.append(g)

    def _pull(self, row):
        ref = row.get_text().strip()
        if not ref:
            return
        npu = config.HOME / "env" / "bin" / "npu"
        cmd = [str(npu) if npu.exists() else "npu", "pull", ref]
        self.pull_status.set_visible(True)
        self.pull_status.set_title(GLib.markup_escape_text(f"Downloading {ref}…"))
        self.pull_status.set_subtitle("")
        row.set_sensitive(False)

        def work():
            import subprocess
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            last = ""
            for line in p.stdout:
                line = line.strip()
                if line:
                    last = line
                    GLib.idle_add(self.pull_status.set_subtitle, GLib.markup_escape_text(line[-120:]))
            return p.wait(), last

        def done(res):
            rc, last = res
            row.set_sensitive(True)
            self.pull_spin.set_visible(False)
            self.pull_status.set_title(GLib.markup_escape_text(f"{ref}: " + ("done" if rc == 0 else "failed")))
            self.pull_status.set_subtitle(GLib.markup_escape_text(last[-160:]))
            self.win.refresh_models()
        self.pull_spin.set_visible(True)
        run_bg(work, done, lambda e: done((1, str(e))))


# ====================================================================== system
class SystemPage(Gtk.ScrolledWindow):
    def __init__(self, win):
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.win = win
        page = Adw.PreferencesPage()
        self.set_child(page)

        g = Adw.PreferencesGroup(title="Server", description=f"OpenNPU model server at {api.URL}")
        self.srv = Adw.ActionRow(title="Status")
        box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.svc = {}
        for label, action in (("Start", "start"), ("Restart", "restart"), ("Stop", "stop")):
            b = Gtk.Button(label=label, css_classes=["pill"] + (["destructive-action"] if action == "stop" else []))
            b.connect("clicked", lambda _b, a=action: self._service(a))
            box.append(b)
            self.svc[action] = b
        self.srv.add_suffix(box)
        g.add(self.srv)
        self.loaded = Adw.ActionRow(title="Loaded model", subtitle="none")
        unload = Gtk.Button(label="Unload", valign=Gtk.Align.CENTER, css_classes=["pill"],
                            tooltip_text="Free the NPU and its RAM now")
        unload.connect("clicked", lambda *_: run_bg(api.unload, lambda _r: (self.win.toast("Unloaded"), self.win.poll_now()),
                                                     lambda e: self.win.toast(str(e))))
        self.loaded.add_suffix(unload)
        g.add(self.loaded)
        page.add(g)

        g = Adw.PreferencesGroup(title="Live")
        self.busy_row = Adw.ActionRow(title="NPU busy")
        self.level = Gtk.LevelBar(min_value=0, max_value=100, valign=Gtk.Align.CENTER, width_request=160)
        self.busy_lbl = Gtk.Label(css_classes=["numeric"], width_chars=5, xalign=1)
        self.busy_row.add_suffix(self.level)
        self.busy_row.add_suffix(self.busy_lbl)
        g.add(self.busy_row)
        self.vals = {}
        for k, title in (("cpu", "Server CPU"), ("mem", "Server memory"), ("pow", "System power"),
                         ("last", "Last reply"), ("idle", "Unloads in")):
            r = Adw.ActionRow(title=title)
            lab = Gtk.Label(css_classes=["numeric", "dim-label"])
            r.add_suffix(lab)
            self.vals[k] = lab
            g.add(r)
        g.add(Adw.ActionRow(title="About power", subtitle="Whole laptop, read from the battery sensor while unplugged; "
                                                          "the driver reports no NPU-only power.", css_classes=["dim-label"]))
        page.add(g)

        self.dev = Adw.PreferencesGroup(title="Device")
        self.dev_rows = {}
        for k, title in (("device", "NPU"), ("arch", "Architecture"), ("firmware", "Firmware"), ("xrt", "XRT")):
            r = Adw.ActionRow(title=title)
            lab = Gtk.Label(label="unknown", css_classes=["dim-label"], selectable=True)
            r.add_suffix(lab)
            self.dev_rows[k] = lab
            self.dev.add(r)
        page.add(self.dev)

        g = Adw.PreferencesGroup(title="Connect other apps",
                                 description="Open WebUI, Continue, OpenCode, the openai SDK and Ollama clients work unchanged.")
        for title, val in (("OpenAI-compatible", api.URL + "/v1"), ("Ollama-compatible", api.URL),
                           ("Web UI", api.URL + "/")):
            r = Adw.ActionRow(title=title, subtitle=val, subtitle_selectable=True)
            r.add_suffix(icon_button("edit-copy-symbolic", f"Copy {title} URL", lambda b, v=val: (copy_text(b, v), self.win.toast("Copied"))))
            g.add(r)
        page.add(g)

        self.diag = Adw.PreferencesGroup(title="Diagnostics", description="Checks driver, firmware, XRT, permissions, "
                                                                          "the engine environment and an NPU smoke test.")
        self.diag_btn = Gtk.Button(label="Run checks", css_classes=["pill"], valign=Gtk.Align.CENTER)
        self.diag_btn.connect("clicked", self._doctor)
        self.diag.set_header_suffix(self.diag_btn)
        self.diag_rows = []
        page.add(self.diag)
        self.page = page

    def _service(self, action):
        for b in self.svc.values():
            b.set_sensitive(False)
        past = {"start": "started", "stop": "stopped", "restart": "restarted"}[action]

        def done(res):
            ok, msg = res
            self.win.toast(f"Server {past}" if ok else f"Could not {action} the server: {msg}")
            GLib.timeout_add(1500, lambda: (self.win.poll_now(), self.win.refresh_models(), False)[-1])
        run_bg(lambda: api.service(action), done, lambda e: done((False, str(e))))

    def _doctor(self, *_):
        st = self.win.stats or {}
        if st.get("generating"):
            self.win.toast("The NPU is generating right now; run the checks when it is idle")
            return
        if st.get("loaded"):
            self.win.toast("A model is loaded: the NPU smoke test shares the NPU with it")
        self.diag_btn.set_sensitive(False)
        self.diag_btn.set_label("Checking…")

        def done(checks):
            try:
                self._show_checks(checks)
            finally:
                self.diag_btn.set_sensitive(True)
                self.diag_btn.set_label("Run again")

        def fail(e):
            self.win.toast(f"Checks failed: {e}")
            self.diag_btn.set_sensitive(True)
            self.diag_btn.set_label("Run checks")
        run_bg(api.doctor, done, fail)

    def _show_checks(self, checks):
        if True:
            for r in self.diag_rows:
                self.diag.remove(r)
            self.diag_rows = []
            icons = {"ok": "object-select-symbolic", "warn": "dialog-warning-symbolic", "fail": "dialog-error-symbolic"}
            for c in checks:
                sub = str(c.get("detail", "")) + (f"\nFix: {c['fix']}" if c.get("fix") else "")
                r = Adw.ActionRow(title=GLib.markup_escape_text(str(c.get("check", "?"))), subtitle=GLib.markup_escape_text(sub),
                                  subtitle_selectable=True)
                img = Gtk.Image(icon_name=icons.get(c["status"], "dialog-question-symbolic"))
                img.add_css_class({"ok": "success", "warn": "warning", "fail": "error"}.get(c["status"], ""))
                r.add_prefix(img)
                self.diag.add(r)
                self.diag_rows.append(r)

    def update(self, st):
        self.svc["start"].set_sensitive(st is None)
        self.svc["restart"].set_sensitive(True)
        self.svc["stop"].set_sensitive(st is not None)
        if st is None:
            self.srv.set_subtitle("not running")
            self.loaded.set_subtitle("none")
            for lab in self.vals.values():
                lab.set_text("-")
            self.level.set_value(0)
            self.busy_lbl.set_text("-")
            return
        self.srv.set_subtitle("generating on the NPU" if st.get("generating") else "running")
        self.loaded.set_subtitle(GLib.markup_escape_text(st.get("loaded") or "none"))
        busy = st.get("npu_busy_pct", 0)
        self.level.set_value(busy)
        self.busy_lbl.set_text(f"{busy:.0f}%")
        self.vals["cpu"].set_text(f"{st.get('cpu_pct', 0):.0f}%")
        self.vals["mem"].set_text(gb(st.get("rss_bytes", 0)))
        p = st.get("system_power_w")
        self.vals["pow"].set_text("plugged in" if p is None else f"{p:.1f} W")
        last = st.get("last") or {}
        self.vals["last"].set_text(" · ".join(x for x in (
            f"{last['decode_tok_s']} tok/s" if last.get("decode_tok_s") else "",
            f"first token {last['ttft_s']} s" if last.get("ttft_s") is not None else "",
            f"{last['joules_per_token']} J/token" if last.get("joules_per_token") else "") if x) or "-")
        u = st.get("unload_in_s")
        self.vals["idle"].set_text("-" if u is None else f"{u // 60}:{u % 60:02d}")
        for k, lab in self.dev_rows.items():
            v = (st.get("npu") or {}).get(k)
            if v:
                lab.set_text(v)


# ====================================================================== window
class Window(Adw.ApplicationWindow):
    PAGES = (("chat", "Chat", "opennpu-chat-circle-text-symbolic"), ("create", "Create", "opennpu-sparkle-symbolic"),
             ("dashboard", "Dashboard", "opennpu-chart-bar-symbolic"),
             ("models", "Models", "opennpu-stack-symbolic"), ("discover", "Discover", "opennpu-compass-symbolic"),
             ("system", "System", "opennpu-cpu-symbolic"))

    def __init__(self, app):
        super().__init__(application=app, title="OpenNPU", default_width=1180, default_height=780)
        self.set_size_request(360, 480)
        self.settings = settings()
        self.up = None

        self.chat = ChatPage(self)
        self.create = CreatePage(self)
        self.models = ModelsPage(self)
        self.system = SystemPage(self)
        self.dashboard = DashboardPage(self)
        self.discover = DiscoverPage(self)
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE, transition_duration=120)
        for name, page in (("chat", self.chat), ("create", self.create), ("dashboard", self.dashboard),
                           ("models", self.models), ("discover", self.discover), ("system", self.system)):
            self.stack.add_named(page, name)

        # ---- sidebar
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.nav = Gtk.ListBox(css_classes=["navigation-sidebar"])
        self.nav.connect("row-activated", lambda _l, row: self.show(row.page))
        for name, title, icon in self.PAGES:
            row = Gtk.ListBoxRow()
            row.page = name
            b = Gtk.Box(spacing=12, margin_start=6, margin_top=4, margin_bottom=4)
            b.append(Gtk.Image(icon_name=icon))
            b.append(Gtk.Label(label=title, xalign=0))
            row.set_child(b)
            self.nav.append(row)
        side.append(self.nav)
        head = Gtk.Box(margin_start=18, margin_end=8, margin_top=14, margin_bottom=2)
        head.append(Gtk.Label(label="Conversations", xalign=0, hexpand=True, css_classes=["caption-heading", "dim-label"]))
        side.append(head)
        self.convo_list = Gtk.ListBox(css_classes=["navigation-sidebar"])
        self.convo_list.connect("row-activated", lambda _l, row: (self.chat.show_convo(row.cid), self.show("chat")))
        csw = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER, child=self.convo_list)
        side.append(csw)

        side_tv = Adw.ToolbarView()
        sh = Adw.HeaderBar()
        sh.set_title_widget(Adw.WindowTitle(title="OpenNPU", subtitle=""))
        self.side_title = sh.get_title_widget()
        sh.pack_end(icon_button("list-add-symbolic", "New conversation (Ctrl+N)", lambda *_: self.new_chat()))
        menu = Gio.Menu()
        style = Gio.Menu()
        for label, val in (("Follow system", "system"), ("Light", "light"), ("Dark", "dark")):
            style.append(label, f"app.theme::{val}")
        menu.append_section("Appearance", style)
        menu.append("Open web UI", "app.web")
        menu.append("Keyboard shortcuts", "win.show-help-overlay")
        menu.append("About OpenNPU", "app.about")
        mb = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Menu")
        mb.update_property([Gtk.AccessibleProperty.LABEL], ["Menu"])
        sh.pack_start(mb)
        side_tv.add_top_bar(sh)
        side_tv.set_content(side)

        # ---- content
        self.content_tv = Adw.ToolbarView()
        self.header = Adw.HeaderBar()
        self.title = Adw.WindowTitle(title="Chat")
        self.header.set_title_widget(self.title)
        self.content_tv.add_top_bar(self.header)
        self.banner = Adw.Banner(title="The OpenNPU server is not running", button_label="Start server")
        self.banner.connect("button-clicked", lambda *_: self.system._service("start"))
        self.content_tv.add_top_bar(self.banner)
        self.toasts = Adw.ToastOverlay(child=self.stack)
        self.content_tv.set_content(self.toasts)
        self.status_chip = Gtk.Label(css_classes=["status-chip"])
        self.header.pack_end(self.status_chip)

        self.split = Adw.NavigationSplitView(min_sidebar_width=230, max_sidebar_width=300)
        self.split.set_sidebar(Adw.NavigationPage.new(side_tv, "OpenNPU"))
        self.content_page = Adw.NavigationPage.new(self.content_tv, "Chat")
        self.split.set_content(self.content_page)
        self.set_content(self.split)
        bp = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 720sp"))
        bp.add_setter(self.split, "collapsed", True)
        self.add_breakpoint(bp)
        narrow = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 480sp"))
        narrow.add_setter(self.status_chip, "visible", False)          # the picker needs the room
        narrow.add_setter(self.create.switcher, "policy", Adw.ViewSwitcherPolicy.NARROW)
        self.add_breakpoint(narrow)
        self.connect("close-request", self._closing)

        self._actions()
        self.show(self.settings.get("page", "chat"))
        self.refresh_convos()
        self.chat.convos.go_remote(self.settings, self._sessions_ready)
        self.poll_now()
        self.refresh_models()
        GLib.timeout_add_seconds(2, self._poll)
        snap = os.environ.get("OPENNPU_DESKTOP_SNAPSHOT")    # dev/test: render self to PNG, then quit
        if snap:
            GLib.timeout_add(int(float(os.environ.get("OPENNPU_DESKTOP_SNAPSHOT_DELAY", "4")) * 1000),
                             lambda: (self._snapshot(snap), self.get_application().quit(), False)[-1])

    def _sessions_ready(self, remote):
        if not remote:
            return
        if self.chat.live:                       # don't pull the rug from under a streaming reply
            return
        last = self.settings.get("convo")
        self.chat.cur = None
        self.chat.show_convo(last if self.chat.convos.get(last) else None)

    def speech_model(self):
        ms = [m["id"] for m in self.create.models if m.get("task") == "speech"]
        return "mms-tts-eng" if "mms-tts-eng" in ms else (ms[0] if ms else None)

    def _closing(self, *_):
        self.chat.player.stop()
        if self.chat.recorder.recording:
            self.chat.recorder.cancel()
        self.chat.save_partial()                    # a reply mid-stream is kept, marked as stopped
        self.settings.save()
        return False

    def _snapshot(self, path):
        w, h = self.get_width(), self.get_height()
        snap = Gtk.Snapshot()
        Gtk.WidgetPaintable(widget=self).snapshot(snap, w, h)
        node = snap.to_node()
        if node is not None:
            self.get_renderer().render_texture(node, None).save_to_png(path)

    # ---- settings
    def save_settings(self):
        self.settings.save()

    # ---- navigation
    def show(self, page):
        titles = dict((n, t) for n, t, _ in self.PAGES)
        if page not in titles:
            page = "chat"
        self.stack.set_visible_child_name(page)
        title = titles[page]
        self.content_page.set_title(title)
        for i, (n, _, _) in enumerate(self.PAGES):
            if n == page:
                self.nav.select_row(self.nav.get_row_at_index(i))
        self.header.set_title_widget(self.chat.picker if page == "chat" else
                                     self.create.switcher if page == "create" else self.title)
        self.title.set_title(title)
        self.split.set_show_content(True)
        if page == "dashboard":
            self.dashboard.refresh()
        if page == "discover":
            self.discover.on_show()
        self.settings["page"] = page
        self.save_settings()

    def new_chat(self):
        self.chat.new_chat()
        self.show("chat")

    def chat_with(self, mid):
        self.chat.new_chat()
        self.chat.select_model(mid)
        self.show("chat")

    def create_with(self, m):
        f = self.create.forms[m["task"]]
        ids = [x["id"] for x in f.get("items") or []]
        if m["id"] in ids:
            f["picker"].set_selected(ids.index(m["id"]))
        self.create.stack.set_visible_child_name(m["task"])
        self.show("create")

    def refresh_convos(self):
        while (r := self.convo_list.get_row_at_index(0)) is not None:
            self.convo_list.remove(r)
        cur = self.chat.cur["id"] if self.chat.cur else None
        for c in self.chat.convos.items:
            row = Gtk.ListBoxRow()
            row.cid = c["id"]
            b = Gtk.Box(spacing=4)
            b.append(Gtk.Label(label=c["title"], xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.END,
                               margin_start=6))
            b.append(icon_button("user-trash-symbolic", f"Delete conversation: {c['title'][:40]}", lambda _b, cid=c["id"]: self._delete(cid)))
            row.set_child(b)
            self.convo_list.append(row)
            if c["id"] == cur:
                self.convo_list.select_row(row)

    def _delete(self, cid):
        items = self.chat.convos.items
        idx = next((i for i, c in enumerate(items) if c["id"] == cid), None)
        if idx is None:
            return
        convo = items[idx]
        if self.chat.live and self.chat.live["convo"] is convo:
            self.chat.stop()
        if self.chat.cur is convo:
            self.chat.cur = None
            self.chat.show_convo(None)
        remote = self.chat.convos.remote
        self.chat.convos.delete(cid, now=not remote)
        self.refresh_convos()
        state = {"undone": False}
        if remote and not convo.get("_new"):              # really delete once Undo can't be pressed
            def commit():
                if not state["undone"]:
                    run_bg(lambda: api.session_delete(cid), None,
                           lambda e: self.toast(f"Could not delete on the server: {e}"))
                return False
            GLib.timeout_add_seconds(6, commit)

        def undo(*_):
            state["undone"] = True
            self.chat.convos.items.insert(min(idx, len(self.chat.convos.items)), convo)
            self.chat.convos.save()
            self.chat.show_convo(convo["id"])
            self.show("chat")
        self.toast("Conversation deleted", "Undo", undo)

    def toast(self, text, button=None, cb=None):
        t = Adw.Toast(title=GLib.markup_escape_text(str(text)), timeout=5 if button else 3)
        if button:
            t.set_button_label(button)
            t.connect("button-clicked", cb)
        self.toasts.add_toast(t)

    # ---- server state
    def refresh_models(self):
        def work():
            return api.chat_models(), api.gen_models()

        def done(res):
            chat, gen = res
            self.chat.set_models(chat)
            self.create.set_models(gen)
            self.models.set_models(chat, gen, (self.stats or {}).get("loaded"))

        def fail(_e):
            self.chat.set_models([])
            self.models.set_models([], [], None)
        run_bg(work, done, fail)

    stats = None

    def poll_now(self):
        def done(st):
            was = self.up
            self.up, self.stats = True, st
            self.banner.set_revealed(False)
            self.system.update(st)
            if st.get("loaded") != self.models.loaded and self.models.groups:
                self.refresh_models()                       # keep "loaded on the NPU" current
            if st.get("loading"):
                txt = "Loading a model"
            elif st.get("generating"):
                txt = "Generating on the NPU"
            elif st.get("loaded"):
                txt = "Ready on the NPU"
            else:
                txt = "NPU idle"
            self.status_chip.set_text(txt)
            self.side_title.set_subtitle(f"NPU {st.get('npu_busy_pct', 0):.0f}% busy" if st.get("generating") else txt)
            if was is False:
                self.refresh_models()

        def fail(_e):
            self.up, self.stats = False, None
            self.banner.set_revealed(True)
            self.system.update(None)
            self.status_chip.set_text("Server offline")
            self.side_title.set_subtitle("server offline")
        run_bg(api.stats, done, fail)

    def _poll(self):
        if self.is_visible():
            self.poll_now()
        return True

    def _actions(self):
        for name, cb, accels in (("new-chat", lambda *_: self.new_chat(), ["<Control>n"]),
                                 ("page-chat", lambda *_: self.show("chat"), ["<Control>1"]),
                                 ("page-create", lambda *_: self.show("create"), ["<Control>2"]),
                                 ("page-dashboard", lambda *_: self.show("dashboard"), ["<Control>3"]),
                                 ("page-models", lambda *_: self.show("models"), ["<Control>4"]),
                                 ("page-system", lambda *_: self.show("system"), ["<Control>5"]),
                                 ("page-discover", lambda *_: self.show("discover"), ["<Control>6"])):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", cb)
            self.add_action(a)
            self.get_application().set_accels_for_action(f"win.{name}", accels)
        self.set_help_overlay(self._shortcuts())

    def _shortcuts(self):
        ui = """<interface><object class="GtkShortcutsWindow" id="help_overlay"><property name="modal">1</property>
        <child><object class="GtkShortcutsSection"><child><object class="GtkShortcutsGroup">
        <property name="title">General</property>""" + "".join(
            f'<child><object class="GtkShortcutsShortcut"><property name="title">{t}</property>'
            f'<property name="accelerator">{a}</property></object></child>'
            for t, a in (("New conversation", "&lt;Control&gt;n"), ("Chat", "&lt;Control&gt;1"), ("Create", "&lt;Control&gt;2"),
                         ("Dashboard", "&lt;Control&gt;3"),
                         ("Models", "&lt;Control&gt;4"), ("System", "&lt;Control&gt;5"), ("Discover", "&lt;Control&gt;6"),
                         ("Send message", "Return"),
                         ("New line", "&lt;Shift&gt;Return"), ("Quit", "&lt;Control&gt;q"))) + \
            "</object></child></object></child></object></interface>"
        return Gtk.Builder.new_from_string(ui, -1).get_object("help_overlay")


_SCHEMES = {"system": Adw.ColorScheme.DEFAULT, "light": Adw.ColorScheme.FORCE_LIGHT, "dark": Adw.ColorScheme.FORCE_DARK}


def _apply_theme(t):
    Adw.StyleManager.get_default().set_color_scheme(_SCHEMES.get(t, Adw.ColorScheme.DEFAULT))


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE)
        GLib.set_application_name("OpenNPU")
        self.connect("activate", self._activate)
        self.connect("command-line", self._command_line)
        self.connect("startup", self._startup)

    def _startup(self, *_):
        theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
        theme.add_search_path(str(HERE / "icons"))
        css = Gtk.CssProvider()
        css.load_from_path(str(HERE / "style.css"))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        for name, cb, accels in (("quit", lambda *_: self.quit(), ["<Control>q"]),
                                 ("about", self._about, []),
                                 ("web", lambda *_: Gtk.UriLauncher.new(api.URL + "/").launch(self.get_active_window(), None, None, None), [])):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", cb)
            self.add_action(a)
            if accels:
                self.set_accels_for_action(f"app.{name}", accels)
        # light / dark / follow-system, remembered across launches
        saved = settings().get("theme", "system")
        if saved not in _SCHEMES:
            saved = "system"
        theme_action = Gio.SimpleAction.new_stateful("theme", GLib.VariantType.new("s"), GLib.Variant.new_string(saved))
        theme_action.connect("change-state", self._set_theme)
        self.add_action(theme_action)
        _apply_theme(saved)

    def _set_theme(self, action, value):
        action.set_state(value)
        t = value.get_string()
        _apply_theme(t)
        settings()["theme"] = t
        settings().save()

    def _activate(self, *_):
        win = self.get_active_window() or Window(self)
        win.present()
        return win

    def _command_line(self, _app, cmdline):
        args = cmdline.get_arguments()[1:]
        self.activate()                          # the normal activation path (and its signal)
        if "--new-chat" in args:
            self.get_active_window().new_chat()
        return 0

    def _about(self, *_):
        kw = dict(application_name="OpenNPU", application_icon=APP_ID, version=__version__,
                  developer_name="OpenNPU contributors", license_type=Gtk.License.APACHE_2_0,
                  comments="Run local AI models on AMD Ryzen AI NPUs under Linux: chat, images, video and "
                           "speech, all on the NPU and all on this machine.")
        if hasattr(Adw, "AboutDialog"):                  # libadwaita >= 1.5
            Adw.AboutDialog(**kw).present(self.get_active_window())
        else:
            Adw.AboutWindow(transient_for=self.get_active_window(), **kw).present()


def main(argv=None):
    return App().run(argv)
