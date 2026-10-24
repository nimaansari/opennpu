"""Desktop Discover page: search Hugging Face through the local server (/api/hub/*), see how each
model runs on the NPU, download it with progress and cancel, then chat with it (or open it in
Create). Downloads run in the server, so the web UI, this app and the CLI share one queue."""
import threading

from gi.repository import Adw, GLib, Gtk

from opennpu.desktop import api

TASKS = (("chat", "Chat"), ("vision", "Vision"), ("embed", "Embeddings"), ("speech", "Speech"),
         ("image", "Image"), ("video", "Video"), ("all", "All"))
SORTS = (("downloads", "Most downloaded"), ("likes", "Most liked"), ("trending_score", "Trending"),
         ("last_modified", "Recently updated"))
TASK_NAME = {"text-generation": "Chat", "image-text-to-text": "Vision", "feature-extraction": "Embeddings",
             "sentence-similarity": "Embeddings", "automatic-speech-recognition": "Speech to text",
             "text-to-speech": "Text to speech", "text-to-audio": "Text to audio", "text-to-image": "Image",
             "image-to-image": "Image to image", "text-to-video": "Video", "image-to-video": "Image to video"}
CREATE_TASK = {"text-to-image": "image", "image-to-image": "image", "text-to-video": "video",
               "image-to-video": "video", "text-to-speech": "speech", "text-to-audio": "speech"}


def run_bg(fn, ok=None, err=None):
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


def spinner():
    """Adw.Spinner needs libadwaita 1.6; Ubuntu 24.04 / Fedora 40 ship 1.5."""
    return Adw.Spinner() if hasattr(Adw, "Spinner") else Gtk.Spinner(spinning=True)


def esc(s):
    return GLib.markup_escape_text(str(s))


def fmt_bytes(b):
    if b is None:
        return "?"
    if b < 1e6:
        return f"{max(1, round(b / 1e3))} KB"
    if b < 1e9:
        return f"{round(b / 1e6)} MB"
    return f"{b / 1e9:.0f} GB" if b >= 1e11 else f"{b / 1e9:.1f} GB"


def fmt_params(p):
    if not p:
        return ""
    return (f"{p / 1e9:.0f}B" if p >= 1e11 else f"{p / 1e9:.1f}B") if p >= 1e9 else f"{round(p / 1e6)}M"


def fmt_count(n):
    n = n or 0
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suf
    return str(n)


def fmt_eta(s):
    if s is None:
        return "estimating"
    if s < 60:
        return f"{max(1, round(s))} s left"
    if s < 3600:
        return f"{round(s / 60)} min left"
    return f"{int(s // 3600)} h {round((s % 3600) / 60)} min left"


def hint(e):
    """An error in words a user can act on."""
    m = str(e)
    if getattr(e, "code", None) == 502 or "offline" in m or "reach huggingface" in m:
        return m + ". Discover needs internet access for search and downloads; installed models keep working offline."
    if "gated" in m and "hf auth login" not in m:
        return m + ". Accept the license on huggingface.co, run `hf auth login` in a terminal, then try again."
    if getattr(e, "code", None) == 404 and "Hugging Face" not in m:
        return "This OpenNPU server has no Discover support yet: update OpenNPU."
    return m


def badge(text, css=None, tooltip=None):
    lab = Gtk.Label(label=text, valign=Gtk.Align.CENTER, css_classes=["hub-badge"] + ([css] if css else []))
    if tooltip:
        lab.set_tooltip_text(tooltip)
    return lab


def make_dialog(win, title, child, width=520, height=560):
    """Adw.Dialog needs libadwaita 1.5; older ones get a modal Adw.Window."""
    tv = Adw.ToolbarView()
    tv.add_top_bar(Adw.HeaderBar())
    tv.set_content(child)
    if hasattr(Adw, "Dialog"):
        d = Adw.Dialog(title=title, content_width=width, content_height=height, child=tv)
        d.present(win)
    else:
        d = Adw.Window(title=title, modal=True, transient_for=win, default_width=width, default_height=height, content=tv)
        d.present()
    return d


class DiscoverPage(Gtk.Box):
    def __init__(self, win):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self.cards = []
        self.pulls = {}                 # id -> job, from /api/hub/pulls
        self.was_downloading = set()
        self.confirm = None             # id whose Remove waits for confirmation
        self.dialog = None              # the open details dialog: {"win", "id", "info", "download", "error", "gguf"}
        self.searched = False
        self.error = ""
        self._seq = 0
        self._poll_id = 0
        self.free_disk = None

        bar = Gtk.Box(spacing=8, margin_start=18, margin_end=18, margin_top=12, margin_bottom=6)
        self.entry = Gtk.SearchEntry(placeholder_text="Search Hugging Face models, e.g. qwen3", hexpand=True)
        self.entry.update_property([Gtk.AccessibleProperty.LABEL], ["Search Hugging Face models"])
        if hasattr(self.entry, "set_search_delay"):                     # GTK >= 4.8
            self.entry.set_search_delay(350)
        self.entry.connect("search-changed", lambda *_: self.search())
        self.entry.connect("activate", lambda *_: self.search())
        bar.append(self.entry)
        self.task = Gtk.DropDown.new_from_strings([t for _, t in TASKS])
        self.task.set_tooltip_text("Task")
        self.task.update_property([Gtk.AccessibleProperty.LABEL], ["Task"])
        self.sort = Gtk.DropDown.new_from_strings([t for _, t in SORTS])
        self.sort.set_tooltip_text("Sort by")
        self.sort.update_property([Gtk.AccessibleProperty.LABEL], ["Sort by"])
        st = self.win.settings
        keys = [k for k, _ in TASKS]
        self.task.set_selected(keys.index(st.get("hub_task")) if st.get("hub_task") in keys else 0)
        skeys = [k for k, _ in SORTS]
        self.sort.set_selected(skeys.index(st.get("hub_sort")) if st.get("hub_sort") in skeys else 0)
        self.task.connect("notify::selected", lambda *_: self._filters_changed())
        self.sort.connect("notify::selected", lambda *_: self._filters_changed())
        bar.append(self.task)
        bar.append(self.sort)
        self.append(Adw.Clamp(maximum_size=980, child=bar))

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, margin_start=18, margin_end=18,
                       margin_top=6, margin_bottom=24)
        self.error_label = Gtk.Label(xalign=0, wrap=True, visible=False, css_classes=["error"], selectable=True)
        body.append(self.error_label)
        self.pulls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, visible=False)
        self.pulls_box.append(Gtk.Label(label="Downloads", xalign=0, css_classes=["heading"]))
        self.pull_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE, css_classes=["boxed-list"])
        self.pulls_box.append(self.pull_list)
        body.append(self.pulls_box)
        self.status = Gtk.Box(spacing=8, visible=False)
        self.status_spin = spinner()
        self.status_label = Gtk.Label(xalign=0, wrap=True, css_classes=["dim-label"])
        self.status.append(self.status_spin)
        self.status.append(self.status_label)
        body.append(self.status)
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE, css_classes=["boxed-list"], visible=False)
        self.list.update_property([Gtk.AccessibleProperty.LABEL], ["Search results"])
        body.append(self.list)
        body.append(Gtk.Label(label="Searches and downloads go to huggingface.co through the OpenNPU server.",
                              xalign=0, wrap=True, css_classes=["caption", "dim-label"]))
        scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroll.set_child(Adw.Clamp(maximum_size=980, child=body))
        self.append(scroll)
        self.rows = {}                  # id -> result row
        self.pull_rows = {}             # id -> {"row", "bar", "box", "key"}

    # ---- search
    def on_show(self):
        if not self.searched:
            self.search()
        self.refresh_pulls()

    def _filters_changed(self):
        self.win.settings["hub_task"] = TASKS[self.task.get_selected()][0]
        self.win.settings["hub_sort"] = SORTS[self.sort.get_selected()][0]
        self.win.save_settings()
        self.search()

    def search(self):
        self._seq += 1
        seq = self._seq
        q = self.entry.get_text().strip()
        task, sort = TASKS[self.task.get_selected()][0], SORTS[self.sort.get_selected()][0]
        self.status.set_visible(True)
        self.status_spin.set_visible(True)
        self.status_label.set_text("Searching Hugging Face")

        def ok(cards):
            if seq != self._seq:
                return
            self.cards, self.error, self.searched = cards, "", True
            self._render()

        def fail(e):
            if seq != self._seq:
                return
            self.cards, self.error, self.searched = [], "Search failed: " + hint(e), True
            self._render()
        run_bg(lambda: api.hub_search(q, task, sort), ok, fail)

    def card(self, rid):
        return next((c for c in self.cards if c["id"] == rid), None)

    def _render(self):
        self.error_label.set_visible(bool(self.error))
        self.error_label.set_text(self.error)
        while (r := self.list.get_row_at_index(0)) is not None:
            self.list.remove(r)
        self.rows = {}
        n = len(self.cards)
        self.status_spin.set_visible(False)
        self.status.set_visible(not self.error)
        self.status_label.set_text(f"{n} model{'s' if n != 1 else ''} found" if n else "No models match")
        for c in self.cards:
            row = Adw.ActionRow(title=esc(c["id"]), subtitle_lines=0, title_lines=0)
            row.card_id = c["id"]
            self.rows[c["id"]] = row
            self.list.append(row)
            self._paint_row(c)
        self.list.set_visible(bool(self.cards))
        self._render_pulls()

    def _subtitle(self, c):
        p = c.get("plan") or {}
        facts = [TASK_NAME.get(c.get("task"), c.get("task") or "Unknown task")]
        if c.get("params"):
            facts.append(fmt_params(c["params"]))
        facts += [f"{fmt_count(c.get('downloads'))} downloads", f"{fmt_count(c.get('likes'))} likes"]
        if c.get("updated"):
            facts.append(c["updated"])
        lines = ["  ·  ".join(facts), p.get("label") or p.get("engine") or "Unknown engine"]
        if p.get("fits") is False and p.get("note"):
            lines.append(f"Warning: {p['note']}")
        return esc("\n".join(lines))

    def _paint_row(self, c):
        """Rebuild one result row's subtitle and suffix widgets for its current state."""
        old = self.rows.get(c["id"])
        if old is None:
            return
        idx = old.get_index()
        row = Adw.ActionRow(title=esc(c["id"]), subtitle=self._subtitle(c), subtitle_lines=0, title_lines=0)
        row.card_id = c["id"]
        p = c.get("plan") or {}
        badges = Gtk.Box(spacing=4, valign=Gtk.Align.CENTER)
        if p.get("fast"):
            badges.append(badge("Fastest", "fast", p.get("label")))
        if c.get("gated"):
            badges.append(badge("Gated", "gated", "Accept the license on huggingface.co and run `hf auth login`"))
        if c.get("installed"):
            badges.append(badge("Installed", "ok"))
        if p.get("fits") is False:
            w = Gtk.Image(icon_name="dialog-warning-symbolic", tooltip_text=p.get("note") or "Needs more RAM")
            w.update_property([Gtk.AccessibleProperty.LABEL], [p.get("note") or "Needs more RAM"])
            badges.append(w)
        row.add_suffix(badges)
        row.add_suffix(self._actions(c))
        row.badges = badges
        self.list.remove(old)
        self.list.insert(row, idx)
        self.rows[c["id"]] = row

    def _actions(self, c):
        rid = c["id"]
        box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        job = self.pulls.get(rid)
        if job and job["state"] == "downloading":
            pct = int(100 * job["bytes"] / job["total"]) if job.get("total") else 0
            box.append(Gtk.Label(label=f"Downloading {pct}%", css_classes=["dim-label"]))
            return box
        if job and job["state"] == "failed":
            box.append(Gtk.Label(label="Download failed", css_classes=["error"]))
        if c.get("installed") or (job and job["state"] == "done"):
            return self._installed_actions(c, job, box)
        b = Gtk.Button(label="Download", css_classes=["pill"])
        b.update_property([Gtk.AccessibleProperty.LABEL], [f"Download {rid}"])
        b.connect("clicked", lambda *_: self.open_details(rid))
        box.append(b)
        return box

    def _installed_actions(self, c, job, box):
        rid = c["id"]
        if self.confirm == rid:
            box.append(Gtk.Label(label="Remove its files?"))
            yes = Gtk.Button(label="Remove", css_classes=["pill", "destructive-action"])
            yes.update_property([Gtk.AccessibleProperty.LABEL], [f"Confirm removing {rid}"])
            yes.connect("clicked", lambda *_: self.remove(rid))
            no = Gtk.Button(label="Keep", css_classes=["pill"])
            no.connect("clicked", lambda *_: self._set_confirm(None, rid))
            box.append(yes)
            box.append(no)
            GLib.idle_add(no.grab_focus)
            return box
        main = self._use_button(c, job, "Chat")
        if main:
            box.append(main)
        rm = Gtk.Button(label="Remove", css_classes=["pill"])
        rm.update_property([Gtk.AccessibleProperty.LABEL], [f"Remove {rid}"])
        rm.connect("clicked", lambda *_: self._set_confirm(rid, rid))
        box.append(rm)
        return box

    def _use_button(self, c, job, label):
        """Chat / Chat now / Open in Create, or None when the model is not something you open."""
        eng = (c.get("plan") or {}).get("engine") or (job or {}).get("engine")
        if eng == "gen" and CREATE_TASK.get(c.get("task")):
            b = Gtk.Button(label="Open in Create", css_classes=["pill", "suggested-action"])
            b.connect("clicked", lambda *_: self.win.create_with({"id": c["id"], "task": CREATE_TASK[c["task"]]}))
            return b
        if eng in ("gen", "embed"):
            return None
        b = Gtk.Button(label=label, css_classes=["pill", "suggested-action"])
        b.update_property([Gtk.AccessibleProperty.LABEL], [f"{label} with {c['id']}"])
        b.connect("clicked", lambda *_: self.chat_now(self.chat_id(c, job)))
        return b

    def _set_confirm(self, rid, row_id):
        self.confirm = rid
        c = self.card(row_id)
        if c:
            self._paint_row(c)
        self._render_pulls(force=True)

    # ---- which /v1/models id to chat with
    def chat_id(self, c, job=None):
        if job and job.get("model"):
            return job["model"]
        repo, name = c["id"], c["id"].split("/")[-1]
        models = self.win.chat.models
        hit = next((m["id"] for m in models if m["id"] in ("npu4:" + repo, "universal:" + repo)), None) or \
            next((m["id"] for m in models if m["engine"] in ("native", "moe") and m["id"].split(":", 1)[1] == name), None)
        if hit:
            return hit
        eng = (c.get("plan") or {}).get("engine")
        return f"{eng}:{name}" if eng in ("native", "moe") else ("npu4:" if eng == "npu4" else "universal:") + repo

    def chat_now(self, mid, tries=10):
        """A finished download reaches /v1/models within a few seconds: wait for it, then chat."""
        if mid in [m["id"] for m in self.win.chat.models]:
            self.win.chat_with(mid)
            return
        if tries <= 0:
            self.win.toast("Downloaded. It appears in the model list in a few seconds")
            return
        self.win.refresh_models()
        GLib.timeout_add(1000, lambda: (self.chat_now(mid, tries - 1), False)[1])

    # ---- details + download
    def open_details(self, rid):
        c = self.card(rid) or {"id": rid}
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_start=18, margin_end=18,
                       margin_top=12, margin_bottom=18)
        loading = Gtk.Box(spacing=8)
        loading.append(spinner())
        loading.append(Gtk.Label(label="Reading the model details", css_classes=["dim-label"]))
        page.append(loading)
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True, child=page)
        d = {"id": rid, "info": None, "page": page, "download": None, "error": None, "gguf": None}
        d["win"] = make_dialog(self.win, rid, scroll)
        self.dialog = d
        d["win"].connect("closed" if hasattr(Adw, "Dialog") else "close-request",
                         lambda *_: (setattr(self, "dialog", None) if self.dialog is d else None, False)[1])

        def ok(info):
            if self.dialog is not d:
                return
            page.remove(loading)
            d["info"] = info
            self._fill_details(d, c, info)

        def fail(e):
            if self.dialog is not d:
                return
            loading.get_first_child().set_visible(False)
            loading.get_last_child().set_text("Could not read the model details: " + hint(e))
            loading.get_last_child().set_wrap(True)
            loading.get_last_child().set_css_classes(["error"])
        run_bg(lambda: api.hub_model(rid), ok, fail)

    def _fill_details(self, d, c, info):
        page, p = d["page"], info.get("plan") or {}
        files = info.get("download", {}).get("files") or []
        ggufs = [g for g in info.get("gguf") or [] if "mmproj" not in g["name"].lower()]
        default = next((f for f in files if f.endswith(".gguf")), None)
        g = Adw.PreferencesGroup(title=esc(info["id"]), description=esc(p.get("label") or p.get("engine") or ""))
        d["size_row"] = Adw.ActionRow(title="Download", subtitle_selectable=True)
        rows = [d["size_row"],
                Adw.ActionRow(title="Free disk", subtitle=esc(fmt_bytes(info.get("free_disk")))),
                Adw.ActionRow(title="Parameters", subtitle=esc(fmt_params(info.get("params") or c.get("params")) or "unknown")),
                Adw.ActionRow(title="RAM needed", subtitle=esc(f"about {p['ram_gb']} GB" if p.get("ram_gb") else "unknown")),
                Adw.ActionRow(title="License", subtitle=esc(info.get("license") or "not stated"))]
        if p.get("note"):
            rows.insert(1, Adw.ActionRow(title="Note", subtitle=esc(p["note"]), subtitle_lines=0))
        for r in rows:
            g.add(r)
        if ggufs:
            names = Gtk.StringList()
            for x in ggufs:
                names.append(f"{x['name']}  ({fmt_bytes(x['size'])})")
            combo = Adw.ComboRow(title="GGUF variant", model=names)
            combo.set_selected(next((i for i, x in enumerate(ggufs) if x["name"] == default), 0))
            combo.connect("notify::selected", lambda *_: self._update_size(d))
            d["gguf"], d["ggufs"] = combo, ggufs
            g.add(combo)
        page.append(g)
        d["default_gguf"] = default
        if info.get("gated"):
            page.append(Gtk.Label(label="Gated: accept the license on huggingface.co and run `hf auth login` before downloading.",
                                  xalign=0, wrap=True, css_classes=["warning"]))
        d["error"] = Gtk.Label(xalign=0, wrap=True, visible=False, css_classes=["error"], selectable=True)
        page.append(d["error"])
        btns = Gtk.Box(spacing=8, halign=Gtk.Align.END)
        close = Gtk.Button(label="Close", css_classes=["pill"])
        close.connect("clicked", lambda *_: d["win"].close())
        d["download"] = Gtk.Button(css_classes=["pill", "suggested-action"])
        d["download"].connect("clicked", lambda *_: self.start_pull(d))
        btns.append(close)
        btns.append(d["download"])
        page.append(btns)
        self._update_size(d)
        d["download"].grab_focus()

    def _chosen_gguf(self, d):
        if not d.get("gguf"):
            return None
        return d["ggufs"][d["gguf"].get_selected()]["name"]

    def _update_size(self, d):
        info = d["info"]
        size = {x["name"]: x["size"] for x in info.get("gguf") or []}
        b = info["download"]["bytes"]
        pick = self._chosen_gguf(d)
        if pick and d.get("default_gguf"):
            b = b - size.get(d["default_gguf"], 0) + size.get(pick, 0)
        d["bytes"] = b
        n = len(info["download"].get("files") or [])
        d["size_row"].set_subtitle(esc(f"{fmt_bytes(b)} ({n} files)"))
        d["download"].set_label(f"Download {fmt_bytes(b)}")

    def start_pull(self, d=None, rid=None):
        rid = d["id"] if d else rid
        gguf = self._chosen_gguf(d) if d else None
        if d and d.get("download"):
            d["download"].set_sensitive(False)

        def ok(job):
            self.pulls[rid] = job
            self.was_downloading.add(rid)
            self.confirm = None
            if d and self.dialog is d:
                d["win"].close()
                self.dialog = None
            c = self.card(rid)
            if c:
                self._paint_row(c)
            self._render_pulls()
            self._ensure_polling()

        def fail(e):
            if d and self.dialog is d:
                d["download"].set_sensitive(True)
                d["error"].set_text(hint(e))
                d["error"].set_visible(True)
            else:
                self.win.toast("Could not start the download: " + hint(e))
        run_bg(lambda: api.hub_pull(rid, gguf), ok, fail)

    # ---- downloads (progress rows)
    def refresh_pulls(self):
        run_bg(api.hub_pulls, self._got_pulls, lambda _e: None)

    def _ensure_polling(self):
        if not self._poll_id:
            self._poll_id = GLib.timeout_add(1000, self._tick)

    def _tick(self):
        if not any(j["state"] == "downloading" for j in self.pulls.values()):
            self._poll_id = 0
            return False
        self.refresh_pulls()
        return True

    def _got_pulls(self, d):
        self.pulls = {j["id"]: j for j in d.get("data") or []}
        self.free_disk = d.get("free_disk")
        finished = False
        for rid, j in self.pulls.items():
            if j["state"] == "downloading":
                self.was_downloading.add(rid)
                continue
            if rid in self.was_downloading:
                self.was_downloading.discard(rid)
                c = self.card(rid)
                if j["state"] == "done":
                    finished = True
                    if c:
                        c["installed"] = True
                if c:
                    self._paint_row(c)
        if finished:
            self.win.refresh_models()
        for c in self.cards:
            j = self.pulls.get(c["id"])
            if j and j["state"] == "downloading":
                self._paint_row(c)
        self._render_pulls()
        if any(j["state"] == "downloading" for j in self.pulls.values()):
            self._ensure_polling()

    def _render_pulls(self, force=False):
        jobs = [j for j in self.pulls.values() if j["state"] != "cancelled"]
        for rid in list(self.pull_rows):
            if rid not in {j["id"] for j in jobs}:
                self.pull_list.remove(self.pull_rows.pop(rid)["row"])
        for j in jobs:
            pr = self.pull_rows.get(j["id"])
            key = (j["state"], self.confirm == j["id"])
            if pr is None or pr["key"] != key or force:
                if pr is not None:
                    self.pull_list.remove(pr["row"])
                pr = self._pull_row(j)
                pr["key"] = key
                self.pull_rows[j["id"]] = pr
                self.pull_list.append(pr["row"])
            if j["state"] == "downloading":
                frac = j["bytes"] / j["total"] if j.get("total") else 0.0
                pr["bar"].set_fraction(min(1.0, frac))
                rate = f"  ·  {fmt_bytes(j['rate'])}/s" if j.get("rate") else ""
                pr["row"].set_subtitle(esc(f"{fmt_bytes(j['bytes'])} of {fmt_bytes(j['total'])}{rate}  ·  {fmt_eta(j.get('eta_s'))}"))
        self.pulls_box.set_visible(bool(jobs))

    def _pull_row(self, j):
        rid = j["id"]
        row = Adw.ActionRow(title=esc(rid), subtitle_lines=0)
        pr = {"row": row, "bar": None}
        box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        if j["state"] == "downloading":
            bar = Gtk.ProgressBar(valign=Gtk.Align.CENTER, width_request=160)
            bar.update_property([Gtk.AccessibleProperty.LABEL], [f"Download progress for {rid}"])
            pr["bar"] = bar
            box.append(bar)
            b = Gtk.Button(label="Cancel", css_classes=["pill"])
            b.update_property([Gtk.AccessibleProperty.LABEL], [f"Cancel download of {rid}"])
            b.connect("clicked", lambda *_: self.cancel(rid))
            box.append(b)
        elif j["state"] == "done":
            row.set_subtitle("Downloaded")
            c = self.card(rid) or {"id": rid, "plan": {"engine": j.get("engine")}}
            if self.confirm == rid:
                self._installed_actions(c, j, box)
            else:
                use = self._use_button(c, j, "Chat now")
                if use:
                    box.append(use)
                pr["chat"] = use
                dis = Gtk.Button(label="Dismiss", css_classes=["pill", "flat"])
                dis.update_property([Gtk.AccessibleProperty.LABEL], [f"Dismiss {rid}"])
                dis.connect("clicked", lambda *_: self.cancel(rid, quiet=True))
                box.append(dis)
        else:                                                   # failed
            row.set_subtitle(esc("Download failed: " + (j.get("error") or "unknown error")))
            row.add_css_class("error")
            retry = Gtk.Button(label="Retry", css_classes=["pill"])
            retry.update_property([Gtk.AccessibleProperty.LABEL], [f"Retry download of {rid}"])
            retry.connect("clicked", lambda *_: self.start_pull(rid=rid))
            box.append(retry)
            dis = Gtk.Button(label="Dismiss", css_classes=["pill", "flat"])
            dis.connect("clicked", lambda *_: self.cancel(rid, quiet=True))
            box.append(dis)
        row.add_suffix(box)
        pr["box"] = box
        return pr

    def cancel(self, rid, quiet=False):
        def ok(_r):
            self.was_downloading.discard(rid)
            if not quiet:
                self.win.toast(f"Download of {rid} cancelled")
            self.refresh_pulls()
            c = self.card(rid)
            if c:
                GLib.timeout_add(300, lambda: (self._paint_row(c), False)[1])
        run_bg(lambda: api.hub_cancel(rid), ok, lambda e: self.win.toast(f"Could not cancel: {hint(e)}"))

    def remove(self, rid):
        job = self.pulls.get(rid)

        def work():
            r = api.hub_remove(rid)
            if job and job["state"] != "downloading":            # a finished job would still offer Chat now
                try:
                    api.hub_cancel(rid)
                except api.ApiError:
                    pass
            return r

        def ok(_r):
            self.confirm = None
            self.pulls.pop(rid, None)
            c = self.card(rid)
            if c:
                c["installed"] = False
                self._paint_row(c)
            self._render_pulls(force=True)
            self.win.refresh_models()
            self.win.toast(f"Removed {rid}")

        def fail(e):
            self.confirm = None
            c = self.card(rid)
            if c:
                self._paint_row(c)
            self.win.toast(f"Could not remove: {hint(e)}")
        run_bg(work, ok, fail)

