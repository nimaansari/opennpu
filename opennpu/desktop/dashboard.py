"""Desktop Dashboard: local usage analytics + hardware history, from the server's /api/usage and
/api/hardware. Everything lives in a SQLite file on this computer; nothing is sent anywhere.
Charts are drawn with cairo: one measure per chart, hairline grid, thin marks, hover tooltips."""
import threading
import time

from gi.repository import Adw, GLib, Gtk, Pango

from opennpu.desktop import api

# validated chart palette slot 1 (light / dark), same as the web UI
SERIES = {False: (0x2A / 255, 0x78 / 255, 0xD6 / 255), True: (0x39 / 255, 0x87 / 255, 0xE5 / 255)}
KIND = {"chat": "Chat", "embed": "Embeddings", "image": "Images", "video": "Video", "speech": "Voice"}
APP = {"web": "Web UI", "desktop": "Desktop app", "cli": "npu CLI", "openai sdk": "OpenAI client", "ollama client": "Ollama client"}
STATUS = {"ok": "Done", "error": "Failed", "cancelled": "Stopped"}


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


def gib(b):
    return "-" if b is None else f"{b / 2 ** 30:.1f} GB"


def dur(s):
    return "-" if s is None else f"{s:.1f} s" if s < 60 else f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"


def when(ts, day=False):
    return time.strftime("%b %d" if day else "%b %d, %H:%M", time.localtime(ts))


def compact(v):
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.1f}k" if v >= 1e3 else (str(int(v)) if float(v).is_integer() else f"{v:.1f}")


def nice_max(v):
    if not v or v <= 0:
        return 1
    import math
    p = 10 ** math.floor(math.log10(v))
    n = v / p
    return (1 if n <= 1 else 2 if n <= 2 else 2.5 if n <= 2.5 else 5 if n <= 5 else 10) * p


class Chart(Gtk.DrawingArea):
    """kind="bars": points [(label, value, tooltip)]; kind="line": points [(t, value, peak)]."""

    def __init__(self, kind, height=170, fmt=compact, unit="", vmin=0, vmax=None, xfmt=None):
        super().__init__(content_height=height, hexpand=True, has_tooltip=True)
        self.kind, self.fmt, self.unit, self.vmin, self.vmax, self.xfmt = kind, fmt, unit, vmin, vmax, xfmt
        self.points, self.hover = [], None
        self.set_draw_func(self._draw)
        self.connect("query-tooltip", self._tip)
        m = Gtk.EventControllerMotion()
        m.connect("motion", lambda _c, x, y: self._move(x))
        m.connect("leave", lambda *_: (setattr(self, "hover", None), self.queue_draw()))
        self.add_controller(m)
        self.update_property([Gtk.AccessibleProperty.LABEL], ["chart"])

    def set_points(self, pts, label=""):
        self.points = pts
        self.update_property([Gtk.AccessibleProperty.LABEL], [label])
        self.queue_draw()

    # geometry
    L, B, T, R = 44, 20, 8, 8

    def _x(self, i, w):
        n = max(1, len(self.points))
        if self.kind == "bars":
            return self.L + (w - self.L - self.R) * (i + 0.5) / n
        t0, t1 = self.points[0][0], self.points[-1][0]
        return self.L + (w - self.L - self.R) * ((self.points[i][0] - t0) / max(1, t1 - t0))

    def _range(self):
        vals = [p[1] for p in self.points if p[1] is not None]
        hi = self.vmax if self.vmax is not None else nice_max(max(vals) * (1.05 if self.kind == "line" else 1) if vals else 1)
        return self.vmin, hi

    def _draw(self, _a, cr, w, h):
        fg = self.get_color()
        dark = Adw.StyleManager.get_default().get_dark()
        sr, sg, sb = SERIES[dark]
        vals = [p[1] for p in self.points if p[1] is not None]
        cr.select_font_face("Sans")
        cr.set_font_size(10)
        try:                                     # grayscale text AA: no colored fringes on dark surfaces
            import cairo
            fo = cairo.FontOptions()
            fo.set_antialias(cairo.ANTIALIAS_GRAY)
            cr.set_font_options(fo)
        except (ImportError, AttributeError):
            pass
        if (self.kind == "bars" and not any(vals)) or (self.kind == "line" and len(vals) < 2):
            cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.6)
            msg = "Nothing in this range yet." if self.kind == "bars" else "Not enough history yet (one point per minute)."
            ext = cr.text_extents(msg)
            cr.move_to((w - ext.width) / 2, h / 2)
            cr.show_text(msg)
            return
        lo, hi = self._range()
        Y = lambda v: self.T + (h - self.T - self.B) * (1 - (v - lo) / ((hi - lo) or 1))  # noqa: E731
        # grid + y labels
        for f in (0, 0.5, 1):
            v = lo + (hi - lo) * f
            y = round(Y(v)) + 0.5
            cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.14)
            cr.set_line_width(1)
            cr.move_to(self.L, y)
            cr.line_to(w - self.R, y)
            cr.stroke()
            cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.65)
            t = self.fmt(v) + self.unit
            cr.move_to(self.L - 6 - cr.text_extents(t).width, y + 3.5)
            cr.show_text(t)
        n = len(self.points)
        if self.kind == "bars":
            band = (w - self.L - self.R) / n
            bw = max(2, min(24, band - 2))
            every = max(1, round(n / 8))
            for i, (lab, v, _tip) in enumerate(self.points):
                cx = self._x(i, w)
                if v:
                    top, base, r = Y(v), Y(lo), min(4, bw / 2, Y(lo) - Y(v))
                    cr.set_source_rgba(sr, sg, sb, 0.75 if self.hover is not None and self.hover != i else 1)
                    x0 = cx - bw / 2
                    cr.new_path()
                    cr.move_to(x0, base)
                    cr.line_to(x0, top + r)
                    cr.arc(x0 + r, top + r, r, 3.14159, 4.71239)
                    cr.line_to(x0 + bw - r, top)
                    cr.arc(x0 + bw - r, top + r, r, 4.71239, 0)
                    cr.line_to(x0 + bw, base)
                    cr.close_path()
                    cr.fill()
                if i % every == 0:
                    cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.65)
                    cr.move_to(cx - cr.text_extents(lab).width / 2, h - 5)
                    cr.show_text(lab)
            return
        # line: area wash + 2px line, broken at gaps in the history
        pts = [(i, p) for i, p in enumerate(self.points) if p[1] is not None]
        steps = [self.points[b][0] - self.points[a][0] for (a, _), (b, _) in zip(pts, pts[1:])]
        gap = min(s for s in steps if s > 0) * 3 if any(s > 0 for s in steps) else 1e18
        segs, cur = [], [pts[0]]
        for (a, pa), (b, pb) in zip(pts, pts[1:]):
            if pb[0] - pa[0] > gap:
                segs.append(cur)
                cur = []
            cur.append((b, pb))
        segs.append(cur)
        for seg in segs:
            cr.set_source_rgba(sr, sg, sb, 0.1)
            cr.move_to(self._x(seg[0][0], w), Y(lo))
            for i, p in seg:
                cr.line_to(self._x(i, w), Y(p[1]))
            cr.line_to(self._x(seg[-1][0], w), Y(lo))
            cr.close_path()
            cr.fill()
            cr.set_source_rgb(sr, sg, sb)
            cr.set_line_width(2)
            cr.set_line_join(1)
            for k, (i, p) in enumerate(seg):
                (cr.line_to if k else cr.move_to)(self._x(i, w), Y(p[1]))
            cr.stroke()
        cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.65)
        for f, align in ((0, 0), (0.5, 0.5), (1, 1)):
            t = self.points[0][0] + (self.points[-1][0] - self.points[0][0]) * f
            s = self.xfmt(t) if self.xfmt else ""
            x = self.L + (w - self.L - self.R) * f - cr.text_extents(s).width * align
            cr.move_to(x, h - 5)
            cr.show_text(s)
        if self.hover is not None and self.points[self.hover][1] is not None:
            x, y = self._x(self.hover, w), Y(self.points[self.hover][1])
            cr.set_source_rgba(fg.red, fg.green, fg.blue, 0.4)
            cr.set_line_width(1)
            cr.move_to(round(x) + 0.5, self.T)
            cr.line_to(round(x) + 0.5, h - self.B)
            cr.stroke()
            cr.set_source_rgb(sr, sg, sb)
            cr.arc(x, y, 4, 0, 6.2832)
            cr.fill()

    def _index(self, x):
        if not self.points:
            return None
        w = self.get_width()
        return min(range(len(self.points)), key=lambda i: abs(self._x(i, w) - x))

    def _move(self, x):
        i = self._index(x)
        if i != self.hover:
            self.hover = i
            self.queue_draw()

    def _tip(self, _w, x, _y, _kbd, tooltip):
        i = self._index(x)
        if i is None:
            return False
        p = self.points[i]
        if self.kind == "bars":
            tooltip.set_text(p[2] or f"{p[0]}: {self.fmt(p[1])}")
        else:
            if p[1] is None:
                return False
            peak = f" (peak {self.fmt(p[2])}{self.unit})" if len(p) > 2 and p[2] is not None else ""
            tooltip.set_text(f"{self.fmt(p[1])}{self.unit}{peak}\n{when(p[0])}")
        return True


def card(title, child, note=None):
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, css_classes=["card", "dash-card"])
    head = Gtk.Box(spacing=8)
    head.append(Gtk.Label(label=title, xalign=0, css_classes=["heading"]))
    if note:
        head.append(Gtk.Label(label=note, xalign=0, hexpand=True, css_classes=["caption", "dim-label"]))
    box.append(head)
    box.append(child)
    box.head = head
    return box


def tiles(items):
    fb = Gtk.FlowBox(min_children_per_line=2, max_children_per_line=6, selection_mode=Gtk.SelectionMode.NONE,
                     homogeneous=True, column_spacing=10, row_spacing=10)
    for label, value, sub in items:
        b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, css_classes=["card", "dash-tile"])
        b.append(Gtk.Label(label=label, xalign=0, css_classes=["caption", "dim-label"]))
        b.append(Gtk.Label(label=value, xalign=0, css_classes=["title-2", "numeric"], ellipsize=Pango.EllipsizeMode.END))
        if sub:
            b.append(Gtk.Label(label=sub, xalign=0, css_classes=["caption", "dim-label"], wrap=True))
        fb.append(b)
    return fb


def rows(group_title, items, description=None):
    g = Adw.PreferencesGroup(title=group_title, description=description)
    for title, sub, value in items:
        r = Adw.ActionRow(title=GLib.markup_escape_text(str(title)), subtitle=GLib.markup_escape_text(str(sub or "")))
        r.add_suffix(Gtk.Label(label=str(value), css_classes=["numeric", "dim-label"]))
        g.add(r)
    if not items:
        g.add(Adw.ActionRow(title="Nothing in this range yet"))
    return g


def segmented(options, active, on_change):
    box = Gtk.Box(css_classes=["linked"])
    first = None
    for key, label in options:
        b = Gtk.ToggleButton(label=label, active=key == active)
        if first:
            b.set_group(first)
        first = first or b
        b.connect("toggled", lambda btn, k=key: btn.get_active() and on_change(k))
        box.append(b)
    return box


class DashboardPage(Gtk.ScrolledWindow):
    RANGES = {"usage": [("24h", "24 h"), ("7d", "7 days"), ("30d", "30 days"), ("all", "All")],
              "hardware": [("1h", "1 h"), ("24h", "24 h"), ("7d", "7 days"), ("30d", "30 days")]}

    def __init__(self, win):
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.win = win
        s = win.settings
        self.tab = s.get("dash_tab", "usage") if s.get("dash_tab") in ("usage", "hardware") else "usage"
        self.range = {"usage": s.get("dash_ur", "7d"), "hardware": s.get("dash_hr", "24h")}
        self.metric = "tokens"
        self._busy = False
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16, margin_top=16, margin_bottom=24,
                        margin_start=16, margin_end=16)
        self.set_child(Adw.Clamp(maximum_size=1200, child=outer))
        outer.append(Gtk.Label(label="Your usage and your hardware. Kept only on this computer; prompts and replies are never recorded.",
                               xalign=0, wrap=True, css_classes=["dim-label"]))
        bar = Gtk.Box(spacing=10)
        bar.append(segmented([("usage", "Usage"), ("hardware", "Hardware")], self.tab, self._set_tab))
        self.range_box = Gtk.Box()
        bar.append(self.range_box)
        bar.append(Gtk.Box(hexpand=True))
        self.record = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="Record usage on this computer")
        self.record.update_property([Gtk.AccessibleProperty.LABEL], ["Record usage on this computer"])
        self._record_handler = self.record.connect("notify::active", self._toggle_record)
        bar.append(Gtk.Label(label="Recording"))
        bar.append(self.record)
        self.export_btn = Gtk.Button(label="Export CSV", css_classes=["pill"])
        self.export_btn.connect("clicked", self._export)
        self.clear_btn = Gtk.Button(label="Clear history", css_classes=["pill"])
        self.clear_btn.connect("clicked", self._confirm_clear)
        bar.append(self.export_btn)
        bar.append(self.clear_btn)
        outer.append(bar)
        self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        outer.append(self.body)
        self.where = Gtk.Label(xalign=0, wrap=True, selectable=True, css_classes=["caption", "dim-label"])
        outer.append(self.where)
        GLib.timeout_add_seconds(15, self._tick)

    # ---- state
    def _save(self):
        self.win.settings.update(dash_tab=self.tab, dash_ur=self.range["usage"], dash_hr=self.range["hardware"])
        self.win.save_settings()

    def _set_tab(self, t):
        self.tab = t
        self._save()
        self.refresh()

    def _set_range(self, r):
        self.range[self.tab] = r
        self._save()
        self.refresh()

    def _tick(self):
        if self.win.stack.get_visible_child() is self and self.win.is_visible():
            self.refresh()
        return True

    def refresh(self):
        if self._busy:
            return
        self._busy = True
        while (c := self.range_box.get_first_child()) is not None:
            self.range_box.remove(c)
        self.range_box.append(segmented(self.RANGES[self.tab], self.range[self.tab], self._set_range))
        self.export_btn.set_visible(self.tab == "usage")
        self.clear_btn.set_visible(self.tab == "usage")
        fn = (lambda: api.usage(self.range["usage"])) if self.tab == "usage" else (lambda: api.hardware(self.range["hardware"]))
        run_bg(fn, self._show, self._fail)

    def _clear_body(self):
        while (c := self.body.get_first_child()) is not None:
            self.body.remove(c)

    def _fail(self, e):
        self._busy = False
        self._clear_body()
        self.body.append(Adw.StatusPage(icon_name="opennpu-chart-bar-symbolic", title="Could not load the dashboard",
                                        description=GLib.markup_escape_text(str(e))))

    def _set_record(self, on):
        self.record.handler_block(self._record_handler)
        self.record.set_active(on)
        self.record.handler_unblock(self._record_handler)

    def _show(self, d):
        self._busy = False
        self._clear_body()
        self._set_record(d.get("enabled", True))
        try:
            (self._usage if self.tab == "usage" else self._hardware)(d)
        except Exception as e:  # noqa: BLE001 - a surprising payload must not break the page
            self._fail(e)

    # ---- usage
    def _usage(self, u):
        t = u["totals"]
        self.where.set_text(f"Stored only on this computer: {u.get('stored_at', '')}")
        if not t["requests"]:
            self.body.append(Adw.StatusPage(icon_name="opennpu-chart-bar-symbolic", title="No usage in this range yet",
                                            description="Chat, embeddings, images, video and voice from any app show up here."
                                            if u.get("enabled") else "Recording is off. Turn it on above."))
            return
        self.body.append(tiles([
            ("Requests", f"{t['requests']:,}", f"{t['errors']} failed, {t['cancelled']} stopped" if t["errors"] or t["cancelled"] else "all completed"),
            ("Tokens generated", f"{t['completion_tokens']:,}", f"{t['prompt_tokens']:,} tokens read"),
            ("Average speed", f"{t['avg_tok_s']:.1f} tok/s" if t.get("avg_tok_s") else "-", "decode, weighted by tokens"),
            ("First token", f"{t['avg_ttft_s']} s" if t.get("avg_ttft_s") is not None else "-", "average for chat"),
            ("NPU working time", dur(t["busy_s"]), "time spent generating"),
            ("Energy", f"{t['energy_wh']:.2f} Wh" if t.get("energy_wh") else "-", "measured while on battery"),
        ]))
        daily = u["bucket_s"] >= 86400
        chart = Chart("bars", height=190)
        pts = [(when(p["t"], True) if daily else time.strftime("%H:%M", time.localtime(p["t"])), p[self.metric],
                f"{when(p['t'], daily)}\n{p['tokens']:,} tokens, {p['requests']} requests") for p in u["timeline"]]
        chart.set_points(pts, f"{self.metric} over time")
        c = card("Over time", chart)
        c.head.append(Gtk.Box(hexpand=True))
        c.head.append(segmented([("tokens", "Tokens"), ("requests", "Requests")], self.metric,
                                lambda m: (setattr(self, "metric", m), self.refresh())))
        self.body.append(c)
        hours = Chart("bars", height=150)
        hours.set_points([(f"{h:02d}", v, f"{h:02d}:00\n{v} requests") for h, v in enumerate(u["hour_of_day"])],
                         "Requests by hour of day")
        self.body.append(card("When you use it", hours, "requests by hour of day"))
        self.body.append(rows("By model", [(m["model"], f"{m['requests']} requests"
                                            + (f", {m['avg_tok_s']} tok/s" if m.get("avg_tok_s") else "")
                                            + (f", last used {when(m['last_ts'])}" if m.get("last_ts") else ""),
                                            f"{m['completion_tokens']:,} tokens") for m in u["by_model"]]))
        two = Gtk.Box(spacing=16, homogeneous=True)
        two.append(rows("By kind", [(KIND.get(k, k), None, v) for k, v in u["by_kind"].items()]))
        two.append(rows("By app", [(APP.get(k, k), None, v) for k, v in u["by_client"].items()]))
        self.body.append(two)
        self.body.append(rows("Recent requests", [
            (f"{r['model']}", f"{when(r['ts'])}  {KIND.get(r['kind'], r['kind'])}  {APP.get(r['client'], r['client'])}  "
                              f"{STATUS.get(r['status'], r['status'])}"
                              + (f"  {r['decode_tok_s']} tok/s" if r.get("decode_tok_s") else ""),
             f"{r['completion_tokens']} tokens" if r.get("completion_tokens") is not None else dur(r.get("total_s")))
            for r in u["recent"][:20]], "latest 20 in this range"))

    # ---- hardware
    def _hardware(self, h):
        i, n = h.get("info") or {}, h.get("now") or {}
        npu = i.get("npu") or {}
        disk = (i.get("disks") or {}).get("models") or (i.get("disks") or {}).get("home")
        self.where.set_text("Hardware history: one point per minute, kept 30 days, stored only on this computer.")
        self.body.append(tiles([
            ("NPU busy", f"{n['npu_busy']:.0f}%" if n.get("npu_busy") is not None else "-", "last 5 s"),
            ("CPU", f"{n['cpu']:.0f}%" if n.get("cpu") is not None else "-", f"{i.get('cores')} threads"),
            ("Memory", gib(n.get("ram_used")), f"of {gib(i.get('ram_total'))}"),
            ("CPU temperature", f"{n['cpu_temp']:.0f} °C" if n.get("cpu_temp") is not None else "-",
             f"GPU {n['gpu_temp']:.0f} °C" if n.get("gpu_temp") is not None else ""),
            ("Power", f"{n['power_w']:.1f} W" if n.get("power_w") is not None else "plugged in", "whole laptop, on battery"),
            ("Model disk", f"{gib(disk['free'])} free" if disk else "-", f"of {gib(disk['total'])}" if disk else ""),
        ]))
        s = h.get("series") or []
        long = self.range["hardware"] in ("7d", "30d")
        xfmt = (lambda t: when(t, True)) if long else (lambda t: time.strftime("%H:%M", time.localtime(t)))
        grid = Gtk.FlowBox(min_children_per_line=1, max_children_per_line=2, selection_mode=Gtk.SelectionMode.NONE,
                           homogeneous=True, column_spacing=12, row_spacing=12)
        ram_max = i.get("ram_total") and round(i["ram_total"] / 2 ** 30)
        for title, key, unit, kw in (("NPU busy", "npu_busy", "%", dict(vmax=100, peak="npu_busy_max")),
                                     ("CPU", "cpu", "%", dict(vmax=100, peak="cpu_max")),
                                     ("Memory used", "ram_used", " GB", dict(scale=1 / 2 ** 30, vmax=ram_max)),
                                     ("CPU temperature", "cpu_temp", " °C", dict(vmin=20, peak="cpu_temp_max")),
                                     ("Power", "power_w", " W", {}),
                                     ("Server memory", "server_rss", " GB", dict(scale=1 / 2 ** 30, fmt=lambda v: f"{v:.1f}"))):
            sc, pk = kw.pop("scale", 1), kw.pop("peak", None)
            ch = Chart("line", height=150, unit=unit, xfmt=xfmt, fmt=kw.pop("fmt", lambda v: f"{v:.0f}"), **kw)
            ch.set_points([(p["t"], p[key] * sc if p.get(key) is not None else None,
                            p[pk] * sc if pk and p.get(pk) is not None else None) for p in s], title)
            grid.append(card(title, ch))
        self.body.append(grid)
        info = [("Processor", i.get("cpu")), ("NPU", f"{npu.get('device')} ({npu.get('arch', '?')})" if npu.get("device") else "not detected"),
                ("NPU firmware", npu.get("firmware")), ("XRT", npu.get("xrt")), ("Graphics", i.get("gpu")),
                ("Memory", gib(i.get("ram_total"))), ("System", i.get("os")), ("Kernel", i.get("kernel"))]
        b = i.get("battery")
        if b:
            info.append(("Battery", f"{b.get('capacity')}%, {(b.get('status') or '').lower()}"
                                    + (f", health {b['health']}%" if b.get("health") else "")))
        for k, dsk in (i.get("disks") or {}).items():
            info.append((f"Disk ({k})", f"{gib(dsk['free'])} free of {gib(dsk['total'])}  {dsk['path']}"))
        self.body.append(rows("This machine", [(k, v, "") for k, v in info if v]))

    # ---- actions
    def _toggle_record(self, sw, _p):
        on = sw.get_active()
        run_bg(lambda: api.usage_settings(on),
               lambda r: self.win.toast("Recording usage on this computer" if r.get("enabled") else "Recording paused"),
               lambda e: (self._set_record(not on), self.win.toast(f"Could not change recording: {e}")))

    def _export(self, *_):
        dlg = Gtk.FileDialog(initial_name="opennpu-usage.csv")

        def picked(d, res):
            try:
                dest = d.save_finish(res)
            except GLib.Error:
                return

            def write(data):
                try:
                    dest.replace_contents(data, None, False, 0, None)
                    self.win.toast(f"Saved {dest.get_basename()}")
                except GLib.Error as e:
                    self.win.toast(f"Could not save: {e.message}")
            run_bg(api.usage_csv, write, lambda e: self.win.toast(f"Could not export: {e}"))
        dlg.save(self.win, None, picked)

    def _confirm_clear(self, *_):
        heading, body = "Delete usage history?", "All recorded usage is deleted from this computer. Hardware history is kept."

        def do(resp):
            if resp == "delete":
                run_bg(lambda: api.usage_clear("usage"), lambda _r: (self.win.toast("Usage history deleted"), self.refresh()),
                       lambda e: self.win.toast(f"Could not delete: {e}"))
        if hasattr(Adw, "AlertDialog"):                   # libadwaita >= 1.5
            d = Adw.AlertDialog(heading=heading, body=body)
            d.add_response("cancel", "Cancel")
            d.add_response("delete", "Delete")
            d.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
            d.connect("response", lambda _d, r: do(r))
            d.present(self.win)
        else:
            d = Adw.MessageDialog(transient_for=self.win, heading=heading, body=body)
            d.add_response("cancel", "Cancel")
            d.add_response("delete", "Delete")
            d.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
            d.connect("response", lambda _d, r: do(r))
            d.present()
