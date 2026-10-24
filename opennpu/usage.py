"""Local usage analytics and hardware history. Everything stays on this machine.

    $OPENNPU_HOME/usage.db  (SQLite, WAL)

What is recorded, per request: time, kind (chat / embed / image / video / speech), model, engine,
client app, status, token counts, time to first token, duration, decode speed, energy, and small
generation details (steps, size, frames). Never prompt or reply text. Hardware is sampled from
/proc and /sys every 5 s and stored as per-minute averages / maxima, kept for 30 days.

Recording can be switched off (OPENNPU_USAGE=0, or the dashboard toggle) and cleared at any time.
Writes go through one background thread, so a slow disk can never stall a streaming reply.
"""
import json
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path

from . import config

DB = Path(os.environ.get("OPENNPU_USAGE_DB", config.HOME / "usage.db"))
HW_KEEP_DAYS = 30
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  ts REAL NOT NULL, kind TEXT, model TEXT, engine TEXT, client TEXT, status TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, ttft_s REAL, total_s REAL, decode_tok_s REAL,
  joules REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS hw (
  ts INTEGER PRIMARY KEY, cpu REAL, cpu_max REAL, ram_used REAL, ram_total REAL, swap_used REAL,
  npu_busy REAL, npu_busy_max REAL, cpu_temp REAL, cpu_temp_max REAL, gpu_temp REAL, nvme_temp REAL,
  power_w REAL, server_rss REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
RANGES = {"1h": 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "all": None}


def _connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB), timeout=10, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.executescript(SCHEMA)
    return c


class Store:
    def __init__(self):
        self.q = queue.Queue(maxsize=10000)
        self._read = None
        self._lock = threading.Lock()
        env_off = os.environ.get("OPENNPU_USAGE", "1") == "0"
        self.enabled = not env_off and self._meta("enabled", "1") == "1"
        threading.Thread(target=self._writer, daemon=True, name="usage-writer").start()

    # ---- writes: queued, applied by one thread
    def _writer(self):
        db = _connect()
        while True:
            op, args = self.q.get()
            try:
                if op == "req":
                    db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", args)
                elif op == "hw":
                    db.execute("INSERT OR REPLACE INTO hw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", args)
                    db.execute("DELETE FROM hw WHERE ts < ?", (time.time() - HW_KEEP_DAYS * 86400,))
                elif op == "sql":
                    db.execute(*args)
                elif op == "vacuum":
                    db.commit()
                    db.execute("VACUUM")
                db.commit()
            except sqlite3.Error as e:
                print(f"usage: {e}", flush=True)
            finally:
                if op == "sync":
                    args.set()

    def flush(self, timeout=5):
        ev = threading.Event()
        self.q.put(("sync", ev))
        ev.wait(timeout)

    def record(self, kind, model, engine=None, client=None, status="ok", prompt_tokens=None, completion_tokens=None,
               ttft_s=None, total_s=None, decode_tok_s=None, joules=None, **detail):
        if not self.enabled:
            return
        try:
            self.q.put_nowait(("req", (time.time(), kind, model, engine, client or "api", status, prompt_tokens,
                                       completion_tokens, ttft_s, total_s, decode_tok_s, joules,
                                       json.dumps(detail) if detail else None)))
        except queue.Full:
            pass

    def hw_row(self, row):
        try:
            self.q.put_nowait(("hw", row))
        except queue.Full:
            pass

    def clear(self, what="all"):
        if what in ("all", "usage"):
            self.q.put(("sql", ("DELETE FROM requests",)))
        if what in ("all", "hardware"):
            self.q.put(("sql", ("DELETE FROM hw",)))
        self.q.put(("vacuum", None))
        self.flush(30)

    def set_enabled(self, on):
        self.enabled = bool(on)
        self.q.put(("sql", ("INSERT OR REPLACE INTO meta VALUES ('enabled', ?)", ("1" if on else "0",))))
        self.flush()

    # ---- reads
    def _rc(self):
        with self._lock:
            if self._read is None:
                self._read = _connect()
            return self._read

    def _meta(self, k, d):
        try:
            r = self._rc().execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
            return r[0] if r else d
        except sqlite3.Error:
            return d

    def _rows(self, sql, args=()):
        with self._lock:
            if self._read is None:
                self._read = _connect()
            return self._read.execute(sql, args).fetchall()

    def summary(self, rng="24h"):
        span = RANGES.get(rng, 86400)
        now = time.time()
        since = now - span if span else 0
        where = "ts >= ?"
        tot = self._rows(f"""SELECT COUNT(*), SUM(status='ok'), SUM(status='error'), SUM(status='cancelled'),
            SUM(COALESCE(prompt_tokens,0)), SUM(COALESCE(completion_tokens,0)), SUM(COALESCE(total_s,0)),
            SUM(COALESCE(joules,0)), SUM(CASE WHEN decode_tok_s>0 THEN completion_tokens END),
            SUM(CASE WHEN decode_tok_s>0 THEN completion_tokens / decode_tok_s END),
            AVG(CASE WHEN kind='chat' AND status='ok' THEN ttft_s END), MIN(ts)
            FROM requests WHERE {where}""", (since,))[0]
        n, ok, err, canc, tin, tout, busy, joules, wtok, wsec, ttft, first = tot
        totals = {"requests": n or 0, "ok": ok or 0, "errors": err or 0, "cancelled": canc or 0,
                  "prompt_tokens": tin or 0, "completion_tokens": tout or 0, "busy_s": round(busy or 0, 1),
                  "energy_wh": round((joules or 0) / 3600, 3),
                  "avg_tok_s": round(wtok / wsec, 2) if wtok and wsec else None,
                  "avg_ttft_s": round(ttft, 2) if ttft is not None else None,
                  "first_ts": first}
        by_model = [dict(zip(("model", "engine", "kind", "requests", "completion_tokens", "prompt_tokens", "avg_tok_s",
                              "busy_s", "last_ts"), r)) for r in self._rows(f"""
            SELECT model, MAX(engine), MAX(kind), COUNT(*), SUM(COALESCE(completion_tokens,0)), SUM(COALESCE(prompt_tokens,0)),
                   ROUND(SUM(CASE WHEN decode_tok_s>0 THEN completion_tokens END) /
                         NULLIF(SUM(CASE WHEN decode_tok_s>0 THEN completion_tokens / decode_tok_s END), 0), 2),
                   ROUND(SUM(COALESCE(total_s,0)), 1), MAX(ts)
            FROM requests WHERE {where} GROUP BY model ORDER BY COUNT(*) DESC, model""", (since,))]
        group = lambda col: {k or "unknown": v for k, v in self._rows(  # noqa: E731
            f"SELECT {col}, COUNT(*) FROM requests WHERE {where} GROUP BY {col} ORDER BY 2 DESC", (since,))}
        # time buckets in local time: hourly for a day or less, daily beyond
        bucket = 3600 if span and span <= 86400 else 86400
        off = -time.localtime(now).tm_gmtoff
        start = (since if span else (first or now))
        b0 = int((start - off) // bucket * bucket + off)
        series = {int(b): (r, t) for b, r, t in self._rows(f"""
            SELECT CAST((ts - ?) / ? AS INTEGER) * ? + ?, COUNT(*), SUM(COALESCE(completion_tokens,0))
            FROM requests WHERE {where} GROUP BY 1""", (b0, bucket, bucket, b0, since))}
        timeline = []
        t = b0
        while t <= now and len(timeline) < 400:
            r, tk = series.get(t, (0, 0))
            timeline.append({"t": t, "requests": r, "tokens": tk})
            t += bucket
        hours = [0] * 24
        for ts, in self._rows(f"SELECT ts FROM requests WHERE {where}", (since,)):
            hours[time.localtime(ts).tm_hour] += 1
        recent = [dict(zip(("ts", "kind", "model", "engine", "client", "status", "prompt_tokens", "completion_tokens",
                            "ttft_s", "total_s", "decode_tok_s", "joules", "detail"), r)) for r in self._rows(f"""
            SELECT * FROM requests WHERE {where} ORDER BY ts DESC LIMIT 50""", (since,))]
        for r in recent:
            r["detail"] = json.loads(r["detail"]) if r["detail"] else None
        return {"range": rng, "enabled": self.enabled, "bucket_s": bucket, "totals": totals, "by_model": by_model,
                "by_kind": group("kind"), "by_client": group("client"), "by_status": group("status"),
                "timeline": timeline, "hour_of_day": hours, "recent": recent, "stored_at": str(DB)}

    def hardware(self, rng="24h"):
        span = RANGES.get(rng, 86400) or HW_KEEP_DAYS * 86400
        since = time.time() - span
        step = 60 if span <= 3 * 3600 else 900 if span <= 86400 else 3600 if span <= 7 * 86400 else 4 * 3600
        cols = ("cpu", "cpu_max", "ram_used", "ram_total", "swap_used", "npu_busy", "npu_busy_max", "cpu_temp",
                "cpu_temp_max", "gpu_temp", "nvme_temp", "power_w", "server_rss")
        agg = ", ".join(f"{'MAX' if c.endswith('_max') else 'AVG'}({c})" for c in cols)
        rows = self._rows(f"SELECT CAST(ts / ? AS INTEGER) * ?, {agg} FROM hw WHERE ts >= ? GROUP BY 1 ORDER BY 1",
                          (step, step, since))
        series = [{"t": r[0], **{c: (round(v, 2) if v is not None else None) for c, v in zip(cols, r[1:])}} for r in rows]
        return {"range": rng, "step_s": step, "series": series}

    def csv(self):
        import csv
        import io
        out = io.StringIO()
        w = csv.writer(out)
        cols = ["time", "kind", "model", "engine", "client", "status", "prompt_tokens", "completion_tokens",
                "ttft_s", "total_s", "decode_tok_s", "joules", "detail"]
        w.writerow(cols)
        for r in self._rows("SELECT * FROM requests ORDER BY ts"):
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])), *r[1:]])
        return out.getvalue()


# ------------------------------------------------------------------ hardware sampling (cheap: /proc, /sys)
def _read(p, cast=float):
    try:
        return cast(Path(p).read_text().strip())
    except (OSError, ValueError):
        return None


def _hwmon():
    """name -> hwmon dir, plus the temperature input we want from each."""
    out = {}
    for d in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = _read(d / "name", str)
        if name:
            out.setdefault(name, d)
    return out


def _temp(dirs, name, label=None):
    d = dirs.get(name)
    if not d:
        return None
    for f in sorted(d.glob("temp*_input")):
        if label is None or _read(str(f).replace("_input", "_label"), str) in (label, None):
            v = _read(f)
            return v / 1000 if v is not None else None
    return None


def static_info(npu=None):
    """What this machine is (read once, cheap)."""
    import platform
    import shutil
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown")
    mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "SwapTotal")}
    osname = "Linux"
    try:
        osname = next(l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release") if l.startswith("PRETTY_NAME="))
    except (OSError, StopIteration):
        pass
    gpu = None
    for d in Path("/sys/class/drm").glob("card*/device"):
        if _read(d / "vendor", str) == "0x1002":
            gpu = "AMD Radeon (" + (_read(d / "device", str) or "?") + ")"
            break
    disks = {}
    for label, p in (("models", config.MODELS), ("cache", config.CACHE), ("home", config.HOME)):
        try:
            u = shutil.disk_usage(p if Path(p).exists() else Path.home())
            disks[label] = {"path": str(p), "free": u.free, "total": u.total}
        except OSError:
            pass
    bat = Path("/sys/class/power_supply/BAT0")
    return {"cpu": cpu, "cores": os.cpu_count(), "ram_total": mem.get("MemTotal"), "swap_total": mem.get("SwapTotal"),
            "os": osname, "kernel": platform.release(), "gpu": gpu, "npu": npu or {}, "disks": disks,
            "battery": {"capacity": _read(bat / "capacity", int), "status": _read(bat / "status", str),
                        "health": (round(100 * _read(bat / "energy_full") / _read(bat / "energy_full_design"))
                                   if _read(bat / "energy_full") and _read(bat / "energy_full_design") else None)}
            if bat.exists() else None}


class Sampler:
    """Every 5 s: CPU, RAM, NPU busy, temperatures, power. One row per minute (avg + max)."""

    def __init__(self, store, npu_busy, server_rss, interval=5):
        self.store, self.npu_busy, self.server_rss, self.interval = store, npu_busy, server_rss, interval
        self.now = {}
        self._buf = []
        self._cpu = self._cpu_times()
        self._dirs = _hwmon()
        threading.Thread(target=self._loop, daemon=True, name="hw-sampler").start()

    @staticmethod
    def _cpu_times():
        f = open("/proc/stat").readline().split()[1:]
        v = [int(x) for x in f]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)

    def sample(self):
        tot, idle = self._cpu_times()
        dt, di = tot - self._cpu[0], idle - self._cpu[1]
        self._cpu = (tot, idle)
        mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")
               if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
        bat = Path("/sys/class/power_supply/BAT0")
        power = _read(bat / "power_now") / 1e6 if _read(bat / "status", str) == "Discharging" and _read(bat / "power_now") else None
        s = {"cpu": 100.0 * (dt - di) / dt if dt else 0.0,
             "ram_used": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0), "ram_total": mem.get("MemTotal"),
             "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
             "npu_busy": self.npu_busy(),
             "cpu_temp": _temp(self._dirs, "k10temp") or _temp(self._dirs, "acpitz"),
             "gpu_temp": _temp(self._dirs, "amdgpu"), "nvme_temp": _temp(self._dirs, "nvme", "Composite"),
             "power_w": power, "server_rss": self.server_rss()}
        self.now = dict(s, ts=time.time())
        return s

    def _loop(self):
        minute = int(time.time() // 60)
        time.sleep(1)
        try:
            self.sample()                       # live values within a second of start, not after a full interval
        except Exception:  # noqa: BLE001
            pass
        while True:
            time.sleep(self.interval)
            try:
                s = self.sample()
            except Exception as e:  # noqa: BLE001 - a sensor hiccup must not kill sampling
                print(f"hw sampler: {e!r}", flush=True)
                continue
            m = int(time.time() // 60)
            if m != minute and self._buf:
                self._flush(minute * 60)
                minute = m
            self._buf.append(s)

    def _flush(self, ts):
        b, self._buf = self._buf, []
        if not self.store.enabled:
            return

        def avg(k):
            v = [x[k] for x in b if x.get(k) is not None]
            return sum(v) / len(v) if v else None

        def mx(k):
            v = [x[k] for x in b if x.get(k) is not None]
            return max(v) if v else None
        self.store.hw_row((ts, avg("cpu"), mx("cpu"), avg("ram_used"), avg("ram_total"), avg("swap_used"),
                           avg("npu_busy"), mx("npu_busy"), avg("cpu_temp"), mx("cpu_temp"), avg("gpu_temp"),
                           avg("nvme_temp"), avg("power_w"), avg("server_rss")))
