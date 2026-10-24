"""Chat sessions, saved on this computer and shared by every app (web UI, desktop, CLI, API).

    $OPENNPU_HOME/sessions.db          SQLite: one row per session (messages as JSON)
    $OPENNPU_HOME/sessions/media/      images attached to messages, by content hash

A message: {"role", "content", "images"?: ["/v1/sessions/media/<hash>.<ext>"], "stats"?, "error"?,
"stopped"?}. Nothing leaves the machine.
"""
import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from . import config

DB = Path(os.environ.get("OPENNPU_SESSIONS_DB", config.HOME / "sessions.db"))
MEDIA = DB.parent / "sessions" / "media"
MEDIA_URL = "/v1/sessions/media/"
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
MAX_IMAGE_BYTES = 10 * 2**20
MAX_IMAGES = 4


class SessionError(ValueError):
    pass


def _magic(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def save_image(data):
    """Bytes -> media URL (deduplicated by content)."""
    if len(data) > MAX_IMAGE_BYTES:
        raise SessionError(f"image is {len(data) / 2**20:.1f} MB; the limit is {MAX_IMAGE_BYTES // 2**20} MB")
    ext = _magic(data)
    if not ext:
        raise SessionError("images must be PNG, JPEG, WebP or GIF")
    name = hashlib.sha256(data).hexdigest()[:32] + "." + ext
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = MEDIA / name
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return MEDIA_URL + name


def image_bytes(url):
    """An image reference from a message -> bytes. data: URLs or our own media URLs only (no
    fetching from the network, no local paths)."""
    if not isinstance(url, str):
        raise SessionError("image_url.url must be a string")
    if url.startswith("data:"):
        head, _, b64 = url.partition(",")
        mime = head[5:].split(";")[0]
        if mime not in IMAGE_TYPES:
            raise SessionError("images must be PNG, JPEG, WebP or GIF")
        try:
            data = base64.b64decode(b64, validate=False)
        except ValueError:
            raise SessionError("image data is not valid base64")
        if len(data) > MAX_IMAGE_BYTES:
            raise SessionError(f"image is {len(data) / 2**20:.1f} MB; the limit is {MAX_IMAGE_BYTES // 2**20} MB")
        return data
    if MEDIA_URL in url:
        name = url.split(MEDIA_URL, 1)[1]
        p = (MEDIA / name).resolve()
        if p.parent != MEDIA.resolve() or not p.is_file():
            raise SessionError("unknown image")
        return p.read_bytes()
    raise SessionError("images must be data: URLs (or images already in a session); remote URLs are not fetched")


def media_path(name):
    p = (MEDIA / name).resolve()
    return p if p.parent == MEDIA.resolve() and p.is_file() else None


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        DB.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(DB), check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, model TEXT,
            created REAL, updated REAL, messages TEXT)""")
        self.db.commit()

    def _row(self, r, full=True):
        msgs = json.loads(r[5] or "[]")
        d = {"id": r[0], "title": r[1] or "New conversation", "model": r[2], "created": r[3], "updated": r[4]}
        d["messages"] = msgs if full else len(msgs)
        return d

    def list(self):
        with self._lock:
            rows = self.db.execute("SELECT * FROM sessions ORDER BY updated DESC").fetchall()
        return [self._row(r, full=False) for r in rows]

    def get(self, sid):
        with self._lock:
            r = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not r:
            raise KeyError(f"no session {sid!r}")
        return self._row(r)

    def exists(self, sid):
        with self._lock:
            return self.db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone() is not None

    def create(self, title=None, model=None, messages=None, sid=None, created=None):
        sid = sid or uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                            (sid, title or "New conversation", model, created or now, now, json.dumps(messages or [])))
            self.db.commit()
        return self.get(sid)

    def update(self, sid, title=None, model=None, messages=None):
        cur = self.get(sid)
        with self._lock:
            self.db.execute("UPDATE sessions SET title=?, model=?, messages=?, updated=? WHERE id=?",
                            (title if title is not None else cur["title"], model if model is not None else cur["model"],
                             json.dumps(messages if messages is not None else cur["messages"]), time.time(), sid))
            self.db.commit()
        return self.get(sid)

    def delete(self, sid):
        with self._lock:
            n = self.db.execute("DELETE FROM sessions WHERE id=?", (sid,)).rowcount
            self.db.commit()
        if not n:
            raise KeyError(f"no session {sid!r}")

    def import_many(self, items):
        """Migrate conversations from an app's old local store; existing ids are skipped."""
        added = 0
        for c in items:
            if not isinstance(c, dict) or not isinstance(c.get("messages"), list):
                continue
            sid = str(c.get("id") or uuid.uuid4().hex)
            if self.exists(sid):
                continue
            msgs = [clean_message(m) for m in c["messages"] if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")]
            self.create(c.get("title"), c.get("model"), msgs, sid=sid, created=c.get("t") or c.get("created"))
            added += 1
        return added


def clean_message(m):
    """Keep the fields a session stores, with images turned into media URLs."""
    out = {"role": m["role"], "content": ""}
    c = m.get("content")
    images = list(m.get("images") or [])
    if isinstance(c, list):
        texts = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type", "text") == "text":
                texts.append(str(p.get("text", "")))
            elif p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
                images.append(url)
        c = "".join(texts)
    out["content"] = c if isinstance(c, str) else ""
    if images:
        out["images"] = [u if isinstance(u, str) and MEDIA_URL in u else save_image(image_bytes(u)) for u in images]
    for k in ("stats", "error", "stopped"):
        if m.get(k) is not None:
            out[k] = m[k]
    return out
