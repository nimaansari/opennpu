"""Chat media for the desktop app: microphone recording (GStreamer), image attachments, and
read-aloud playback. Nothing here talks to the server; app.py wires it to the API."""
import base64
import mimetypes
import os
import re
import tempfile
import time

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGE_BYTES = 10 * 2**20
MAX_IMAGES = 4


class Recorder:
    """Records the default microphone to a 16 kHz mono WAV: what speech recognition expects.
    start() / stop(callback(path or None, error or None)); stop() lets the WAV header finish."""

    def __init__(self):
        self.pipe = None
        self.path = None
        self.started = None
        self._done = None

    @staticmethod
    def available():
        try:
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
            return all(Gst.ElementFactory.find(e) for e in ("autoaudiosrc", "wavenc", "audioresample"))
        except (ValueError, ImportError):
            return False

    @property
    def recording(self):
        return self.pipe is not None

    def start(self):
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        fd, self.path = tempfile.mkstemp(prefix="opennpu-mic-", suffix=".wav")
        os.close(fd)
        self.pipe = Gst.parse_launch(
            "autoaudiosrc ! audioconvert ! audioresample ! audio/x-raw,rate=16000,channels=1 "
            f"! wavenc ! filesink location={GLib.shell_quote(self.path)}")
        bus = self.pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._message)
        if self.pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._cleanup()
            raise RuntimeError("could not open the microphone")
        self.started = time.time()

    def elapsed(self):
        return time.time() - self.started if self.started else 0.0

    def stop(self, done):
        """Finish the file (EOS), then done(path, None); done(None, msg) on failure or silence."""
        if not self.pipe:
            return
        from gi.repository import Gst
        self._done = done
        self.pipe.send_event(Gst.Event.new_eos())
        GLib.timeout_add(3000, self._timeout)

    def cancel(self):
        path = self.path
        self._cleanup()
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _message(self, _bus, msg):
        from gi.repository import Gst
        if msg.type == Gst.MessageType.EOS:
            self._finish(None)
        elif msg.type == Gst.MessageType.ERROR:
            err, _dbg = msg.parse_error()
            self._finish(err.message)

    def _timeout(self):
        if self.pipe:
            self._finish(None)
        return False

    def _finish(self, error):
        done, path, secs = self._done, self.path, self.elapsed()
        self._cleanup()
        self._done = None
        if done is None:
            return
        if not error and secs < 0.4:
            error = "too short: hold the mic button a little longer"
        if error:
            try:
                os.unlink(path)
            except (OSError, TypeError):
                pass
            done(None, error)
        else:
            done(path, None)

    def _cleanup(self):
        if self.pipe:
            from gi.repository import Gst
            self.pipe.set_state(Gst.State.NULL)
            self.pipe.get_bus().remove_signal_watch()
        self.pipe, self.started = None, None


def read_image(path):
    """(mime, bytes) for an attachable image, or raises ValueError with a user-facing reason."""
    mime = mimetypes.guess_type(path)[0] or ""
    if mime not in IMAGE_TYPES:
        raise ValueError(f"{os.path.basename(path)}: only PNG, JPEG, WebP or GIF images")
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"{os.path.basename(path)} is {size / 2**20:.0f} MB; the limit is {MAX_IMAGE_BYTES // 2**20} MB")
    with open(path, "rb") as f:
        return mime, f.read()


def data_url(mime, data):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def content_parts(text, images):
    """OpenAI chat content: a plain string, or text + image_url parts when images are attached."""
    if not images:
        return text
    parts = [{"type": "text", "text": text}] if text else []
    parts += [{"type": "image_url", "image_url": {"url": data_url(m, d)}} for m, d in images]
    return parts


def speakable(markdown_text, limit=4000):
    """Reply text as plain sentences for text-to-speech: no code blocks, links or markup."""
    s = re.sub(r"```.*?(```|\Z)", " (code omitted) ", markdown_text, flags=re.S)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", "", s, flags=re.M)
    s = re.sub(r"[*_>|]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


class Player:
    """Plays one clip at a time (read-aloud); play() while playing stops the previous clip."""

    def __init__(self):
        self.media = None
        self.path = None

    def play(self, wav_bytes, on_end=None):
        self.stop()
        fd, self.path = tempfile.mkstemp(prefix="opennpu-say-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        self.media = Gtk.MediaFile.new_for_filename(self.path)
        if on_end:
            self.media.connect("notify::ended", lambda m, _p: m.get_ended() and on_end())
        self.media.play()

    @property
    def playing(self):
        return bool(self.media and self.media.get_playing())

    def stop(self):
        if self.media:
            self.media.pause()
            self.media = None
        if self.path:
            try:
                os.unlink(self.path)
            except OSError:
                pass
            self.path = None
