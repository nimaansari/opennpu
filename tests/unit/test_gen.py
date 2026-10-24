"""Generation: catalog, memory planning, weight counting, encoders, HTTP validation (no NPU needed)."""
import json
import struct

import numpy as np
import pytest

from opennpu.gen import catalog
from opennpu.gen import runtime as rt


def test_resolve_curated_prefixed_and_repo():
    assert catalog.resolve("sd-turbo")[:2] == ("image", "stabilityai/sd-turbo")
    assert catalog.resolve("video:foo/bar")[:2] == ("video", "foo/bar")
    assert catalog.resolve("stabilityai/sd-turbo")[3] == "sd-turbo"           # repo id of a curated entry
    assert catalog.resolve("x/y", task="speech")[0] == "speech"
    with pytest.raises(KeyError):
        catalog.resolve("x/y")                                                # unknown, no task


def test_every_catalog_entry_is_complete():
    for name, e in catalog.CATALOG.items():
        assert e["task"] in catalog.TASKS, name
        assert e["repo"].count("/") == 1, name
        assert e["ram_gb"] > 0 and e["note"], name


def _write_safetensors(path, tensors):
    header = {k: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]} for k, shape in tensors.items()}
    b = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(b)) + b)


def test_weights_params_reads_headers_and_skips_duplicate_full_copy(tmp_path):
    _write_safetensors(tmp_path / "unet/diffusion_pytorch_model.fp16.safetensors", {"w": (10, 20)})
    _write_safetensors(tmp_path / "unet/diffusion_pytorch_model.safetensors", {"w": (10, 20)})
    _write_safetensors(tmp_path / "vae/diffusion_pytorch_model.safetensors", {"a": (3, 3), "b": (4,)})
    assert rt.weights_params(tmp_path) == 200 + 13
    assert rt.weights_params(tmp_path, ["unet"]) == 200


def test_plan_memory(monkeypatch):
    gb = 2**30
    monkeypatch.setattr(rt, "available_ram", lambda: 40 * gb)
    assert rt.plan_memory(2e9, "x") == "auto"                    # 4 GB bf16 * 2.3 fits
    assert rt.plan_memory(10e9, "x") == "npu"                    # 20 GB * 1.4 fits, * 2.3 does not
    with pytest.raises(rt.NotEnoughMemory):
        rt.plan_memory(30e9, "x")
    # only the denoiser gets NPU copies: a big text encoder alone is cheap
    assert rt.plan_memory(7e9, "x", offloaded=1.3e9) == "auto"


def test_wav_bytes_header():
    w = rt.wav_bytes(np.array([0.0, 0.5, -1.0], dtype=np.float32), 16000)
    assert w[:4] == b"RIFF" and w[8:12] == b"WAVE"
    assert struct.unpack("<I", w[24:28])[0] == 16000
    assert len(w) == 44 + 3 * 2


def test_repo_id_validation():
    with pytest.raises(rt.ModelUnavailable):
        rt._repo_files("not-a-repo-id")


class _Mgr:
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.key = self.engine = None


def test_http_rejects_bad_requests_before_loading():
    from opennpu.gen import http
    for fn, req in ((http.images, {}), (http.images, {"prompt": "x", "size": "huge"}),
                    (http.images, {"prompt": "x", "size": "10x10"}),
                    (http.images, {"prompt": "x", "response_format": "url"}),
                    (http.speech, {}), (http.speech, {"input": "hi", "response_format": "midi"}),
                    (http.videos, {})):
        with pytest.raises(http.GenError) as e:
            fn(_Mgr(), None, print, req)
        assert e.value.code == 400


def test_http_task_mismatch():
    from opennpu.gen import http
    with pytest.raises(http.GenError) as e:
        http.images(_Mgr(), None, print, {"prompt": "x", "model": "mms-tts-eng"})
    assert e.value.code == 400 and "speech" in str(e.value)


def test_placement_roundtrip(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import sys
    import opennpu.offload  # noqa: F401
    O = sys.modules["opennpu.offload"]
    from opennpu import config
    monkeypatch.setattr(config, "CACHE", tmp_path)

    class Fake(O.AutoPlace):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.choice, self.stats = {}, {}

    m = torch.nn.Sequential(Fake(), Fake())
    m[0].choice[(1, 4, 64, 64)] = "npu"
    m[1].choice[(1, 8, 32, 32)] = "cpu"
    assert rt.save_placement(m, "model-x", "unet") == 2
    n = torch.nn.Sequential(Fake(), Fake())
    assert rt.load_placement(n, "model-x", "unet") == 2
    assert n[0].choice == {(1, 4, 64, 64): "npu"} and n[1].choice == {(1, 8, 32, 32): "cpu"}


def test_dtype_anchor_for_fully_offloaded_submodule():
    torch = pytest.importorskip("torch")

    class Emb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    m = torch.nn.Sequential(Emb())
    before = {"0": torch.bfloat16}
    m[0].lin = torch.nn.Identity()             # what offload does: weights move out of the module
    assert next(m[0].parameters(), None) is None
    assert rt._dtype_anchors(m, before) == 1
    assert next(m[0].parameters()).dtype == torch.bfloat16


def test_http_input_types_and_limits():
    from opennpu.gen import http
    bad = [(http.images, {"prompt": ["a", "b"]}), (http.images, {"prompt": "x" * 2001}),
           (http.images, {"prompt": "   "}), (http.images, {"prompt": "x", "negative_prompt": 5}),
           (http.videos, {"prompt": "x", "frames": 500}), (http.videos, {"prompt": "x", "size": "1920x1080"}),
           (http.videos, {"prompt": "x", "fps": 0}), (http.speech, {"input": "x" * 5000}),
           (http.speech, {"input": {"t": 1}}), (http.speech, {"input": "hi", "response_format": 3})]
    for fn, req in bad:
        with pytest.raises(http.GenError) as e:
            fn(_Mgr(), None, print, req)
        assert e.value.code == 400, req


def test_bad_model_does_not_unload_the_loaded_model():
    from opennpu.gen import http
    m = _Mgr()
    m.key, m.engine = "native:qwen3-8b", object()
    m._unload = lambda: (_ for _ in ()).throw(AssertionError("unloaded before validating"))
    with pytest.raises(http.GenError) as e:
        http.images(m, None, print, {"prompt": "x", "model": "nope"})
    assert e.value.code == 404 and m.key == "native:qwen3-8b"


def test_progress_and_cancel_when_idle():
    from opennpu.gen import http
    rt.set_progress(task=None, phase="idle", step=0, steps=0, started=None)
    assert http.progress()["phase"] == "idle"
    assert http.cancel()["cancelled"] is False and not rt.CANCEL.is_set()


def test_cancel_stops_at_next_step_and_returns_499():
    import time as _t
    from opennpu.gen import http

    class Eng:
        def generate(self, prompt, **kw):
            rt.set_progress(phase="generating", step=1, steps=10)
            assert http.progress()["step"] == 1
            assert http.cancel()["cancelled"] is True
            if rt.CANCEL.is_set():                     # what the diffusers step callback does
                raise rt.GenCancelled("generation cancelled")

    m = _Mgr()
    m.key, m.engine, m.gen_start, m.last = "image:sd-turbo", Eng(), None, _t.time()

    class Tel:
        generating_now = False
        def generating(self, a, b): pass
    with pytest.raises(http.GenError) as e:
        http.images(m, Tel(), print, {"prompt": "x", "model": "sd-turbo"})
    assert e.value.code == 499
    assert http.progress()["phase"] == "idle" and not rt.CANCEL.is_set()


def _multipart_body(fields, file_bytes=None, fname="a.webm"):
    import uuid
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if file_bytes is not None:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                f"Content-Type: audio/webm\r\n\r\n").encode() + file_bytes + b"\r\n"
    return out + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def test_multipart_parser_keeps_binary_file():
    from opennpu.gen import http
    blob = bytes(range(256)) * 4
    body, ct = _multipart_body({"model": "whisper-base", "language": "en"}, blob)
    form = http._multipart(body, ct)
    assert form["model"] == "whisper-base" and form["language"] == "en"
    assert form["file"] == ("a.webm", blob)


def test_transcriptions_rejects_bad_requests():
    from opennpu.gen import http
    cases = [({}, b"", "application/json"),                                    # nothing
             ({"audio_b64": "!!!"}, b"", "application/json"),                  # not base64
             ({"audio_b64": "AAAA", "response_format": "srt"}, b"", "application/json"),
             (None, *_multipart_body({"model": "whisper-base"}))]               # multipart without a file
    for req, body, ct in cases:
        with pytest.raises(http.GenError) as e:
            http.transcriptions(_Mgr(), None, print, req, body, ct)
        assert e.value.code == 400, (req, ct)


def test_transcribe_catalog():
    assert catalog.resolve("whisper-small")[:2] == ("transcribe", "openai/whisper-small")
    assert catalog.resolve("transcribe:openai/whisper-tiny")[0] == "transcribe"
