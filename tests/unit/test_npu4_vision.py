"""Images in chat on the 4-bit NPU engine: message -> processor inputs, session-cache keys (no NPU)."""
import io
import json

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")

from opennpu.native.npu4 import engine as E  # noqa: E402

IMG = 99                                          # image placeholder token id in these tests


def _png(w, h, color="red"):
    b = io.BytesIO()
    PIL.new("RGB", (w, h), color).save(b, "PNG")
    return b.getvalue()


def test_vision_conversation_parts_and_resize():
    big, small = _png(1000, 800), _png(64, 64, "blue")
    msgs = [{"role": "user", "content": "what is this?", "images": [big, small]},
            {"role": "assistant", "content": "two squares"},
            {"role": "user", "content": "colors?"}]
    conv, digests = E.vision_conversation(msgs, max_pixels=448 * 448)
    assert [m["role"] for m in conv] == ["user", "assistant", "user"]
    parts = conv[0]["content"]
    assert [p["type"] for p in parts] == ["image", "image", "text"]       # images first, then the text
    assert parts[0]["image"].width * parts[0]["image"].height <= 448 * 448
    assert abs(parts[0]["image"].width / parts[0]["image"].height - 1.25) < 0.01
    assert parts[1]["image"].size == (64, 64)                            # small images untouched
    assert conv[1]["content"] == [{"type": "text", "text": "two squares"}]
    assert digests == [E.image_digest(big), E.image_digest(small)] and digests[0] != digests[1]


def test_keys_tell_same_size_images_apart():
    ids = [1, 2, IMG, IMG, IMG, 3, IMG, IMG, 4]
    spans = E.image_spans(ids, IMG)
    assert spans == [(2, 5), (6, 8)]
    a = E.prompt_keys(ids, spans, ["img:a", "img:b"])
    b = E.prompt_keys(ids, spans, ["img:a", "img:c"])
    assert a[:6] == b[:6] and a[6] != b[6]        # same text, different second image
    assert [k for k in a if isinstance(k, int)] == [1, 2, 3, 4]
    u = E.prompt_keys(ids, spans, ["img:a"])       # runs don't pair with the images: never shared
    assert u[2] != E.prompt_keys(ids, spans, ["img:a"])[2]


def test_safe_cut_never_inside_an_image():
    spans = [(2, 5), (6, 8)]
    assert E.safe_cut(4, spans) == 2
    assert E.safe_cut(5, spans) == 5 and E.safe_cut(2, spans) == 2 and E.safe_cut(9, spans) == 9


class _Cache:
    def __init__(self):
        self.cropped = None

    def crop(self, n):
        self.cropped = n


def _engine():
    eng = E.Npu4Engine.__new__(E.Npu4Engine)
    eng._sess = {}
    eng._new_cache = lambda: "fresh"
    return eng


def test_session_reuse_text_and_images():
    eng = _engine()
    c = _Cache()
    eng._keep("s", c, [1, 2, 3])
    assert eng._cache_for("s", [1, 2, 3, 4, 5]) == (c, 3)               # text-only: as before
    # the cached prompt had image A; the new one has image B of the same size: cut before the image
    have = E.prompt_keys([1, IMG, IMG, 2, 7], [(1, 3)], ["img:a"])
    new = E.prompt_keys([1, IMG, IMG, 2, 8], [(1, 3)], ["img:b"])
    eng._keep("s", c, have)
    assert eng._cache_for("s", new, [(1, 3)]) == (c, 1) and c.cropped == 1
    # a text-only prompt never matches a cache built with an image in that place
    eng._keep("s", c, have)
    assert eng._cache_for("s", [1, IMG, IMG, 2, 7, 9]) == (c, 1)
    # same image again (follow-up turn): the whole cache is reused
    eng._keep("s", c, have)
    assert eng._cache_for("s", have + [5, 6], [(1, 3)]) == (c, 5)


def test_vision_feed_only_uncached_images():
    eng = _engine()
    ids = torch.tensor([[1, IMG, IMG, 2, IMG, IMG, IMG, IMG, 3]])
    grid = torch.tensor([[1, 2, 4], [1, 4, 4]])                         # 8 and 16 patch rows
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": torch.zeros_like(ids),
           "pixel_values": torch.arange(24.0)[:, None].repeat(1, 3), "image_grid_thw": grid}
    pos = torch.arange(9).view(1, 1, -1).expand(3, 1, -1)
    spans = E.image_spans(ids[0].tolist(), IMG)
    kw = eng._vision_feed(enc, spans, pos, 0)
    assert kw["pixel_values"].shape[0] == 24 and kw["image_grid_thw"].shape[0] == 2
    assert kw["position_ids"].shape == (3, 1, 9)
    kw = eng._vision_feed(enc, spans, pos, 4)                            # first image already cached
    assert kw["pixel_values"][0, 0] == 8 and kw["pixel_values"].shape[0] == 16
    assert kw["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert kw["position_ids"][0, 0].tolist() == [4, 5, 6, 7, 8]
    kw = eng._vision_feed(enc, spans, pos, 8)                            # all images cached: text only
    assert "pixel_values" not in kw and "mm_token_type_ids" not in kw and "attention_mask" not in kw


def test_text_only_chat_refuses_images_without_a_vision_tower():
    eng = _engine()
    eng.vision = False
    with pytest.raises(ValueError, match="vision"):
        eng.chat([{"role": "user", "content": "hi", "images": [_png(8, 8)]}], 4, 0, False, print)


VLM_MOE = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
           "text_config": {"num_experts": 256, "num_experts_per_tok": 8}, "vision_config": {"depth": 27}}


@pytest.mark.parametrize("env,vision", [("cpu", True), ("off", False)])
def test_describe_marks_npu4_vlms_as_vision(monkeypatch, tmp_path, env, vision):
    import importlib
    monkeypatch.setenv("OPENNPU_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("OPENNPU_NPU4_VISION", env)
    snap = tmp_path / "hub" / "models--org--vlm-moe" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(VLM_MOE))
    (snap / "model.safetensors").write_text("")
    import opennpu.native.server as srv
    srv = importlib.reload(srv)
    d = srv.describe("npu4:org/vlm-moe", "npu4", (str(snap),))
    assert d["vision"] is vision and d["engine"] == "npu4"
