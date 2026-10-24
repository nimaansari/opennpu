"""Desktop app logic that needs no display: reply rendering + install files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opennpu.desktop import markdown  # noqa: E402


def test_blocks_split_prose_and_code():
    b = markdown.blocks("Intro **bold**\n\n```python\nprint(1)\n```\nAfter")
    assert [x[0] for x in b] == ["text", "code", "text"]
    assert "<b>bold</b>" in b[0][1]
    assert b[1][1:] == ("python", "print(1)")


def test_unterminated_fence_while_streaming_is_code():
    b = markdown.blocks("Here:\n```js\nconst a = 1")
    assert b[-1] == ("code", "js", "const a = 1")


def test_markup_is_escaped():
    out = markdown.blocks("<script>alert(1)</script> & <b>x</b>")[0][1]
    assert "<script>" not in out and "&lt;script&gt;" in out and "&amp;" in out


def test_lists_and_headings():
    out = markdown.blocks("# Title\n- one\n  - nested\n2. two")[0][1]
    assert 'weight="bold"' in out and "•  one" in out and "    •  nested" in out and "2.  two" in out


def test_reasoning_split():
    assert markdown.split_reasoning("<think>hm</think>Answer") == ("hm", "Answer", False)
    assert markdown.split_reasoning("<think>still going") == ("still going", "", True)
    assert markdown.split_reasoning("plain") == ("", "plain", False)


def test_desktop_entry_and_icon_exist():
    assert (ROOT / "packaging/io.github.opennpu.OpenNPU.desktop").read_text().count("Exec=") >= 1
    assert (ROOT / "opennpu/desktop/icons/hicolor/scalable/apps/io.github.opennpu.OpenNPU.svg").exists()


def _media():
    import pytest
    pytest.importorskip("gi")
    from opennpu.desktop import media
    return media


def test_content_parts_plain_and_with_images():
    media = _media()
    assert media.content_parts("hi", []) == "hi"
    parts = media.content_parts("what is this", [("image/png", b"\x89PNG")])
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_read_image_limits(tmp_path):
    import pytest
    media = _media()
    ok = tmp_path / "a.png"
    ok.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    assert media.read_image(str(ok))[0] == "image/png"
    with pytest.raises(ValueError):
        bad = tmp_path / "notes.txt"
        bad.write_text("x")
        media.read_image(str(bad))
    with pytest.raises(ValueError):
        big = tmp_path / "big.jpg"
        big.write_bytes(b"0" * (media.MAX_IMAGE_BYTES + 1))
        media.read_image(str(big))


def test_speakable_strips_code_and_markup():
    media = _media()
    s = media.speakable("# Title\n**Bold** and `x`.\n```python\nprint(1)\n```\n- item [link](https://a.b)")
    assert "print" not in s and "**" not in s and "`" not in s and "https" not in s
    assert "Bold and x." in s and "item link" in s
