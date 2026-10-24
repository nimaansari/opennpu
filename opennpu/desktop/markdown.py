"""Model replies (markdown) -> blocks the chat view renders natively: ("text", pango markup) for
prose and lists, ("code", lang, source) for fenced code. Small and safe: everything is escaped
first, only a known subset becomes markup."""
import re
from html import escape


def _inline(s):
    # code spans and links are swapped out first so emphasis never reaches inside them
    keep = []

    def stash(markup):
        keep.append(markup)
        return f"\x00{len(keep) - 1}\x00"
    s = re.sub(r"`([^`\n]+)`", lambda m: stash(f'<span font_family="monospace" background="#8884">{escape(m.group(1), quote=False)}</span>'), s)
    s = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)",
               lambda m: stash(f'<a href="{escape(m.group(2), quote=True)}">{escape(m.group(1), quote=False)}</a>'), s)
    s = escape(s, quote=False)
    s = re.sub(r"\*\*(?=\S)([^*\n]*?\S)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*(?=\S)([^*\n]*?\S)\*(?![*\w])", r"<i>\1</i>", s)       # not "a * b * c"
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)


def _prose(text):
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.append(f'<span weight="bold" size="{"large" if len(m.group(1)) <= 2 else "medium"}">{_inline(m.group(2))}</span>')
            continue
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            indent = "    " * (len(m.group(1)) // 2)
            bullet = "•" if m.group(2) in "-*+" else m.group(2)
            out.append(f"{indent}{bullet}  {_inline(m.group(3))}")
            continue
        m = re.match(r"^&gt;\s?(.*)$", escape(line, quote=False))
        if m and line.startswith(">"):
            out.append(f"<i>{_inline(line[1:].lstrip())}</i>")
            continue
        out.append(_inline(line))
    return "\n".join(out).strip("\n")


def blocks(md):
    """Split into prose and fenced code blocks. Fences count only at the start of a line; an
    unterminated fence (still streaming) is code. Each text block also carries its source, so a
    renderer can fall back to plain text if the markup is ever rejected."""
    res, pos = [], 0
    for m in re.finditer(r"(?m)^[ \t]*```([\w+.-]*)[^\n]*\n(.*?)(^[ \t]*```[ \t]*$|\Z)", md, re.S):
        if m.start() > pos and md[pos:m.start()].strip():
            src = md[pos:m.start()].strip("\n")
            res.append(("text", _prose(src), src))
        res.append(("code", m.group(1), m.group(2).rstrip("\n")))
        pos = m.end()
    if md[pos:].strip():
        src = md[pos:].strip("\n")
        res.append(("text", _prose(src), src))
    return res


def split_reasoning(text):
    """('reasoning', 'answer', still_thinking) from '<think>...</think>answer', or from
    '...</think>answer' when the chat template opened the <think> itself (DeepSeek-R1 distills)."""
    m = re.match(r"^\s*<think>(.*?)(</think>|\Z)(.*)$", text, re.S)
    if m:
        return m.group(1).strip(), m.group(3).lstrip(), not m.group(2)
    m = re.match(r"^(.*?)</think>(.*)$", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).lstrip(), False
    return "", text, False
