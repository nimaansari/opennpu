"""Chat engine for the universal 4-bit NPU model (model.NPU4Model): sessions, streaming, stop.

Vision-language models (NPU4Model.vision): a chat whose messages carry images (m["images"]: raw
image bytes) goes through the model's AutoProcessor; the first forward gets the pixels (the vision
tower runs once, its features replace the image placeholder tokens), decode steps are text-only on
the cache. Qwen-VL style M-RoPE positions are computed for the whole prompt here (get_rope_index)
and passed explicitly, so session caches never depend on the model's own rope_deltas state.
Text-only chats on the same model run exactly as before.
"""
import hashlib
import inspect
import io
import json
import os
import threading
import time
from pathlib import Path

import torch

from .model import NPU4Model, device

KV_SESSIONS = 3            # conversations whose KV cache stays warm (they are small: a few hundred MB)
# CPU threads for the small CPU-side ops: few, because busy CPU cores slow the NPU (shared power
# budget); measured best 2-4 (gpt-oss 14.1 tok/s at 2, Qwen3.6 7.9 at 4)
THREADS = int(os.environ.get("OPENNPU_NPU4_THREADS", "4"))
# images are scaled down to at most this many pixels: every image token is one token of prefill
# (~8-10 tok/s on the NPU); 448x448 is 196 tokens on Qwen3.5/3.6 (32x32 pixels per token)
IMAGE_PIXELS = int(os.environ.get("OPENNPU_NPU4_IMAGE_PIXELS", str(448 * 448)))


def image_digest(raw):
    return "img:" + hashlib.sha1(raw).hexdigest()[:16]


def vision_conversation(messages, max_pixels=IMAGE_PIXELS):
    """Server messages ({"role", "content", "images": [bytes]}) -> the processor's chat format
    (images first, then the text, like UniversalEngine) and the images' digests in prompt order."""
    from PIL import Image
    conv, digests = [], []
    for m in messages:
        parts = []
        for raw in m.get("images") or []:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if im.width * im.height > max_pixels:
                f = (max_pixels / (im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.BICUBIC)
            parts.append({"type": "image", "image": im})
            digests.append(image_digest(raw))
        parts.append({"type": "text", "text": m["content"]})
        conv.append({"role": m["role"], "content": parts})
    return conv, digests


def image_spans(ids, image_token_id):
    """[start, end) of each run of image placeholder tokens."""
    spans, i = [], 0
    while i < len(ids):
        if ids[i] == image_token_id:
            j = i
            while j < len(ids) and ids[j] == image_token_id:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def prompt_keys(ids, spans, digests):
    """What a session cache is matched on: the token ids, with each image's placeholder tokens
    replaced by that image's digest (two images of one size have the same placeholder ids).
    If the runs don't pair up with the images, every image token gets a unique key (no reuse)."""
    keys = list(ids)
    paired = len(spans) == len(digests)
    for n, (a, b) in enumerate(spans):
        for j in range(a, b):
            keys[j] = digests[n] if paired else f"img:?{time.time_ns()}:{j}"
    return keys


def safe_cut(n, spans):
    """A cache prefix length must not end inside an image (its features are computed whole)."""
    for a, b in spans:
        if a < n < b:
            return a
    return n


class _Harmony:
    """gpt-oss "harmony" output -> the <think>...</think> + answer text the apps already show.
    <|channel|>analysis<|message|>... is reasoning, <|channel|>final<|message|>... the answer."""

    def __init__(self):
        self.buf, self.mode, self.out = "", None, ""

    def feed(self, s):
        self.buf += s
        res = ""
        while True:
            if self.mode is None:
                i = self.buf.find("<|message|>")
                if i < 0:
                    return res
                head = self.buf[:i]
                self.buf = self.buf[i + len("<|message|>"):]
                self.mode = "think" if "analysis" in head or "commentary" in head else "final"
                if self.mode == "think":
                    res += "<think>"
            else:
                j = min([k for k in (self.buf.find("<|end|>"), self.buf.find("<|return|>"),
                                     self.buf.find("<|start|>"), self.buf.find("<|call|>")) if k >= 0], default=-1)
                if j < 0:
                    keep = 10                   # a marker may be split across pieces
                    res += self.buf[:-keep] if len(self.buf) > keep else ""
                    self.buf = self.buf[-keep:] if len(self.buf) > keep else self.buf
                    return res
                res += self.buf[:j]
                if self.mode == "think":
                    res += "</think>\n\n"
                self.buf = self.buf[j:]
                k = self.buf.find("|>")
                self.buf = self.buf[k + 2:] if k >= 0 else ""
                self.mode = None


class Npu4Engine:
    sessions = True

    def __init__(self, ref, log=print):
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoTokenizer
        d = Path(ref)
        if not d.is_dir():                     # an HF cache snapshot (only the files OpenNPU fetched)
            c = try_to_load_from_cache(ref, "config.json")
            if not isinstance(c, str):
                raise FileNotFoundError(f"{ref} is not downloaded")
            d = Path(c).parent
        self.dir = d
        torch.set_num_threads(8)                 # loading: quantization uses all cores
        self.m = NPU4Model(d, log=log)
        self.model = self.m.model
        self.tok = AutoTokenizer.from_pretrained(d)
        self.vision = self.m.vision              # the server lets users attach images when True
        self.proc = None
        if self.vision:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(d)
            fc = self.m.full_cfg
            self.image_token_id = getattr(fc, "image_token_id", None) or getattr(fc, "image_token_index", None)
        self.harmony = self.m.cfg.model_type == "gpt_oss"
        eos = set()
        gc = d / "generation_config.json"
        if gc.exists():
            e = json.loads(gc.read_text()).get("eos_token_id")
            eos |= set(e if isinstance(e, list) else [e] if e is not None else [])
        if self.tok.eos_token_id is not None:
            eos.add(self.tok.eos_token_id)
        for t in ("<|im_end|>", "<|endoftext|>", "<|return|>", "<|call|>", "<|eot_id|>", "<end_of_turn>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                eos.add(i)
        self.eos = eos
        self.ctx = getattr(self.m.cfg, "max_position_embeddings", None) or 4096
        self.ctx = min(self.ctx, 32768)
        self._sess = {}                         # session -> (cache, token ids in it)
        self._ckpt = {}                         # session -> (prompt keys, cache right after the prompt)
        lt = getattr(self.m.cfg, "layer_types", None) or []
        # linear-attention (recurrent) layers can't rewind: keep a copy of the cache at each prompt's end
        self.recurrent = any(t in ("linear_attention", "mamba", "ssm") for t in lt)
        self.lock = threading.Lock()
        import inspect
        # prompts: only the last position's logits (the LM head over every prompt token is the
        # vocabulary times the prompt length of wasted NPU work)
        self._last = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(self.model.forward).parameters else {}
        self._warm_up()

    def _warm_up(self):
        """One short prompt and one decode step at load, so the first real request doesn't pay for
        setting up the NPU runs (each matrix's 1-token and 8-token dispatch)."""
        try:
            t = self.tok.eos_token_id or 0
            with torch.no_grad():
                c = self._new_cache()
                self.model(input_ids=torch.tensor([[t] * 9]), past_key_values=c, use_cache=True, **self._last)
                self.model(input_ids=torch.tensor([[t]]), past_key_values=c, use_cache=True)
            if self.vision and hasattr(self, "_reset_rope"):
                self._reset_rope()
        except Exception:  # noqa: BLE001 - only an optimization
            pass

    # ------------------------------------------------------------------ helpers
    def _ids(self, messages, think, gen=True):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        kw = {"enable_thinking": think}
        if self.harmony:
            kw = {"reasoning_effort": "medium" if think else "low"}
        try:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=True, **kw)
        except (TypeError, ValueError):
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return [int(i) for i in (ids[0] if ids and isinstance(ids[0], list) else ids)]

    def _stable_len(self, messages, think, ids):
        """Tokens of the prompt that a later turn renders the same (the conversation up to the end of
        the last message, without the "assistant, answer now" tail): where to checkpoint."""
        try:
            base = self._ids(messages, think, gen=False)
        except Exception:  # noqa: BLE001
            return 0
        n = 0
        while n < min(len(base), len(ids)) and base[n] == ids[n]:
            n += 1
        return n

    def _vision_prompt(self, messages, think):
        """-> (encoding, keys, image spans, M-RoPE positions [3, 1, L] or None)."""
        conv, digests = vision_conversation(messages)
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            enc = self.proc.apply_chat_template(conv, enable_thinking=think, **kw)
        except (TypeError, ValueError):
            enc = self.proc.apply_chat_template(conv, **kw)
        ids = [int(i) for i in enc["input_ids"][0]]
        spans = image_spans(ids, self.image_token_id)
        keys = prompt_keys(ids, spans, digests)
        if len(spans) != len(digests):
            spans = [(0, len(ids))]                  # unknown layout: never cut a cache inside the prompt
        return enc, keys, spans, self._positions(enc)

    def _positions(self, enc):
        inner = getattr(self.model, "model", None)
        f = getattr(inner, "get_rope_index", None)
        if f is None:
            return None
        ok = inspect.signature(f).parameters
        kw = {k: enc[k] for k in ("mm_token_type_ids", "image_grid_thw", "video_grid_thw") if k in ok and k in enc}
        pos, _ = f(enc["input_ids"], **kw)
        return pos

    def _reset_rope(self):
        inner = getattr(self.model, "model", None)
        if inner is not None and getattr(inner, "rope_deltas", None) is not None:
            inner.rope_deltas = None             # stale M-RoPE state from an earlier image chat

    def _vision_feed(self, enc, spans, pos, cached):
        """Forward kwargs for the uncached part of a vision prompt: pixels of the images in it only."""
        kw = {"position_ids": pos[:, :, cached:]} if pos is not None else {}
        first = sum(1 for a, _ in spans if a < cached)          # images already in the cache
        if "image_grid_thw" in enc and "pixel_values" in enc:
            grid = enc["image_grid_thw"]
            if first < len(grid):
                rows = grid.prod(-1)
                r0 = int(rows[:first].sum())
                kw["pixel_values"] = enc["pixel_values"][r0:]
                kw["image_grid_thw"] = grid[first:]
        elif "pixel_values" in enc and first < len(spans):
            kw["pixel_values"] = enc["pixel_values"][first:]
        for k, v in enc.items():                    # per-token extras (token_type_ids, ...)
            if k not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "video_grid_thw",
                         "mm_token_type_ids") and torch.is_tensor(v) and v.dim() == 2 \
                    and v.shape[1] == enc["input_ids"].shape[1]:
                kw[k] = v[:, cached:]
        return kw

    def _new_cache(self):
        from transformers import DynamicCache
        try:
            return DynamicCache(config=self.model.config)
        except TypeError:
            return DynamicCache()

    def _cache_for(self, session, ids, spans=()):
        """A cache holding a prefix of ids (reused from the session when it is one) and its length.
        ids: the prompt's keys (token ids; image tokens keyed by the image), spans: image runs."""
        s = self._sess.pop(session or "_anon", None)
        if s:
            cache, have = s
            n = 0
            while n < min(len(have), len(ids) - 1) and have[n] == ids[n]:
                n += 1
            n = safe_cut(n, spans)
            if n == len(have):
                return cache, n
            if n > 0 and hasattr(cache, "crop") and not getattr(self, "recurrent", False):
                try:
                    cache.crop(n)
                    return cache, n
                except Exception:  # noqa: BLE001 - recurrent (linear-attention) state can't rewind
                    pass
        ck = getattr(self, "_ckpt", {}).get(session or "_anon")
        if ck and not spans and len(ck[0]) < len(ids) and ids[:len(ck[0])] == ck[0]:
            import copy                        # the previous prompt is a prefix: resume after it
            return copy.deepcopy(ck[1]), len(ck[0])
        return self._new_cache(), 0

    def _keep(self, session, cache, ids):
        self._sess[session or "_anon"] = (cache, ids)
        while len(self._sess) > KV_SESSIONS:
            self._sess.pop(next(iter(self._sess)))

    # ------------------------------------------------------------------ chat
    def chat(self, messages, max_tokens, temperature, think, on_text, session=None):
        vis = any(m.get("images") for m in messages)
        if vis and not self.vision:
            raise ValueError("this model has no vision tower; pick a vision model (marked 'vision') for images")
        if vis:
            enc, keys, spans, pos = self._vision_prompt(messages, think)
            ids = [int(i) for i in enc["input_ids"][0]]
        else:
            ids = self._ids(messages, think)
            keys, spans = ids, ()
        if len(ids) >= self.ctx:
            raise ValueError(f"this conversation is {len(ids)} tokens; the model's context here is {self.ctx}. "
                             "Start a new conversation (or shorten the message).")
        dev = device()
        torch.set_num_threads(THREADS)
        with self.lock, torch.no_grad():
            cache, cached = self._cache_for(session, keys, spans)
            t0 = time.perf_counter()
            feed = ids[cached:]
            kw, nxt = {}, None
            if vis:
                if "image_grid_thw" not in enc and any(a < cached for a, _ in spans) \
                        and any(a >= cached for a, _ in spans):
                    cache, cached = self._new_cache(), 0   # can't split this processor's pixels per image
                    feed = ids
                kw = self._vision_feed(enc, spans, pos, cached)
                if pos is not None:
                    nxt = int(pos.max()) + 1           # M-RoPE: generated tokens continue from here
            elif self.vision:
                self._reset_rope()
            cut = self._stable_len(messages, think, ids) if self.recurrent and not vis else 0
            if cached < cut < len(ids):
                # recurrent state can't rewind: checkpoint where the next turn's prompt still agrees
                import copy
                self.model(input_ids=torch.tensor([ids[cached:cut]]), past_key_values=cache, use_cache=True, **getattr(self, "_last", {}))
                self._ckpt[session or "_anon"] = (list(keys[:cut]), copy.deepcopy(cache))
                while len(self._ckpt) > KV_SESSIONS:
                    self._ckpt.pop(next(iter(self._ckpt)))
                feed = ids[cut:]
            out = self.model(input_ids=torch.tensor([feed]), past_key_values=cache, use_cache=True, **kw, **getattr(self, "_last", {}))
            lg = out.logits[0, -1].float()
            t1 = time.perf_counter()
            fed = list(keys)
            gen, text, shown = [], "", ""
            parser = _Harmony() if self.harmony else None
            gen_s = torch.Generator().manual_seed(int(time.time()))
            n = min(max_tokens, self.ctx - len(ids))
            c0 = dev.calls
            for _ in range(n):
                if temperature > 0:
                    p = torch.softmax(lg / temperature, -1)
                    tok = int(torch.multinomial(p, 1, generator=gen_s))
                else:
                    tok = int(lg.argmax())
                if tok in self.eos:
                    break
                gen.append(tok)
                piece = self.tok.decode(gen, skip_special_tokens=not self.harmony)
                if not piece.endswith("�"):
                    new = piece[len(text):]
                    text = piece
                    if parser:
                        new = parser.feed(new)
                    if new:
                        on_text(new)
                        shown += new
                step = {} if nxt is None else {"position_ids": torch.full((pos.shape[0], 1, 1), nxt + len(gen) - 1)}
                out = self.model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True, **step)
                fed.append(tok)
                lg = out.logits[0, -1].float()
            if parser:
                rest = parser.feed("<|end|>") if parser.mode else ""
                if rest:
                    on_text(rest)
            t2 = time.perf_counter()
            self._keep(session, cache, fed)
        return {"prompt_tokens": len(ids), "cached_tokens": cached, "completion_tokens": len(gen),
                "prefill_s": round(t1 - t0, 2), "decode_tok_s": round(len(gen) / max(t2 - t1, 1e-9), 2),
                "npu_dispatches_per_token": round((dev.calls - c0) / max(len(gen), 1))}

    def embed(self, texts):
        raise ValueError("this is a chat model; load an embedding model for /v1/embeddings")

    def close(self):
        self._sess.clear()
        self._ckpt.clear()
        del self.model, self.m
