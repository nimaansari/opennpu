"""GGUF checkpoint reader with the HFCheckpoint interface (HF tensor names, float access).

llama.cpp GGUF -> HF names, dequantized on demand (memory-mapped, one tensor or one
expert at a time). Lets the native engine take the GGUF files people already have.

    ck = GGUFCheckpoint("model.gguf", config_dir)   # config_dir: HF config.json + tokenizer
    ck.f32("model.layers.0.self_attn.q_proj.weight")    # [out, in] float32
    ck.expert(0, "gate", 5)                             # [768, 2048] float32
"""
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader, dequantize

_LAYER = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",            # MoE router
    "ffn_gate_exps": "mlp.experts.gate",   # stacked [E, out, in]
    "ffn_up_exps": "mlp.experts.up",
    "ffn_down_exps": "mlp.experts.down",
}
_GLOBAL = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}


class GGUFCheckpoint:
    def __init__(self, gguf_path, config_dir):
        self.path = Path(gguf_path)
        self.cfg = json.loads((Path(config_dir) / "config.json").read_text())
        self.r = GGUFReader(str(gguf_path))
        self._t = {}
        for t in self.r.tensors:
            if t.name in _GLOBAL:
                self._t[_GLOBAL[t.name]] = t
                continue
            m = re.match(r"blk\.(\d+)\.(\w+)\.(weight|bias)$", t.name)
            if m and m.group(2) in _LAYER:
                self._t[f"model.layers.{m.group(1)}.{_LAYER[m.group(2)]}.{m.group(3)}"] = t

    def __contains__(self, name):
        return name in self._t

    def _deq(self, t, data=None):
        return np.asarray(dequantize(t.data if data is None else data, t.tensor_type), np.float32)

    def f32(self, name):
        t = self._t[name]
        a = self._deq(t)
        shape = tuple(int(s) for s in reversed(t.shape))       # GGUF ne is innermost-first
        return a.reshape(shape)

    def raw(self, name):
        """bf16-compatible float32 view; rows can be gathered lazily for embeddings."""
        if name == "model.embed_tokens.weight":
            return _RowView(self, self._t[name])
        return self.f32(name)

    def expert(self, layer, kind, e):
        """One expert matrix [out, in] (kind: gate|up|down) without dequantizing the rest."""
        t = self._t[f"model.layers.{layer}.mlp.experts.{kind}.weight"]
        n_e = int(t.shape[-1])
        rows = t.data.reshape(n_e, -1, t.data.shape[-1])[e]     # data is [E*out, bytes_per_row]
        out = self._deq(t, rows)
        return out.reshape(int(t.shape[1]), int(t.shape[0]))


class _RowView:
    """Embedding table accessed one row at a time (dequantize only the requested rows)."""

    def __init__(self, ck, t):
        self.ck, self.t = ck, t
        self.shape = (int(t.shape[1]), int(t.shape[0]))

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.asarray(idx))
        rows = self.t.data.reshape(self.shape[0], -1)[idx]
        out = self.ck._deq(self.t, rows).reshape(len(idx), self.shape[1])
        return out[0] if out.shape[0] == 1 and np.ndim(idx) == 1 and len(idx) == 1 else out
