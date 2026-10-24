"""CPU reference forward for the native engine, streamed one layer at a time.

`quant=True` uses the same per-32 affine 4-bit weights the NPU runs (so any gap to the
device is the device's), `quant=False` the original bf16 checkpoint.

    python -m opennpu.native.reference <ckpt_dir> <id> <id> ...
"""
import sys

import numpy as np
import torch

from opennpu.native.runtime import HFCheckpoint, _inv_freq, _requant_q4k


def _w(ck, name, quant):
    W = ck.f32(name)
    if quant and W.ndim == 2 and W.shape[1] % 32 == 0:
        q, s, m = _requant_q4k(W)
        W = (q.reshape(W.shape[0], -1, 32) * s[..., None] + m[..., None]).reshape(W.shape)
    return torch.from_numpy(np.ascontiguousarray(W, np.float32))


def _wq(W, quant):
    if quant:
        q, s, m = _requant_q4k(W)
        W = (q.reshape(W.shape[0], -1, 32) * s[..., None] + m[..., None]).reshape(W.shape)
    return torch.from_numpy(np.ascontiguousarray(W, np.float32))


def forward(ckpt, ids, quant=True, layers=None, ffn=True, attn=True):
    ck = ckpt if hasattr(ckpt, "f32") else HFCheckpoint(ckpt)
    c = ck.cfg
    H, KV = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    DH = c.get("head_dim") or c["hidden_size"] // H
    eps = float(c.get("rms_norm_eps", 1e-6))
    qkn = "model.layers.0.self_attn.q_norm.weight" in ck
    P = len(ids)
    rms = lambda x, w: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w  # noqa: E731
    t = lambda n: torch.from_numpy(ck.f32(n))  # noqa: E731
    x = torch.from_numpy(np.asarray(ck.raw("model.embed_tokens.weight")[ids], np.float32))
    inv = torch.from_numpy(_inv_freq(c, DH)).float()
    ang = torch.arange(P)[:, None].float() * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]

    def rope(v):
        a, b = v[..., :DH // 2], v[..., DH // 2:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)

    mask = torch.full((P, P), float("-inf")).triu(1)
    for k in range(layers or c["num_hidden_layers"]):
        k_l = k
        p = f"model.layers.{k}."
        h = rms(x, t(p + "input_layernorm.weight"))
        q = h @ _w(ck, p + "self_attn.q_proj.weight", quant).T
        kk = h @ _w(ck, p + "self_attn.k_proj.weight", quant).T
        v = h @ _w(ck, p + "self_attn.v_proj.weight", quant).T
        if p + "self_attn.q_proj.bias" in ck:
            q, kk, v = q + t(p + "self_attn.q_proj.bias"), kk + t(p + "self_attn.k_proj.bias"), v + t(p + "self_attn.v_proj.bias")
        q, kk, v = q.view(P, H, DH), kk.view(P, KV, DH), v.view(P, KV, DH)
        if qkn:
            q, kk = rms(q, t(p + "self_attn.q_norm.weight")), rms(kk, t(p + "self_attn.k_norm.weight"))
        q, kk = rope(q), rope(kk)
        kk, v = kk.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        a = torch.einsum("phd,qhd->hpq", q, kk) / DH ** 0.5 + mask
        o = torch.einsum("hpq,qhd->phd", a.softmax(-1), v).reshape(P, H * DH)
        x = x + (o @ _w(ck, p + "self_attn.o_proj.weight", quant).T if attn else 0)
        h = rms(x, t(p + "post_attention_layernorm.weight"))
        if not ffn:
            continue
        if p + "mlp.gate.weight" in ck:            # Mixture of experts (qwen3_moe)
            logits = h @ t(p + "mlp.gate.weight").T                       # [P, E]
            prob = logits.softmax(-1)
            w, sel = prob.topk(c["num_experts_per_tok"], -1)
            if c.get("norm_topk_prob", True):
                w = w / w.sum(-1, keepdim=True)
            out = torch.zeros_like(x)
            for e in sel.unique().tolist():
                rows = (sel == e).any(-1).nonzero().flatten()
                we = (w * (sel == e)).sum(-1)[rows, None]
                ew = {k: _wq(ck.expert(k_l, k, e), quant) for k in ("gate", "up", "down")}
                hh = h[rows]
                out[rows] += we * ((torch.nn.functional.silu(hh @ ew["gate"].T) * (hh @ ew["up"].T)) @ ew["down"].T)
            x = x + out
        else:
            g = h @ _w(ck, p + "mlp.gate_proj.weight", quant).T
            u = h @ _w(ck, p + "mlp.up_proj.weight", quant).T
            x = x + (torch.nn.functional.silu(g) * u) @ _w(ck, p + "mlp.down_proj.weight", quant).T
    x = rms(x, t("model.norm.weight"))
    lm = "lm_head.weight" if "lm_head.weight" in ck else "model.embed_tokens.weight"
    return (x @ _w(ck, lm, quant).T).numpy()


if __name__ == "__main__":
    torch.set_num_threads(8)
    ids = [int(a) for a in sys.argv[2:]]
    lg = forward(sys.argv[1], ids)
    print("argmax per position:", lg.argmax(-1).tolist())
