"""OpenNPU native engine: any supported dense HF checkpoint on the MLIR-AIR fused decode.

One NPU dispatch per token runs every decoder layer plus the LM head. This module is
the model-agnostic host side:

  HFCheckpoint   memory-maps the safetensors shards (no torch, no RAM copy)
  build_weights  bf16 -> 4-bit Q4NX (per-32 affine) in the decode's cascade order,
                 streamed one layer at a time into a raw cache file
  NativeDecoder  uploads the weights once and runs one token per dispatch

Prompts go through the batched NPU prefill (prefill.py) when it is built for the model;
otherwise they feed through the same decode dispatch, one token at a time, and the KV cache
is written by the device as it goes.

Run inside the MLIR-AIR environment (pyxrt + mlir-air; see opennpu.config.AIRENV).
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from opennpu import config

AIR = config.AIR
DEC = AIR / "programming_examples" / "fused_decode"
CACHE = Path(os.environ.get("OPENNPU_NATIVE_CACHE", config.NATIVE_CACHE))

# HF model_type -> (fused_decode table entry, kernel MODEL_TYPE macro) per hidden size.
# The table entry carries the device geometry; specgen derives it from config.json.
FAMILY = {"qwen3": "qwen3", "qwen2": "qwen2", "llama": "llama", "mistral": "llama"}


# ----------------------------------------------------------------------------- weights
class HFCheckpoint:
    """Zero-copy view of a safetensors checkpoint directory."""

    _DT = {"BF16": bfloat16, "F16": np.float16, "F32": np.float32}

    def __init__(self, path):
        self.path = Path(path)
        self.cfg = json.loads((self.path / "config.json").read_text())
        idx = self.path / "model.safetensors.index.json"
        files = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
                 if idx.exists() else ["model.safetensors"])
        self._t = {}
        for f in files:
            fp = self.path / f
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            mm = np.memmap(fp, dtype=np.uint8, mode="r")
            for name, m in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = m["data_offsets"]
                self._t[name] = (mm, 8 + n + a, 8 + n + b, self._DT[m["dtype"]], m["shape"])

    def __contains__(self, name):
        return name in self._t

    def raw(self, name):
        mm, a, b, dt, shape = self._t[name]
        return mm[a:b].view(dt).reshape(shape)

    def f32(self, name):
        return np.asarray(self.raw(name), np.float32)


def _requant_q4k(W, group=32):
    """Per-32-column affine 4-bit: w ~= q*scale + min (the Q4NX device codec)."""
    M, K = W.shape
    Wg = W.reshape(M, K // group, group)
    mn = Wg.min(2)
    sc = (Wg.max(2) - mn) / 15.0
    sc = np.where(sc <= 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.rint((Wg - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8)
    return q.reshape(M, K), sc, mn.astype(np.float32)


def _interleave(up, gate, chunk):
    n = up.shape[0] // chunk
    return np.concatenate([(up if h == 0 else gate)[s * chunk:(s + 1) * chunk]
                           for s in range(n) for h in (0, 1)])


# ----------------------------------------------------------------------------- engine
def load_engine(decode_model, attn_maxl):
    """Import fused_decode.py configured for `decode_model` (module constants read env)."""
    import importlib.util
    os.environ.update(DECODE_MODEL=decode_model, UNIFIED="1", LM_HEAD="0", NLAYERS="1",
                      DECODE_GOLDEN="1", DECODE_GOLDEN_L=str(attn_maxl))
    if str(DEC) not in sys.path:
        sys.path.insert(0, str(DEC))
    spec = importlib.util.spec_from_file_location(f"fd_{decode_model}", str(DEC / "fused_decode.py"))
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def build_weights(ck, fd, out, verbose=True):
    """Stream the checkpoint into the decode's weight image + norm stacks.

    W (int16 view of bf16 blocks) goes to `out`.W.bin layer by layer; norms/biases are
    tiny and go to `out`.aux.npz."""
    from q4_0_codec import pack_q4k_cascade_fast as pack

    NCX, NCY, NPH = fd.NCX, fd.NCY, fd.NPH
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    cfg = ck.cfg
    L, K = fd.UNI_DEC, fd.K
    assert cfg["intermediate_size"] % 512 == 0, "INTER padding not implemented yet"
    pre = "model.layers"
    tmp = Path(str(out) + ".W.bin.part")
    rms_in, rms_post = [], []
    t0 = time.time()

    def packed(W):
        q, s, m = _requant_q4k(W)
        return q, s, m

    fold = bool(getattr(fd, "NORM_FOLD", False))
    with open(tmp, "wb") as f:
        for k in range(L):
            g = lambda n: ck.f32(f"{pre}.{k}.{n}.weight")  # noqa: E731
            n_in = ck.f32(f"{pre}.{k}.input_layernorm.weight")
            n_post = ck.f32(f"{pre}.{k}.post_attention_layernorm.weight")
            # NORM_FOLD builds run a weightless RMSNorm on device: fold each norm
            # weight into the input columns of the projections that read it.
            fi = n_in[None, :] if fold else 1.0
            fp = n_post[None, :] if fold else 1.0
            qkv = np.concatenate([g("self_attn.q_proj"), g("self_attn.k_proj"), g("self_attn.v_proj")]) * fi
            ph = [None] * NPH
            ph[0] = packed(qkv)
            ph[OP] = packed(g("self_attn.o_proj"))
            up, gate = packed(g("mlp.up_proj") * fp), packed(g("mlp.gate_proj") * fp)
            ph[GP] = tuple(_interleave(up[i], gate[i], fd.GLU_CHUNK) for i in range(3))
            ph[DP] = packed(g("mlp.down_proj"))
            slab = np.concatenate([pack(q, s, NCX, NCY, dual_chan=DUAL, mins=m) for q, s, m in ph])
            assert slab.size == fd.W_LAYER, (slab.size, fd.W_LAYER)
            slab.astype(np.int16).tofile(f)
            rms_in.append(np.asarray(ck.raw(f"{pre}.{k}.input_layernorm.weight"), bfloat16))
            rms_post.append(np.asarray(ck.raw(f"{pre}.{k}.post_attention_layernorm.weight"), bfloat16))
            if verbose:
                el = time.time() - t0
                print(f"\r[opennpu] quantize layer {k + 1}/{L}  ({el:.0f}s, ~{el / (k + 1) * (L - k - 1):.0f}s left)",
                      end="", flush=True)
        # LM head, padded to the full vocab grid, one cascade per vocab wave.
        lm_name = "lm_head.weight" if "lm_head.weight" in ck else "model.embed_tokens.weight"
        fn = ck.f32("model.norm.weight")[None, :] if fold else 1.0
        # the checkpoint's own vocab (fine-tunes may add tokens inside the padded grid)
        V, VP, VPF = ck.raw(lm_name).shape[0], fd.VOCAB_SIZE_PADDED, fd.VOCAB_SIZE_PADDED_FULL
        assert V <= VPF, (V, VPF)
        for w in range(fd.UNI_LM):
            rows = np.zeros((VP, K), np.float32)
            lo, hi = w * VP, min((w + 1) * VP, V)
            if hi > lo:
                rows[:hi - lo] = ck.raw(lm_name)[lo:hi].astype(np.float32) * fn
            q, s, m = _requant_q4k(rows)
            pack(q, s, NCX, NCY, dual_chan=DUAL, mins=m).astype(np.int16).tofile(f)
    tmp.rename(str(out) + ".W.bin")
    np.savez(str(out) + ".aux.npz", RMS_in=np.stack(rms_in).view(np.int16),
             RMS_post=np.stack(rms_post).view(np.int16))
    if verbose:
        print(f"\n[opennpu] weights -> {out}.W.bin ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _inv_freq(cfg, dh):
    theta = float(cfg.get("rope_theta", 10000.0))
    half = dh // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type", rs.get("type")) == "llama3":
        f, lo, hi = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
        old = rs["original_max_position_embeddings"]
        wl = 2 * np.pi / inv
        smooth = (old / wl - lo) / (hi - lo)
        inv = np.where(wl > old / lo, inv / f,
                       np.where(wl < old / hi, inv, (1 - smooth) * inv / f + smooth * inv))
    elif rs.get("rope_type", rs.get("type")) == "linear":
        inv = inv / rs["factor"]
    return inv


class NativeDecoder:
    """One fused-decode xclbin + the model's weights resident on the NPU."""

    def __init__(self, ckpt_dir, decode_model, template_dir, max_L=None, verbose=True):
        import pyxrt as xrt
        sys.path.insert(0, str(DEC))
        import decode_staircase as stair
        from decode_insts_gen import DecodeInstsGen

        self.xrt, self.stair = xrt, stair
        self.ck = HFCheckpoint(ckpt_dir)
        cfg = self.ck.cfg
        self.gen = DecodeInstsGen(str(template_dir), max_L=max_L)
        self.windows = stair.resolve_windows(self.gen)
        self.ATTN_MAXL = max(self.windows)
        fd = self.fd = load_engine(decode_model, self.ATTN_MAXL)
        self.family = FAMILY[cfg["model_type"]]
        self.K, self.DH, self.L = fd.K, fd.DH, fd.UNI_DEC
        lmn = "lm_head.weight" if "lm_head.weight" in self.ck else "model.embed_tokens.weight"
        self.VOCAB = self.ck.raw(lmn).shape[0]

        # weights: quantize once per (checkpoint, engine geometry)
        tag = (f"{Path(ckpt_dir).name}-{fd.MODEL_CONFIG_STAMP.split('MODEL_SHA=')[-1]}"
               + ("-fold" if getattr(fd, "NORM_FOLD", False) else ""))
        self.cache = CACHE / tag
        CACHE.mkdir(parents=True, exist_ok=True)
        if not Path(str(self.cache) + ".W.bin").exists():
            build_weights(self.ck, fd, self.cache, verbose)
        W = np.memmap(str(self.cache) + ".W.bin", dtype=np.int16, mode="r")
        aux = np.load(str(self.cache) + ".aux.npz")
        rin, rpost = aux["RMS_in"].view(bfloat16), aux["RMS_post"].view(bfloat16)
        rms_slabs = np.concatenate([np.concatenate([rin[k], rpost[k]]) for k in range(self.L)])

        # per-layer rope payload pieces (qk-norm or qkv-bias), host-side
        pre = "model.layers"
        if fd.HAS_QK_NORM:
            self.extra = [np.concatenate([self.ck.f32(f"{pre}.{k}.self_attn.q_norm.weight"),
                                          self.ck.f32(f"{pre}.{k}.self_attn.k_norm.weight")])
                          for k in range(self.L)]
        elif fd.HAS_QKV_BIAS:
            self.extra = [np.concatenate([self.ck.f32(f"{pre}.{k}.self_attn.{n}_proj.bias")
                                          for n in ("q", "k", "v")]) for k in range(self.L)]
        else:
            self.extra = None
        self.per_layer = bool(fd.ROPE_W_PER_LAYER)
        self.inv = _inv_freq(cfg, self.DH)
        self.embed = self.ck.raw("model.embed_tokens.weight")  # memmap [V, K] bf16
        final_norm = np.asarray(self.ck.raw("model.norm.weight"), bfloat16)

        # device buffers (layout from the engine; see fused_decode.py rms_t / y)
        self.ROPE_W_LEN, self.RMS_LAYER = fd.ROPE_W_LEN, fd.RMS_LAYER
        n_rope = (self.L if self.per_layer else 1) * self.ROPE_W_LEN
        self._rope_base = self.L * self.RMS_LAYER
        RMS_SIZE = self._rope_base + n_rope + self.K
        self.KVSZ_TOK, self.REGION_W, self.NGRP = fd.KVSZ_TOK, fd.REGION_W, fd.NGRP
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.VP, self.UNI_LM = fd.VOCAB_SIZE_PADDED, fd.UNI_LM
        ny = self.decode_y + self.UNI_LM * self.VP

        dev = self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._kern = stair.open_windows(dev, xrt, self.gen, self.windows)
        self.cur = self.ATTN_MAXL
        kern = self._kern[self.cur][1]
        g = kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(dev, self.K * 2, HO, g(3))
        WL, G = fd.W_LAYER, fd.W_GROUP
        if fd.W_SPLIT:
            parts = [(gi * G * WL, min((gi + 1) * G, self.L) * WL) for gi in range(fd.N_WGRP)]
            parts.append((self.L * WL, W.size))
        else:
            parts = [(0, W.size)]
        self.w_bos = []
        t0 = time.time()
        for i, (a, b) in enumerate(parts):
            bo = xrt.bo(dev, (b - a) * 2, HO, g(4 if i == 0 else 7 + i))
            mv = np.frombuffer(bo.map(), dtype=np.int16, count=b - a)
            step = 1 << 26
            for o in range(a, b, step):             # stream from the page cache, no RAM copy
                mv[o - a:min(o + step, b) - a] = W[o:min(o + step, b)]
            bo.sync(TO)
            self.w_bos.append(bo)
        if verbose:
            print(f"[opennpu] {W.size * 2 / 2**30:.2f} GiB of weights resident in {time.time() - t0:.1f}s "
                  f"({len(parts)} BOs), ATTN_MAXL={self.ATTN_MAXL}", flush=True)
        self.r_bo = xrt.bo(dev, RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(dev, ny * 2, HO, g(6))
        self.kvc = xrt.bo(dev, self.L * self.LREG * 2, HO, g(7))
        self._ist = stair.make_insts_states(self.gen, xrt, dev, g(1), self.windows)
        self._st = self._ist[self.cur]
        self.kern, self.ib = kern, self._st["ib"]
        rbuf = np.concatenate([rms_slabs, np.zeros(n_rope, bfloat16), final_norm])
        assert rbuf.size == RMS_SIZE, (rbuf.size, RMS_SIZE)
        self.r_bo.write(rbuf.view(np.int16), 0)
        self.r_bo.sync(TO)
        self.reset()

    # ------------------------------------------------------------------ per token
    def reset(self):
        """Empty the KV cache (new conversation)."""
        z = np.zeros(self.L * self.LREG, np.int16)
        self.kvc.write(z, 0)
        self.kvc.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.pos = 0

    def _rope(self, p):
        ang = p * self.inv
        lut = np.concatenate([np.cos(ang), np.sin(ang)]).astype(bfloat16).astype(np.float32)
        if not self.per_layer:
            return lut.astype(bfloat16)
        return np.concatenate([np.concatenate([lut, e]) for e in self.extra]).astype(bfloat16)

    def step(self, tok):
        """Feed one token at the next position; return the next-token logits (f32)."""
        xrt, np_ = self.xrt, np
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        p = self.pos
        if p >= self.ATTN_MAXL:
            raise RuntimeError(f"context full ({self.ATTN_MAXL})")
        n = self.stair.patch_insts(self._st, p + 1, xrt, TO)
        rope = self._rope(p)
        self.r_bo.write(rope.view(np_.int16), self._rope_base * 2)
        self.r_bo.sync(TO, rope.size * 2, self._rope_base * 2)
        self.x_bo.write(np_.asarray(self.embed[tok], bfloat16).view(np_.int16), 0)
        self.x_bo.sync(TO)
        st = self.kern(3, self.ib, n, self.x_bo, self.w_bos[0], self.r_bo, self.y_bo, self.kvc,
                       *self.w_bos[1:]).wait(60000)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        nv = self.UNI_LM * self.VP
        self.y_bo.sync(FROM, nv * 2, self.decode_y * 2)
        y = np_.frombuffer(self.y_bo.map(), dtype=bfloat16, count=nv, offset=self.decode_y * 2)
        self.pos = p + 1
        return y[:self.VOCAB].astype(np_.float32)

    def generate(self, ids, n_new, eos=(), on_token=None, temperature=0.0, seed=0, logits=None):
        """Prompt `ids` -> up to n_new tokens. Returns (new_ids, prefill_s, decode_s).
        With `logits`, the prompt is already in the KV cache (batched prefill) and these are its
        last token's logits; `ids` is ignored."""
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        if logits is not None:
            lg = logits
        else:
            for t in ids[:-1]:
                self.step(t)
            lg = self.step(ids[-1])
        t1 = time.perf_counter()
        out = []
        for _ in range(n_new):
            if temperature > 0:
                z = lg / temperature
                pr = np.exp(z - z.max())
                tok = int(rng.choice(len(pr), p=pr / pr.sum()))
            else:
                tok = int(lg.argmax())
            if tok in eos:
                break
            out.append(tok)
            if on_token:
                on_token(tok)
            if len(out) == n_new or self.pos >= self.ATTN_MAXL:
                break
            lg = self.step(tok)
        return out, t1 - t0, time.perf_counter() - t1
