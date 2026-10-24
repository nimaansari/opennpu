"""Mixture-of-experts decode on the NPU (own fused engine).

Qwen3-MoE (qwen3_moe, e.g. Qwen3-30B-A3B) on the rotated per-layer program built from
fused_decode.py's "qwen3-30b-a3b" entry (see MOE_DESIGN_fused_decode.md):

  dispatch k = [ gate-up / glu*w / down of layer k-1's 8 routed experts -> residual ]
               [ input-norm -> QKV(k) -> rope / attention -> o(k) -> residual ]
  host       : router of layer k on the post-attention hidden state -> top-8 -> bind
               those 8 expert sub-buffers (zero copy) + their weights for dispatch k+1
  k = 0  runs the FFN half with routing weights 0;  k = L uses an all-zero attention
  slab, so it is FFN-only;  then one LM-head dispatch (same xclbin, separate stream).

Weights are re-quantized to the engine's Q4NX once into attn.bin
[attn_0 .. attn_{L-1} | zero_attn | vocab chunks] and experts_L{l}.bin per layer
(E slabs of [gate-up cascade | down cascade]).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from opennpu.native.runtime import CACHE, DEC, _inv_freq, _interleave, _requant_q4k, load_engine

DECODE_MODEL = "qwen3-30b-a3b"


def _pack(fd, W):
    from q4_0_codec import pack_q4k_cascade_fast as pack
    q, s, m = _requant_q4k(np.ascontiguousarray(W, np.float32))
    return pack(q, s, fd.NCX, fd.NCY, dual_chan=True, mins=m)


def _expert_slab(fd, gate, up, down):
    """One expert -> [gate-up cascade | down cascade], packed like a dense phase."""
    gu = _interleave(up, gate, fd.GLU_CHUNK)        # [2*EI, K], U/G interleaved per chunk
    return np.concatenate([_pack(fd, gu), _pack(fd, down)])


def build_weights(ck, fd, out, verbose=True):
    """Re-quantize the checkpoint into the attn image + per-layer expert files + aux."""
    L, K, NL = fd.UNI_DEC, fd.K, ck.cfg["num_hidden_layers"]
    assert fd.UNI_DEC == 1
    pre = "model.layers"
    t0 = time.time()
    rms_in, rms_post, qn, kn, routers = [], [], [], [], []
    with open(str(out) + ".attn.bin", "wb") as f:
        for k in range(NL):
            g = lambda n: ck.f32(f"{pre}.{k}.{n}.weight")  # noqa: E731
            qkv = np.concatenate([g("self_attn.q_proj"), g("self_attn.k_proj"), g("self_attn.v_proj")])
            slab = np.concatenate([_pack(fd, qkv), _pack(fd, g("self_attn.o_proj"))])
            assert slab.size == fd.W_LAYER, (slab.size, fd.W_LAYER)
            slab.astype(np.int16).tofile(f)
            rms_in.append(np.asarray(ck.raw(f"{pre}.{k}.input_layernorm.weight"), bfloat16))
            rms_post.append(np.asarray(ck.raw(f"{pre}.{k}.post_attention_layernorm.weight"), bfloat16))
            qn.append(np.asarray(ck.f32(f"{pre}.{k}.self_attn.q_norm.weight"), bfloat16))
            kn.append(np.asarray(ck.f32(f"{pre}.{k}.self_attn.k_norm.weight"), bfloat16))
            routers.append(ck.f32(f"{pre}.{k}.mlp.gate.weight").astype(np.float32))
            if verbose:
                print(f"\r[moe] attn+router layer {k + 1}/{NL}", end="", flush=True)
        # zero-attention slab (k=L FFN-only dispatch): all weights zero -> o == 0
        np.zeros(fd.W_LAYER, np.int16).tofile(f)
        # vocab chunks for the LM-head stream
        lm = "lm_head.weight" if "lm_head.weight" in ck else "model.embed_tokens.weight"
        V, VP = fd.VOCAB_SIZE, fd.VOCAB_SIZE_PADDED
        for w in range(fd.UNI_LM):
            rows = np.zeros((VP, K), np.float32)
            lo, hi = w * VP, min((w + 1) * VP, V)
            if hi > lo:
                rows[:hi - lo] = ck.raw(lm)[lo:hi].astype(np.float32)
            _pack(fd, rows).astype(np.int16).tofile(f)
    if verbose:
        print(f"\n[moe] experts ({NL} layers x {ck.cfg['num_experts']}) ...", flush=True)
    for k in range(NL):
        with open(str(out) + f".exp.L{k}.bin", "wb") as f:
            for e in range(ck.cfg["num_experts"]):
                _expert_slab(fd, ck.expert(k, "gate", e), ck.expert(k, "up", e),
                             ck.expert(k, "down", e)).astype(np.int16).tofile(f)
        if verbose:
            el = time.time() - t0
            print(f"\r[moe] experts layer {k + 1}/{NL}  ({el:.0f}s)", end="", flush=True)
    np.savez(str(out) + ".aux.npz", rms_in=np.stack(rms_in).view(np.int16),
             rms_post=np.stack(rms_post).view(np.int16), qn=np.stack(qn).view(np.int16),
             kn=np.stack(kn).view(np.int16), final_norm=np.asarray(ck.raw("model.norm.weight"), bfloat16).view(np.int16),
             routers=np.stack(routers))
    if verbose:
        print(f"\n[moe] weights -> {out}.* ({(time.time() - t0) / 60:.1f} min)", flush=True)


class MoeDecoder:
    def __init__(self, ckpt_dir, max_L=None, verbose=True):
        import pyxrt as xrt
        sys.path.insert(0, str(DEC))
        import decode_staircase as stair
        from decode_insts_gen import DecodeInstsGen
        from opennpu.native.gguf_ckpt import GGUFCheckpoint
        from opennpu.native.runtime import HFCheckpoint

        self.xrt, self.stair = xrt, stair
        d = Path(ckpt_dir)
        gg = d / "model.gguf"
        self.ck = GGUFCheckpoint(gg, d) if gg.exists() else HFCheckpoint(d)
        cfg = self.ck.cfg
        self.NL = cfg["num_hidden_layers"]
        self.E = cfg["num_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.norm_topk = cfg.get("norm_topk_prob", True)

        # templates: layer stream (L-patched) + LM stream (static), from one xclbin
        tdir = Path(os.environ["OPENNPU_MOE_TEMPLATES"])
        self.gen = DecodeInstsGen(str(tdir / "layer"))
        self.windows = stair.resolve_windows(self.gen)
        self.ATTN_MAXL = max(self.windows)
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL
        fd = self.fd = load_engine(DECODE_MODEL, self.ATTN_MAXL)
        self.K, self.DH, self.VOCAB = fd.K, fd.DH, fd.VOCAB_SIZE
        self.inv = _inv_freq(cfg, self.DH)

        tag = Path(ckpt_dir).name + "-moe"
        self.cache = CACHE / tag
        CACHE.mkdir(parents=True, exist_ok=True)
        if not Path(str(self.cache) + ".aux.npz").exists():
            build_weights(self.ck, fd, self.cache, verbose)
        aux = np.load(str(self.cache) + ".aux.npz")
        self.rms_in = aux["rms_in"].view(bfloat16)
        self.rms_post = aux["rms_post"].view(bfloat16)
        self.qn, self.kn = aux["qn"].view(bfloat16), aux["kn"].view(bfloat16)
        self.final_norm = aux["final_norm"].view(bfloat16)
        self.routers = aux["routers"]              # [NL, E, K] f32
        # LM head on CPU (correct + tiny): the NPU vocab dispatch's geometry is tied to the
        # pre-rotation phase order and is still wrong (logit cosine 0.82); the 48 MoE layers
        # stay on the NPU. TODO: fix the NPU vocab stream.
        # fp16 via torch: half the bytes of f32 -> ~15 ms/token instead of ~28 (bandwidth-bound).
        lmn = "lm_head.weight" if "lm_head.weight" in self.ck else "model.embed_tokens.weight"
        lm = self.ck.f32(lmn)                                                  # [V, K]
        # int8 weights (per-row scales), fp32 activations: 1/4 the bytes of f32, ~5.6 ms/token
        # on 8 Zen5 cores vs ~15.6 for fp16; top-1 unchanged (logit cosine ~0.9998).
        self._lm_q = self._lm_t = self._lm = None
        try:
            import torch
            try:
                import warnings
                from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic
                lin = torch.nn.Linear(lm.shape[1], lm.shape[0], bias=False)
                lin.weight.data = torch.from_numpy(np.ascontiguousarray(lm, np.float32))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._lm_q = quantize_dynamic(torch.nn.Sequential(lin), {torch.nn.Linear: per_channel_dynamic_qconfig})
                del lin
            except Exception:  # noqa: BLE001  (no quantized engine in this torch build)
                self._lm_t = torch.from_numpy(np.asarray(lm, np.float16))
        except ImportError:
            self._lm = np.ascontiguousarray(lm, dtype=np.float32)
        del lm
        self._fnorm = np.asarray(self.final_norm, bfloat16).astype(np.float32)

        # device buffers
        dev = self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self._TO = TO
        self._kl = stair.open_windows(dev, xrt, self.gen, self.windows)
        self.cur = self.ATTN_MAXL
        self.kern = self._kl[self.cur][1]
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.RMS_LAYER, self.ROPE_W_LEN = fd.RMS_LAYER, fd.ROPE_W_LEN
        self._rope_base = fd.RMS_LAYER
        self._final_off = self._rope_base + self.ROPE_W_LEN
        self._rw_off = self._final_off + self.K
        self.RMS_SIZE = self._rw_off + 16
        self.W_LAYER = fd.W_LAYER
        self.KVSZ_TOK, self.REGION_W, self.NGRP = fd.KVSZ_TOK, fd.REGION_W, fd.NGRP
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK
        self.EXP_LEN = fd.EXP_LEN
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.VP, self.UNI_LM = fd.VOCAB_SIZE_PADDED, fd.UNI_LM
        ny = self.decode_y + self.UNI_LM * self.VP

        # x, y, rms (per dispatch), KV parent (NL layers + 1 zero scratch)
        self.x_bo = xrt.bo(dev, self.K * 2, HO, g(3))
        self.y_bo = xrt.bo(dev, ny * 2, HO, g(6))
        self.r_bo = xrt.bo(dev, self.RMS_SIZE * 2, HO, g(5))
        self.kv_parent = xrt.bo(dev, (self.NL + 1) * self.LREG * 2, HO, g(7))
        np.frombuffer(self.kv_parent.map(), np.uint8)[:] = 0
        self.kv_parent.sync(TO)
        self.kv = [xrt.bo(self.kv_parent, self.LREG * 2, k * self.LREG * 2) for k in range(self.NL + 1)]

        # attention parent: [attn_0..attn_{NL-1} | zero_attn | vocab]
        attn = np.memmap(str(self.cache) + ".attn.bin", dtype=np.int16, mode="r")
        self.attn_parent = xrt.bo(dev, attn.size * 2, HO, g(4))
        mv = np.frombuffer(self.attn_parent.map(), np.int16, count=attn.size)
        step = 1 << 25
        for o in range(0, attn.size, step):
            mv[o:o + step] = attn[o:o + step]
        self.attn_parent.sync(TO)
        self.attn = [xrt.bo(self.attn_parent, self.W_LAYER * 2, k * self.W_LAYER * 2) for k in range(self.NL)]
        self.zero_attn = xrt.bo(self.attn_parent, self.W_LAYER * 2, self.NL * self.W_LAYER * 2)

        # experts: one parent per layer, E sub-buffers each
        self.exp_parent, self.E_bo, self._bomap = [], [], []
        for k in range(self.NL):
            em = np.memmap(str(self.cache) + f".exp.L{k}.bin", dtype=np.int16, mode="r")
            bo = xrt.bo(dev, em.size * 2, HO, g(9 + 0))
            dmv = np.frombuffer(bo.map(), np.int16, count=em.size)
            for o in range(0, em.size, step):
                dmv[o:o + step] = em[o:o + step]
            bo.sync(TO)
            self.exp_parent.append(bo)
            self.E_bo.append([xrt.bo(bo, self.EXP_LEN * 2, e * self.EXP_LEN * 2) for e in range(self.E)])
            self._bomap.append(np.frombuffer(bo.map(), bfloat16, count=em.size))
        # down staging (one dispatch's 8 experts, dense [cx][rh][i][j=e*3+je][cy] order)
        self.DSTAGE_LEN = self.EXP_LEN  # EXP_DN*topk == per-expert EXP_LEN here (EXP_DN*8)
        self.dstage = xrt.bo(dev, fd.EXP_DN * self.topk * 2, HO, g(9))
        self._dmv = np.frombuffer(self.dstage.map(), bfloat16, count=fd.EXP_DN * self.topk)

        self.EXP_GU, self.EXP_DN = fd.EXP_GU, fd.EXP_DN
        BLK = fd.BLOCK_BF16
        # host view of every expert's packed down, shaped [E, cx,rh,i, je, cy, blk]
        self._G = fd.NCX * 2 * (fd.I2P[1] * 2)          # 32 (cx,rh,i) groups
        self._IN = (fd.MOE_EI // 256) * 2 * BLK           # 15360 (je,cy,blk)
        # zero-copy down views into the already-resident expert BOs (no duplicate RAM)
        self._down = [m.reshape(self.E, self.EXP_LEN)[:, self.EXP_GU:].reshape(self.E, self._G, self._IN)
                      for m in self._bomap]
        # gather straight into the staging BO, one thread per expert slot: the copy is bound
        # by cold reads of the 17.6 GB expert set, so parallel reads help and a scratch hop doesn't
        self._dview = self._dmv.reshape(self._G, self.topk, self._IN)
        self._pool = ThreadPoolExecutor(self.topk)
        if verbose:
            gb = (attn.size * 2 + sum(b.size() for b in self.exp_parent)) / 2**30
            print(f"[moe] {gb:.1f} GiB resident on the NPU, ATTN_MAXL={self.ATTN_MAXL}", flush=True)

        self._ist = stair.make_insts_states(self.gen, xrt, dev, g(1), self.windows)
        self._st = self._ist[self.cur]
        self.ib = self._st["ib"]
        # LM-head stream shares the xclbin; its own insts + a run
        self.lm_gen = DecodeInstsGen(str(tdir / "lm"))
        self._lst = stair.make_insts_states(self.lm_gen, xrt, dev, g(1), [self.ATTN_MAXL])[self.ATTN_MAXL]
        self.reset()

    def reset(self):
        np.frombuffer(self.kv_parent.map(), np.uint8)[:] = 0
        self.kv_parent.sync(self._TO)
        self.pos = 0
        self._sel = None  # (experts, weights) for the next dispatch's FFN half

    def _write_rms(self, k, p):
        """rms slab for dispatch k at position p: [post_ln(k-1)|in_ln(k)|cos/sin|qn|kn|final|rw]."""
        r = np.zeros(self.RMS_SIZE, bfloat16)
        post = self.rms_post[k - 1] if k >= 1 else np.zeros(self.K, bfloat16)
        inln = self.rms_in[k] if k < self.NL else np.zeros(self.K, bfloat16)
        r[:self.K] = post
        r[self.K:2 * self.K] = inln
        ang = p * self.inv
        lut = np.concatenate([np.cos(ang), np.sin(ang)]).astype(bfloat16)
        qn = self.qn[k] if k < self.NL else np.zeros(self.DH, bfloat16)
        kn = self.kn[k] if k < self.NL else np.zeros(self.DH, bfloat16)
        r[self._rope_base:self._rope_base + self.ROPE_W_LEN] = np.concatenate([lut, qn, kn])
        r[self._final_off:self._final_off + self.K] = self.final_norm
        if self._sel is not None:
            r[self._rw_off:self._rw_off + self.topk] = np.asarray(self._sel[1], bfloat16)
        self.r_bo.write(r.view(np.int16), 0)
        self.r_bo.sync(self._TO)


    def _rmsnorm(self, x, w):
        xf = x.astype(np.float32)
        return xf / np.sqrt((xf * xf).mean() + 1e-6) * w.astype(np.float32)

    def step(self, tok, want_logits=True):
        xrt, np_ = self.xrt, np
        p = self.pos
        if p >= self.ATTN_MAXL:
            raise RuntimeError(f"context full ({self.ATTN_MAXL})")
        n = self.stair.patch_insts(self._st, p + 1, xrt, self._TO)
        self.x_bo.write(np_.asarray(self.ck.raw("model.embed_tokens.weight")[tok], bfloat16).view(np_.int16), 0)
        self.x_bo.sync(self._TO)
        for k in range(self.NL + 1):
            self._cur_layer = k
            # FFN of layer k-1 runs in this dispatch; k==0 is a zero-weight FFN edge.
            if k == 0:
                exp, w, idx = self.E_bo[0][:self.topk], np.zeros(self.topk, np.float32), list(range(self.topk))
            else:
                exp, w, idx = self._sel
            self._sel = (exp, w, idx)
            self._fill_down(k - 1 if k >= 1 else 0, idx)   # gather this FFN's experts' down weights
            self._write_rms(k, p)
            attn_bo = self.attn[k] if k < self.NL else self.zero_attn
            kv_bo = self.kv[k] if k < self.NL else self.kv[self.NL]
            st = self.kern(3, self.ib, n, self.x_bo, attn_bo, self.r_bo, self.y_bo, kv_bo,
                           *exp, self.dstage).wait(60000)
            if not str(st).endswith("COMPLETED"):
                raise RuntimeError(f"dispatch layer {k} pos{p} state={st}")
            if k < self.NL:
                self.x_bo.sync(self._FROM)
                h = np_.frombuffer(self.x_bo.map(), bfloat16, count=self.K).astype(np.float32)
                hn = self._rmsnorm(h, self.rms_post[k].astype(np.float32))
                self._sel = self._route_with(k, hn) if k + 1 <= self.NL else None
        self.pos = p + 1
        if not want_logits:
            return None
        # LM head on CPU from the device's final hidden (see __init__).
        self.x_bo.sync(self._FROM)
        xf = np_.frombuffer(self.x_bo.map(), bfloat16, count=self.K).astype(np.float32)
        hn = xf / np.sqrt((xf * xf).mean() + 1e-6) * self._fnorm
        if self._lm_q is not None:
            import torch
            with torch.no_grad():
                y = self._lm_q(torch.from_numpy(hn.astype(np.float32))[None])[0].numpy()
        elif self._lm_t is not None:
            import torch
            y = torch.mv(self._lm_t, torch.from_numpy(hn.astype(np.float16))).float().numpy()
        else:
            y = self._lm @ hn
        return y[:self.VOCAB]

    def _fill_down(self, k, sel_idx):
        """Gather the selected experts' down weights into the staging BO (dense order).

        Strided copy (no transpose): each expert's [cx,rh,i,je,cy,blk] goes to its slot
        in the dense [cx,rh,i,slot,je,cy,blk] view."""
        src, dv = self._down[k], self._dview

        def put(se):
            dv[:, se[0], :] = src[se[1]]
        list(self._pool.map(put, enumerate(sel_idx)))
        self.dstage.sync(self._TO)

    def _route_with(self, k, hn):
        logits = self.routers[k] @ hn
        e = np.exp(logits - logits.max())
        prob = e / e.sum()
        top = np.sort(np.argsort(-prob)[:self.topk])
        w = prob[top]
        if self.norm_topk:
            w = w / w.sum()
        return [self.E_bo[k][int(x)] for x in top], w.astype(np.float32), [int(x) for x in top]

    def generate(self, ids, n_new, eos=(), on_token=None, temperature=0.0, seed=0):
        t0 = time.perf_counter()
        for t in ids[:-1]:
            self.step(t, want_logits=False)
        lg = self.step(ids[-1])
        t1 = time.perf_counter()
        out = []
        for _ in range(n_new):
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
