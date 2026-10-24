"""Fast prompt prefill for the fused dense decoder (NativeDecoder).

The fused decode is one token per NPU dispatch, so feeding a prompt through it runs at decode
speed (~12 tok/s on an 8B). This runs the prompt through MLIR-AIR's fused prefill device instead
(Xilinx/mlir-air programming_examples/llms/shared/fused_prefill, MIT): 128-token chunks, int4
GEMMs that dequantize the 4-bit weights on the cores, attention on the NPU. ~300-400 tok/s on
Qwen3-8B (a 238-token prompt in 0.8 s instead of 22 s).

  weights   the decoder's own 4-bit values (runtime._requant_q4k of the same, possibly
            norm-folded, matrices), packed once for the prefill GEMMs and cached on disk
            (<decoder cache>.prefill/, ~0.63 B/param); resident only while loaded
  output    the last token's logits + roped K / raw V for every prompt position, written into
            the decoder's KV regions, so decode continues at pos = len(prompt) unchanged
  build     one device per model geometry (~5 min, one time), compiled with the newer
            toolchain in config.PREFILL_ENV; build_prefill.py does it

Whole-prompt only: prefill(ids) recomputes the prompt from position 0.
"""
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from opennpu import config
from opennpu.native.runtime import _requant_q4k

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("OPENNPU_PREFILL_DIR", config.CACHE / "fused_prefill"))
TOK_S = 300.0                       # prefill speed used to choose prefill vs per-token feeding
DECODE_TOK_S = 11.0


def _air_examples():
    """Expose the checkout's programming_examples as the air_examples package."""
    ex = config.PREFILL_AIR / "programming_examples"
    if not (ex / "llms/shared/fused_prefill/dense.py").exists():
        raise FileNotFoundError(f"no fused prefill sources in {config.PREFILL_AIR}")
    sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [str(ex)]


def desc_fields(cfg):
    """The prefill device's model description (models.Desc fields) from a HF config.json."""
    h = cfg["num_attention_heads"]
    return dict(repo="", d=cfg["hidden_size"], dh=cfg.get("head_dim") or cfg["hidden_size"] // h,
                heads=h, kv_heads=cfg.get("num_key_value_heads", h), inter=cfg["intermediate_size"],
                layers=cfg["num_hidden_layers"], vocab=cfg["vocab_size"],
                qk_norm=cfg["model_type"] == "qwen3", qkv_bias=cfg["model_type"] == "qwen2",
                tied=False)


MAX_K = 16384          # deeper GEMMs hang the device (Qwen3-14B's down, K = 17408): split them


def down_split(inter):
    """Parts the down projection's K (= inter) is split into, each a multiple of 256."""
    for s in range(1, 9):
        if inter % (256 * s) == 0 and inter // s <= MAX_K:
            return s
    raise ValueError(f"intermediate size {inter} cannot be split for the prefill GEMM")


def geometry_tag(f):
    ds = down_split(f["inter"])
    return "d{d}_h{heads}x{dh}_kv{kv_heads}_i{inter}_l{layers}_v{vocab}{q}{b}{s}".format(
        **f, q="_qkn" if f["qk_norm"] else "", b="_bias" if f["qkv_bias"] else "",
        s=f"_ds{ds}" if ds > 1 else "")


def build_dir(f):
    return ROOT / geometry_tag(f)


def built(f):
    return (build_dir(f) / "manifest.json").exists()


def supported(cfg):
    """Dense models the prefill device reproduces (full attention, full rotary)."""
    return (cfg.get("model_type") in ("qwen3", "qwen2", "llama", "mistral")
            and not (cfg.get("sliding_window") and cfg.get("use_sliding_window", True))
            and cfg.get("partial_rotary_factor", 1.0) == 1.0
            and (cfg.get("rope_scaling") or {}).get("rope_type", "default") in ("default", "llama3", "linear"))


def build(f, jobs=6, log=None):
    """Compile the device for geometry f with the prefill toolchain (one time, ~5 min)."""
    out = build_dir(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    env_sh = config.PREFILL_ENV
    if not Path(env_sh).exists():
        raise FileNotFoundError(f"fused prefill toolchain missing ({env_sh}); run ./install.sh --native")
    cmd = (f"set +u; source {env_sh} >/dev/null; exec python3 {HERE / 'build_prefill.py'} "
           f"'{json.dumps({**f, 'down_split': down_split(f['inter'])})}' {out} {jobs}")
    with open(out.parent / f"{out.name}.build.log", "w") as lf:
        subprocess.run(["bash", "-c", cmd], check=True, stdout=lf, stderr=subprocess.STDOUT,
                       env={**os.environ, "OPENNPU_PREFILL_AIR": str(config.PREFILL_AIR)})
    return out


class Prefiller:
    """The fused prefill device for one NativeDecoder. load() is slow the first time (packs and
    caches the weights); run(ids) prefills and seeds the decoder."""

    def __init__(self, dec):
        _air_examples()
        from air_examples.llms.shared.fused_prefill import dense
        from air_examples.llms.shared.fused_prefill.models import Desc
        self.dec, self.dense = dec, dense
        self.fold = bool(getattr(dec.fd, "NORM_FOLD", False))
        self.f = desc_fields(dec.ck.cfg)
        if not built(self.f):
            raise FileNotFoundError(f"fused prefill not built for {geometry_tag(self.f)}")
        self.desc = Desc(**self.f)                    # built for max_len 2048 KV records
        self.max_len = min(self.desc.max_len, dec.ATTN_MAXL)
        self.split = down_split(self.desc.inter)
        self.lock = threading.Lock()
        self.pf = None

    # ------------------------------------------------------------------ weights
    def _packed(self):
        """Iterator of (site, k, n, packed) in load order, from the disk cache or built."""
        from air_examples.llms.shared.fused_prefill import packing as P
        d = self.dec
        cache = Path(str(d.cache) + ".prefill" + (f"-ds{self.split}" if self.split > 1 else ""))
        index = cache / "index.json"
        if index.exists():
            for site, k, n, shape in json.loads(index.read_text()):
                yield site, k, n, np.load(cache / f"{site}.npy", mmap_mode="r")
            return
        ck, e, fold = d.ck, self.desc, self.fold
        tmp = Path(str(cache) + ".part")
        tmp.mkdir(parents=True, exist_ok=True)
        entries = []

        def q4(W):
            return _requant_q4k(np.ascontiguousarray(W, np.float32))

        def emit(site, k, n, W):
            a = P.pack_q4(*q4(W))
            np.save(tmp / f"{site}.npy", a)
            entries.append((site, k, n, list(a.shape)))
            return a

        dq, dk = e.heads * e.dh, e.kv_heads * e.dh
        for L in range(e.layers):
            p = f"model.layers.{L}."
            g = lambda n: ck.f32(p + n + ".weight")  # noqa: E731
            fi = g("input_layernorm")[None, :] if fold else 1.0
            fp = g("post_attention_layernorm")[None, :] if fold else 1.0
            yield f"{L}.qkv", e.d, dq + 2 * dk, emit(f"{L}.qkv", e.d, dq + 2 * dk, np.concatenate(
                [g("self_attn.q_proj"), g("self_attn.k_proj"), g("self_attn.v_proj")]) * fi)
            yield f"{L}.o", dq, e.d, emit(f"{L}.o", dq, e.d, g("self_attn.o_proj"))
            yield f"{L}.gate", e.d, e.inter, emit(f"{L}.gate", e.d, e.inter, g("mlp.gate_proj") * fp)
            yield f"{L}.up", e.d, e.inter, emit(f"{L}.up", e.d, e.inter, g("mlp.up_proj") * fp)
            ds, W = self.split, g("mlp.down_proj")
            if ds == 1:
                yield f"{L}.down", e.inter, e.d, emit(f"{L}.down", e.inter, e.d, W)
            h = e.inter // ds
            for j in range(ds if ds > 1 else 0):
                yield f"{L}.down.{j}", h, e.d, emit(f"{L}.down.{j}", h, e.d, W[:, j * h:(j + 1) * h])
        lm = "lm_head.weight" if "lm_head.weight" in ck else "model.embed_tokens.weight"
        fn = ck.f32("model.norm.weight")[None, :] if fold else 1.0
        for i, (n0, n1) in enumerate(self._lm_split()):
            yield f"lm.{i}", e.d, n1 - n0, emit(f"lm.{i}", e.d, n1 - n0, ck.f32(lm)[n0:n1] * fn)
        (tmp / "index.json").write_text(json.dumps(entries))
        tmp.rename(cache)

    def _lm_split(self):
        from air_examples.llms.shared.fused_prefill import device as D
        return D.n_split(self.desc.vocab)

    def nbytes(self):
        """Resident bytes once loaded (weights; activations are small)."""
        e = self.desc
        p = e.layers * (e.d * (e.heads + 2 * e.kv_heads) * e.dh + e.heads * e.dh * e.d + 3 * e.d * e.inter)
        return int((p + e.vocab * e.d) * 0.63)

    def load(self):
        """Pack (first time, cached) and upload the weights; open the device."""
        if self.pf is not None:
            return
        from air_examples.llms.shared.fused_prefill import device as D  # noqa: F401
        dense, d, e = self.dense, self.dec, self.desc
        ck = d.ck
        pf = _split_class(dense, self.split).__new__(_split_class(dense, self.split))
        pf.split = self.split
        pf.desc = e
        dense.Engine.__init__(pf, str(build_dir(self.f)), dense.spec(e).CFG, e.max_len)
        pf.current_context_length, pf.rope_cache, pf.kv = 0, {}, []
        c = ck.cfg
        pf.eps = float(c.get("rms_norm_eps", 1e-6))
        pf.inv_freq, pf.rot, pf.rope_scale = {"a": d.inv[: e.dh // 2]}, e.dh, 1.0
        for site, k, n, packed in self._packed():
            pf.gemm_site(site, k, n, np.ascontiguousarray(packed), 1)
        one = np.ones(e.d, np.float32)
        pf.norms, pf.bias = [], []
        for L in range(e.layers):
            p = f"model.layers.{L}."
            nm = {"input": one if self.fold else ck.f32(p + "input_layernorm.weight"),
                  "post_attn": one if self.fold else ck.f32(p + "post_attention_layernorm.weight")}
            if e.qk_norm:
                nm["q_norm"] = ck.f32(p + "self_attn.q_norm.weight")
                nm["k_norm"] = ck.f32(p + "self_attn.k_norm.weight")
            pf.norms.append(nm)
            if e.qkv_bias:
                pf.bias.append(np.concatenate([ck.f32(p + f"self_attn.{w}_proj.bias") for w in "qkv"]))
        pf.final_norm = one if self.fold else ck.f32("model.norm.weight")
        pf.lm = self._lm_split()
        pf._embed = lambda ids: np.asarray(d.embed[ids], np.float32)
        pf.attn_setup()
        pf.kvb = [pf.kv_bo("a") for _ in range(e.layers)]
        pf.warm("0.up")
        self.pf = pf

    def unload(self):
        self.pf = None

    # ------------------------------------------------------------------ run
    def worth_it(self, n_total, n_new):
        """Prefill the whole prompt rather than feed n_new tokens one by one?"""
        return n_total <= self.max_len - 1 and n_total / TOK_S + 0.3 < n_new / DECODE_TOK_S

    def run(self, ids):
        """Prefill `ids` from position 0, seed the decoder's KV, set dec.pos; returns the last
        token's logits (f32, vocab-sized)."""
        with self.lock:
            self.load()
            lg = self.pf.prefill(ids)
            K, V = self.pf.kv_stack()                     # [layers, P, kv_heads*dh] f32
        self._seed(K, V, len(ids))
        return np.asarray(lg[: self.dec.VOCAB], np.float32)

    def _seed(self, K, V, P):
        """K/V into the decoder's regions: per layer [K_g0 | K_g1 | ... | V_g0 | ...], position p at
        p*REGION_W, each group holding REGION_W/DH consecutive KV heads."""
        d = self.dec
        RW, NG, ML, LREG = d.REGION_W, d.NGRP, d.ATTN_MAXL, d.LREG
        kv = np.frombuffer(d.kvc.map(), dtype=np.int16, count=d.L * LREG)
        kb = K.astype(bfloat16).view(np.int16).reshape(d.L, P, NG, RW)
        vb = V.astype(bfloat16).view(np.int16).reshape(d.L, P, NG, RW)
        for L in range(d.L):
            base = L * LREG
            for gi in range(NG):
                o = base + gi * ML * RW
                kv[o:o + P * RW] = kb[L, :, gi].reshape(-1)
                o = base + (NG + gi) * ML * RW
                kv[o:o + P * RW] = vb[L, :, gi].reshape(-1)
        d.kvc.sync(d.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        d.pos = P


def _split_class(dense, ds):
    """DensePrefill, or a subclass whose down projection runs as ds K-slices summed on the host."""
    if ds == 1:
        return dense.DensePrefill
    from air_examples.llms.shared.fused_prefill import device as D
    from air_examples.llms.shared.fused_prefill import hostops as H
    TO = dense.TO

    class SplitDown(dense.DensePrefill):
        def _ffn_all(self, L, x, chunks, pend):
            e, nm, eps, TM, TK = self.desc, self.norms[L], self.eps, D.TM, D.TK
            rows = [x[r0:r0 + t] for r0, t in chunks]
            h = e.inter // ds
            gu = []
            for c, (r0, t) in enumerate(chunks):
                o = self.gemm_wait(pend[c], t)
                a = self.a_chunk(e.d, c)
                H.add_rms_tile(rows[c], o, t, None, eps, nm["post_attn"], a[1], TM, TK)
                a[0].sync(TO)
                gu.append((self.gemm_start(f"{L}.gate", c, 1), self.gemm_start(f"{L}.up", c)))
            for c, (r0, t) in enumerate(chunks):
                g = self.gemm_wait(gu[c][0], t)
                u = self.gemm_wait(gu[c][1], t)
                parts = []
                for j in range(ds):
                    a = self.a_chunk(h, c * ds + j)
                    H.glu_tile(np.ascontiguousarray(g[:, j * h:(j + 1) * h]),
                               np.ascontiguousarray(u[:, j * h:(j + 1) * h]), t, h, a[1], TM, TK)
                    a[0].sync(TO)
                    parts.append(self.gemm_start(f"{L}.down.{j}", c * ds + j, role=f"down{j}"))
                f = None
                for p in parts:
                    r = self.gemm_wait(p, t)[:t, :e.d].astype(np.float32)
                    f = r if f is None else f + r
                H.add_rms_tile(rows[c], f.astype(bfloat16), t, None, eps, None, None, TM, TK)

    return SplitDown


def for_decoder(dec, build_if_missing=False, log=None):
    """A Prefiller for dec, or None when the model or toolchain is not supported/built."""
    if os.environ.get("OPENNPU_PREFILL", "1") == "0" or not supported(dec.ck.cfg):
        return None
    f = desc_fields(dec.ck.cfg)
    try:
        if not built(f):
            if not build_if_missing:
                return None
            if log:
                log(f"building the fused prefill for {geometry_tag(f)} (one time, ~5 min) ...")
            t = time.time()
            build(f)
            if log:
                log(f"fused prefill built in {time.time() - t:.0f}s")
        return Prefiller(dec)
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"fused prefill unavailable ({type(e).__name__}: {e}); prompts feed per token")
        return None
