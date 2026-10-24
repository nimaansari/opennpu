"""Universal 4-bit NPU engine: any Hugging Face causal LM, dense or mixture-of-experts.

The model runs as its own transformers code (attention variants, routers, linear attention,
norms: whatever the architecture needs, exactly as written upstream), but every large matrix
product goes to the NPU through the int4 GEMV device (gemv_device.py):

  * the model is built on the meta device, so its full-size weights never exist in RAM
  * every nn.Linear above MIN_PARAMS and every 3-D expert bank is quantized straight from the
    checkpoint (bf16/fp16/fp32, or gpt-oss MXFP4) to 4-bit Q4NX (runtime._requant_q4k: per 32
    weights a scale and a min) and packed into NPU buffers: ~0.56 B/param resident
  * sibling projections that read the same input (q/k/v, gate/up, DeltaNet's qkv/z) run as one
    dispatch; a token's routed experts run as one dispatch per projection (8 columns -> experts)
  * the rest (embeddings, norms, small layers, biases, activations, attention) stays on the CPU

Decode is one token per forward; prompts also run token by token through the NPU layers (each
row is a GEMV), so prefill speed ~= decode speed for now.

Vision-language checkpoints (a config with a vision_config, e.g. Qwen3.5/3.6-MoE) are built as the
full image-text-to-text model, so checkpoint keys map 1:1 (model.language_model.*, model.visual.*,
lm_head). The vision tower's Linear layers run where OPENNPU_NPU4_VISION says:

  * "cpu" (default): bf16 on the CPU. A vision tower sees hundreds of rows (one per image patch) per
    layer, and the NPU device is a GEMV: it re-streams the int4 weights for every row. Measured on
    Qwen3.6-35B-A3B's tower (448x448 image = 784 patches): one 1152 -> 1152 layer NPU 113 ms
    (8 rows per dispatch) vs CPU bf16 13 ms; the whole tower NPU 8.2 s vs CPU 1.35 s, and int4
    vision weights read text in images less reliably.
  * "npu": int4 on the NPU, NPURowsLinear (8 // ncols rows per dispatch, the matrix on as few
    columns as fit its K). Works (same answers on simple images), ~6x slower per image.
  * "off": build the text-only causal LM as before (images are refused).
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opennpu import config
from opennpu.native.runtime import _requant_q4k

from . import gemv_device as G

MIN_PARAMS = int(os.environ.get("OPENNPU_NPU4_MIN_PARAMS", str(1 << 20)))
VISION = os.environ.get("OPENNPU_NPU4_VISION", "cpu").lower()          # cpu | npu | off
# submodules that make up the vision side of an image-text-to-text model
VISION_PARTS = ("visual", "vision_tower", "vision_model", "multi_modal_projector", "mm_projector")
BUILD = Path(os.environ.get("OPENNPU_GEMV_DIR", config.CACHE / "gemv"))
HERE = Path(__file__).resolve().parent
BUILD8 = Path(os.environ.get("OPENNPU_GEMM8_DIR", config.CACHE / "gemm8"))
BATCHED = os.environ.get("OPENNPU_NPU4_BATCHED", "1") != "0"      # 8-token device for prompts
# weight format: q4nx (asymmetric 4-bit, scale + min per 32: 5 bits/weight; the most accurate with
# the search quantizer, the default) or q4s (symmetric, scale per 32: 4.5 bits/weight, 10% fewer
# bytes to stream: ~5-10% faster decode, ~6% higher perplexity measured on gpt-oss-20b).
FORMAT = os.environ.get("OPENNPU_NPU4_FORMAT", "q4nx").lower()
# matrices (name substrings, comma separated) stored as two 4-bit passes, the weights and the
# rounding left over (~8-bit precision, twice the bytes), in one dispatch
HIPREC = [p for p in os.environ.get("OPENNPU_NPU4_HIPREC", "").split(",") if p]
_DEV = {}
_DEV_LOCK = threading.Lock()


def build_dir(batch=1, sym=False):
    d = BUILD if batch == 1 else BUILD8
    return Path(str(d) + "-s") if sym else d


def device(batch=1, sym=False):
    """The GEMV device (decode, batch 1) or the 8-token device (prompts, batch 8); sym: for q4s weights."""
    with _DEV_LOCK:
        if (batch, sym) not in _DEV:
            from .gemv import GemvDevice
            _DEV[batch, sym] = GemvDevice(build_dir(batch, sym), batch)
        return _DEV[batch, sym]


def _build(build, batch, shapes, log, sym=False):
    import subprocess
    man = build / "manifest.json"
    have = {tuple(s) for s in json.loads(man.read_text())["shapes"]} if man.exists() else set()
    todo = sorted(set(shapes) - have)
    if not todo:
        return
    kind = ("GEMV" if batch == 1 else "8-token GEMM") + (" (q4s)" if sym else "")
    # ~6 s per shape with 6 parallel builds (7 shapes in 44 s on a Ryzen AI 7 350); big shapes longer
    log(f"building {len(todo)} NPU {kind} shape(s) (one time, about {max(1, round(len(todo) * 0.15))} min): {todo}")
    env_sh = config.PREFILL_ENV
    if not Path(env_sh).exists():
        raise FileNotFoundError(f"NPU kernel toolchain missing ({env_sh}); run ./install.sh --native")
    args = " ".join(f"{t}:{k}" for t, k in todo)
    build.mkdir(parents=True, exist_ok=True)
    with open(build / "build.log", "a") as lf:
        subprocess.run(["bash", "-c", f"set +u; source {env_sh} >/dev/null; exec python3 {HERE / 'build_gemv.py'} "
                        f"{build} {args} -j 6 --batch {batch}" + (" --sym" if sym else "")], check=True, stdout=lf, stderr=subprocess.STDOUT)
    if (batch, sym) in _DEV:                       # pick up the new shapes
        _DEV[batch, sym].man = json.loads(man.read_text())


def ensure_shapes(shapes, log=print, sym=False):
    """Build any (tiles, kc) shape not built yet, for both devices (~1-2 min per shape, once)."""
    _build(build_dir(1, sym), 1, shapes, log, sym)
    if BATCHED:
        try:
            _build(build_dir(G.BATCH, sym), G.BATCH, shapes, log, sym)
        except Exception as e:  # noqa: BLE001 - prompts fall back to one token per pass
            log(f"8-token device unavailable ({e}); prompts run one token at a time")


_B8 = {}


def batched_ok(m):
    """The 8-token device has this matrix's shape (manifest re-read at most every 10 s)."""
    if not BATCHED:
        return False
    t, have = _B8.get(m.sym, (0.0, set()))
    if time.time() - t > 10:
        p = build_dir(G.BATCH, m.sym) / "manifest.json"
        have = {tuple(x) for x in json.loads(p.read_text())["shapes"]} if p.exists() else set()
        _B8[m.sym] = (time.time(), have)
    return (m.tiles, m.kc) in have


def quant_sym(W, group=32):
    """q4s: per 32 columns, w ~= (q - 8) * scale (q 0..15, scale bf16-exact). Of a few scales
    around absmax/8 the one with the least squared error is kept per group."""
    from ml_dtypes import bfloat16 as _bf
    M_, K = W.shape
    g = W.reshape(M_, K // group, group)
    mx = np.take_along_axis(g, np.abs(g).argmax(2)[..., None], 2)[..., 0]
    best_q = best_d = best_e = None
    for div in (-8.0, -7.5, -7.0, 7.0, 7.5):
        d = np.asarray(np.asarray(mx / div, np.float32).astype(_bf), np.float32)
        d = np.where(d == 0, 1.0, d).astype(np.float32)
        q = np.clip(np.rint(g / d[..., None]), -8, 7)
        e = ((q * d[..., None] - g) ** 2).sum(2)
        if best_e is None:
            best_q, best_d, best_e = q, d, e
        else:
            m = e < best_e
            best_q = np.where(m[..., None], q, best_q)
            best_d = np.where(m, d, best_d)
            best_e = np.minimum(e, best_e)
    return (best_q + 8).astype(np.uint8).reshape(M_, K), best_d


# ------------------------------------------------------------------ packed matrices
class Mat:
    """Geometry of an [N, K] matrix on `ncols` device columns: R row blocks x S K-slices
    (R * S = ncols), every block the same (tiles, kc) shape, so the whole matrix is one
    dispatch; the S partial products are summed on the host. (R, S) is picked to stream the
    fewest bytes (padding to whole 64-row tiles and 512-wide K steps), K/S <= KMAX."""

    def __init__(self, N, K, ncols, sym=False):
        self.N, self.K, self.ncols, self.sym = N, K, ncols, sym
        best = None
        for S in [d for d in range(1, ncols + 1) if ncols % d == 0]:
            kc = -(-K // (S * G.GK))                     # odd allowed: a pair of tiles is 2 * kc packets
            if kc > G.KMAX // G.GK:
                continue
            R = ncols // S
            tiles = -(-N // (R * G.NR * G.GN))          # odd allowed when tiles * kc stays even
            if (tiles * kc) % 2:                         # (the ping-pong buffers must stay in step)
                if kc + 1 <= G.KMAX // G.GK and tiles * (kc + 1) < (tiles + 1) * kc:
                    kc += 1
                else:
                    tiles += 1
            cost = (R * S * tiles * kc * (1 + 0.03 * (S - 1)), S)     # fewer host sums on near-ties
            if best is None or cost < best[0]:
                best = (cost, R, S, tiles, kc)
        if best is None:
            raise ValueError(f"[{N}, {K}] needs more than {ncols} columns of K <= {G.KMAX}")
        _, self.R, self.S, self.tiles, self.kc = best
        self.ks = self.kc * G.GK                         # K per slice (padded)
        self.rows_col = G.NR * self.tiles * G.GN
        self.cb = G.col_bytes(self.tiles, self.kc, sym)
        self.nbytes = self.R * self.S * self.cb

    def shapes(self):
        return {(self.tiles, self.kc)}

    def blocks(self):
        """(column index within this matrix, row block r, K slice s)."""
        return [(r * self.S + s, r, s) for r in range(self.R) for s in range(self.S)]

    def pack(self, W, threads=1):
        """W [N, K] float -> packed bytes (uint8). libonpu's quantizers (scale search with a
        least-squares refit per 32 weights) when available, else the numpy ones below."""
        lib = clib()
        if lib is not None and hasattr(lib, "onpu_pack"):
            import ctypes as C
            W = np.ascontiguousarray(W, np.float32)
            out = np.empty(self.nbytes, np.uint8)
            n = lib.onpu_pack(W.ctypes.data_as(C.POINTER(C.c_float)), self.N, self.K, self.ncols, int(self.sym),
                              out.ctypes.data_as(C.POINTER(C.c_uint8)), threads)
            if n == self.nbytes:
                return out
        if self.sym:
            q, sc = quant_sym(np.ascontiguousarray(W, np.float32))
            mn = None
        else:
            q, sc, mn = _requant_q4k(np.ascontiguousarray(W, np.float32))
        out = np.empty(self.nbytes, np.uint8)
        for c, r, s in self.blocks():
            r0, r1 = r * self.rows_col, min((r + 1) * self.rows_col, self.N)
            k0, k1 = s * self.ks, min((s + 1) * self.ks, self.K)
            o = c * self.cb
            if r1 > r0 and k1 > k0:
                out[o:o + self.cb] = G.pack_block(q[r0:r1, k0:k1], sc[r0:r1, k0 // 32:k1 // 32],
                                                  None if mn is None else mn[r0:r1, k0 // 32:k1 // 32],
                                                  self.tiles, self.kc)
            else:
                out[o:o + self.cb] = 0
        return out

    def x_cols(self, x):
        """The S input slices, padded to the slice width: [S, ks]."""
        xp = np.zeros(self.S * self.ks, np.float32)
        xp[:self.K] = x
        return xp.reshape(self.S, self.ks)

    def x_dev(self, xs):
        """Inputs [n, K] -> the device's per-column inputs [n * R * S, ks] (column c = r*S + s)."""
        n = xs.shape[0]
        xp = np.zeros((n, self.S * self.ks), np.float32)
        xp[:, :self.K] = xs
        xp = xp.reshape(n, 1, self.S, self.ks)
        return np.broadcast_to(xp, (n, self.R, self.S, self.ks)).reshape(n * self.R * self.S, self.ks)

    def x_dev8(self, xs):
        """Up to 8 rows [n, K] -> the 8-token device's per-column inputs [R * S, ks * 8]."""
        n = xs.shape[0]
        xp = np.zeros((n, self.S * self.ks), np.float32)
        xp[:, :self.K] = xs
        cols = [G.x_block(xp[:, s * self.ks:(s + 1) * self.ks]) for s in range(self.S)]
        return np.stack([cols[s] for _, _, s in self.blocks()])

    def gather8(self, y, copies=1):
        """8-token device output (copies * R * S columns) -> [copies, 8, N]."""
        out = np.zeros((copies, G.BATCH, self.R * self.rows_col), np.float32)
        for e in range(copies):
            for c, r, s in self.blocks():
                out[e, :, r * self.rows_col:(r + 1) * self.rows_col] += G.unpack_col8(y[e * self.R * self.S + c], self.tiles)
        return out[:, :, :self.N]

    def gather_n(self, y, n):
        """Device output of n copies (n * R * S columns) -> [n, N]: unpack, sum the K slices."""
        to = G.out_tiles(self.tiles)
        y = y[:n * self.R * self.S].reshape(n, self.R, self.S, to // G.FLUSH, G.NR, G.FLUSH * G.GN)
        y = y.transpose(0, 1, 2, 4, 3, 5).reshape(n, self.R, self.S, G.NR, to * G.GN)[..., :self.tiles * G.GN]
        y = y.reshape(n, self.R, self.S, self.rows_col).sum(2)
        return y.reshape(n, -1)[:, :self.N]


class Arena:
    """One NPU buffer holding many packed matrices (offsets < 1 GiB, the device's W window)."""

    def __init__(self, nbytes, sym=False):
        self.dev = device(1, sym)
        self.n = int(nbytes)
        self.bo = self.dev.bo(self.n, 0)
        self.view = np.frombuffer(self.bo.map(), np.uint8, count=self.n)
        self.used = 0

    def put(self, blob):
        o = self.used
        self.view[o:o + blob.size] = blob
        self.used += blob.size
        return o

    def sync(self):
        self.bo.sync(self.dev.TO)


_CLIB = [False, None]


def clib():
    """libonpu (llama/build.sh): the NPU core in C++, so each matrix product is one C call. None when
    missing or OPENNPU_NPU4_CLIB=0; the pyxrt path is then used."""
    if _CLIB[0]:
        return _CLIB[1]
    _CLIB[0] = True
    if os.environ.get("OPENNPU_NPU4_CLIB", "1") == "0":
        return None
    import ctypes as C
    p = Path(os.environ.get("OPENNPU_LIBONPU", config.HOME / "lib/libonpu.so"))
    if not p.exists():
        return None
    try:
        lib = C.CDLL(str(p))
    except OSError:
        return None
    P, I64, F = C.c_void_p, C.c_int64, C.POINTER(C.c_float)
    lib.onpu_ok.restype = C.c_int
    lib.onpu_geometry.argtypes = [I64, I64, C.c_int, C.POINTER(I64)]
    lib.onpu_dense_new.restype, lib.onpu_dense_new.argtypes = P, [I64, I64]
    if hasattr(lib, "onpu_quant_dequant"):
        lib.onpu_quant_dequant.argtypes = [F, I64, I64, C.c_int, F, C.c_int]
    if hasattr(lib, "onpu_pack"):
        lib.onpu_pack.restype = I64
        lib.onpu_pack.argtypes = [F, I64, I64, C.c_int, C.c_int, C.POINTER(C.c_uint8), C.c_int]
    if not hasattr(lib, "onpu_dense_new2"):
        return None                                  # an older libonpu: rebuild with llama/build.sh
    lib.onpu_geometry2.argtypes = [I64, I64, C.c_int, C.c_int, C.POINTER(I64)]
    lib.onpu_dense_new2.restype, lib.onpu_dense_new2.argtypes = P, [I64, I64, C.c_int]
    lib.onpu_bank_new2.restype, lib.onpu_bank_new2.argtypes = P, [I64, I64, I64, C.c_int, C.c_int]
    lib.onpu_dense_buf.restype, lib.onpu_dense_buf.argtypes = C.POINTER(C.c_uint8), [P]
    lib.onpu_dense_sync.argtypes = [P]
    lib.onpu_dense_free.argtypes = [P]
    lib.onpu_dense_mv.argtypes = [P, F, I64, F]
    lib.onpu_bank_new.restype, lib.onpu_bank_new.argtypes = P, [I64, I64, I64, C.c_int]
    lib.onpu_bank_buf.restype, lib.onpu_bank_buf.argtypes = C.POINTER(C.c_uint8), [P]
    lib.onpu_bank_sync.argtypes = [P]
    lib.onpu_bank_free.argtypes = [P]
    lib.onpu_bank_run.argtypes = [P, F, C.c_int, I64, C.POINTER(C.c_int32), C.c_int, F]
    if hasattr(lib, "onpu_ffn_new"):
        I32P = C.POINTER(C.c_int32)
        lib.onpu_ffn_new.restype, lib.onpu_ffn_new.argtypes = P, [I64, I64, I64, I64, I64, C.c_int]
        for n in ("onpu_ffn_gu", "onpu_ffn_dn"):
            getattr(lib, n).restype, getattr(lib, n).argtypes = P, [P]
        lib.onpu_ffn_fused.argtypes = lib.onpu_ffn_free.argtypes = [P]
        lib.onpu_ffn_perm.argtypes = [P, I32P]
        lib.onpu_ffn_run.argtypes = [P, F, I32P, C.c_int, F, F]
        lib.onpu_ffn_run_many.argtypes = [P, F, I64, I32P, C.c_int, F, F]
    if hasattr(lib, "onpu_moe_ffn"):
        lib.onpu_moe_ffn.argtypes = [P, P, F, I64, C.POINTER(C.c_int32), C.c_int, F, C.c_int, C.c_float, C.c_float,
                                     F, F, F]
    if not lib.onpu_ok():
        return None
    _CLIB[1] = lib
    return lib


def _fp(a):
    import ctypes as C
    return a.ctypes.data_as(C.POINTER(C.c_float))


def quant_dequant(W, sym=False):
    """What the NPU makes of W [N, K] (each 32-weight group quantized as pack does), dequantized."""
    import ctypes as C
    W = np.ascontiguousarray(W, np.float32)
    out = np.empty_like(W)
    clib().onpu_quant_dequant(W.ctypes.data_as(C.POINTER(C.c_float)), W.shape[0], W.shape[1], int(sym),
                              out.ctypes.data_as(C.POINTER(C.c_float)), 8)
    return out


def check_geometry(m):
    """The C core must pack exactly like Mat (same R, S, tiles, kc, bytes)."""
    import ctypes as C
    out = (C.c_int64 * 8)()
    clib().onpu_geometry2(m.N, m.K, m.ncols, int(m.sym), out)
    want = (m.R, m.S, m.tiles, m.kc, m.ks, m.rows_col, m.cb, m.nbytes)
    if tuple(out) != want:
        raise RuntimeError(f"libonpu geometry {tuple(out)} != {want}: rebuild llama/build.sh")


class CArena:
    """An Arena over a libonpu buffer (Python packs into it; sync hands it to the device)."""

    def __init__(self, ptr, nbytes, sync):
        self.n = int(nbytes)
        self.view = np.ctypeslib.as_array(ptr, shape=(self.n,))
        self.used = 0
        self._sync = sync

    def put(self, blob):
        o = self.used
        self.view[o:o + blob.size] = blob
        self.used += blob.size
        return o

    def sync(self):
        self._sync()


def fast_cpu_paths(model):
    """Swap the per-token CPU work PyTorch runs as dozens of tiny ops for one libonpu call.
    Today: Gated DeltaNet decode (Qwen3.5 / Qwen3.6 linear attention). OPENNPU_NPU4_FASTCPU=0 turns it off."""
    lib = clib()
    if lib is None or not hasattr(lib, "onpu_gdn_step") or os.environ.get("OPENNPU_NPU4_FASTCPU", "1") == "0":
        return 0
    import ctypes as C
    import types
    P, I, F = C.POINTER(C.c_float), C.c_int, C.c_float
    lib.onpu_gdn_step.argtypes = [P, I, P, P, P, P, P, I, P, I, P, P, P, F, P, I, I, I, I, P]
    if hasattr(lib, "onpu_attn_step"):
        I64 = C.c_int64
        lib.onpu_attn_step.argtypes = [P, I, I, P, P, I, I64, I64, I64, P, P, F, P]
        if hasattr(lib, "onpu_attn_step_bf16"):
            lib.onpu_attn_step_bf16.argtypes = lib.onpu_attn_step.argtypes
        _patch_attention(model)
    n = 0
    for mod in model.modules():
        if not type(mod).__name__.endswith("GatedDeltaNet"):
            continue
        need = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "A_log", "dt_bias", "norm", "out_proj")
        if not all(hasattr(mod, a) for a in need) or getattr(mod, "activation", "silu") != "silu" \
                or not isinstance(mod.in_proj_b, nn.Linear) or not isinstance(mod.in_proj_a, nn.Linear) \
                or mod.conv1d.bias is not None or mod.in_proj_b.bias is not None or mod.in_proj_a.bias is not None:
            continue
        f32 = (lambda t: t.detach().float().contiguous())
        mod._gdn = dict(Wb=f32(mod.in_proj_b.weight), Wa=f32(mod.in_proj_a.weight),
                        conv=f32(mod.conv1d.weight.squeeze(1)), A=f32(mod.A_log), dt=f32(mod.dt_bias),
                        nw=f32(mod.norm.weight), eps=float(getattr(mod.norm, "variance_epsilon", 1e-6)))
        mod._slow_forward = mod.forward
        mod.forward = types.MethodType(_gdn_forward, mod)
        n += 1
    if os.environ.get("OPENNPU_NPU4_LEAN", "1") != "0":
        _patch_lean_attention(model)
    _patch_kv_cache()
    return n


# attention classes whose forward is the plain pattern (q/k/v projections -> [q/k norm] -> RoPE ->
# cache -> attention -> o_proj): name -> (q/k norms, sinks)
_LEAN = {"LlamaAttention": (False, False), "MistralAttention": (False, False), "Qwen2Attention": (False, False),
         "Qwen3Attention": (True, False), "Qwen3MoeAttention": (True, False), "GptOssAttention": (False, True)}


def _patch_lean_attention(model):
    """One token: q/k/v from a single libonpu call with their biases pre-joined, no per-projection
    module calls (each would convert, add its bias and reshape on its own)."""
    import sys
    import types
    for mod in model.modules():
        spec = _LEAN.get(type(mod).__name__)
        projs = [getattr(mod, n, None) for n in ("q_proj", "k_proj", "v_proj")]
        if spec is None or not all(isinstance(p, NPULinear) for p in projs) or getattr(projs[0].group, "ch", None) is None \
                or not (projs[0].group is projs[1].group is projs[2].group) or not hasattr(mod, "o_proj"):
            continue
        mf = sys.modules[type(mod).__module__]
        if not (hasattr(mf, "apply_rotary_pos_emb") and hasattr(mf, "eager_attention_forward")):
            continue
        g = projs[0].group
        bias = np.zeros(g.N1, np.float32)          # outputs (a residual group stores 2x rows)
        for p in projs:
            if p.bias is not None:
                o = sum(g.sizes[:p.i])
                bias[o:o + p.out_features] = p.bias.detach().float().numpy()
        mod._lean = dict(g=g, offs=[sum(g.sizes[:p.i]) for p in projs], sizes=[p.out_features for p in projs],
                         bias=torch.from_numpy(bias) if any(p.bias is not None for p in projs) else None,
                         norm=spec[0], sinks=spec[1], mf=mf)
        mod._slow_forward = mod.forward
        mod.forward = types.MethodType(_lean_attn_forward, mod)


def _lean_attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                       **kwargs):
    if hidden_states.shape[0] * hidden_states.shape[1] != 1 or position_embeddings is None or self.training:
        return self._slow_forward(hidden_states, position_embeddings=position_embeddings,
                                  attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    L = self._lean
    y = torch.from_numpy(L["g"].full(hidden_states))
    if L["bias"] is not None:
        y = y + L["bias"]
    D = self.head_dim
    (oq, ok, ov), (nq, nk, nv) = L["offs"], L["sizes"]
    q, k, v = (y[:, o:o + n].view(1, 1, -1, D) for o, n in ((oq, nq), (ok, nk), (ov, nv)))
    if L["norm"]:
        q, k = self.q_norm(q), self.k_norm(k)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = position_embeddings
    q, k = L["mf"].apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)
    fn = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, L["mf"].eager_attention_forward)
    extra = {"s_aux": self.sinks} if L["sinks"] else {}
    if getattr(self, "sliding_window", None) is not None:
        extra["sliding_window"] = self.sliding_window
    out, w = fn(self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, **extra, **kwargs)
    return self.o_proj(out.reshape(1, 1, -1)), w


def _grow_update(self, key_states, value_states, *args, **kwargs):
    """transformers' DynamicLayer.update, but new tokens are written into a buffer with spare
    capacity (doubling) instead of torch.cat-ing the whole cache every token: per token cost stops
    growing with the conversation. keys/values stay views of the buffer; anything that replaced them
    (crop, deepcopy, another shape) makes the next update reallocate, so it never trusts stale state."""
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    k, v = self.keys, self.values
    n = k.shape[-2] if k is not None and k.numel() else 0
    t = key_states.shape[-2]
    kb, vb = getattr(self, "_kbuf", None), getattr(self, "_vbuf", None)
    dt = torch.bfloat16 if KV16 and _KV16_OK[0] and key_states.dtype == torch.float32 else key_states.dtype
    fits = (kb is not None and vb is not None and n + t <= kb.shape[-2] and kb.dtype == dt
            and kb.shape[:2] == key_states.shape[:2] and kb.shape[-1] == key_states.shape[-1]
            and vb.shape[:2] == value_states.shape[:2] and vb.shape[-1] == value_states.shape[-1]
            and (n == 0 or (k.data_ptr() == kb.data_ptr() and v.data_ptr() == vb.data_ptr()
                            and k.stride() == kb.stride() and v.stride() == vb.stride())))
    if not fits:
        cap = max(256, -(-2 * (n + t) // 256) * 256)
        kb = key_states.new_empty((*key_states.shape[:2], cap, key_states.shape[-1]), dtype=dt)
        vb = value_states.new_empty((*value_states.shape[:2], cap, value_states.shape[-1]), dtype=dt)
        if n:
            kb[..., :n, :] = k
            vb[..., :n, :] = v
        self._kbuf, self._vbuf = kb, vb
    kb[..., n:n + t, :] = key_states
    vb[..., n:n + t, :] = value_states
    self.keys, self.values = kb[..., :n + t, :], vb[..., :n + t, :]
    return self.keys, self.values


# the KV cache in bf16 (half the bytes attention reads per token); decode attention reads it
# directly, everything else gets it upcast. OPENNPU_NPU4_KV16=0 keeps fp32.
KV16 = os.environ.get("OPENNPU_NPU4_KV16", "1") != "0"
_KV16_OK = [True]          # off (for the process) once a model's attention isn't ours to read bf16 with


def _patch_kv_cache():
    """Growing (no-copy) KV cache for every transformers model in this process.
    OPENNPU_NPU4_GROWCACHE=0 keeps transformers' own."""
    if os.environ.get("OPENNPU_NPU4_GROWCACHE", "1") == "0":
        return
    from transformers import cache_utils
    if getattr(cache_utils.DynamicLayer.update, "_onpu", False):
        return
    _grow_update._onpu = True
    _grow_update._orig = cache_utils.DynamicLayer.update
    cache_utils.DynamicLayer.update = _grow_update


_ATTN_PATCHED = set()


def _patch_attention(model):
    """Decode attention (one token, fp32) through libonpu: the model file's eager function and the
    shared sdpa entry are wrapped; anything else (prompts, masks per head, softcapping...) goes to
    the original."""
    import sys
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "sdpa" not in _ATTN_PATCHED and "sdpa" in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _attn_fast(ALL_ATTENTION_FUNCTIONS["sdpa"])
        _ATTN_PATCHED.add("sdpa")
    for mod in model.modules():
        mf = sys.modules.get(type(mod).__module__)
        if mf is not None and mf.__name__ not in _ATTN_PATCHED and hasattr(mf, "eager_attention_forward"):
            mf.eager_attention_forward = _attn_fast(mf.eager_attention_forward)
            _ATTN_PATCHED.add(mf.__name__)
        # a bf16 KV cache needs every attention to go through _attn_fast (which upcasts when needed)
        if type(mod).__name__.endswith("Attention") and (mf is None or not hasattr(mf, "eager_attention_forward")):
            _KV16_OK[0] = False


def _attn_prompt(module, query, key, value, attention_mask, scaling, sinks):
    """Many query tokens: PyTorch's CPU flash attention (its log-sum-exp gives the sinks' share)
    instead of the eager path's full score matrices. None if the inputs don't fit."""
    B, Hq, T, D = query.shape
    S = key.shape[2]
    m = attention_mask
    if m is None:                                   # causal, the queries are the last T positions
        i = torch.arange(T)[:, None] + (S - T)
        m = torch.where(torch.arange(S)[None, :] <= i, 0.0, float("-inf"))
    elif m.dim() == 4 and m.shape[0] == 1 and m.shape[1] == 1 and m.shape[2] == T and m.shape[3] >= S:
        m = m[0, 0, :, :S]
        m = torch.where(m, 0.0, float("-inf")) if m.dtype == torch.bool else m.float()
    else:
        return None
    rep = Hq // key.shape[1]
    k, v = (t if rep == 1 else t[:, :, None].expand(B, t.shape[1], rep, S, D).reshape(B, Hq, S, D)
            for t in (key, value))
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        query, k, v, 0.0, False, attn_mask=m[None, None], scale=float(scaling))[:2]
    if sinks is not None:
        out = out * torch.sigmoid(lse - sinks.detach().float()[None, :, None]).unsqueeze(-1)
    return out.transpose(1, 2).contiguous(), None


def _attn_fast(orig):
    def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        B, Hq, T, D = query.shape
        kv16 = key.dtype == value.dtype == torch.bfloat16 and query.dtype == torch.float32
        if kv16 and T > 1:                        # prompts: upcast once
            key, value, kv16 = key.float(), value.float(), False
        if T > 1 and B == 1 and query.dtype == key.dtype == value.dtype == torch.float32 and not \
                (dropout and module.training) and kw.get("softcap") is None and kw.get("head_mask") is None \
                and key.shape[1] and Hq % key.shape[1] == 0 and key.shape[-1] == D == value.shape[-1] \
                and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu") \
                and (kw.get("s_aux") is None or kw["s_aux"].numel() == Hq):
            r = _attn_prompt(module, query, key, value, attention_mask,
                             scaling if scaling is not None else D ** -0.5, kw.get("s_aux"))
            if r is not None:
                return r
        ok = (B == 1 and T == 1 and query.dtype == torch.float32 and key.dtype == value.dtype
              and key.dtype in (torch.float32, torch.bfloat16)
              and not (dropout and module.training) and kw.get("softcap") is None and kw.get("head_mask") is None
              and key.shape[1] and Hq % key.shape[1] == 0 and key.stride(-1) == 1 and value.stride() == key.stride()
              and key.shape[-1] == D and value.shape[-1] == D)
        mask = None
        if ok and attention_mask is not None:
            m = attention_mask
            ok = m.dim() == 4 and m.shape[0] == 1 and m.shape[1] == 1 and m.shape[-1] >= key.shape[2]
            if ok:
                m = m[0, 0, -1, :key.shape[2]]
                mask = (torch.where(m, 0.0, float("-inf")) if m.dtype == torch.bool else m.float()).contiguous()
        sinks = kw.get("s_aux")
        if not ok or (sinks is not None and sinks.numel() != Hq):
            if key.dtype != query.dtype:
                key, value = key.to(query.dtype), value.to(query.dtype)
            return orig(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        sinks = None if sinks is None else sinks.detach().float().contiguous()
        q = query.reshape(Hq, D).contiguous()
        out = torch.empty(1, 1, Hq, D)
        f = clib().onpu_attn_step_bf16 if kv16 else clib().onpu_attn_step
        f(_tp(q), Hq, D, _tp(key), _tp(value), key.shape[1], key.shape[2], key.stride(1),
          key.stride(2), None if mask is None else _tp(mask), None if sinks is None else _tp(sinks),
          float(scaling if scaling is not None else D ** -0.5), _tp(out))
        return out, None
    attn._onpu_orig = orig
    return attn


def _tp(t):
    import ctypes as C
    return C.cast(t.data_ptr(), C.POINTER(C.c_float))


def _gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    B, T, H = hidden_states.shape
    lay = None
    # a few tokens (decode, or checking draft tokens): the exact recurrent step once per token in C++;
    # transformers' chunked form pads every call to 64 tokens
    if B == 1 and T <= GDN_STEP_MAX and cache_params is not None \
            and cache_params.has_previous_state(self.layer_idx, state_idx=0):
        lay = cache_params.layers[self.layer_idx]
        if getattr(lay, "record_past", False):
            lay = None
    if lay is not None:
        conv, S = lay.conv_states[0], lay.recurrent_states[0]
        ok = all(t.dtype == torch.float32 and t.is_contiguous() for t in (conv, S, hidden_states))
    if lay is None or not ok:
        return self._slow_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    g = self._gdn
    qkv = self.in_proj_qkv(hidden_states).float().contiguous()
    z = self.in_proj_z(hidden_states).float().contiguous()
    out = torch.empty(1, T, self.num_v_heads * self.head_v_dim)
    for t in range(T):
        clib().onpu_gdn_step(_tp(hidden_states[0, t]), H, _tp(qkv[0, t]), _tp(z[0, t]), _tp(g["Wb"]), _tp(g["Wa"]),
                             _tp(g["conv"]), g["conv"].shape[1], _tp(conv), conv.shape[-1], _tp(g["A"]), _tp(g["dt"]),
                             _tp(g["nw"]), g["eps"], _tp(S), self.num_k_heads, self.num_v_heads, self.head_k_dim,
                             self.head_v_dim, _tp(out[0, t]))
    return self.out_proj(out.to(hidden_states.dtype))


_SIDE = ThreadPoolExecutor(1, thread_name_prefix="npu4-experts")
FUSE_FFN = os.environ.get("OPENNPU_NPU4_FUSE_FFN", "1") != "0"   # gate_up + down in one dispatch
# fused dispatches run on the 1-token device: only for single tokens, since several tokens use the
# 8-token device for everything else and switching devices costs ~2.4 ms each time
FUSE_T_MAX = 1
FUSE_MANY = os.environ.get("OPENNPU_NPU4_FUSE_MANY", "0") == "1"   # several tokens fused (8-token device): measured no faster, opt-in
GDN_STEP_MAX = 16          # up to this many new tokens take the per-token C++ path (DeltaNet, MoE block)


def _q8(W):
    """int8 rows with a scale each (absmax / 127): (W8 [N, K], scale [N])."""
    s = (W.abs().amax(1) / 127.0).clamp_min(1e-12)
    return torch.round(W / s[:, None]).to(torch.int8).contiguous(), s.float().contiguous()


def _gemv_i8(W, s, x, y):
    import ctypes as C
    lib = clib()
    if not getattr(lib.onpu_gemv_i8, "argtypes", None):
        lib.onpu_gemv_i8.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_void_p, C.c_void_p]
    lib.onpu_gemv_i8(W.data_ptr(), s.data_ptr(), W.shape[0], W.shape[1], x.data_ptr(), y.data_ptr())


def _moe_shared_forward(self, hidden_states):
    """Qwen3.5-MoE sparse block, one token: routed experts on the NPU (another thread; the C call
    releases the GIL) while the shared expert runs on the CPU."""
    B, T, H = hidden_states.shape
    if B * T > GDN_STEP_MAX or (B * T > 1 and self._shared_cpu is not None):
        return self._slow_forward(hidden_states)
    h = hidden_states.reshape(-1, H)
    wr = getattr(self, "_router_cpu", None)
    if wr is not None:              # router + shared-expert gate in one product; top-k of the logits,
        lg = h.float() @ wr         # then a softmax over those k = the router's softmax-topk-renormalize
        v, idx = torch.topk(lg[:, :-1], self.gate.top_k, dim=-1)
        w = torch.softmax(v, dim=-1)
        sg = lg[:, -1:]
    else:
        _, w, idx = self.gate(h)
        sg = self.shared_expert_gate(h)
    fut = _SIDE.submit(self.experts, h, idx, w)
    if self._shared_cpu is None:
        (gw, gs), (dw, ds), inter = self._shared_i8
        x = h.float().contiguous()
        n = x.shape[0]
        gu = torch.empty(n, gw.shape[0])
        for r in range(n):
            _gemv_i8(gw, gs, x[r], gu[r])
        a = (self.shared_expert.act_fn(gu[:, :inter]) * gu[:, inter:]).contiguous()
        sh = torch.empty(n, dw.shape[0])
        for r in range(n):
            _gemv_i8(dw, ds, a[r], sh[r])
    else:
        wgu, wd, inter = self._shared_cpu
        gu = h.float() @ wgu
        sh = (self.shared_expert.act_fn(gu[:, :inter]) * gu[:, inter:]) @ wd
    sh = torch.sigmoid(sg) * sh.to(h.dtype)
    return (fut.result() + sh).reshape(B, T, H)


class PackCache:
    """Packed NPU weights on disk, so a model's second load skips reading and quantizing the
    checkpoint (minutes) and just reads the packed bytes (seconds). One file per NPU buffer, keyed
    by the checkpoint (file names, sizes, mtimes) and the packing version; written only while
    plenty of disk stays free (OPENNPU_NPU4_CACHE=0 turns it off)."""
    VERSION = "q4-gemv-4"          # 2: search quantizers (libonpu); 3: odd K packet counts; 4: odd tiles
    KEEP_FREE = 20 * 2**30

    def __init__(self, ckpt_dir):
        import hashlib
        d = Path(ckpt_dir).resolve()
        sig = [self.VERSION, str(d)]
        for f in sorted(d.glob("*.safetensors")) + sorted(d.glob("*.json")):
            st = f.stat()
            sig.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
        self.root = config.CACHE / "npu4" / (d.name + "-" + hashlib.sha1("|".join(sig).encode()).hexdigest()[:12])
        self.enabled = os.environ.get("OPENNPU_NPU4_CACHE", "1") != "0"
        self.hits = self.misses = 0

    def _path(self, name):
        import hashlib
        return self.root / (hashlib.sha1(name.encode()).hexdigest()[:20] + ".bin")

    def load(self, name, arena):
        """Fill arena from the cache; True on a hit."""
        p = self._path(name)
        if not self.enabled or not p.exists() or p.stat().st_size != arena.n:
            self.misses += 1
            return False
        with open(p, "rb") as f:
            f.readinto(memoryview(arena.view))
        arena.used = arena.n
        arena.sync()
        self.hits += 1
        return True

    def save(self, name, arena):
        import shutil
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < arena.n + self.KEEP_FREE:
            return
        p = self._path(name)
        tmp = p.with_suffix(".part")
        with open(tmp, "wb") as f:
            f.write(memoryview(arena.view[:arena.n]))
        tmp.replace(p)


def mv(h, m, x):
    """y [N] = W x for a matrix spread over all 8 columns (h: its prepared dispatch)."""
    return m.gather_n(device(1, m.sym).call(h, m.x_dev(x[None])), 1)[0]


def mv_experts(arena, bases, m, xs):
    """Several experts' [N, K] products, m.ncols columns per expert, 8 // m.ncols experts per
    dispatch. xs: one input per expert ([n_e, K]) or one shared [K]. Returns [n_e, N]."""
    per = G.NC // m.ncols
    n_e = len(bases)
    xs = np.broadcast_to(xs, (n_e, m.K)) if xs.ndim == 1 else xs
    col = np.arange(m.ncols) * m.cb
    out = []
    for e0 in range(0, n_e, per):
        grp = np.asarray(bases[e0:e0 + per], np.int64)
        k = grp.size
        xe = xs[e0:e0 + k]
        if k < per:                                   # spare columns repeat the last expert
            grp = np.concatenate([grp, np.full(per - k, grp[-1])])
            xe = np.concatenate([xe, np.broadcast_to(xe[-1:], (per - k, m.K))])
        offs = (grp[:, None] + col[None, :]).reshape(-1)
        y = arena.dev.run(arena.bo, offs, m.x_dev(xe), m.tiles, m.kc)
        out.append(m.gather_n(y, k))
    return np.concatenate(out)


# ------------------------------------------------------------------ modules
class NPULinearGroup:
    """Sibling Linear layers reading the same input, packed as one matrix (rows stacked) in its
    own NPU buffer: the first call computes all of them for that input, the others take their slice."""

    def __init__(self, names, weights, sizes=None, K=None, cache=None, key=None, sym=False, residual=False):
        """weights: the matrices, or a callable returning them (only called on a cache miss).
        sym: the q4s weight format. residual: rows [W; W - what the NPU makes of W] (two 4-bit
        passes, ~8-bit precision): the two halves are added after each product."""
        self.names = names
        self.sizes = sizes or [w.shape[0] for w in weights]
        self.residual, self.N1 = residual, sum(self.sizes)
        self.m = Mat(self.N1 * (2 if residual else 1), K if K is not None else weights[0].shape[1], G.NC, sym)
        lib = clib()
        self.ch = None
        if lib is not None:
            check_geometry(self.m)
            self.ch = lib.onpu_dense_new2(self.m.N, self.m.K, int(sym))
        if self.ch:
            self.arena = CArena(lib.onpu_dense_buf(self.ch), self.m.nbytes, lambda: lib.onpu_dense_sync(self.ch))
        else:                                       # no libonpu, or its device lacks the shape
            self.ch = None
            self.arena = Arena(self.m.nbytes, sym)
        self.base = 0
        if not (cache and cache.load(key, self.arena)):
            ws = weights() if callable(weights) else weights
            W = np.ascontiguousarray(np.concatenate(ws) if len(ws) > 1 else ws[0], np.float32)
            if residual:
                W = np.concatenate([W, W - quant_dequant(W, self.m.sym)])
            self.arena.put(self.m.pack(W, threads=8))
            self.arena.sync()
            if cache:
                cache.save(key, self.arena)
        self.h = self.h8 = None                     # prepared dispatches (1-token, 8-token)
        self._key, self._y = None, None

    def compute(self, x, i):
        """x: the caller's input tensor itself. Siblings get the same object (q/k/v_proj(h));
        holding a reference keeps its address from being reused by the next token's tensor."""
        if self._key is None or self._key[0] is not x or self._key[1] != x._version:
            if self.ch is not None:                         # the C core: one call for all rows
                xf = np.ascontiguousarray(x.detach().reshape(-1, x.shape[-1]).float().numpy())
                self._y = np.empty((len(xf), self.m.N), np.float32)
                clib().onpu_dense_mv(self.ch, _fp(xf), len(xf), _fp(self._y))
                if self.residual:
                    self._y = self._y[:, :self.N1] + self._y[:, self.N1:]
                self._key = (x, x._version)
                o = sum(self.sizes[:i])
                return self._y[:, o:o + self.sizes[i]]
            if self.h is None:
                self.h = device(1, self.m.sym).prepare(self.arena.bo, [self.base + c * self.m.cb for c in range(G.NC)],
                                          self.m.tiles, self.m.kc)
            xf = x.detach().reshape(-1, x.shape[-1]).float().numpy()
            if len(xf) > 1 and batched_ok(self.m):            # prompt: 8 tokens per weight pass
                if self.h8 is None:
                    self.h8 = device(G.BATCH, self.m.sym).prepare(self.arena.bo, [self.base + c * self.m.cb for c in range(G.NC)],
                                                      self.m.tiles, self.m.kc)
                ys = []
                for r0 in range(0, len(xf), G.BATCH):
                    chunk = xf[r0:r0 + G.BATCH]
                    y = device(G.BATCH, self.m.sym).call(self.h8, self.m.x_dev8(chunk))
                    ys.append(self.m.gather8(y)[0, :len(chunk)])
                self._y = np.concatenate(ys)
            else:
                self._y = np.stack([mv(self.h, self.m, row) for row in xf])
            if self.residual:
                self._y = self._y[:, :self.N1] + self._y[:, self.N1:]
            self._key = (x, x._version)
        o = sum(self.sizes[:i])
        return self._y[:, o:o + self.sizes[i]]

    def full(self, x):
        """All siblings' outputs for x: [rows, N]."""
        self.compute(x, 0)
        return self._y


class NPULinear(nn.Module):
    def __init__(self, group, i, in_features, out_features, bias):
        super().__init__()
        self.group, self.i = group, i
        self.in_features, self.out_features = in_features, out_features
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        shp = x.shape
        y = torch.from_numpy(np.ascontiguousarray(self.group.compute(x, self.i)))
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*shp[:-1], self.out_features)

    def extra_repr(self):
        return f"{self.in_features} -> {self.out_features}, int4 on the NPU"


class NPUExperts(nn.Module):
    """A 3-D expert bank (Qwen3-MoE / Qwen3.5-MoE / gpt-oss style) on the NPU.

    forward(hidden [T, H], indices [T, k], weights [T, k]) like the modules it replaces."""

    def __init__(self, orig, gu, dn, topk, gu_bias=None, dn_bias=None, pool=None, shapes=None, E=None,
                 cache=None, key=None, sym=False):
        """gu, dn: per-expert [out, in] matrices, or a callable returning (gu, dn) (cache misses only;
        then shapes = ((N, K) of gate_up, (N, K) of down) and E are given)."""
        super().__init__()
        gs, ds = shapes if shapes else (gu[0].shape, dn[0].shape)
        self.E = E if E is not None else len(gu)
        self.topk = topk
        self.mgu = Mat(*gs, expert_cols(topk, *gs), sym)
        self.mdn = Mat(*ds, expert_cols(topk, *ds), sym)
        step = self.mgu.nbytes + self.mdn.nbytes
        self.bgu = [e * step for e in range(self.E)]
        self.bdn = [e * step + self.mgu.nbytes for e in range(self.E)]
        self.gu_bias, self.dn_bias = gu_bias, dn_bias
        self.apply_gate = getattr(orig, "_apply_gate", None)
        self.act_fn = getattr(orig, "act_fn", None)
        # the activation libonpu's fused expert FFN can do itself (None: Python does it)
        self.act_kind, self.alpha, self.limit = None, 0.0, 0.0
        if self.apply_gate is not None and hasattr(orig, "alpha") and hasattr(orig, "limit"):
            self.act_kind, self.alpha, self.limit = 1, float(orig.alpha), float(orig.limit)
        elif self.act_fn is torch.nn.functional.silu or type(self.act_fn).__name__ in ("SiLU", "SiLUActivation"):
            self.act_kind = 0
        lib = clib()
        self.cgu = self.cdn = None
        if lib is not None and self._init_c(lib, gu, dn, pool, cache, key, step):
            return
        self.arena = Arena(self.E * step, sym)
        if not (cache and cache.load(key, self.arena)):
            if callable(gu):
                gu, dn = gu()
            packed = (pool.map if pool else map)(lambda e: (self.mgu.pack(gu[e]), self.mdn.pack(dn[e])), range(self.E))
            for a, b in packed:
                self.arena.put(a)
                self.arena.put(b)
            self.arena.sync()
            if cache:
                cache.save(key, self.arena)

    def _init_c(self, lib, gu, dn, pool, cache, key, step):
        """Two libonpu banks (gate_up, down); the disk cache keeps the interleaved [gu_e | dn_e] layout."""
        check_geometry(self.mgu)
        check_geometry(self.mdn)
        E, g, d = self.E, self.mgu.nbytes, self.mdn.nbytes
        self.cffn, perm = None, None
        if FUSE_FFN and hasattr(lib, "onpu_ffn_new") and self.act_kind == 0 and self.gu_bias is None \
                and self.dn_bias is None and self.mgu.ncols == 1 and self.mdn.ncols == 1 and self.topk == G.NC:
            f = lib.onpu_ffn_new(self.mgu.N, self.mgu.K, self.mdn.N, self.mdn.K, E, int(self.mgu.sym))
            if f and lib.onpu_ffn_fused(f):         # gate_up + down in one dispatch per token
                self.cffn, self.cgu, self.cdn = f, lib.onpu_ffn_gu(f), lib.onpu_ffn_dn(f)
                import ctypes as C
                perm = np.empty(self.mgu.N, np.int32)
                lib.onpu_ffn_perm(f, perm.ctypes.data_as(C.POINTER(C.c_int32)))
                key = key.replace("exp", "expf", 1)   # gate_up rows are packed in the fused order
            elif f:
                lib.onpu_ffn_free(f)
        if self.cffn is None:
            self.cgu = lib.onpu_bank_new2(self.mgu.N, self.mgu.K, E, self.mgu.ncols, int(self.mgu.sym))
            self.cdn = lib.onpu_bank_new2(self.mdn.N, self.mdn.K, E, self.mdn.ncols, int(self.mdn.sym))
        if not (self.cgu and self.cdn):              # its device lacks a shape: the pyxrt path
            for h in (self.cgu, self.cdn):
                if h:
                    lib.onpu_bank_free(h)
            self.cgu = self.cdn = None
            return False
        vg = np.ctypeslib.as_array(lib.onpu_bank_buf(self.cgu), shape=(E, g))
        vd = np.ctypeslib.as_array(lib.onpu_bank_buf(self.cdn), shape=(E, d))
        path = cache._path(key) if (cache and cache.enabled) else None
        if path is not None and path.exists() and path.stat().st_size == E * step:
            mm = np.memmap(path, np.uint8, mode="r", shape=(E, step))
            for e0 in range(0, E, 16):
                vg[e0:e0 + 16] = mm[e0:e0 + 16, :g]
                vd[e0:e0 + 16] = mm[e0:e0 + 16, g:]
            del mm
            cache.hits += 1
        else:
            if callable(gu):
                gu, dn = gu()
            def one(e):
                vg[e] = self.mgu.pack(gu[e] if perm is None else np.asarray(gu[e])[perm])
                vd[e] = self.mdn.pack(dn[e])
            list((pool.map if pool else map)(one, range(E)))
            if path is not None:
                import shutil
                path.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(path.parent).free > E * step + PackCache.KEEP_FREE:
                    tmp = path.with_suffix(".part")
                    with open(tmp, "wb") as f:
                        for e in range(E):
                            f.write(memoryview(vg[e]))
                            f.write(memoryview(vd[e]))
                    tmp.replace(path)
        lib.onpu_bank_sync(self.cgu)
        lib.onpu_bank_sync(self.cdn)
        return True

    def _forward_c(self, h, idx, w):
        lib = clib()
        T, k = idx.shape
        ids = np.ascontiguousarray(idx, np.int32)
        import ctypes as C
        ip = ids.ctypes.data_as(C.POINTER(C.c_int32))
        h = np.ascontiguousarray(h, np.float32)
        if T <= FUSE_T_MAX and self.cffn is not None:     # one fused dispatch per token
            wf = np.ascontiguousarray(w, np.float32)
            y = np.empty((T, self.mdn.N), np.float32)
            if all(lib.onpu_ffn_run(self.cffn, _fp(h[t:t + 1]), ids[t].ctypes.data_as(C.POINTER(C.c_int32)), k,
                                    _fp(wf[t:t + 1]), _fp(y[t:t + 1])) for t in range(T)):
                return y
        if T > 1 and self.cffn is not None and FUSE_MANY:      # tokens grouped per expert, both phases per dispatch
            wf = np.ascontiguousarray(w, np.float32)
            y = np.empty((T, self.mdn.N), np.float32)
            if lib.onpu_ffn_run_many(self.cffn, _fp(h), T, ip, k, _fp(wf), _fp(y)):
                return y
        if self.act_kind is not None and hasattr(lib, "onpu_moe_ffn"):
            if not hasattr(self, "_bias_c"):
                f = (lambda b: None if b is None else np.ascontiguousarray(np.asarray(b), np.float32))
                self._bias_c = (f(self.gu_bias), f(self.dn_bias))
            gb, db = self._bias_c
            w = np.ascontiguousarray(w, np.float32)
            y = np.empty((T, self.mdn.N), np.float32)
            lib.onpu_moe_ffn(self.cgu, self.cdn, _fp(h), T, ip, k, _fp(w), self.act_kind, self.alpha, self.limit,
                             None if gb is None else _fp(gb), None if db is None else _fp(db), _fp(y))
            return y
        gu = np.empty((T, k, self.mgu.N), np.float32)
        lib.onpu_bank_run(self.cgu, _fp(h), 0, T, ip, k, _fp(gu))
        if self.gu_bias is not None:
            gu += self.gu_bias[idx]
        act = np.ascontiguousarray(self._act(gu.reshape(T * k, -1)), np.float32)
        dn = np.empty((T, k, self.mdn.N), np.float32)
        lib.onpu_bank_run(self.cdn, _fp(act), 1, T, ip, k, _fp(dn))
        if self.dn_bias is not None:
            dn += self.dn_bias[idx]
        return (dn * w[:, :, None]).sum(1)

    def shapes(self):
        return self.mgu.shapes() | self.mdn.shapes()

    def _act(self, gu):
        t = torch.from_numpy(gu)
        if self.apply_gate is not None:                  # gpt-oss: interleaved, clamped SwiGLU
            return self.apply_gate(t).numpy()
        gate, up = t.chunk(2, dim=-1)
        return (self.act_fn(gate) * up).numpy()

    def _batched(self, m, items, bases, xs_of):
        """items: [(expert, (token, slot) pairs)] with <= 8 tokens each -> outputs per item.
        8 // ncols items share a dispatch (each on its own columns with its own 8-token input)."""
        dev = device(G.BATCH, m.sym)
        per = G.NC // m.ncols
        col = np.arange(m.ncols) * m.cb
        res = []
        for i0 in range(0, len(items), per):
            grp = items[i0:i0 + per]
            pad = grp + [grp[-1]] * (per - len(grp))
            offs = np.concatenate([bases[e] + col for e, _ in pad])
            xin = np.concatenate([m.x_dev8(xs_of(i0 + min(j, len(grp) - 1))) for j in range(per)])
            y = dev.run(self.arena.bo, offs, xin, m.tiles, m.kc)
            g8 = m.gather8(y, len(grp))
            res += [g8[i, :len(p)] for i, (_, p) in enumerate(grp)]
        return res

    def _forward_batched(self, h, idx, w):
        """Many tokens (a prompt): group them by expert, 8 tokens per weight pass."""
        items = []
        for e in np.unique(idx):
            pos = np.argwhere(idx == e)                     # (token, slot)
            for r0 in range(0, len(pos), G.BATCH):
                items.append((int(e), pos[r0:r0 + G.BATCH]))
        gu = self._batched(self.mgu, items, np.asarray(self.bgu, np.int64), lambda i: h[items[i][1][:, 0]])
        acts = []
        for (e, p), g in zip(items, gu):
            if self.gu_bias is not None:
                g = g + self.gu_bias[e]
            acts.append(self._act(np.ascontiguousarray(g, np.float32)))
        dn = self._batched(self.mdn, items, np.asarray(self.bdn, np.int64), lambda i: acts[i])
        out = np.zeros_like(h)
        for (e, p), d in zip(items, dn):
            if self.dn_bias is not None:
                d = d + self.dn_bias[e]
            np.add.at(out, p[:, 0], d * w[p[:, 0], p[:, 1]][:, None])
        return out

    def forward(self, hidden_states, indices, weights):
        h = hidden_states.detach().float().numpy()
        out = np.zeros_like(h)
        idx = indices.detach().cpu().numpy()
        w = weights.detach().float().cpu().numpy()
        if self.cgu is not None:
            return torch.from_numpy(self._forward_c(h, idx, w)).to(hidden_states.dtype)
        if h.shape[0] > 1 and batched_ok(self.mgu) and batched_ok(self.mdn):
            return torch.from_numpy(self._forward_batched(h, idx, w)).to(hidden_states.dtype)
        for t in range(h.shape[0]):
            es = [int(e) for e in idx[t]]
            gu = mv_experts(self.arena, [self.bgu[e] for e in es], self.mgu, h[t])
            if self.gu_bias is not None:
                gu = gu + self.gu_bias[es]
            act = self._act(gu)
            dn = mv_experts(self.arena, [self.bdn[e] for e in es], self.mdn, act)
            if self.dn_bias is not None:
                dn = dn + self.dn_bias[es]
            out[t] = (dn * w[t][:, None]).sum(0)
        return torch.from_numpy(out).to(hidden_states.dtype)


class NPURowsLinear(nn.Module):
    """A Linear that sees many rows per call (a vision tower: one row per image patch). The GEMV
    device re-streams the weights for each row, so the matrix sits on as few columns as its K allows
    (ncols) and each dispatch runs 8 // ncols rows on their own columns. K is zero-padded to 32."""

    def __init__(self, W, bias, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.kp = _pad32(in_features)
        if self.kp != W.shape[1]:
            W = np.pad(W, ((0, 0), (0, self.kp - W.shape[1])))
        self.m = rows_mat(out_features, self.kp)
        self.arena = Arena(self.m.nbytes)
        self.base = self.arena.put(self.m.pack(W))
        self.arena.sync()
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        shp = x.shape
        xf = x.detach().reshape(-1, shp[-1]).float().numpy()
        if self.kp != self.in_features:
            xf = np.pad(xf, ((0, 0), (0, self.kp - self.in_features)))
        y = torch.from_numpy(np.ascontiguousarray(mv_experts(self.arena, [self.base] * xf.shape[0], self.m, xf)))
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(x.dtype).reshape(*shp[:-1], self.out_features)

    def extra_repr(self):
        return f"{self.in_features} -> {self.out_features}, int4 on the NPU, {G.NC // self.m.ncols} rows per dispatch"


class BF16Linear(nn.Module):
    """A CPU Linear computed in bf16 (oneDNN bf16 GEMMs are ~4x faster than fp32 here); fp32 in/out."""

    def __init__(self, lin):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = nn.Parameter(lin.weight.detach().to(torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(lin.bias.detach().float(), requires_grad=False) if lin.bias is not None else None

    def forward(self, x):
        y = torch.nn.functional.linear(x.to(torch.bfloat16), self.weight).to(x.dtype)
        return y + self.bias.to(x.dtype) if self.bias is not None else y

    def extra_repr(self):
        return f"{self.in_features} -> {self.out_features}, bf16 on the CPU"


def _pad32(k):
    return -(-k // 32) * 32


def rows_mat(N, K):
    """Mat for NPURowsLinear: the fewest columns (1, 2, 4, 8) that fit K <= KMAX per slice."""
    for nc in (1, 2, 4, 8):
        try:
            return Mat(N, K, nc)
        except ValueError:
            continue
    raise ValueError(f"[{N}, {K}] does not fit the NPU GEMV device")


def vision_class(full):
    """The image-text-to-text class for a config with a vision tower (None: text-only build)."""
    if getattr(full, "vision_config", None) is None:
        return None
    import transformers
    from transformers import AutoModelForImageTextToText
    try:
        return AutoModelForImageTextToText._model_mapping[type(full)]
    except KeyError:
        for a in getattr(full, "architectures", None) or []:
            cls = getattr(transformers, a, None)
            if cls is not None and not a.endswith("ForCausalLM"):
                return cls
    return None


def _is_vision_module(name):
    return any(p in VISION_PARTS for p in name.split("."))


def expert_cols(topk, N=None, K=None):
    """Columns per expert so a token's experts fill the 8 columns (1 when topk > 8 or uneven); more
    when K needs more slices than that leaves (then fewer experts per dispatch)."""
    c = G.NC // topk if topk <= G.NC and G.NC % topk == 0 else 1
    while N is not None and c < G.NC:
        try:
            Mat(N, K, c)
            break
        except ValueError:
            c *= 2
    return c


# ------------------------------------------------------------------ checkpoint
class Checkpoint:
    """safetensors tensors by name (one shard open at a time is enough)."""

    def __init__(self, d):
        from safetensors import safe_open
        self.d = Path(d)
        idx = self.d / "model.safetensors.index.json"
        if idx.exists():
            self.map = json.loads(idx.read_text())["weight_map"]
        else:
            with safe_open(self.d / "model.safetensors", "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self._open = {}
        self._safe_open = safe_open

    def __contains__(self, k):
        return k in self.map

    def get(self, k):
        f = self.map[k]
        if f not in self._open:
            self._open[f] = self._safe_open(self.d / f, "pt")
        return self._open[f].get_tensor(k)

    def get_f32(self, k):
        return self.get(k).float().numpy()


_FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                np.float32)


def mxfp4(blocks, scales):
    """gpt-oss MXFP4 expert weights ([E, out, K/32, 16] uint8 + [E, out, K/32] exponents) ->
    float32 [E, out, K]."""
    b = np.asarray(blocks, np.uint8)
    lo, hi = _FP4[b & 0x0F], _FP4[b >> 4]
    v = np.empty(b.shape[:-1] + (32,), np.float32)
    v[..., 0::2], v[..., 1::2] = lo, hi
    v *= np.exp2(np.asarray(scales, np.int32) - 127)[..., None].astype(np.float32)
    return v.reshape(*b.shape[:-2], -1)


# ------------------------------------------------------------------ model
def text_config(cfg):
    return getattr(cfg, "text_config", None) or cfg


class NPU4Model:
    """load(ckpt_dir) -> a transformers causal LM whose big matmuls run on the NPU."""

    def __init__(self, ckpt_dir, log=print, threads=8, vision=None):
        """vision: "cpu" / "npu" (where the vision tower's Linears run) or "off" (text-only build);
        default OPENNPU_NPU4_VISION. Only used when the config has a vision tower."""
        from transformers import AutoConfig, AutoModelForCausalLM
        self.dir = Path(ckpt_dir)
        self.log = log
        full = AutoConfig.from_pretrained(self.dir)
        cfg = text_config(full)
        self.cfg, self.full_cfg = cfg, full
        ck = self.ck = Checkpoint(self.dir)
        mode = (vision or VISION).lower()
        cls = vision_class(full) if mode != "off" else None
        model = None
        if cls is not None:                      # the full image-text-to-text model: keys map 1:1
            with torch.device("meta"):
                model = cls._from_config(full, dtype=torch.float32)
            names = [n for n, _ in model.named_parameters()]
            miss = [n for n in names if n not in ck and n != "lm_head.weight"]
            if len(miss) > len(names) // 10:     # another key layout (old llava-style checkpoints)
                self.log(f"[npu4] vision checkpoint keys don't match ({miss[:3]}...): text-only")
                model = None
        self.vision = model is not None
        self.vision_mode = mode if self.vision else "off"
        if model is None:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)   # CPU side: fp32 is fastest
        model.eval()
        # checkpoint key for a model parameter name (VLM checkpoints nest the LM)
        pref = ""
        if not self.vision and not any(k.startswith("model.layers.") for k in ck.map):
            for p in ("model.language_model.", "language_model.model.", "model.text_model."):
                if any(k.startswith(p + "layers.") for k in ck.map):
                    pref = p
                    break
        self.key = (lambda n: pref + n[len("model."):] if pref and n.startswith("model.") else n)
        t0 = time.time()
        self.topk = getattr(cfg, "num_experts_per_tok", 1) or 1
        f = FORMAT if FORMAT in ("q4s", "q4nx") else "q4nx"
        self.sym = f == "q4s"
        npu_bytes = self._plan(model)
        self.log(f"[npu4] {npu_bytes / 2**30:.1f} GiB of int4 weights ({f}) for the NPU")
        ensure_shapes(self._all_shapes, log, self.sym)   # geometry only: build missing GEMV shapes first
        if self._vrow_shapes:
            ensure_shapes(self._vrow_shapes, log)
        pool = ThreadPoolExecutor(threads)
        self.pcache = PackCache(self.dir)
        self._replace(model, pool)
        if self.pcache.hits:
            self.log(f"[npu4] {self.pcache.hits} packed buffers from the disk cache")
        pool.shutdown()
        model.to_empty(device="cpu")                 # only the small CPU-side parameters remain
        self._load_small(model)
        self._fix_buffers(model, cfg)
        for pname, n in self._vcpu:                  # vision Linears that stay on the CPU: bf16
            parent = model.get_submodule(pname) if pname else model
            setattr(parent, n, BF16Linear(getattr(parent, n)))
        self.model = model
        n = fast_cpu_paths(model)
        if n:
            self.log(f"[npu4] {n} linear-attention layers decode through libonpu's fused CPU step")
        n = self._shared_on_cpu(model)
        if n:
            self.log(f"[npu4] {n} shared experts decode on the CPU while the routed experts run on the NPU")
        self.log(f"[npu4] loaded in {time.time() - t0:.0f}s")

    # -- which modules go to the NPU
    def _plan(self, model):
        self._groups, self._experts, self._vrows, self._vcpu = [], [], [], []
        self._vrow_shapes = set()
        shapes, total = set(), 0
        for pname, parent in model.named_modules():
            if self.vision and _is_vision_module(pname):
                for n, c in parent.named_children():
                    if not isinstance(c, nn.Linear):
                        continue
                    if self.vision_mode == "npu" and c.in_features * c.out_features >= MIN_PARAMS:
                        m = rows_mat(c.out_features, _pad32(c.in_features))
                        self._vrows.append((pname, n, c))
                        self._vrow_shapes |= m.shapes()          # q4nx (NPURowsLinear)
                        total += m.nbytes
                    else:
                        self._vcpu.append((pname, n))
                continue
            lin = [(n, c) for n, c in parent.named_children()
                   if isinstance(c, nn.Linear) and c.in_features * c.out_features >= MIN_PARAMS
                   and c.in_features % 32 == 0]
            by_in = {}
            for n, c in lin:
                by_in.setdefault(c.in_features, []).append((n, c))
            for k, items in by_in.items():
                N = sum(c.out_features for _, c in items)
                res = any(p in (f"{pname}.{n}" if pname else n) for n, _ in items for p in HIPREC)
                m = Mat(N * (2 if res else 1), k, G.NC, self.sym)
                self._groups.append((pname, items, k, m))
                shapes |= m.shapes()
                total += m.nbytes
            for n, c in parent.named_children():
                if self._is_experts(c):
                    gu, dn = self._expert_shapes(f"{pname}.{n}" if pname else n, c)
                    mg = Mat(*gu, expert_cols(self.topk, *gu), self.sym)
                    md = Mat(*dn, expert_cols(self.topk, *dn), self.sym)
                    self._experts.append((pname, n, (gu, dn)))
                    shapes |= mg.shapes() | md.shapes()
                    total += c.gate_up_proj.shape[0] * (mg.nbytes + md.nbytes)
        self._all_shapes = shapes
        return total

    @staticmethod
    def _is_experts(c):
        p = dict(c.named_parameters(recurse=False))
        return "gate_up_proj" in p and "down_proj" in p and p["gate_up_proj"].dim() == 3

    def _expert_shapes(self, name, c):
        """((N, K) of gate_up, (N, K) of down) in [out, in] form."""
        E, a, b = c.gate_up_proj.shape
        H = self.cfg.hidden_size
        gu = (a, b) if b == H else (b, a)            # [E, 2I, H] (F.linear) or [E, H, 2I] (x @ W)
        _, a, b = c.down_proj.shape
        dn = (a, b) if a == H else (b, a)
        return gu, dn

    def _shared_on_cpu(self, model):
        """MoE blocks with a small always-on shared expert (Qwen3.5 / Qwen3.6): when decoding, run it on
        the CPU (fp32 copy) at the same time as the routed experts' NPU dispatches, instead of two
        tiny NPU dispatches of its own. Prompts keep the NPU copy. OPENNPU_NPU4_SHARED_CPU=0: off."""
        if os.environ.get("OPENNPU_NPU4_SHARED_CPU", "1") == "0":
            return 0
        import types
        n = 0
        for pname, mod in model.named_modules():
            se, sg = getattr(mod, "shared_expert", None), getattr(mod, "shared_expert_gate", None)
            if se is None or sg is None or not isinstance(getattr(mod, "experts", None), NPUExperts) \
                    or not all(hasattr(se, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")):
                continue
            ks = [self.key(f"{pname}.shared_expert.{p}.weight") for p in ("gate_proj", "up_proj", "down_proj")]
            if not all(k in self.ck for k in ks) or se.gate_proj.out_features * se.gate_proj.in_features > (8 << 20):
                continue
            g, u, d = (torch.from_numpy(np.ascontiguousarray(self.ck.get_f32(k))) for k in ks)
            mod._shared_cpu = (torch.cat([g, u]).T.contiguous(), d.T.contiguous(), g.shape[0])
            if hasattr(clib(), "onpu_gemv_i8"):          # int8 rows: 4x fewer bytes for the CPU to read
                mod._shared_i8 = (_q8(torch.cat([g, u])), _q8(d), g.shape[0])
                mod._shared_cpu = None
            r = getattr(mod, "gate", None)
            if type(r).__name__ == "Qwen3_5MoeTopKRouter" and getattr(r, "bias", None) is None \
                    and isinstance(sg, nn.Linear) and sg.bias is None:
                mod._router_cpu = torch.cat([r.weight.detach().float(), sg.weight.detach().float()]).T.contiguous()
            mod._slow_forward = mod.forward
            mod.forward = types.MethodType(_moe_shared_forward, mod)
            n += 1
        return n

    # -- quantize + pack into NPU buffers, swap the modules
    def _replace(self, model, pool):
        ck, key = self.ck, self.key
        n_done = 0
        for pname, items, k, m in self._groups:
            res = m.N != sum(c.out_features for _, c in items)
            parent = model.get_submodule(pname) if pname else model
            names = [f"{pname}.{n}" if pname else n for n, _ in items]
            bs = []
            for n, c in items:
                bk = key(f"{pname}.{n}.bias" if pname else f"{n}.bias")
                bs.append(ck.get(bk).float() if c.bias is not None and bk in ck else None)
            grp = NPULinearGroup([n for n, _ in items], lambda names=names: [ck.get_f32(key(f"{n}.weight")) for n in names],
                                 sizes=[c.out_features for _, c in items], K=k, cache=self.pcache,
                                 key=("lin-s:" if self.sym else "lin:") + ("r2:" if res else "") + ",".join(names),
                                 sym=self.sym, residual=res)
            for i, (n, c) in enumerate(items):
                setattr(parent, n, NPULinear(grp, i, c.in_features, c.out_features, bs[i]))
            n_done += 1
            if n_done % 20 == 0:
                self.log(f"[npu4] dense layers {n_done}/{len(self._groups)}")
        for pname, n, c in self._vrows:
            parent = model.get_submodule(pname)
            w = ck.get_f32(key(f"{pname}.{n}.weight"))
            bk = key(f"{pname}.{n}.bias")
            b = ck.get(bk).float() if c.bias is not None and bk in ck else None
            setattr(parent, n, NPURowsLinear(w, b, c.in_features, c.out_features))
        if self._vrows:
            self.log(f"[npu4] vision tower: {len(self._vrows)} Linear layers on the NPU")
        for li, (pname, n, (gu_s, dn_s)) in enumerate(self._experts):
            parent = model.get_submodule(pname)
            orig = getattr(parent, n)
            base = key(f"{pname}.{n}")
            gub = ck.get_f32(base + ".gate_up_proj_bias") if base + ".gate_up_proj_bias" in ck else None
            dnb = ck.get_f32(base + ".down_proj_bias") if base + ".down_proj_bias" in ck else None
            setattr(parent, n, NPUExperts(
                orig, lambda base=base, orig=orig, gu_s=gu_s, dn_s=dn_s: self._expert_weights(base, orig, gu_s, dn_s, pool),
                None, self.topk, gub, dnb, pool, shapes=(tuple(gu_s), tuple(dn_s)), E=orig.gate_up_proj.shape[0],
                cache=self.pcache, key=f"{'exp-s' if self.sym else 'exp'}:{pname}.{n}", sym=self.sym))
            self.log(f"[npu4] expert layer {li + 1}/{len(self._experts)}")

    def _expert_weights(self, base, orig, gu_s, dn_s, pool):
        ck = self.ck
        if base + ".gate_up_proj_blocks" in ck:          # gpt-oss MXFP4: [E, out, K/32, 16]
            gu = mxfp4(ck.get(base + ".gate_up_proj_blocks").numpy(), ck.get(base + ".gate_up_proj_scales").numpy())
            dn = mxfp4(ck.get(base + ".down_proj_blocks").numpy(), ck.get(base + ".down_proj_scales").numpy())
        else:
            gu = ck.get(base + ".gate_up_proj").float().numpy()
            dn = ck.get(base + ".down_proj").float().numpy()
            if gu.shape[1:] != tuple(gu_s):
                gu = gu.transpose(0, 2, 1)
            if dn.shape[1:] != tuple(dn_s):
                dn = dn.transpose(0, 2, 1)
        return [np.ascontiguousarray(a) for a in gu], [np.ascontiguousarray(a) for a in dn]

    # -- everything else from the checkpoint
    def _load_small(self, model):
        ck, key = self.ck, self.key
        missing = []
        with torch.no_grad():
            for n, p in list(model.named_parameters()) + list(model.named_buffers()):
                k = key(n)
                if k in ck:
                    p.copy_(ck.get(k).to(p.dtype))
                elif n in ("lm_head.weight",) and getattr(self.cfg, "tie_word_embeddings", False):
                    pass
                else:
                    missing.append(n)
        if getattr(self.cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
            model.tie_weights()
        missing = [m for m in missing if not _owner_is_npu(model, m) and "rotary" not in m and "inv_freq" not in m]
        if missing:
            self.log(f"[npu4] not in the checkpoint (left as initialized): {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    def _fix_buffers(self, model, cfg):
        """Non-persistent buffers (rotary inv_freq) were never materialized: rebuild those modules."""
        for name, mod in list(model.named_modules()):
            if type(mod).__name__.endswith("RotaryEmbedding") and hasattr(mod, "inv_freq"):
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                fresh = type(mod)(mod.config) if hasattr(mod, "config") else None
                if fresh is not None:
                    setattr(parent, name.rsplit(".", 1)[-1], fresh)


def _owner_is_npu(model, pname):
    mod = pname.rsplit(".", 1)[0]
    try:
        return isinstance(model.get_submodule(mod), (NPULinear, NPUExperts, NPURowsLinear))
    except AttributeError:
        return False
