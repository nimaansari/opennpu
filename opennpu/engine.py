"""XDNA2 universal GEMM engine.

One xclbin, one hardware context for the whole process. Every GEMM of any shape is decomposed into
the prebuilt tiles listed in kernels/<precision>/manifest.json:

* K is zero-padded to a multiple of 1024 and split into 1024-wide chunks, accumulated in fp32 on host.
* N is padded to a multiple of 256 and split greedily into the compiled N tiles (4096 ... 256).
* M is rounded up to the next compiled M bucket; above the largest bucket it is processed in slices.

Precisions:
* fast      bf16 operands, bfp16-emulated multiply, fp32 accumulate/output
* accurate  bf16 operands, native bf16 multiply, fp32 accumulate/output
* int8      W8A8: int8 weights (per-output-channel scale), int8 activations (per-token dynamic
            scale), int32 accumulate/output, rescaled to fp32 on host

Switching between tiles never reconfigures the NPU: the core program is identical for all of them,
only the DMA instruction stream changes. Weights are uploaded once and stay resident.
"""
import json
import os
import threading

import numpy as np

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
PRECISIONS = ("flm", "fast", "accurate", "int8")


class NPUUnavailable(RuntimeError):
    pass


def bits_bf16(x_f32: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit pattern (round to nearest even), as uint16."""
    u = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def quant_rows_int8(x: np.ndarray):
    """Symmetric per-row int8 quantization. Returns (int8 array, fp32 scale per row)."""
    amax = np.abs(x).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


class Weight:
    """A (N, K) weight matrix resident on the NPU as a grid of [k-chunk][n-tile] buffers."""

    def __init__(self, engine, w_nk: np.ndarray):
        e = engine
        self.N, self.K = w_nk.shape
        self.Kp = -(-self.K // e.K) * e.K
        self.Np = -(-self.N // e.n_align) * e.n_align
        self.tiles = []                      # [(n0, n_tile)]
        n0, rest = 0, self.Np
        for t in sorted(e.n_tiles, reverse=True):
            while rest >= t:
                self.tiles.append((n0, t))
                n0 += t
                rest -= t
        self.scale = None
        if e.int8:
            q, s = quant_rows_int8(w_nk)
            wp = np.zeros((self.Np, self.Kp), dtype=np.int8)
            wp[: self.N, : self.K] = q
            self.scale = np.ones(self.Np, dtype=np.float32)
            self.scale[: self.N] = s
        else:
            wp = np.zeros((self.Np, self.Kp), dtype=np.uint16)
            wp[: self.N, : self.K] = bits_bf16(w_nk)
        self.bos = []                        # [kc][ti] -> bo
        for kc in range(self.Kp // e.K):
            row = []
            for n0, t in self.tiles:
                blk = np.ascontiguousarray(wp[n0:n0 + t, kc * e.K:(kc + 1) * e.K])
                row.append(e._upload(blk, e.GID_B))
            self.bos.append(row)
        self.nbytes = wp.nbytes


class Engine:
    GID_INSTS, GID_A, GID_B, GID_C = 1, 3, 4, 5

    def __init__(self, precision="fast"):
        kernel_dir = os.path.join(KDIR, precision)
        if not os.path.isdir(kernel_dir):
            raise NPUUnavailable(f"unknown precision {precision!r}; have {sorted(os.listdir(KDIR))}")
        self.precision = precision
        try:
            import pyxrt as xrt
        except ImportError as e:
            raise NPUUnavailable("pyxrt not found: install XRT python bindings (apt install python3-xrt)") from e
        self.xrt = xrt
        man = json.load(open(os.path.join(kernel_dir, "manifest.json")))
        self.K = man["K"]
        self.m_buckets = man["M_buckets"]
        self.n_tiles = man["N_tiles"]
        self.n_align = min(self.n_tiles)
        self.int8 = man.get("dtype_in", "bf16") == "i8"
        self.a_dtype = np.int8 if self.int8 else np.uint16
        self.c_dtype = np.int32 if self.int8 else np.float32
        try:
            self.dev = xrt.device(0)
            xb = xrt.xclbin(os.path.join(kernel_dir, man["xclbin"]))
            self.dev.register_xclbin(xb)
            self.ctx = xrt.hw_context(self.dev, xb.get_uuid())
            self.kernel = xrt.kernel(self.ctx, xb.get_kernels()[0].get_name())
        except Exception as e:  # noqa: BLE001
            raise NPUUnavailable(f"could not open NPU / load overlay: {e}") from e
        self.insts = {}
        for M in self.m_buckets:
            for N in self.n_tiles:
                raw = np.fromfile(os.path.join(kernel_dir, f"insts_{M}_{N}.bin"), dtype=np.uint32)
                bo = xrt.bo(self.dev, raw.nbytes, xrt.bo.cacheable, self.kernel.group_id(self.GID_INSTS))
                np.frombuffer(bo.map(), dtype=np.uint32)[:] = raw
                bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                self.insts[(M, N)] = (bo, raw.nbytes)
        self._a = {}
        self._c = {}
        self.lock = threading.Lock()
        self.calls = 0
        self.weight_bytes = 0

    # -- buffers -----------------------------------------------------------------------------
    def _upload(self, arr: np.ndarray, gid):
        xrt = self.xrt
        bo = xrt.bo(self.dev, arr.nbytes, xrt.bo.host_only, self.kernel.group_id(gid))
        np.frombuffer(bo.map(), dtype=arr.dtype)[:] = arr.reshape(-1)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.weight_bytes += arr.nbytes
        return bo

    def _buf(self, cache, key, shape, gid, dtype):
        if key not in cache:
            xrt = self.xrt
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            bo = xrt.bo(self.dev, nbytes, xrt.bo.host_only, self.kernel.group_id(gid))
            cache[key] = (bo, np.frombuffer(bo.map(), dtype=dtype).reshape(shape))
        return cache[key]

    def bucket(self, M):
        for b in self.m_buckets:
            if M <= b:
                return b
        return self.m_buckets[-1]

    def weight(self, w_nk: np.ndarray) -> Weight:
        return Weight(self, np.asarray(w_nk, dtype=np.float32))

    # -- compute -----------------------------------------------------------------------------
    def matmul(self, x: np.ndarray, w: Weight) -> np.ndarray:
        """x (M, K) fp32 -> x @ W^T (M, N) fp32."""
        x = np.asarray(x, dtype=np.float32)
        M = x.shape[0]
        out = np.empty((M, w.N), dtype=np.float32)
        step = self.m_buckets[-1]
        with self.lock:
            for m0 in range(0, M, step):
                xs = x[m0:m0 + step]
                out[m0:m0 + xs.shape[0]] = self._matmul_slice(xs, w)
        return out

    def _matmul_slice(self, x, w: Weight):
        xrt = self.xrt
        M = x.shape[0]
        Mb = self.bucket(M)
        if self.int8:
            xq, sx = quant_rows_int8(x)
        else:
            xq, sx = bits_bf16(x), None
        acc = np.zeros((M, w.Np), dtype=np.float32)
        abo, a = self._buf(self._a, Mb, (Mb, self.K), self.GID_A, self.a_dtype)
        for kc, row in enumerate(w.bos):
            k0 = kc * self.K
            kw = min(self.K, w.K - k0)
            a[:M, :kw] = xq[:, k0:k0 + kw]
            a[:M, kw:] = 0
            a[M:] = 0
            abo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            for (n0, t), wbo in zip(w.tiles, row):
                cbo, c = self._buf(self._c, (Mb, t), (Mb, t), self.GID_C, self.c_dtype)
                ibo, ib = self.insts[(Mb, t)]
                r = self.kernel(3, ibo, ib, abo, wbo, cbo).wait()
                if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                    raise RuntimeError(f"NPU kernel failed: {r}")
                cbo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                acc[:, n0:n0 + t] += c[:M]
                self.calls += 1
        if self.int8:
            acc *= sx[:, None]
            acc *= w.scale[None, :]
        return acc[:, : w.N]


class IronWeight:
    def __init__(self, N, K, Np, Kp, packed):
        self.N, self.K, self.Np, self.Kp = N, K, Np, Kp
        self.packed = packed            # XRT tensor, bfp16 blocks in kernel consumption order
        self.nbytes = packed.numpy().nbytes


class IronEngine:
    """Backend on AMD IRON's open-source `iron.operators.flm.GEMM` (Apache-2.0; the operator's name
    comes from the matmul overlay it was ported from, nothing else of that project is used).

    One xclbin for every shape (M/K/N are runtime parameters), bf16 activations, B pre-packed to
    bfp16 in consumption order, fp32 accumulate. ~10-13 TFLOPS at projection shapes on XDNA2.
    Constraints M%256, K%512, N%64 -- handled here by padding. Per-shape instruction streams are
    generated once (~4 s) and cached on disk by IRON.
    """

    M_BUCKETS = (256, 512, 1024, 2048)
    K_ALIGN, N_ALIGN = 512, 64

    def __init__(self):
        try:
            import torch  # noqa: F401
            from iron.operators.flm import GEMM  # noqa: F401
            from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: F401
        except ImportError as e:
            raise NPUUnavailable(f"flm backend needs amd/iron + mlir_aie 1.4.x ({e})") from e
        # IRON compiles into ./build by default, i.e. relative to wherever the user happens to
        # run from: a different cwd meant recompiling (or failing). Use one shared cache.
        from iron.common.base import AIEOperatorBase
        from iron.common.context import AIEContext
        from . import config
        if AIEOperatorBase._default_context is None:
            AIEOperatorBase._default_context = AIEContext(build_dir=config.CACHE / "iron")
        self.precision = "flm"
        self.int8 = False
        self.m_buckets = list(self.M_BUCKETS)
        self.ops = {}            # (Mb, Kp, Np) -> callable
        self._a = {}             # (Mb, Kp) -> XRTTensor
        self._a_used = {}        # (Mb, Kp) -> (rows, cols) written by the last call
        self._c = {}             # (Mb, Np) -> XRTTensor
        self.lock = threading.Lock()
        self.calls = 0
        self.weight_bytes = 0
        self._packer = {}

    def _op(self, Mb, Kp, Np):
        key = (Mb, Kp, Np)
        if key not in self.ops:
            from iron.operators.flm import GEMM
            op = GEMM(M=Mb, K=Kp, N=Np, tile_n=64)
            op.compile()
            self.ops[key] = (op, op.get_callable())
        return self.ops[key]

    def bucket(self, M):
        for b in self.m_buckets:
            if M <= b:
                return b
        return self.m_buckets[-1]

    def weight(self, w_nk):
        import torch
        from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
        w = torch.as_tensor(np.asarray(w_nk, dtype=np.float32))
        N, K = w.shape
        Kp = -(-K // self.K_ALIGN) * self.K_ALIGN
        Np = -(-N // self.N_ALIGN) * self.N_ALIGN
        B = torch.zeros(Kp, Np, dtype=torch.bfloat16)
        B[:K, :N] = w.t().to(torch.bfloat16)
        op, _ = self._op(self.m_buckets[0], Kp, Np)      # packing depends on tile_n/device only
        packed = np.asarray(op.pack_B(B))
        t = XRTTensor(packed, dtype=packed.dtype)
        self.weight_bytes += packed.nbytes
        return IronWeight(N, K, Np, Kp, t)

    def matmul(self, x, w: IronWeight):
        from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
        x = np.asarray(x, dtype=np.float32)
        M = x.shape[0]
        out = np.empty((M, w.N), dtype=np.float32)
        step = self.m_buckets[-1]
        with self.lock:
            for m0 in range(0, M, step):
                xs = x[m0:m0 + step]
                ms = xs.shape[0]
                Mb = self.bucket(ms)
                _, fn = self._op(Mb, w.Kp, w.Np)
                ka = (Mb, w.Kp)
                if ka not in self._a:
                    self._a[ka] = XRTTensor((Mb, w.Kp), dtype=np.uint16)
                A = self._a[ka]
                a = A.data
                a[:ms, :w.K] = bits_bf16(xs)
                a[:ms, w.K:] = 0
                a[ms:] = 0
                self._a_used[ka] = (ms, w.K)
                A._sync_to_device()
                kc = (Mb, w.Np)
                if kc not in self._c:
                    self._c[kc] = XRTTensor((Mb, w.Np), dtype=np.uint16)
                C = self._c[kc]
                fn(A, w.packed, C)
                c = C.numpy()
                out[m0:m0 + ms] = (c[:ms, : w.N].astype(np.uint32) << 16).view(np.float32)
                self.calls += 1
        return out


    def matmul_t(self, x, w: IronWeight):
        """torch in -> torch out (x's dtype). bf16 activations go to the device as raw bits: no
        fp32 round trip and no single-threaded numpy casts, which were most of the per-call cost
        for diffusion-sized GEMMs (~1 GFLOP calls: ~0.5 ms on the NPU, several ms of host glue)."""
        import torch
        from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
        xb = x.detach().to("cpu", torch.bfloat16).contiguous()
        M = xb.shape[0]
        bits = xb.view(torch.int16).numpy().view(np.uint16)
        out = torch.empty((M, w.N), dtype=x.dtype if x.dtype.is_floating_point else torch.float32)
        step = self.m_buckets[-1]
        with self.lock:
            for m0 in range(0, M, step):
                ms = min(step, M - m0)
                Mb = self.bucket(ms)
                _, fn = self._op(Mb, w.Kp, w.Np)
                ka = (Mb, w.Kp)
                if ka not in self._a:
                    self._a[ka] = XRTTensor((Mb, w.Kp), dtype=np.uint16)
                    self._a_used[ka] = (Mb, w.Kp)          # fresh buffer: assume dirty
                A = self._a[ka]
                a = A.data
                a[:ms, :w.K] = bits[m0:m0 + ms]
                # zero only what a previous call may have left non-zero (wider K / more rows)
                um, uk = self._a_used.get(ka, (0, 0))
                if uk > w.K:
                    a[:max(um, ms), w.K:uk] = 0
                if um > ms:
                    a[ms:um] = 0
                self._a_used[ka] = (ms, w.K)
                A._sync_to_device()
                kc = (Mb, w.Np)
                if kc not in self._c:
                    self._c[kc] = XRTTensor((Mb, w.Np), dtype=np.uint16)
                C = self._c[kc]
                fn(A, w.packed, C)
                c = torch.from_numpy(C.numpy()[:ms, :w.N].view(np.int16)).view(torch.bfloat16)
                out[m0:m0 + ms] = c                         # one multithreaded cast
                self.calls += 1
        return out


_ENGINE = None
_ENGINE_ERR = None


def get_engine(precision=None):
    """Process-wide engine, or raises NPUUnavailable (cached).

    precision: "flm" (AMD IRON flm.GEMM, fastest, needs amd/iron), "fast", "accurate" or "int8".
    Defaults to $OPENNPU_PRECISION, else "flm" when IRON is importable, else "fast".
    Fixed for the life of the process (one overlay, one hardware context).
    """
    global _ENGINE, _ENGINE_ERR
    if not precision:
        precision = os.environ.get("OPENNPU_PRECISION")
    if not precision:
        try:
            import iron.operators.flm  # noqa: F401
            precision = "flm"
        except ImportError:
            precision = "fast"
    if _ENGINE is not None and _ENGINE.precision != precision:
        raise RuntimeError(f"engine already running with precision={_ENGINE.precision!r}")
    if _ENGINE is None and _ENGINE_ERR is None:
        try:
            _ENGINE = IronEngine() if precision in ("flm", "iron") else Engine(precision)
        except NPUUnavailable as e:
            _ENGINE_ERR = e
    if _ENGINE is None:
        raise _ENGINE_ERR
    return _ENGINE


FLMEngine, FLMWeight = IronEngine, IronWeight      # old names, kept for code that imported them
