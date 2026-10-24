"""Host side of the int4 GEMV device (gemv_device.py): one hw_context, insts per shape, per-column
weight offsets patched per call. Runs in the main env (pyxrt + numpy)."""
import json
import threading
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from . import gemv_device as G

class LinearInsts:
    """insts(offs) = base + sum_c (variant_c - base) * offs[c] / delta (the BD address words are
    linear in each column's offset). Built from the base build and one build per column."""

    def __init__(self, base, variants, delta):
        self.base = np.frombuffer(base, np.uint32).astype(np.int64)
        self.delta = delta
        idx, dd, col = [], [], []
        for c, buf in enumerate(variants):
            w = np.frombuffer(buf, np.uint32).astype(np.int64)
            if w.shape != self.base.shape:
                raise ValueError("GEMV insts change length with the column offset")
            i = np.nonzero(w - self.base)[0]
            idx.append(i)
            dd.append((w - self.base)[i])
            col.append(np.full(i.size, c))
        self.idx, self.d, self.col = np.concatenate(idx), np.concatenate(dd), np.concatenate(col)
        if np.any(np.bincount(self.idx) > 1):
            raise ValueError("two columns patch the same insts word")
        self.base_at = self.base[self.idx]

    def at(self, offs, out=None):
        num = self.d * np.asarray(offs, np.int64)[self.col]
        if np.any(num % self.delta):
            raise ValueError("column offset is not a multiple of the insts' lattice")
        vals = self.base_at + num // self.delta
        if vals.min() < 0 or vals.max() >= 1 << 32:
            raise ValueError("insts word out of range")
        if out is None:
            out = self.base.astype(np.uint32)
        out[self.idx] = vals.astype(np.uint32)
        return out


class GemvDevice:
    """batch 1: the GEMV device (decode); batch 8: the 8-token device (prompts), same weights."""

    def __init__(self, build_dir, batch=1):
        import pyxrt as xrt
        self.xrt = xrt
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.dir = Path(build_dir)
        self.T = batch
        self.prefix = "gemv" if batch == 1 else "gemm8"
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.dev = xrt.device(0)
        self.xb = xrt.xclbin(str(self.dir / f"{self.prefix}.xclbin"))
        self.dev.register_xclbin(self.xb)
        self.ctx = xrt.hw_context(self.dev, self.xb.get_uuid())
        self.kern = xrt.kernel(self.ctx, "MLIR_AIE")
        self.lock = threading.Lock()
        self.shapes = {}
        xn = G.KMAX * self.T
        self.x = self.bo(G.NC * xn * 2, 1)
        self.xm = np.frombuffer(self.x.map(), bfloat16, count=G.NC * xn).reshape(G.NC, xn)
        self.ys = {}
        self.calls, self.dev_s = 0, 0.0

    def bo(self, nbytes, arg):
        return self.xrt.bo(self.dev, max(int(nbytes), 4096), self.xrt.bo.host_only, self.kern.group_id(3 + arg))

    def weights(self, blob):
        """Upload packed weights (uint8) -> a W buffer object."""
        b = self.bo(blob.nbytes, 0)
        np.frombuffer(b.map(), np.uint8, count=blob.nbytes)[:] = blob
        b.sync(self.TO)
        return b

    def _shape(self, tiles, kc):
        key = (tiles, kc)
        if key not in self.shapes:
            if [tiles, kc] not in [list(s) for s in self.man["shapes"]]:
                raise KeyError(f"GEMV shape {key} not built")
            rd = lambda n: (self.dir / f"{n}.insts.bin").read_bytes()  # noqa: E731
            name = f"{self.prefix}_{tiles}_{kc}"
            lin = LinearInsts(rd(name), [rd(f"{name}_c{c}") for c in range(G.NC)], self.man["delta"])
            n = lin.base.size
            ib = self.xrt.bo(self.dev, n * 4, self.xrt.bo.cacheable, self.kern.group_id(1))
            self.shapes[key] = dict(lin=lin, ib=ib, n=n, offs=None, view=None)
            if key not in self.ys:
                ny = G.NC * G.out_tiles(tiles) * G.NR * G.GN * self.T
                y = self.bo(ny * 2, 2)
                self.ys[key] = (y, np.frombuffer(y.map(), bfloat16, count=ny))
        return self.shapes[key]

    def prepare(self, w, offs, tiles, kc):
        """A fixed matrix (offsets never change): its own insts buffer and run, built once."""
        s = self._shape(tiles, kc)
        ib = self.xrt.bo(self.dev, s["n"] * 4, self.xrt.bo.cacheable, self.kern.group_id(1))
        np.frombuffer(ib.map(), np.uint32, count=s["n"])[:] = s["lin"].at(offs)
        ib.sync(self.TO)
        y, ym = self.ys[(tiles, kc)]
        r = self.xrt.run(self.kern)
        for i, v in enumerate((3, ib, s["n"], w, self.x, y)):
            r.set_arg(i, v)
        return {"run": r, "ib": ib, "y": y, "ym": ym, "tiles": tiles, "w": w}

    def call(self, h, x):
        """Run a prepared matrix on x ([K] or [NC, K]); returns y [NC, tiles*NR*GN] f32."""
        with self.lock:
            k = x.shape[-1]
            self.xm[:, :k] = x
            self.xm[:, k:] = 0
            self.x.sync(self.TO)
            t = time.perf_counter()
            h["run"].start()
            st = h["run"].wait(30000)
            self.dev_s += time.perf_counter() - t
            self.calls += 1
            if "COMPLETED" not in str(st):
                raise RuntimeError(f"GEMV dispatch: {st}")
            h["y"].sync(self.FROM)
            return h["ym"].astype(np.float32).reshape(G.NC, -1)

    def run(self, w, offs, x, tiles, kc):
        """y [NC, tiles*NR*GN] (bf16 as f32) = per column c, the packed block at offs[c] of w times
        x[c] (x: [K] for every column, or [NC, K] one input per column)."""
        s = self._shape(tiles, kc)
        with self.lock:
            if s["offs"] != tuple(offs):
                if s["view"] is None:
                    s["view"] = np.frombuffer(s["ib"].map(), np.uint32, count=s["n"])
                    s["view"][:] = s["lin"].base.astype(np.uint32)
                s["lin"].at(offs, s["view"])
                s["ib"].sync(self.TO)
                s["offs"] = tuple(offs)
            k = x.shape[-1]
            self.xm[:, :k] = x
            self.xm[:, k:] = 0
            self.x.sync(self.TO)
            y, ym = self.ys[(tiles, kc)]
            t = time.perf_counter()
            r = self.kern(3, s["ib"], s["n"], w, self.x, y)
            st = r.wait(30000)
            self.dev_s += time.perf_counter() - t
            self.calls += 1
            if "COMPLETED" not in str(st):
                raise RuntimeError(f"GEMV dispatch: {st}")
            y.sync(self.FROM)
            return ym.astype(np.float32).reshape(G.NC, -1)
