"""Per-token time breakdown of the native decode: host prep vs NPU dispatch vs readback.
    python -m opennpu.native.profile <ckpt> <decode_model> <template_dir> [ctx...]"""
import sys
import time
import numpy as np
from ml_dtypes import bfloat16
from opennpu.native.runtime import NativeDecoder

ck, dm, td = sys.argv[1:4]
ctxs = [int(c) for c in sys.argv[4:]] or [16, 512, 1500]
d = NativeDecoder(ck, dm, td)
xrt = d.xrt
TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
for ctx in ctxs:
    d.reset()
    d.pos = ctx - 1                       # KV contents irrelevant for timing
    T = {"insts": 0, "rope": 0, "x": 0, "npu": 0, "logits": 0}
    n = 20
    for i in range(n):
        p = d.pos
        t0 = time.perf_counter(); ni = d.stair.patch_insts(d._st, p + 1, xrt, TO)
        t1 = time.perf_counter(); r = d._rope(p); d.r_bo.write(r.view(np.int16), d._rope_base * 2); d.r_bo.sync(TO, r.size * 2, d._rope_base * 2)
        t2 = time.perf_counter(); d.x_bo.write(np.asarray(d.embed[100], bfloat16).view(np.int16), 0); d.x_bo.sync(TO)
        t3 = time.perf_counter(); d.kern(3, d.ib, ni, d.x_bo, d.w_bos[0], d.r_bo, d.y_bo, d.kvc, *d.w_bos[1:]).wait(60000)
        t4 = time.perf_counter(); nv = d.UNI_LM * d.VP; d.y_bo.sync(FROM, nv * 2, d.decode_y * 2)
        y = np.frombuffer(d.y_bo.map(), dtype=bfloat16, count=nv, offset=d.decode_y * 2)[:d.VOCAB].astype(np.float32); int(y.argmax())
        t5 = time.perf_counter()
        for k, a, b in (("insts", t0, t1), ("rope", t1, t2), ("x", t2, t3), ("npu", t3, t4), ("logits", t4, t5)):
            T[k] += (b - a) * 1000 / n
        d.pos = p + 1
    tot = sum(T.values())
    print(f"ctx {ctx:5d}: total {tot:6.1f} ms ({1000/tot:5.2f} tok/s) | " + " ".join(f"{k} {v:.2f}" for k, v in T.items()), flush=True)
wb = sum(b.size() for b in d.w_bos) if hasattr(d.w_bos[0], "size") else None
print("weight bytes", wb)
