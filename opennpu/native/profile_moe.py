"""Per-token time breakdown of the own MoE decode (gather / rms / NPU / route / LM head).
    python -m opennpu.native.profile_moe [ckpt_dir] [n_tokens]      (inside the native env)"""
import os
import sys
import time

from opennpu import config
from opennpu.native.moe_runtime import MoeDecoder

ck = sys.argv[1] if len(sys.argv) > 1 else str(config.MODELS / "Qwen3-30B-A3B-Instruct-2507-npu")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
os.environ.setdefault("OPENNPU_MOE_TEMPLATES", str(config.MOE_TEMPLATES))
d = MoeDecoder(ck)
T = {}


def timed(name, f):
    def g(*a, **k):
        t = time.perf_counter()
        r = f(*a, **k)
        T[name] = T.get(name, 0) + time.perf_counter() - t
        return r
    return g


d._fill_down = timed("gather", d._fill_down)
d._write_rms = timed("rms", d._write_rms)
d._route_with = timed("route", d._route_with)
k0 = d.kern
d.kern = lambda *a: type("Run", (), {"wait": lambda s, t: timed("npu", lambda: k0(*a).wait(t))()})()
for t in (9707, 11, 1879, 374):          # warm up
    d.step(t)
T.clear()
t0 = time.perf_counter()
for i in range(n):
    d.step(100 + i)
tot = (time.perf_counter() - t0) / n * 1000
print(f"total {tot:.1f} ms/token = {1000 / tot:.2f} tok/s")
for k, v in T.items():
    print(f"  {k:7s} {v / n * 1000:6.1f} ms")
print(f"  other   {tot - sum(T.values()) / n * 1000:6.1f} ms  (LM head, host glue)")
