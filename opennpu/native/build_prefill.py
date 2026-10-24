"""Build MLIR-AIR's fused prefill device for one model geometry (run by prefill.build, inside the
prefill toolchain env, so it imports nothing from opennpu).

    python build_prefill.py '<Desc fields as JSON>' <out_dir> [jobs]
"""
import json
import os
import sys
import types
from pathlib import Path

ex = Path(os.environ["OPENNPU_PREFILL_AIR"]) / "programming_examples"
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [str(ex)]

from air_examples.llms.shared.fused_prefill.build import main  # noqa: E402
from air_examples.llms.shared.fused_prefill import models  # noqa: E402

fields, out = json.loads(sys.argv[1]), sys.argv[2]
jobs = sys.argv[3] if len(sys.argv) > 3 else "6"
split = fields.pop("down_split", 1)
models.MODELS["opennpu"] = models.Desc(**fields)       # workers fork and inherit it
if split > 1:                     # down's K split into `split` GEMMs (prefill.MAX_K)
    _spec = models.spec

    def spec(desc):
        s = _spec(desc)
        n = -(-desc.d // models.D.NR) * models.D.NR
        shapes = (set(s.shapes) - {(desc.inter, n, 0, 1)}) | {(desc.inter // split, n, 0, 1)}
        return s._replace(shapes=tuple(sorted(shapes)))
    models.spec = spec
main("dense:opennpu", [out, "-j", jobs])
