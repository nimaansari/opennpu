"""Build the int4 GEMV device (gemv_device.py) for a set of shapes, inside the prefill toolchain env.

    python build_gemv.py OUT_DIR TILES:KC [TILES:KC ...] [-j N] [--batch 8] [--sym]

--batch 8 builds the 8-token device (gemm8.cc, for prompts) into its own directory: same shapes,
same packed weights, names gemm8_*. --sym builds for the q4s weight format (its own directory).

Per shape: the insts at all-zero column offsets plus one variant per column with that column's
offset moved by DELTA bytes (insts.LinearInsts turns them into insts for any offsets). Every
build must configure the same device (one PDI): the CDOs are compared. Writes manifest.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemv_device as G  # noqa: E402

DELTA = 1 << 16
BATCH = int(os.environ.get("OPENNPU_GEMV_BATCH", "1"))     # set from --batch; workers fork and inherit
SYM = False                                                # set from --sym (the q4s weight format)


def _flags():
    inc = os.environ.get("MLIR_AIE_INSTALL_DIR") or str(Path(shutil.which("aie-opt")).parents[1])
    return ["-O2", "-std=c++20", "--target=aie2p-none-unknown-elf", "-DNDEBUG", "-Wno-parentheses",
            "-Wno-attributes", "-Wno-macro-redefined", "-Wno-empty-body", "-Wno-deprecated-declarations",
            f"-I{inc}/include", "-D__AIE_API_AIE_ADF_HPP__"]


def obj():
    return G.OBJ if BATCH == 1 else G.OBJ8


def build_kernel(out):
    cc = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin" / "clang++"
    src = "gemv_any.cc" if BATCH == 1 else "gemm8.cc"
    extra = [] if BATCH == 1 or os.environ.get("OPENNPU_GEMM8_NATIVE_BF16") else \
        ["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"]       # the fast bf16 mmul (as MLIR-AIR's GEMMs)
    if SYM:
        extra.append("-DGV_SYM")
    subprocess.run([str(cc), *_flags(), *extra, "-c", str(HERE / "kernels" / src), "-o", str(out / obj())],
                   check=True)


def prefix():
    return "gemv" if BATCH == 1 else "gemm8"


def name_of(tiles, kc, col=None):
    return f"{prefix()}_{tiles}_{kc}" + ("" if col is None else f"_c{col}")


def _compile(args):
    out, tiles, kc, col = args
    from air.backend.xrt import XRTBackend
    name = name_of(tiles, kc, col)
    work = Path(out) / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    os.symlink(Path(out) / obj(), work / obj())
    os.chdir(work)
    offs = [0] * G.NC
    if col is not None:
        offs[col] = DELTA
    mod = G.build(tiles, kc, tuple(offs), batch=BATCH, sym=SYM).build(target="npu2")
    be = XRTBackend(verbose=False, omit_while_true_loop=False, output_format="xclbin", kernel_name="MLIR_AIE",
                    instance_name=name)
    be.compile(mod, output_binary_name=name, insts=f"{name}.insts.bin")
    be.unload()
    for ext in (".xclbin", ".insts.bin"):
        shutil.move(work / f"{name}{ext}", Path(out) / f"{name}{ext}")
    cdo = b"".join(p.read_bytes() for p in sorted((work / "air_project/cdo_seg").glob("*.bin")))
    shutil.rmtree(work)
    return name, hashlib.sha256(cdo).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("shapes", nargs="+")
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--base-only", action="store_true", help="debug: no per-column variants")
    ap.add_argument("--batch", type=int, default=1, choices=(1, G.BATCH))
    ap.add_argument("--sym", action="store_true", help="the q4s (symmetric) weight format")
    a = ap.parse_args()
    global BATCH, SYM
    BATCH, SYM = a.batch, a.sym
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"shapes": [], "delta": DELTA, "device": None,
                                                               "format": "q4s" if SYM else "q4nx"}
    if man.get("format", "q4nx") != ("q4s" if SYM else "q4nx"):
        raise SystemExit(f"{out} holds {man.get('format', 'q4nx')} builds")
    have = {tuple(s) for s in man["shapes"]}
    shapes = [tuple(int(v) for v in s.split(":")) for s in a.shapes]
    todo_shapes = [s for s in shapes if s not in have]
    if not (out / obj()).exists():
        build_kernel(out)
    cols = [] if a.base_only else list(range(G.NC))
    todo = [(str(out), t, k, c) for t, k in todo_shapes for c in [None, *cols]]
    with ProcessPoolExecutor(a.j) as ex:
        digests = dict(ex.map(_compile, todo))
    devs = set(digests.values()) | ({man["device"]} if man["device"] else set())
    if len(devs) != 1:
        raise SystemExit(f"GEMV builds disagree on the device configuration: {digests}")
    for t, k in todo_shapes:                   # one xclbin is enough: keep the first, drop the rest
        for c in range(G.NC):
            (out / f"{name_of(t, k, c)}.xclbin").unlink(missing_ok=True)
        if man["device"] or (out / f"{prefix()}.xclbin").exists():
            (out / f"{name_of(t, k)}.xclbin").unlink(missing_ok=True)
        else:
            (out / f"{name_of(t, k)}.xclbin").rename(out / f"{prefix()}.xclbin")
    man.update(shapes=sorted(have | set(todo_shapes)), device=devs.pop(), delta=DELTA)
    man_p.write_text(json.dumps(man, indent=1))
    print(f"built {len(todo_shapes)} shapes into {out}")


if __name__ == "__main__":
    main()
