#!/usr/bin/env python3
"""Build OpenNPU's universal GEMM overlay for AMD XDNA2 (aie2p / npu2).

One persistent-core bf16 GEMM design at K=1024 compiles to byte-identical core ELFs/CDOs for every
(M, N), so a single xclbin + hw_context serves all shapes; only the runtime DMA sequence (insts)
differs. Any Linear layer of any model is decomposed into these tiles at run time.

Requires the mlir-aie (IRON) + llvm-aie (Peano) wheels and `source mlir-aie/utils/env_setup.sh`.
End users do NOT need this: the generated artifacts ship in opennpu/kernels/.
"""
import argparse, hashlib, json, os, subprocess, sys, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "opennpu", "kernels")
BUILD = os.path.join(HERE, "..", "build", "f32")
K = 1024
M_BUCKETS = [128, 256, 384, 512, 768, 1024]
N_TILES = [256, 512, 1024, 2048, 4096]
# precision -> (dtype_in, dtype_out, bfp16 emulation, k tile)
PRECISIONS = {
    "fast": ("bf16", "f32", "1", 64),       # bf16 via bfp16-emulated mmul (~2x accurate)
    "accurate": ("bf16", "f32", "0", 64),   # native bf16 mmul, fp32 accumulate
    "int8": ("i8", "i32", "0", 128),        # W8A8: per-channel int8 weights, per-token int8 activations
}


def common(precision):
    din, dout, emu, k = PRECISIONS[precision]
    return ["-K", str(K), "-m", "16", "-k", str(k), "-n", "32", "--n-aie-cols", "8", "--b-col-maj", "1",
            "--dtype_in", din, "--dtype_out", dout, "--emulate-bf16-mmul-with-bfp16", emu, "--dev", "npu2"]


def build(M, N, precision):
    stem = os.path.join(BUILD, precision, f"g_{M}_{N}")
    if not os.path.exists(stem + ".bin"):
        subprocess.run([sys.executable, os.path.join(HERE, "gemm_design.py"), "-M", str(M), "-N", str(N), *common(precision),
                        "--xclbin-path", stem + ".xclbin", "--insts-path", stem + ".bin"],
                       cwd=os.path.dirname(stem), check=True, capture_output=True)
    return M, N, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=sorted(PRECISIONS), default="fast")
    precision = ap.parse_args().precision
    out = os.path.join(OUT, precision)
    os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(BUILD, precision), exist_ok=True)
    jobs = [(M, N) for M in M_BUCKETS for N in N_TILES]
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda mn: build(*mn, precision), jobs))
    ref = None
    for M, N, stem in results:
        cdo = open(os.path.join(stem + ".prj", "main_aie_cdo_elfs.bin"), "rb").read()
        h = hashlib.sha256(cdo).hexdigest()
        ref = ref or h
        if h != ref:
            sys.exit(f"core image differs for M={M} N={N}: overlay is not shape-independent")
        open(os.path.join(out, f"insts_{M}_{N}.bin"), "wb").write(open(stem + ".bin", "rb").read())
    M0, N0, stem0 = results[0]
    open(os.path.join(out, "gemm.xclbin"), "wb").write(open(stem0 + ".xclbin", "rb").read())
    json.dump({"arch": "aie2p", "K": K, "M_buckets": M_BUCKETS, "N_tiles": N_TILES, "dtype_in": PRECISIONS[precision][0], "dtype_out": PRECISIONS[precision][1],
               "precision": precision, "core_sha256": ref, "xclbin": "gemm.xclbin"},
              open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"[{precision}] built {len(results)} instruction streams, shared core {ref[:12]}")


if __name__ == "__main__":
    main()
