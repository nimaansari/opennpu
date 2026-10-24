"""`opennpu doctor`: check every layer of the Linux NPU stack and say exactly how to fix what's missing."""
import grp
import glob
import os
import platform
import shutil
import subprocess
import time

OK, WARN, FAIL = "\033[32m✔\033[0m", "\033[33m!\033[0m", "\033[31m✘\033[0m"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return ""


def checks():
    """Yield (status, title, detail, fix)."""
    rel = platform.release()
    major, minor = (int(x) for x in rel.split(".")[:2])
    yield ((OK if (major, minor) >= (6, 14) else WARN), "Linux kernel", rel,
           None if (major, minor) >= (6, 14) else "kernel >= 6.14 recommended (amdxdna upstream); Ubuntu: linux-oem-24.04")

    cpu = _run(["grep", "-m1", "model name", "/proc/cpuinfo"]).split(":")[-1].strip()
    is_ryzen_ai = "Ryzen AI" in cpu or "Ryzen 7 8" in cpu or "Ryzen 9 8" in cpu
    yield ((OK if is_ryzen_ai else WARN), "CPU", cpu or "unknown",
           None if is_ryzen_ai else "OpenNPU v0.1 targets AMD Ryzen AI (XDNA2: Strix / Krackan / Strix Halo)")

    loaded = "amdxdna" in _run(["lsmod"])
    yield ((OK if loaded else FAIL), "amdxdna driver", "loaded" if loaded else "not loaded",
           None if loaded else "sudo add-apt-repository ppa:amd-team/xrt && sudo apt install amdxdna-dkms && sudo modprobe amdxdna")

    nodes = glob.glob("/dev/accel/accel*")
    yield ((OK if nodes else FAIL), "NPU device node", ", ".join(nodes) or "none",
           None if nodes else "no /dev/accel/accel* — driver did not bind; check `sudo dmesg | grep amdxdna` (firmware?)")

    try:
        in_render = "render" in [grp.getgrgid(g).gr_name for g in os.getgroups()]
    except KeyError:
        in_render = False
    yield ((OK if in_render else FAIL), "render group", "member" if in_render else "not a member",
           None if in_render else "sudo usermod -aG render $USER  (then log out and back in)")

    smi = shutil.which("xrt-smi")
    out = _run([smi, "examine"]) if smi else ""
    npu = next((l.strip() for l in out.splitlines() if "npu" in l.lower() and "|" in l and "BDF" not in l), "")
    fw = next((l.split(":")[-1].strip() for l in out.splitlines() if "NPU Firmware" in l), "")
    yield ((OK if npu else FAIL), "XRT runtime", (npu + (f"  fw {fw}" if fw else "")) or ("xrt-smi missing" if not smi else "no NPU listed"),
           None if npu else "sudo apt install libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu")
    arch_ok = "aie2p" in out
    if npu:
        yield ((OK if arch_ok else WARN), "NPU architecture", "aie2p (XDNA2)" if arch_ok else "not aie2p",
               None if arch_ok else "bundled overlays are built for XDNA2 (aie2p); XDNA1 (Phoenix/Hawk) needs `tools/build_kernels.py --dev npu`")

    try:
        import subprocess
        ml = subprocess.run(["bash", "-c", "ulimit -l"], capture_output=True, text=True).stdout.strip()
        ok = ml == "unlimited"
        yield ((OK if ok else WARN), "memlock limit", ml,
               None if ok else "add '* soft memlock unlimited' and '* hard memlock unlimited' to /etc/security/limits.conf")
    except Exception:  # noqa: BLE001
        pass

    yield from _native_checks()

    try:
        import pyxrt  # noqa: F401
        yield (OK, "pyxrt bindings", pyxrt.__file__, None)
    except ImportError:
        yield (FAIL, "pyxrt bindings", "not importable",
               "sudo apt install python3-xrt   (venv: ln -s /usr/lib/python3/dist-packages/pyxrt*.so $VIRTUAL_ENV/lib/python3*/site-packages/)")
        return

    try:
        import numpy as np
        from .engine import get_engine
        e = get_engine()
        rng = np.random.default_rng(0)
        x = rng.standard_normal((256, 2048), dtype=np.float32)
        w_np = rng.standard_normal((4096, 2048), dtype=np.float32)
        w = e.weight(w_np)
        y = e.matmul(x, w)
        t = time.perf_counter()
        for _ in range(10):
            e.matmul(x, w)
        dt = (time.perf_counter() - t) / 10
        rel = float(np.linalg.norm(y - x @ w_np.T) / np.linalg.norm(x @ w_np.T))
        good = rel < 0.05
        yield ((OK if good else FAIL), f"NPU GEMM smoke test ({e.precision})",
               f"256x2048x4096 in {dt*1e3:.2f} ms = {2*256*2048*4096/dt/1e12:.2f} TFLOPS, rel err {rel:.4f}",
               None if good else "numerical mismatch — please file an issue with `opennpu doctor` output")
    except Exception as ex:  # noqa: BLE001
        yield (FAIL, "NPU GEMM smoke test", str(ex), "see messages above")


def _native_checks():
    """The fused engines + server run in their own env (pyxrt + mlir-air). An OS upgrade that
    swaps the system Python breaks that env silently: the server then crash-loops on import."""
    from . import config
    env = config.AIRENV
    if not env.exists():
        yield (WARN, "native engine env", f"{env} missing",
               "optional (own fused NPU engine + server): run ./install.sh --native (README, Install), or set OPENNPU_AIRENV")
        return
    py = env.parent / "bin" / "python"          # dev layout: env.sh lives inside the venv
    if os.path.islink(py) and not py.exists():
        yield (FAIL, "native engine env", f"interpreter gone: {py} -> {os.path.realpath(py)}",
               "the system Python was replaced (OS upgrade?). Repoint it: "
               "uv python install 3.12 && ln -sfn $(uv python find 3.12) " + str(py))
        return
    probe = _run(["bash", "-c", f"set +u; source {env} >/dev/null 2>&1; python3 -c "
                  "'import pyxrt, numpy, transformers; print(\"ok\")'"]).strip()
    yield ((OK if probe == "ok" else FAIL), "native engine env", str(env) if probe == "ok" else "imports fail (pyxrt/numpy/transformers)",
           None if probe == "ok" else f"source {env} && python3 -c 'import pyxrt'  to see the error; "
           "pyxrt must match the env's Python (python3-xrt for 3.12)")
    pre_ok = (config.PREFILL_ENV.exists()
              and (config.PREFILL_AIR / "programming_examples/llms/shared/fused_prefill/dense.py").exists())
    yield ((OK if pre_ok else WARN), "batched prefill toolchain",
           str(config.PREFILL_AIR) if pre_ok else "missing: prompts feed one token at a time (slow first token)",
           None if pre_ok else "./install.sh --native")
    from opennpu.native import llamacpp
    b = llamacpp.server_bin()
    yield ((OK if b else WARN), "llama.cpp NPU backend", str(b) if b else "not built: GGUF models are unavailable",
           None if b else "llama/build.sh   (or ./install.sh --native)")
    lib = config.HOME / "lib/libonpu.so"
    yield ((OK if lib.exists() else WARN), "libonpu (4-bit engine C core)",
           str(lib) if lib.exists() else "not built: the 4-bit engine runs its slower Python path",
           None if lib.exists() else "llama/build.sh   (or ./install.sh --native)")
    import urllib.request
    try:
        urllib.request.urlopen(config.URL + "/health", timeout=2)
        yield (OK, "model server", f"{config.URL}  (web UI at {config.URL}/)", None)
    except Exception:  # noqa: BLE001
        state = _run(["systemctl", "--user", "is-active", config.SERVICE]).strip() or "not installed"
        yield (WARN, "model server", f"not reachable (service: {state})",
               f"npu start   (logs: journalctl --user -u {config.SERVICE} -n 50)")


def main(argv=None):
    import json
    import sys
    as_json = "--json" in (argv if argv is not None else sys.argv[1:])
    if as_json:
        res = [{"status": {OK: "ok", WARN: "warn", FAIL: "fail"}[st], "check": t, "detail": d, "fix": f}
               for st, t, d, f in checks()]
        print(json.dumps(res, indent=1))
        return 1 if any(r["status"] == "fail" for r in res) else 0
    print("OpenNPU doctor\n")
    bad = 0
    for st, title, detail, fix in checks():
        print(f" {st} {title:28s} {detail}")
        if fix:
            print(f"     → {fix}")
            bad += st == FAIL
    print("\nAll good — your NPU is ready." if not bad else f"\n{bad} problem(s) to fix.")
    return 1 if bad else 0
