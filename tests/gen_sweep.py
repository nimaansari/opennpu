"""Run every generation model end to end on the NPU; save outputs and a results table.

    python tests/gen_sweep.py                 # whole catalog (skips what does not fit)
    python tests/gen_sweep.py sd-turbo bark-small
    python tests/gen_sweep.py --task image

Each model runs in its own process (memory is fully returned between models). Results go to
reports/gen/: one output file per model plus results.json and results.md.
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "gen"
PROMPTS = {
    "image": "a photo of a red fox sitting in fresh snow, golden hour, sharp focus",
    "video": "a red fox walking through fresh snow, cinematic",
    "speech": "OpenNPU runs speech generation on the Ryzen AI NPU under Linux.",
}

CHILD = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
from opennpu.gen.runtime import load, png_bytes, mp4_bytes, wav_bytes, NotEnoughMemory, ModelUnavailable
name, prompt, out = sys.argv[2], sys.argv[3], sys.argv[4]
res = {"model": name}
try:
    t0 = time.time()
    eng = load(name, log=lambda m: print(m, file=sys.stderr, flush=True))
    res["load_s"] = round(time.time() - t0, 1)
    res["npu_layers"] = sum(len(r.offloaded) for _, r in eng.report)
    res["npu_params_pct"] = round(100 * sum(r.npu_params for _, r in eng.report) /
                                  max(sum(r.total_params for _, r in eng.report), 1), 1)
    runs = []
    for _ in range(int(sys.argv[5])):
        if eng.task == "speech":
            (audio, sr), st = eng.generate(prompt, seed=0)
        elif eng.task == "video":
            frames, st = eng.generate(prompt, seed=0, frames=9, steps=12)
        else:
            images, st = eng.generate(prompt, seed=0)
        runs.append(st)
    res["runs"] = runs
    if eng.task == "speech":
        open(out + ".wav", "wb").write(wav_bytes(audio, sr)); res["file"] = out + ".wav"
    elif eng.task == "video":
        open(out + ".mp4", "wb").write(mp4_bytes(frames, st["fps"])); res["file"] = out + ".mp4"
    else:
        open(out + ".png", "wb").write(png_bytes(images[0])); res["file"] = out + ".png"
    res["status"] = "ok"
except (NotEnoughMemory, ModelUnavailable) as e:
    res["status"] = "skipped"; res["reason"] = str(e)
except Exception as e:
    res["status"] = "failed"; res["reason"] = f"{type(e).__name__}: {e}"[:500]
print("RESULT " + json.dumps(res), flush=True)
'''


def _mem_cap():
    """Hard cgroup cap per model (systemd-run), so a bad estimate can never OOM the desktop."""
    import shutil
    if not shutil.which("systemd-run"):
        return []
    avail = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    cap = max(4 << 30, avail - (3 << 30))
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0"]


def run_one(py, name, task, runs):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(_mem_cap() + [py, "-c", CHILD, str(ROOT), name, PROMPTS[task], str(OUT / name), str(runs)],
                       capture_output=True, text=True)
    if p.returncode in (-9, 137) and "RESULT " not in p.stdout:
        return {"model": name, "task": task, "status": "failed", "wall_s": round(time.time() - t0, 1),
                "reason": "killed at the memory cap (needs more RAM than was free)"}
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    res = json.loads(line[7:]) if line else {"model": name, "status": "failed",
                                             "reason": (p.stderr or "no output")[-500:]}
    res.update(task=task, wall_s=round(time.time() - t0, 1), peak_rss_gb_max_so_far=round(peak / 2**20, 1))
    return res


def table(results):
    rows = ["| Model | Task | Status | NPU params | Load | First run | Steady run | Output |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        runs = r.get("runs") or []
        first = f"{runs[0]['seconds']} s" if runs else ""
        steady = f"{runs[-1]['seconds']} s" if len(runs) > 1 else ""
        note = r.get("file", "").replace(str(ROOT) + "/", "") if r["status"] == "ok" else r.get("reason", "")[:120]
        rows.append(f"| {r['model']} | {r['task']} | {r['status']} | {r.get('npu_params_pct', '')}% | "
                    f"{r.get('load_s', '')} s | {first} | {steady} | {note} |")
    return "\n".join(rows) + "\n"


def main():
    sys.path.insert(0, str(ROOT))
    from opennpu.gen.catalog import CATALOG
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--task")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    names = a.models or [n for n, e in CATALOG.items() if not a.task or e["task"] == a.task]
    old = {}
    f = OUT / "results.json"
    if f.exists():
        old = {r["model"]: r for r in json.loads(f.read_text())}
    for name in names:
        task = CATALOG[name]["task"] if name in CATALOG else (a.task or "image")
        print(f"== {name} ({task})", flush=True)
        r = run_one(sys.executable, name, task, a.runs)
        print(f"   {r['status']}: {r.get('reason') or r.get('runs')}", flush=True)
        old[name] = r
        OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(list(old.values()), indent=1))
        (OUT / "results.md").write_text(table(list(old.values())))


if __name__ == "__main__":
    main()
