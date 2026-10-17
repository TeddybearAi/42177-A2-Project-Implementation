"""
run_all.py - run every (regime x seed) training job, spreading them over the
available GPUs. Each GPU works through a shared queue one job at a time, so
with two GPUs two runs train in parallel.
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

Finished runs (runs/<run name>/results.json exists) are skipped, so the script can
simply be restarted after an interruption. Each run's console output goes to
runs/logs/<run name>.log.

Example (two GPUs, 3 regimes x 3 seeds = 9 runs)
------------------------------------------------
python run_all.py --pv-root D:/data/PlantVillage --pd-root D:/data/PlantDoc_cropped \
                  --devices cuda:0,cuda:1
EfficientNet-B0 alternative (frozen + full, 3 seeds = 6 runs):
python run_all.py ... --arch efficientnet_b0 --regimes full,frozen
Extra arguments after `--` are passed to every train.py call, e.g.
python run_all.py ... -- --epochs 25 --batch-size 96
"""
import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Longest jobs first so the two GPUs finish at roughly the same time.
REGIME_ORDER = {"full": 0, "layer4": 1, "frozen": 2}


def run_name(arch, regime, seed):
    """Must match train.run_name."""
    return f"{regime}_s{seed}" if arch == "resnet50" else f"effb0_{regime}_s{seed}"


def main():
    """Build the job queue, start one worker thread per GPU and report any failed runs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--pv-root", required=True)
    ap.add_argument("--pd-root", required=True)
    ap.add_argument("--devices", default="cuda:0", help="comma-separated, e.g. cuda:0,cuda:1")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--regimes", default="full,layer4,frozen")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--runs-dir", default="runs")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]

    a.manifest = str(Path(a.manifest).resolve())           # absolute: child processes run from HERE
    runs = Path(a.runs_dir).resolve()
    logs = runs / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = sorted(((r, int(s)) for r in a.regimes.split(",") for s in a.seeds.split(",")),
                  key=lambda j: (REGIME_ORDER.get(j[0], 9), j[1]))
    q = queue.Queue()
    for r, s in jobs:
        if (runs / run_name(a.arch, r, s) / "results.json").exists():
            print(f"skip {run_name(a.arch, r, s)} (already finished)")
        else:
            q.put((r, s))
    failed, lock = [], threading.Lock()

    def worker(device):
        """Take the next job from the queue and run train.py on `device` until the queue is empty."""
        while True:
            try:
                r, s = q.get_nowait()
            except queue.Empty:
                return
            name = run_name(a.arch, r, s)
            cmd = [sys.executable, str(HERE / "train.py"), "--manifest", a.manifest,
                   "--pv-root", a.pv_root, "--pd-root", a.pd_root, "--arch", a.arch, "--regime", r, "--seed", str(s),
                   "--device", device, "--out", str(runs / name)] + extra
            t0 = time.time()
            print(f"[{device}] start {name}  (log: {logs / (name + '.log')})", flush=True)
            with open(logs / f"{name}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            print(f"[{device}] {status} {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if rc != 0:
                with lock:
                    failed.append(name)

    threads = [threading.Thread(target=worker, args=(d.strip(),)) for d in a.devices.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\nall jobs finished" + (f"; failed: {failed} (see runs/logs)" if failed else ""))
    print("next: python aggregate.py")


if __name__ == "__main__":
    main()
