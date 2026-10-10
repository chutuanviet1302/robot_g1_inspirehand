"""Reproduce every number in RESULTS.md, one stage after the other (Windows and Linux).

    python scripts/run_all.py                 # ~5 h on the development laptop
    python scripts/run_all.py --quick         # small sizes, to check that everything runs (~20 min)

Every stage is skipped when its output already exists (the pipeline keeps its own per-stage status), so after
a reboot the same command continues where it stopped. Evaluation runs left `running` by an interrupted run are
marked `aborted` first, so they never reach the report.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sh(*args: str) -> None:
    print(f"\n=== {' '.join(args)}  [{time.strftime('%H:%M:%S')}]", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    from homehand import paths
    from homehand.eval import store
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = a.quick
    with store.connect() as con:
        con.execute("UPDATE runs SET status='aborted' WHERE status='running'")
    cli = ["-m", "homehand.cli"]
    sh(*cli, "pipeline", "--episodes", "20" if q else "300", "--yolo-images", "100" if q else "1500",
       "--yolo-epochs", "2" if q else "40", "--steps", "500" if q else "25000",
       "--diffusion-steps", "500" if q else "30000", "--eval-n", "4" if q else "50")
    data = paths.DATA_DIR
    if not (data / "grasp_bench.json").exists():
        sh(*cli, "grasp-bench", "-n", "1" if q else "6", "--out", str(data / "grasp_bench.json"))
    if not (data / "motion_quality.json").exists():
        sh("scripts/motion_quality.py", "--seeds", "0-3" if q else "0-23", "--out", str(data / "motion_quality.json"))
    if not (data / "simgap_ablation.json").exists():
        sh("scripts/simgap_ablation.py", "--seeds", "1" if q else "16", "--out", str(data / "simgap_ablation.json"))
    sh("scripts/report.py")


if __name__ == "__main__":
    main()
