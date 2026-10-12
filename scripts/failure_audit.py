"""Failure attribution for full tidy episodes: for every object that did not end up in the bin, what moved it.

    python scripts/failure_audit.py --seeds 0-99 --perception ground_truth --tag baseline
    python scripts/failure_audit.py --seeds 0-49 --perception yolo --tag baseline --workers 2

The environment records, for each object, the first time it was disturbed before it was ever lifted (slid > 1 cm
or tilted > 10 deg) with the body touching it and the controller phase, and how it last left the hand. This
script runs the expert, groups the failures by outcome x culprit (forearm, wrist, palm, finger, thumb, the object
being carried, another object, the bin, nothing) x phase, and writes data/failure_audit_<perception>_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def culprit_kind(by: str | None, phase: str, carried: str | None) -> str:
    """Coarse class of the body that disturbed an object."""
    from homehand.model.objects import OBJECTS
    if not by or by == "none":
        return "none"
    if by in OBJECTS:
        return "carried_object" if by == carried else "other_object"
    if by == "bin":
        return "bin"
    side = "left" if by.startswith(("left_", "L_")) else "right" if by.startswith(("right_", "R_")) else ""
    acting = phase.split(":")[0] if ":" in phase else ""
    prefix = "" if not acting or side == acting else "idle_hand:"
    if "elbow" in by or "shoulder" in by:
        part = "forearm"
    elif "wrist" in by:
        part = "wrist"
    elif "hand_base" in by:
        part = "palm"
    elif "thumb" in by:
        part = "thumb"
    else:
        part = "finger"
    return prefix + part


def run_seed(args: tuple[int, str, str]) -> dict:
    seed, perception, randomization = args
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode

    env = KitchenEnv()
    pl = TidyPlanner(env, perception=perception)
    r = run_episode(env, pl, seed=seed, randomization=randomization)
    failures = []
    for name, c in r["causes"].items():
        if c["outcome"] == "success":
            continue
        d, rel, tip, out = c.get("disturbed"), c.get("release"), c.get("tipped"), c.get("left_bin")
        f = {"object": name, "outcome": c["outcome"], "rest": dict((l[0], l[-1]) for l in r["layout"]).get(name)}
        # the decisive event first: it came out of the bin again > it tipped / fell > it was nudged before
        # being lifted > it left the hand
        if out or (tip and tip.get("after_bin")):
            e = out or tip
            f.update(cause="bin_exit:" + culprit_kind(e["by"], e["phase"], None), by=e["by"], phase=e["phase"],
                     t=e["t"], ago=e.get("ago"))
        elif tip and (not d or tip["by"] != "none"):
            f.update(cause=culprit_kind(tip["by"], tip["phase"], None), by=tip["by"], phase=tip["phase"],
                     t=tip["t"], ago=tip.get("ago"), fell_off=tip.get("fell_off"),
                     was_target=tip.get("was_target"))
        elif d:
            f.update(cause=culprit_kind(d["by"], d["phase"], d["target"]), by=d["by"], phase=d["phase"],
                     was_target=d["was_target"], t=d["t"], shift=d["shift"], tilt_deg=d["tilt_deg"])
        elif rel and c["outcome"] in ("dropped", "misplaced"):
            f.update(cause="released", phase=rel["phase"], t=rel["t"], palm_speed=rel["palm_speed"],
                     lift=rel["lift"])
        else:
            f.update(cause="undisturbed", last_toucher=c.get("last_toucher"))
        failures.append(f)
    return {"seed": seed, "outcome": r["outcome"], "cleared": r["n_cleared"], "n": r["n_objects"],
            "safety_stop": bool(r["safety"].get("protective_stop")), "failures": failures,
            "disturbed_ok": [  # objects that were nudged but still made it: near misses
                {"object": n, **c["disturbed"]} for n, c in r["causes"].items()
                if c["outcome"] == "success" and c.get("disturbed") and not c["disturbed"]["was_target"]]}


def summarise(res: list[dict]) -> dict:
    from homehand.eval.metrics import wilson
    n = len(res)
    clean = sum(r["outcome"] == "success" for r in res)
    cleared, total = sum(r["cleared"] for r in res), sum(r["n"] for r in res)
    fails = [f for r in res for f in r["failures"]]
    phase = lambda f: (f.get("phase") or "").split(":")[-1] or "-"
    return {
        "episodes": n,
        "clean": wilson(clean, n),
        "objects": wilson(cleared, total),
        "safety_stops": sum(r["safety_stop"] for r in res),
        "by_outcome": dict(Counter(f["outcome"] for f in fails).most_common()),
        "by_cause": dict(Counter(f["cause"] for f in fails).most_common()),
        "by_outcome_cause_phase": dict(Counter(f"{f['outcome']} | {f['cause']} | {phase(f)}"
                                               for f in fails).most_common()),
        "by_object": dict(Counter(f"{f['object']} ({f['rest']})" for f in fails).most_common()),
        "near_misses": dict(Counter(f"{culprit_kind(d['by'], d['phase'], d['target'])} | "
                                    f"{(d['phase'] or '').split(':')[-1]}"
                                    for r in res for d in r["disturbed_ok"]).most_common()),
    }


def main() -> None:
    from homehand.resources import WORKER_MB, WORKER_TORCH_MB, safe_workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--perception", default="ground_truth", choices=["ground_truth", "oracle", "yolo"])
    ap.add_argument("--randomization", default="nominal")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    seeds = []
    for part in a.seeds.split(","):        # "0-99" or "4,12,21" or a mix
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    per = WORKER_TORCH_MB if a.perception == "yolo" else WORKER_MB
    workers = safe_workers(a.workers, per_worker_mb=per)
    print(f"{len(seeds)} episodes, {a.perception}, {a.randomization}, {workers} workers", flush=True)
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(run_seed, [(s, a.perception, a.randomization) for s in seeds]):
            res.append(r)
            if r["failures"]:
                fs = ", ".join(f"{f['object']}:{f['outcome']}<-{f['cause']}@{(f.get('phase') or '-')}"
                               for f in r["failures"])
                print(f"[{len(res)}/{len(seeds)}] seed {r['seed']}: {r['cleared']}/{r['n']}  {fs}", flush=True)
            elif len(res) % 10 == 0:
                print(f"[{len(res)}/{len(seeds)}]", flush=True)
    res.sort(key=lambda r: r["seed"])
    summary = summarise(res)
    print(json.dumps(summary, indent=1))
    out = ROOT / "data" / f"failure_audit_{a.perception}_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "episodes": res}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
