"""Generate RESULTS.md from the evaluation database, training logs and perception benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from homehand import paths
from homehand.eval import store

ROOT = Path(__file__).resolve().parents[1]


def label(r) -> str:
    """Controller column: the run name's first part (it carries variants such as the DDIM step count)."""
    return (r.get("name") or r["controller"]).split(" | ")[0]


def pct(w):
    return f"{100 * w['p']:.0f}% [{100 * w['lo']:.0f}–{100 * w['hi']:.0f}]"


def main():
    runs = [r for r in store.list_runs() if r["status"] == "done" and r["summary"]]
    runs.sort(key=lambda r: r["id"])
    n_rand = sorted({r["summary"]["n_episodes"] for r in runs if r["randomization"] != "nominal"})
    lines = ["# Results", "",
             "Measured on the development laptop (i5-11400H, GTX 1650 4 GB, 16 GB RAM) with `homehand eval` (seeds 0..n-1,",
             "4 objects per episode). Intervals are 95 % Wilson score intervals. *Task success* = every object of the",
             "episode ended in the bin; *objects cleared* = fraction of all objects that ended in the bin.",
             f"Randomised runs use {'/'.join(map(str, n_rand)) or '-'} episodes each, so neighbouring levels have overlapping",
             "intervals; the one-factor table further down is the clearer sim-gap measurement.",
             "Reproduce everything with `python scripts/run_all.py`.", ""]
    lines += ["## Controllers × sim-gap level", "",
              "| run | controller | perception | randomisation | episodes | task success | objects cleared | protective stops |",
              "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r["summary"]
        lines.append(f"| {r['id']} | {label(r)} | {r['perception']} | {r['randomization']} | {s['n_episodes']} | "
                     f"{pct(s['task_success'])} | {pct(s['object_clear_rate'])} | {s['safety']['protective_stops']} |")
    lines += ["", "## Failure taxonomy (objects)", "",
              "| run | " + " | ".join(runs[0]["summary"]["object_outcomes"]) + " |" if runs else "",
              "|---|" + "---|" * (len(runs[0]["summary"]["object_outcomes"]) if runs else 0)]
    for r in runs:
        lines.append(f"| {r['id']} {label(r)} / {r['randomization']} | "
                     + " | ".join(str(v) for v in r["summary"]["object_outcomes"].values()) + " |")
    lines += ["", "## Per object / grasp type (nominal runs)", "", "| run | object | grasp | success |", "|---|---|---|---|"]
    for r in runs:
        if r["randomization"] != "nominal":
            continue
        for o in r["summary"]["per_object"].values():
            lines.append(f"| {r['id']} {label(r)} | {o['label']} | {o['grasp']} | {pct(o)} |")
    gb = paths.DATA_DIR / "grasp_bench.json"
    if gb.exists():
        from homehand.eval.grasp_bench import summarize as gb_table
        lines += ["", "## Grasp benchmark (single object, ground-truth pose)", "",
                  "`homehand grasp-bench --out data/grasp_bench.json`: every object with both hands, inner and outer",
                  "slot, yaws spread over the full circle. Failure modes: *missed* (fingers closed on nothing), *pushed*",
                  "(object swept before the grasp), *slipped* (fell out while carried), *bounced* (left the bin).", "",
                  gb_table(json.loads(gb.read_text()))]
    mq = paths.DATA_DIR / "motion_quality.json"
    if mq.exists():
        q = json.loads(mq.read_text())["summary"]
        lines += ["", "## Motion quality (expert, full tidy episodes, ground-truth poses)", "",
                  "`python scripts/motion_quality.py --out data/motion_quality.json`. Joint acceleration / jerk from the",
                  "measured arm joint positions at 25 Hz; interpenetration = deepest overlap between any robot mesh (visual",
                  "or collision) and the objects, bin or counter.", "",
                  "| episodes | clean | objects cleared | protective stops | mean duration | RMS joint acc. | RMS joint jerk | max interpenetration |",
                  "|---|---|---|---|---|---|---|---|",
                  f"| {q['episodes']} | {q['clean']} | {q['objects_cleared']} | {q['safety_stops']} | {q['mean_seconds']} s | "
                  f"{q['rms_joint_acc']} rad/s² | {q['rms_joint_jerk']} rad/s³ | {q['max_penetration_mm']} mm |"]
    ab = paths.DATA_DIR / "simgap_ablation.json"
    if ab.exists():
        t = json.loads(ab.read_text())["table"]
        lines += ["", "## Sim-gap factors one at a time (expert, ground-truth poses)", "",
                  "`python scripts/simgap_ablation.py --out data/simgap_ablation.json`: each factor fixed at the extreme",
                  "of the `high` preset, everything else nominal. The random presets above mix all factors and use few",
                  "episodes, so this table is the better guide to what the controller depends on.", "",
                  "| factor | clean episodes | objects cleared | protective stops | failures |", "|---|---|---|---|---|"]
        for f, v in t.items():
            fails = ", ".join(f"{k} {c}" for k, c in sorted(v["failures"].items())) or "-"
            lines.append(f"| {f} | {v['clean']}/{v['episodes']} | {v['objects']} | {v['safety_stops']} | {fails} |")
    lines += ["", "## Learning", ""]
    for d in sorted(paths.MODELS_DIR.glob("*/train_log.json")):
        if d.parent.name.startswith("smoke"):
            continue
        log = json.loads(d.read_text())
        last = log["history"][-1] if log["history"] else {}
        lines.append(f"- **{d.parent.name}**: {log['params_M']} M params, {log['steps']} steps, batch {log['batch_size']}, "
                     f"{log['device']}, {log['train_seconds'] / 60:.1f} min, final validation action-MSE "
                     f"{last.get('val_action_mse')}")
    meta = paths.DATASET_DIR / "expert_v1" / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        lines.append(f"- dataset **expert_v1**: {m['n_demos']} successful skill executions / {m['n_frames']} frames from "
                     f"{m['n_tidy_episodes']} tidy episodes; per object {m['per_object']}; per hand {m['per_side']}")
    det = paths.MODELS_DIR / "detector"
    if (det / "metrics.json").exists():
        mm = json.loads((det / "metrics.json").read_text())
        lines += ["", "## Perception", "",
                  f"YOLOv8n-seg on held-out synthetic images: box mAP50 {mm['box_map50']:.3f} (mAP50-95 {mm['box_map50_95']:.3f}), "
                  f"mask mAP50 {mm['mask_map50']:.3f} (mAP50-95 {mm['mask_map50_95']:.3f}).", ""]
        if (det / "bench.json").exists():
            lines += ["| detector | recall | position error mean | p90 | yaw error | upright / lying correct | latency |",
                      "|---|---|---|---|---|---|---|"]
            for b in json.loads((det / "bench.json").read_text()).values():
                lines.append(f"| {b['detector']} | {100 * b['recall']:.1f}% | {b['pos_err_mean_mm']} mm | {b['pos_err_p90_mm']} mm | "
                             f"{b['yaw_err_mean_deg']}° | {100 * b.get('rest_pose_accuracy', 1.0):.0f}% | {b['detect_ms']} ms |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")   # (Windows default: cp1252)
    print(f"wrote {ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
