"""Evaluation metrics: success rates with Wilson score intervals and the failure taxonomy."""

from __future__ import annotations

import math
from collections import Counter

from homehand.control.grasps import GRASPS
from homehand.env.kitchen_env import OUTCOMES
from homehand.model.objects import OBJECTS


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial proportion (well behaved for small n and p near 0 or 1)."""
    if n == 0:
        return {"k": 0, "n": 0, "p": 0.0, "lo": 0.0, "hi": 0.0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, centre - half), 4),
            "hi": round(min(1.0, centre + half), 4)}


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    succ = sum(bool(e["success"]) for e in episodes)
    obj_total = sum(e["n_objects"] for e in episodes)
    obj_ok = sum(e["n_cleared"] for e in episodes)
    episode_outcomes = Counter(e["outcome"] for e in episodes)
    object_outcomes = Counter(o for e in episodes for o in e["object_outcomes"].values())
    per_object = {}
    for name in OBJECTS:
        outs = [e["object_outcomes"][name] for e in episodes if name in e["object_outcomes"]]
        if outs:
            per_object[name] = {"label": OBJECTS[name].label, "grasp": GRASPS[OBJECTS[name].grasp].label,
                                **wilson(sum(o == "success" for o in outs), len(outs)),
                                "outcomes": dict(Counter(outs))}
    per_side = {}
    for e in episodes:
        for ev in e.get("log", []):
            if ev.get("event") == "pick":
                s = per_side.setdefault(ev["side"], {"picks": 0})
                s["picks"] += 1
    return {
        "n_episodes": n,
        "task_success": wilson(succ, n),
        "object_clear_rate": wilson(obj_ok, obj_total),
        "episode_outcomes": {k: episode_outcomes.get(k, 0) for k in OUTCOMES},
        "object_outcomes": {k: object_outcomes.get(k, 0) for k in OUTCOMES},
        "per_object": per_object,
        "picks_per_side": {s: v["picks"] for s, v in per_side.items()},
        "safety": {
            "protective_stops": sum(bool(e.get("safety", {}).get("protective_stop")) for e in episodes),
            "self_collision_steps": sum(e.get("safety", {}).get("self_collision", 0) for e in episodes),
            "high_force_steps": sum(e.get("safety", {}).get("high_force", 0) for e in episodes),
            "workspace_steps": sum(e.get("safety", {}).get("workspace", 0) for e in episodes),
            "rate_limited_steps": sum(e.get("safety", {}).get("clipped_velocity", 0) for e in episodes),
            "max_contact_force_n": max((e.get("safety", {}).get("max_contact_force", 0) for e in episodes), default=0),
        },
        "mean_sim_time_s": round(sum(e["sim_time"] for e in episodes) / max(n, 1), 2),
        "mean_perception_ms": round(sum(e.get("perception_ms", 0) for e in episodes) / max(n, 1), 1),
        "mean_picks": round(sum(sum(ev.get("event") == "pick" for ev in e.get("log", [])) for e in episodes)
                            / max(n, 1), 2),
    }
