"""Core tests: model, IK, safety, metrics, grasping, perception, API. Rendering tests skip without OpenGL."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from homehand import paths

pytestmark = pytest.mark.skipif(not paths.SCENE_XML.exists(), reason="run `homehand build-model` first")


def test_model_compiles_and_is_stable():
    import mujoco

    from homehand.model import spec
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    d = mujoco.MjData(m)
    assert [m.actuator(i).name for i in range(m.nu)] == spec.ACTUATORS
    assert m.neq == 12  # 6 mimic joints per Inspire hand
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()


def test_mimic_joints_track():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(paths.ROBOT_XML))
    d = mujoco.MjData(m)
    d.ctrl[m.actuator("R_index_proximal_joint").id] = 1.0
    for _ in range(1500):
        mujoco.mj_step(m, d)
    q = lambda n: d.qpos[m.jnt_qposadr[m.joint(n).id]]
    assert abs(q("R_index_intermediate_joint") - (1.06399 * q("R_index_proximal_joint") - 0.04545)) < 5e-3


def test_ik_reaches_workspace():
    from homehand.control.ik import ArmIK
    ik = ArmIK()
    q, err = ik.solve(ik.q_home, {"right": (np.array([0.30, -0.18, 0.88]), np.array([1.0, 0, 0, 0]))}, iters=300)
    assert err["right"] < 5e-3


def test_wilson_interval():
    from homehand.eval.metrics import wilson
    w = wilson(8, 10)
    assert w["lo"] == pytest.approx(0.4902, abs=1e-3) and w["hi"] == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0)["p"] == 0.0


def test_grasp_geometry_wrist_yaw_bounded():
    from homehand.model.objects import MAX_HAND_YAW, OBJECTS, grasp_geometry
    for yaw in np.linspace(-np.pi, np.pi, 73):
        for o in OBJECTS.values():
            hy, along, towards = grasp_geometry(o, yaw)
            assert abs(hy) <= MAX_HAND_YAW + 1e-9
            assert {round(along, 4), round(towards, 4)} == {round(o.half_x, 4), round(o.half_y, 4)}


def test_grasp_yaw_stays_in_reachable_window():
    from homehand.model.objects import OBJECTS, grasp_geometry, hand_yaw_window
    for side in ("left", "right"):
        for y in (0.12, 0.2, 0.29):
            ys = y if side == "left" else -y
            lo, hi = hand_yaw_window(side, ys)
            for yaw in np.linspace(-np.pi, np.pi, 37):
                for o in OBJECTS.values():
                    hy, _, _ = grasp_geometry(o, yaw, side, ys)
                    assert lo - 1e-9 <= hy <= hi + 1e-9 or not o.box
            # with the inward turn capped at 0 the fingers never point towards the midline
            _, hi0 = hand_yaw_window(side, ys, max_inward=0.0)
            lo0, _ = hand_yaw_window(side, ys, max_inward=0.0)
            assert (hi0 <= 1e-9) if side == "right" else (lo0 >= -1e-9)


def test_min_jerk_and_rounded_path():
    from homehand.control.skills import min_jerk, rounded_path
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0 and min_jerk(0.5) == pytest.approx(0.5)
    eps = 1e-4   # zero velocity at both ends
    assert min_jerk(eps) / eps < 1e-3 and (1 - min_jerk(1 - eps)) / eps < 1e-3
    pts = [np.zeros(3), np.array([0.0, 0.0, 0.2]), np.array([0.2, 0.0, 0.2])]
    path = rounded_path(pts, radius=0.05)
    assert np.allclose(path[0], pts[0]) and np.allclose(path[-1], pts[-1])
    assert np.linalg.norm(path - pts[1], axis=1).min() > 0.01      # the corner is cut ...
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert seg.sum() < 0.4 and seg.max() < 0.16                      # ... and the path is shorter


def test_mass_randomisation_updates_derived_constants():
    """Regression: changing body_mass without mj_setConst left stale subtree masses / inverse weights, which
    made grasped objects slip and the simulation diverge under the sim-gap presets."""
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    b = env.obj_body["soup_can"]
    env.reset("pick_place", seed=0, objects=["soup_can"], randomization={"mass": (1.5, 1.5)})
    assert env.model.body_subtreemass[b] == pytest.approx(1.5 * env.nominal["mass"][b])
    w15 = env.model.body_invweight0[b, 0]
    env.reset("pick_place", seed=0, objects=["soup_can"], randomization="nominal")
    assert env.model.body_subtreemass[b] == pytest.approx(env.nominal["mass"][b])
    assert env.model.body_invweight0[b, 0] == pytest.approx(1.5 * w15, rel=1e-3)


def test_lying_grasps_reachable():
    """Every lying object, both hands, across its spawn range: the top-grasp pose and the drop pose are within
    reach of the IK (the palm-down hand reaches far less than the side grasp; see top_grasp_plan)."""
    from homehand.control.ik import ArmIK
    from homehand.control.skills import Target, grasp_plan
    from homehand.env.kitchen_env import (BIN_HALF_INNER, BIN_XY, LONG_LYING, LONG_LYING_ABSY, LYING_X,
                                          SHORT_LYING_YAW, SPAWN_ZONES)
    from homehand.model.objects import LYING, OBJECTS
    ik = ArmIK()
    for name in LYING:
        o = OBJECTS[name]
        for side, sg in (("left", 1.0), ("right", -1.0)):
            for x in LYING_X:
                absy = LONG_LYING_ABSY[0] if name in LONG_LYING else SPAWN_ZONES["outer"]["absy"][0]
                yaw = np.pi / 2 if name in LONG_LYING else SHORT_LYING_YAW[name][0]
                t = Target(name, np.array([x, sg * absy, 0.8]), yaw, 0.06, rest="lying")
                t.drop_xy = BIN_XY + np.array([-BIN_HALF_INNER[0] + max(o.half_x, o.half_y) + 0.035, sg * 0.08])
                p = grasp_plan(side, t)
                for key, qkey in (("grasp", "quat"), ("drop", "drop_quat")):
                    _, err = ik.solve_best(side, p[key], p[qkey])
                    assert err < 0.01, (name, side, x, key, err)


def test_safety_layer_limits_commands():
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    env.reset("pick_place", seed=0)
    s = env.safety
    crazy = env.last_action + 10.0
    out = s.filter(crazy, env.last_action)
    assert np.all(out <= s.hi + 1e-9)
    assert np.all(np.abs(out - env.last_action) <= s.max_step + 1e-9)


def test_reset_layout_is_collision_free():
    from homehand.env.kitchen_env import KitchenEnv
    env = KitchenEnv()
    from homehand.model.objects import rest_of
    n_lying = 0
    for seed in range(20):
        obs = env.reset("tidy_table", seed=seed)
        for name, x, y, _, rest in env.layout:
            o = obs["objects"][name]
            c = o["centre"] if rest == "lying" else o["pos"]
            # (a lying bottle may roll a little while it settles)
            assert np.linalg.norm(c[:2] - [x, y]) < (0.02 if rest == "lying" else 0.005), (seed, name, rest)
            assert rest_of(name, o["quat"])[0] == rest, (seed, name)   # settled in the pose it was put in
            assert not env.tracks[name].tipped
            n_lying += rest == "lying"
    assert n_lying > 0


def test_dataset_merge_and_load(tmp_path, monkeypatch):
    """Shards of kept demos -> data.npz + meta.json in the format the trainer reads."""
    from homehand.data import record
    from homehand.policy.features import ACT_DIM, OBS_DIM
    monkeypatch.setattr(paths, "DATASET_DIR", tmp_path)
    shards = tmp_path / "t" / "shards"
    shards.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(shards / "seed00000.npz", n=np.array(2),
             obs0=rng.normal(size=(5, OBS_DIM)).astype(np.float32), act0=rng.normal(size=(5, ACT_DIM)).astype(np.float32),
             side0=np.array("left"), object0=np.array("soup_can"),
             obs1=rng.normal(size=(3, OBS_DIM)).astype(np.float32), act1=rng.normal(size=(3, ACT_DIM)).astype(np.float32),
             side1=np.array("right"), object1=np.array("mustard"))
    np.savez(shards / "seed00001.npz", n=np.array(0))
    meta = record.merge("t", sorted(shards.glob("*.npz")))
    (tmp_path / "t" / "meta.json").write_text(__import__("json").dumps(meta))
    data, m = record.load("t")
    assert m["n_demos"] == 2 and m["n_frames"] == 8 and m["n_tidy_episodes"] == 2
    assert m["per_side"] == {"left": 1, "right": 1} and m["demo_length"]["max"] == 5
    assert data["obs"].shape == (8, OBS_DIM) and data["action"].shape == (8, ACT_DIM)
    assert list(data["episode_index"]) == [0] * 5 + [1] * 3
    assert min(m["stats"]["obs_std"]) >= record.STD_FLOOR


def test_inspire_bridge_mapping():
    """Sim joint targets -> RH56DFX normalised commands: order, open/closed ends, inverse, rate limit, NaN hold."""
    from homehand.control.inspire_real import REAL_ORDER, SIM_RANGE, InspireBridge
    b = InspireBridge()
    assert np.allclose(b.to_real(np.zeros(6)), 1.0)                       # sim open -> real 1
    b.last = None
    assert np.allclose(b.to_real(SIM_RANGE), 0.0)                         # sim closed -> real 0
    b.last = None
    q = np.array([0.0, 0.0, 1.47, 0.0, 0.0, 0.0])                         # only the index closed
    r = b.to_real(q)
    assert r[REAL_ORDER.index("index")] == 0.0 and np.allclose(np.delete(r, REAL_ORDER.index("index")), 1.0)
    assert np.allclose(b.to_sim(r), q)
    nxt = b.to_real(np.zeros(6))                                          # one control step later: rate-limited
    assert np.max(np.abs(nxt - r)) <= 3.0 / SIM_RANGE.max() * b.dt + 1e-9
    assert np.allclose(b.to_real([np.nan] * 6), nxt)                      # non-finite -> hold the last command
    assert b.force_registers().max() <= 1000


@pytest.mark.slow
def test_expert_picks_soup_can():
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode
    env = KitchenEnv()
    planner = TidyPlanner(env, perception="ground_truth")
    res = run_episode(env, planner, task="pick_place", seed=0, layout=[("soup_can", 0.31, -0.18, 0.0)])
    assert res["outcome"] == "success", res


def _can_render() -> bool:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        r = mujoco.Renderer(m, 16, 16)
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_render(), reason="no OpenGL context available")
def test_perception_localizes_objects():
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.perception.camera import HeadCamera
    from homehand.perception.detector import OracleDetector
    from homehand.perception.localize import localize
    env = KitchenEnv()
    cam = HeadCamera(env)
    det = OracleDetector(env)
    obs = env.reset("tidy_table", seed=1)
    frame = cam.capture(with_seg=True)
    errs = [np.linalg.norm(localize(d, frame).pos[:2] - obs["objects"][d.name]["pos"][:2]) for d in det.detect(frame)]
    assert len(errs) >= 2 and np.median(errs) < 0.02


def test_api_smoke():
    os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    from fastapi.testclient import TestClient

    from homehand.server.app import create_app
    client = TestClient(create_app())
    cfg = client.get("/api/config").json()
    assert cfg["controllers"][0]["id"] == "expert"
    assert len(cfg["objects"]) == 4
    assert client.get("/api/runs").status_code == 200
    assert "ram_available_mb" in client.get("/api/system").json()
    # the UI's module scripts must be served as JavaScript (Windows registry MIME types say text/plain)
    for js in (paths.WEB_DIST_DIR / "assets").glob("*.js"):
        assert "javascript" in client.get(f"/assets/{js.name}").headers["content-type"]
