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
    from homehand.model.objects import LYING, OBJECTS, TIDY_OBJECTS
    ik = ArmIK()
    for name in (n for n in LYING if n in TIDY_OBJECTS):     # (the tidy-table task's lying objects)
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


def test_human_motion_profiles():
    """Reach-to-grasp profile: rest at both ends, single speed peak at 40 %; hand pre-shaped by 65 % of the reach."""
    from homehand.control.human_motion import APERTURE_PEAK, ASYM_PEAK, asym_min_jerk, fitts_time, preshape
    tau = np.linspace(0, 1, 2001)
    s = np.array([asym_min_jerk(x) for x in tau])
    v = np.gradient(s, tau)
    assert s[0] == 0 and s[-1] == pytest.approx(1.0) and np.all(np.diff(s) >= -1e-12)
    assert v[0] < 1e-3 and v[-1] < 1e-3
    assert tau[np.argmax(v)] == pytest.approx(0.4, abs=0.01) and v.max() == pytest.approx(ASYM_PEAK, rel=1e-2)
    assert fitts_time(0.4, 0.05) > fitts_time(0.2, 0.05) > fitts_time(0.2, 0.1)
    start, shaped = np.zeros(6), np.ones(6)
    assert np.allclose(preshape(APERTURE_PEAK, start, shaped), shaped)
    assert np.all(preshape(0.3, start, shaped)[1:] < 1) and preshape(0.3, start, shaped)[0] > preshape(0.3, start, shaped)[1]


def test_clearance_detects_neighbour():
    """The arm probes report contact with a box under them and clearance from a far one."""
    from homehand.control.clearance import box_distance, obstacle_boxes
    from homehand.control.skills import Target
    box = obstacle_boxes([Target("sugar_box", np.array([0.3, 0.13, 0.8]), 0.0, 0.175)])[0]
    assert box_distance(np.array([0.3, 0.13, 0.9]), box) < 0                      # inside
    assert box_distance(np.array([0.3, 0.13, 0.8 + 0.175 + 0.05]), box) == pytest.approx(0.05, abs=1e-6)


def test_clearance_probes_follow_the_hand_command():
    """The IK model keeps the fingers at 0: the probes must put them where the commanded hand shape is (the
    open thumb sits ~4 cm from its zero position), and cover the forearm."""
    from homehand.control import clearance as C
    from homehand.control.grasps import GRASPS, LYING_GRASP
    from homehand.control.ik import ArmIK
    ik = ArmIK()
    p0, _ = C.probes(ik, ik.q_home, "left")
    p1, _ = C.probes(ik, C.with_hand(ik, ik.q_home, "left", GRASPS[LYING_GRASP["soup_can"]].open_cmd()), "left")
    assert np.linalg.norm(p1 - p0, axis=1).max() > 0.03
    elbow = ik.cfg.data.xpos[ik.model.body("left_elbow_link").id]
    assert np.linalg.norm(p0 - elbow, axis=1).min() < 0.05


def test_drop_spots_keep_objects_inside_the_bin():
    """Every drop spot keeps the object's whole footprint inside the bin walls (a sugar box dropped with its
    edge over the side wall tipped out of the bin)."""
    from homehand.control.planner import TidyPlanner
    from homehand.control.skills import Target
    from homehand.env.kitchen_env import BIN_HALF_INNER, BIN_XY
    from homehand.model.objects import OBJECTS
    pl = object.__new__(TidyPlanner)
    for name, o in OBJECTS.items():
        r = max(o.half_x, o.half_y)
        for side in ("left", "right"):
            pl.in_bin_xy, pl.used_spots = [], []
            for _ in range(4):
                xy = pl.choose_drop_spot(side, Target(name, np.array([0.3, 0.2, 0.8]), 0.0, 0.1))
                assert np.all(np.abs(xy - BIN_XY) + r <= BIN_HALF_INNER - 0.019), (name, side, xy)


def test_wrist_cameras_look_along_the_fingers():
    import mujoco
    from homehand.model import spec
    m = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side in spec.SIDES:
        cam = m.camera(spec.wrist_camera(side)).id
        look = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
        palm = d.site_xmat[m.site(spec.palm_site(side)).id].reshape(3, 3)
        assert look @ palm[:, 0] > np.cos(np.radians(40)) and look @ palm[:, 2] < 0   # along the fingers, tilted down


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


def test_zero_residual_is_the_expert():
    """Expert + a residual policy that outputs zeros commands exactly what the expert commands."""
    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.rl.residual import ACT_DIM, OBS_DIM
    from homehand.runner import run_episode

    class Zero:
        kind = "residual"
        seen = []

        def act(self, x):
            assert x.shape == (OBS_DIM,) and np.all(np.isfinite(x))
            self.seen.append(1)
            return np.zeros(ACT_DIM)

    layout = [("soup_can", 0.31, -0.18, 0.0)]
    acts = {}
    for name, policy in (("expert", None), ("residual", Zero())):
        env = KitchenEnv()
        planner = TidyPlanner(env, perception="ground_truth", policy=policy)
        acts[name] = []
        run_episode(env, planner, task="pick_place", seed=0, layout=layout,
                    on_step=lambda obs, a, out=acts[name]: out.append(np.array(a)))
    assert Zero.seen, "the residual was never asked"
    assert len(acts["expert"]) == len(acts["residual"])
    assert np.array_equal(np.array(acts["expert"]), np.array(acts["residual"]))


def test_flow_matching_policy_learns_a_mapping():
    """Shapes, and a tiny overfit: two observations -> two constant action chunks; Euler sampling recovers them."""
    import torch
    from homehand.policy.models import MODELS
    torch.manual_seed(0)
    m = MODELS["flow"](obs_dim=4, act_dim=3, chunk=8, n_infer_steps=10, cond_dim=64, dims=(32, 64))
    obs = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]).repeat(32, 1)
    target = torch.stack([torch.full((8, 3), 1.0), torch.full((8, 3), -1.0)]).repeat(32, 1, 1)
    mask = torch.ones(64, 8)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(300):
        loss, _ = m.loss(obs, target, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    out = m.predict(obs[:2], generator=torch.Generator().manual_seed(1))
    assert out.shape == (2, 8, 3)
    assert torch.allclose(out.mean((1, 2)), torch.tensor([1.0, -1.0]), atol=0.2), out.mean((1, 2))


def test_bimanual_expert_lifts_the_box():
    """Both palms squeeze the wide box, lift it, carry it to the goal and put it down (nominal physics)."""
    from homehand import paths
    if not paths.BIMANUAL_XML.exists():
        from homehand.model.build import build_bimanual_scene
        build_bimanual_scene()
    from homehand.control.bimanual import BimanualExpert
    from homehand.env.bimanual_env import BimanualEnv
    from homehand.runner import run_episode
    env = BimanualEnv()
    r = run_episode(env, BimanualExpert(env), task="bimanual_lift", seed=0)
    assert r["outcome"] == "success", env.info.details
    assert not env.safety.stats.protective_stop


def _conveyor_scene():
    from homehand import paths
    if not paths.CONVEYOR_XML.exists():
        from homehand.model.build import build_conveyor_scene
        build_conveyor_scene()


def test_conveyor_carries_objects_at_belt_speed():
    """Objects ride the belt by friction at its speed, straight (no drift across it), and leave at the far end
    as `missed` when nobody picks them; with the belt stopped they stay put."""
    _conveyor_scene()
    from homehand.env.conveyor_env import ConveyorEnv
    env = ConveyorEnv()
    env.reset(seed=0, speed=0.04, objects=["foam_brick", "soup_can"])
    track = []
    done = False
    while not done:
        _, done = env.step(env.home_ctrl)
        if "foam_brick" in env.tracks:
            p = env.object_pose("foam_brick")[0]
            if -0.3 < p[1] < 0.3:
                track.append((env.t, p[0], p[1]))
    a = np.array(track)
    speed = (a[-1, 2] - a[0, 2]) / ((a[-1, 0] - a[0, 0]) * 0.04)
    assert abs(speed - 0.04) < 0.002 and np.ptp(a[:, 1]) < 0.005
    assert env.info.object_outcomes == {"foam_brick": "missed", "soup_can": "missed"}
    env.reset(seed=0, speed=0.0, objects=["foam_brick"])
    p0 = env.object_pose("foam_brick")[0]
    for _ in range(50):
        env.step(env.home_ctrl)
    assert np.linalg.norm(env.object_pose("foam_brick")[0][:2] - p0[:2]) < 0.002    # (it settles ~4 mm onto the belt)


def test_pose6d_from_one_view_and_tracking():
    """ICP pose from oracle masks: the right rest pose and ADD-S < 7 mm for every object of a static belt scene;
    the tracker then predicts where an object on the running belt will be."""
    if not _can_render():
        pytest.skip("no OpenGL context available")
    _conveyor_scene()
    from homehand.env.conveyor_env import BELT_Z, ConveyorEnv
    from homehand.perception.camera import HeadCamera
    from homehand.perception.detector import make_detector
    from homehand.perception.pose6d import estimate_pose, pose_errors, quat2mat
    env = ConveyorEnv()
    cam, det, cache = HeadCamera(env), make_detector("oracle", env), {}
    env.reset(seed=2, speed=0.0)
    for _ in range(10):
        env.step(env.home_ctrl)
    frame = cam.capture(with_seg=True)
    rests = {n: r for n, *_, r in env.layout}
    dets = det.detect(frame)
    assert len(dets) == 4
    for d in dets:
        est = estimate_pose(d, frame, env.model, env.obj_info, BELT_Z, cache)
        p, q = env.object_pose(d.name)
        assert est.rest == rests[d.name]
        assert pose_errors(cache[d.name], est.R, est.pos, quat2mat(q), p)["add_s"] < 0.007
    # tracking on the running belt (the perception pipeline: no fits at the image edge, refits when lost)
    from homehand.perception.conveyor_perception import ConveyorPerception
    per = ConveyorPerception(env, "oracle")
    env.reset(seed=3, speed=0.04, objects=["foam_brick", "gelatin_box"])
    per.reset((0.0, 0.04))
    checked = 0
    for k in range(400):
        env.step(env.home_ctrl)
        if k % 5 == 0:
            per.update(env.t * 0.04)
            for n, tr in per.tracks.items():
                p, q = env.object_pose(n)
                centre = quat2mat(q) @ per.cache[n].mean(0) + p     # (a symmetric object's body origin is not
                if -0.36 < centre[1] < -0.10:                        # observable: compare the centroids)
                    pred = tr.predict(env.t * 0.04)
                    assert pred.rest == rests_of(env)[n]
                    assert np.linalg.norm(pred.centre[:2] - centre[:2]) < 0.02
                    assert abs(tr.velocity[1] - 0.04) < 0.006
                    checked += 1
    assert checked > 20


def rests_of(env) -> dict:
    return {n: r for n, *_, r in env.layout}


def test_grip_env_reward_terms():
    pytest.importorskip("gymnasium")
    from homehand import paths
    if not paths.BIMANUAL_XML.exists():
        from homehand.model.build import build_bimanual_scene
        build_bimanual_scene()
    from homehand.rl.grip import ACT_DIM, OBS_DIM
    from homehand.rl.grip_env import GripEnv
    env = GripEnv(randomization="nominal", seed=0)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs))
    for _ in range(4):
        obs, r, done, _, info = env.step(np.zeros(ACT_DIM))
        assert np.isfinite(r) and abs(r - sum(info["terms"].values())) < 1e-9
        if done:
            break


def test_residual_env_reward_terms():
    pytest.importorskip("gymnasium")
    from homehand.rl.residual import ACT_DIM, OBS_DIM
    from homehand.rl.residual_env import ResidualPickEnv
    env = ResidualPickEnv(randomization="nominal", seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs))
    for _ in range(5):
        obs, r, done, trunc, info = env.step(np.zeros(ACT_DIM))
        assert np.isfinite(r) and abs(r - sum(info["terms"].values())) < 1e-9
        if done or trunc:
            break


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
