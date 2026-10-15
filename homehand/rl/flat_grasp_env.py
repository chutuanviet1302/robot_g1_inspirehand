"""Flat-box grasping: the RL environment (MJX, for training on a GPU) and its plain-MuJoCo twin (evaluation, the
expert baseline, videos).

The arm starts in the expert's pre-grasp pose (`flat_grasp_model`). Each episode draws:

* the box: +-3 cm around the nominal spot, yaw +-0.6 rad;
* what the camera tells about it: a constant per-episode bias (6 mm in x / y, 3 mm in z, 4 deg in yaw: the size of
  the pose errors of the conveyor / tidy perception), plus 2 mm of noise per frame;
* physics: grip friction (the finger pads) 0.6-1.5, box mass 80-300 g.

Both controllers see the same thing: the joints, the palm and fingertips, the observed box pose, and which fingers
touch the box (the hand's force sensors). The expert plans once from the observed pose (as `skills.top_grasp_plan`):
turn, come down, close, lift - joint targets from the arm IK. The policy acts at 50 Hz on the same observation, so it
can feel where the box really is and how it moves in the hand.

Action (13, in [-1, 1]): rate of change of the position targets of the 7 right-arm joints and the 6 hand actuators.
Success: at the end of the 5 s episode the box is >= 6 cm above the counter, held by the thumb and a finger.

The observation / reward code below takes the array module (`np` or `jax.numpy`) so both twins share it; it only
needs `flat_grasp_model`'s two generated files (no `homehand` import: it is copied into the Kaggle notebook).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DT = 0.02                          # policy at 50 Hz
EPISODE_STEPS = 250                # 5 s
TABLE_Z = 0.80
BOX_XY_RANGE, BOX_YAW_RANGE = 0.03, 0.6
BIAS_XY, BIAS_Z, BIAS_YAW, NOISE = 0.006, 0.003, 0.07, 0.002
GRIP_FRICTION = (0.6, 1.5)
BOX_MASS = (0.08, 0.30)
LIFT_OK = 0.06
FINGER_OPEN = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # thumb turned in, as the expert's pre-grasp shape
FINGER_CLOSE = np.array([1.0, 0.6, 1.47, 1.47, 1.47, 1.47])  # grasps.py: lying_block_pinch
N_ARM = 7
RATE = np.array([1.5] * N_ARM + [4.0] * 6)                   # target change per second at |action| = 1
PARTS = ("thumb", "index", "middle", "ring", "pinky")
OBS_SIZE = 73


@dataclass
class Layout:
    """Indices into the model, and the pre-grasp the episode starts from."""
    act_qadr: np.ndarray
    act_dadr: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    box_qadr: int
    box_dadr: int
    box_body: int
    box_geom: int
    pad_geoms: np.ndarray         # the finger pads (contact priority 1: theirs is the grip friction)
    part_of_geom: np.ndarray      # geom -> 0..4 finger (thumb first), 5 palm, -1 other
    palm_site: int
    tip_sites: np.ndarray
    qpos0: np.ndarray
    ctrl0: np.ndarray
    box_xy: np.ndarray
    z_rest: float
    palm_pos: np.ndarray          # the pre-grasp palm pose (the expert plans from it)
    palm_quat: np.ndarray
    heading_axis: int             # the palm axis that lies flattest at the pre-grasp: the hand's heading
    heading0: float

    @classmethod
    def of(cls, m: mujoco.MjModel, frame_json: str | Path) -> "Layout":
        frame = json.loads(Path(frame_json).read_text())
        joints = m.actuator_trnid[:, 0]
        part = -np.ones(m.ngeom, dtype=np.int32)
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = m.body(m.geom_bodyid[g]).name
            for k, name in enumerate(PARTS):
                if body.startswith(f"R_{name}_"):
                    part[g] = k
            if body == "R_hand_base_link":
                part[g] = 5
        pads = np.flatnonzero((part >= 0) & (m.geom_priority > 0)).astype(np.int32)
        qadr = m.jnt_qposadr[joints].astype(np.int32)
        qpos0 = m.qpos0.copy()
        ctrl0 = np.r_[frame["arm_q"], FINGER_OPEN]
        qpos0[qadr] = ctrl0
        d = mujoco.MjData(m)
        d.qpos[:] = qpos0
        mujoco.mj_kinematics(m, d)
        palm = m.site("R_palm").id
        R = d.site_xmat[palm].reshape(3, 3)
        axis = int(np.argmin(np.abs(R[2, :])))
        return cls(
            act_qadr=qadr, act_dadr=m.jnt_dofadr[joints].astype(np.int32),
            lo=m.actuator_ctrlrange[:, 0].copy(), hi=m.actuator_ctrlrange[:, 1].copy(),
            box_qadr=int(m.jnt_qposadr[m.joint("box_free").id]), box_dadr=int(m.jnt_dofadr[m.joint("box_free").id]),
            box_body=m.body("box").id,
            box_geom=m.geom("box").id, pad_geoms=pads, part_of_geom=part, palm_site=palm,
            tip_sites=np.array([m.site(f"R_{p}_tip").id for p in PARTS], dtype=np.int32),
            qpos0=qpos0, ctrl0=ctrl0, box_xy=np.array(frame["box_xy"]), z_rest=TABLE_Z + frame["box_half"][2],
            palm_pos=np.array(frame["palm_pos"]), palm_quat=np.array(frame["palm_quat"]),
            heading_axis=axis, heading0=float(np.arctan2(R[1, axis], R[0, axis])))


# ---------------------------------------------------------------------------------------- shared (np / jnp)
def box_yaw(xp, quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def observe(xp, lay: Layout, qpos, qvel, site_xpos, site_xmat, ctrl, touch, box_obs, yaw_obs, t_frac):
    """Policy input (73): positions / speeds / targets of the 13 actuated joints, the observed box from the palm and
    from each fingertip, the palm's orientation, the box's heading against the hand's, its observed height, touch,
    time."""
    mid, half = (lay.hi + lay.lo) / 2, (lay.hi - lay.lo) / 2
    q = qpos[lay.act_qadr]
    palm = site_xpos[lay.palm_site]
    tips = site_xpos[lay.tip_sites]
    R = site_xmat[lay.palm_site].reshape(3, 3)
    a = lay.heading_axis
    heading = xp.arctan2(R[1, a], R[0, a]) - lay.heading0
    dyaw = 2 * (yaw_obs - heading)                    # the box looks the same turned by half a turn
    return xp.concatenate([
        (q - mid) / half, 0.1 * qvel[lay.act_dadr], (ctrl - mid) / half,
        10 * (box_obs - palm), 10 * (box_obs[None, :] - tips).reshape(-1), R[:, 0], R[:, 2],
        xp.array([xp.sin(dyaw), xp.cos(dyaw), 10 * (box_obs[2] - lay.z_rest)]), touch, xp.array([t_frac])])


def reward(xp, lay: Layout, qpos, site_xpos, touch, action, box_xy0):
    """Shaped: reach (palm and fingertips to the box), grip (thumb + fingers on it), lift while gripped, success;
    small costs for pushing the box away on the counter and for jerky actions."""
    box = qpos[lay.box_qadr:lay.box_qadr + 3]
    d_palm = xp.linalg.norm(box - site_xpos[lay.palm_site])
    d_tips = xp.linalg.norm(box[None, :] - site_xpos[lay.tip_sites], axis=-1).mean()
    thumb, fingers = touch[0], touch[1:5].sum()
    held = thumb * (fingers > 0)
    h = xp.clip(box[2] - lay.z_rest, 0.0, 0.10)
    pushed = xp.linalg.norm(box[:2] - box_xy0) * (h < 0.01)
    success = held * (box[2] - lay.z_rest > LIFT_OK)
    r = (0.5 * (1 - xp.tanh(10 * d_palm)) + 0.5 * (1 - xp.tanh(10 * d_tips))
         + 0.5 * thumb + 0.25 * xp.minimum(fingers, 2.0)
         + 10.0 * held * h / 0.10 + 2.0 * success
         - 5.0 * pushed - 0.01 * xp.sum(action ** 2))
    return r, success


def draw_episode(rng: np.random.Generator) -> dict:
    """Box pose, perception bias and physics of one episode (numpy twin; the MJX env draws the same laws)."""
    return {"xy": rng.uniform(-BOX_XY_RANGE, BOX_XY_RANGE, 2), "yaw": rng.uniform(-BOX_YAW_RANGE, BOX_YAW_RANGE),
            "bias": np.r_[rng.normal(0, BIAS_XY, 2), rng.normal(0, BIAS_Z), rng.normal(0, BIAS_YAW)],
            "grip_friction": rng.uniform(*GRIP_FRICTION), "mass": rng.uniform(*BOX_MASS)}


def touch_np(m: mujoco.MjModel, d: mujoco.MjData, box_geoms, part_of) -> np.ndarray:
    """Which hand parts touch the box (thumb, index .. pinky, palm); `part_of`: geom -> part index or None."""
    out = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0.001:
            continue
        other = c.geom2 if c.geom1 in box_geoms else c.geom1 if c.geom2 in box_geoms else None
        if other is None:
            continue
        k = part_of(other)
        if k is not None and k >= 0:
            out[k] = 1.0
    return out


# ---------------------------------------------------------------------------------------- plain MuJoCo twin
class FlatGraspSim:
    """CPU twin of the training env: same model, same laws, same observation."""

    def __init__(self, xml: str | Path, frame_json: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.lay = Layout.of(self.model, frame_json)
        self.n_sub = round(DT / self.model.opt.timestep)
        self._mass0 = self.model.body_mass[self.lay.box_body]
        self._inertia0 = self.model.body_inertia[self.lay.box_body].copy()

    def reset(self, seed: int) -> np.ndarray:
        m, d, lay = self.model, self.data, self.lay
        self.rng = np.random.default_rng(seed)
        ep = self.ep = draw_episode(self.rng)
        m.geom_friction[lay.pad_geoms, 0] = ep["grip_friction"]
        m.body_mass[lay.box_body] = ep["mass"]
        m.body_inertia[lay.box_body] = self._inertia0 * ep["mass"] / self._mass0
        # masses changed: the derived constants too (the constraint solver's inverse weights; stale ones made
        # grasped boxes jump and the simulation diverge)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = lay.qpos0
        a = lay.box_qadr
        d.qpos[a:a + 3] = [*(lay.box_xy + ep["xy"]), lay.z_rest + 0.0005]
        d.qpos[a + 3:a + 7] = [np.cos(ep["yaw"] / 2), 0, 0, np.sin(ep["yaw"] / 2)]
        self.ctrl = lay.ctrl0.copy()
        d.ctrl[:] = self.ctrl
        mujoco.mj_forward(m, d)
        self.box_xy0 = d.qpos[a:a + 2].copy()
        self.t = 0
        return self.observe()

    # the controller-facing interface (the robot twin in `flat_grasp_deploy` has the same)
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.lay.act_qadr[:N_ARM]].copy()

    def touch(self) -> np.ndarray:
        return touch_np(self.model, self.data, {self.lay.box_geom}, lambda g: self.lay.part_of_geom[g])

    def box_seen(self) -> tuple[np.ndarray, float]:
        a, b = self.lay.box_qadr, self.ep["bias"]
        pos = self.data.qpos[a:a + 3] + b[:3] + self.rng.normal(0, NOISE, 3)
        return pos, float(box_yaw(np, self.data.qpos[a + 3:a + 7]) + b[3])

    def box_true(self) -> np.ndarray:
        return self.data.qpos[self.lay.box_qadr:self.lay.box_qadr + 3].copy()

    def observe(self) -> np.ndarray:
        d = self.data
        pos, yaw = self.box_seen()
        return observe(np, self.lay, d.qpos, d.qvel, d.site_xpos, d.site_xmat, self.ctrl, self.touch(), pos, yaw,
                       self.t / EPISODE_STEPS).astype(np.float32)

    def step(self, action: np.ndarray | None = None, ctrl: np.ndarray | None = None, on_substep=None):
        """One 50 Hz step: from a policy action (rates) or, for the expert, from position targets directly."""
        lay = self.lay
        if ctrl is None:
            action = np.clip(action, -1, 1)
            ctrl = self.ctrl + action * RATE * DT
        else:
            action = np.zeros(lay.lo.size)
        self.ctrl = np.clip(ctrl, lay.lo, lay.hi)
        self.data.ctrl[:] = self.ctrl
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            if on_substep is not None:
                on_substep()
        self.t += 1
        r, success = reward(np, lay, self.data.qpos, self.data.site_xpos, self.touch(), action, self.box_xy0)
        return self.observe(), float(r), bool(success), self.t >= EPISODE_STEPS

    def lift_cm(self) -> float:
        return 100 * (self.box_true()[2] - self.lay.z_rest)

    def success(self) -> bool:
        t = self.touch()
        return bool(t[0] and t[1:5].any() and self.lift_cm() / 100 > LIFT_OK)


class Expert:
    """The expert's top grasp, planned once from the observed pose (the first frame), as `skills.top_grasp_plan`:
    the palm turns to the box's heading and moves over it (the pre-grasp swung about the box), comes down 10 cm,
    the hand closes, the palm lifts 15 cm. Arm joint targets from the arm IK at the key poses, interpolated in
    joint space (the descent and lift through intermediate poses, so the palm moves straight)."""

    def __init__(self, twin, ik=None):
        from homehand.control.ik import ArmIK
        lay = twin.lay
        self.ik = ik or ArmIK()
        pos, yaw = twin.box_seen()
        yaw = float(np.clip((yaw + np.pi / 2) % np.pi - np.pi / 2, -0.8, 0.8))
        b0 = np.r_[lay.box_xy, lay.z_rest]
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p_above = pos + Rz @ (lay.palm_pos - b0)
        p_above[2] = lay.palm_pos[2] + (pos[2] - b0[2])
        Rp = np.zeros(9)
        mujoco.mju_quat2Mat(Rp, lay.palm_quat)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (Rz @ Rp.reshape(3, 3)).reshape(-1))
        q = self.ik.q_work.copy()
        adr = self.ik.arm_qadr["right"]
        q[adr] = twin.arm_q()
        keys = [(0.0, q[adr].copy())]
        down = [p_above + np.array([0, 0, -0.10 * f]) for f in (0.25, 0.5, 0.75, 1.0)]
        up = [p_above + np.array([0, 0, -0.10 + 0.15 * f]) for f in (0.25, 0.5, 0.75, 1.0)]
        for t, p in [(1.0, p_above)] + [(1.0 + 0.25 * (k + 1), p) for k, p in enumerate(down)] \
                + [(3.0 + 0.25 * (k + 1), p) for k, p in enumerate(up)]:
            q, _ = self.ik.solve(q, {"right": (p, quat)}, iters=100, converge_orientation=True)
            keys.append((t, q[adr].copy()))
        keys.insert(6, (3.0, keys[5][1]))           # hold the grasp pose while the hand closes (2-3 s)
        self.times = np.array([k[0] for k in keys])
        self.arm = np.array([k[1] for k in keys])

    def ctrl(self, t: float) -> np.ndarray:
        arm = np.array([np.interp(t, self.times, self.arm[:, j]) for j in range(N_ARM)])
        f = float(np.clip(t - 2.0, 0, 1))
        return np.r_[arm, FINGER_OPEN + (FINGER_CLOSE - FINGER_OPEN) * f]


RES_RATE = np.array([0.5] * N_ARM + [2.0] * 6)   # residual policy: offset change per second at |action| = 1
RES_MAX = np.array([0.35] * N_ARM + [0.6] * 6)   # residual policy: largest offset from the expert's targets


class Residual:
    """The learned policy as a correction of the expert: it integrates an offset on the expert's 13 joint
    targets (rate- and size-limited) and observes the plain observation plus that offset."""

    def __init__(self, twin, policy, ik=None):
        self.twin, self.policy = twin, policy
        self.expert = Expert(twin, ik)
        self.offset = np.zeros(N_ARM + 6)

    def obs(self, obs: np.ndarray) -> np.ndarray:
        return np.r_[obs, self.offset / RES_MAX].astype(np.float32)

    def target(self, obs: np.ndarray) -> np.ndarray:
        a = np.clip(self.policy(self.obs(obs)), -1, 1)
        self.offset = np.clip(self.offset + a * RES_RATE * DT, -RES_MAX, RES_MAX)
        return self.expert.ctrl(self.twin.t * DT) + self.offset


def run_policy(twin, policy, seed: int, ik=None, **kw) -> bool:
    """One episode of the residual policy (on the CPU twin or the whole robot)."""
    obs = twin.reset(seed, **kw)
    res = Residual(twin, policy, ik)
    for _ in range(EPISODE_STEPS):
        obs, *_ = twin.step(ctrl=res.target(obs), **kw)
    return twin.success()


def run_expert(twin, seed: int, ik=None, **kw) -> bool:
    twin.reset(seed)
    ex = Expert(twin, ik)
    for k in range(EPISODE_STEPS):
        twin.step(ctrl=ex.ctrl(k * DT), **kw)
    return twin.success()


# ---------------------------------------------------------------------------------------- MJX (training)
def make_mjx_env(xml: str | Path, frame_json: str | Path):
    """Brax environment on MJX (imports jax / brax here: the plain-MuJoCo twin above needs neither)."""
    import jax
    import jax.numpy as jp
    from brax.envs.base import Env, State
    from mujoco import mjx

    class FlatGraspEnv(Env):
        def __init__(self):
            self.mj_model = mujoco.MjModel.from_xml_path(str(xml))
            self.lay = Layout.of(self.mj_model, frame_json)
            self.sys = mjx.put_model(self.mj_model)
            self.n_sub = round(DT / self.mj_model.opt.timestep)
            self._rate = jp.asarray(RATE * DT)

        @property
        def observation_size(self):
            return OBS_SIZE

        @property
        def action_size(self):
            return int(self.lay.lo.size)

        @property
        def backend(self):
            return "mjx"

        def _touch(self, data):
            c = data._impl.contact
            lay = self.lay
            g0, g1 = c.geom[:, 0], c.geom[:, 1]
            on_box = ((g0 == lay.box_geom) | (g1 == lay.box_geom)) & (c.dist <= 0.001)
            other = jp.where(g0 == lay.box_geom, g1, g0)
            part = jp.asarray(lay.part_of_geom)[other]
            return (on_box[:, None] & (part[:, None] == jp.arange(6)[None, :])).any(axis=0).astype(jp.float32)

        def _obs(self, data, info, rng, t):
            a = self.lay.box_qadr
            b = info["bias"]
            pos = data.qpos[a:a + 3] + b[:3] + NOISE * jax.random.normal(rng, (3,))
            yaw = box_yaw(jp, data.qpos[a + 3:a + 7]) + b[3]
            return observe(jp, self.lay, data.qpos, data.qvel, data.site_xpos, data.site_xmat, info["ctrl"],
                           self._touch(data), pos, yaw, t / EPISODE_STEPS)

        def reset(self, rng):
            lay = self.lay
            rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
            xy = jp.asarray(lay.box_xy) + jax.random.uniform(k1, (2,), minval=-BOX_XY_RANGE, maxval=BOX_XY_RANGE)
            yaw = jax.random.uniform(k2, (), minval=-BOX_YAW_RANGE, maxval=BOX_YAW_RANGE)
            bias = jax.random.normal(k3, (4,)) * jp.array([BIAS_XY, BIAS_XY, BIAS_Z, BIAS_YAW])
            a = lay.box_qadr
            qpos = jp.asarray(lay.qpos0)
            qpos = qpos.at[a:a + 3].set(jp.array([xy[0], xy[1], lay.z_rest + 0.0005]))
            qpos = qpos.at[a + 3:a + 7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
            ctrl = jp.asarray(lay.ctrl0, dtype=jp.float32)
            data = mjx.make_data(self.sys).replace(qpos=qpos, ctrl=ctrl)
            data = mjx.forward(self.sys, data)
            info = {"rng": rng, "ctrl": ctrl, "bias": bias, "box_xy0": xy, "t": jp.zeros(())}
            obs = self._obs(data, info, k4, 0.0)
            metrics = {"success": jp.zeros(()), "lift_cm": jp.zeros(())}
            return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

        def step(self, state, action):
            lay = self.lay
            action = jp.clip(action, -1, 1)
            ctrl = jp.clip(state.info["ctrl"] + action * self._rate, lay.lo, lay.hi)
            data = state.pipeline_state.replace(ctrl=ctrl)
            data = jax.lax.fori_loop(0, self.n_sub, lambda _, d: mjx.step(self.sys, d), data)
            rng, k = jax.random.split(state.info["rng"])
            t = state.info["t"] + 1
            info = {**state.info, "rng": rng, "ctrl": ctrl, "t": t}
            r, success = reward(jp, lay, data.qpos, data.site_xpos, self._touch(data), action,
                                state.info["box_xy0"])
            obs = self._obs(data, info, k, t)
            box_z = data.qpos[lay.box_qadr + 2]
            bad = jp.isnan(data.qpos).any() | jp.isnan(obs).any() | (box_z < TABLE_Z - 0.05)
            obs = jp.where(jp.isnan(obs), 0.0, obs)
            r = jp.where(jp.isnan(r), 0.0, r)
            metrics = {**state.metrics, "success": success.astype(jp.float32),
                       "lift_cm": 100 * (box_z - lay.z_rest)}
            return state.replace(pipeline_state=data, obs=obs, reward=r, done=bad.astype(jp.float32),
                                 metrics=metrics, info=info)

    return FlatGraspEnv()


def randomize(sys, rng, lay: Layout):
    """Per-env physics for brax's domain-randomisation wrapper: grip friction and box mass (inertia scaled)."""
    import jax

    mass0 = sys.body_mass[lay.box_body]
    pads = np.asarray(lay.pad_geoms)

    box_dofs = np.arange(6) + int(lay.box_dadr)

    @jax.vmap
    def one(r):
        k1, k2 = jax.random.split(r)
        f = jax.random.uniform(k1, (), minval=GRIP_FRICTION[0], maxval=GRIP_FRICTION[1])
        mass = jax.random.uniform(k2, (), minval=BOX_MASS[0], maxval=BOX_MASS[1])
        # the derived inverse weights scale with 1 / mass too (mj_setConst on the CPU)
        w = mass0 / mass
        return (sys.geom_friction.at[pads, 0].set(f), sys.body_mass.at[lay.box_body].set(mass),
                sys.body_inertia.at[lay.box_body].set(sys.body_inertia[lay.box_body] / w),
                sys.body_invweight0.at[lay.box_body].set(sys.body_invweight0[lay.box_body] * w),
                sys.dof_invweight0.at[box_dofs].set(sys.dof_invweight0[box_dofs] * w))

    fields = ("geom_friction", "body_mass", "body_inertia", "body_invweight0", "dof_invweight0")
    values = one(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    sys = sys.tree_replace(dict(zip(fields, values)))
    return sys, in_axes
