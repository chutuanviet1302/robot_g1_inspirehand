"""Differential IK for the two G1 arms with mink.

IK runs on the robot-only model (no scene objects), so it only ever touches the 14 arm joints;
finger joints are frozen. Targets are palm-site poses (see `model.build.PALM_SITE`).
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from homehand import paths
from homehand.control.safety import ARM_MAX_VEL, LIMIT_MARGIN

POSTURE_WEIGHTS = (0.01, 0.01, 0.005, 0.005)  # shoulder yaw, wrist roll, wrist pitch, wrist yaw
from homehand.model import spec


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


class ArmIK:
    def __init__(self, xml_path=None, posture_cost: float = 1e-3, solver: str = "daqp"):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or paths.ROBOT_XML))
        # IK works inside the safety layer's joint limits (+ a little), so commands never get clipped
        for j in spec.ARM_JOINTS:
            jid = self.model.joint(j).id
            lo, hi = self.model.jnt_range[jid]
            m = (LIMIT_MARGIN + 0.01) * (hi - lo)
            self.model.jnt_range[jid] = (lo + m, hi - m)
        self.cfg = mink.Configuration(self.model)
        self.solver = solver
        self.tasks: dict[str, mink.FrameTask] = {
            side: mink.FrameTask(spec.palm_site(side), "site", position_cost=1.0, orientation_cost=0.8, lm_damping=1e-3)
            for side in spec.SIDES
        }
        # Posture regularisation: keep the arms near the natural ready pose; twisting the wrist or the
        # shoulder yaw costs more than moving shoulder pitch / roll / elbow.
        cost = np.full(self.model.nv, posture_cost)
        for j, w in zip(("shoulder_yaw", "wrist_roll", "wrist_pitch", "wrist_yaw"), POSTURE_WEIGHTS):
            for side in spec.SIDES:
                cost[self.model.jnt_dofadr[self.model.joint(f"{side}_{j}_joint").id]] = w
        self.posture = mink.PostureTask(self.model, cost=cost)
        hand_dofs = [
            self.model.jnt_dofadr[j] for j in range(self.model.njnt)
            if self.model.joint(j).name not in spec.ARM_JOINTS
        ]
        self.freeze = mink.DofFreezingTask(self.model, dof_indices=hand_dofs)
        self.limits = [mink.ConfigurationLimit(self.model), self._collision_limit()]
        self.arm_qadr = {
            side: np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in spec.arm_joints(side)])
            for side in spec.SIDES
        }
        self.q_home = self.home_qpos()
        self.q_work = self.home_qpos(self.WORK)

    def _collision_limit(self) -> mink.CollisionAvoidanceLimit:
        """Keep each arm/hand >= 6 cm away from the torso and from the other arm."""
        m = self.model

        def geoms_under(body_name):
            root = m.body(body_name).id
            out = []
            for g in range(m.ngeom):
                b = m.geom_bodyid[g]
                while b > 0 and b != root:
                    b = m.body_parentid[b]
                if b == root and m.geom_contype[g] | m.geom_conaffinity[g]:
                    out.append(g)
            return out

        torso = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("torso_link").id
                 and m.geom_contype[g] | m.geom_conaffinity[g]]
        # the shoulder links sit against the torso by design; guard from the upper arm outwards
        arms = {s: geoms_under(f"{s}_shoulder_yaw_link") for s in spec.SIDES}
        pairs = [(arms["left"], torso), (arms["right"], torso), (arms["left"], arms["right"])]
        return mink.CollisionAvoidanceLimit(m, geom_pairs=pairs, minimum_distance_from_collisions=0.06,
                                            collision_detection_distance=0.10)

    # Natural ready pose, in joint space (right arm; the left arm mirrors roll/yaw), like a person standing at a
    # counter: upper arms hanging along the torso (elbows at the sides, 0.9 m high), forearms forward and ~45 deg
    # up, wrists straight, palms facing each other, fingers relaxed. The hand stays behind x = 0.31 m and above
    # z = 1.08 m, clear of the tallest object (0.99 m): with the forearms lower (palms at 1.03 m) the waiting
    # hand brushed the sugar box and the clean-episode rate fell from 7/8 to 4/8. (G1: elbow 0 = bent 90 deg.)
    NATURAL = {"shoulder_pitch": 0.0, "shoulder_roll": 0.2, "shoulder_yaw": 0.45, "elbow": -0.8,
               "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0}

    # Working posture: the redundancy reference while the hand is out at the object or the bin (the ready pose
    # above is where the arms rest). Pulling the IK towards the resting pose (elbows at the hips) left far
    # targets 4-6 cm out of reach; this keeps the elbows moderately raised and close to the body.
    WORK = {"shoulder_pitch": -0.7, "shoulder_roll": 0.35, "shoulder_yaw": 0.15, "elbow": 0.4,
            "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0}

    def home_qpos(self, pose: dict | None = None) -> np.ndarray:
        q = np.zeros(self.model.nq)
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            for joint, v in (pose or self.NATURAL).items():
                mirror = sgn if joint in ("shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw") else 1.0
                q[self.model.jnt_qposadr[self.model.joint(f"{side}_{joint}_joint").id]] = mirror * v
        return q

    def palm_pose(self, q: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        self.cfg.update(q)
        T = self.cfg.get_transform_frame_to_world(spec.palm_site(side), "site")
        return T.translation(), T.rotation().wxyz

    def solve(
        self,
        q0: np.ndarray,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
        iters: int = 60,
        dt: float = 0.05,
        tol: float = 2e-3,
        posture_target: np.ndarray | None = None,
        converge_orientation: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """targets: side -> (pos, quat wxyz). An arm without a target holds its current configuration.

        Returns (q, position error per side).
        """
        self.cfg.update(q0.copy())
        posture = (self.q_work if posture_target is None else posture_target).copy()
        for side in spec.SIDES:
            if side not in targets:  # keep the idle arm where it is
                posture[self.arm_qadr[side]] = q0[self.arm_qadr[side]]
        self.posture.set_target(posture)
        tasks = [self.posture]
        for side, (pos, quat) in targets.items():
            t = self.tasks[side]
            t.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(quat, float)), np.asarray(pos, float)))
            tasks.append(t)
        idle = [self.tasks[s] for s in spec.SIDES if s not in targets]
        for t in idle:  # pin idle palm in place too, so posture alone doesn't drift it
            t.set_target(self.cfg.get_transform_frame_to_world(t.frame_name, "site"))
            tasks.append(t)
        errs: dict[str, float] = {}
        for _ in range(iters):
            try:
                v = mink.solve_ik(self.cfg, tasks, dt, self.solver, damping=1e-3, limits=self.limits,
                                  constraints=[self.freeze])
            except mink.NoSolutionFound:  # infeasible constraints at this configuration: stay put
                break
            self.cfg.integrate_inplace(v, dt)
            full = {s: self.tasks[s].compute_error(self.cfg) for s in targets}
            errs = {s: float(np.linalg.norm(e[:3])) for s, e in full.items()}
            if all(e < tol for e in errs.values()) and (
                    not converge_orientation or all(np.linalg.norm(e[3:]) < 10 * tol for e in full.values())):
                break
        return self.cfg.q.copy(), errs

    def track(self, q: np.ndarray, targets: dict, iters: int = 15, tol: float = 0.01) -> np.ndarray:
        """One control step of differential IK, rate-limited like the safety layer.

        Differential IK can get trapped near joint limits; when the warm-started solution misses by more than
        `tol`, it is re-solved from the ready pose and that solution is used if it is clearly better. Either
        way the arm joints move at most ARM_MAX_VEL * dt per step, so the commanded trajectory is exactly
        what the safety layer lets through (the controller's internal state stays consistent with reality).
        """
        q1, e1 = self.solve(q, targets, iters=iters)
        best = q1
        if max(e1.values(), default=0.0) > tol:
            q2, e2 = self.solve(self.q_work, targets, iters=200)
            if max(e2.values()) < 0.5 * max(e1.values()):
                best = q2
        out = q.copy()
        idx = np.concatenate([self.arm_qadr["left"], self.arm_qadr["right"]])
        step = 0.9 * ARM_MAX_VEL * spec.CONTROL_DT
        out[idx] = q[idx] + np.clip(best[idx] - q[idx], -step, step)
        return out

    def plan_path(self, q0: np.ndarray, side: str, poses: list[tuple[np.ndarray, np.ndarray]],
                  tol: float = 0.01) -> np.ndarray:
        """Joint configurations for a dense sequence of palm poses (warm-started, fully converged IK).

        Consecutive solutions are close, so linear interpolation between them is a continuous, collision-
        checked joint path. When the warm start gets trapped, the pose is re-solved from the ready pose (the
        resulting jump is given time by the caller's joint-aware timing). The other arm keeps q0."""
        qs = [q0.copy()]
        other = self.arm_qadr["right" if side == "left" else "left"]
        for pos, quat in poses:
            q1, e1 = self.solve(qs[-1], {side: (pos, quat)}, iters=100, converge_orientation=True)
            if e1[side] > tol:
                q2, e2 = self.solve(self.q_work, {side: (pos, quat)}, iters=200, converge_orientation=True)
                if e2[side] < 0.5 * e1[side]:
                    q1 = q2
            q1[other] = q0[other]
            qs.append(q1)
        return np.array(qs)

    def solve_best(self, side: str, pos, quat, seeds: list | None = None) -> tuple[np.ndarray, float]:
        """Fully converged IK for one palm pose from several starts (working posture, ready pose, extra seeds);
        returns the solution with the smallest position error."""
        best, best_err = None, np.inf
        for q0 in [self.q_work, self.q_home] + list(seeds or []):
            q, e = self.solve(q0, {side: (np.asarray(pos, float), np.asarray(quat, float))}, iters=200,
                              converge_orientation=True)
            if e[side] < best_err:
                best, best_err = q, e[side]
            if best_err < 2e-3:
                break
        return best, best_err

    def plan_joint_path(self, q0: np.ndarray, side: str, poses: list[tuple[np.ndarray, np.ndarray]],
                        samples: int = 12, final_q: np.ndarray | None = None) -> np.ndarray:
        """Joint-space path q0 -> IK(pose 1) -> ... (each pose solved from the working posture, i.e. on the
        well-conditioned branch of the redundancy), linearly interpolated, `samples` points per leg. Used for
        free-space moves (reach, return), the way a person swings the arm instead of tracing a straight line.
        `final_q`: the configuration to end in (e.g. the ready pose) instead of an IK solution."""
        other = self.arm_qadr["right" if side == "left" else "left"]
        keys = [q0.copy()]
        for k, (pos, quat) in enumerate(poses):
            if final_q is not None and k == len(poses) - 1:
                q1 = final_q.copy()
            else:
                q1, _ = self.solve_best(side, pos, quat, seeds=[keys[-1]])
            q1[other] = q0[other]
            keys.append(q1)
        path = [keys[0]]
        for a, b in zip(keys[:-1], keys[1:]):
            for t in np.linspace(0, 1, samples + 1)[1:]:
                path.append(a + (b - a) * t)
        return np.array(path)

    def arm_targets(self, q: np.ndarray) -> np.ndarray:
        """14 arm joint targets [left 7, right 7] from a full robot qpos."""
        return np.concatenate([q[self.arm_qadr["left"]], q[self.arm_qadr["right"]]])
