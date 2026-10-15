"""Home task: the robot walks on its own legs from across the room, around obstacles, stops at the counter and
clears it into the bin.

The scene is the kitchen counter of the tidy-table task in a room (`model.build.build_home_scene`); the robot is the
same G1 + Inspire hands with leg joints and a free-floating base. Its legs are driven by Unitree's pretrained
walking policy (`locomotion.walker`) inside every physics step; the arms and hands keep their 26 position actuators
and the safety layer exactly as in the fixed-base tasks. Obstacles (boxes and stools, 2-4 per episode, random
sizes and places on the way, always leaving a way through) are mocap bodies moved at reset.

Episode outcome: as in the tidy-table task per object, plus `info.details["nav"]`: whether the robot arrived,
how far from the stop point, bumps into obstacles, falls.
"""

from __future__ import annotations

import mujoco
import numpy as np

from homehand import paths
from homehand.env.conveyor_env import ConveyorEnv
from homehand.env.kitchen_env import KitchenEnv
from homehand.locomotion.walker import Walker
from homehand.model import spec
from homehand.model.build import HOME_DOCK, HOME_OBSTACLE_SLOTS, HOME_START

NAV_STEPS = 1500                   # control steps (60 s) added to the horizon for walking to the counter
OBSTACLE_AREA = ((-2.05, -0.75), (-1.1, 1.1))     # x, y range of obstacle centres
PARK = (3.0, 0.0)                  # unused objects wait behind the counter, out of the robot's way


class WalkingMixin:
    """The legged robot in a room with obstacles: walker, obstacle placement, bump / fall bookkeeping. Mixed into
    the tidy-table and the conveyor environments (before them, so its `_simulate` / `step` come first)."""

    def _init_walking(self) -> None:
        m = self.model
        self.walker = Walker(m, self.data)
        self.pelvis = m.body("pelvis").id
        self.obstacle_geoms = [m.geom(f"obstacle{k}").id for k in range(HOME_OBSTACLE_SLOTS)]
        self.obstacle_bodies = {m.geom_bodyid[g] for g in self.obstacle_geoms}
        self.robot_all = {b for b in range(m.nbody) if self._under_pelvis(b)}

    def _under_pelvis(self, b: int) -> bool:
        while b > 0:
            if b == self.pelvis:
                return True
            b = self.model.body_parentid[b]
        return False

    # ------------------------------------------------------------------ reset
    def _after_reset(self, task: str, n_obstacles: int | None) -> None:
        """Called at the end of the task's own reset: obstacles, robot at the start, walking bookkeeping."""
        self.task = task
        self.info.task = task
        m, d = self.model, self.data
        self.obstacles = self._place_obstacles(n_obstacles)
        self.walker.place(*HOME_START)
        mujoco.mj_forward(m, d)
        for _ in range(100):                     # stand still for 0.2 s before anything happens
            self.walker.physics_step()
            mujoco.mj_step(m, d)
        self.safety.check_workspace = False      # the palms leave the counter's workspace while walking
        self.horizon += NAV_STEPS
        self.nav = {"arrived": False, "dock_error_m": None, "dock_yaw_error_deg": None, "bumps": 0,
                    "fell": False, "walk_time_s": None, "path_m": 0.0}
        self._bumping = False
        self._prev_xy = d.qpos[self.walker.base_q:self.walker.base_q + 2].copy()
        self.traj_qpos = []
        self._log_qpos()

    def _place_obstacles(self, n: int | None) -> list[dict]:
        """2-4 boxes / stools between the start and the counter, never blocking every way through."""
        from homehand.locomotion import nav
        m, d = self.model, self.data
        n = int(self.rng.integers(2, 5)) if n is None else n
        for _ in range(50):
            placed = []
            for k in range(HOME_OBSTACLE_SLOTS):
                g = self.obstacle_geoms[k]
                mid = m.body_mocapid[m.geom_bodyid[g]]
                if k >= n:
                    d.mocap_pos[mid] = [10 + k, 5, 0]
                    continue
                for _ in range(100):
                    x = self.rng.uniform(*OBSTACLE_AREA[0])
                    y = self.rng.uniform(*OBSTACLE_AREA[1])
                    hx, hy = self.rng.uniform(0.15, 0.28, 2)
                    hz = self.rng.uniform(0.22, 0.4)
                    if np.hypot(x - HOME_START[0], y - HOME_START[1]) < 0.75:
                        continue
                    if any(abs(x - o["x"]) < hx + o["hx"] + 0.1 and abs(y - o["y"]) < hy + o["hy"] + 0.1
                           for o in placed):
                        continue
                    break
                yaw = self.rng.uniform(-np.pi, np.pi)
                m.geom_size[g] = [hx, hy, hz]
                m.geom_pos[g] = [0, 0, hz]
                d.mocap_pos[mid] = [x, y, 0]
                d.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                placed.append({"x": float(x), "y": float(y), "hx": float(hx), "hy": float(hy), "hz": float(hz),
                               "yaw": float(yaw)})
            mujoco.mj_forward(m, d)
            # a way through must exist on the true map (the robot itself only learns it with its lidar)
            grid = nav.OccupancyGrid()
            for o in placed:
                c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
                for u in np.linspace(-o["hx"], o["hx"], 12):
                    for v in np.linspace(-o["hy"], o["hy"], 12):
                        grid.mark(np.array([[o["x"] + c * u - s * v, o["y"] + s * u + c * v]]))
            if nav.astar(grid, HOME_START[:2], (HOME_DOCK[0] - 0.5, HOME_DOCK[1])) is not None:
                return placed
        return placed

    # ------------------------------------------------------------------ physics
    def _simulate(self) -> None:
        m, d = self.model, self.data
        for _ in range(spec.N_SUBSTEPS):
            self.walker.physics_step()
            mujoco.mj_step(m, d)

    def step(self, action):
        # the palm workspace moves with the robot (it is defined for the pelvis at the stop point)
        base = self.data.qpos[self.walker.base_q:self.walker.base_q + 3]
        self.safety.workspace_offset = base - np.array([HOME_DOCK[0], HOME_DOCK[1], 0.793])
        obs, done = super().step(action)
        d = self.data
        xy = d.qpos[self.walker.base_q:self.walker.base_q + 2]
        self.nav["path_m"] += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy.copy()
        bump = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in self.obstacle_bodies and b2 in self.robot_all) or (b2 in self.obstacle_bodies and b1 in self.robot_all):
                bump = True
                break
        if bump and not self._bumping:
            self.nav["bumps"] += 1
            self.events.append({"t": self.t, "event": "bumped"})
        self._bumping = bump
        if self.walker.fallen() and not self.nav["fell"]:
            self.nav["fell"] = True
            self.events.append({"t": self.t, "event": "fell"})
            self._finish()
            self.info.outcome = "fell"
            self.info.success = False
            return self.observe(), True
        return obs, done

    def _finish(self) -> None:
        super()._finish()
        self.info.details["nav"] = dict(self.nav)
        self.info.details["obstacles"] = list(getattr(self, "obstacles", []))

    def base_pose(self):
        return self.walker.base_pose()


class HomeEnv(WalkingMixin, KitchenEnv):
    """Walk to the counter, tidy it."""

    def __init__(self, record: bool = False):
        KitchenEnv.__init__(self, xml_path=paths.HOME_XML, record=record)
        self._init_walking()

    def reset(self, task: str = "home", seed: int = 0, objects=None, randomization=None, layout=None,
              n_obstacles: int | None = None) -> dict:
        KitchenEnv.reset(self, "tidy_table", seed=seed, objects=objects, randomization=randomization, layout=layout)
        used = {n for n, *_ in self.layout}
        for k, name in enumerate(n for n in self.OBJECT_NAMES if n not in used):
            a = self.obj_qadr[name]
            self.data.qpos[a:a + 3] = [PARK[0], PARK[1] + 0.25 * k, 0.001]
        self._after_reset(task, n_obstacles)
        return self.observe()


class HomeConveyorEnv(WalkingMixin, ConveyorEnv):
    """Walk to the conveyor, then pick the objects it brings. The belt starts only once the robot stands at it
    (`start_belt`, called by the controller): objects passing while it walks would just ride past."""

    def __init__(self, record: bool = False):
        ConveyorEnv.__init__(self, record=record, xml_path=paths.HOME_CONVEYOR_XML)
        self._init_walking()

    def reset(self, task: str = "home_conveyor", seed: int = 0, objects=None, randomization=None, layout=None,
              speed: float = 0.03, n_objects: int = 4, gap=None, n_obstacles: int | None = None) -> dict:
        from homehand.env.conveyor_env import GAP
        ConveyorEnv.reset(self, "conveyor", seed=seed, objects=objects, randomization=randomization,
                          speed=speed, n_objects=n_objects, gap=gap or GAP)
        # hold the belt: the spawn plan starts when the robot is ready (the conveyor's reset already put the first
        # object on the belt: back into the plan with it)
        spawned = [(0, *entry) for entry in self.layout]
        for name in list(self.tracks):
            self._park(name)
        self.tracks, self.layout = {}, []
        self.events = [e for e in self.events if e.get("event") != "spawned"]
        self._pending = (spawned + list(self.queue), self.speed, self.horizon)
        self.queue, self.speed = [], 0.0
        self.data.ctrl[self.belt_act] = 0.0
        self.data.qvel[self.belt_dof] = 0.0
        self.belt_started = False
        self._after_reset(task, n_obstacles)
        return self.observe()

    def start_belt(self) -> None:
        queue, speed, horizon = self._pending
        self.queue = [(s + self.t, *rest) for s, *rest in queue]
        self.speed = speed
        self.horizon = self.t + horizon
        self.belt_started = True
        self.events.append({"t": self.t, "event": "belt_start"})

    def _check_done(self) -> bool:
        if not self.belt_started:
            if self.t >= self.horizon:
                self._finish()
                return True
            return False
        return super()._check_done()
