"""Live simulation session for the web app.

One worker thread owns the MuJoCo model, data and OpenGL renderer (OpenGL contexts are bound to the thread
that created them) and executes commands from a queue: run a tidy_table episode in (scaled) real time, run a
conveyor episode (simulated first, then played back: its planner thinks for up to a few seconds at a time, which
would freeze a live stream), replay a recorded episode, or idle. The latest JPEG frames + telemetry are published
for the WebSocket.
"""

from __future__ import annotations

import base64
import io
import queue
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from homehand import paths
from homehand.model import spec
from homehand.model.objects import TIDY_OBJECTS as OBJECTS

COLORS = {"soup_can": (230, 60, 50), "mustard": (240, 200, 30), "sugar_box": (80, 160, 255), "meat_can": (90, 220, 120),
          "foam_brick": (200, 80, 200), "apple": (255, 140, 40)}


@dataclass
class LiveState:
    mode: str = "idle"              # idle | running | replay | loading | error
    config: dict = field(default_factory=dict)
    frame: str | None = None        # base64 JPEG of the main camera
    head: str | None = None         # base64 JPEG of the head camera with detections
    seq: int = 0
    telemetry: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # [{t, grip_l, grip_r, cleared}]
    log: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None


def _jpeg(img: np.ndarray, quality: int = 75) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class LiveSession:
    def __init__(self):
        self.state = LiveState()
        self.lock = threading.Lock()
        self.cmds: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.camera = "front"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ public API (any thread)
    def start(self, config: dict) -> None:
        self.stop_flag.set()
        self.cmds.put(("run", config))

    def replay(self, episode: dict) -> None:
        self.stop_flag.set()
        self.cmds.put(("replay", episode))

    def stop(self) -> None:
        self.stop_flag.set()

    def set_camera(self, camera: str) -> None:
        self.camera = camera

    def snapshot(self) -> dict:
        with self.lock:
            s = self.state
            return {"mode": s.mode, "config": s.config, "frame": s.frame, "head": s.head, "seq": s.seq,
                    "telemetry": s.telemetry, "history": s.history[-300:], "log": s.log[-60:], "result": s.result,
                    "error": s.error}

    # ------------------------------------------------------------------ worker thread
    def _use(self, task: str) -> None:
        """Make the scene of `task` the current one (each scene is built once, in this thread)."""
        if task not in self.envs:
            self._set(mode="loading")
            if task == "conveyor":
                from homehand.env.conveyor_env import ConveyorEnv
                self.envs[task] = ConveyorEnv(record=True)
            else:
                from homehand.env.kitchen_env import KitchenEnv
                self.envs[task] = KitchenEnv(record=True)
        if self.ik is None:
            from homehand.control.ik import ArmIK
            self.ik = ArmIK()
        self.env = self.envs[task]

    def _loop(self) -> None:
        self.env = None
        self.envs: dict = {}
        self.ik = None
        while True:
            cmd, arg = self.cmds.get()
            self.stop_flag.clear()
            try:
                if cmd == "run" and arg.get("task") == "conveyor":
                    self._use("conveyor")
                    self._run_conveyor(arg)
                elif cmd == "run":
                    self._use("tidy_table")
                    self._run(arg)
                elif cmd == "replay":
                    self._use("tidy_table")
                    self._replay(arg)
            except Exception as e:  # surface errors in the UI instead of killing the thread
                import traceback
                traceback.print_exc()
                self._set(mode="error", error=f"{type(e).__name__}: {e}")

    def _set(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self.state, k, v)
            self.state.seq += 1

    def _render(self, w: int = 640, h: int = 480) -> str:
        return _jpeg(self.env.render(self.camera, w, h))

    def _head_view(self, planner) -> str | None:
        """Head camera image with the detections of the last perception step."""
        img = Image.fromarray(self.env.render("head", 480, 360))
        p = planner.last_perception if planner is not None else None
        if p is not None and p.detections:
            d = ImageDraw.Draw(img)
            sx, sy = 480 / 640, 360 / 480
            for det in p.detections:
                x1, y1, x2, y2 = det["bbox"]
                c = COLORS.get(det["name"], (255, 255, 255))
                d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=c, width=3)
                d.rectangle([x1 * sx, y1 * sy - 14, x1 * sx + 7 * len(det["label"]) + 40, y1 * sy], fill=c)
                d.text((x1 * sx + 3, y1 * sy - 13), f"{det['label']} {det['score']:.2f}", fill=(0, 0, 0))
        return _jpeg(np.asarray(img), 70)

    def _run(self, cfg: dict) -> None:
        from homehand.control.planner import TidyPlanner
        from homehand.runner import run_episode

        policy = None
        if cfg.get("controller", "expert") != "expert":
            from homehand.policy.skill import load_policy
            policy = load_policy(cfg["controller"])
        planner = TidyPlanner(self.env, perception=cfg.get("perception", "auto"), policy=policy, ik=self.ik)
        self.camera = cfg.get("camera", self.camera)
        speed = float(cfg.get("speed", 1.0))
        n_obj = int(cfg.get("n_objects", 0)) or None
        objects = None
        if n_obj:
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            objects = list(rng.choice(list(OBJECTS), size=min(n_obj, len(OBJECTS)), replace=False))
        self._set(mode="running", config={**cfg, "perception": planner.perception_mode,
                                          "skill": planner.skill_kind},
                  history=[], log=[], result=None, error=None)
        t_wall = time.perf_counter()
        last_frame = 0.0
        last_perc = [None]

        def on_step(obs, action):
            nonlocal t_wall, last_frame
            if self.stop_flag.is_set():
                return True
            env = self.env
            tel = env.telemetry()
            tel["phase"] = planner.phase
            tel["current"] = planner.current
            with self.lock:
                self.state.history.append({"t": tel["t"], "grip_left": tel["grip_force"]["left"],
                                           "grip_right": tel["grip_force"]["right"], "cleared": tel["n_cleared"]})
                self.state.log = planner.log + [e for e in env.events]
            now = time.perf_counter()
            if now - last_frame > 1 / 20:  # ~20 fps stream
                head = None
                if planner.last_perception is not last_perc[0] or self.state.head is None:
                    head = self._head_view(planner)
                    last_perc[0] = planner.last_perception
                frame = self._render()
                upd = {"frame": frame, "telemetry": tel}
                if head:
                    upd["head"] = head
                self._set(**upd)
                last_frame = now
            # pace the simulation: speed x real time
            t_wall += spec.CONTROL_DT / max(speed, 0.05)
            delay = t_wall - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_wall = time.perf_counter()
            return False

        res = run_episode(self.env, planner, seed=int(cfg.get("seed", 0)), randomization=cfg.get("randomization"),
                          objects=objects, on_step=on_step)
        tel = self.env.telemetry()
        self._set(mode="idle", frame=self._render(), head=self._head_view(planner), telemetry=tel,
                  result={k: res[k] for k in ("outcome", "success", "n_cleared", "n_objects", "object_outcomes",
                                               "steps", "sim_time", "safety", "perception_ms")})

    def _run_conveyor(self, cfg: dict) -> None:
        """Conveyor episode: simulate it (progress shown), then play it back at `speed` x real time with the
        telemetry, grip forces and log recorded along the way."""
        from homehand.control.conveyor_planner import ConveyorPlanner
        from homehand.runner import run_episode

        env = self.env
        perception = cfg.get("perception", "auto")
        if perception == "auto":
            perception = "yolo" if env.YOLO_WEIGHTS.exists() else "oracle"
        planner = ConveyorPlanner(env, perception=perception, ik=self.ik)
        self.camera = cfg.get("camera", self.camera)
        n_obj = int(cfg.get("n_objects", 0)) or 4
        belt = float(cfg.get("belt_speed", 0.03))
        reset = env.reset
        env.reset = lambda task="conveyor", seed=0, objects=None, randomization=None, layout=None: reset(
            task, seed=seed, randomization=randomization, speed=belt, n_objects=n_obj)
        conf = {**cfg, "perception": perception, "skill": planner.skill_kind, "n_objects": n_obj}
        self._set(mode="simulating", config=conf, history=[], log=[], result=None, error=None, head=None,
                  telemetry={"progress": 0.0})
        rec: list = []          # per control step: (qpos, telemetry or None)

        def current():
            sk = planner.skill
            if sk is None:
                return None
            side = next(s for s, k in planner.skills.items() if k is sk)
            return {"side": side, "target": {"name": sk.name}, "grasp": sk.grasp_type.label, "attempt": 1}

        def on_step(obs, action):
            if self.stop_flag.is_set():
                return True
            tel = None
            if env.t % 2 == 0:
                tel = env.telemetry()
                tel["phase"] = planner.phase
                tel["current"] = current()
            rec.append((env.data.qpos.copy(), tel))
            if env.t % 25 == 0:
                self._set(telemetry={"progress": round(min(1.0, env.t / max(env.horizon, 1)), 3),
                                     "t": round(env.t * spec.CONTROL_DT, 2)})
            return False

        try:
            res = run_episode(env, planner, task="conveyor", seed=int(cfg.get("seed", 0)),
                              randomization=cfg.get("randomization"), on_step=on_step)
        finally:
            env.reset = reset
        if self.stop_flag.is_set():
            self._set(mode="idle")
            return
        log = planner.log + list(env.events)
        result = {k: res[k] for k in ("outcome", "success", "n_cleared", "n_objects", "object_outcomes", "steps",
                                      "sim_time", "safety", "perception_ms") if k in res}
        # play back
        self._set(mode="running")
        speed = float(cfg.get("speed", 1.0))
        t_wall = time.perf_counter()
        last_frame, last_head = 0.0, -1.0
        tel = {}
        for k, (q, t) in enumerate(rec):
            if self.stop_flag.is_set():
                break
            env.data.qpos[:] = q
            mujoco.mj_forward(env.model, env.data)
            if t is not None:
                tel = t
                with self.lock:
                    self.state.history.append({"t": t["t"], "grip_left": t["grip_force"]["left"],
                                               "grip_right": t["grip_force"]["right"], "cleared": t["n_cleared"]})
                    self.state.log = [e for e in log if e.get("t", 0) <= k + 1]
            now = time.perf_counter()
            if now - last_frame > 1 / 20:
                upd = {"frame": self._render(), "telemetry": tel}
                if now - last_head > 0.2:
                    upd["head"] = _jpeg(env.render("head", 480, 360), 70)
                    last_head = now
                self._set(**upd)
                last_frame = now
            t_wall += spec.CONTROL_DT / max(speed, 0.05)
            delay = t_wall - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_wall = time.perf_counter()
        with self.lock:
            self.state.log = log
        self._set(mode="idle", frame=self._render(), telemetry=env.telemetry(), result=result)

    def _replay(self, ep: dict) -> None:
        path = paths.DATA_DIR / ep["replay"]
        qpos = np.load(path)["qpos"]
        env = self.env
        env.reset("tidy_table", seed=int(ep["seed"]), layout=ep["data"].get("layout"))
        self._set(mode="replay", config={"replay": ep["id"], "seed": ep["seed"]}, history=[], result=None,
                  log=ep["data"].get("log", []) + ep["data"].get("events", []), error=None)
        t_wall = time.perf_counter()
        for k in range(0, len(qpos), 1):
            if self.stop_flag.is_set():
                break
            env.data.qpos[: qpos.shape[1]] = qpos[k]
            mujoco.mj_forward(env.model, env.data)
            if k % 2 == 0:
                self._set(frame=self._render(), telemetry={"t": round(k * spec.CONTROL_DT, 2), "step": k,
                                                           "replay": True, "outcome": ep["outcome"]})
            t_wall += spec.CONTROL_DT
            delay = t_wall - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self._set(mode="idle", result={"outcome": ep["outcome"], "n_cleared": ep["n_cleared"],
                                       "n_objects": ep["n_objects"], "object_outcomes": ep["data"]["object_outcomes"],
                                       "replay": True})
