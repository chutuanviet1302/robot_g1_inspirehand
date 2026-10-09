"""Live simulation session for the web app.

One worker thread owns the MuJoCo model, data and OpenGL renderer (OpenGL contexts are bound to the thread
that created them) and executes commands from a queue: run a tidy_table episode in (scaled) real time, replay
a recorded episode, or idle. The latest JPEG frames + telemetry are published for the WebSocket.
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
from homehand.model.objects import OBJECTS

COLORS = {"soup_can": (230, 60, 50), "mustard": (240, 200, 30), "sugar_box": (80, 160, 255), "meat_can": (90, 220, 120)}


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
    def _loop(self) -> None:
        self.env = None
        self.ik = None
        while True:
            cmd, arg = self.cmds.get()
            self.stop_flag.clear()
            try:
                if self.env is None:
                    self._set(mode="loading")
                    from homehand.control.ik import ArmIK
                    from homehand.env.kitchen_env import KitchenEnv
                    self.env = KitchenEnv(record=True)
                    self.ik = ArmIK()
                if cmd == "run":
                    self._run(arg)
                elif cmd == "replay":
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
