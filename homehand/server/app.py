"""FastAPI backend: live simulation stream, evaluation jobs, results, replays and model info.

Everything runs in one process started by `homehand serve`; the React UI is served as static files.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from homehand import paths
from homehand.env.randomize import PRESETS


class RunConfig(BaseModel):
    controller: str = "expert"
    perception: str = "auto"
    randomization: str = "nominal"
    seed: int = 0
    n_objects: int = 0          # 0 = random 3-4
    camera: str = "front"
    speed: float = 1.0


class EvalConfig(BaseModel):
    controller: str = "expert"
    perception: str = "auto"
    randomization: str = "nominal"
    n: int = 20
    seed0: int = 0


class EvalJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.job: dict | None = None

    def start(self, cfg: EvalConfig) -> dict:
        from homehand.eval import store
        from homehand.eval.runner import run_eval
        from homehand.perception.detector import YOLO_WEIGHTS

        with self.lock:
            if self.job and self.job["status"] == "running":
                raise HTTPException(409, "an evaluation is already running")
            perception = cfg.perception
            if perception == "auto":
                perception = "yolo" if YOLO_WEIGHTS.exists() else "oracle"
            name = f"{cfg.controller} | {perception} | {cfg.randomization}"
            run_id = store.create_run(name, "tidy_table", cfg.controller, perception, cfg.randomization, cfg.n,
                                      cfg.seed0)
            self.job = {"run_id": run_id, "done": 0, "total": cfg.n, "status": "running", "started": time.time()}

        def progress(done, total):
            self.job["done"] = done

        def work():
            try:
                run_eval(cfg.controller, perception, cfg.randomization, n=cfg.n, seed0=cfg.seed0, progress=progress,
                         run_id=run_id)
                self.job["status"] = "done"
            except Exception as e:
                self.job["status"] = "failed"
                self.job["error"] = str(e)

        threading.Thread(target=work, daemon=True).start()
        return self.job


def available_controllers() -> list[dict]:
    out = [{"id": "expert", "label": "Scripted expert (grasp library + IK)", "kind": "expert"}]
    if paths.MODELS_DIR.exists():
        for d in sorted(paths.MODELS_DIR.iterdir()):
            if (d / "policy.pt").exists() and not d.name.startswith("smoke"):
                log = json.loads((d / "train_log.json").read_text()) if (d / "train_log.json").exists() else {}
                kind = log.get("policy", "policy")
                label = {"act": "ACT", "diffusion": "Diffusion Policy"}.get(kind, kind)
                out.append({"id": d.name, "label": f"{label} ({d.name})", "kind": kind})
    return out


def create_app() -> FastAPI:
    from homehand.server.session import LiveSession

    app = FastAPI(title="HomeHand", version="1.0")
    session = LiveSession()
    jobs = EvalJobs()

    @app.get("/api/config")
    def config():
        from homehand.control.grasps import GRASPS, LYING_GRASP
        from homehand.model.objects import OBJECTS
        from homehand.perception.detector import YOLO_WEIGHTS

        perception = [{"id": "auto", "label": "Auto (YOLO if trained)"},
                      {"id": "yolo", "label": "YOLOv8n-seg + depth", "available": YOLO_WEIGHTS.exists()},
                      {"id": "oracle", "label": "Oracle masks + depth"},
                      {"id": "ground_truth", "label": "Ground-truth poses"}]
        return {
            "controllers": available_controllers(),
            "perception": perception,
            "randomization": {k: v.to_dict() for k, v in PRESETS.items()},
            "cameras": ["front", "overview", "top", "head"],
            "objects": [{"id": k, "label": o.label, "grasp": GRASPS[o.grasp].label,
                         "grasp_lying": GRASPS[LYING_GRASP[k]].label if k in LYING_GRASP else None, "mass": o.mass}
                        for k, o in OBJECTS.items()],
            "grasps": {k: {"label": g.label, "approach": g.approach, "thumb_yaw": g.thumb_yaw, "close": g.close,
                           "z_frac": g.z_frac, "tilt_deg": g.tilt_deg} for k, g in GRASPS.items()},
        }

    # ------------------------------------------------------------------ live
    @app.post("/api/live/start")
    def live_start(cfg: RunConfig):
        session.start(cfg.model_dump())
        return {"ok": True}

    @app.post("/api/live/stop")
    def live_stop():
        session.stop()
        return {"ok": True}

    @app.post("/api/live/camera/{camera}")
    def live_camera(camera: str):
        session.set_camera(camera)
        return {"ok": True}

    @app.websocket("/ws/live")
    async def live_ws(ws: WebSocket):
        await ws.accept()
        last = -1
        try:
            while True:
                snap = session.snapshot()
                if snap["seq"] != last:
                    last = snap["seq"]
                    await ws.send_text(json.dumps(snap))
                await asyncio.sleep(0.05)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ------------------------------------------------------------------ evaluation
    @app.post("/api/eval")
    def eval_start(cfg: EvalConfig):
        if cfg.n < 1 or cfg.n > 500:
            raise HTTPException(400, "n must be in 1..500")
        return jobs.start(cfg)

    @app.get("/api/eval/job")
    def eval_job():
        return jobs.job or {}

    @app.get("/api/runs")
    def runs():
        from homehand.eval import store
        return store.list_runs()

    @app.get("/api/runs/{run_id}")
    def run(run_id: int):
        from homehand.eval import store
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404)
        return r

    @app.delete("/api/runs/{run_id}")
    def run_delete(run_id: int):
        from homehand.eval import store
        store.delete_run(run_id)
        return {"ok": True}

    @app.get("/api/episodes")
    def episodes(run_id: int | None = None, outcome: str | None = None, limit: int = 300):
        from homehand.eval import store
        return store.list_episodes(run_id, outcome, limit)

    @app.post("/api/episodes/{ep_id}/replay")
    def replay(ep_id: int):
        from homehand.eval import store
        ep = store.get_episode(ep_id)
        if not ep or not ep.get("replay"):
            raise HTTPException(404, "episode or replay file not found")
        session.replay(ep)
        return {"ok": True}

    # ------------------------------------------------------------------ models / system
    @app.get("/api/models")
    def models():
        out = {"policies": [], "datasets": [], "detector": None}
        if paths.MODELS_DIR.exists():
            for d in sorted(paths.MODELS_DIR.iterdir()):
                if (d / "train_log.json").exists() and not d.name.startswith("smoke"):
                    out["policies"].append({"name": d.name, **json.loads((d / "train_log.json").read_text())})
            det = paths.MODELS_DIR / "detector"
            if (det / "metrics.json").exists():
                out["detector"] = json.loads((det / "metrics.json").read_text())
                if (det / "bench.json").exists():
                    out["detector"]["bench"] = json.loads((det / "bench.json").read_text())
        if paths.DATASET_DIR.exists():
            for d in sorted(paths.DATASET_DIR.iterdir()):
                if (d / "meta.json").exists() and d.name != "smoke":
                    m = json.loads((d / "meta.json").read_text())
                    out["datasets"].append({k: v for k, v in m.items() if k not in ("stats", "feature_names")})
        return out

    @app.get("/api/system")
    def system():
        from homehand.resources import available_mb, safe_workers
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu = {"name": torch.cuda.get_device_name(0), "free_mb": free // 2**20, "total_mb": total // 2**20}
        except Exception:
            pass
        return {"ram_available_mb": available_mb(), "safe_workers": safe_workers(), "gpu": gpu,
                "cpu_count": os.cpu_count(), "platform": os.name}

    # ------------------------------------------------------------------ static UI
    static = paths.WEB_DIST_DIR
    if (static / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = static / path
            if path and f.is_file():
                return FileResponse(f)
            return FileResponse(static / "index.html")
    else:
        @app.get("/")
        def no_ui():
            return JSONResponse({"error": "web UI not built: run `npm run build` in web/"}, status_code=503)

    return app
