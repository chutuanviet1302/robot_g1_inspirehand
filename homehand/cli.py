"""HomeHand command line. Run `homehand --help`."""

from __future__ import annotations

import json
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux: offscreen rendering

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="HomeHand: Unitree G1 + two Inspire hands tidying a kitchen counter.")
perception_app = typer.Typer(help="Head-camera detector (YOLOv8n-seg) data, training and benchmarks.")
app.add_typer(perception_app, name="perception")


@app.command("fetch-assets")
def fetch_assets(force: bool = False):
    """Download Unitree G1 (Menagerie), Inspire hand (dex-urdf) and YCB objects at pinned versions."""
    from homehand.assets_fetch import fetch_all
    fetch_all(force=force)


@app.command("build-model")
def build_model(source: str = typer.Option("auto", help="auto | repo (third_party/unitree_g1_inspire) | menagerie")):
    """Assemble G1 + 2x Inspire + kitchen scene into generated/."""
    from homehand.model.build import build
    build(source=source)


@app.command()
def demo(seed: int = 0, perception: str = "auto", controller: str = "expert", out: str = "demo.mp4",
         camera: str = "front", fps: int = 25):
    """Run one tidy_table episode and save a video (mp4, or gif when imageio-ffmpeg is missing)."""
    import numpy as np
    from PIL import Image

    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.runner import run_episode

    env = KitchenEnv()
    policy = None
    if controller != "expert":
        from homehand.policy.skill import load_policy
        policy = load_policy(controller)
    planner = TidyPlanner(env, perception=perception, policy=policy)
    frames = []

    def grab(obs, action):
        if env.t % 2 == 0:
            frames.append(env.render(camera, 640, 480).copy())

    res = run_episode(env, planner, seed=seed, on_step=grab)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out, frames, fps=fps // 2)
    except Exception as e:  # imageio / imageio-ffmpeg missing: fall back to a GIF
        typer.echo(f"mp4 export unavailable ({type(e).__name__}: {e}); writing a GIF")
        out = out.rsplit(".", 1)[0] + ".gif"
        imgs = [Image.fromarray(f).resize((480, 360)) for f in frames[::2]]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(4000 / fps), loop=0)
    env.close()
    typer.echo(json.dumps({k: res[k] for k in ("outcome", "n_cleared", "n_objects", "object_outcomes")}, indent=2))
    typer.echo(f"video: {out}")


@app.command()
def view(seed: int = 0, perception: str = "oracle", controller: str = "expert", speed: float = 1.0,
         loop: bool = typer.Option(False, help="keep running the next seed after each episode")):
    """Open the native MuJoCo viewer and watch the robot tidy the counter (orbit with the mouse)."""
    import time

    import mujoco.viewer

    from homehand.control.planner import TidyPlanner
    from homehand.env.kitchen_env import KitchenEnv
    from homehand.model import spec

    env = KitchenEnv()
    policy = None
    if controller != "expert":
        from homehand.policy.skill import load_policy
        policy = load_policy(controller)
    planner = TidyPlanner(env, perception=perception, policy=policy)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = (0.38, 0.0, 0.9)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 150, -22
        while v.is_running():
            with v.lock():
                obs = env.reset("tidy_table", seed=seed)
                planner.reset(obs)
            typer.echo(f"seed {seed}: objects {[str(o) for o in env.info.objects]}")
            done, settle, t_wall = False, 0, time.perf_counter()
            while v.is_running() and not done:
                with v.lock():
                    obs, done = env.step(planner.act(obs))
                    if planner.done and not done:
                        settle += 1
                        if settle >= 15:
                            env.force_finish()
                            done = True
                v.sync()
                t_wall += spec.CONTROL_DT / speed
                if (delay := t_wall - time.perf_counter()) > 0:
                    time.sleep(delay)
            if not v.is_running():
                break
            i = env.info
            stop = env.safety.stats.stop_reason
            typer.echo(f"  -> {i.outcome}: {i.n_cleared}/{len(i.objects)} in the bin {({str(k): v for k, v in i.object_outcomes.items()})}"
                       + (f"  [protective stop: {stop}]" if stop else ""))
            if not loop:
                typer.echo("episode finished; close the window to exit")
                while v.is_running():
                    time.sleep(0.2)
                break
            seed += 1


@app.command()
def collect(name: str = "expert_v1", n: int = 400, workers: int = 6, perception: str = "oracle",
            randomization: str = "low"):
    """Record pick-and-place demonstrations from the scripted expert."""
    from homehand.data.record import collect as run
    run(name, n_episodes=n, workers=workers, perception=perception, randomization=randomization)


@app.command()
def train(policy: str = typer.Option("act", help="act | diffusion"), dataset: str = "expert_v1",
          steps: int = 20000, batch_size: int = 256, lr: float = 3e-4, name: str = "",
          init: str = typer.Option("", help="model directory to continue training from"),
          log_every: int = 500):
    """Train ACT or Diffusion Policy on a recorded dataset."""
    from homehand.policy.train import train as run
    run(policy, dataset, steps=steps, batch_size=batch_size, lr=lr, out_name=name or None, init=init or None,
        log_every=log_every)


@app.command("eval")
def evaluate(controller: str = "expert", perception: str = "auto", randomization: str = "nominal",
             n: int = typer.Option(50, "-n", "--n"), seed0: int = 0, workers: int = 0, name: str = ""):
    """Batch evaluation (parallel workers, RAM-guarded); results go to the SQLite store and the web UI."""
    from homehand.eval import store
    from homehand.eval.runner import run_eval

    def progress(done, total):
        typer.echo(f"\r  {done}/{total}", nl=False)

    run_id = run_eval(controller, perception, randomization, n=n, seed0=seed0, workers=workers or None,
                      name=name or None, progress=progress)
    s = store.get_run(run_id)["summary"]
    typer.echo("")
    typer.echo(json.dumps({"run": run_id, "task_success": s["task_success"],
                           "object_clear_rate": s["object_clear_rate"], "episode_outcomes": s["episode_outcomes"],
                           "safety": s["safety"]}, indent=2))


@perception_app.command("gen-data")
def perception_gen(n: int = 1500, seed: int = 0):
    from homehand.perception.train_detector import gen_data
    gen_data(n, seed)


@perception_app.command("train")
def perception_train(epochs: int = 40, imgsz: int = 512, batch: int = 8):
    from homehand.perception.train_detector import train as run
    run(epochs, imgsz, batch)


@perception_app.command("bench")
def perception_bench(detector: str = "yolo", n: int = 60):
    """Detection recall + 3D localisation error vs. ground truth."""
    from homehand.perception.train_detector import localization_benchmark
    typer.echo(json.dumps(localization_benchmark(detector, n), indent=2))


@app.command("grasp-bench")
def grasp_bench(objects: str = typer.Option("", help="comma separated object names (default: all)"),
                n: int = typer.Option(6, "-n", "--n", help="yaw samples per object / hand / slot"),
                workers: int = 0, out: str = "",
                rest: str = typer.Option("upright,lying", help="rest poses to test: upright, lying or both")):
    """Pick-and-place of single objects over positions and yaws: success and failure mode per grasp."""
    import json as _json

    from homehand.eval.grasp_bench import run_bench, summarize

    names = [o for o in objects.split(",") if o] or None
    res = run_bench(names, n_yaw=n, workers=workers or None, rests=tuple(rest.split(",")),
                    progress=lambda d, t: typer.echo(f"  {d}/{t}") if d % 16 == 0 or d == t else None)
    typer.echo(summarize(res))
    if out:
        Path(out).write_text(_json.dumps(res, indent=1))


@app.command()
def pipeline(episodes: int = 300, yolo_images: int = 1500, yolo_epochs: int = 40, steps: int = 10000,
             batch_size: int = 128, eval_n: int = 50, skip: str = "",
             diffusion_steps: int = 30000, evaluate: bool = typer.Option(True, help="run the evaluations")):
    """Everything, one stage after the other (keeps RAM / VRAM use bounded):
    demos -> YOLO data -> YOLO -> ACT -> Diffusion -> evaluations (expert / ACT / DP, nominal + sim-gap)."""
    from homehand import paths
    from homehand.eval.runner import run_eval

    skip_set = set(filter(None, skip.split(",")))
    log = paths.DATA_DIR / "logs" / "pipeline.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(log.read_text()) if log.exists() else {}

    def stage(name, fn):
        if name in skip_set or status.get(name) == "done":
            typer.echo(f"== {name}: skipped")
            return
        typer.echo(f"== {name}")
        fn()
        status[name] = "done"
        log.write_text(json.dumps(status, indent=2))

    from homehand.data.record import collect as rec
    from homehand.perception import train_detector as td
    from homehand.policy.train import train as tr

    stage("collect", lambda: rec("expert_v1", n_episodes=episodes, perception="oracle", randomization="low"))
    stage("yolo_data", lambda: td.gen_data(yolo_images))
    stage("yolo_train", lambda: td.train(yolo_epochs))
    stage("perception_bench", lambda: (paths.MODELS_DIR / "detector" / "bench.json").write_text(json.dumps(
        {d: td.localization_benchmark(d) for d in ("oracle", "yolo")}, indent=2)))
    stage("train_act", lambda: tr("act", "expert_v1", steps=steps, batch_size=batch_size, out_name="act_expert_v1"))
    # Diffusion Policy needs ~3x the steps of ACT here (10 k steps: 1 % of objects, 30 k: 78 %, see RESULTS.md)
    stage("train_diffusion", lambda: tr("diffusion", "expert_v1", steps=diffusion_steps, batch_size=batch_size,
                                          out_name="diffusion_expert_v1", log_every=2500))
    if not evaluate:
        typer.echo("pipeline finished (training only)")
        return
    for ctrl in ("expert", "act_expert_v1", "diffusion_expert_v1"):
        stage(f"eval_{ctrl}_nominal", lambda c=ctrl: run_eval(c, "yolo", "nominal", n=eval_n))
    for preset in ("low", "medium", "high"):
        for ctrl in ("expert", "act_expert_v1", "diffusion_expert_v1"):
            stage(f"eval_{ctrl}_{preset}", lambda c=ctrl, p=preset: run_eval(c, "yolo", p, n=eval_n // 2))
    stage("eval_expert_oracle", lambda: run_eval("expert", "oracle", "nominal", n=eval_n))
    typer.echo("pipeline finished")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Start the web app (simulation, live camera, evaluation dashboard)."""
    import threading
    import webbrowser

    import uvicorn

    from homehand.server.app import create_app

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def main():
    app()


if __name__ == "__main__":
    main()
