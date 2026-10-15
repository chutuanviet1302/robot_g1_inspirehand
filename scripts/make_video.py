"""Render one episode of any task as an HD video (H.264 MP4).

    python scripts/make_video.py --task conveyor --seed 0 --perception yolo --out media/conveyor_hd.mp4
    python scripts/make_video.py --task tidy_table --seed 8 --perception yolo --out media/tidy_hd.mp4
    python scripts/make_video.py --task bimanual_lift --seed 4 --out media/bimanual_hd.mp4

The episode runs first, recording the joint and object positions; the frames are then rendered from that record
with a fixed free camera (no stalls while a planner thinks), at `--speedup` x real time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# free camera per task: lookat, azimuth, elevation, distance
CAMERAS = {
    "conveyor": ([0.22, 0.0, 0.95], 180.0, -30.0, 1.30),
    "tidy_table": ([0.32, 0.0, 0.92], 180.0, -38.0, 1.35),
    "bimanual_lift": ([0.33, 0.0, 0.93], 140.0, -22.0, 1.20),
}


def simulate(task: str, seed: int, perception: str, objects: int):
    from homehand.runner import run_episode
    if task == "conveyor":
        from homehand.control.conveyor_planner import ConveyorPlanner
        from homehand.env.conveyor_env import ConveyorEnv
        env = ConveyorEnv()
        controller = ConveyorPlanner(env, perception=perception)
        reset = env.reset
        env.reset = lambda task="conveyor", seed=0, objects=None, randomization=None, layout=None: reset(
            task, seed=seed, n_objects=objects or 4)
    elif task == "bimanual_lift":
        from homehand.control.bimanual import BimanualExpert
        from homehand.env.bimanual_env import BimanualEnv
        env = BimanualEnv()
        controller = BimanualExpert(env)
    else:
        from homehand.control.planner import TidyPlanner
        from homehand.env.kitchen_env import KitchenEnv
        env = KitchenEnv()
        controller = TidyPlanner(env, perception=perception)
    qpos = []
    res = run_episode(env, controller, task=task, seed=seed, on_step=lambda obs, a: qpos.append(env.data.qpos.copy()))
    return env, qpos, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="conveyor", choices=list(CAMERAS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception", default="oracle")
    ap.add_argument("--objects", type=int, default=0, help="conveyor: objects per episode (default 4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--speedup", type=int, default=2, help="playback speed (integer: every n-th control step)")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better, larger)")
    a = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from homehand.model import spec

    env, qpos, res = simulate(a.task, a.seed, a.perception, a.objects)
    print("outcome:", res["outcome"], {str(k): v for k, v in res["object_outcomes"].items()})
    m, d = env.model, env.data
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
    m.vis.quality.shadowsize = max(m.vis.quality.shadowsize, 4096)
    renderer = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    look, az, el, dist = CAMERAS[a.task]
    cam.lookat[:] = look
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    # skip the idle start (nothing moves until the first object or reach)
    first = next((i for i, q in enumerate(qpos) if i > 0 and np.abs(q - qpos[0]).max() > 1e-3), 0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / spec.CONTROL_DT))            # every control step at 1x = 25 fps
    parked = [env.obj_qadr[n] for n in env.obj_qadr]     # objects waiting off the counter: moved out of view
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-crf", str(a.crf), "-preset", "slow", "-movflags", "+faststart"])
    n = 0
    for q in qpos[max(0, first - 25)::a.speedup]:
        d.qpos[:] = q
        for adr in parked:
            if d.qpos[adr] < -1.0 and d.qpos[adr + 2] < 0.1:     # parked on the floor behind the robot
                d.qpos[adr] = 6.0
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        writer.append_data(renderer.render())
        n += 1
    writer.close()
    renderer.close()
    print(f"wrote {out}: {n} frames, {n / fps:.1f} s, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
