"""Render one conveyor episode as an animated GIF for the README.

    python scripts/make_gif.py --seed 7 --out media/conveyor_seed7.gif

The episode runs first (camera 6-D poses on oracle masks), recording the joint and object positions; the frames
are then rendered from that record with a fixed free camera in front of the counter, played back at 3x.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--speed", type=float, default=0.03, help="belt speed, m/s")
    ap.add_argument("--perception", default="oracle")
    ap.add_argument("--out", default="media/conveyor_seed7.gif")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--every", type=int, default=4, help="render every n-th control step")
    ap.add_argument("--speedup", type=float, default=3.0, help="playback speed")
    ap.add_argument("--crop-bottom", type=float, default=0.17, help="fraction of the image height cut at the bottom "
                                                                    "(the empty counter front)")
    a = ap.parse_args()

    import mujoco
    import numpy as np
    from PIL import Image

    from homehand.control.conveyor_planner import ConveyorPlanner
    from homehand.env.conveyor_env import ConveyorEnv
    from homehand.model import spec
    from homehand.runner import run_episode

    env = ConveyorEnv()
    planner = ConveyorPlanner(env, perception=a.perception)
    reset = env.reset
    env.reset = lambda task="conveyor", seed=0, objects=None, randomization=None, layout=None: reset(
        task, seed=seed, speed=a.speed, n_objects=a.objects, gap=(0.40, 0.55))
    qpos = []
    res = run_episode(env, planner, task="conveyor", seed=a.seed,
                      on_step=lambda obs, act: qpos.append(env.data.qpos.copy()))
    print("outcome:", res["outcome"], dict(res["object_outcomes"]))

    m, d = env.model, env.data
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)
    renderer = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.22, 0.0, 0.95]
    cam.azimuth, cam.elevation, cam.distance = 180.0, -36.0, 1.25
    # start shortly before the first object reaches the right hand's zone
    first = next((i for i, q in enumerate(qpos) if i > 0 and np.abs(q - qpos[0]).max() > 1e-3), 0)
    frames = []
    for q in qpos[max(0, first - 25)::a.every]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        frames.append(Image.fromarray(img[: int(round(a.height * (1 - a.crop_bottom)))]))
    renderer.close()
    duration = int(round(1000 * a.every * spec.CONTROL_DT / a.speedup))
    # one adaptive palette for the whole clip (no colour flicker between frames)
    palette = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration, loop=0, optimize=True)
    print(f"wrote {out}: {len(imgs)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
