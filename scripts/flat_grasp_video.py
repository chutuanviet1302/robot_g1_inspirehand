"""Expert vs learned policy on the same flat-box episode, side by side, on the whole robot (HD H.264 MP4).

    python scripts/flat_grasp_video.py --policy models/flat_grasp/policy.npz --seeds 100003,100011 \
        --out media/flat_grasp_rl_vs_expert.mp4

Each seed is one episode (same box pose, perception bias, friction and mass for both); the left half is the
expert, the right half the policy, each labelled with its outcome.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 960, 1080                 # each half of a 1920 x 1080 frame
CAMERA = ([0.30, -0.18, 0.98], 205.0, -18.0, 0.95)       # lookat, azimuth, elevation, distance


def episode(twin, seed: int, policy, renderer, cam) -> tuple[list[np.ndarray], bool]:
    from homehand.rl import flat_grasp_env as E
    twin.frames = []
    if policy is None:
        twin.reset(seed, renderer, cam)
        ex = E.Expert(twin)
        for k in range(E.EPISODE_STEPS):
            twin.step(ctrl=ex.ctrl(k * E.DT), renderer=renderer, cam=cam)
    else:
        obs = twin.reset(seed, renderer, cam)
        res = E.Residual(twin, policy)
        for _ in range(E.EPISODE_STEPS):
            obs, *_ = twin.step(ctrl=res.target(obs), renderer=renderer, cam=cam)
    return list(twin.frames), twin.success()


def label(frame: np.ndarray, text: str, ok: bool | None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    color = (255, 255, 255) if ok is None else ((60, 200, 90) if ok else (230, 70, 60))
    d.rectangle([0, 0, W, 70], fill=(20, 20, 24))
    d.text((24, 12), text, fill=color, font=font)
    return np.asarray(img)


def main() -> None:
    import imageio.v2 as imageio
    import mujoco
    from homehand.rl.flat_grasp_deploy import RobotTwin
    from homehand.rl.flat_grasp_policy import Policy
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", default="100003")
    ap.add_argument("--out", default="media/flat_grasp_rl_vs_expert.mp4")
    a = ap.parse_args()
    twin = RobotTwin()
    twin.model.vis.global_.offwidth, twin.model.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(twin.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = CAMERA
    policy = Policy(a.policy)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=50, codec="libx264", quality=None, pixelformat="yuv420p",
                                macro_block_size=1, ffmpeg_params=["-crf", "18", "-preset", "slow"])
    for seed in [int(s) for s in a.seeds.split(",")]:
        left, ok_e = episode(twin, seed, None, renderer, cam)
        right, ok_p = episode(twin, seed, policy, renderer, cam)
        print(f"seed {seed}: expert {'ok' if ok_e else 'fail'}, policy {'ok' if ok_p else 'fail'}")
        n = max(len(left), len(right))
        left += [left[-1]] * (n - len(left))
        right += [right[-1]] * (n - len(right))
        for k, (l, r) in enumerate(zip(left, right)):
            last = k >= n - 1
            tl = f"Expert (scripted) - seed {seed}" + ("" if not last else (": lifted" if ok_e else ": failed"))
            tr = f"Expert + RL residual (PPO)" + ("" if not last else (": lifted" if ok_p else ": failed"))
            frame = np.concatenate([label(l, tl, ok_e if last else None), label(r, tr, ok_p if last else None)], 1)
            for _ in range(25 if last else 1):              # hold the last frame 0.5 s
                writer.append_data(frame)
    writer.close()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
