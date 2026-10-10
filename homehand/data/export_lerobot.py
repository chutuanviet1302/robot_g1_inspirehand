"""Convert the raw VLA recordings (`homehand collect-vla`) into a LeRobotDataset for SmolVLA fine-tuning.

Run inside the LeRobot environment (it does not need MuJoCo), from the project root:

    .venv-vla\\Scripts\\python -m homehand.data.export_lerobot --name vla_v1

The dataset is written to `data/lerobot/<name>/` (zip it and upload it to Kaggle, see kaggle/README.md).
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from homehand import paths
from homehand.data.record_vla import FPS, IMG, VLA_DIR
from homehand.model import spec

# SmolVLA (smolvla_base) names its three image inputs camera1..3: head, left wrist, right wrist
IMAGE_KEYS = {"head": "observation.images.camera1", "L_wrist_cam": "observation.images.camera2",
              "R_wrist_cam": "observation.images.camera3"}


def features() -> dict:
    names = list(spec.ACTUATORS)
    out = {"observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
           "action": {"dtype": "float32", "shape": (len(names),), "names": names}}
    for key in IMAGE_KEYS.values():
        out[key] = {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channel"]}
    return out


def read_video(path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def export(name: str, limit: int = 0) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    src = VLA_DIR / name
    out = paths.DATA_DIR / "lerobot" / (name + ("_smoke" if 0 < limit < 50 else ""))
    if out.exists():
        shutil.rmtree(out)
    from lerobot.configs.video import RGBEncoderConfig
    # H.264 instead of the default AV1: AV1 (SVT) encoding took 26 s per episode on the laptop CPU (4 h for the
    # dataset), H.264 (veryfast) a few seconds; GOP 2 keeps random access cheap for the training data loader
    enc = RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23, g=2)
    ds = LeRobotDataset.create(repo_id=f"homehand/{name}", fps=FPS, features=features(), root=out,
                               robot_type="unitree_g1_inspire", use_videos=True, rgb_encoder=enc,
                               streaming_encoding=True,   # no temporary PNG per frame (that took most of the time)
                               video_backend="pyav")   # torchcodec has no build for every torch on Windows
    demos = [d for p in sorted(src.glob("seed*.json")) for d in json.loads(p.read_text())["demos"]]
    if limit:
        demos = demos[:limit]
    for i, d in enumerate(demos):
        z = np.load(src / f"{d['stem']}.npz")
        vids = {cam: read_video(src / f"{d['stem']}_{cam}.mp4") for cam in IMAGE_KEYS}
        n = min(len(z["state"]), *(len(v) for v in vids.values()))
        for t in range(n):
            frame = {"observation.state": z["state"][t], "action": z["action"][t], "task": d["instruction"]}
            for cam, key in IMAGE_KEYS.items():
                frame[key] = vids[cam][t]
            ds.add_frame(frame)
        ds.save_episode()
        if (i + 1) % 25 == 0 or i + 1 == len(demos):
            print(f"[export] {i + 1}/{len(demos)} episodes", flush=True)
    ds.finalize()
    print(f"[export] LeRobotDataset at {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vla_v1")
    ap.add_argument("--limit", type=int, default=0, help="only the first N demonstrations (smoke test)")
    a = ap.parse_args()
    export(a.name, a.limit)


if __name__ == "__main__":
    main()
