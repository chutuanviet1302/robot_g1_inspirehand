"""Head-mounted RGB-D camera: rendering, intrinsics and deprojection to world coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, metres along the optical axis
    seg: np.ndarray | None       # (H, W) int32 geom id (-1 = background), only for ground-truth labelling
    K: np.ndarray                # 3x3 intrinsics
    cam_pos: np.ndarray          # camera position (world)
    cam_mat: np.ndarray          # 3x3 camera-to-world rotation (MuJoCo convention: looks along -z, y up)


class HeadCamera:
    def __init__(self, env, camera: str = "head", width: int = 640, height: int = 480):
        self.env = env
        self.camera = camera
        self.cam_id = env.model.camera(camera).id
        self.width, self.height = width, height
        fovy = np.deg2rad(env.model.cam_fovy[self.cam_id])
        f = 0.5 * height / np.tan(fovy / 2)
        self.K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])

    def capture(self, with_seg: bool = False) -> Frame:
        m, d = self.env.model, self.env.data
        r = self.env.renderer(self.width, self.height)
        r.update_scene(d, camera=self.camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.camera)
        depth = r.render().astype(np.float32).copy()
        r.disable_depth_rendering()
        seg = None
        if with_seg:
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=self.camera)
            s = r.render()
            r.disable_segmentation_rendering()
            seg = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
        return Frame(rgb, depth, seg, self.K, d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy())


def deproject(frame: Frame, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Pixels (u right, v down) -> world points (N, 3) using the depth image."""
    z = frame.depth[vs, us]
    K = frame.K
    x = (us - K[0, 2]) / K[0, 0] * z
    y = -(vs - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, -z], axis=1)          # camera frame: x right, y up, looking along -z
    return pc @ frame.cam_mat.T + frame.cam_pos
