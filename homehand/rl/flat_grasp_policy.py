"""The trained flat-box grasp policy in numpy (no jax / torch at run time): normaliser + MLP, exported either by
`train_flat_grasp.export_npz` (Brax PPO on MJX: swish layers, tanh(mean)) or by `train_flat_grasp_cpu.export_npz`
(SB3 PPO: tanh layers, the Gaussian's mean clipped to the action box)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Policy:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.sb3 = "format" in z.files and str(z["format"]) == "sb3"
        if self.sb3:
            self.mean, self.std = z["obs_mean"], np.sqrt(z["obs_var"] + float(z["eps"]))
            self.clip = float(z["clip"])
        else:
            self.mean, self.std, self.clip = z["obs_mean"], z["obs_std"], None
        n = len([k for k in z.files if k.startswith("w")])
        self.layers = [(z[f"w{k}"], z[f"b{k}"]) for k in range(n)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        for k, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if k < len(self.layers) - 1:
                x = np.tanh(x) if self.sb3 else x / (1 + np.exp(-x))     # (swish for Brax)
        if self.sb3:
            return np.clip(x, -1.0, 1.0)
        return np.tanh(x[..., : x.shape[-1] // 2])     # Brax: (loc, scale) -> tanh(loc)
