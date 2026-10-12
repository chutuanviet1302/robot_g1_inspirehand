"""Train the residual policy with PPO (Stable-Baselines3), several simulation workers on the CPU.

    homehand rl-train --steps 1000000 --workers 4 --randomization medium

Writes models/<name>/policy.zip, obs_stats.npz (observation normalisation, used by `ResidualPolicy`) and
progress.csv (per rollout: reward per term summed per pick, pick_success = share of picks still in the hand at the release point,
steps/s) for the learning curves.
Resumable: an existing policy.zip in the output directory is loaded and training continues.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _make(rank: int, randomization: str, neighbour_penalty: float):
    def thunk():
        import os
        os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
        import torch
        torch.set_num_threads(1)
        from homehand.rl.residual_env import ResidualPickEnv
        return ResidualPickEnv(randomization=randomization, seed=1000 + rank, neighbour_penalty=neighbour_penalty)
    return thunk


def train(steps: int = 1_000_000, workers: int = 4, randomization: str = "medium", name: str = "residual_ppo",
          neighbour_penalty: float | None = None, seed: int = 0) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from homehand import paths
    from homehand.resources import safe_workers
    from homehand.rl.residual_env import NEIGHBOUR_PENALTY

    import os
    # one BLAS thread per simulation worker (no oversubscription; read by the workers at import). The throughput
    # limit is elsewhere: the vectorised env waits for its slowest worker at every step, and a step can be one
    # control step or a reset / a long expert-driven stretch (3 workers: ~22 agent steps/s in total)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    out = paths.MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    npen = NEIGHBOUR_PENALTY if neighbour_penalty is None else neighbour_penalty
    n = safe_workers(workers)
    venv = SubprocVecEnv([_make(i, randomization, npen) for i in range(n)], start_method="spawn")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    if (out / "policy.zip").exists():
        model = PPO.load(out / "policy.zip", env=venv, device="cpu")
        st = np.load(out / "obs_stats.npz")
        venv.obs_rms.mean, venv.obs_rms.var = st["mean"], st["var"]
        print(f"[rl] continuing from {out / 'policy.zip'} ({model.num_timesteps} steps)")
    else:
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
                    seed=seed, device="cpu", verbose=0)

    log_path = out / "progress.csv"

    class Log(BaseCallback):
        """Mean reward per term and the share of picks that end in the bin, per rollout."""

        def _on_training_start(self):
            self.terms, self.ends, self.t0 = defaultdict(list), [], time.time()
            self.f = open(log_path, "a", newline="")
            self.w = None

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for k, v in info.get("terms", {}).items():
                    self.terms[k].append(v)
                if done:
                    self.ends.append(float(info.get("carried", False)))
            return True

        def _on_rollout_end(self):
            row = {"steps": self.num_timesteps, "picks": len(self.ends),
                   "pick_success": round(float(np.mean(self.ends)), 4) if self.ends else "",
                   "steps_per_s": round(self.num_timesteps / max(time.time() - self.t0, 1e-6), 1)}
            row.update({k: round(float(np.sum(v)) / max(len(self.ends), 1), 4) for k, v in self.terms.items()})
            if self.w is None:
                self.w = csv.DictWriter(self.f, fieldnames=list(row))
                if self.f.tell() == 0:
                    self.w.writeheader()
            self.w.writerow(row)
            self.f.flush()
            print(f"[rl] {row}", flush=True)
            self.terms, self.ends = defaultdict(list), []

        def _on_training_end(self):
            self.f.close()

    save_every = 50_000
    done = model.num_timesteps
    try:
        while done < steps:
            chunk = min(save_every, steps - done)
            model.learn(total_timesteps=chunk, callback=Log(), reset_num_timesteps=False)
            done = model.num_timesteps
            model.save(out / "policy.zip")
            np.savez(out / "obs_stats.npz", mean=venv.obs_rms.mean, var=venv.obs_rms.var, clip=10.0)
    finally:
        venv.close()
    return out
