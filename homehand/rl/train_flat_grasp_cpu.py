"""PPO (Stable-Baselines3) on the flat-box grasp, plain MuJoCo on the CPU, several worker processes.

Why not MJX: on a Kaggle T4 this contact-rich model (mesh-hull fingers, elliptic cones, the robot's solver settings
that the grasp outcomes depend on) ran at ~850 policy steps/s per GPU; plain MuJoCo on 7 CPU cores does ~3000.
The MJX env (`flat_grasp_env.make_mjx_env`, `train_flat_grasp.py`) stays for a larger GPU.

    python -m homehand.rl.train_flat_grasp_cpu --steps 20000000 --workers 7

By default the policy is a residual on the expert (`FlatGraspGym`); --from-scratch trains it alone.

Writes models/flat_grasp/: policy.zip (SB3, resumable), policy.npz (numpy inference: `flat_grasp_policy.Policy`),
progress.csv (per rollout: episode success, lift, reward, steps/s).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

XML = "generated/flat_grasp_train.xml"
FRAME = "generated/flat_grasp.json"


RES_RATE = np.array([0.5] * 7 + [2.0] * 6)       # residual: offset change per second at |action| = 1
RES_MAX = np.array([0.35] * 7 + [0.6] * 6)       # residual: largest offset from the expert's targets


class FlatGraspGym:
    """Gymnasium wrapper of `FlatGraspSim`: each reset draws a fresh episode.

    residual=True: the policy corrects the expert. The expert plans its grasp from the first frame as usual
    (`flat_grasp_env.Expert`); the policy integrates an offset on its 13 joint targets (rate- and size-limited),
    observing everything the plain policy does plus that offset. Zero action = the expert."""

    def __init__(self, seed: int, residual: bool = True):
        import gymnasium as gym
        from homehand import paths
        from homehand.rl import flat_grasp_env as E
        self.E = E
        root = paths.GENERATED_DIR.parent
        self.sim = E.FlatGraspSim(root / XML, root / FRAME)
        n = self.sim.lay.lo.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (E.OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.residual = residual
        if residual:
            from homehand.control.ik import ArmIK
            self.ik = ArmIK()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (E.OBS_SIZE + n,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.metadata, self.render_mode, self.spec = {}, None, None

    def reset(self, seed=None, options=None):
        obs = self.sim.reset(int(self.rng.integers(2 ** 31)))
        self.ret = 0.0
        if self.residual:
            self.expert = self.E.Expert(self.sim, self.ik)
            self.offset = np.zeros(self.sim.lay.lo.size)
            obs = self._res_obs(obs)
        return obs, {}

    def _res_obs(self, obs):
        return np.r_[obs, self.offset / RES_MAX].astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        if self.residual:
            self.offset = np.clip(self.offset + action * RES_RATE * self.E.DT, -RES_MAX, RES_MAX)
            target = self.expert.ctrl(self.sim.t * self.E.DT) + self.offset
            obs, r, success, end = self.sim.step(ctrl=target)
            r -= 0.01 * float(np.sum(action ** 2))
            obs = self._res_obs(obs)
        else:
            obs, r, success, end = self.sim.step(action)
        self.ret += r
        bad = not np.isfinite(obs).all() or self.sim.box_true()[2] < self.E.TABLE_Z - 0.05
        if bad:
            obs = np.nan_to_num(obs).astype(np.float32)
        info = {}
        if end or bad:
            info = {"success": float(self.sim.success()), "lift_cm": self.sim.lift_cm(), "ret": self.ret}
        return obs, r, bool(bad), bool(end and not bad), info

    def close(self):
        pass


def _make(rank: int, seed: int, residual: bool = True):
    def thunk():
        import gymnasium as gym
        import torch
        torch.set_num_threads(1)

        class Env(FlatGraspGym, gym.Env):
            pass
        return Env(seed * 1000 + rank, residual)
    return thunk


def export_npz(model, venv, path: Path) -> None:
    """SB3 MlpPolicy (tanh hidden layers) + VecNormalize statistics as plain arrays."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    arrays = {"format": np.array("sb3"), "obs_mean": venv.obs_rms.mean, "obs_var": venv.obs_rms.var,
              "clip": np.array(venv.clip_obs), "eps": np.array(venv.epsilon)}
    k = 0
    while f"mlp_extractor.policy_net.{2 * k}.weight" in sd:
        arrays[f"w{k}"] = sd[f"mlp_extractor.policy_net.{2 * k}.weight"].T
        arrays[f"b{k}"] = sd[f"mlp_extractor.policy_net.{2 * k}.bias"]
        k += 1
    arrays[f"w{k}"] = sd["action_net.weight"].T
    arrays[f"b{k}"] = sd["action_net.bias"]
    np.savez(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/flat_grasp")
    ap.add_argument("--from-scratch", action="store_true", help="the policy alone (no expert underneath)")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([_make(i, a.seed, not a.from_scratch) for i in range(a.workers)], start_method="spawn")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.98)
    if (out / "policy.zip").exists():
        model = PPO.load(out / "policy.zip", env=venv, device="cpu")
        st = np.load(out / "vecnorm.npz")
        venv.obs_rms.mean, venv.obs_rms.var = st["mean"], st["var"]
        venv.ret_rms.var = st["ret_var"]
        print(f"continuing from {out / 'policy.zip'} ({model.num_timesteps} steps)", flush=True)
    else:
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=5, gamma=0.98, gae_lambda=0.95,
                    learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       log_std_init=-0.5 if a.from_scratch else -1.0),
                    seed=a.seed, device="cpu", verbose=0)

    class Log(BaseCallback):
        def _on_training_start(self):
            self.eps, self.t0, self.n0 = [], time.time(), self.num_timesteps
            path = out / "progress.csv"
            self.f = open(path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=["steps", "episodes", "success", "lift_cm", "return",
                                                        "steps_per_s", "wall_s"])
            if path.stat().st_size == 0:
                self.w.writeheader()

        def _on_step(self):
            self.eps += [i for i in self.locals["infos"] if "success" in i]
            return True

        def _on_rollout_end(self):
            if not self.eps:
                return
            row = {"steps": self.num_timesteps, "episodes": len(self.eps),
                   "success": round(float(np.mean([e["success"] for e in self.eps])), 4),
                   "lift_cm": round(float(np.mean([e["lift_cm"] for e in self.eps])), 2),
                   "return": round(float(np.mean([e["ret"] for e in self.eps])), 1),
                   "steps_per_s": round((self.num_timesteps - self.n0) / (time.time() - self.t0), 1),
                   "wall_s": round(time.time() - self.t0)}
            self.w.writerow(row)
            self.f.flush()
            print(row, flush=True)
            self.eps = []

        def _on_training_end(self):
            self.f.close()

    done = model.num_timesteps
    try:
        while done < a.steps:
            model.learn(total_timesteps=min(500_000, a.steps - done), callback=Log(), reset_num_timesteps=False)
            done = model.num_timesteps
            model.save(out / "policy.zip")
            np.savez(out / "vecnorm.npz", mean=venv.obs_rms.mean, var=venv.obs_rms.var, ret_var=venv.ret_rms.var)
            export_npz(model, venv, out / "policy.npz")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
