"""PPO (Brax) on the MJX flat-box grasping env. Runs where jax + brax + mujoco-mjx are (a GPU: Kaggle; see
kaggle/flat_grasp_ppo.ipynb). Needs only this folder's `flat_grasp_env.py` and the two generated files.

    python train_flat_grasp.py --xml flat_grasp_train.xml --frame flat_grasp.json --steps 60000000 --out out/

Writes out/params.pkl (the latest policy, also every evaluation), out/progress.jsonl (one line per evaluation) and
out/policy.npz (the policy for numpy inference: `flat_grasp_policy.Policy`).
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_grasp_env as E  # noqa: E402

PPO = dict(num_envs=2048, batch_size=512, num_minibatches=32, unroll_length=10, num_updates_per_batch=8,
           discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, reward_scaling=1.0, clipping_epsilon=0.2,
           normalize_observations=True, max_grad_norm=1.0)
POLICY_LAYERS = (256, 256, 128)
VALUE_LAYERS = (256, 256, 256)


def export_npz(params, path: Path) -> None:
    """Observation normaliser + policy MLP as plain arrays (brax: swish hidden layers, tanh(mean) for the
    deterministic action)."""
    norm, policy = params[0], params[1]
    arrays = {"obs_mean": np.asarray(norm.mean), "obs_std": np.asarray(norm.std)}
    layers = policy["params"]
    for k in range(len(layers)):
        arrays[f"w{k}"] = np.asarray(layers[f"hidden_{k}"]["kernel"])
        arrays[f"b{k}"] = np.asarray(layers[f"hidden_{k}"]["bias"])
    np.savez(path, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="flat_grasp_train.xml")
    ap.add_argument("--frame", default="flat_grasp.json")
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=PPO["num_envs"])
    ap.add_argument("--batch-size", type=int, default=PPO["batch_size"])
    ap.add_argument("--minibatches", type=int, default=PPO["num_minibatches"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = E.make_mjx_env(a.xml, a.frame)
    eval_env = E.make_mjx_env(a.xml, a.frame)
    print("devices:", jax.devices(), flush=True)
    t0 = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "time_s": round(time.time() - t0, 1)}
        row.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/") and np.ndim(v) == 0})
        with open(out / "progress.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{step:>11,d}  {row['time_s']:>7.0f}s  reward {row.get('eval/episode_reward', 0):8.1f}  "
              f"success-steps {row.get('eval/episode_success', 0):6.1f}  "
              f"lift {row.get('eval/episode_lift_cm', 0) / E.EPISODE_STEPS:5.2f} cm", flush=True)

    def save(step, make_policy, params):
        with open(out / "params.pkl", "wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)

    cfg = dict(PPO, num_envs=a.num_envs, batch_size=a.batch_size, num_minibatches=a.minibatches)
    make_policy, params, _ = ppo.train(
        environment=env, eval_env=eval_env, num_timesteps=a.steps, num_evals=a.evals,
        episode_length=E.EPISODE_STEPS, action_repeat=1, seed=a.seed,
        network_factory=functools.partial(ppo_networks.make_ppo_networks,
                                          policy_hidden_layer_sizes=POLICY_LAYERS,
                                          value_hidden_layer_sizes=VALUE_LAYERS),
        randomization_fn=functools.partial(E.randomize, lay=env.lay),
        progress_fn=progress, policy_params_fn=save, **cfg)
    save(None, make_policy, params)
    export_npz(params, out / "policy.npz")
    (out / "config.json").write_text(json.dumps({"ppo": cfg, "steps": a.steps, "policy_layers": POLICY_LAYERS,
                                                 "value_layers": VALUE_LAYERS, "seed": a.seed}, indent=2))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
