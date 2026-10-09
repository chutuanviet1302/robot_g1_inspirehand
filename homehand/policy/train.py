"""Train ACT / Diffusion Policy on a recorded dataset (GPU if available, CPU otherwise).

    homehand train --policy act --dataset expert_v1
    homehand train --policy diffusion --dataset expert_v1
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from homehand import paths
from homehand.data.record import load
from homehand.policy.models import MODELS


class ChunkSampler:
    """Vectorised minibatches of (normalised obs_t, normalised actions t..t+H-1, validity mask).

    Chunks running past the episode end are padded with the last action and masked out of the loss.
    """

    def __init__(self, data: dict, stats: dict, chunk: int, episodes: np.ndarray, device: str):
        keep = np.isin(data["episode_index"], episodes)
        ep = data["episode_index"][keep]
        end = np.zeros(len(ep), dtype=np.int64)  # exclusive end index of each frame's episode
        for e in np.unique(ep):
            idx = np.nonzero(ep == e)[0]
            end[idx] = idx[-1] + 1
        j = np.arange(len(ep))[:, None] + np.arange(chunk)[None]
        self.mask = torch.from_numpy((j < end[:, None]).astype(np.float32)).to(device)
        self.chunk_idx = torch.from_numpy(np.minimum(j, end[:, None] - 1)).to(device)
        self.obs = torch.from_numpy((data["obs"][keep] - stats["obs_mean"]) / stats["obs_std"]).float().to(device)
        self.act = torch.from_numpy((data["action"][keep] - stats["act_mean"]) / stats["act_std"]).float().to(device)
        self.n = len(ep)

    def __len__(self):
        return self.n

    def batch(self, idx: torch.Tensor):
        return self.obs[idx], self.act[self.chunk_idx[idx]], self.mask[idx]

    def random_batch(self, batch_size: int, gen: torch.Generator):
        return self.batch(torch.randint(0, self.n, (batch_size,), generator=gen).to(self.obs.device))

    def iterate(self, batch_size: int):
        for i in range(0, self.n, batch_size):
            yield self.batch(torch.arange(i, min(i + batch_size, self.n), device=self.obs.device))


def train(policy: str = "act", dataset: str = "expert_v1", steps: int = 20_000, batch_size: int = 256,
          lr: float = 3e-4, out_name: str | None = None, device: str | None = None, seed: int = 0,
          log_every: int = 500, init: str | None = None) -> Path:
    """`init`: a model directory (or policy.pt) to continue training from (same policy kind and dataset)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load(dataset)
    stats = {k: np.asarray(v, np.float32) for k, v in meta["stats"].items()}
    model = MODELS[policy](meta["obs_dim"], meta["act_dim"]).to(device)
    prev_steps = 0
    if init:
        ip = Path(init)
        ip = ip / "policy.pt" if ip.is_dir() else ip
        ck = torch.load(ip, map_location=device, weights_only=False)
        assert ck["kind"] == policy, f"{ip} is a {ck['kind']} checkpoint"
        model.load_state_dict(ck["state_dict"])
        prev_log = ip.parent / "train_log.json"
        prev_steps = json.loads(prev_log.read_text())["steps"] if prev_log.exists() else 0
        print(f"[train] continuing from {ip} ({prev_steps} steps)")
    chunk = model.chunk
    eps = np.unique(data["episode_index"])
    rng = np.random.default_rng(seed)
    val_eps = rng.choice(eps, size=max(1, len(eps) // 10), replace=False)
    train_eps = np.setdiff1d(eps, val_eps)
    tr = ChunkSampler(data, stats, chunk, train_eps, device)
    va = ChunkSampler(data, stats, chunk, val_eps, device)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4 if policy == "act" else 1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    ema = copy.deepcopy(model).eval() if policy == "diffusion" else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {policy}: {n_params / 1e6:.2f} M params, {len(tr)} train / {len(va)} val frames, device={device}")

    history = []
    step, t0 = 0, time.time()
    # Resumable: a checkpoint of the full training state is written at every log point, so a reboot loses at
    # most `log_every` steps (two reboots during a 100-minute Diffusion Policy run had lost everything).
    out_dir = paths.MODELS_DIR / (out_name or f"{policy}_{dataset}")
    ckpt = out_dir / "training_state.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        if st.get("config") == [policy, dataset, steps, batch_size, lr, str(init)]:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            if ema is not None:
                ema.load_state_dict(st["ema"])
            gen.set_state(st["gen"].cpu())
            step, history = st["step"], st["history"]
            t0 -= history[-1]["elapsed_s"] if history else 0.0
            print(f"[train] resumed from {ckpt} at step {step}")
    while step < steps:
        obs, act, mask = tr.random_batch(batch_size, gen)
        model.train()
        loss, parts = model.loss(obs, act, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            decay = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        step += 1
        if step % log_every == 0 or step == steps:
            val = evaluate_mse(ema or model, va, batch_size)
            history.append({"step": step, "train_loss": round(loss.item(), 5), "val_action_mse": round(val, 5),
                            **{k: round(v, 5) for k, v in parts.items()}, "elapsed_s": round(time.time() - t0, 1)})
            print(f"[train] step {step:6d} loss {loss.item():.4f} val action-MSE {val:.4f} ({time.time() - t0:.0f}s)")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"config": [policy, dataset, steps, batch_size, lr, str(init)], "step": step,
                        "history": history, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                        "gen": gen.get_state()}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)                # atomic: a crash never leaves half a file

    out_dir.mkdir(parents=True, exist_ok=True)
    final = ema or model
    torch.save({"kind": policy, "cfg": final.cfg, "state_dict": final.state_dict(),
                "stats": {k: v.tolist() for k, v in stats.items()}, "dataset": dataset,
                "demo_length": meta["demo_length"], "feature_names": meta["feature_names"]}, out_dir / "policy.pt")
    (out_dir / "train_log.json").write_text(json.dumps({
        "policy": policy, "dataset": dataset, "params_M": round(n_params / 1e6, 3), "steps": prev_steps + steps,
        "continued_from": str(init) if init else None,
        "batch_size": batch_size, "lr": lr, "device": device, "history": history,
        "train_seconds": round(time.time() - t0, 1)}, indent=2))
    ckpt.unlink(missing_ok=True)
    print(f"[train] saved {out_dir / 'policy.pt'}")
    return out_dir


@torch.no_grad()
def evaluate_mse(model, sampler: ChunkSampler, batch_size: int) -> float:
    """MSE of the predicted chunk vs. the demonstration (normalised units, valid steps only)."""
    model.eval()
    tot, n = 0.0, 0.0
    for obs, act, mask in sampler.iterate(batch_size):
        pred = model.predict(obs)
        tot += (((pred - act) ** 2).mean(-1) * mask).sum().item()
        n += mask.sum().item()
    return tot / max(n, 1.0)
