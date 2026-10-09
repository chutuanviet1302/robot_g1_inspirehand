"""Imitation-learning policies over the pick-and-place skill features (state-based, laptop-sized).

ACT              Action Chunking with Transformers (Zhao et al. 2023): CVAE + transformer decoder predicting a
                 chunk of future joint targets; temporal ensembling at inference.
DiffusionPolicy  Diffusion Policy (Chi et al. 2023), CNN variant: a 1D temporal U-Net with FiLM conditioning
                 denoises an action chunk; DDPM training, DDIM sampling; receding-horizon execution.

Both are trained on the same dataset with the same normalisation and evaluated in the same harness.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------- ACT
class ACT(nn.Module):
    kind = "act"

    def __init__(self, obs_dim: int, act_dim: int, chunk: int = 25, d_model: int = 256, n_heads: int = 8,
                 enc_layers: int = 3, dec_layers: int = 4, latent_dim: int = 32, dropout: float = 0.1,
                 kl_weight: float = 10.0):
        super().__init__()
        self.cfg = dict(obs_dim=obs_dim, act_dim=act_dim, chunk=chunk, d_model=d_model, n_heads=n_heads,
                        enc_layers=enc_layers, dec_layers=dec_layers, latent_dim=latent_dim, dropout=dropout,
                        kl_weight=kl_weight)
        self.chunk, self.latent_dim, self.kl_weight = chunk, latent_dim, kl_weight

        def enc(n):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n)

        # CVAE posterior q(z | obs, actions) - only used during training
        self.post_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.post_obs = nn.Linear(obs_dim, d_model)
        self.post_act = nn.Linear(act_dim, d_model)
        self.post_pos = nn.Parameter(torch.randn(1, chunk + 2, d_model) * 0.02)
        self.post_enc = enc(enc_layers)
        self.post_out = nn.Linear(d_model, 2 * latent_dim)
        # policy: memory = [z, obs] tokens -> decoder queries -> action chunk
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.mem_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.mem_enc = enc(enc_layers)
        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.head = nn.Linear(d_model, act_dim)

    def _decode(self, obs, z):
        mem = torch.stack([self.z_proj(z), self.obs_proj(obs)], 1) + self.mem_pos
        mem = self.mem_enc(mem)
        q = self.queries.expand(obs.shape[0], -1, -1)
        return self.head(self.decoder(q, mem))

    def loss(self, obs, actions, mask):
        b = obs.shape[0]
        toks = torch.cat([self.post_cls.expand(b, -1, -1), self.post_obs(obs)[:, None], self.post_act(actions)], 1)
        h = self.post_enc(toks + self.post_pos)[:, 0]
        mu, logvar = self.post_out(h).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self._decode(obs, z)
        l1 = (F.l1_loss(pred, actions, reduction="none").mean(-1) * mask).sum() / mask.sum()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}

    @torch.no_grad()
    def predict(self, obs):
        z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        return self._decode(obs, z)


# ----------------------------------------------------------------------------------------- Diffusion
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], -1)


class CondResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.Mish())
        self.c2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(x)
        scale, bias = self.film(cond).unsqueeze(-1).chunk(2, 1)
        h = self.c2(h * (1 + scale) + bias)
        return h + self.res(x)


class UNet1D(nn.Module):
    def __init__(self, act_dim, cond_dim, dims=(128, 256, 512), t_dim=128):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(), nn.Linear(4 * t_dim, t_dim))
        c = t_dim + cond_dim
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        chans = [act_dim, *dims]
        for i in range(len(dims)):
            last = i == len(dims) - 1
            self.downs.append(nn.ModuleList([CondResBlock1D(chans[i], chans[i + 1], c),
                                             CondResBlock1D(chans[i + 1], chans[i + 1], c),
                                             nn.Conv1d(chans[i + 1], chans[i + 1], 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondResBlock1D(dims[-1], dims[-1], c), CondResBlock1D(dims[-1], dims[-1], c)])
        for i in reversed(range(1, len(dims))):
            self.ups.append(nn.ModuleList([CondResBlock1D(2 * dims[i], dims[i - 1], c),
                                           CondResBlock1D(dims[i - 1], dims[i - 1], c),
                                           nn.ConvTranspose1d(dims[i - 1], dims[i - 1], 4, 2, 1)]))
        self.out = nn.Sequential(nn.Conv1d(dims[0], dims[0], 5, padding=2), nn.GroupNorm(8, dims[0]), nn.Mish(),
                                 nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, cond):  # x: (B, T, A)
        x = x.transpose(1, 2)
        c = torch.cat([self.t_emb(t), cond], -1)
        skips = []
        for r1, r2, down in self.downs:
            x = r2(r1(x, c), c)
            skips.append(x)
            x = down(x)
        for m in self.mid:
            x = m(x, c)
        for r1, r2, up in self.ups:
            x = torch.cat([x, skips.pop()], 1)
            x = up(r2(r1(x, c), c))
        return self.out(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    kind = "diffusion"

    def __init__(self, obs_dim: int, act_dim: int, chunk: int = 16, n_train_steps: int = 100,
                 n_infer_steps: int = 30, cond_dim: int = 256):
        super().__init__()
        self.cfg = dict(obs_dim=obs_dim, act_dim=act_dim, chunk=chunk, n_train_steps=n_train_steps,
                        n_infer_steps=n_infer_steps, cond_dim=cond_dim)
        self.chunk, self.T, self.n_infer = chunk, n_train_steps, n_infer_steps
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, cond_dim), nn.Mish(), nn.Linear(cond_dim, cond_dim))
        self.net = UNet1D(act_dim, cond_dim)
        # squared-cosine noise schedule (as in the Diffusion Policy reference implementation)
        s = 0.008
        steps = torch.arange(n_train_steps + 1, dtype=torch.float64) / n_train_steps
        ac = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = torch.clamp(1 - ac[1:] / ac[:-1], max=0.999).float()
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0))

    def loss(self, obs, actions, mask):
        b = obs.shape[0]
        t = torch.randint(0, self.T, (b,), device=obs.device)
        noise = torch.randn_like(actions)
        a = self.alphas_cumprod[t][:, None, None]
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        pred = self.net(noisy, t, self.obs_enc(obs))
        mse = ((pred - noise) ** 2).mean(-1)
        loss = (mse * mask).sum() / mask.sum()
        return loss, {"mse": loss.item()}

    @torch.no_grad()
    def predict(self, obs, generator: torch.Generator | None = None):
        """DDIM (eta=0) sampling with `n_infer` steps."""
        b = obs.shape[0]
        cond = self.obs_enc(obs)
        x = torch.randn(b, self.chunk, self.cfg["act_dim"], device=obs.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, self.n_infer, device=obs.device).long()
        for i, t in enumerate(ts):
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
            eps = self.net(x, t.expand(b), cond)
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(-5, 5)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x


MODELS = {"act": ACT, "diffusion": DiffusionPolicy}
