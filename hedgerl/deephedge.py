"""Deep hedging baseline (Buehler, Gonon, Teichmann, Wood 2019): a single policy network trained
end-to-end by minimising a convex risk measure of the terminal P&L over simulated paths, with the
same accounting as the RL environment but written in torch so it is differentiable. The loss is
pfhedge's ExpectedShortfall (5%). This is the strongest *learned* baseline and, as Godin (2026)
notes, is Monte Carlo policy gradient rather than a proper RL algorithm."""

import math

import numpy as np
import torch
import torch.nn as nn
from pfhedge.nn import ExpectedShortfall

from hedgerl import book
from hedgerl.config import EnvConfig
from hedgerl.sim import PathBank

_SQRT2 = math.sqrt(2.0)


def _ncdf(x):
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _npdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _bs(s, k, tau, vol, call=True):
    tau = torch.clamp(torch.as_tensor(tau, dtype=s.dtype), min=1e-12)
    vol = torch.clamp(vol, min=1e-12)
    d1 = (torch.log(s / k) + 0.5 * vol**2 * tau) / (vol * torch.sqrt(tau))
    d2 = d1 - vol * torch.sqrt(tau)
    if call:
        price, delta = s * _ncdf(d1) - k * _ncdf(d2), _ncdf(d1)
    else:
        price, delta = k * _ncdf(-d2) - s * _ncdf(-d1), _ncdf(d1) - 1.0
    gamma = _npdf(d1) / (s * vol * torch.sqrt(tau))
    return price, delta, gamma


def _liability(cfg, s, tau, vol):
    if tau <= 1e-12:  # expiry: intrinsic value, no Greeks needed
        v = torch.clamp(s - cfg.strike, min=0.0)
        if cfg.phase == 2:
            v = v + torch.clamp(cfg.strike - s, min=0.0)
        return v, torch.zeros_like(s), torch.zeros_like(s)
    v, d, g = _bs(s, cfg.strike, tau, vol, True)
    if cfg.phase == 2:
        vp, dp, _ = _bs(s, cfg.strike, tau, vol, False)
        v, d, g = v + vp, d + dp, 2.0 * g
    return v, d, g


class DeepHedger(nn.Module):
    def __init__(self, cfg: EnvConfig, hidden: int = 64):
        super().__init__()
        self.cfg = cfg
        n_out = 1 if cfg.phase == 1 else 2
        self.net = nn.Sequential(nn.Linear(book.n_features(cfg), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out), nn.Tanh())

    def holdings(self, obs):
        a = self.net(obs)
        cfg, q = self.cfg, self.cfg.liability_qty
        h = q * (cfg.h_min + 0.5 * (a[:, 0] + 1.0) * (cfg.h_max - cfg.h_min))
        n = q * (cfg.n_min + 0.5 * (a[:, 1] + 1.0) * (cfg.n_max - cfg.n_min)) if cfg.phase == 2 else torch.zeros_like(h)
        return h, n

    def terminal_pnl(self, spot: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
        """Differentiable version of hedgerl.env.rollout for a batch of paths."""
        cfg, q = self.cfg, self.cfg.liability_qty
        B, T = spot.shape[0], spot.shape[1] - 1
        h_prev, n_prev = torch.zeros(B, dtype=spot.dtype), torch.zeros(B, dtype=spot.dtype)
        pnl = torch.zeros(B, dtype=spot.dtype)
        for t in range(T):
            s_t, s_t1, v_t, v_t1 = spot[:, t], spot[:, t + 1], vol[:, t], vol[:, t + 1]
            tau, tau1 = book.tau_liability(cfg, t), book.tau_liability(cfg, t + 1)
            V_t, d, g = _liability(cfg, s_t, tau, v_t)
            V_t1, _, _ = _liability(cfg, s_t1, tau1, v_t1)
            cols = [torch.log(s_t / cfg.strike), torch.full_like(s_t, tau / (cfg.liability_days * cfg.dt)), v_t,
                    h_prev / q, d, g * s_t]
            if cfg.phase == 2:
                th, th1 = book.tau_hedge(cfg, t), book.tau_hedge(cfg, t + 1)
                C_t, dh, gh = _bs(s_t, cfg.s0, th, v_t, True)
                C_t1, _, _ = _bs(s_t1, cfg.s0, th1, v_t1, True)
                cols += [n_prev / q, dh, gh * s_t, torch.full_like(s_t, th / (cfg.hedge_option_days * cfg.dt))]
            obs = torch.stack(cols, dim=-1).float()
            h, n = self.holdings(obs)
            h, n = h.to(spot.dtype), n.to(spot.dtype)
            dw = -q * (V_t1 - V_t) + h * (s_t1 - s_t) - cfg.cost * s_t * torch.abs(h - h_prev)
            if cfg.phase == 2:
                dw = dw + n * (C_t1 - C_t) - cfg.hedge_option_spread * C_t * torch.abs(n - n_prev)
            if t + 1 == T:
                dw = dw - cfg.cost * s_t1 * torch.abs(h)
                if cfg.phase == 2:
                    dw = dw - cfg.hedge_option_spread * C_t1 * torch.abs(n)
            pnl = pnl + dw
            h_prev, n_prev = h, n
        return pnl

    def as_policy(self):
        """Adapter to the policy(obs, t) interface used by hedgerl.env.rollout."""
        def policy(obs, t):
            with torch.no_grad():
                return self.net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        return policy


def train_deep_hedger(cfg: EnvConfig, bank: PathBank, epochs: int = 300, batch: int = 4096, lr: float = 1e-3,
                      alpha: float = 0.05, seed: int = 0, verbose: bool = True) -> DeepHedger:
    torch.manual_seed(seed)
    model = DeepHedger(cfg)
    loss_fn = ExpectedShortfall(alpha)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    spot = torch.as_tensor(bank.spot, dtype=torch.float64)
    vol = torch.as_tensor(bank.mark_vol, dtype=torch.float64)
    assert cfg.pricer == "bs" and np.allclose(bank.mark_vol, bank.price_vol), \
        "deep hedger trains on the training measure only (market vol == hedger vol)"
    premium = book.initial_premium(cfg, float(bank.price_vol[0, 0]))
    n = spot.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch,), generator=g)
        pnl = model.terminal_pnl(spot[idx], vol[idx]) / premium
        loss = loss_fn(pnl)  # ES of the loss -pnl (pfhedge convention: input is P&L)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"[deephedge] epoch {ep:4d}  ES5%(loss/premium) = {loss.item():.4f}")
    return model
