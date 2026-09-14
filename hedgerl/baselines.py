"""Classical hedging policies, expressed in the same policy(obs, t) interface as the RL agents.
They read the Greeks that the environment puts in the observation, so they are marked with the
same volatility as the agent (fair comparison under misspecification)."""

import numpy as np

from hedgerl import book
from hedgerl.config import EnvConfig
from hedgerl.pricing import bs_delta, bs_gamma, leland_vol, ww_halfwidth


def _unpack(cfg: EnvConfig, obs):
    s = cfg.strike * np.exp(obs[:, 0].astype(float))
    vol = obs[:, 2].astype(float)
    h_prev = obs[:, 3].astype(float) * cfg.liability_qty
    delta = obs[:, 4].astype(float)
    gamma = obs[:, 5].astype(float) / s
    out = dict(s=s, vol=vol, h_prev=h_prev, delta=delta, gamma=gamma)
    if cfg.phase == 2:
        out.update(n_prev=obs[:, 6].astype(float) * cfg.liability_qty, delta_h=obs[:, 7].astype(float),
                   gamma_h=obs[:, 8].astype(float) / s)
    return out


class BSDelta:
    """Cost-blind Black-Scholes delta hedge, rebalanced every step. Hedge option untouched."""
    name = "bs_delta"

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def __call__(self, obs, t):
        u = _unpack(self.cfg, obs)
        h = self.cfg.liability_qty * u["delta"]
        n = u.get("n_prev", None)
        return book.holdings_to_action(self.cfg, h, n)


class LelandDelta:
    """Delta hedge using Leland's cost-adjusted volatility. Recomputes delta with sigma_L."""
    name = "leland"

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def __call__(self, obs, t):
        cfg = self.cfg
        u = _unpack(cfg, obs)
        tau = book.tau_liability(cfg, t)
        vol_l = leland_vol(u["vol"], cfg.cost, cfg.dt)
        d = bs_delta(u["s"], cfg.strike, tau, vol_l, call=True)
        if cfg.phase == 2:
            d = d + bs_delta(u["s"], cfg.strike, tau, vol_l, call=False)
        return book.holdings_to_action(cfg, cfg.liability_qty * d, u.get("n_prev"))


class WhalleyWilmott:
    """No-transaction band around the BS delta; trade to the nearest band edge when outside."""
    name = "whalley_wilmott"

    def __init__(self, cfg: EnvConfig, risk_aversion: float = 1.0):
        self.cfg, self.ra = cfg, risk_aversion

    def __call__(self, obs, t):
        cfg = self.cfg
        u = _unpack(cfg, obs)
        target = cfg.liability_qty * u["delta"]
        hw = cfg.liability_qty * ww_halfwidth(u["s"], u["gamma"], cfg.cost, self.ra)
        h = np.clip(u["h_prev"], target - hw, target + hw)
        return book.holdings_to_action(cfg, h, u.get("n_prev"))


class DeltaGamma:
    """Phase 2: neutralise gamma with the hedge option every `rebalance_every` steps, then delta
    with the underlying every step. The textbook dealer baseline."""
    name = "delta_gamma"

    def __init__(self, cfg: EnvConfig, rebalance_every: int = 5):
        assert cfg.phase == 2
        self.cfg, self.k = cfg, rebalance_every

    def __call__(self, obs, t):
        cfg = self.cfg
        u = _unpack(cfg, obs)
        q = cfg.liability_qty
        if t % self.k == 0:
            n = q * u["gamma"] / np.maximum(u["gamma_h"], 1e-10)
            n = np.clip(n, q * cfg.n_min, q * cfg.n_max)
        else:
            n = u["n_prev"]
        h = q * u["delta"] - n * u["delta_h"]
        return book.holdings_to_action(cfg, h, n)


class NoHedge:
    name = "no_hedge"

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def __call__(self, obs, t):
        z = np.zeros(obs.shape[0])
        return book.holdings_to_action(self.cfg, z, z if self.cfg.phase == 2 else None)
