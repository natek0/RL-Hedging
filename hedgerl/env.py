"""Gymnasium environment: one hedging episode = one simulated path from a PathBank."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hedgerl import book
from hedgerl.config import EnvConfig
from hedgerl.sim import PathBank


class HedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, bank: PathBank, seed: int = 0, sequential: bool = False):
        super().__init__()
        assert bank.n_steps == cfg.n_steps
        self.cfg, self.bank = cfg, bank
        self.sequential = sequential
        self._rng = np.random.default_rng(seed)
        self._next_path = 0
        n_act = 1 if cfg.phase == 1 else 2
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(book.n_features(cfg),), dtype=np.float32)
        self.premium = book.initial_premium(cfg, float(bank.price_vol[0, 0]))
        self.path = self.t = self.h_prev = self.n_prev = None

    def _obs(self):
        s = self.bank.spot[self.path, self.t:self.t + 1]
        vol = self.bank.mark_vol[self.path, self.t:self.t + 1]
        return book.features(self.cfg, s, vol, self.t, np.array([self.h_prev]), np.array([self.n_prev]))[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.sequential:
            self.path = self._next_path % self.bank.n_paths
            self._next_path += 1
        else:
            self.path = int(self._rng.integers(self.bank.n_paths))
        self.t, self.h_prev, self.n_prev = 0, 0.0, 0.0
        return self._obs(), {}

    def step(self, action):
        cfg, t, p = self.cfg, self.t, self.path
        h, n = book.action_to_holdings(cfg, np.asarray(action)[None, :])
        s_t, s_t1 = self.bank.spot[p, t:t + 1], self.bank.spot[p, t + 1:t + 2]
        pv_t, pv_t1 = self.bank.price_vol[p, t:t + 1], self.bank.price_vol[p, t + 1:t + 2]
        dw, cost = book.step_pnl(cfg, s_t, s_t1, pv_t, pv_t1, t, h, n, np.array([self.h_prev]), np.array([self.n_prev]))
        r = float(book.reward(cfg, dw, self.premium)[0])
        self.h_prev, self.n_prev = float(h[0]), float(n[0])
        self.t += 1
        terminated = self.t >= cfg.n_steps
        info = {"dw": float(dw[0]), "cost": float(cost[0]), "h": self.h_prev, "n": self.n_prev}
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, r, terminated, False, info


def rollout(cfg: EnvConfig, bank: PathBank, policy, premium: float | None = None):
    """Vectorised episode over every path in `bank`. `policy(obs, t) -> raw actions in [-1,1]^k`.
    Returns a dict with per-path P&L, per-step increments, holdings, costs. Accounting is identical
    to HedgingEnv.step (tested)."""
    n, T = bank.n_paths, bank.n_steps
    premium = premium if premium is not None else book.initial_premium(cfg, float(bank.price_vol[0, 0]))
    h_prev, n_prev = np.zeros(n), np.zeros(n)
    dws, costs, hs, ns, rs = [], [], [], [], []
    for t in range(T):
        s_t, s_t1 = bank.spot[:, t], bank.spot[:, t + 1]
        pv_t, pv_t1 = bank.price_vol[:, t], bank.price_vol[:, t + 1]
        obs = book.features(cfg, s_t, bank.mark_vol[:, t], t, h_prev, n_prev)
        a = policy(obs, t)
        h, n_ = book.action_to_holdings(cfg, a)
        dw, cost = book.step_pnl(cfg, s_t, s_t1, pv_t, pv_t1, t, h, n_, h_prev, n_prev)
        dws.append(dw); costs.append(cost); hs.append(h); ns.append(n_); rs.append(book.reward(cfg, dw, premium))
        h_prev, n_prev = h, n_
    dws, costs, hs, ns, rs = map(lambda x: np.stack(x, axis=1), (dws, costs, hs, ns, rs))
    return {"pnl": dws.sum(1), "dw": dws, "cost": costs, "h": hs, "n": ns, "reward": rs, "premium": premium}
