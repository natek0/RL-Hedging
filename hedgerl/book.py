"""Shared accounting for the option book. Both the gymnasium environment (one path at a time) and
the vectorised evaluator (all paths at once) call these functions, so a P&L number produced by an
SB3 agent, a baseline, or the deep hedger is computed by the same code path.

Two volatilities are kept apart on purpose:
  * price_vol  - what the *market* uses to value the options (liability marks, hedge-option quotes).
                 Under a misspecified test measure this is the true model's price, so nobody can
                 earn P&L by trading a mispriced hedge option.
  * mark_vol   - what the *hedger* believes; used for the Greeks in the observation and by every
                 baseline. Under misspecification the hedger is wrong, the market is not.
Under the training measure (Heston, instantaneous vol) the two coincide.

Conventions: r = 0, daily steps, t = 0..T-1 are decision times, spot S_t observed, holdings chosen
at t are held over (t, t+1]. Liability is short `q` units of each leg. Wealth increment

    dw_t = -q (V_{t+1} - V_t) + h_t (S_{t+1} - S_t) + n_t (C_{t+1} - C_t)
           - cost * S_t |h_t - h_{t-1}| - spread * C_t |n_t - n_{t-1}|

and at the final step both hedge positions are closed at cost. Sum over t of dw_t is the total
P&L including the premium received at t = 0 (it enters through -q(V_T - V_0))."""

import numpy as np

from hedgerl.config import EnvConfig
from hedgerl.pricing import bs_delta, bs_gamma, bs_price, merton_price


def _option_value(cfg: EnvConfig, s, k, tau, price_vol, call):
    if cfg.pricer == "bs":
        return bs_price(s, k, tau, price_vol, call=call)
    if cfg.pricer == "merton":
        return merton_price(s, k, tau, call=call, **cfg.merton)
    raise ValueError(cfg.pricer)


def liability_value(cfg: EnvConfig, s, tau, price_vol):
    v = _option_value(cfg, s, cfg.strike, tau, price_vol, True)
    if cfg.phase == 2:
        v = v + _option_value(cfg, s, cfg.strike, tau, price_vol, False)
    return v


def liability_greeks(cfg: EnvConfig, s, tau, mark_vol):
    """Hedger's Black-Scholes delta and gamma of ONE unit of the liability legs."""
    d = bs_delta(s, cfg.strike, tau, mark_vol, call=True)
    g = bs_gamma(s, cfg.strike, tau, mark_vol)
    if cfg.phase == 2:
        d = d + bs_delta(s, cfg.strike, tau, mark_vol, call=False)
        g = 2.0 * g
    return d, g


def hedge_option_value(cfg: EnvConfig, s, tau_h, price_vol):
    return _option_value(cfg, s, cfg.s0, tau_h, price_vol, True)


def hedge_option_greeks(cfg: EnvConfig, s, tau_h, mark_vol):
    return bs_delta(s, cfg.s0, tau_h, mark_vol, call=True), bs_gamma(s, cfg.s0, tau_h, mark_vol)


def tau_liability(cfg: EnvConfig, t):
    return (cfg.liability_days - t) * cfg.dt


def tau_hedge(cfg: EnvConfig, t):
    return (cfg.hedge_option_days - t) * cfg.dt


def initial_premium(cfg: EnvConfig, price_vol0: float) -> float:
    v0 = liability_value(cfg, np.array([cfg.s0]), tau_liability(cfg, 0), np.array([price_vol0]))
    return float(cfg.liability_qty * v0[0])


def n_features(cfg: EnvConfig) -> int:
    return 6 if cfg.phase == 1 else 10


def features(cfg: EnvConfig, s, mark_vol, t, h_prev, n_prev):
    """Observation for a batch. Greeks are included as features: the agent is allowed to know the
    Black-Scholes hedge, the question is whether it learns to deviate from it usefully."""
    tau = tau_liability(cfg, t)
    d, g = liability_greeks(cfg, s, tau, mark_vol)
    q = cfg.liability_qty
    cols = [np.log(s / cfg.strike), np.full_like(s, tau / (cfg.liability_days * cfg.dt)), mark_vol,
            h_prev / q, d, g * s]
    if cfg.phase == 2:
        tau_h = tau_hedge(cfg, t)
        dh, gh = hedge_option_greeks(cfg, s, tau_h, mark_vol)
        cols += [n_prev / q, dh, gh * s, np.full_like(s, tau_h / (cfg.hedge_option_days * cfg.dt))]
    return np.stack(cols, axis=-1).astype(np.float32)


def action_to_holdings(cfg: EnvConfig, a):
    """Map raw actions in [-1, 1]^k to holdings in units of the underlying / hedge option."""
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    q = cfg.liability_qty
    h = q * (cfg.h_min + 0.5 * (a[..., 0] + 1.0) * (cfg.h_max - cfg.h_min))
    if cfg.phase == 2:
        n = q * (cfg.n_min + 0.5 * (a[..., 1] + 1.0) * (cfg.n_max - cfg.n_min))
    else:
        n = np.zeros_like(h)
    return h, n


def holdings_to_action(cfg: EnvConfig, h, n=None):
    """Inverse of action_to_holdings (clipped), so baselines can be expressed as policies."""
    q = cfg.liability_qty
    a_h = 2.0 * (h / q - cfg.h_min) / (cfg.h_max - cfg.h_min) - 1.0
    if cfg.phase == 2:
        a_n = 2.0 * (n / q - cfg.n_min) / (cfg.n_max - cfg.n_min) - 1.0
        return np.clip(np.stack([a_h, a_n], axis=-1), -1.0, 1.0)
    return np.clip(a_h[..., None], -1.0, 1.0)


def step_pnl(cfg: EnvConfig, s_t, s_t1, pv_t, pv_t1, t, h, n, h_prev, n_prev):
    """Wealth increment over (t, t+1] for a batch, plus the cost paid. `pv_*` are the market's
    pricing vols. Closes positions at the end."""
    q = cfg.liability_qty
    v_t = liability_value(cfg, s_t, tau_liability(cfg, t), pv_t)
    v_t1 = liability_value(cfg, s_t1, tau_liability(cfg, t + 1), pv_t1)
    dw = -q * (v_t1 - v_t) + h * (s_t1 - s_t)
    cost = cfg.cost * s_t * np.abs(h - h_prev)
    if cfg.phase == 2:
        c_t = hedge_option_value(cfg, s_t, tau_hedge(cfg, t), pv_t)
        c_t1 = hedge_option_value(cfg, s_t1, tau_hedge(cfg, t + 1), pv_t1)
        dw = dw + n * (c_t1 - c_t)
        cost = cost + cfg.hedge_option_spread * c_t * np.abs(n - n_prev)
    if (t + 1) == cfg.liability_days:
        cost = cost + cfg.cost * s_t1 * np.abs(h)
        if cfg.phase == 2:
            cost = cost + cfg.hedge_option_spread * c_t1 * np.abs(n)
    return dw - cost, cost


def reward(cfg: EnvConfig, dw, premium):
    """Kolm-Ritter mean-variance reward on the premium-scaled increment."""
    x = dw / premium
    return x - 0.5 * cfg.kappa * x**2
