import numpy as np
import pytest
import torch

from hedgerl import book
from hedgerl.baselines import BSDelta, NoHedge, WhalleyWilmott, DeltaGamma
from hedgerl.config import EnvConfig
from hedgerl.deephedge import DeepHedger
from hedgerl.env import HedgingEnv, rollout
from hedgerl.pricing import bs_delta, bs_gamma, bs_price, merton_price
from hedgerl.sim import generate_paths


@pytest.fixture(params=[1, 2])
def cfg(request):
    return EnvConfig(phase=request.param, cost=0.002)


def test_bs_put_call_parity():
    s, k, tau, vol = 105.0, 100.0, 0.3, 0.25
    assert bs_price(s, k, tau, vol, True) - bs_price(s, k, tau, vol, False) == pytest.approx(s - k, abs=1e-10)
    eps = 1e-4
    fd = (bs_price(s + eps, k, tau, vol) - bs_price(s - eps, k, tau, vol)) / (2 * eps)
    assert bs_delta(s, k, tau, vol) == pytest.approx(fd, abs=1e-6)
    fd2 = (bs_delta(s + eps, k, tau, vol) - bs_delta(s - eps, k, tau, vol)) / (2 * eps)
    assert bs_gamma(s, k, tau, vol) == pytest.approx(fd2, abs=1e-5)


def test_env_matches_vectorised_rollout(cfg):
    """The gym env (one path, step by step) and the vectorised evaluator must agree exactly."""
    bank = generate_paths(cfg, 8, seed=1)
    pol = BSDelta(cfg)
    vec = rollout(cfg, bank, pol)
    env = HedgingEnv(cfg, bank, sequential=True)
    for p in range(bank.n_paths):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = pol(obs[None, :], env.t)[0]
            obs, r, done, _, info = env.step(a)
            total += info["dw"]
        assert total == pytest.approx(vec["pnl"][p], rel=1e-9, abs=1e-9)


def test_pnl_identity_no_hedge(cfg):
    """With no hedging and zero cost, P&L is exactly premium minus payoff."""
    cfg.cost = 0.0
    bank = generate_paths(cfg, 50, seed=2)
    res = rollout(cfg, bank, NoHedge(cfg))
    sT = bank.spot[:, -1]
    payoff = np.maximum(sT - cfg.strike, 0.0)
    if cfg.phase == 2:
        payoff = payoff + np.maximum(cfg.strike - sT, 0.0)
    assert np.allclose(res["pnl"], res["premium"] - cfg.liability_qty * payoff, atol=1e-9)


def test_delta_hedge_reduces_variance_gbm():
    cfg = EnvConfig(phase=1, model="gbm", cost=0.0, mark_vol=0.2, price_vol=0.2)
    bank = generate_paths(cfg, 2000, seed=3)
    unhedged = rollout(cfg, bank, NoHedge(cfg))["pnl"].std()
    hedged = rollout(cfg, bank, BSDelta(cfg))["pnl"].std()
    assert hedged < 0.25 * unhedged


def test_ww_trades_less_than_delta(cfg):
    bank = generate_paths(cfg, 200, seed=4)
    t_delta = np.abs(np.diff(rollout(cfg, bank, BSDelta(cfg))["h"], axis=1)).sum()
    t_ww = np.abs(np.diff(rollout(cfg, bank, WhalleyWilmott(cfg))["h"], axis=1)).sum()
    assert t_ww < t_delta


def test_torch_accounting_matches_numpy(cfg):
    """Deep hedger's differentiable P&L must equal the NumPy accounting for the same policy."""
    bank = generate_paths(cfg, 16, seed=5)
    model = DeepHedger(cfg)
    torch_pnl = model.terminal_pnl(torch.as_tensor(bank.spot), torch.as_tensor(bank.mark_vol)).detach().numpy()
    np_pnl = rollout(cfg, bank, model.as_policy())["pnl"]
    assert np.allclose(torch_pnl, np_pnl, atol=1e-6)


def test_delta_gamma_flattens_gamma():
    # Under GBM marking (constant vol) the only residual of a delta hedge is gamma P&L, so the
    # gamma-neutral book must have smaller daily dispersion. Under Heston this need not hold: the
    # longer-dated hedge option over-hedges vega, which is a finding, not a bug.
    cfg = EnvConfig(phase=2, model="gbm", mark_vol=0.2, price_vol=0.2, cost=0.0, hedge_option_spread=0.0)
    bank = generate_paths(cfg, 500, seed=6)
    dg = rollout(cfg, bank, DeltaGamma(cfg, rebalance_every=1))
    d = rollout(cfg, bank, BSDelta(cfg))
    # gamma-hedged book should have smaller daily P&L dispersion than delta-only over most of the life
    assert np.abs(dg["dw"][:, :40]).mean() < np.abs(d["dw"][:, :40]).mean()


def test_merton_pricer_matches_monte_carlo():
    """Exact Merton series price vs the pfhedge jump simulator it is meant to price."""
    p = dict(sigma=0.15, jump_per_year=2.0, jump_mean=-0.03, jump_std=0.05)
    cfg = EnvConfig(phase=1, model="merton", merton=p, mark_vol=0.2, price_vol=0.2, pricer="merton")
    bank = generate_paths(cfg, 200_000, seed=7)
    sT = bank.spot[:, -1]
    for k in (95.0, 100.0, 105.0):
        mc = np.maximum(sT - k, 0.0)
        assert merton_price(100.0, k, 60 / 250, **p) == pytest.approx(mc.mean(), abs=4 * mc.std() / np.sqrt(len(sT)))
    assert merton_price(100.0, 100.0, 0.24, 0.2, 0.0, 0.0, 0.0) == pytest.approx(bs_price(100.0, 100.0, 0.24, 0.2), abs=1e-9)


def test_misspecified_market_prices_at_true_vol():
    """Under the wrong-vol test, the liability is sold at the market's 30% price while the hedger's
    delta is computed at 20%: premium must be the 30% price."""
    cfg = EnvConfig(phase=1, model="gbm", gbm_sigma=0.30, mark_vol=0.2, price_vol=0.30)
    bank = generate_paths(cfg, 4, seed=8)
    from hedgerl.env import HedgingEnv
    env = HedgingEnv(cfg, bank)
    assert env.premium == pytest.approx(float(bs_price(100.0, 100.0, 60 / 250, 0.30)), abs=1e-9)
    obs, _ = env.reset()
    assert obs[2] == pytest.approx(0.2)
