"""Paired-path evaluation: every policy is run on the identical PathBank, metrics are computed on
terminal P&L, and differences against a reference policy get paired-bootstrap confidence intervals."""

import numpy as np
import pandas as pd

from hedgerl.config import EnvConfig
from hedgerl.env import rollout
from hedgerl.sim import PathBank


def cvar(pnl, alpha=0.05):
    """Expected shortfall of the loss at level alpha, reported as a positive number (bigger = worse)."""
    k = max(1, int(np.ceil(alpha * len(pnl))))
    worst = np.sort(pnl)[:k]
    return float(-worst.mean())


def metrics(res, premium):
    pnl = res["pnl"] / premium  # in units of initial premium
    turnover = np.abs(np.diff(res["h"], axis=1, prepend=0.0)).sum(1).mean()
    return dict(mean=float(pnl.mean()), std=float(pnl.std()), cvar5=cvar(pnl, 0.05), cvar1=cvar(pnl, 0.01),
                mean_abs_daily=float(np.abs(res["dw"] / premium).mean()),
                turnover=float(turnover), cost_paid=float((res["cost"].sum(1) / premium).mean()))


def paired_bootstrap(pnl_a, pnl_b, stat, n_boot=1000, seed=0):
    """CI for stat(a) - stat(b) resampling *paths* jointly (same path indices for both policies)."""
    rng = np.random.default_rng(seed)
    n = len(pnl_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = stat(pnl_a[idx]) - stat(pnl_b[idx])
    return float(stat(pnl_a) - stat(pnl_b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_policies(cfg: EnvConfig, bank: PathBank, policies: dict, reference: str = "whalley_wilmott",
                      n_boot: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Run all policies on `bank`; return a metrics table (with bootstrap CIs vs `reference` for
    mean and CVaR5) and the raw results."""
    results = {name: rollout(cfg, bank, pol) for name, pol in policies.items()}
    premium = results[next(iter(results))]["premium"]
    rows = {}
    ref = results[reference]["pnl"] / premium if reference in results else None
    for name, res in results.items():
        m = metrics(res, premium)
        if ref is not None and name != reference:
            pnl = res["pnl"] / premium
            d, lo, hi = paired_bootstrap(pnl, ref, lambda x: x.mean(), n_boot)
            m["d_mean_vs_ref"], m["d_mean_lo"], m["d_mean_hi"] = d, lo, hi
            d, lo, hi = paired_bootstrap(pnl, ref, lambda x: cvar(x, 0.05), n_boot)
            m["d_cvar5_vs_ref"], m["d_cvar5_lo"], m["d_cvar5_hi"] = d, lo, hi
        rows[name] = m
    return pd.DataFrame(rows).T, results


def family(name: str) -> str:
    """'ppo_s2' -> 'ppo', 'deep_hedge_es5_s0' -> 'deep_hedge_es5', 'whalley_wilmott' -> itself."""
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1][0] == "s" and parts[1][1:].isdigit() else name


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows into one row per policy family with mean and std across seeds.
    Classical baselines have one row and std 0. `n_seeds` says how many runs the row is built from."""
    g = df.groupby(df.index.map(family))
    cols = ["mean", "std", "cvar5", "cvar1", "turnover", "cost_paid", "d_cvar5_vs_ref"]
    cols = [c for c in cols if c in df.columns]
    out = g[cols].mean().add_suffix("_seedmean").join(g[cols].std(ddof=0).add_suffix("_seedstd"))
    out["n_seeds"] = g.size()
    if "d_cvar5_vs_ref" in df.columns:
        out["seeds_beating_ref_cvar5"] = g["d_cvar5_hi"].apply(lambda x: int((x < 0).sum()))
    return out
