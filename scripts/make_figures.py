"""Figures for the README: P&L distributions, the learned hedge ratio vs Black-Scholes delta and the
Whalley-Wilmott band, and the CVaR misspecification grid."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, TEST_MEASURES, base_parser, make_cfg, tag  # noqa: E402
from run_eval import load_policies  # noqa: E402

from hedgerl import book  # noqa: E402
from hedgerl.pricing import ww_halfwidth  # noqa: E402

COLORS = {"bs_delta": "#2a78d6", "whalley_wilmott": "#eb6834", "deep_hedge_es5": "#1baf7a", "ppo": "#eda100",
          "sac": "#e87ba4", "leland": "#008300", "delta_gamma_weekly": "#4a3aa7", "no_hedge": "#9a9a9a"}
LABELS = {"bs_delta": "BS delta", "whalley_wilmott": "Whalley-Wilmott", "deep_hedge_es5": "Deep hedge (ES 5%)",
          "ppo": "PPO", "sac": "SAC", "leland": "Leland", "delta_gamma_weekly": "Delta-gamma (weekly)",
          "no_hedge": "Unhedged"}
SEED0 = {"deep_hedge_es5": "deep_hedge_es5_s0", "ppo": "ppo_s0", "sac": "sac_s0"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.axisbelow": True})


def fig_pnl(out: Path, t: str):
    d = np.load(out / "pnl_heston.npz")
    keys = [k for k in ["bs_delta", "whalley_wilmott", "deep_hedge_es5", "ppo", "sac"] if SEED0.get(k, k) in d]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(-2.5, 1.5, 120)
    for k in keys:
        ax.hist(d[SEED0.get(k, k)], bins=bins, histtype="step", lw=2, color=COLORS[k], label=LABELS[k], density=True)
    ax.set_xlabel("terminal P&L (units of initial premium), Heston test paths")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title(f"{t}: P&L distributions on identical paths", loc="left")
    fig.tight_layout(); fig.savefig(out / "pnl_distributions.png", dpi=160); plt.close(fig)


def fig_hedge_ratio(out: Path, cfg, pols):
    s = np.linspace(85, 115, 121)
    vol = np.full_like(s, 0.2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, t in zip(axes, [10, 50]):
        obs0 = book.features(cfg, s, vol, t, np.zeros_like(s), np.zeros_like(s))
        delta = obs0[:, 4].astype(float)
        # previous holding set to the BS delta so the WW band is centred where it would be in practice
        obs = book.features(cfg, s, vol, t, cfg.liability_qty * delta, np.zeros_like(s))
        hw = ww_halfwidth(s, obs[:, 5].astype(float) / s, cfg.cost, 1.0)
        ax.fill_between(s, delta - hw, delta + hw, color=COLORS["whalley_wilmott"], alpha=0.18, lw=0,
                        label="Whalley-Wilmott band")
        ax.plot(s, delta, color=COLORS["bs_delta"], lw=2, label=LABELS["bs_delta"])
        for k in ["deep_hedge_es5", "ppo", "sac"]:
            if SEED0[k] in pols:
                h, _ = book.action_to_holdings(cfg, pols[SEED0[k]](obs, t))
                ax.plot(s, h / cfg.liability_qty, color=COLORS[k], lw=2, label=LABELS[k])
        ax.set_title(f"day {t} of {cfg.liability_days}, prev holding = BS delta", loc="left")
        ax.set_xlabel("spot (strike = 100)")
    axes[0].set_ylabel("hedge ratio (units of underlying per short call)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "hedge_ratio.png", dpi=160); plt.close(fig)


def fig_misspec(out: Path, t: str, phase: int):
    rows, errs = [], []
    for m in TEST_MEASURES:
        if phase == 2 and m == "rbergomi":
            continue  # no market-consistent option price under rough vol; see README
        f = out / f"seeds_{m}.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            rows.append(df["cvar5_seedmean"].rename(m)); errs.append(df["cvar5_seedstd"].rename(m))
    if not rows:
        return
    tab, err = pd.concat(rows, axis=1), pd.concat(errs, axis=1)
    keys = [k for k in ["bs_delta", "leland", "whalley_wilmott", "deep_hedge_es5", "ppo", "sac", "delta_gamma_weekly"] if k in tab.index]
    tab, err = tab.loc[keys], err.loc[keys]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tab.columns)); w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(x + i * w - 0.4 + w / 2, tab.loc[k], width=w * 0.92, color=COLORS[k], label=LABELS[k],
               yerr=err.loc[k], ecolor="#333333", capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels(tab.columns)
    ax.set_ylabel("5% CVaR of P&L (premium units, lower is better)")
    ax.set_title(f"{t}: trained on Heston, tested across measures (bars: mean over seeds, whiskers: seed std)", loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "misspecification_cvar5.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    a = base_parser("Make figures").parse_args()
    cfg = make_cfg(a.phase, a.cost, a.kappa)
    t = tag(a.phase, a.cost, a.kappa)
    out = RESULTS / t
    pols = load_policies(cfg, t)
    fig_pnl(out, t)
    if cfg.phase == 1:
        fig_hedge_ratio(out, cfg, pols)
    fig_misspec(out, t, cfg.phase)
    print("figures written to", out)
