"""Evaluate every available policy on paired test paths across the misspecification grid.
Writes results/<tag>/metrics_<measure>.csv, a combined markdown table, and raw P&L arrays."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, N_TEST_PATHS, RESULTS, TEST_MEASURES, TEST_SEED, base_parser, make_cfg, tag  # noqa: E402

from stable_baselines3 import PPO, SAC  # noqa: E402

from hedgerl.baselines import BSDelta, DeltaGamma, LelandDelta, NoHedge, WhalleyWilmott  # noqa: E402
from hedgerl.config import EnvConfig  # noqa: E402
from hedgerl.deephedge import DeepHedger  # noqa: E402
from hedgerl.evaluate import aggregate_seeds, evaluate_policies  # noqa: E402
from hedgerl.sim import generate_paths  # noqa: E402


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs, t):
        a, _ = self.model.predict(obs, deterministic=True)
        return a


def load_policies(cfg: EnvConfig, t: str) -> dict:
    pols = {"no_hedge": NoHedge(cfg), "bs_delta": BSDelta(cfg), "leland": LelandDelta(cfg),
            "whalley_wilmott": WhalleyWilmott(cfg, risk_aversion=1.0)}
    if cfg.phase == 2:
        pols["delta_gamma_weekly"] = DeltaGamma(cfg, rebalance_every=5)
    for f in sorted(MODELS.glob(f"deephedge_{t}_s*.pt")):
        m = DeepHedger(cfg); m.load_state_dict(torch.load(f)); m.eval()
        pols[f"deep_hedge_es5_{f.stem.split('_')[-1]}"] = m.as_policy()
    for algo, cls in (("ppo", PPO), ("sac", SAC)):
        for f in sorted(MODELS.glob(f"{algo}_{t}_s*.zip")):
            pols[f"{algo}_{f.stem.split('_')[-1]}"] = SB3Policy(cls.load(f, device="cpu"))
    return pols


if __name__ == "__main__":
    p = base_parser("Evaluate policies")
    p.add_argument("--n-paths", type=int, default=N_TEST_PATHS)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--measures", nargs="*", default=list(TEST_MEASURES))
    a = p.parse_args()
    cfg = make_cfg(a.phase, a.cost, a.kappa)
    t = tag(a.phase, a.cost, a.kappa)
    out = RESULTS / t
    out.mkdir(parents=True, exist_ok=True)
    pols = load_policies(cfg, t)
    print("policies:", list(pols))
    tables = {}
    for name in a.measures:
        spec = TEST_MEASURES[name]
        test_cfg = replace(cfg, gbm_sigma=spec.get("gbm_sigma", cfg.gbm_sigma), pricer=spec.get("pricer", "bs"))
        bank = generate_paths(test_cfg, a.n_paths, seed=TEST_SEED, model=spec["model"], mark_vol=spec["mark_vol"],
                              price_vol=spec["price_vol"])
        df, raw = evaluate_policies(test_cfg, bank, pols, reference="whalley_wilmott", n_boot=a.n_boot)
        df.to_csv(out / f"metrics_{name}.csv")
        agg = aggregate_seeds(df)
        agg.to_csv(out / f"seeds_{name}.csv")
        np.savez_compressed(out / f"pnl_{name}.npz", **{k: v["pnl"] / v["premium"] for k, v in raw.items()})
        tables[name] = (df, agg)
        print(f"\n== {t} | test measure: {name} | {a.n_paths} paired paths | P&L in units of premium ==")
        print(df[["mean", "std", "cvar5", "turnover", "cost_paid"]].round(3).to_string())
        print("-- across seeds (mean +- std over training seeds; n_seeds = 1 for classical baselines) --")
        show = agg[["cvar5_seedmean", "cvar5_seedstd", "mean_seedmean", "mean_seedstd", "n_seeds"]].copy()
        if "seeds_beating_ref_cvar5" in agg:
            show["seeds_beating_WW"] = agg["seeds_beating_ref_cvar5"]
        print(show.round(3).to_string())
    with open(out / "summary.md", "w") as fh:
        for name, (df, agg) in tables.items():
            fh.write(f"\n### {t}, test measure `{name}`\n\n")
            if cfg.phase == 2 and name == "rbergomi":
                fh.write("WARNING: no closed-form option price exists under rough Bergomi, so the hedge option is "
                         "valued with Black-Scholes at instantaneous vol, which is not the market's price. Option-leg "
                         "P&L in this table is partly a marking artefact. Not reported in the README headline table.\n\n")
            cols = ["mean", "std", "cvar5", "cvar1", "turnover", "cost_paid", "d_cvar5_vs_ref", "d_cvar5_lo", "d_cvar5_hi"]
            fh.write(df[cols].round(3).to_markdown() + "\n")
            fh.write("\nAcross training seeds (paired-bootstrap column counts seeds whose 95% CI for the CVaR "
                     "difference vs Whalley-Wilmott lies entirely below zero):\n\n")
            acols = [c for c in ["cvar5_seedmean", "cvar5_seedstd", "mean_seedmean", "mean_seedstd", "turnover_seedmean",
                                 "n_seeds", "seeds_beating_ref_cvar5"] if c in agg.columns]
            fh.write(agg[acols].round(3).to_markdown() + "\n")
    json.dump({"phase": a.phase, "cost": a.cost, "kappa": a.kappa, "n_paths": a.n_paths}, open(out / "config.json", "w"))
    print("wrote", out)
