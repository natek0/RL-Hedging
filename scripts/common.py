"""Shared CLI helpers: one place that defines the experiment configs and the seeds."""

import argparse
from pathlib import Path

from hedgerl.config import EnvConfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TRAIN_SEED, TEST_SEED = 1000, 2000
N_TRAIN_PATHS, N_TEST_PATHS = 100_000, 20_000

# Test measures for the misspecification grid. `mark_vol` is what the hedger believes (Greeks);
# `price_vol` / `pricer` is how the market values options. For GBM the market prices at the true 30%
# while the hedger marks Greeks at 20%; for Merton the market uses the exact jump-diffusion price.
# For rough Bergomi there is no closed-form price, so options are valued with BS at instantaneous
# vol: that column is a caveat for Phase 2 (see README) and is exact for Phase 1 only at expiry.
TEST_MEASURES = {
    "heston": dict(model="heston", mark_vol="instantaneous", price_vol="instantaneous"),
    "rbergomi": dict(model="rbergomi", mark_vol="instantaneous", price_vol="instantaneous"),
    "merton_jumps": dict(model="merton", mark_vol=0.2, price_vol=0.2, pricer="merton"),
    "gbm_vol30_marked20": dict(model="gbm", mark_vol=0.2, price_vol=0.30, gbm_sigma=0.30),
}


def make_cfg(phase: int, cost: float, kappa: float = 1.0) -> EnvConfig:
    return EnvConfig(phase=phase, cost=cost, kappa=kappa)


def tag(phase: int, cost: float, kappa: float = 1.0) -> str:
    return f"p{phase}_c{int(round(cost * 1e4))}bps_k{kappa:g}"


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--cost", type=float, default=0.002)
    p.add_argument("--kappa", type=float, default=1.0)
    return p
