"""Path generation. All simulators are pulled from pfhedge.stochastic (Heston uses the Andersen
QE scheme, rough Bergomi the hybrid scheme). We only add seeding, the choice of "marking vol"
the hedger sees, and a NumPy container so every policy consumes identical arrays."""

from dataclasses import dataclass

import numpy as np
import torch
from pfhedge import stochastic as st

from hedgerl.config import EnvConfig


@dataclass
class PathBank:
    spot: np.ndarray       # (n_paths, n_steps + 1)
    mark_vol: np.ndarray   # (n_paths, n_steps + 1): vol the hedger believes (Greeks, features)
    price_vol: np.ndarray  # (n_paths, n_steps + 1): vol the market prices options with
    true_var: np.ndarray | None  # instantaneous variance if the model has one, else None
    model: str

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1


def generate_paths(cfg: EnvConfig, n_paths: int, seed: int, model: str | None = None,
                   mark_vol: float | str | None = None, price_vol: float | str | None = None) -> PathBank:
    """Simulate `n_paths` under `model` (defaults to cfg.model). `mark_vol` / `price_vol` override the
    config so a policy trained on Heston can be evaluated on GBM(30%) while believing sigma = 0.2 and
    while the market still prices options at 30%."""
    model = model or cfg.model
    mark_vol = cfg.mark_vol if mark_vol is None else mark_vol
    price_vol = cfg.price_vol if price_vol is None else price_vol
    n_steps = cfg.n_steps
    torch.manual_seed(seed)
    true_var = None

    if model == "heston":
        p = cfg.heston
        out = st.generate_heston(n_paths, n_steps + 1, init_state=(cfg.s0, p["v0"]), kappa=p["kappa"],
                                 theta=p["theta"], sigma=p["sigma"], rho=p["rho"], dt=cfg.dt, dtype=torch.float64)
        spot, true_var = out.spot.numpy(), out.variance.numpy()
    elif model == "gbm":
        spot = st.generate_geometric_brownian(n_paths, n_steps + 1, init_state=(cfg.s0,), sigma=cfg.gbm_sigma,
                                              dt=cfg.dt, dtype=torch.float64).numpy()
    elif model == "merton":
        p = cfg.merton
        spot = st.generate_merton_jump(n_paths, n_steps + 1, init_state=(cfg.s0,), sigma=p["sigma"],
                                       jump_per_year=p["jump_per_year"], jump_mean=p["jump_mean"],
                                       jump_std=p["jump_std"], dt=cfg.dt, dtype=torch.float64).numpy()
    elif model == "rbergomi":
        p = cfg.rbergomi
        out = st.generate_rough_bergomi(n_paths, n_steps + 1, init_state=(cfg.s0, p["xi"]), alpha=p["alpha"],
                                        rho=p["rho"], eta=p["eta"], xi=p["xi"], dt=cfg.dt, dtype=torch.float64)
        spot, true_var = out.spot.numpy(), out.variance.numpy()
    else:
        raise ValueError(model)

    def _vol(v, name):
        if v == "instantaneous":
            if true_var is None:
                raise ValueError(f"{model} has no instantaneous variance; pass a float {name}")
            return np.sqrt(np.maximum(true_var, 1e-8)).astype(np.float64)
        return np.full_like(spot, float(v), dtype=np.float64)

    return PathBank(spot=spot.astype(np.float64), mark_vol=_vol(mark_vol, "mark_vol"),
                    price_vol=_vol(price_vol, "price_vol"), true_var=true_var, model=model)
