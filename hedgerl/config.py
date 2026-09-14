from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig:
    """Everything that defines one hedging problem. Shared by the gym env, the vectorised
    evaluator, the baselines and the deep hedger, so all policies face identical accounting."""

    # Liability: short `liability_qty` units of each leg. Phase 1 = short ATM call.
    # Phase 2 = short ATM straddle hedged with the underlying and one longer-dated ATM call.
    phase: Literal[1, 2] = 1
    s0: float = 100.0
    strike: float = 100.0
    liability_days: int = 60
    liability_qty: float = 1.0
    dt: float = 1.0 / 250.0

    # Hedging option (phase 2 only): ATM call struck at s0 with this tenor, held and marked daily.
    hedge_option_days: int = 90
    hedge_option_spread: float = 0.01  # proportional bid-ask on the option premium

    # Proportional transaction cost on the underlying, as a fraction of notional traded.
    cost: float = 0.002  # 20 bps

    # Position bounds in units of liability_qty.
    h_min: float = -0.5
    h_max: float = 1.5
    n_min: float = -3.0
    n_max: float = 3.0

    # Kolm-Ritter per-step reward r = dw - kappa/2 * dw^2, with dw scaled by the initial premium.
    kappa: float = 1.0

    # Market model used for the *training* measure. Test measures are chosen at eval time.
    model: Literal["heston", "gbm", "merton", "rbergomi"] = "heston"
    heston: dict = field(default_factory=lambda: dict(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04))
    gbm_sigma: float = 0.2
    merton: dict = field(default_factory=lambda: dict(sigma=0.15, jump_per_year=2.0, jump_mean=-0.03, jump_std=0.05))
    rbergomi: dict = field(default_factory=lambda: dict(alpha=-0.4, rho=-0.9, eta=1.9, xi=0.04))

    # Volatility the hedger *believes* when marking. "instantaneous" uses the simulator's variance
    # (available for Heston and rBergomi); a float pins it, which is how misspecification is tested.
    mark_vol: float | Literal["instantaneous"] = "instantaneous"

    # Volatility the *market* uses to price the liability and the hedge option. Under the training
    # measure it equals mark_vol. A misspecification test sets the true vol here (e.g. 0.30) while
    # the hedger keeps marking Greeks at 0.20, so no policy can profit from a mispriced hedge option.
    price_vol: float | Literal["instantaneous"] = "instantaneous"
    # Pricing model for option values: "bs" (Black-Scholes at price_vol) or "merton" (exact
    # jump-diffusion price with cfg.merton parameters; price_vol is then ignored).
    pricer: Literal["bs", "merton"] = "bs"

    def __post_init__(self):
        # Phase 2 needs room for the underlying leg that offsets the hedge option's delta
        # (n * delta_h can reach ~2 units), so widen the bounds unless the user set them.
        if self.phase == 2 and (self.h_min, self.h_max, self.n_min, self.n_max) == (-0.5, 1.5, -3.0, 3.0):
            self.h_min, self.h_max, self.n_min, self.n_max = -3.0, 2.0, -4.0, 4.0

    @property
    def n_steps(self) -> int:
        return self.liability_days
