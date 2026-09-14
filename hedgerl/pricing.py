"""Black-Scholes closed forms (r = 0), vectorised over NumPy arrays. Used to mark the liability
and the hedging option inside the environment and by every baseline."""

import numpy as np
from scipy.stats import norm

_EPS = 1e-12


def _d1(s, k, tau, vol):
    tau = np.maximum(tau, _EPS)
    vol = np.maximum(vol, _EPS)
    return (np.log(s / k) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))


def bs_price(s, k, tau, vol, call=True):
    s, k, tau, vol = map(np.asarray, (s, k, tau, vol))
    d1 = _d1(s, k, tau, vol)
    d2 = d1 - vol * np.sqrt(np.maximum(tau, _EPS))
    if call:
        p = s * norm.cdf(d1) - k * norm.cdf(d2)
        intrinsic = np.maximum(s - k, 0.0)
    else:
        p = k * norm.cdf(-d2) - s * norm.cdf(-d1)
        intrinsic = np.maximum(k - s, 0.0)
    return np.where(tau <= _EPS, intrinsic, p)


def bs_delta(s, k, tau, vol, call=True):
    s, k, tau, vol = map(np.asarray, (s, k, tau, vol))
    d1 = _d1(s, k, tau, vol)
    d = norm.cdf(d1) if call else norm.cdf(d1) - 1.0
    expired = (s > k).astype(float) if call else -(s < k).astype(float)
    return np.where(tau <= _EPS, expired, d)


def bs_gamma(s, k, tau, vol):
    s, k, tau, vol = map(np.asarray, (s, k, tau, vol))
    d1 = _d1(s, k, tau, vol)
    g = norm.pdf(d1) / (s * np.maximum(vol, _EPS) * np.sqrt(np.maximum(tau, _EPS)))
    return np.where(tau <= _EPS, 0.0, g)


def bs_vega(s, k, tau, vol):
    s, k, tau, vol = map(np.asarray, (s, k, tau, vol))
    d1 = _d1(s, k, tau, vol)
    return np.where(tau <= _EPS, 0.0, s * norm.pdf(d1) * np.sqrt(np.maximum(tau, _EPS)))


def _bs_fwd(f, k, tau, vol, call=True):
    """Black formula on a forward f with zero discounting."""
    tau = np.maximum(tau, _EPS)
    vol = np.maximum(vol, _EPS)
    d1 = (np.log(f / k) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    if call:
        return f * norm.cdf(d1) - k * norm.cdf(d2)
    return k * norm.cdf(-d2) - f * norm.cdf(-d1)


def merton_price(s, k, tau, sigma, jump_per_year, jump_mean, jump_std, call=True, n_terms=25):
    """Merton (1976) jump-diffusion price with r = 0 and compensated drift (spot is a martingale),
    log-jumps ~ N(jump_mean, jump_std^2), matching pfhedge.stochastic.generate_merton_jump.
    Conditional on n jumps the log-price is normal, so the price is a Poisson mixture of Black
    prices with forward s * exp((-lam*kappa + n*ln(1+kappa)/tau) * tau) and vol sqrt(sigma^2 + n*jump_std^2/tau)."""
    s, k, tau = map(lambda x: np.asarray(x, dtype=float), (s, k, tau))
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam_p = jump_per_year * (1.0 + kappa)
    tau_c = np.maximum(tau, _EPS)
    price = np.zeros(np.broadcast(s, k, tau).shape)
    log_w = -lam_p * tau_c
    for n in range(n_terms):
        w = np.exp(log_w + n * np.log(np.maximum(lam_p * tau_c, _EPS)) - np.sum(np.log(np.arange(1, n + 1))))
        fwd = s * np.exp(-jump_per_year * kappa * tau_c + n * np.log1p(kappa))
        vol_n = np.sqrt(sigma**2 + n * jump_std**2 / tau_c)
        price = price + w * _bs_fwd(fwd, k, tau_c, vol_n, call)
    intrinsic = np.maximum(s - k, 0.0) if call else np.maximum(k - s, 0.0)
    return np.where(tau <= _EPS, intrinsic, price)


def leland_vol(vol, cost, dt):
    """Leland (1985) adjusted volatility for a *short* option hedged at interval dt with
    proportional cost `cost`: sigma_L^2 = sigma^2 (1 + sqrt(2/pi) * cost / (sigma sqrt(dt)))."""
    vol = np.asarray(vol, dtype=float)
    a = np.sqrt(2.0 / np.pi) * cost / (np.maximum(vol, _EPS) * np.sqrt(dt))
    return vol * np.sqrt(1.0 + a)


def ww_halfwidth(s, gamma, cost, risk_aversion=1.0):
    """Whalley-Wilmott (1997) no-transaction band half-width around the BS delta:
    H = (3/2 * cost * S * Gamma^2 / risk_aversion)^(1/3), with r = 0."""
    s, gamma = np.asarray(s), np.asarray(gamma)
    return (1.5 * cost * s * gamma**2 / risk_aversion) ** (1.0 / 3.0)
