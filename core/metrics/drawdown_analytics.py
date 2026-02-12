"""Drawdown analytics — Bailey & Lopez de Prado framework.

Implements the analytical drawdown model from:

    Bailey, D.H. & Lopez de Prado, M. (2014). "The Strategy Approval Decision:
    A Sharpe Ratio Indifference Curve Approach." Algorithmic Finance, 3(1-2),
    99-109. DOI: 10.3233/AF-140035

All formulas assume strategy PnL modeled as arithmetic diffusion:
    dX_t = μdt + σdB_t

where μ is the drift (expected daily return) and σ is the daily volatility.

Key insight: all drawdown properties depend on the Sharpe ratio (SR)
and can be expressed in "vol units" (b_σ = b/σ) for universality.

Equations implemented:
    Eq. 6:  P(D ≥ b)     = exp(−2·b_σ·SR)         — stationary drawdown probability
    Eq. 4-5: P(b, T)     = inverse Gaussian CDF     — finite-horizon DD probability
    Eq. 7:  E[D]/σ       = 1/(2·SR)                 — expected drawdown (vol units)
    Eq. 8:  E[τ]         = b_σ/SR                   — expected recovery time
    Eq. 9:  std(τ)       = √(b_σ/SR)/SR             — recovery time std dev
    Eq. 10: E[time in DD] = 1/(2·SR²)               — fraction of time in drawdown
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _normalize_drawdown(b: float, sigma: float) -> float:
    """Convert drawdown b into vol units: b_σ = b / σ."""
    if sigma <= 0:
        return float("inf")
    return b / sigma


def expected_drawdown_vol_units(sharpe: float) -> float:
    """
    Expected drawdown in vol units.

    E(D_t) / σ = 1 / (2 · SR)

    Args:
        sharpe: annualized Sharpe ratio

    Returns:
        Expected drawdown as a multiple of volatility.
        Returns inf if SR ≤ 0.
    """
    if sharpe <= 0:
        return float("inf")
    return 1.0 / (2.0 * sharpe)


def expected_drawdown_pct(sharpe: float, ann_vol: float) -> float:
    """
    Expected drawdown as a percentage.

    E(D_t) = σ / (2 · SR)

    Args:
        sharpe: annualized Sharpe ratio
        ann_vol: annualized volatility (e.g. 0.10 for 10%)

    Returns:
        Expected drawdown as a fraction (e.g. 0.05 for 5%).
        Returns inf if SR ≤ 0.
    """
    if sharpe <= 0:
        return float("inf")
    return ann_vol / (2.0 * sharpe)


def drawdown_probability(b_sigma: float, sharpe: float) -> float:
    """
    Probability of ever hitting a drawdown of depth b (stationary distribution).

    P(D ≥ b) = exp(-2 · b_σ · SR)

    This is the memoryless (stationary) probability — valid for any time
    in the future given an infinite horizon. Equation 6 from the paper.

    Args:
        b_sigma: drawdown depth in vol units (b / σ)
        sharpe: annualized Sharpe ratio

    Returns:
        Probability in [0, 1]. Returns 1.0 if SR ≤ 0 (certain to hit any drawdown).
    """
    if sharpe <= 0:
        return 1.0
    if b_sigma <= 0:
        return 1.0
    exponent = -2.0 * b_sigma * sharpe
    # Guard against overflow
    if exponent < -700:
        return 0.0
    return math.exp(exponent)


def drawdown_probability_horizon(
    b: float,
    mu: float,
    sigma: float,
    T: float,
) -> float:
    """
    Probability of hitting drawdown b within time horizon T.

    Uses the inverse Gaussian CDF (equations 4-5 from the paper):

    P(b, T) = Φ((-b - μT) / (σ√T)) + exp(-2μb/σ²) · Φ((-b + μT) / (σ√T))

    Args:
        b: drawdown depth (positive value, e.g. 0.10 for 10%)
        mu: drift (daily expected excess return)
        sigma: daily volatility
        T: time horizon (in trading days)

    Returns:
        Probability in [0, 1].
    """
    if b <= 0:
        return 1.0
    if sigma <= 0:
        return 0.0
    if T <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    if sigma_sqrt_T == 0:
        return 0.0

    # First term: Φ((-b - μT) / (σ√T))
    z1 = (-b - mu * T) / sigma_sqrt_T
    term1 = norm.cdf(z1)

    # Second term: exp(-2μb/σ²) · Φ((-b + μT) / (σ√T))
    sigma_sq = sigma * sigma
    if sigma_sq == 0:
        return 0.0

    exponent = -2.0 * mu * b / sigma_sq
    # Guard against overflow
    if exponent > 700:
        return min(term1, 1.0)

    z2 = (-b + mu * T) / sigma_sqrt_T
    term2 = math.exp(exponent) * norm.cdf(z2)

    return min(term1 + term2, 1.0)


def expected_recovery_time(b_sigma: float, sharpe: float) -> float:
    """
    Expected time to recover from a drawdown of depth b.

    E(τ) = b_σ / SR

    Expressed as a fraction of a year (since SR is annualized).
    To get trading days, multiply by 252.

    Args:
        b_sigma: drawdown in vol units (b / σ)
        sharpe: annualized Sharpe ratio

    Returns:
        Expected recovery time in years. Returns inf if SR ≤ 0.
    """
    if sharpe <= 0:
        return float("inf")
    if b_sigma <= 0:
        return 0.0
    return b_sigma / sharpe


def recovery_time_std(b_sigma: float, sharpe: float) -> float:
    """
    Standard deviation of recovery time from drawdown b.

    std(τ) = √(b_σ / SR) · (1 / SR)

    Equation 9 from the paper. Also in years.

    Args:
        b_sigma: drawdown in vol units (b / σ)
        sharpe: annualized Sharpe ratio

    Returns:
        Std of recovery time in years. Returns inf if SR ≤ 0.
    """
    if sharpe <= 0:
        return float("inf")
    if b_sigma <= 0:
        return 0.0
    return math.sqrt(b_sigma / sharpe) * (1.0 / sharpe)


def expected_time_in_drawdown(sharpe: float) -> float:
    """
    Expected fraction of time spent in drawdown.

    E(time in DD) = 1 / (2 · SR²)

    Equation 10 from the paper. Inversely proportional to SR squared.
    A strategy with SR=1 spends ~50% of time in drawdown.
    A strategy with SR=2 spends ~12.5% of time in drawdown.

    Args:
        sharpe: annualized Sharpe ratio

    Returns:
        Fraction of time in drawdown (e.g. 0.50 for 50%). Returns inf if SR ≤ 0.
    """
    if sharpe <= 0:
        return float("inf")
    return 1.0 / (2.0 * sharpe * sharpe)


def time_in_drawdown_pct(sharpe: float) -> float:
    """
    Expected time in drawdown as a percentage.

    Convenience wrapper: expected_time_in_drawdown × 100.

    Args:
        sharpe: annualized Sharpe ratio

    Returns:
        Percentage of time in drawdown (e.g. 50.0 for 50%). Returns inf if SR ≤ 0.
    """
    t = expected_time_in_drawdown(sharpe)
    if math.isinf(t):
        return float("inf")
    return t * 100.0


def drawdown_table(
    sharpe: float,
    ann_vol: float,
    current_drawdown: float = 0.0,
) -> dict[str, float | str]:
    """
    Build a summary dict of drawdown analytics for display.

    Combines all the analytical formulas into a single convenient output.

    Args:
        sharpe: annualized Sharpe ratio
        ann_vol: annualized portfolio volatility
        current_drawdown: current drawdown depth (positive value)

    Returns:
        Dict with keys: expected_dd_pct, time_in_dd_pct, dd_probability_10pct,
        recovery_time_days, recovery_time_std_days
    """
    result: dict[str, float | str] = {}

    # Expected drawdown
    if sharpe > 0:
        result["expected_dd_pct"] = expected_drawdown_pct(sharpe, ann_vol)
        result["time_in_dd_pct"] = time_in_drawdown_pct(sharpe)

        # P(DD ≥ 10%)
        if ann_vol > 0:
            b_sigma_10 = _normalize_drawdown(0.10, ann_vol)
            result["dd_prob_10pct"] = drawdown_probability(b_sigma_10, sharpe)
        else:
            result["dd_prob_10pct"] = 0.0

        # Recovery from current drawdown
        if current_drawdown > 0 and ann_vol > 0:
            b_sigma_curr = _normalize_drawdown(current_drawdown, ann_vol)
            rec_years = expected_recovery_time(b_sigma_curr, sharpe)
            rec_std_years = recovery_time_std(b_sigma_curr, sharpe)
            result["recovery_time_days"] = rec_years * 252
            result["recovery_time_std_days"] = rec_std_years * 252
        else:
            result["recovery_time_days"] = 0.0
            result["recovery_time_std_days"] = 0.0
    else:
        result["expected_dd_pct"] = float("inf")
        result["time_in_dd_pct"] = float("inf")
        result["dd_prob_10pct"] = 1.0
        result["recovery_time_days"] = float("inf")
        result["recovery_time_std_days"] = float("inf")

    return result
