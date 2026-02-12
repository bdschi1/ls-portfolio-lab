"""Risk metrics: volatility, beta, idiosyncratic vol, VaR, CVaR, marginal contribution.

All functions are pure. Calculations use numpy for linear algebra where needed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


# --- Position-level risk metrics ---


def position_beta(
    position_returns: pl.Series,
    market_returns: pl.Series,
) -> float:
    """
    OLS beta of a position vs market.

    β = cov(Ri, Rm) / var(Rm)
    """
    min_len = min(position_returns.len(), market_returns.len())
    if min_len < 20:
        return 1.0  # default if insufficient data

    ri = position_returns.tail(min_len).to_numpy()
    rm = market_returns.tail(min_len).to_numpy()

    cov = np.cov(ri, rm, ddof=1)
    if cov[1, 1] == 0:
        return 1.0
    return float(cov[0, 1] / cov[1, 1])


def position_alpha(
    position_returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Jensen's alpha (annualized) from CAPM regression.

    α = E[Ri] - Rf - β(E[Rm] - Rf)
    """
    min_len = min(position_returns.len(), market_returns.len())
    if min_len < 20:
        return 0.0

    beta = position_beta(position_returns, market_returns)
    ann_ri = float(position_returns.tail(min_len).mean() or 0) * TRADING_DAYS_PER_YEAR
    ann_rm = float(market_returns.tail(min_len).mean() or 0) * TRADING_DAYS_PER_YEAR

    return ann_ri - risk_free_rate - beta * (ann_rm - risk_free_rate)


def idiosyncratic_volatility(
    position_returns: pl.Series,
    market_returns: pl.Series,
) -> float:
    """
    Annualized idiosyncratic (residual) volatility from CAPM regression.

    σ_idio = std(Ri - β × Rm) × √252
    """
    min_len = min(position_returns.len(), market_returns.len())
    if min_len < 20:
        return 0.0

    ri = position_returns.tail(min_len).to_numpy()
    rm = market_returns.tail(min_len).to_numpy()

    beta = position_beta(position_returns, market_returns)
    residuals = ri - beta * rm

    return float(np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_beta(
    position_returns: pl.Series,
    market_returns: pl.Series,
    window: int = 63,
) -> pl.Series:
    """Rolling beta over a window of trading days."""
    min_len = min(position_returns.len(), market_returns.len())
    ri = position_returns.tail(min_len).to_numpy()
    rm = market_returns.tail(min_len).to_numpy()

    betas = np.full(min_len, np.nan)
    for i in range(window, min_len):
        ri_w = ri[i - window : i]
        rm_w = rm[i - window : i]
        cov = np.cov(ri_w, rm_w, ddof=1)
        if cov[1, 1] != 0:
            betas[i] = cov[0, 1] / cov[1, 1]

    return pl.Series("rolling_beta", betas)


def days_to_liquidate(shares: float, avg_daily_volume: float, pct_of_volume: float = 0.20) -> float:
    """
    Estimated days to liquidate a position.

    Assumes you can trade at most pct_of_volume (default 20%) of ADV per day.
    """
    if avg_daily_volume <= 0 or pct_of_volume <= 0:
        return float("inf")
    daily_capacity = avg_daily_volume * pct_of_volume
    return shares / daily_capacity


# --- Portfolio-level risk metrics ---


def covariance_matrix(returns_df: pl.DataFrame, tickers: list[str]) -> np.ndarray:
    """
    Compute the sample covariance matrix from daily returns.

    returns_df: DataFrame with one column per ticker (daily returns)
    tickers: ordered list of tickers (defines column/row order of matrix)

    Returns: (n, n) numpy array
    """
    available = [t for t in tickers if t in returns_df.columns]
    if not available:
        return np.array([])

    mat = returns_df.select(available).to_numpy()
    return np.cov(mat, rowvar=False, ddof=1)


def portfolio_volatility(
    weights: dict[str, float],
    returns_df: pl.DataFrame,
) -> float:
    """
    Annualized portfolio volatility: √(w'Σw) × √252

    weights: {ticker: signed_weight} (positive=long, negative=short)
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if len(tickers) == 0:
        return 0.0

    if len(tickers) == 1:
        # Single position: vol = |weight| × position_vol
        ticker = tickers[0]
        col = returns_df[ticker]
        if col.len() < 2:
            return 0.0
        pos_vol = float(col.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        return abs(weights[ticker]) * pos_vol

    w = np.array([weights[t] for t in tickers])
    cov = covariance_matrix(returns_df, tickers)

    if cov.size == 0:
        return 0.0

    port_var = w @ cov @ w
    if port_var < 0:
        # Can happen with numerical issues
        return 0.0
    return float(np.sqrt(port_var) * np.sqrt(TRADING_DAYS_PER_YEAR))


def portfolio_beta(
    weights: dict[str, float],
    betas: dict[str, float],
) -> float:
    """
    Portfolio beta = Σ(wi × βi)

    weights are signed (+long, -short). This is the NET beta.
    """
    total = 0.0
    for ticker, weight in weights.items():
        beta = betas.get(ticker, 1.0)
        total += weight * beta
    return total


def marginal_contribution_to_risk(
    weights: dict[str, float],
    returns_df: pl.DataFrame,
) -> dict[str, float]:
    """
    Marginal contribution of each position to portfolio volatility.

    MCR_i = w_i × (Σw)_i / σ_p

    Returns: {ticker: marginal_contribution}
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if len(tickers) < 2:
        return {}

    w = np.array([weights[t] for t in tickers])
    cov = covariance_matrix(returns_df, tickers)
    if cov.size == 0:
        return {}

    port_var = w @ cov @ w
    if port_var <= 0:
        return {}
    port_vol = np.sqrt(port_var)

    # Σw gives the marginal risk for each position
    sigma_w = cov @ w
    mcr = w * sigma_w / port_vol

    return {tickers[i]: float(mcr[i]) for i in range(len(tickers))}


def var_historical(daily_rets: pl.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.

    Returns the loss at the given confidence level (positive number).
    E.g., VaR(95%) = 0.018 means 95% of days, daily loss is < 1.8%.
    """
    if daily_rets.len() < 10:
        return 0.0
    quantile = 1.0 - confidence
    var = float(daily_rets.quantile(quantile, interpolation="linear"))
    return abs(var)


def var_parametric(daily_rets: pl.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) Value at Risk.

    VaR = μ - z × σ
    """
    if daily_rets.len() < 10:
        return 0.0
    mu = float(daily_rets.mean() or 0)
    sigma = float(daily_rets.std() or 0)
    z = stats.norm.ppf(1.0 - confidence)
    return abs(mu + z * sigma)


def cvar_historical(daily_rets: pl.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).

    Average of returns below the VaR threshold.
    """
    if daily_rets.len() < 10:
        return 0.0
    quantile = 1.0 - confidence
    threshold = float(daily_rets.quantile(quantile, interpolation="linear"))
    tail = daily_rets.filter(daily_rets <= threshold)
    if tail.len() == 0:
        return 0.0
    return abs(float(tail.mean() or 0))


def rolling_volatility(
    daily_rets: pl.Series,
    window: int = 63,
) -> pl.Series:
    """Rolling annualized volatility."""
    return daily_rets.rolling_std(window_size=window) * np.sqrt(TRADING_DAYS_PER_YEAR)
