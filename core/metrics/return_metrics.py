"""Return-based metrics: Sharpe, Sortino, Calmar, Information Ratio, period returns.

All functions are pure — take Polars DataFrames and scalars, return results.
No side effects, no state.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import kurtosis, norm, skew

from core.metrics.sharpe_inference import (
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio_variance,
)

TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pl.Series) -> pl.Series:
    """Compute daily log returns from a price series."""
    return prices.log().diff().drop_nulls()


def daily_simple_returns(prices: pl.Series) -> pl.Series:
    """Compute daily simple (arithmetic) returns from a price series."""
    return prices.pct_change().drop_nulls()


def annualized_return(daily_rets: pl.Series) -> float:
    """Annualized return from daily simple returns (geometric)."""
    if daily_rets.len() == 0:
        return 0.0
    total = (1.0 + daily_rets).product()
    n_days = daily_rets.len()
    if n_days == 0:
        return 0.0
    return float(total ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)


def annualized_volatility(daily_rets: pl.Series) -> float:
    """Annualized volatility from daily returns."""
    if daily_rets.len() < 2:
        return 0.0
    return float(daily_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_rets: pl.Series, risk_free_rate: float = 0.05) -> float:
    """
    Annualized Sharpe ratio.

    (annualized_return - risk_free_rate) / annualized_volatility
    """
    ann_ret = annualized_return(daily_rets)
    ann_vol = annualized_volatility(daily_rets)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(daily_rets: pl.Series, risk_free_rate: float = 0.05) -> float:
    """
    Annualized Sortino ratio.

    Uses downside deviation (only negative returns) in denominator.
    """
    ann_ret = annualized_return(daily_rets)
    downside = daily_rets.filter(daily_rets < 0)
    if downside.len() < 2:
        return 0.0
    downside_std = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_std


def calmar_ratio(daily_rets: pl.Series) -> float:
    """
    Calmar ratio = annualized return / max drawdown.

    Uses absolute value of max drawdown in denominator.
    """
    ann_ret = annualized_return(daily_rets)
    cum = (1.0 + daily_rets).cum_prod()
    peak = cum.cum_max()
    dd = ((cum - peak) / peak).min()
    if dd is None or dd == 0:
        return 0.0
    return ann_ret / abs(float(dd))


def information_ratio(
    portfolio_daily_rets: pl.Series,
    benchmark_daily_rets: pl.Series,
) -> float:
    """
    Information ratio = alpha / tracking error.

    Alpha = annualized excess return vs benchmark.
    Tracking error = annualized std of excess returns.
    """
    # Align lengths
    min_len = min(portfolio_daily_rets.len(), benchmark_daily_rets.len())
    if min_len < 2:
        return 0.0

    port = portfolio_daily_rets.tail(min_len)
    bench = benchmark_daily_rets.tail(min_len)
    excess = port - bench

    ann_excess = annualized_return(excess)
    te = float(excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if te == 0:
        return 0.0
    return ann_excess / te


def tracking_error(
    portfolio_daily_rets: pl.Series,
    benchmark_daily_rets: pl.Series,
) -> float:
    """Annualized tracking error (std of excess returns)."""
    min_len = min(portfolio_daily_rets.len(), benchmark_daily_rets.len())
    if min_len < 2:
        return 0.0
    port = portfolio_daily_rets.tail(min_len)
    bench = benchmark_daily_rets.tail(min_len)
    excess = port - bench
    return float(excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def period_return(prices: pl.Series, days: int) -> float:
    """Return over the last N trading days."""
    if prices.len() < days + 1:
        return 0.0
    start_price = float(prices[-(days + 1)])
    end_price = float(prices[-1])
    if start_price == 0:
        return 0.0
    return (end_price - start_price) / start_price


def multi_period_returns(prices: pl.Series) -> dict[str, float]:
    """
    Compute returns over standard periods.

    Returns dict: {"1D": float, "1W": float, "1M": float, "3M": float,
                    "6M": float, "1Y": float, "YTD": float}
    """

    results: dict[str, float] = {}
    n = prices.len()

    periods = {
        "1D": 1,
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    for label, days in periods.items():
        if n > days:
            results[label] = period_return(prices, days)
        else:
            results[label] = 0.0

    return results


def portfolio_daily_returns(
    weights: dict[str, float],
    returns_df: pl.DataFrame,
) -> pl.Series:
    """
    Compute portfolio daily returns from position weights and a returns DataFrame.

    weights: {ticker: signed_weight} (positive=long, negative=short)
    returns_df: DataFrame with columns "date" + one column per ticker containing daily returns

    Returns a polars Series of daily portfolio returns.
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        return pl.Series("portfolio_return", dtype=pl.Float64)

    # Weighted sum of daily returns
    w_array = np.array([weights[t] for t in tickers])
    ret_matrix = returns_df.select(tickers).to_numpy()

    port_rets = ret_matrix @ w_array

    return pl.Series("portfolio_return", port_rets)


def cumulative_return(daily_rets: pl.Series) -> pl.Series:
    """Cumulative return series from daily returns (starts at 1.0)."""
    return (1.0 + daily_rets).cum_prod()


def rolling_sharpe(
    daily_rets: pl.Series,
    window: int = 63,
    risk_free_rate: float = 0.05,
) -> pl.Series:
    """Rolling Sharpe ratio over a window of trading days."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_rets - daily_rf

    rolling_mean = excess.rolling_mean(window_size=window)
    rolling_std = excess.rolling_std(window_size=window)

    # Annualize
    return (rolling_mean * TRADING_DAYS_PER_YEAR) / (rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def deflated_sharpe_ratio(
    daily_rets: pl.Series,
    risk_free_rate: float = 0.05,
    num_trials: int = 1,
) -> float:
    """
    Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014).

    Adjusts the observed Sharpe ratio for non-normality (skewness, kurtosis)
    and multiple testing (number of strategy trials).

    Uses sharpe_ratio_variance() for the full asymptotic SE and
    expected_maximum_sharpe_ratio() for the exact E[max] of K normals
    (replacing the √(2·ln(K)) approximation).

    Args:
        daily_rets: daily portfolio return series
        risk_free_rate: annualized risk-free rate
        num_trials: number of independent strategy variants tested
                    (set to 1 if this is the only strategy evaluated)

    Returns:
        DSR as a probability in [0, 1]. Values > 0.95 suggest the Sharpe ratio
        is statistically significant at the 5% level.
    """
    n = daily_rets.len()
    if n < 30:
        return 0.0

    sr = sharpe_ratio(daily_rets, risk_free_rate)

    rets_np = daily_rets.to_numpy()
    gamma3 = float(skew(rets_np, bias=False))
    gamma4 = float(kurtosis(rets_np, bias=False, fisher=False))  # regular kurtosis

    # Full asymptotic variance with non-normality corrections
    sr_var = sharpe_ratio_variance(sr, n, skew=gamma3, kurtosis=gamma4)
    if sr_var <= 0:
        sr_var = 1.0 / n
    sr_std = math.sqrt(sr_var)

    # Expected max SR under the null (exact for small K)
    sr_benchmark = expected_maximum_sharpe_ratio(num_trials, sr_var, sr0=0.0)

    if sr_std == 0:
        return 1.0 if sr > sr_benchmark else 0.0
    dsr = float(norm.cdf((sr - sr_benchmark) / sr_std))

    return dsr


def sharpe_psr(
    daily_rets: pl.Series,
    risk_free_rate: float = 0.05,
    sr0: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio — P(true SR > SR₀).

    Convenience wrapper that extracts distribution moments from a
    Polars return series and delegates to sharpe_inference.

    Args:
        daily_rets: daily portfolio return series.
        risk_free_rate: annualized risk-free rate.
        sr0: benchmark Sharpe ratio to test against.

    Returns:
        PSR in [0, 1].
    """
    n = daily_rets.len()
    if n < 10:
        return 0.0

    sr = sharpe_ratio(daily_rets, risk_free_rate)
    rets_np = daily_rets.to_numpy()
    gamma3 = float(skew(rets_np, bias=False))
    gamma4 = float(kurtosis(rets_np, bias=False, fisher=False))

    return probabilistic_sharpe_ratio(sr, sr0, t=n, skew=gamma3, kurtosis=gamma4)


def sharpe_confidence_interval(
    daily_rets: pl.Series,
    risk_free_rate: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Sharpe ratio confidence interval with PSR and MinTRL.

    Args:
        daily_rets: daily portfolio return series.
        risk_free_rate: annualized risk-free rate.
        confidence: confidence level (default 0.95).

    Returns:
        Dict with keys: sr, ci_lower, ci_upper, psr, min_track_record.
    """
    n = daily_rets.len()
    if n < 10:
        return {
            "sr": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "psr": 0.0, "min_track_record": float("inf"),
        }

    sr = sharpe_ratio(daily_rets, risk_free_rate)
    rets_np = daily_rets.to_numpy()
    gamma3 = float(skew(rets_np, bias=False))
    gamma4 = float(kurtosis(rets_np, bias=False, fisher=False))

    var = sharpe_ratio_variance(sr, n, skew=gamma3, kurtosis=gamma4)
    se = math.sqrt(max(var, 1e-20))

    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))

    psr = probabilistic_sharpe_ratio(sr, sr0=0.0, t=n, skew=gamma3, kurtosis=gamma4)
    min_trl = minimum_track_record_length(sr, sr0=0.0, skew=gamma3, kurtosis=gamma4)

    return {
        "sr": sr,
        "ci_lower": sr - z * se,
        "ci_upper": sr + z * se,
        "psr": psr,
        "min_track_record": min_trl,
    }
