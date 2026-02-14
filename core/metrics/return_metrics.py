"""Return-based metrics: Sharpe, Sortino, Calmar, Information Ratio, period returns.

All functions are pure — take Polars DataFrames and scalars, return results.
No side effects, no state.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import kurtosis, norm, skew

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

    DSR = Φ[ (SR_observed − SR_benchmark) / σ(SR) ]

    where:
        SR_benchmark = √(V[SR_hat]) × ((1 − γ)·z_α + γ·z_α^(N))
                     ≈ expected max SR under null (from N independent trials)
        σ(SR) = √[ (1 − γ₃·SR + (γ₄−1)/4·SR²) / T ]
        γ₃ = skewness of returns
        γ₄ = excess kurtosis of returns
        T  = number of observations

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
    gamma4 = float(kurtosis(rets_np, bias=False))  # excess kurtosis

    # Standard error of the Sharpe ratio (Lo, 2002 / Bailey & Lopez de Prado)
    sr_var = (1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr ** 2) / n
    if sr_var <= 0:
        sr_var = 1.0 / n
    sr_std = math.sqrt(sr_var)

    # Expected max SR under the null (from num_trials independent tests)
    # E[max(Z_1..Z_N)] ≈ Φ^{-1}(1 − 1/(N·e)) × √(2·ln(N)) for large N
    # Simplified: for N=1, benchmark SR = 0
    if num_trials <= 1:
        sr_benchmark = 0.0
    else:
        # Euler-Mascheroni approximation for expected max of N standard normals
        euler_mascheroni = 0.5772156649
        z = math.sqrt(2.0 * math.log(num_trials))
        sr_benchmark = sr_std * (z - (math.log(math.pi) + euler_mascheroni) / (2.0 * z))

    # DSR = probability that the true SR > 0 given observed SR and its std error
    if sr_std == 0:
        return 1.0 if sr > sr_benchmark else 0.0
    dsr = float(norm.cdf((sr - sr_benchmark) / sr_std))

    return dsr
