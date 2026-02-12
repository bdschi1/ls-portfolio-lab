"""Technical / signal metrics: RSI, momentum, SMA proximity.

Position-level technical indicators for situational awareness.
Not for alpha generation — for risk monitoring (e.g., is a short getting
overbought? Is a long in freefall?).
"""

from __future__ import annotations

import polars as pl


def rsi(prices: pl.Series, period: int = 14) -> pl.Series:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    RSI = 100 - (100 / (1 + RS))
    RS = avg_gain / avg_loss over the period

    Uses EWM with span = (2 * period - 1) to approximate Wilder's smoothing
    where alpha = 1/period.

    Returns a Series of RSI values (first `period` values will be null/NaN).
    """
    delta = prices.diff()

    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)

    # Wilder's smoothing: EMA with alpha = 1/period
    # polars ewm_mean uses span parameter where alpha = 2/(span+1)
    # To get alpha = 1/period, span = 2*period - 1
    span = 2 * period - 1
    avg_gain = gain.ewm_mean(span=span, adjust=False)
    avg_loss = loss.ewm_mean(span=span, adjust=False)

    # Avoid division by zero
    rs = avg_gain / avg_loss.map_elements(lambda x: x if x != 0 else 1e-10, return_dtype=pl.Float64)

    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values.alias("rsi")


def rsi_current(prices: pl.Series, period: int = 14) -> float:
    """Get the most recent RSI value."""
    r = rsi(prices, period)
    if r.len() == 0:
        return 50.0  # neutral default
    val = r[-1]
    return float(val) if val is not None else 50.0


def rsi_classification(rsi_value: float) -> str:
    """Classify RSI into zones."""
    if rsi_value >= 70:
        return "OVERBOUGHT"
    elif rsi_value >= 60:
        return "BULLISH"
    elif rsi_value >= 40:
        return "NEUTRAL"
    elif rsi_value >= 30:
        return "BEARISH"
    else:
        return "OVERSOLD"


def sma(prices: pl.Series, period: int) -> pl.Series:
    """Simple moving average."""
    return prices.rolling_mean(window_size=period).alias(f"sma_{period}")


def price_vs_sma(prices: pl.Series, period: int = 50) -> float:
    """
    Current price relative to SMA as percentage.

    Positive = above SMA, negative = below.
    E.g., +5.0 means price is 5% above the 50-day SMA.
    """
    if prices.len() < period:
        return 0.0
    current = float(prices[-1])
    sma_val = float(prices.tail(period).mean() or 0)
    if sma_val == 0:
        return 0.0
    return (current - sma_val) / sma_val * 100.0


def high_low_52w(prices: pl.Series) -> dict[str, float]:
    """
    52-week (252 trading day) high/low analysis.

    Returns:
        high_52w: 52-week high price
        low_52w: 52-week low price
        pct_from_high: % distance from 52w high (negative)
        pct_from_low: % distance from 52w low (positive)
    """
    lookback = min(prices.len(), 252)
    window = prices.tail(lookback)

    high = float(window.max() or 0)
    low = float(window.min() or 0)
    current = float(prices[-1]) if prices.len() > 0 else 0

    pct_from_high = ((current - high) / high * 100.0) if high > 0 else 0.0
    pct_from_low = ((current - low) / low * 100.0) if low > 0 else 0.0

    return {
        "high_52w": high,
        "low_52w": low,
        "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low,
    }


def momentum(prices: pl.Series, days: int) -> float:
    """
    Price momentum over N trading days.

    Simple total return over the period.
    """
    if prices.len() < days + 1:
        return 0.0
    start = float(prices[-(days + 1)])
    end = float(prices[-1])
    if start == 0:
        return 0.0
    return (end - start) / start


def multi_period_momentum(prices: pl.Series) -> dict[str, float]:
    """Momentum over standard periods: 1M, 3M, 6M, 12M."""
    return {
        "momentum_1m": momentum(prices, 21),
        "momentum_3m": momentum(prices, 63),
        "momentum_6m": momentum(prices, 126),
        "momentum_12m": momentum(prices, 252),
    }


def position_technical_summary(prices: pl.Series, rsi_period: int = 14) -> dict[str, float | str]:
    """
    Full technical summary for a single position.

    Returns dict with RSI, SMA distances, 52w high/low, momentum.
    """
    result: dict[str, float | str] = {}

    # RSI
    rsi_val = rsi_current(prices, rsi_period)
    result["rsi"] = rsi_val
    result["rsi_zone"] = rsi_classification(rsi_val)

    # SMA
    result["vs_sma_20"] = price_vs_sma(prices, 20)
    result["vs_sma_50"] = price_vs_sma(prices, 50)
    result["vs_sma_200"] = price_vs_sma(prices, 200)

    # 52-week
    hl = high_low_52w(prices)
    result.update(hl)

    # Momentum
    mom = multi_period_momentum(prices)
    result.update(mom)

    return result
