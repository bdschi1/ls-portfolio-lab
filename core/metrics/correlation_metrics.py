"""Correlation metrics: pairwise correlation, portfolio average, crowding indicators.

Monitors diversification quality and identifies when positions are too correlated.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def pairwise_correlation_matrix(
    returns_df: pl.DataFrame,
    tickers: list[str],
) -> np.ndarray:
    """
    Compute pairwise correlation matrix from daily returns.

    returns_df: DataFrame with one column per ticker (daily returns)
    tickers: ordered list of tickers

    Returns (n, n) numpy correlation matrix.
    """
    available = [t for t in tickers if t in returns_df.columns]
    if len(available) < 2:
        return np.array([])

    mat = returns_df.select(available).drop_nulls().to_numpy()
    if mat.shape[0] < 10:
        return np.eye(len(available))

    return np.corrcoef(mat, rowvar=False)


def average_pairwise_correlation(
    returns_df: pl.DataFrame,
    tickers: list[str],
) -> float:
    """
    Average pairwise correlation across all positions.

    Excludes diagonal (self-correlation = 1.0).
    Lower = more diversified.
    """
    corr = pairwise_correlation_matrix(returns_df, tickers)
    if corr.size == 0:
        return 0.0

    n = corr.shape[0]
    if n < 2:
        return 0.0

    # Extract upper triangle (excluding diagonal)
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def long_short_correlation(
    returns_df: pl.DataFrame,
    long_tickers: list[str],
    short_tickers: list[str],
    long_weights: dict[str, float] | None = None,
    short_weights: dict[str, float] | None = None,
) -> float:
    """
    Correlation between the long book and short book.

    Ideally moderately positive — you want your hedge to move with the longs
    (but you're positioned opposite), giving you negative portfolio-level correlation.

    High positive correlation = good hedge (shorts move with longs).
    Low/negative correlation = shorts don't hedge the longs well.

    Uses equal weights if weights not provided.
    """
    available_long = [t for t in long_tickers if t in returns_df.columns]
    available_short = [t for t in short_tickers if t in returns_df.columns]

    if not available_long or not available_short:
        return 0.0

    # Compute book-level returns
    if long_weights:
        lw = np.array([long_weights.get(t, 1.0 / len(available_long)) for t in available_long])
    else:
        lw = np.ones(len(available_long)) / len(available_long)

    if short_weights:
        sw = np.array([short_weights.get(t, 1.0 / len(available_short)) for t in available_short])
    else:
        sw = np.ones(len(available_short)) / len(available_short)

    # Normalize weights
    lw = lw / lw.sum()
    sw = sw / sw.sum()

    long_rets = returns_df.select(available_long).drop_nulls().to_numpy() @ lw
    short_rets = returns_df.select(available_short).drop_nulls().to_numpy() @ sw

    min_len = min(len(long_rets), len(short_rets))
    if min_len < 10:
        return 0.0

    corr = np.corrcoef(long_rets[:min_len], short_rets[:min_len])
    return float(corr[0, 1])


def most_correlated_pairs(
    returns_df: pl.DataFrame,
    tickers: list[str],
    top_n: int = 10,
) -> list[tuple[str, str, float]]:
    """
    Find the most correlated position pairs.

    Returns list of (ticker_a, ticker_b, correlation) sorted by |correlation|.
    """
    available = [t for t in tickers if t in returns_df.columns]
    corr = pairwise_correlation_matrix(returns_df, available)
    if corr.size == 0:
        return []

    n = corr.shape[0]
    pairs: list[tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((available[i], available[j], float(corr[i, j])))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]


def least_correlated_pairs(
    returns_df: pl.DataFrame,
    tickers: list[str],
    top_n: int = 10,
) -> list[tuple[str, str, float]]:
    """
    Find the least correlated (or most negatively correlated) position pairs.

    Useful for finding diversification opportunities.
    """
    available = [t for t in tickers if t in returns_df.columns]
    corr = pairwise_correlation_matrix(returns_df, available)
    if corr.size == 0:
        return []

    n = corr.shape[0]
    pairs: list[tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((available[i], available[j], float(corr[i, j])))

    pairs.sort(key=lambda x: x[2])
    return pairs[:top_n]


def correlation_summary(
    returns_df: pl.DataFrame,
    long_tickers: list[str],
    short_tickers: list[str],
) -> dict[str, float]:
    """
    Summary correlation stats for the portfolio.

    Returns:
        avg_pairwise: average pairwise correlation
        avg_long_corr: average correlation among longs
        avg_short_corr: average correlation among shorts
        long_short_corr: correlation between long and short books
    """
    all_tickers = long_tickers + short_tickers

    return {
        "avg_pairwise": average_pairwise_correlation(returns_df, all_tickers),
        "avg_long_corr": average_pairwise_correlation(returns_df, long_tickers),
        "avg_short_corr": average_pairwise_correlation(returns_df, short_tickers),
        "long_short_corr": long_short_correlation(returns_df, long_tickers, short_tickers),
    }
