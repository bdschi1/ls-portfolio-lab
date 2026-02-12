"""Drawdown metrics: max drawdown, current drawdown, recovery analysis.

Tracks peak-to-trough declines and recovery characteristics —
critical for PM performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


@dataclass
class DrawdownEpisode:
    """A single drawdown episode from peak to recovery."""

    peak_date: date | None
    trough_date: date | None
    recovery_date: date | None  # None if not yet recovered
    peak_value: float
    trough_value: float
    depth: float  # negative percentage
    days_to_trough: int
    days_to_recovery: int | None  # None if not recovered
    is_active: bool  # True if currently in this drawdown


def drawdown_series(prices_or_nav: pl.Series) -> pl.Series:
    """
    Compute drawdown series (underwater curve).

    Returns series of drawdown percentages (negative values, 0 = at peak).
    """
    peak = prices_or_nav.cum_max()
    dd = (prices_or_nav - peak) / peak
    return dd


def max_drawdown(prices_or_nav: pl.Series) -> float:
    """
    Maximum drawdown as a negative percentage.

    E.g., -0.083 means the worst peak-to-trough decline was 8.3%.
    """
    dd = drawdown_series(prices_or_nav)
    min_dd = dd.min()
    return float(min_dd) if min_dd is not None else 0.0


def current_drawdown(prices_or_nav: pl.Series) -> float:
    """
    Current drawdown from the most recent peak.

    Returns 0.0 if at all-time high, negative otherwise.
    """
    dd = drawdown_series(prices_or_nav)
    if dd.len() == 0:
        return 0.0
    return float(dd[-1])


def max_drawdown_from_returns(daily_rets: pl.Series) -> float:
    """Max drawdown computed from a daily returns series."""
    cum = (1.0 + daily_rets).cum_prod()
    return max_drawdown(cum)


def drawdown_duration(prices_or_nav: pl.Series) -> int:
    """
    Days in current drawdown (0 if at peak).

    Counts backward from today to the last peak.
    """
    dd = drawdown_series(prices_or_nav)
    if dd.len() == 0 or float(dd[-1]) == 0.0:
        return 0

    # Count from end backward to last zero
    dd_np = dd.to_numpy()
    for i in range(len(dd_np) - 1, -1, -1):
        if dd_np[i] == 0.0:
            return len(dd_np) - 1 - i
    return len(dd_np)


def identify_drawdown_episodes(
    prices_or_nav: pl.Series,
    dates: pl.Series | None = None,
    min_depth: float = -0.02,  # only flag drawdowns deeper than 2%
) -> list[DrawdownEpisode]:
    """
    Identify all drawdown episodes in a price/NAV series.

    A drawdown episode starts when the series drops below its running peak
    and ends when it recovers to a new peak.

    Args:
        prices_or_nav: price or NAV series
        dates: corresponding date series (optional, for date tracking)
        min_depth: minimum depth to count as an episode (e.g., -0.02 = 2%)

    Returns list of DrawdownEpisode objects, newest first.
    """
    if prices_or_nav.len() < 2:
        return []

    vals = prices_or_nav.to_numpy()
    n = len(vals)

    if dates is not None and dates.len() == n:
        date_arr = dates.to_list()
    else:
        date_arr = [None] * n

    episodes: list[DrawdownEpisode] = []
    peak_val = vals[0]
    peak_idx = 0
    in_drawdown = False
    trough_val = vals[0]
    trough_idx = 0

    for i in range(1, n):
        if vals[i] >= peak_val:
            # New peak — close any active drawdown
            if in_drawdown:
                depth = (trough_val - peak_val) / peak_val
                if depth <= min_depth:
                    episodes.append(
                        DrawdownEpisode(
                            peak_date=date_arr[peak_idx],
                            trough_date=date_arr[trough_idx],
                            recovery_date=date_arr[i],
                            peak_value=float(peak_val),
                            trough_value=float(trough_val),
                            depth=float(depth),
                            days_to_trough=trough_idx - peak_idx,
                            days_to_recovery=i - trough_idx,
                            is_active=False,
                        )
                    )
                in_drawdown = False
            peak_val = vals[i]
            peak_idx = i
            trough_val = vals[i]
            trough_idx = i
        else:
            in_drawdown = True
            if vals[i] < trough_val:
                trough_val = vals[i]
                trough_idx = i

    # Handle active (unrecovered) drawdown
    if in_drawdown:
        depth = (trough_val - peak_val) / peak_val
        if depth <= min_depth:
            episodes.append(
                DrawdownEpisode(
                    peak_date=date_arr[peak_idx],
                    trough_date=date_arr[trough_idx],
                    recovery_date=None,
                    peak_value=float(peak_val),
                    trough_value=float(trough_val),
                    depth=float(depth),
                    days_to_trough=trough_idx - peak_idx,
                    days_to_recovery=None,
                    is_active=True,
                )
            )

    return list(reversed(episodes))


def drawdown_summary(prices_or_nav: pl.Series) -> dict[str, float]:
    """
    Summary statistics of drawdown behavior.

    Returns dict with:
    - max_drawdown: worst peak-to-trough (negative)
    - current_drawdown: current distance from peak (negative or 0)
    - current_drawdown_duration: days in current drawdown
    - avg_drawdown_depth: mean depth of all episodes
    - avg_recovery_days: mean recovery time (for recovered episodes)
    - num_episodes: count of drawdown episodes > 2%
    """
    episodes = identify_drawdown_episodes(prices_or_nav)

    recovered = [e for e in episodes if not e.is_active and e.days_to_recovery is not None]
    all_depths = [e.depth for e in episodes]

    return {
        "max_drawdown": max_drawdown(prices_or_nav),
        "current_drawdown": current_drawdown(prices_or_nav),
        "current_drawdown_duration": drawdown_duration(prices_or_nav),
        "avg_drawdown_depth": float(np.mean(all_depths)) if all_depths else 0.0,
        "avg_recovery_days": float(np.mean([e.days_to_recovery for e in recovered])) if recovered else 0.0,
        "num_episodes": len(episodes),
    }
