"""Time-weighted return and performance calculation from snapshots.

Computes portfolio performance metrics from the daily snapshot history.
This is the bridge between raw snapshots and the PM scorecard.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

from core.metrics.pm_performance import ClosedTrade, PMScorecard, build_scorecard
from history.snapshot import DailySnapshot, SnapshotStore
from history.trade_log import TradeLog

logger = logging.getLogger(__name__)


def nav_to_returns(snapshots: list[DailySnapshot]) -> pl.Series:
    """Convert daily NAV snapshots to a daily returns series."""
    if len(snapshots) < 2:
        return pl.Series("daily_return", dtype=pl.Float64)

    navs = [s.nav for s in snapshots]
    returns = [(navs[i] - navs[i - 1]) / navs[i - 1] for i in range(1, len(navs))]
    return pl.Series("daily_return", returns)


def time_weighted_return(snapshots: list[DailySnapshot]) -> float:
    """
    Compute time-weighted return from daily snapshots.

    TWR = ∏(1 + Ri) - 1 across all daily periods.
    """
    daily_rets = nav_to_returns(snapshots)
    if daily_rets.len() == 0:
        return 0.0
    return float((1.0 + daily_rets).product() - 1.0)


def annualized_twr(snapshots: list[DailySnapshot]) -> float:
    """Annualized time-weighted return."""
    twr = time_weighted_return(snapshots)
    n_days = len(snapshots)
    if n_days <= 1:
        return 0.0
    return (1.0 + twr) ** (252.0 / n_days) - 1.0


def realized_volatility(snapshots: list[DailySnapshot]) -> float:
    """Annualized realized volatility from snapshot NAVs."""
    daily_rets = nav_to_returns(snapshots)
    if daily_rets.len() < 2:
        return 0.0
    return float(daily_rets.std() * np.sqrt(252))


def realized_sharpe(snapshots: list[DailySnapshot], risk_free_rate: float = 0.05) -> float:
    """Realized Sharpe ratio from snapshots."""
    ann_ret = annualized_twr(snapshots)
    vol = realized_volatility(snapshots)
    if vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / vol


def compute_turnover(trade_log: TradeLog, avg_nav: float, trading_days: int) -> float:
    """
    Portfolio turnover rate.

    Turnover = total traded notional / (avg_nav × trading_days) × 252
    Annualized.
    """
    records = trade_log.read_all()
    total_traded = sum(r.notional for r in records)
    if avg_nav == 0 or trading_days == 0:
        return 0.0
    return (total_traded / avg_nav) * (252.0 / trading_days)


def build_closed_trades(trade_log: TradeLog) -> list[ClosedTrade]:
    """
    Convert trade log's closed trade dicts into ClosedTrade objects
    for the PM scorecard.
    """
    raw = trade_log.identify_closed_trades()
    closed: list[ClosedTrade] = []

    for r in raw:
        closed.append(
            ClosedTrade(
                ticker=r["ticker"],
                side=r["side"],
                sector=r.get("sector", ""),
                subsector=r.get("subsector", ""),
                entry_date=r["entry_date"],
                exit_date=r["exit_date"],
                entry_price=r["entry_price"],
                exit_price=r["exit_price"],
                shares=r["shares"],
                pnl_dollars=r["pnl_dollars"],
                pnl_pct=r["pnl_pct"],
                holding_days=r["holding_days"],
            )
        )

    return closed


def generate_pm_scorecard(
    snapshot_store: SnapshotStore,
    trade_log: TradeLog,
    market_returns: pl.Series | None = None,
    risk_free_rate: float = 0.05,
) -> PMScorecard:
    """
    Generate a complete PM scorecard from paper portfolio history.

    This is the main entry point for PM performance evaluation.
    """
    snapshots = snapshot_store.read_all()
    if not snapshots:
        return PMScorecard()

    # Time-weighted return
    twr = time_weighted_return(snapshots)
    ann_twr = annualized_twr(snapshots)
    sharpe = realized_sharpe(snapshots, risk_free_rate)

    # Alpha (vs market) — simple approximation
    alpha = 0.0
    if market_returns is not None:
        daily_rets = nav_to_returns(snapshots)
        min_len = min(daily_rets.len(), market_returns.len())
        if min_len > 20:
            port_ann = float(daily_rets.tail(min_len).mean() or 0) * 252
            mkt_ann = float(market_returns.tail(min_len).mean() or 0) * 252
            alpha = port_ann - mkt_ann  # simple excess return as proxy

    # Drawdown analysis from NAV series
    from core.metrics.drawdown_metrics import drawdown_summary

    nav_series = pl.Series("nav", [s.nav for s in snapshots])
    dd_stats = drawdown_summary(nav_series)

    # Closed trades
    closed_trades = build_closed_trades(trade_log)

    # Turnover
    avg_nav = float(np.mean([s.nav for s in snapshots]))
    trading_days = len(snapshots)
    turnover = compute_turnover(trade_log, avg_nav, trading_days)

    return build_scorecard(
        closed_trades=closed_trades,
        total_return_pct=twr * 100,
        annualized_return_pct=ann_twr * 100,
        sharpe=sharpe,
        alpha=alpha * 100,
        max_drawdown=dd_stats["max_drawdown"] * 100,
        avg_drawdown_depth=dd_stats["avg_drawdown_depth"] * 100,
        avg_recovery_days=dd_stats["avg_recovery_days"],
        num_drawdowns=int(dd_stats["num_episodes"]),
        turnover_pct=turnover * 100,
        trading_days=trading_days,
    )
