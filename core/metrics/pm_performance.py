"""PM performance metrics: hit rate, slugging %, sector skill, drawdown response.

The PM scorecard — evaluates the quality of portfolio management decisions
over the paper trading period. Slugging percentage is weighted as the
most important metric (more so than hit rate).

A PM with 45% hit rate but 3.0x slugging is far superior to
one with 60% hit rate and 1.0x slugging. The math:
  - PM A: 0.45 × 3.0W - 0.55 × W = +0.80W expected per trade
  - PM B: 0.60 × 1.0W - 0.40 × W = +0.20W expected per trade
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np


@dataclass
class ClosedTrade:
    """A completed (closed) trade for PM analytics."""

    ticker: str
    side: str  # LONG or SHORT
    sector: str
    subsector: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: float
    pnl_dollars: float
    pnl_pct: float
    holding_days: int
    market_cap_bucket: str = ""  # "large", "mid", "small"

    @property
    def is_winner(self) -> bool:
        return self.pnl_dollars > 0


@dataclass
class PMScorecard:
    """Comprehensive PM performance evaluation."""

    # Headline numbers
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe: float = 0.0
    alpha: float = 0.0

    # Trading stats — THE CORE
    hit_rate: float = 0.0  # % of winners
    slugging_pct: float = 0.0  # avg win / avg loss — THE KEY METRIC
    expected_value_per_trade: float = 0.0  # hit_rate × avg_win - miss_rate × avg_loss
    win_loss_ratio: float = 0.0  # total wins $ / total losses $
    total_trades: int = 0
    total_winners: int = 0
    total_losers: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_win_dollars: float = 0.0
    avg_loss_dollars: float = 0.0
    best_trade_pnl: float = 0.0
    best_trade_ticker: str = ""
    worst_trade_pnl: float = 0.0
    worst_trade_ticker: str = ""
    avg_holding_period_days: float = 0.0

    # Side breakdown
    long_hit_rate: float = 0.0
    long_slugging: float = 0.0
    short_hit_rate: float = 0.0
    short_slugging: float = 0.0

    # Sector skill
    sector_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Drawdown behavior
    max_drawdown: float = 0.0
    avg_drawdown_depth: float = 0.0
    avg_recovery_days: float = 0.0
    num_drawdowns: int = 0

    # Activity
    turnover_pct: float = 0.0
    trades_per_month: float = 0.0


def compute_hit_rate(trades: list[ClosedTrade]) -> float:
    """Percentage of winning trades."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.is_winner)
    return winners / len(trades)


def compute_slugging_pct(trades: list[ClosedTrade]) -> float:
    """
    Slugging percentage = avg_win_magnitude / avg_loss_magnitude.

    This is the single most important PM metric. It measures
    how much you make when you're right vs how much you lose when wrong.

    Interpretation:
    - < 1.0: losers are bigger than winners (bad)
    - 1.0-1.5: roughly even (mediocre)
    - 1.5-2.5: winners meaningfully larger than losers (good)
    - > 2.5: excellent risk management / conviction sizing
    - > 3.0: elite
    """
    wins = [abs(t.pnl_dollars) for t in trades if t.pnl_dollars > 0]
    losses = [abs(t.pnl_dollars) for t in trades if t.pnl_dollars < 0]

    if not wins or not losses:
        return 0.0

    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return float("inf")
    return avg_win / avg_loss


def compute_expected_value(trades: list[ClosedTrade]) -> float:
    """
    Expected value per trade in dollars.

    EV = hit_rate × avg_win - miss_rate × avg_loss

    Positive EV means the PM has edge. The magnitude tells you
    how much each trade is "worth" in expectation.
    """
    if not trades:
        return 0.0

    wins = [t.pnl_dollars for t in trades if t.pnl_dollars > 0]
    losses = [abs(t.pnl_dollars) for t in trades if t.pnl_dollars < 0]

    hit_rate = len(wins) / len(trades)
    miss_rate = 1.0 - hit_rate

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return hit_rate * avg_win - miss_rate * avg_loss


def compute_sector_stats(trades: list[ClosedTrade]) -> dict[str, dict[str, float]]:
    """
    Compute hit rate, slugging, and alpha by sector.

    Returns: {
        sector: {
            "hit_rate": float,
            "slugging": float,
            "total_pnl": float,
            "num_trades": int,
            "avg_pnl_pct": float,
        }
    }
    """
    by_sector: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        sector = t.sector or "Unknown"
        if sector not in by_sector:
            by_sector[sector] = []
        by_sector[sector].append(t)

    result: dict[str, dict[str, float]] = {}
    for sector, sector_trades in by_sector.items():
        result[sector] = {
            "hit_rate": compute_hit_rate(sector_trades),
            "slugging": compute_slugging_pct(sector_trades),
            "total_pnl": sum(t.pnl_dollars for t in sector_trades),
            "num_trades": float(len(sector_trades)),
            "avg_pnl_pct": float(np.mean([t.pnl_pct for t in sector_trades])),
        }

    return result


def compute_side_stats(
    trades: list[ClosedTrade],
) -> dict[str, dict[str, float]]:
    """Compute hit rate and slugging for long vs short book separately."""
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    return {
        "long": {
            "hit_rate": compute_hit_rate(longs),
            "slugging": compute_slugging_pct(longs),
            "num_trades": float(len(longs)),
            "total_pnl": sum(t.pnl_dollars for t in longs),
        },
        "short": {
            "hit_rate": compute_hit_rate(shorts),
            "slugging": compute_slugging_pct(shorts),
            "num_trades": float(len(shorts)),
            "total_pnl": sum(t.pnl_dollars for t in shorts),
        },
    }


def compute_holding_period_stats(trades: list[ClosedTrade]) -> dict[str, float]:
    """Analyze hit rate and slugging by holding period bucket."""
    if not trades:
        return {}

    buckets = {
        "< 1 week": (0, 5),
        "1-4 weeks": (5, 20),
        "1-3 months": (20, 63),
        "3-6 months": (63, 126),
        "> 6 months": (126, 99999),
    }

    result: dict[str, float] = {}
    for label, (lo, hi) in buckets.items():
        bucket_trades = [t for t in trades if lo <= t.holding_days < hi]
        if bucket_trades:
            result[f"{label}_hit_rate"] = compute_hit_rate(bucket_trades)
            result[f"{label}_slugging"] = compute_slugging_pct(bucket_trades)
            result[f"{label}_count"] = float(len(bucket_trades))

    return result


def build_scorecard(
    closed_trades: list[ClosedTrade],
    total_return_pct: float = 0.0,
    annualized_return_pct: float = 0.0,
    sharpe: float = 0.0,
    alpha: float = 0.0,
    max_drawdown: float = 0.0,
    avg_drawdown_depth: float = 0.0,
    avg_recovery_days: float = 0.0,
    num_drawdowns: int = 0,
    turnover_pct: float = 0.0,
    trading_days: int = 1,
) -> PMScorecard:
    """
    Build the complete PM scorecard from closed trades and portfolio-level stats.

    This is the master function that computes everything.
    """
    if not closed_trades:
        return PMScorecard(
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe=sharpe,
            alpha=alpha,
        )

    # Core metrics
    winners = [t for t in closed_trades if t.is_winner]
    losers = [t for t in closed_trades if not t.is_winner]
    win_pnls = [t.pnl_dollars for t in winners]
    loss_pnls = [abs(t.pnl_dollars) for t in losers]

    # Best/worst trades
    all_pnl = [(t.pnl_dollars, t.ticker) for t in closed_trades]
    best = max(all_pnl, key=lambda x: x[0])
    worst = min(all_pnl, key=lambda x: x[0])

    # Side stats
    side_stats = compute_side_stats(closed_trades)

    # Trades per month
    trades_per_month = len(closed_trades) / max(trading_days / 21, 1)

    scorecard = PMScorecard(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        sharpe=sharpe,
        alpha=alpha,
        hit_rate=compute_hit_rate(closed_trades),
        slugging_pct=compute_slugging_pct(closed_trades),
        expected_value_per_trade=compute_expected_value(closed_trades),
        win_loss_ratio=(sum(win_pnls) / sum(loss_pnls)) if loss_pnls else float("inf"),
        total_trades=len(closed_trades),
        total_winners=len(winners),
        total_losers=len(losers),
        avg_win_pct=float(np.mean([t.pnl_pct for t in winners])) if winners else 0.0,
        avg_loss_pct=float(np.mean([t.pnl_pct for t in losers])) if losers else 0.0,
        avg_win_dollars=float(np.mean(win_pnls)) if win_pnls else 0.0,
        avg_loss_dollars=float(np.mean(loss_pnls)) if loss_pnls else 0.0,
        best_trade_pnl=best[0],
        best_trade_ticker=best[1],
        worst_trade_pnl=worst[0],
        worst_trade_ticker=worst[1],
        avg_holding_period_days=float(np.mean([t.holding_days for t in closed_trades])),
        long_hit_rate=side_stats["long"]["hit_rate"],
        long_slugging=side_stats["long"]["slugging"],
        short_hit_rate=side_stats["short"]["hit_rate"],
        short_slugging=side_stats["short"]["slugging"],
        sector_stats=compute_sector_stats(closed_trades),
        max_drawdown=max_drawdown,
        avg_drawdown_depth=avg_drawdown_depth,
        avg_recovery_days=avg_recovery_days,
        num_drawdowns=num_drawdowns,
        turnover_pct=turnover_pct,
        trades_per_month=trades_per_month,
    )

    return scorecard
