"""Tests for core/metrics/pm_performance.py — the PM scorecard engine."""

from datetime import date

import pytest

from core.metrics.pm_performance import (
    ClosedTrade,
    build_scorecard,
    compute_expected_value,
    compute_hit_rate,
    compute_sector_stats,
    compute_side_stats,
    compute_slugging_pct,
)


def _make_trades() -> list[ClosedTrade]:
    """Generate a set of closed trades for testing."""
    return [
        ClosedTrade(
            ticker="AAPL", side="LONG", sector="Tech", subsector="Hardware",
            entry_date=date(2024, 1, 1), exit_date=date(2024, 3, 1),
            entry_price=150.0, exit_price=180.0, shares=100,
            pnl_dollars=3000.0, pnl_pct=0.20, holding_days=60,
        ),
        ClosedTrade(
            ticker="MSFT", side="LONG", sector="Tech", subsector="Software",
            entry_date=date(2024, 1, 15), exit_date=date(2024, 2, 15),
            entry_price=300.0, exit_price=280.0, shares=50,
            pnl_dollars=-1000.0, pnl_pct=-0.067, holding_days=31,
        ),
        ClosedTrade(
            ticker="UNH", side="SHORT", sector="Healthcare", subsector="HMO",
            entry_date=date(2024, 2, 1), exit_date=date(2024, 4, 1),
            entry_price=500.0, exit_price=460.0, shares=20,
            pnl_dollars=800.0, pnl_pct=0.08, holding_days=60,
        ),
        ClosedTrade(
            ticker="HCA", side="SHORT", sector="Healthcare", subsector="Hospitals",
            entry_date=date(2024, 2, 15), exit_date=date(2024, 3, 15),
            entry_price=250.0, exit_price=270.0, shares=40,
            pnl_dollars=-800.0, pnl_pct=-0.08, holding_days=28,
        ),
        ClosedTrade(
            ticker="ISRG", side="LONG", sector="Healthcare", subsector="MedTech",
            entry_date=date(2024, 3, 1), exit_date=date(2024, 5, 1),
            entry_price=400.0, exit_price=450.0, shares=30,
            pnl_dollars=1500.0, pnl_pct=0.125, holding_days=61,
        ),
    ]


class TestHitRate:
    def test_basic(self):
        trades = _make_trades()
        hr = compute_hit_rate(trades)
        # 3 winners (AAPL, UNH, ISRG), 2 losers (MSFT, HCA) → 60%
        assert hr == pytest.approx(0.60)

    def test_empty(self):
        assert compute_hit_rate([]) == 0.0

    def test_all_winners(self):
        trades = [t for t in _make_trades() if t.is_winner]
        assert compute_hit_rate(trades) == 1.0


class TestSluggingPct:
    def test_basic(self):
        trades = _make_trades()
        slug = compute_slugging_pct(trades)
        # Avg win: (3000 + 800 + 1500) / 3 = 1766.67
        # Avg loss: (1000 + 800) / 2 = 900
        # Slugging: 1766.67 / 900 ≈ 1.96
        expected = (3000 + 800 + 1500) / 3.0 / ((1000 + 800) / 2.0)
        assert slug == pytest.approx(expected, abs=0.01)

    def test_empty(self):
        assert compute_slugging_pct([]) == 0.0

    def test_no_losses(self):
        winners = [t for t in _make_trades() if t.is_winner]
        slug = compute_slugging_pct(winners)
        assert slug == 0.0  # no losses to compare against

    def test_slug_above_1_is_good(self):
        """Verify our test data has slugging > 1."""
        trades = _make_trades()
        slug = compute_slugging_pct(trades)
        assert slug > 1.0


class TestExpectedValue:
    def test_positive_ev(self):
        trades = _make_trades()
        ev = compute_expected_value(trades)
        # With 60% hit rate and 2x slugging, EV should be positive
        assert ev > 0

    def test_ev_formula(self):
        trades = _make_trades()
        ev = compute_expected_value(trades)
        # EV = hit_rate × avg_win - miss_rate × avg_loss
        hr = compute_hit_rate(trades)
        wins = [t.pnl_dollars for t in trades if t.pnl_dollars > 0]
        losses = [abs(t.pnl_dollars) for t in trades if t.pnl_dollars < 0]
        expected = hr * (sum(wins) / len(wins)) - (1 - hr) * (sum(losses) / len(losses))
        assert ev == pytest.approx(expected)


class TestSectorStats:
    def test_sectors_present(self):
        trades = _make_trades()
        stats = compute_sector_stats(trades)
        assert "Tech" in stats
        assert "Healthcare" in stats

    def test_sector_hit_rate(self):
        trades = _make_trades()
        stats = compute_sector_stats(trades)
        # Tech: AAPL won, MSFT lost → 50%
        assert stats["Tech"]["hit_rate"] == pytest.approx(0.50)
        # Healthcare: UNH won, HCA lost, ISRG won → 66.7%
        assert stats["Healthcare"]["hit_rate"] == pytest.approx(2 / 3, abs=0.01)


class TestSideStats:
    def test_long_short_split(self):
        trades = _make_trades()
        stats = compute_side_stats(trades)
        assert "long" in stats
        assert "short" in stats
        assert stats["long"]["num_trades"] == 3.0
        assert stats["short"]["num_trades"] == 2.0


class TestBuildScorecard:
    def test_scorecard_complete(self):
        trades = _make_trades()
        sc = build_scorecard(
            closed_trades=trades,
            total_return_pct=14.2,
            sharpe=1.42,
            alpha=5.8,
            trading_days=120,
        )
        assert sc.hit_rate == pytest.approx(0.60)
        assert sc.slugging_pct > 1.0
        assert sc.total_trades == 5
        assert sc.total_winners == 3
        assert sc.total_losers == 2
        assert sc.best_trade_ticker == "AAPL"
        assert sc.worst_trade_ticker == "MSFT"
        assert sc.sharpe == 1.42
        assert "Tech" in sc.sector_stats
        assert "Healthcare" in sc.sector_stats
        assert sc.trades_per_month > 0

    def test_empty_scorecard(self):
        sc = build_scorecard(closed_trades=[])
        assert sc.total_trades == 0
        assert sc.hit_rate == 0.0
        assert sc.slugging_pct == 0.0
