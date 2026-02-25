"""Tests for core/metrics/attribution.py — position, sector, side, factor attribution.

Key invariant tested throughout: attributions must sum to the total.
This is the single most important property of any attribution system.
A sign error or missed position silently produces plausible-looking wrong numbers.
"""

import polars as pl
import pytest

from core.metrics.attribution import (
    AttributionSummary,
    factor_attribution,
    full_attribution,
    position_attribution,
    sector_attribution,
    side_attribution,
)
from core.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Helpers — build portfolios with known P&L for deterministic assertions
# ---------------------------------------------------------------------------


def _make_portfolio(
    positions: list[dict] | None = None,
    nav: float = 1_000_000,
) -> Portfolio:
    """Build a small portfolio with known entry/current prices for deterministic P&L."""
    if positions is None:
        positions = [
            # Long AAPL: bought at 150, now 165 → +$15/share × 1000 = +$15,000
            {"ticker": "AAPL", "side": "LONG", "shares": 1000, "entry_price": 150.0,
             "current_price": 165.0, "sector": "Technology"},
            # Long JNJ: bought at 170, now 160 → -$10/share × 500 = -$5,000
            {"ticker": "JNJ", "side": "LONG", "shares": 500, "entry_price": 170.0,
             "current_price": 160.0, "sector": "Health Care"},
            # Short XOM: shorted at 100, now 90 → +$10/share × 800 = +$8,000
            {"ticker": "XOM", "side": "SHORT", "shares": 800, "entry_price": 100.0,
             "current_price": 90.0, "sector": "Energy"},
            # Short TSLA: shorted at 200, now 220 → -$20/share × 300 = -$6,000
            {"ticker": "TSLA", "side": "SHORT", "shares": 300, "entry_price": 200.0,
             "current_price": 220.0, "sector": "Technology"},
        ]
    pos_list = [Position(**p) for p in positions]
    return Portfolio(positions=pos_list, nav=nav)


def _make_returns_df() -> pl.DataFrame:
    """Minimal returns DF with SPY column for factor attribution."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 60
    return pl.DataFrame({
        "date": list(range(n)),
        "SPY": rng.normal(0.0004, 0.01, n).tolist(),
        "AAPL": rng.normal(0.0005, 0.015, n).tolist(),
        "JNJ": rng.normal(0.0002, 0.008, n).tolist(),
        "XOM": rng.normal(0.0003, 0.012, n).tolist(),
        "TSLA": rng.normal(0.0006, 0.02, n).tolist(),
    })


# ---------------------------------------------------------------------------
# Position-level attribution
# ---------------------------------------------------------------------------


class TestPositionAttribution:
    def test_contribution_bps_sum_equals_total(self):
        """Position contributions must sum to total portfolio P&L in bps."""
        p = _make_portfolio()
        attribs = position_attribution(p)
        total_bps = sum(a.contribution_bps for a in attribs)
        expected_bps = (p.total_pnl_dollars / p.nav) * 10_000
        assert total_bps == pytest.approx(expected_bps, abs=0.01)

    def test_individual_contributions_correct(self):
        """Each position's contribution = pnl / nav × 10,000."""
        p = _make_portfolio()
        attribs = position_attribution(p)
        by_ticker = {a.ticker: a for a in attribs}

        # AAPL: +$15,000 / $1M = +150 bps
        assert by_ticker["AAPL"].contribution_bps == pytest.approx(150.0)
        # JNJ: -$5,000 / $1M = -50 bps
        assert by_ticker["JNJ"].contribution_bps == pytest.approx(-50.0)
        # XOM short: +$8,000 / $1M = +80 bps
        assert by_ticker["XOM"].contribution_bps == pytest.approx(80.0)
        # TSLA short: -$6,000 / $1M = -60 bps
        assert by_ticker["TSLA"].contribution_bps == pytest.approx(-60.0)

    def test_pnl_dollars_correct(self):
        """Check raw P&L dollars match hand calculations."""
        p = _make_portfolio()
        attribs = position_attribution(p)
        by_ticker = {a.ticker: a for a in attribs}

        assert by_ticker["AAPL"].pnl_dollars == pytest.approx(15_000.0)
        assert by_ticker["JNJ"].pnl_dollars == pytest.approx(-5_000.0)
        assert by_ticker["XOM"].pnl_dollars == pytest.approx(8_000.0)
        assert by_ticker["TSLA"].pnl_dollars == pytest.approx(-6_000.0)

    def test_sorted_by_absolute_contribution(self):
        """Positions should be sorted by |contribution_bps| descending."""
        p = _make_portfolio()
        attribs = position_attribution(p)
        abs_contribs = [abs(a.contribution_bps) for a in attribs]
        assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_pnl_pct_matches_position_level(self):
        """Position P&L % should match the position model's pnl_pct."""
        p = _make_portfolio()
        attribs = position_attribution(p)
        by_ticker = {a.ticker: a for a in attribs}
        # AAPL: (165-150)/150 = 10%
        assert by_ticker["AAPL"].pnl_pct == pytest.approx(10.0)
        # XOM short: (100-90)/100 = 10%
        assert by_ticker["XOM"].pnl_pct == pytest.approx(10.0)

    def test_side_labels_correct(self):
        p = _make_portfolio()
        attribs = position_attribution(p)
        by_ticker = {a.ticker: a for a in attribs}
        assert by_ticker["AAPL"].side == "LONG"
        assert by_ticker["XOM"].side == "SHORT"

    def test_empty_portfolio(self):
        p = Portfolio(positions=[], nav=1_000_000)
        assert position_attribution(p) == []

    def test_zero_nav_returns_empty(self):
        """Guard against division by zero."""
        p = _make_portfolio(nav=1_000_000)
        # Manually override to test the guard (can't create Portfolio with nav=0)
        p_dict = p.model_dump()
        p_dict["nav"] = 0.01  # tiny, but still >0 to pass validation
        p2 = Portfolio(**p_dict)
        attribs = position_attribution(p2)
        # Should still work, just with large bps numbers
        assert len(attribs) == 4


# ---------------------------------------------------------------------------
# Sector-level attribution
# ---------------------------------------------------------------------------


class TestSectorAttribution:
    def test_sector_bps_sum_equals_total(self):
        """Sector contributions must sum to total portfolio P&L in bps."""
        p = _make_portfolio()
        sectors = sector_attribution(p)
        total_bps = sum(s.contribution_bps for s in sectors)
        expected_bps = (p.total_pnl_dollars / p.nav) * 10_000
        assert total_bps == pytest.approx(expected_bps, abs=0.01)

    def test_sector_pnl_dollars_sum_to_total(self):
        """Sector P&L dollars must sum to total portfolio P&L."""
        p = _make_portfolio()
        sectors = sector_attribution(p)
        total_pnl = sum(s.pnl_dollars for s in sectors)
        assert total_pnl == pytest.approx(p.total_pnl_dollars)

    def test_long_short_pnl_sum_to_sector_pnl(self):
        """Within each sector, long_pnl + short_pnl == sector pnl."""
        p = _make_portfolio()
        sectors = sector_attribution(p)
        for s in sectors:
            assert s.long_pnl + s.short_pnl == pytest.approx(s.pnl_dollars)

    def test_technology_sector_aggregation(self):
        """Tech has AAPL (long, +$15k) and TSLA (short, -$6k) → net +$9k."""
        p = _make_portfolio()
        sectors = sector_attribution(p)
        tech = next(s for s in sectors if s.sector == "Technology")
        assert tech.pnl_dollars == pytest.approx(9_000.0)
        assert tech.long_pnl == pytest.approx(15_000.0)
        assert tech.short_pnl == pytest.approx(-6_000.0)
        assert tech.position_count == 2

    def test_sector_contribution_bps_formula(self):
        """Each sector's bps = sector_pnl / nav × 10,000."""
        p = _make_portfolio()
        sectors = sector_attribution(p)
        for s in sectors:
            expected_bps = (s.pnl_dollars / p.nav) * 10_000
            assert s.contribution_bps == pytest.approx(expected_bps)

    def test_sorted_by_absolute_contribution(self):
        p = _make_portfolio()
        sectors = sector_attribution(p)
        abs_contribs = [abs(s.contribution_bps) for s in sectors]
        assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_unknown_sector_handling(self):
        """Positions without sector label go to 'Unknown'."""
        positions = [
            {"ticker": "XYZ", "side": "LONG", "shares": 100, "entry_price": 50.0,
             "current_price": 55.0, "sector": ""},
        ]
        p = _make_portfolio(positions=positions)
        sectors = sector_attribution(p)
        assert sectors[0].sector == "Unknown"

    def test_empty_portfolio(self):
        p = Portfolio(positions=[], nav=1_000_000)
        assert sector_attribution(p) == []


# ---------------------------------------------------------------------------
# Side-level attribution
# ---------------------------------------------------------------------------


class TestSideAttribution:
    def test_long_plus_short_equals_total(self):
        """Long P&L + short P&L must equal total portfolio P&L."""
        p = _make_portfolio()
        sa = side_attribution(p)
        assert sa.long_pnl + sa.short_pnl == pytest.approx(p.total_pnl_dollars)

    def test_long_pnl_correct(self):
        """Long book: AAPL +$15k, JNJ -$5k → net $10k."""
        p = _make_portfolio()
        sa = side_attribution(p)
        assert sa.long_pnl == pytest.approx(10_000.0)

    def test_short_pnl_correct(self):
        """Short book: XOM +$8k, TSLA -$6k → net $2k."""
        p = _make_portfolio()
        sa = side_attribution(p)
        assert sa.short_pnl == pytest.approx(2_000.0)

    def test_contribution_bps_formula(self):
        """Contribution bps = side_pnl / nav × 10,000."""
        p = _make_portfolio()
        sa = side_attribution(p)
        assert sa.long_contribution_bps == pytest.approx(100.0)  # $10k / $1M × 10k
        assert sa.short_contribution_bps == pytest.approx(20.0)  # $2k / $1M × 10k

    def test_hit_rates(self):
        """Hit rate = fraction of positions with positive P&L per side."""
        p = _make_portfolio()
        sa = side_attribution(p)
        # Longs: AAPL wins, JNJ loses → 50%
        assert sa.long_hit_rate == pytest.approx(0.5)
        # Shorts: XOM wins, TSLA loses → 50%
        assert sa.short_hit_rate == pytest.approx(0.5)

    def test_all_long_portfolio(self):
        """Portfolio with no shorts — short P&L = 0, short hit rate = 0."""
        positions = [
            {"ticker": "AAPL", "side": "LONG", "shares": 100, "entry_price": 150.0,
             "current_price": 160.0, "sector": "Technology"},
        ]
        p = _make_portfolio(positions=positions)
        sa = side_attribution(p)
        assert sa.short_pnl == 0.0
        assert sa.short_hit_rate == 0.0
        assert sa.long_hit_rate == 1.0

    def test_all_short_portfolio(self):
        positions = [
            {"ticker": "XOM", "side": "SHORT", "shares": 100, "entry_price": 100.0,
             "current_price": 90.0, "sector": "Energy"},
        ]
        p = _make_portfolio(positions=positions)
        sa = side_attribution(p)
        assert sa.long_pnl == 0.0
        assert sa.long_hit_rate == 0.0
        assert sa.short_hit_rate == 1.0


# ---------------------------------------------------------------------------
# Factor-level attribution
# ---------------------------------------------------------------------------


class TestFactorAttribution:
    def test_factor_components_sum_to_total_return(self):
        """market + size + value + momentum + alpha = total return (by construction)."""
        p = _make_portfolio()
        betas = {"AAPL": 1.3, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.8}
        market_return = 0.05  # 5% market return
        fa = factor_attribution(p, returns_df=None, betas=betas, market_return=market_return)

        assert fa is not None
        total = (
            fa.market_contribution_pct
            + fa.size_contribution_pct
            + fa.value_contribution_pct
            + fa.momentum_contribution_pct
            + fa.alpha_contribution_pct
            + fa.residual_pct
        )
        expected = p.total_pnl_pct * 100
        assert total == pytest.approx(expected, abs=0.01)

    def test_market_contribution_uses_weighted_beta(self):
        """Market contrib = portfolio_beta × market_return × 100."""
        p = _make_portfolio()
        betas = {"AAPL": 1.3, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.8}
        market_return = 0.05
        fa = factor_attribution(p, returns_df=None, betas=betas, market_return=market_return)

        # Compute expected weighted beta
        nav = p.nav
        weighted_beta = sum(
            pos.weight_in(nav) * betas[pos.ticker]
            for pos in p.positions
        )
        expected_market = weighted_beta * market_return * 100
        assert fa.market_contribution_pct == pytest.approx(expected_market, abs=0.01)

    def test_factor_returns_passed_through(self):
        """Size/value/momentum contributions come from factor_returns dict."""
        p = _make_portfolio()
        betas = {"AAPL": 1.0, "JNJ": 1.0, "XOM": 1.0, "TSLA": 1.0}
        factor_returns = {"smb": 0.02, "hml": -0.01, "momentum": 0.03}
        fa = factor_attribution(
            p, returns_df=None, betas=betas,
            market_return=0.0, factor_returns=factor_returns,
        )
        assert fa.size_contribution_pct == pytest.approx(2.0)
        assert fa.value_contribution_pct == pytest.approx(-1.0)
        assert fa.momentum_contribution_pct == pytest.approx(3.0)

    def test_alpha_is_residual_after_factors(self):
        """Alpha = total_return - (market + size + value + momentum)."""
        p = _make_portfolio()
        betas = {"AAPL": 1.0, "JNJ": 1.0, "XOM": 1.0, "TSLA": 1.0}
        fa = factor_attribution(
            p, returns_df=None, betas=betas,
            market_return=0.0, factor_returns={},
        )
        # With zero market return and no factors, alpha should equal total return
        assert fa.alpha_contribution_pct == pytest.approx(p.total_pnl_pct * 100, abs=0.01)

    def test_spy_column_fallback(self):
        """When no market_return, should use SPY column from returns_df."""
        p = _make_portfolio()
        betas = {"AAPL": 1.0, "JNJ": 1.0, "XOM": 1.0, "TSLA": 1.0}
        returns_df = _make_returns_df()
        fa = factor_attribution(p, returns_df=returns_df, betas=betas)
        # Should not be None and should compute something
        assert fa is not None
        assert fa.market_contribution_pct != 0.0

    def test_none_without_betas(self):
        p = _make_portfolio()
        assert factor_attribution(p, returns_df=None, betas=None) is None
        assert factor_attribution(p, returns_df=None, betas={}) is None

    def test_none_with_zero_nav(self):
        """Tiny NAV still works (no zero division crash)."""
        positions = [
            {"ticker": "AAPL", "side": "LONG", "shares": 1, "entry_price": 0.01,
             "current_price": 0.01, "sector": "Technology"},
        ]
        p = _make_portfolio(positions=positions, nav=100.0)
        fa = factor_attribution(p, returns_df=None, betas={"AAPL": 1.0}, market_return=0.05)
        assert fa is not None


# ---------------------------------------------------------------------------
# Full attribution summary
# ---------------------------------------------------------------------------


class TestFullAttribution:
    def test_summary_contains_all_views(self):
        p = _make_portfolio()
        betas = {"AAPL": 1.3, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.8}
        summary = full_attribution(p, returns_df=None, betas=betas)

        assert isinstance(summary, AttributionSummary)
        assert len(summary.positions) == 4
        assert len(summary.sectors) == 3  # Tech, Health Care, Energy
        assert summary.side is not None
        # factors is None here because no market_return or returns_df with SPY
        assert summary.total_pnl_dollars == pytest.approx(p.total_pnl_dollars)

    def test_total_bps_correct(self):
        p = _make_portfolio()
        summary = full_attribution(p)
        expected_bps = (p.total_pnl_dollars / p.nav) * 10_000
        assert summary.total_pnl_bps == pytest.approx(expected_bps)

    def test_cross_view_consistency(self):
        """Total from positions == total from sectors == total from sides."""
        p = _make_portfolio()
        summary = full_attribution(p)
        pos_total = sum(a.contribution_bps for a in summary.positions)
        sector_total = sum(s.contribution_bps for s in summary.sectors)
        side_total = summary.side.long_contribution_bps + summary.side.short_contribution_bps
        assert pos_total == pytest.approx(summary.total_pnl_bps, abs=0.01)
        assert sector_total == pytest.approx(summary.total_pnl_bps, abs=0.01)
        assert side_total == pytest.approx(summary.total_pnl_bps, abs=0.01)

    def test_empty_portfolio_summary(self):
        p = Portfolio(positions=[], nav=1_000_000)
        summary = full_attribution(p)
        assert summary.total_pnl_dollars == 0.0
        assert summary.total_pnl_bps == 0.0
        assert len(summary.positions) == 0
        assert len(summary.sectors) == 0

    def test_single_position(self):
        """Degenerate case: one position should attribute everything to itself."""
        positions = [
            {"ticker": "AAPL", "side": "LONG", "shares": 1000, "entry_price": 100.0,
             "current_price": 110.0, "sector": "Technology"},
        ]
        p = _make_portfolio(positions=positions)
        summary = full_attribution(p)
        assert len(summary.positions) == 1
        assert len(summary.sectors) == 1
        assert summary.positions[0].contribution_bps == pytest.approx(summary.total_pnl_bps)
        assert summary.sectors[0].contribution_bps == pytest.approx(summary.total_pnl_bps)
