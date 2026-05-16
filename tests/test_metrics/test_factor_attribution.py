"""Tests for core/metrics/factor_attribution.py"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from core.metrics.factor_attribution import (
    FactorAttribution,
    NameContrib,
    factor_pnl_attribution,
    hit_rate,
    slugging,
)
from core.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _trading_dates(n: int, end: date | None = None) -> list[date]:
    """Generate n consecutive weekday dates ending at `end` (default today)."""
    end = end or date(2026, 5, 15)  # Friday — gives WTD = Mon-Fri
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def _build_market_returns(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, n)


def _make_position(
    ticker: str,
    side: str = "LONG",
    shares: float = 1000.0,
    price: float = 100.0,
    sector: str = "",
) -> Position:
    return Position(
        ticker=ticker,
        side=side,  # type: ignore[arg-type]
        shares=shares,
        entry_price=price,
        current_price=price,
        sector=sector,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleNameSingleFactor1D:
    def test_pure_beta_reconciles_exactly(self):
        """β=1 name on a single market factor — over 1D, pnl_market should
        equal realized $ P&L (idio ~= 0) and the total should reconcile."""
        n = 60
        dates = _trading_dates(n)
        market = _build_market_returns(n, seed=1)
        # r_i = 1.0 * r_market (exact pure beta, no alpha, no idio)
        name_ret = market.copy()

        returns_df = pl.DataFrame({"date": dates, "AAA": name_ret})
        factor_returns = {"market": pl.Series("market", market)}

        pos = _make_position("AAA", side="LONG", shares=1000, price=100.0)
        port = Portfolio(name="t", positions=[pos], nav=10_000_000.0)

        result = factor_pnl_attribution(port, returns_df, factor_returns, horizon="1D")

        # Realized $ P&L for the last day:
        notional = pos.notional
        expected_pnl = market[-1] * notional

        assert result.total_pnl == pytest.approx(expected_pnl, rel=1e-6, abs=1e-6)
        # Pure beta → market component carries essentially all P&L
        assert result.market_pnl == pytest.approx(expected_pnl, rel=1e-3, abs=1e-3)
        assert abs(result.idio_pnl) < 1e-3


class TestMultiPositionWTD:
    def test_sums_reconcile(self):
        """Per-name pnl_total should sum to realized total; component sum
        should reconcile to realized total within tolerance."""
        n = 60
        dates = _trading_dates(n, end=date(2026, 5, 15))  # Friday
        market = _build_market_returns(n, seed=2)

        # 4 names: mix of pure-beta and alpha-with-noise so idio is non-zero
        rng = np.random.default_rng(7)
        r1 = 1.2 * market + rng.normal(0.0, 0.005, n)
        r2 = 0.8 * market + rng.normal(0.0, 0.004, n)
        r3 = 1.0 * market  # pure beta
        r4 = -0.5 * market + rng.normal(0.0, 0.006, n)

        returns_df = pl.DataFrame({"date": dates, "A": r1, "B": r2, "C": r3, "D": r4})
        factor_returns = {"market": pl.Series("market", market)}

        positions = [
            _make_position("A", "LONG", shares=1000, price=100),
            _make_position("B", "LONG", shares=500, price=200),
            _make_position("C", "SHORT", shares=2000, price=50),
            _make_position("D", "SHORT", shares=300, price=400),
        ]
        port = Portfolio(name="t", positions=positions, nav=10_000_000.0)

        result = factor_pnl_attribution(port, returns_df, factor_returns, horizon="WTD")

        # WTD slice = Monday..Friday (5 rows) at end of synthetic series
        per_name_total = sum(c.pnl_total for c in result.name_contribs)
        assert result.total_pnl == pytest.approx(per_name_total, rel=1e-6, abs=1e-6)

        # Components should reconcile to realized
        components_sum = (
            result.market_pnl + sum(result.style_pnl.values()) + result.sector_pnl + result.idio_pnl
        )
        assert components_sum == pytest.approx(result.total_pnl, rel=1e-6, abs=1e-6)


class TestZeroBetaPortfolio:
    def test_all_pnl_in_idio(self):
        """Names with returns independent of the factor → fit β ≈ 0
        → market_pnl is small relative to idio P&L."""
        n = 250  # long history → sample β converges to true β=0
        dates = _trading_dates(n)
        rng = np.random.default_rng(99)
        market = rng.normal(0.0, 0.01, n)
        # Independent name returns (uncorrelated with market by construction)
        r1 = rng.normal(0.0008, 0.012, n)
        r2 = rng.normal(0.0005, 0.011, n)

        returns_df = pl.DataFrame({"date": dates, "X": r1, "Y": r2})
        factor_returns = {"market": pl.Series("market", market)}

        positions = [
            _make_position("X", "LONG", shares=1000, price=100),
            _make_position("Y", "LONG", shares=1000, price=100),
        ]
        port = Portfolio(name="t", positions=positions, nav=10_000_000.0)

        # MTD slice: realized $ has signal, fitted β ≈ 0 → idio dominates
        result = factor_pnl_attribution(port, returns_df, factor_returns, horizon="MTD")

        # market_pnl should be much smaller than idio_pnl in magnitude
        assert abs(result.market_pnl) < 0.25 * max(abs(result.idio_pnl), 1.0)
        # idio_pnl should approximately equal total_pnl
        if abs(result.total_pnl) > 1.0:
            assert abs(result.idio_pnl / result.total_pnl - 1.0) < 0.25


class TestPureBetaPortfolio:
    def test_market_carries_pnl(self):
        """β=1 on market, exact (no alpha) → market_pnl carries most of P&L,
        idio holds only the multi-day geometric-vs-arithmetic residual."""
        n = 80
        dates = _trading_dates(n)
        market = _build_market_returns(n, seed=3)
        # Three names, all pure-beta with β=1
        returns_df = pl.DataFrame(
            {
                "date": dates,
                "A": market.copy(),
                "B": market.copy(),
                "C": market.copy(),
            }
        )
        factor_returns = {"market": pl.Series("market", market)}

        positions = [
            _make_position("A", "LONG", shares=1000, price=100),
            _make_position("B", "LONG", shares=2000, price=50),
            _make_position("C", "SHORT", shares=500, price=200),
        ]
        port = Portfolio(name="t", positions=positions, nav=10_000_000.0)

        # 1D horizon: geometric == arithmetic, so idio is exactly zero
        result_1d = factor_pnl_attribution(port, returns_df, factor_returns, horizon="1D")
        assert abs(result_1d.idio_pnl) < 1e-6
        if result_1d.total_pnl != 0.0:
            assert result_1d.market_pnl == pytest.approx(result_1d.total_pnl, rel=1e-6)

        # MTD horizon: small residual due to compounding, but market still dominates
        result_mtd = factor_pnl_attribution(port, returns_df, factor_returns, horizon="MTD")
        if abs(result_mtd.total_pnl) > 1.0:
            # Market explains ≥90% of the realized P&L
            assert abs(result_mtd.market_pnl / result_mtd.total_pnl) > 0.9


class TestHitRate:
    def test_six_winners_four_losers(self):
        contribs = [
            NameContrib(ticker=f"W{i}", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=1.0)
            for i in range(6)
        ] + [
            NameContrib(ticker=f"L{i}", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=-1.0)
            for i in range(4)
        ]
        assert hit_rate(contribs) == pytest.approx(0.6)

    def test_empty(self):
        assert hit_rate([]) == 0.0


class TestSlugging:
    def test_avg_win_2_avg_loss_1(self):
        """Two winners avg $2, two losers avg $1 → slugging = 2.0."""
        contribs = [
            NameContrib(ticker="W1", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=1.0),
            NameContrib(ticker="W2", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=3.0),
            NameContrib(ticker="L1", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=-0.5),
            NameContrib(ticker="L2", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=-1.5),
        ]
        # avg win = (1+3)/2 = 2.0; avg loss = (0.5+1.5)/2 = 1.0; ratio = 2.0
        assert slugging(contribs) == pytest.approx(2.0)

    def test_no_losers_returns_nan(self):
        contribs = [
            NameContrib(ticker="W1", side="LONG", pnl_total=0, pnl_market=0, pnl_idio=1.0),
        ]
        result = slugging(contribs)
        import math

        assert math.isnan(result)


class TestTopIdioRankings:
    def test_winners_losers_sorted_and_capped(self):
        n = 60
        dates = _trading_dates(n)
        rng = np.random.default_rng(13)
        market = rng.normal(0.0, 0.01, n)

        # 12 names so cap-at-10 is exercised
        tickers = [f"T{i:02d}" for i in range(12)]
        cols = {"date": dates}
        for i, t in enumerate(tickers):
            # Mix of betas + idio noise so per-name idio P&L varies
            beta = 0.5 + 0.1 * i
            cols[t] = beta * market + rng.normal((-1) ** i * 0.0005, 0.008, n)

        returns_df = pl.DataFrame(cols)
        factor_returns = {"market": pl.Series("market", market)}

        positions = [_make_position(t, "LONG", shares=1000, price=100) for t in tickers]
        port = Portfolio(name="t", positions=positions, nav=10_000_000.0)

        result = factor_pnl_attribution(port, returns_df, factor_returns, horizon="MTD")

        # Capped at 10
        assert len(result.top_idio_winners) == 10
        assert len(result.top_idio_losers) == 10

        # Winners sorted descending by idio
        w_idios = [c.pnl_idio for c in result.top_idio_winners]
        assert w_idios == sorted(w_idios, reverse=True)
        # Losers sorted ascending by idio
        l_idios = [c.pnl_idio for c in result.top_idio_losers]
        assert l_idios == sorted(l_idios)
        # Top winner ≥ top loser
        assert result.top_idio_winners[0].pnl_idio >= result.top_idio_losers[0].pnl_idio


class TestReturnTypes:
    def test_returns_factor_attribution_dataclass(self):
        n = 40
        dates = _trading_dates(n)
        market = _build_market_returns(n, seed=5)
        returns_df = pl.DataFrame({"date": dates, "A": market})
        factor_returns = {"market": pl.Series("market", market)}
        port = Portfolio(
            name="t",
            positions=[_make_position("A", "LONG", shares=100, price=100)],
            nav=1_000_000.0,
        )
        result = factor_pnl_attribution(port, returns_df, factor_returns, horizon="1D")
        assert isinstance(result, FactorAttribution)
        assert result.horizon == "1D"
        assert len(result.name_contribs) == 1
        assert isinstance(result.name_contribs[0], NameContrib)
