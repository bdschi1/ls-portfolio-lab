"""Tests for core/metrics/style_tilts.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.metrics.style_tilts import (
    CrowdingScore,
    SectorTilt,
    StyleTilt,
    StyleTiltsReport,
    style_tilts,
)
from core.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _factor_returns(
    n: int = 252,
    seed: int = 7,
    market_vol: float = 0.01,
) -> dict[str, pl.Series]:
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, market_vol, n)
    smb = rng.normal(0.0001, 0.005, n)
    hml = rng.normal(0.00005, 0.004, n)
    return {
        "MKT": pl.Series("MKT", mkt),
        "SMB": pl.Series("SMB", smb),
        "HML": pl.Series("HML", hml),
    }


def _make_returns_df(
    tickers: list[str],
    factors: dict[str, pl.Series],
    beta_map: dict[str, dict[str, float]],
    noise: float = 0.002,
    seed: int = 11,
) -> pl.DataFrame:
    """Synthesise per-ticker returns as Σ β_k * f_k + idiosyncratic noise.

    Low noise keeps the OLS estimates close to the generative betas, which the
    style-loading tests rely on.
    """
    rng = np.random.default_rng(seed)
    n = next(iter(factors.values())).len()
    cols: dict[str, pl.Series] = {}
    for t in tickers:
        betas = beta_map.get(t, {})
        r = np.zeros(n)
        for fname, fseries in factors.items():
            r = r + betas.get(fname, 0.0) * fseries.to_numpy()
        r = r + rng.normal(0.0, noise, n)
        cols[t] = pl.Series(t, r)
    return pl.DataFrame(cols)


def _equal_weight_portfolio(
    longs: list[tuple[str, str]],
    shorts: list[tuple[str, str]],
    nav: float = 1_000_000.0,
) -> Portfolio:
    """Build a portfolio with each leg equally weighted.

    ``longs``/``shorts`` are lists of (ticker, sector). Each name gets
    ``nav / total_legs`` of notional, so abs weights are all equal.
    """
    total = len(longs) + len(shorts)
    per_leg_notional = nav / total
    price = 100.0
    shares = per_leg_notional / price
    positions: list[Position] = []
    for t, sec in longs:
        positions.append(
            Position(
                ticker=t,
                side="LONG",
                shares=shares,
                entry_price=price,
                current_price=price,
                sector=sec,
            )
        )
    for t, sec in shorts:
        positions.append(
            Position(
                ticker=t,
                side="SHORT",
                shares=shares,
                entry_price=price,
                current_price=price,
                sector=sec,
            )
        )
    return Portfolio(positions=positions, nav=nav)


# ---------------------------------------------------------------------------
# Style loading / drift
# ---------------------------------------------------------------------------


class TestStyleLoading:
    def test_long_only_unit_beta_loading_near_one(self):
        """All-long portfolio of β_MKT=1 names → MKT loading ≈ gross long weight (1.0)."""
        factors = _factor_returns()
        tickers = [f"T{i}" for i in range(5)]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        report = style_tilts(portfolio, returns_df, factors)
        mkt = next(s for s in report.style if s.factor == "MKT")
        assert mkt.portfolio_loading == pytest.approx(1.0, abs=0.1)

    def test_perfect_ls_hedge_mkt_loading_near_zero(self):
        """Equal long & short books of β_MKT=1 names → MKT loading ≈ 0."""
        factors = _factor_returns()
        long_tickers = [f"L{i}" for i in range(3)]
        short_tickers = [f"S{i}" for i in range(3)]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in long_tickers + short_tickers}
        returns_df = _make_returns_df(long_tickers + short_tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in long_tickers],
            shorts=[(t, "Healthcare") for t in short_tickers],
        )
        report = style_tilts(portfolio, returns_df, factors)
        mkt = next(s for s in report.style if s.factor == "MKT")
        assert mkt.portfolio_loading == pytest.approx(0.0, abs=0.1)

    def test_drift_against_zero_target(self):
        """Target MKT=0 with a long-only β=1 book → drift ≈ +1 (matches loading)."""
        factors = _factor_returns()
        tickers = [f"T{i}" for i in range(4)]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        report = style_tilts(
            portfolio,
            returns_df,
            factors,
            targets={"MKT": 0.0},
        )
        mkt = next(s for s in report.style if s.factor == "MKT")
        assert mkt.target_loading == 0.0
        assert mkt.drift is not None
        assert mkt.drift == pytest.approx(mkt.portfolio_loading, abs=1e-9)
        assert mkt.drift == pytest.approx(1.0, abs=0.1)

    def test_no_targets_all_drift_none(self):
        factors = _factor_returns()
        tickers = ["A", "B"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        report = style_tilts(portfolio, returns_df, factors, targets=None)
        for s in report.style:
            assert s.target_loading is None
            assert s.drift is None


# ---------------------------------------------------------------------------
# Sector tilts / active weight
# ---------------------------------------------------------------------------


class TestSectorTilts:
    def test_net_positive_same_sector(self):
        """Two longs + one short in same sector at equal notional → net = +1/3 of gross."""
        factors = _factor_returns()
        long_tickers = ["L1", "L2"]
        short_tickers = ["S1"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in long_tickers + short_tickers}
        returns_df = _make_returns_df(long_tickers + short_tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in long_tickers],
            shorts=[(t, "Technology") for t in short_tickers],
        )
        report = style_tilts(portfolio, returns_df, factors)
        tech = next(s for s in report.sectors if s.sector == "Technology")
        # 3 equal-notional legs at $1M NAV → each leg = 1/3 abs weight.
        assert tech.long_weight == pytest.approx(2 / 3, abs=1e-9)
        assert tech.short_weight == pytest.approx(1 / 3, abs=1e-9)
        assert tech.portfolio_weight == pytest.approx(1 / 3, abs=1e-9)
        assert tech.portfolio_weight > 0

    def test_active_weight_against_benchmark(self):
        """Sector weight 0.45 vs benchmark 0.30 → active +0.15."""
        factors = _factor_returns()
        # Two longs in Technology with equal notional → 0.45 of NAV combined.
        # Build a custom portfolio so we hit exactly 45%.
        nav = 1_000_000.0
        price = 100.0
        positions = [
            Position(
                ticker="TECH1",
                side="LONG",
                shares=(0.225 * nav) / price,
                entry_price=price,
                current_price=price,
                sector="Technology",
            ),
            Position(
                ticker="TECH2",
                side="LONG",
                shares=(0.225 * nav) / price,
                entry_price=price,
                current_price=price,
                sector="Technology",
            ),
            Position(
                ticker="FIN1",
                side="LONG",
                shares=(0.10 * nav) / price,
                entry_price=price,
                current_price=price,
                sector="Financials",
            ),
        ]
        portfolio = Portfolio(positions=positions, nav=nav)
        tickers = [p.ticker for p in positions]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        bench = {"Technology": 0.30, "Financials": 0.05}
        report = style_tilts(
            portfolio,
            returns_df,
            factors,
            benchmark_weights=bench,
        )
        tech = next(s for s in report.sectors if s.sector == "Technology")
        assert tech.portfolio_weight == pytest.approx(0.45, abs=1e-9)
        assert tech.benchmark_weight == 0.30
        assert tech.active_weight == pytest.approx(0.15, abs=1e-9)

    def test_no_benchmark_active_weight_none(self):
        factors = _factor_returns()
        tickers = ["A", "B"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        report = style_tilts(portfolio, returns_df, factors, benchmark_weights=None)
        for s in report.sectors:
            assert s.benchmark_weight is None
            assert s.active_weight is None


# ---------------------------------------------------------------------------
# Crowding
# ---------------------------------------------------------------------------


class TestCrowding:
    def test_placeholder_mode_emits_zeros(self):
        factors = _factor_returns()
        tickers = ["A", "B", "C"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        report = style_tilts(portfolio, returns_df, factors, crowding_data=None)
        assert len(report.crowding) == 3
        for row in report.crowding:
            assert row.score == 0.0
            assert row.source == "placeholder"

    def test_external_scores_pass_through(self):
        factors = _factor_returns()
        tickers = ["A", "B", "C"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.0, "HML": 0.0} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        portfolio = _equal_weight_portfolio(
            longs=[(t, "Technology") for t in tickers],
            shorts=[],
        )
        crowding_in = {"A": 0.9, "B": 0.4}  # C intentionally missing
        report = style_tilts(
            portfolio,
            returns_df,
            factors,
            crowding_data=crowding_in,
        )
        by_ticker = {c.ticker: c for c in report.crowding}
        assert by_ticker["A"].score == pytest.approx(0.9)
        assert by_ticker["A"].source == "external"
        assert by_ticker["B"].score == pytest.approx(0.4)
        assert by_ticker["B"].source == "external"
        # Ticker absent from crowding_data falls back to placeholder.
        assert by_ticker["C"].score == 0.0
        assert by_ticker["C"].source == "placeholder"


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_report_types(self):
        factors = _factor_returns()
        tickers = [f"T{i}" for i in range(6)]
        sectors = ["Technology", "Healthcare", "Financials"]
        beta_map = {t: {"MKT": 1.0, "SMB": 0.2, "HML": -0.1} for t in tickers}
        returns_df = _make_returns_df(tickers, factors, beta_map)
        longs = [(tickers[i], sectors[i % 3]) for i in range(4)]
        shorts = [(tickers[i], sectors[i % 3]) for i in range(4, 6)]
        portfolio = _equal_weight_portfolio(longs=longs, shorts=shorts)
        report = style_tilts(portfolio, returns_df, factors)

        assert isinstance(report, StyleTiltsReport)
        assert all(isinstance(s, StyleTilt) for s in report.style)
        assert all(isinstance(s, SectorTilt) for s in report.sectors)
        assert all(isinstance(c, CrowdingScore) for c in report.crowding)
        assert {s.factor for s in report.style} == {"MKT", "SMB", "HML"}
        assert set(sectors).issubset({s.sector for s in report.sectors})
