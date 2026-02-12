"""Tests for core/metrics/exposure_metrics.py"""

import pytest

from core.metrics.exposure_metrics import (
    check_exposure_limits,
    concentration_hhi,
    exposure_summary,
    net_beta_exposure,
    top_n_concentration,
)
from core.portfolio import Portfolio, Position


def _make_portfolio() -> Portfolio:
    return Portfolio(
        positions=[
            Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0,
                     current_price=150.0, sector="Tech", subsector="Hardware"),
            Position(ticker="MSFT", side="LONG", shares=50, entry_price=300.0,
                     current_price=300.0, sector="Tech", subsector="Software"),
            Position(ticker="UNH", side="SHORT", shares=40, entry_price=500.0,
                     current_price=500.0, sector="Healthcare", subsector="HMO"),
            Position(ticker="HCA", side="SHORT", shares=60, entry_price=250.0,
                     current_price=250.0, sector="Healthcare", subsector="Hospitals"),
        ],
        nav=100000.0,
    )


class TestNetBetaExposure:
    def test_basic(self):
        p = _make_portfolio()
        betas = {"AAPL": 1.2, "MSFT": 1.0, "UNH": 0.8, "HCA": 1.1}
        nbe = net_beta_exposure(p, betas)
        # Long: AAPL wt=0.15 * 1.2 + MSFT wt=0.15 * 1.0
        # Short: UNH wt=-0.20 * 0.8 + HCA wt=-0.15 * 1.1
        expected = (0.15 * 1.2 + 0.15 * 1.0) + (-0.20 * 0.8 + (-0.15) * 1.1)
        assert nbe == pytest.approx(expected, abs=0.01)


class TestConcentration:
    def test_hhi_equal_weights(self):
        """4 equal positions → HHI = 4 * (0.25)² = 0.25"""
        p = Portfolio(
            positions=[
                Position(ticker=f"T{i}", side="LONG", shares=100,
                         entry_price=100.0, current_price=100.0)
                for i in range(4)
            ],
            nav=40000.0,  # each position = 25% of NAV
        )
        hhi = concentration_hhi(p)
        assert hhi == pytest.approx(4 * 0.25**2, abs=0.01)

    def test_top_n(self):
        p = _make_portfolio()
        top5 = top_n_concentration(p, 5)
        # All 4 positions → top 5 = sum of all weights
        total = sum(pos.abs_weight_in(p.nav) for pos in p.positions)
        assert top5 == pytest.approx(total)


class TestExposureLimits:
    def test_no_violations(self):
        p = _make_portfolio()
        warnings = check_exposure_limits(
            p, max_sector_net=1.0, max_gross=5.0, max_single_position=0.50,
        )
        assert len(warnings) == 0

    def test_sector_violation(self):
        p = _make_portfolio()
        warnings = check_exposure_limits(p, max_sector_net=0.01)  # very tight
        assert any("Sector" in w for w in warnings)

    def test_beta_violation(self):
        p = _make_portfolio()
        betas = {"AAPL": 2.0, "MSFT": 2.0, "UNH": 0.5, "HCA": 0.5}
        warnings = check_exposure_limits(p, betas=betas, max_net_beta=0.01)
        assert any("beta" in w.lower() for w in warnings)


class TestExposureSummary:
    def test_summary_keys(self):
        p = _make_portfolio()
        summary = exposure_summary(p)
        assert "gross_exposure" in summary
        assert "net_exposure" in summary
        assert "hhi" in summary
        assert "top_5_concentration" in summary
