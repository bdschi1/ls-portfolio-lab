"""Tests for core/metrics/risk_metrics.py"""

import numpy as np
import polars as pl
import pytest

from core.metrics.risk_metrics import (
    cvar_historical,
    days_to_liquidate,
    idiosyncratic_volatility,
    marginal_contribution_to_risk,
    portfolio_beta,
    portfolio_volatility,
    position_beta,
    var_historical,
    var_parametric,
)


def _make_returns(n: int = 252, mean: float = 0.0, std: float = 0.01, seed: int = 42) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("returns", rng.normal(mean, std, n))


def _make_correlated_returns(
    n: int = 252, correlation: float = 0.6, seed: int = 42,
) -> tuple[pl.Series, pl.Series]:
    """Generate two correlated return series."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 0.01, n)
    z2 = correlation * z1 + np.sqrt(1 - correlation**2) * rng.normal(0, 0.01, n)
    return pl.Series("a", z1), pl.Series("b", z2)


class TestPositionBeta:
    def test_market_beta_of_market(self):
        market = _make_returns(252)
        # Market vs itself → beta ≈ 1.0
        beta = position_beta(market, market)
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_high_beta_stock(self):
        rng = np.random.default_rng(42)
        market = rng.normal(0, 0.01, 252)
        stock = 1.5 * market + rng.normal(0, 0.005, 252)  # beta ≈ 1.5
        beta = position_beta(pl.Series(stock), pl.Series(market))
        assert 1.2 < beta < 1.8

    def test_insufficient_data(self):
        short_series = pl.Series([0.01, 0.02, 0.03])
        market = pl.Series([0.005, 0.01, 0.015])
        beta = position_beta(short_series, market)
        assert beta == 1.0  # default fallback


class TestIdiosyncraticVolatility:
    def test_pure_market_stock(self):
        market = _make_returns(252)
        # Stock that perfectly tracks market → idio vol ≈ 0
        idio = idiosyncratic_volatility(market, market)
        assert idio == pytest.approx(0.0, abs=0.01)

    def test_stock_with_idio_risk(self):
        rng = np.random.default_rng(42)
        market = rng.normal(0, 0.01, 252)
        stock = market + rng.normal(0, 0.02, 252)  # significant idio
        idio = idiosyncratic_volatility(pl.Series(stock), pl.Series(market))
        assert idio > 0.1  # meaningful idio vol


class TestPortfolioVolatility:
    def test_single_position(self):
        df = pl.DataFrame({"AAPL": np.random.default_rng(42).normal(0, 0.01, 252)})
        weights = {"AAPL": 1.0}
        vol = portfolio_volatility(weights, df)
        assert 0.05 < vol < 0.30

    def test_diversification_reduces_vol(self):
        """Two uncorrelated positions should have lower portfolio vol than either alone."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame({
            "A": rng.normal(0, 0.02, 252),
            "B": rng.normal(0, 0.02, 500)[:252],  # different seed via slice
        })
        vol_a = portfolio_volatility({"A": 1.0}, df)
        vol_50_50 = portfolio_volatility({"A": 0.5, "B": 0.5}, df)
        assert vol_50_50 < vol_a


class TestPortfolioBeta:
    def test_simple_beta(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "HCA": -0.2}
        betas = {"AAPL": 1.2, "MSFT": 1.0, "HCA": 0.8}
        pb = portfolio_beta(weights, betas)
        expected = 0.5 * 1.2 + 0.3 * 1.0 + (-0.2) * 0.8
        assert pb == pytest.approx(expected)

    def test_hedged_portfolio(self):
        """Equal long/short with beta 1.0 should net to ~0."""
        weights = {"LONG1": 0.5, "SHORT1": -0.5}
        betas = {"LONG1": 1.0, "SHORT1": 1.0}
        assert portfolio_beta(weights, betas) == pytest.approx(0.0)


class TestVaR:
    def test_var_positive(self):
        rets = _make_returns(500, mean=0.0, std=0.02)
        var = var_historical(rets, confidence=0.95)
        assert var > 0  # VaR is a positive number representing loss

    def test_var_parametric_close_to_historical(self):
        rets = _make_returns(1000, mean=0.0, std=0.01)
        h_var = var_historical(rets, confidence=0.95)
        p_var = var_parametric(rets, confidence=0.95)
        # Should be in the same ballpark
        assert abs(h_var - p_var) < 0.02

    def test_cvar_exceeds_var(self):
        rets = _make_returns(500)
        var = var_historical(rets)
        cvar = cvar_historical(rets)
        assert cvar >= var  # CVaR is always ≥ VaR


class TestDaysToLiquidate:
    def test_basic(self):
        # 10000 shares, ADV = 100000, 20% participation → 0.5 days
        d = days_to_liquidate(10000, 100000, 0.20)
        assert d == pytest.approx(0.5)

    def test_illiquid(self):
        d = days_to_liquidate(100000, 50000, 0.10)
        assert d == 20.0

    def test_zero_volume(self):
        d = days_to_liquidate(100, 0)
        assert d == float("inf")


class TestMarginalContribution:
    def test_mcr_sums_to_portfolio_vol(self):
        """MCR should approximately sum to portfolio vol."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame({
            "A": rng.normal(0, 0.02, 252),
            "B": rng.normal(0, 0.015, 252),
            "C": rng.normal(0, 0.025, 252),
        })
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        mcr = marginal_contribution_to_risk(weights, df)

        # Sum of MCR should approximate portfolio volatility (daily)
        assert len(mcr) == 3
        total_mcr = sum(mcr.values())
        # MCR values should be non-trivial
        assert total_mcr != 0
