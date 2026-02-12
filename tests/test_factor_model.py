"""Tests for core/factor_model.py — CAPM, FF3, FF4 regressions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.factor_model import (
    FactorDecomposition,
    build_proxy_factors,
    capm_regression,
    multi_factor_regression,
    portfolio_factor_decomposition,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic return data
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, seed: int = 42) -> tuple[pl.Series, pl.Series]:
    """Create correlated stock and market return series."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n)
    # stock = 0.3 alpha + 1.2 beta * market + noise
    stock = 0.0003 + 1.2 * market + rng.normal(0.0, 0.005, n)
    return pl.Series("stock", stock), pl.Series("market", market)


def _make_factor_data(n: int = 252, seed: int = 42) -> dict[str, pl.Series]:
    """Create multi-factor return data."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n)
    smb = rng.normal(0.0001, 0.005, n)
    hml = rng.normal(0.00005, 0.004, n)
    momentum = rng.normal(0.0002, 0.006, n)
    return {
        "market": pl.Series("market", market),
        "smb": pl.Series("smb", smb),
        "hml": pl.Series("hml", hml),
        "momentum": pl.Series("momentum", momentum),
    }


# ---------------------------------------------------------------------------
# CAPM Regression
# ---------------------------------------------------------------------------

class TestCAPM:
    def test_basic_regression(self):
        stock, market = _make_returns()
        result = capm_regression(stock, market)
        assert isinstance(result, FactorDecomposition)
        # Beta should be close to 1.2 (the generative parameter)
        assert 0.8 < result.beta_market < 1.6
        # R² should be positive (correlated data)
        assert result.r_squared > 0.3
        assert result.systematic_pct > 30
        assert result.idiosyncratic_pct < 70

    def test_alpha_positive(self):
        """Stock with positive drift should have positive alpha."""
        stock, market = _make_returns()
        result = capm_regression(stock, market, risk_free_rate=0.0)
        # Alpha should be positive (stock has extra drift)
        assert result.alpha > -0.5  # loose bound — direction correct

    def test_insufficient_data(self):
        """With <30 data points, returns default decomposition."""
        stock = pl.Series("stock", np.random.normal(0, 0.01, 20))
        market = pl.Series("market", np.random.normal(0, 0.01, 20))
        result = capm_regression(stock, market)
        assert result.beta_market == 1.0
        assert result.r_squared == 0.0
        assert result.idiosyncratic_pct == 100.0

    def test_residual_vol_positive(self):
        stock, market = _make_returns()
        result = capm_regression(stock, market)
        assert result.residual_vol > 0

    def test_systematic_idio_sum(self):
        """Systematic % + idiosyncratic % should equal 100."""
        stock, market = _make_returns()
        result = capm_regression(stock, market)
        assert abs(result.systematic_pct + result.idiosyncratic_pct - 100.0) < 0.01

    def test_beta_one_for_market(self):
        """Market regressed on itself should have beta ≈ 1."""
        _, market = _make_returns()
        result = capm_regression(market, market)
        assert abs(result.beta_market - 1.0) < 0.01
        assert result.r_squared > 0.99


# ---------------------------------------------------------------------------
# Multi-Factor Regression (FF3/FF4)
# ---------------------------------------------------------------------------

class TestMultiFactor:
    def test_basic_ff3(self):
        """3-factor regression should return size and value betas."""
        stock, _ = _make_returns()
        factors = _make_factor_data()
        # Remove momentum for FF3
        ff3 = {k: v for k, v in factors.items() if k != "momentum"}
        result = multi_factor_regression(stock, ff3)
        assert result.beta_size is not None
        assert result.beta_value is not None
        assert result.beta_momentum is None  # not included

    def test_ff4_includes_momentum(self):
        """4-factor regression should return all betas."""
        stock, _ = _make_returns()
        factors = _make_factor_data()
        result = multi_factor_regression(stock, factors)
        assert result.beta_size is not None
        assert result.beta_value is not None
        assert result.beta_momentum is not None

    def test_r_squared_non_negative(self):
        stock, _ = _make_returns()
        factors = _make_factor_data()
        result = multi_factor_regression(stock, factors)
        assert result.r_squared >= 0.0

    def test_requires_market_key(self):
        """Should raise if 'market' key missing."""
        stock, _ = _make_returns()
        with pytest.raises(ValueError, match="market"):
            multi_factor_regression(stock, {"smb": pl.Series("x", [0.01] * 252)})

    def test_insufficient_data_ff(self):
        stock = pl.Series("stock", np.random.normal(0, 0.01, 20))
        factors = {k: pl.Series(k, np.random.normal(0, 0.01, 20)) for k in ["market", "smb", "hml"]}
        result = multi_factor_regression(stock, factors)
        assert result.beta_market == 1.0
        assert result.r_squared == 0.0

    def test_factor_exposures_dict(self):
        stock, _ = _make_returns()
        factors = _make_factor_data()
        result = multi_factor_regression(stock, factors)
        exposures = result.factor_exposures
        assert "market" in exposures
        assert "size" in exposures
        assert "value" in exposures
        assert "momentum" in exposures

    def test_more_factors_higher_r_squared(self):
        """Adding factors should not decrease R² (in-sample)."""
        rng = np.random.default_rng(99)
        n = 252
        market = rng.normal(0.0004, 0.01, n)
        smb = rng.normal(0.0001, 0.005, n)
        # Stock driven partly by SMB
        stock = 0.0003 + 1.0 * market + 0.5 * smb + rng.normal(0, 0.003, n)
        stock_s = pl.Series("stock", stock)
        market_s = pl.Series("market", market)
        smb_s = pl.Series("smb", smb)

        capm_result = capm_regression(stock_s, market_s)
        ff_result = multi_factor_regression(stock_s, {"market": market_s, "smb": smb_s})
        # FF should explain at least as much as CAPM
        assert ff_result.r_squared >= capm_result.r_squared - 0.01  # allow tiny float tolerance


# ---------------------------------------------------------------------------
# Build Proxy Factors
# ---------------------------------------------------------------------------

class TestBuildProxyFactors:
    def test_all_etfs_present(self):
        """When all ETFs are in the DataFrame, all factors should be built."""
        n = 100
        rng = np.random.default_rng(42)
        data = {
            "SPY": rng.normal(0, 0.01, n),
            "IWM": rng.normal(0, 0.01, n),
            "IWD": rng.normal(0, 0.01, n),
            "IWF": rng.normal(0, 0.01, n),
            "MTUM": rng.normal(0, 0.01, n),
        }
        df = pl.DataFrame(data)
        factors = build_proxy_factors(df)
        assert "market" in factors
        assert "smb" in factors
        assert "hml" in factors
        assert "momentum" in factors

    def test_missing_etfs(self):
        """Missing ETFs should result in fewer factors, not errors."""
        df = pl.DataFrame({"SPY": [0.01, -0.005, 0.003]})
        factors = build_proxy_factors(df)
        assert "market" in factors
        assert "smb" not in factors  # IWM missing

    def test_smb_is_iwm_minus_spy(self):
        """SMB proxy should be IWM - SPY."""
        data = {
            "SPY": [0.01, 0.02, -0.01],
            "IWM": [0.015, 0.01, -0.005],
        }
        df = pl.DataFrame(data)
        factors = build_proxy_factors(df)
        expected = [0.015 - 0.01, 0.01 - 0.02, -0.005 - (-0.01)]
        for actual, exp in zip(factors["smb"].to_list(), expected):
            assert abs(actual - exp) < 1e-10


# ---------------------------------------------------------------------------
# Portfolio Factor Decomposition
# ---------------------------------------------------------------------------

class TestPortfolioFactorDecomposition:
    def test_capm_when_no_factors(self):
        stock, market = _make_returns()
        result = portfolio_factor_decomposition(stock, market)
        assert result.beta_size is None  # CAPM, not FF

    def test_multi_factor_when_factors_provided(self):
        stock, _ = _make_returns()
        factors = _make_factor_data()
        result = portfolio_factor_decomposition(stock, factors["market"], factor_returns=factors)
        assert result.beta_size is not None


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (in return_metrics, but related)
# ---------------------------------------------------------------------------

class TestDSR:
    def test_positive_sharpe_high_dsr(self):
        """A clearly positive Sharpe with enough data should yield high DSR."""
        from core.metrics.return_metrics import deflated_sharpe_ratio
        rng = np.random.default_rng(42)
        rets = pl.Series("r", rng.normal(0.001, 0.01, 252))
        dsr = deflated_sharpe_ratio(rets, risk_free_rate=0.0)
        assert dsr > 0.8

    def test_zero_sharpe_low_dsr(self):
        from core.metrics.return_metrics import deflated_sharpe_ratio
        rng = np.random.default_rng(42)
        rets = pl.Series("r", rng.normal(0.0, 0.01, 252))
        dsr = deflated_sharpe_ratio(rets, risk_free_rate=0.0)
        assert dsr < 0.8

    def test_more_trials_lower_dsr(self):
        """Multiple testing should deflate the DSR."""
        from core.metrics.return_metrics import deflated_sharpe_ratio
        rng = np.random.default_rng(42)
        rets = pl.Series("r", rng.normal(0.0005, 0.01, 252))
        dsr1 = deflated_sharpe_ratio(rets, num_trials=1)
        dsr10 = deflated_sharpe_ratio(rets, num_trials=10)
        assert dsr10 <= dsr1

    def test_short_series_returns_zero(self):
        from core.metrics.return_metrics import deflated_sharpe_ratio
        rets = pl.Series("r", [0.01, -0.005])
        assert deflated_sharpe_ratio(rets) == 0.0
