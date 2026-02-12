"""Tests for core/metrics/return_metrics.py"""

import numpy as np
import polars as pl
import pytest

from core.metrics.return_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_return,
    daily_simple_returns,
    information_ratio,
    portfolio_daily_returns,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)

# --- Helpers ---


def _make_returns(n: int = 252, mean: float = 0.0004, std: float = 0.01, seed: int = 42) -> pl.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(mean, std, n)
    return pl.Series("returns", returns)


# --- Tests ---


class TestDailyReturns:
    def test_simple_returns_length(self):
        prices = pl.Series([100, 105, 103, 108])
        rets = daily_simple_returns(prices)
        assert rets.len() == 3

    def test_simple_returns_values(self):
        prices = pl.Series([100.0, 110.0, 99.0])
        rets = daily_simple_returns(prices)
        assert rets[0] == pytest.approx(0.10)
        assert rets[1] == pytest.approx(-0.10, abs=0.001)


class TestAnnualizedReturn:
    def test_positive_return(self):
        # Use deterministic positive returns to guarantee positive annualized return
        rets = pl.Series([0.001] * 252)  # 0.1% daily → ~28% annualized
        ann = annualized_return(rets)
        assert ann > 0.20

    def test_zero_returns(self):
        rets = pl.Series([0.0] * 100)
        assert annualized_return(rets) == 0.0

    def test_empty_returns(self):
        rets = pl.Series(dtype=pl.Float64)
        assert annualized_return(rets) == 0.0


class TestAnnualizedVolatility:
    def test_known_vol(self):
        # Daily std of 1% → annualized ~15.87%
        rets = _make_returns(252, mean=0.0, std=0.01)
        vol = annualized_volatility(rets)
        assert 0.10 < vol < 0.25

    def test_constant_returns_near_zero_vol(self):
        rets = pl.Series([0.001] * 100)
        vol = annualized_volatility(rets)
        # Constant returns → near-zero vol (floating point may not be exactly 0)
        assert vol < 0.001

    def test_insufficient_data(self):
        rets = pl.Series([0.01])
        assert annualized_volatility(rets) == 0.0


class TestSharpeRatio:
    def test_positive_sharpe(self):
        rets = _make_returns(252, mean=0.0004, std=0.01)
        s = sharpe_ratio(rets, risk_free_rate=0.05)
        assert s != 0.0

    def test_constant_returns_near_zero_vol(self):
        rets = pl.Series([0.001] * 100)
        # Constant returns have near-zero std → Sharpe will be extreme or 0
        # The function should handle this gracefully (not crash)
        s = sharpe_ratio(rets)
        assert isinstance(s, float)


class TestSortinoRatio:
    def test_positive_sortino(self):
        rets = _make_returns(252, mean=0.0004, std=0.01)
        s = sortino_ratio(rets, risk_free_rate=0.05)
        assert s != 0.0

    def test_all_positive_returns(self):
        rets = pl.Series([0.01] * 100)
        # No downside → sortino should be 0 (insufficient downside data)
        s = sortino_ratio(rets)
        assert s == 0.0


class TestCalmarRatio:
    def test_positive_calmar(self):
        # Use deterministic positive returns with a known drawdown
        rets = pl.Series(
            [0.01, 0.01, -0.03, 0.01, 0.01, 0.01, 0.01] * 36  # 252 days
        )
        c = calmar_ratio(rets)
        assert c != 0.0  # has positive return and drawdown


class TestInformationRatio:
    def test_ir_same_returns(self):
        rets = _make_returns(252)
        ir = information_ratio(rets, rets)
        # Same returns → zero excess → IR ≈ 0
        assert abs(ir) < 0.1

    def test_ir_with_clear_outperformance(self):
        # Portfolio always beats benchmark by 0.001 per day
        bench = _make_returns(252, mean=0.0, std=0.01, seed=42)
        port = bench + 0.001  # deterministic outperformance
        ir = information_ratio(port, bench)
        # Clear outperformance → positive IR
        assert ir > 0


class TestTrackingError:
    def test_same_returns_zero_te(self):
        rets = _make_returns(252)
        te = tracking_error(rets, rets)
        assert te == pytest.approx(0.0, abs=0.001)


class TestPortfolioDailyReturns:
    def test_single_position(self):
        df = pl.DataFrame({
            "date": list(range(10)),
            "AAPL": [0.01, -0.02, 0.005, 0.01, -0.005, 0.02, -0.01, 0.005, 0.01, -0.005],
        })
        weights = {"AAPL": 1.0}
        port_rets = portfolio_daily_returns(weights, df)
        assert port_rets.len() == 10
        assert port_rets[0] == pytest.approx(0.01)

    def test_two_positions(self):
        df = pl.DataFrame({
            "date": list(range(5)),
            "AAPL": [0.01, -0.02, 0.01, 0.01, -0.01],
            "MSFT": [0.02, 0.01, -0.01, 0.005, 0.01],
        })
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        port_rets = portfolio_daily_returns(weights, df)
        expected_first = 0.6 * 0.01 + 0.4 * 0.02
        assert port_rets[0] == pytest.approx(expected_first)

    def test_short_weight(self):
        df = pl.DataFrame({
            "date": list(range(3)),
            "AAPL": [0.01, -0.01, 0.02],
            "HCA": [0.02, 0.01, -0.01],
        })
        # Long AAPL, short HCA
        weights = {"AAPL": 0.5, "HCA": -0.3}
        port_rets = portfolio_daily_returns(weights, df)
        expected_first = 0.5 * 0.01 + (-0.3) * 0.02
        assert port_rets[0] == pytest.approx(expected_first)


class TestCumulativeReturn:
    def test_cumulative(self):
        rets = pl.Series([0.01, 0.02, -0.01])
        cum = cumulative_return(rets)
        assert cum[0] == pytest.approx(1.01)
        assert cum[1] == pytest.approx(1.01 * 1.02)
        assert cum[2] == pytest.approx(1.01 * 1.02 * 0.99)
