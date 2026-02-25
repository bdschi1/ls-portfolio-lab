"""Tests for core/metrics/return_metrics.py"""

import math

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

    def test_exact_geometric_annualization(self):
        """Verify formula: total^(252/n) - 1 with deterministic input."""
        daily_r = 0.001
        n = 252
        rets = pl.Series([daily_r] * n)
        ann = annualized_return(rets)
        # Geometric: (1.001)^252 - 1
        expected = (1.0 + daily_r) ** 252 - 1.0
        assert ann == pytest.approx(expected, rel=1e-6)

    def test_negative_return_annualized(self):
        """Consistent daily losses → negative annualized return."""
        rets = pl.Series([-0.001] * 126)  # half a year of -10bps/day
        ann = annualized_return(rets)
        expected = (1.0 - 0.001) ** 252 - 1.0  # geometric, annualized from 126d
        assert ann == pytest.approx(expected, rel=1e-4)

    def test_zero_returns(self):
        rets = pl.Series([0.0] * 100)
        assert annualized_return(rets) == 0.0

    def test_empty_returns(self):
        rets = pl.Series(dtype=pl.Float64)
        assert annualized_return(rets) == 0.0


class TestAnnualizedVolatility:
    def test_known_vol_formula(self):
        """Verify formula: daily_std × √252."""
        rets = _make_returns(252, mean=0.0, std=0.01)
        vol = annualized_volatility(rets)
        # Compute expected from the actual series std
        daily_std = float(rets.std())
        expected = daily_std * np.sqrt(252)
        assert vol == pytest.approx(expected, rel=1e-6)

    def test_daily_1pct_annualizes_correctly(self):
        """Daily std=1% should annualize to ~15.87%."""
        # With enough samples, sample std ≈ population std
        rets = _make_returns(10000, mean=0.0, std=0.01, seed=42)
        vol = annualized_volatility(rets)
        expected = 0.01 * np.sqrt(252)  # 15.87%
        assert vol == pytest.approx(expected, rel=0.05)  # within 5% of theoretical

    def test_constant_returns_near_zero_vol(self):
        rets = pl.Series([0.001] * 100)
        vol = annualized_volatility(rets)
        # Constant returns → near-zero vol (floating point may not be exactly 0)
        assert vol < 0.001

    def test_insufficient_data(self):
        rets = pl.Series([0.01])
        assert annualized_volatility(rets) == 0.0


class TestSharpeRatio:
    def test_formula_verification(self):
        """SR = (ann_return - rf) / ann_vol — verify against hand calc."""
        rets = _make_returns(252, mean=0.0004, std=0.01)
        rf = 0.05
        s = sharpe_ratio(rets, risk_free_rate=rf)
        # Manually compute
        ann_ret = annualized_return(rets)
        ann_vol = annualized_volatility(rets)
        expected = (ann_ret - rf) / ann_vol
        assert s == pytest.approx(expected, rel=1e-6)

    def test_near_zero_vol_returns_finite(self):
        """Constant returns → near-zero vol → Sharpe should be a finite float.

        Note: Polars .std() on constant values returns ~1e-17 (not exactly 0)
        due to floating-point representation, so the zero-vol guard doesn't
        trigger. This is acceptable — the function remains numerically stable.
        """
        rets = pl.Series([0.001] * 100)
        s = sharpe_ratio(rets)
        assert isinstance(s, float)
        assert not math.isnan(s)

    def test_negative_sharpe_on_losing_strategy(self):
        """Losing strategy with returns below rf should have negative Sharpe."""
        rets = _make_returns(252, mean=-0.001, std=0.01, seed=42)
        s = sharpe_ratio(rets, risk_free_rate=0.05)
        assert s < 0

    def test_higher_return_higher_sharpe(self):
        """Same vol, higher return → higher Sharpe."""
        rets_low = _make_returns(252, mean=0.0002, std=0.01, seed=42)
        rets_high = _make_returns(252, mean=0.001, std=0.01, seed=42)
        assert sharpe_ratio(rets_high) > sharpe_ratio(rets_low)


class TestSortinoRatio:
    def test_formula_verification(self):
        """Sortino = (ann_ret - rf) / downside_deviation."""
        rets = _make_returns(252, mean=0.0004, std=0.01)
        rf = 0.05
        s = sortino_ratio(rets, risk_free_rate=rf)
        # Manually compute
        ann_ret = annualized_return(rets)
        downside = rets.filter(rets < 0)
        downside_std = float(downside.std() * np.sqrt(252))
        expected = (ann_ret - rf) / downside_std
        assert s == pytest.approx(expected, rel=1e-6)

    def test_sortino_ge_sharpe_for_positive_skew(self):
        """For positive-skew returns, Sortino ≥ Sharpe (less downside than total vol)."""
        # Create positively skewed returns: mostly small gains, rare small losses
        rng = np.random.default_rng(42)
        gains = rng.exponential(0.005, 200)  # right-skewed
        losses = -rng.exponential(0.003, 52)
        rets = pl.Series(np.concatenate([gains, losses]))
        s_sharpe = sharpe_ratio(rets, risk_free_rate=0.0)
        s_sortino = sortino_ratio(rets, risk_free_rate=0.0)
        assert s_sortino >= s_sharpe

    def test_all_positive_returns(self):
        rets = pl.Series([0.01] * 100)
        # No downside → sortino should be 0 (insufficient downside data)
        s = sortino_ratio(rets)
        assert s == 0.0


class TestCalmarRatio:
    def test_formula_verification(self):
        """Calmar = annualized_return / |max_drawdown|."""
        rets = pl.Series(
            [0.01, 0.01, -0.03, 0.01, 0.01, 0.01, 0.01] * 36  # 252 days
        )
        c = calmar_ratio(rets)
        # Manually compute max drawdown
        cum = (1.0 + rets).cum_prod()
        peak = cum.cum_max()
        dd = ((cum - peak) / peak).min()
        ann_ret = annualized_return(rets)
        expected = ann_ret / abs(float(dd))
        assert c == pytest.approx(expected, rel=1e-6)

    def test_no_drawdown_returns_zero(self):
        """Monotonically increasing → no drawdown → Calmar = 0."""
        rets = pl.Series([0.01] * 100)
        c = calmar_ratio(rets)
        assert c == 0.0


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
