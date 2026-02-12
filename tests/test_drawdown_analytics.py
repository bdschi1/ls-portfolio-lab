"""Tests for core/metrics/drawdown_analytics.py — Bailey & Lopez de Prado framework."""

import math

import pytest

from core.metrics.drawdown_analytics import (
    _normalize_drawdown,
    drawdown_probability,
    drawdown_probability_horizon,
    drawdown_table,
    expected_drawdown_pct,
    expected_drawdown_vol_units,
    expected_recovery_time,
    expected_time_in_drawdown,
    recovery_time_std,
    time_in_drawdown_pct,
)


class TestNormalizeDrawdown:
    def test_basic(self):
        assert _normalize_drawdown(0.10, 0.20) == pytest.approx(0.5)

    def test_zero_sigma(self):
        assert _normalize_drawdown(0.10, 0.0) == float("inf")

    def test_negative_sigma(self):
        assert _normalize_drawdown(0.10, -0.05) == float("inf")


class TestExpectedDrawdownVolUnits:
    def test_sr_1(self):
        # E(D)/σ = 1/(2·1) = 0.5
        assert expected_drawdown_vol_units(1.0) == pytest.approx(0.5)

    def test_sr_2(self):
        # E(D)/σ = 1/(2·2) = 0.25
        assert expected_drawdown_vol_units(2.0) == pytest.approx(0.25)

    def test_sr_half(self):
        # E(D)/σ = 1/(2·0.5) = 1.0
        assert expected_drawdown_vol_units(0.5) == pytest.approx(1.0)

    def test_zero_sr(self):
        assert expected_drawdown_vol_units(0.0) == float("inf")

    def test_negative_sr(self):
        assert expected_drawdown_vol_units(-0.5) == float("inf")


class TestExpectedDrawdownPct:
    def test_sr_1_vol_10(self):
        # E(D) = 0.10 / (2·1) = 0.05 = 5%
        assert expected_drawdown_pct(1.0, 0.10) == pytest.approx(0.05)

    def test_sr_2_vol_20(self):
        # E(D) = 0.20 / (2·2) = 0.05 = 5%
        assert expected_drawdown_pct(2.0, 0.20) == pytest.approx(0.05)

    def test_sr_1_vol_20(self):
        # E(D) = 0.20 / (2·1) = 0.10 = 10%
        assert expected_drawdown_pct(1.0, 0.20) == pytest.approx(0.10)

    def test_zero_sr(self):
        assert expected_drawdown_pct(0.0, 0.10) == float("inf")


class TestDrawdownProbability:
    def test_sr_1_b_sigma_1(self):
        # P = exp(-2·1·1) = exp(-2) ≈ 0.1353
        assert drawdown_probability(1.0, 1.0) == pytest.approx(math.exp(-2), rel=1e-6)

    def test_sr_2_b_sigma_1(self):
        # P = exp(-2·1·2) = exp(-4) ≈ 0.0183
        assert drawdown_probability(1.0, 2.0) == pytest.approx(math.exp(-4), rel=1e-6)

    def test_sr_1_b_sigma_0_5(self):
        # P = exp(-2·0.5·1) = exp(-1) ≈ 0.3679
        assert drawdown_probability(0.5, 1.0) == pytest.approx(math.exp(-1), rel=1e-6)

    def test_zero_drawdown(self):
        # b_σ = 0 means P(DD ≥ 0) = 1 (always at or in drawdown)
        assert drawdown_probability(0.0, 1.0) == 1.0

    def test_zero_sr(self):
        # SR ≤ 0: certain to hit any drawdown
        assert drawdown_probability(1.0, 0.0) == 1.0
        assert drawdown_probability(1.0, -0.5) == 1.0

    def test_large_drawdown_very_small_prob(self):
        # Very deep drawdown with good SR → near zero
        p = drawdown_probability(5.0, 2.0)
        assert p < 1e-8

    def test_overflow_guard(self):
        # Extremely large exponent should return 0, not error
        p = drawdown_probability(1000.0, 2.0)
        assert p == 0.0


class TestDrawdownProbabilityHorizon:
    def test_basic(self):
        # With positive drift, probability should be between 0 and 1
        mu = 0.0005  # daily drift
        sigma = 0.01  # daily vol
        b = 0.05  # 5% drawdown
        T = 252  # 1 year

        p = drawdown_probability_horizon(b, mu, sigma, T)
        assert 0 <= p <= 1

    def test_larger_horizon_higher_prob(self):
        mu = 0.0005
        sigma = 0.01
        b = 0.05

        p_1y = drawdown_probability_horizon(b, mu, sigma, 252)
        p_2y = drawdown_probability_horizon(b, mu, sigma, 504)
        assert p_2y >= p_1y  # longer horizon → higher prob

    def test_deeper_drawdown_lower_prob(self):
        mu = 0.0005
        sigma = 0.01

        p_5 = drawdown_probability_horizon(0.05, mu, sigma, 252)
        p_10 = drawdown_probability_horizon(0.10, mu, sigma, 252)
        assert p_10 < p_5  # deeper drawdown → lower prob

    def test_zero_drawdown(self):
        # b=0 means probability of hitting DD ≥ 0 = 1
        p = drawdown_probability_horizon(0.0, 0.0005, 0.01, 252)
        assert p == 1.0

    def test_zero_sigma(self):
        p = drawdown_probability_horizon(0.05, 0.0005, 0.0, 252)
        assert p == 0.0

    def test_zero_horizon(self):
        p = drawdown_probability_horizon(0.05, 0.0005, 0.01, 0)
        assert p == 0.0

    def test_matches_table_1_approx(self):
        """Approximate validation against Table 1 from the paper.

        For SR=1 (annualized), b_σ=1:
        The stationary probability should be exp(-2) ≈ 0.135.
        The finite-horizon probability for T=252 days (1 year) should be
        somewhat lower than the stationary value.
        """
        # SR=1 annualized → daily: mu = SR * sigma_daily / sqrt(252)
        sigma_daily = 0.01
        sr_annual = 1.0
        mu_daily = sr_annual * sigma_daily / math.sqrt(252)

        # b_σ = 1 → b = 1 * sigma_annual = 0.01 * sqrt(252) ≈ 0.1587
        sigma_annual = sigma_daily * math.sqrt(252)
        b = 1.0 * sigma_annual

        p = drawdown_probability_horizon(b, mu_daily, sigma_daily, 252)
        # Should be less than stationary prob of exp(-2) ≈ 0.135
        assert 0 < p < 0.2


class TestExpectedRecoveryTime:
    def test_sr_2_b_sigma_1(self):
        # E(τ) = 1/2 = 0.5 years
        assert expected_recovery_time(1.0, 2.0) == pytest.approx(0.5)

    def test_sr_1_b_sigma_1(self):
        # E(τ) = 1/1 = 1.0 years
        assert expected_recovery_time(1.0, 1.0) == pytest.approx(1.0)

    def test_sr_1_b_sigma_2(self):
        # E(τ) = 2/1 = 2.0 years
        assert expected_recovery_time(2.0, 1.0) == pytest.approx(2.0)

    def test_zero_sr(self):
        assert expected_recovery_time(1.0, 0.0) == float("inf")

    def test_zero_drawdown(self):
        assert expected_recovery_time(0.0, 1.0) == 0.0


class TestRecoveryTimeStd:
    def test_basic(self):
        # std(τ) = √(1/2) · (1/2) = 0.707 · 0.5 ≈ 0.354
        result = recovery_time_std(1.0, 2.0)
        expected = math.sqrt(1.0 / 2.0) * (1.0 / 2.0)
        assert result == pytest.approx(expected)

    def test_sr_1_b_sigma_1(self):
        # std(τ) = √(1/1) · (1/1) = 1.0
        assert recovery_time_std(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_sr(self):
        assert recovery_time_std(1.0, 0.0) == float("inf")

    def test_zero_drawdown(self):
        assert recovery_time_std(0.0, 1.0) == 0.0


class TestExpectedTimeInDrawdown:
    def test_sr_1(self):
        # 1/(2·1²) = 0.5 = 50% of time in drawdown
        assert expected_time_in_drawdown(1.0) == pytest.approx(0.5)

    def test_sr_2(self):
        # 1/(2·4) = 0.125 = 12.5%
        assert expected_time_in_drawdown(2.0) == pytest.approx(0.125)

    def test_sr_0_5(self):
        # 1/(2·0.25) = 2.0 = 200% (>100%, meaning always in drawdown)
        assert expected_time_in_drawdown(0.5) == pytest.approx(2.0)

    def test_zero_sr(self):
        assert expected_time_in_drawdown(0.0) == float("inf")

    def test_negative_sr(self):
        assert expected_time_in_drawdown(-1.0) == float("inf")


class TestTimeInDrawdownPct:
    def test_sr_1(self):
        assert time_in_drawdown_pct(1.0) == pytest.approx(50.0)

    def test_sr_2(self):
        assert time_in_drawdown_pct(2.0) == pytest.approx(12.5)

    def test_zero_sr(self):
        assert time_in_drawdown_pct(0.0) == float("inf")


class TestDrawdownTable:
    def test_positive_sharpe(self):
        result = drawdown_table(sharpe=1.0, ann_vol=0.10, current_drawdown=0.05)

        assert result["expected_dd_pct"] == pytest.approx(0.05)
        assert result["time_in_dd_pct"] == pytest.approx(50.0)
        assert 0 < result["dd_prob_10pct"] < 1
        assert result["recovery_time_days"] > 0
        assert result["recovery_time_std_days"] > 0

    def test_zero_sharpe(self):
        result = drawdown_table(sharpe=0.0, ann_vol=0.10, current_drawdown=0.05)

        assert result["expected_dd_pct"] == float("inf")
        assert result["time_in_dd_pct"] == float("inf")
        assert result["dd_prob_10pct"] == 1.0

    def test_no_current_drawdown(self):
        result = drawdown_table(sharpe=1.0, ann_vol=0.10, current_drawdown=0.0)
        assert result["recovery_time_days"] == 0.0

    def test_high_sharpe(self):
        result = drawdown_table(sharpe=3.0, ann_vol=0.08)
        assert result["expected_dd_pct"] < 0.02  # very small expected DD
        assert result["time_in_dd_pct"] < 10.0  # less than 10% of time
