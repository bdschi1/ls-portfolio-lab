"""Tests for core.metrics.sharpe_inference — Sharpe ratio statistical inference."""

import math

import numpy as np
import pytest

from core.metrics.sharpe_inference import (
    adjusted_p_values_bonferroni,
    adjusted_p_values_holm,
    adjusted_p_values_sidak,
    control_for_fdr,
    critical_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    observed_fdr,
    posterior_fdr,
    probabilistic_sharpe_ratio,
    sharpe_ratio_power,
    sharpe_ratio_variance,
)


class TestSharpeRatioVariance:
    """Tests for sharpe_ratio_variance()."""

    def test_gaussian_case_reduces_to_lo_formula(self):
        """With γ₃=0, γ₄=3, ρ=0, should match Lo (2002): (1 + SR²/2) / T."""
        sr, t = 1.0, 252
        var = sharpe_ratio_variance(sr, t, skew=0, kurtosis=3, autocorr=0)
        lo_var = (1 + sr ** 2 / 2) / t
        assert var == pytest.approx(lo_var, rel=1e-10)

    def test_zero_sharpe_gaussian(self):
        """SR=0, Gaussian: variance = 1/T."""
        var = sharpe_ratio_variance(0.0, 100, skew=0, kurtosis=3, autocorr=0)
        assert var == pytest.approx(1.0 / 100, rel=1e-10)

    def test_non_normality_increases_variance(self):
        """Excess kurtosis should increase variance for SR > 0."""
        var_normal = sharpe_ratio_variance(1.0, 252, kurtosis=3)
        var_fat = sharpe_ratio_variance(1.0, 252, kurtosis=6)
        assert var_fat > var_normal

    def test_positive_autocorrelation_increases_variance(self):
        """Positive autocorrelation reduces T_eff → increases variance."""
        var_iid = sharpe_ratio_variance(1.0, 252, autocorr=0)
        var_corr = sharpe_ratio_variance(1.0, 252, autocorr=0.1)
        assert var_corr > var_iid

    def test_negative_autocorrelation_decreases_variance(self):
        """Negative autocorrelation increases T_eff → decreases variance."""
        var_iid = sharpe_ratio_variance(1.0, 252, autocorr=0)
        var_neg = sharpe_ratio_variance(1.0, 252, autocorr=-0.1)
        assert var_neg < var_iid

    def test_single_observation_returns_inf(self):
        assert math.isinf(sharpe_ratio_variance(1.0, 1))

    def test_multiple_trials_concentrates_variance(self):
        """Variance of max(Z_1,...,Z_K) decreases as K grows (max concentrates)."""
        var_1 = sharpe_ratio_variance(1.0, 252, num_trials=1)
        var_10 = sharpe_ratio_variance(1.0, 252, num_trials=10)
        # Var[max] < Var[single] because the maximum concentrates
        assert var_10 < var_1


class TestProbabilisticSharpeRatio:
    """Tests for probabilistic_sharpe_ratio()."""

    def test_psr_half_when_sr_equals_sr0(self):
        """PSR = 0.5 when SR = SR₀."""
        psr = probabilistic_sharpe_ratio(1.0, sr0=1.0, t=252)
        assert psr == pytest.approx(0.5, abs=1e-6)

    def test_psr_approaches_one_for_high_sr(self):
        """PSR → 1 as SR >> SR₀."""
        psr = probabilistic_sharpe_ratio(5.0, sr0=0.0, t=252)
        assert psr > 0.99

    def test_psr_approaches_zero_for_negative_sr(self):
        """PSR → 0 as SR << SR₀."""
        psr = probabilistic_sharpe_ratio(-2.0, sr0=0.0, t=252)
        assert psr < 0.01

    def test_psr_increases_with_sample_size(self):
        """More observations → higher PSR (for positive SR)."""
        psr_short = probabilistic_sharpe_ratio(0.5, sr0=0.0, t=50)
        psr_long = probabilistic_sharpe_ratio(0.5, sr0=0.0, t=500)
        assert psr_long > psr_short

    def test_psr_in_unit_interval(self):
        for sr in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            psr = probabilistic_sharpe_ratio(sr, sr0=0.0, t=252)
            assert 0.0 <= psr <= 1.0


class TestMinTrackRecordLength:
    """Tests for minimum_track_record_length()."""

    def test_higher_sr_needs_shorter_track(self):
        trl_low = minimum_track_record_length(0.5, sr0=0.0)
        trl_high = minimum_track_record_length(2.0, sr0=0.0)
        assert trl_high < trl_low

    def test_returns_inf_when_sr_leq_sr0(self):
        assert math.isinf(minimum_track_record_length(0.0, sr0=0.0))
        assert math.isinf(minimum_track_record_length(0.5, sr0=1.0))

    def test_stricter_alpha_needs_longer_track(self):
        trl_5 = minimum_track_record_length(1.0, alpha=0.05)
        trl_1 = minimum_track_record_length(1.0, alpha=0.01)
        assert trl_1 > trl_5

    def test_positive_result(self):
        trl = minimum_track_record_length(1.0, sr0=0.0)
        assert trl > 0 and not math.isinf(trl)


class TestCriticalSharpeRatio:
    """Tests for critical_sharpe_ratio()."""

    def test_increases_with_fewer_observations(self):
        sr_50 = critical_sharpe_ratio(0.0, 50)
        sr_500 = critical_sharpe_ratio(0.0, 500)
        assert sr_50 > sr_500

    def test_increases_with_more_trials(self):
        sr_1 = critical_sharpe_ratio(0.0, 252, num_trials=1)
        sr_100 = critical_sharpe_ratio(0.0, 252, num_trials=100)
        assert sr_100 > sr_1

    def test_positive_for_zero_null(self):
        sr_crit = critical_sharpe_ratio(0.0, 252)
        assert sr_crit > 0


class TestSharpeRatioPower:
    """Tests for sharpe_ratio_power()."""

    def test_power_increases_with_sample_size(self):
        pow_50 = sharpe_ratio_power(0.0, 1.0, 50)
        pow_500 = sharpe_ratio_power(0.0, 1.0, 500)
        assert pow_500 > pow_50

    def test_power_increases_with_effect_size(self):
        pow_small = sharpe_ratio_power(0.0, 0.3, 252)
        pow_large = sharpe_ratio_power(0.0, 1.5, 252)
        assert pow_large > pow_small

    def test_power_in_unit_interval(self):
        pw = sharpe_ratio_power(0.0, 1.0, 252)
        assert 0.0 <= pw <= 1.0

    def test_power_near_zero_when_sr1_equals_sr0(self):
        """When true SR equals null, power ≈ alpha."""
        pw = sharpe_ratio_power(0.0, 0.0, 252, alpha=0.05)
        assert pw < 0.15


class TestFDR:
    """Tests for posterior_fdr and observed_fdr."""

    def test_pfdr_in_unit_interval(self):
        fdr = posterior_fdr(0.1, 0.05, 0.2)
        assert 0.0 <= fdr <= 1.0

    def test_pfdr_zero_when_all_h1(self):
        """If p_H1 = 1, no false discoveries possible."""
        fdr = posterior_fdr(1.0, 0.05, 0.2)
        assert fdr == pytest.approx(0.0)

    def test_pfdr_increases_with_lower_prior(self):
        fdr_high = posterior_fdr(0.5, 0.05, 0.2)
        fdr_low = posterior_fdr(0.01, 0.05, 0.2)
        assert fdr_low > fdr_high

    def test_ofdr_in_unit_interval(self):
        fdr = observed_fdr(1.0, 0.0, 0.5, 252, 0.1)
        assert 0.0 <= fdr <= 1.0

    def test_control_for_fdr_returns_valid_tuple(self):
        sr, alpha, power = control_for_fdr(0.05, sr1=0.5, p_h1=0.1, t=252)
        assert sr >= 0
        assert 0 < alpha < 1
        assert 0 <= power <= 1


class TestMultipleTesting:
    """Tests for FWER corrections."""

    def test_bonferroni_single_test_unchanged(self):
        p = np.array([0.03])
        adj = adjusted_p_values_bonferroni(p)
        assert adj[0] == pytest.approx(0.03)

    def test_bonferroni_multiplies_by_k(self):
        p = np.array([0.01, 0.02, 0.03])
        adj = adjusted_p_values_bonferroni(p)
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.09])

    def test_bonferroni_caps_at_one(self):
        p = np.array([0.5, 0.6])
        adj = adjusted_p_values_bonferroni(p)
        assert all(a <= 1.0 for a in adj)

    def test_sidak_less_conservative_than_bonferroni(self):
        p = np.array([0.01, 0.02, 0.05, 0.10])
        bonf = adjusted_p_values_bonferroni(p)
        sidak = adjusted_p_values_sidak(p)
        assert all(s <= b + 1e-12 for s, b in zip(sidak, bonf))

    def test_holm_less_conservative_than_bonferroni(self):
        p = np.array([0.01, 0.02, 0.05, 0.10])
        bonf = adjusted_p_values_bonferroni(p)
        holm = adjusted_p_values_holm(p)
        assert all(h <= b + 1e-12 for h, b in zip(holm, bonf))

    def test_holm_preserves_order(self):
        p = np.array([0.05, 0.01, 0.10, 0.03])
        adj = adjusted_p_values_holm(p)
        sorted_adj = adj[np.argsort(p)]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-12

    def test_empty_array(self):
        adj = adjusted_p_values_bonferroni(np.array([]))
        assert len(adj) == 0

    def test_holm_sidak_variant(self):
        p = np.array([0.01, 0.02, 0.05])
        adj_bonf = adjusted_p_values_holm(p, variant="bonferroni")
        adj_sidak = adjusted_p_values_holm(p, variant="sidak")
        assert all(s <= b + 1e-12 for s, b in zip(adj_sidak, adj_bonf))


class TestExpectedMaxSR:
    """Tests for expected_maximum_sharpe_ratio()."""

    def test_single_trial_returns_sr0(self):
        e_max = expected_maximum_sharpe_ratio(1, 0.01, sr0=0.5)
        assert e_max == pytest.approx(0.5)

    def test_increases_with_num_trials(self):
        e2 = expected_maximum_sharpe_ratio(2, 0.01)
        e10 = expected_maximum_sharpe_ratio(10, 0.01)
        e100 = expected_maximum_sharpe_ratio(100, 0.01)
        assert e100 > e10 > e2

    def test_increases_with_variance(self):
        e_low = expected_maximum_sharpe_ratio(10, 0.001)
        e_high = expected_maximum_sharpe_ratio(10, 0.01)
        assert e_high > e_low

    def test_zero_trials(self):
        e = expected_maximum_sharpe_ratio(0, 0.01)
        assert e == pytest.approx(0.0)
