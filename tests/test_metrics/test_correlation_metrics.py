"""Tests for core/metrics/correlation_metrics.py — including partial correlations."""

import numpy as np
import polars as pl
import pytest

from core.metrics.correlation_metrics import (
    average_pairwise_correlation,
    correlation_summary,
    long_short_correlation,
    most_partially_correlated_pairs,
    pairwise_correlation_matrix,
    partial_correlation_matrix,
)

# --- Helpers ---


def _make_returns_df(
    n_days: int = 100,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic correlated returns DataFrame."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    rng = np.random.default_rng(seed)
    n = len(tickers)

    # Create correlated returns via a factor model
    factor = rng.normal(0, 0.01, n_days)
    data = {}
    for i, t in enumerate(tickers):
        idio = rng.normal(0, 0.005, n_days)
        beta = 0.5 + 0.3 * i / n  # varying factor loading
        data[t] = factor * beta + idio

    return pl.DataFrame(data)


# --- Existing function tests ---


class TestPairwiseCorrelation:
    def test_returns_correct_shape(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT", "GOOG"]
        corr = pairwise_correlation_matrix(df, tickers)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT"]
        corr = pairwise_correlation_matrix(df, tickers)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_symmetric(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT", "GOOG"]
        corr = pairwise_correlation_matrix(df, tickers)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_missing_tickers_skipped(self):
        df = _make_returns_df()
        corr = pairwise_correlation_matrix(df, ["AAPL", "MISSING"])
        assert corr.size == 0  # only 1 available, need 2

    def test_few_observations_returns_identity(self):
        df = _make_returns_df(n_days=5)
        tickers = ["AAPL", "MSFT"]
        corr = pairwise_correlation_matrix(df, tickers)
        np.testing.assert_allclose(corr, np.eye(2))


class TestAveragePairwise:
    def test_positive_for_correlated_assets(self):
        df = _make_returns_df()
        avg = average_pairwise_correlation(df, ["AAPL", "MSFT", "GOOG"])
        assert avg > 0  # factor model creates positive correlation

    def test_formula_is_mean_of_upper_triangle(self):
        """avg_pairwise = mean(upper triangle of correlation matrix)."""
        df = _make_returns_df(n_days=500)
        tickers = ["AAPL", "MSFT", "GOOG"]
        avg = average_pairwise_correlation(df, tickers)
        # Manually compute
        corr = pairwise_correlation_matrix(df, tickers)
        n = corr.shape[0]
        upper = corr[np.triu_indices(n, k=1)]
        expected = float(np.mean(upper))
        assert avg == pytest.approx(expected, rel=1e-6)

    def test_uncorrelated_assets_near_zero(self):
        """Independent assets (no common factor) → avg pairwise ≈ 0."""
        rng = np.random.default_rng(99)
        df = pl.DataFrame({
            "A": rng.normal(0, 0.01, 1000).tolist(),
            "B": rng.normal(0, 0.01, 1000).tolist(),
            "C": rng.normal(0, 0.01, 1000).tolist(),
        })
        avg = average_pairwise_correlation(df, ["A", "B", "C"])
        assert abs(avg) < 0.10  # should be near zero


# --- Partial correlation tests (Paleologo Insight 4.2) ---


class TestPartialCorrelation:
    def test_returns_correct_shape(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT", "GOOG"]
        pcorr = partial_correlation_matrix(df, tickers)
        assert pcorr.shape == (3, 3)

    def test_diagonal_is_one(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT", "GOOG"]
        pcorr = partial_correlation_matrix(df, tickers)
        np.testing.assert_allclose(np.diag(pcorr), 1.0, atol=1e-10)

    def test_symmetric(self):
        df = _make_returns_df()
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        pcorr = partial_correlation_matrix(df, tickers)
        np.testing.assert_allclose(pcorr, pcorr.T, atol=1e-10)

    def test_partial_lower_than_pairwise_with_common_factor(self):
        """With a common factor, partial correlations should generally
        be lower in magnitude than pairwise correlations."""
        df = _make_returns_df(n_days=500, seed=42)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

        corr = pairwise_correlation_matrix(df, tickers)
        pcorr = partial_correlation_matrix(df, tickers)

        # Average absolute partial correlation should be lower
        n = corr.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        avg_abs_corr = np.mean(np.abs(corr[upper_idx]))
        avg_abs_pcorr = np.mean(np.abs(pcorr[upper_idx]))

        assert avg_abs_pcorr < avg_abs_corr

    def test_three_asset_analytic_formula(self):
        """For 3 assets, partial corr(A,B|C) has a known closed form:
        ρ_AB|C = (ρ_AB - ρ_AC·ρ_BC) / √((1-ρ_AC²)(1-ρ_BC²))
        Validate our precision-matrix implementation against this."""
        # Use large N for stable correlations
        df = _make_returns_df(n_days=2000, seed=42)
        tickers = ["AAPL", "MSFT", "GOOG"]

        corr = pairwise_correlation_matrix(df, tickers)
        pcorr = partial_correlation_matrix(df, tickers)

        # Extract pairwise correlations
        r_ab = corr[0, 1]  # AAPL-MSFT
        r_ac = corr[0, 2]  # AAPL-GOOG
        r_bc = corr[1, 2]  # MSFT-GOOG

        # Analytic partial correlation: ρ(AAPL,MSFT | GOOG)
        expected_partial_ab = (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
        assert pcorr[0, 1] == pytest.approx(expected_partial_ab, abs=0.01)

        # Analytic partial: ρ(AAPL,GOOG | MSFT)
        expected_partial_ac = (r_ac - r_ab * r_bc) / np.sqrt((1 - r_ab**2) * (1 - r_bc**2))
        assert pcorr[0, 2] == pytest.approx(expected_partial_ac, abs=0.01)

    def test_two_assets_equals_pairwise(self):
        """With only 2 assets, partial = pairwise (nothing to control for)."""
        df = _make_returns_df(n_days=500)
        tickers = ["AAPL", "MSFT"]

        corr = pairwise_correlation_matrix(df, tickers)
        pcorr = partial_correlation_matrix(df, tickers)

        # For 2 assets, partial correlation = regular correlation
        np.testing.assert_allclose(pcorr[0, 1], corr[0, 1], atol=1e-6)

    def test_missing_tickers(self):
        df = _make_returns_df()
        pcorr = partial_correlation_matrix(df, ["AAPL", "MISSING"])
        assert pcorr.size == 0

    def test_few_observations_returns_identity(self):
        df = _make_returns_df(n_days=5)
        tickers = ["AAPL", "MSFT"]
        pcorr = partial_correlation_matrix(df, tickers)
        np.testing.assert_allclose(pcorr, np.eye(2))

    def test_values_bounded(self):
        """Partial correlations should be in [-1, 1]."""
        df = _make_returns_df(n_days=500)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        pcorr = partial_correlation_matrix(df, tickers)
        assert np.all(pcorr >= -1.0 - 1e-10)
        assert np.all(pcorr <= 1.0 + 1e-10)


class TestLongShortCorrelation:
    def test_identical_books_correlation_one(self):
        """If long and short books have identical returns, correlation ≈ 1."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 200).tolist()
        df = pl.DataFrame({"A": rets, "B": rets})
        corr = long_short_correlation(df, long_tickers=["A"], short_tickers=["B"])
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_uncorrelated_books(self):
        """Independent long/short books → correlation near 0."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame({
            "LONG1": rng.normal(0, 0.01, 500).tolist(),
            "SHORT1": rng.normal(0, 0.01, 500).tolist(),
        })
        corr = long_short_correlation(df, long_tickers=["LONG1"], short_tickers=["SHORT1"])
        assert abs(corr) < 0.15

    def test_missing_tickers_returns_zero(self):
        df = _make_returns_df()
        corr = long_short_correlation(df, long_tickers=["MISSING"], short_tickers=["ALSO_MISSING"])
        assert corr == 0.0


class TestCorrelationSummary:
    def test_returns_all_keys(self):
        df = _make_returns_df(n_days=200)
        summary = correlation_summary(df, long_tickers=["AAPL", "MSFT"], short_tickers=["GOOG", "AMZN"])
        assert set(summary.keys()) == {"avg_pairwise", "avg_long_corr", "avg_short_corr", "long_short_corr"}

    def test_all_values_bounded(self):
        df = _make_returns_df(n_days=200)
        summary = correlation_summary(df, long_tickers=["AAPL", "MSFT"], short_tickers=["GOOG", "AMZN"])
        for v in summary.values():
            assert -1.0 <= v <= 1.0


class TestMostPartiallyCorrelatedPairs:
    def test_returns_pairs(self):
        df = _make_returns_df(n_days=500)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        pairs = most_partially_correlated_pairs(df, tickers, top_n=3)
        assert len(pairs) <= 3
        assert all(len(p) == 3 for p in pairs)

    def test_sorted_by_absolute_value(self):
        df = _make_returns_df(n_days=500)
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        pairs = most_partially_correlated_pairs(df, tickers, top_n=10)
        abs_vals = [abs(p[2]) for p in pairs]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_empty_on_missing_tickers(self):
        df = _make_returns_df()
        pairs = most_partially_correlated_pairs(df, ["MISSING1", "MISSING2"])
        assert pairs == []
