"""Tests for core/metrics/quality_score.py — composite portfolio quality scoring.

Tests validate:
1. Each subscore function in isolation (known inputs → expected outputs)
2. Letter grade boundaries
3. The full compute_quality_score pipeline with and without returns data
4. Weight normalization and custom weights
5. Edge cases: negative Sharpe, zero vol, all-long, infinite L/S ratio
"""

import math

import numpy as np
import polars as pl
import pytest

from core.metrics.quality_score import (
    DEFAULT_WEIGHTS,
    PortfolioQualityScore,
    _letter_grade,
    _score_alpha,
    _score_diversification,
    _score_drawdown_resilience,
    _score_exposure_balance,
    _score_sharpe,
    _score_tail_risk,
    compute_quality_score,
)
from core.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Subscore unit tests — known inputs, exact expected outputs
# ---------------------------------------------------------------------------


class TestScoreSharpe:
    def test_zero_sharpe(self):
        """SR=0 → slight credit (10 from the linear branch)."""
        s = _score_sharpe(0.0)
        assert s == pytest.approx(10.0)

    def test_negative_sharpe_floors_at_zero(self):
        """SR=-2 → 10 + (-2)*10 = -10 → clamped to 0."""
        s = _score_sharpe(-2.0)
        assert s == 0.0

    def test_slightly_negative(self):
        """SR=-0.5 → 10 + (-0.5)*10 = 5."""
        s = _score_sharpe(-0.5)
        assert s == pytest.approx(5.0)

    def test_sharpe_1(self):
        """SR=1.0 → 100*(1 - exp(-0.8)) ≈ 55.07."""
        expected = 100.0 * (1.0 - math.exp(-0.8))
        assert _score_sharpe(1.0) == pytest.approx(expected, abs=0.1)

    def test_sharpe_2(self):
        """SR=2.0 → 100*(1 - exp(-1.6)) ≈ 79.81."""
        expected = 100.0 * (1.0 - math.exp(-1.6))
        assert _score_sharpe(2.0) == pytest.approx(expected, abs=0.1)

    def test_high_sharpe_caps_at_100(self):
        """Very high Sharpe should asymptote at 100."""
        s = _score_sharpe(10.0)
        assert 99.0 < s <= 100.0

    def test_monotonically_increasing(self):
        """Higher SR → higher score."""
        scores = [_score_sharpe(sr) for sr in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


class TestScoreDrawdownResilience:
    def test_negative_sharpe_returns_floor(self):
        assert _score_drawdown_resilience(-1.0, 0.15) == 10.0

    def test_zero_vol_returns_floor(self):
        assert _score_drawdown_resilience(1.0, 0.0) == 10.0

    def test_high_sharpe_low_vol_scores_well(self):
        """SR=2.0, vol=10% → small E[DD], low time in DD → high score."""
        s = _score_drawdown_resilience(2.0, 0.10)
        assert s > 50.0

    def test_low_sharpe_high_vol_scores_poorly(self):
        """SR=0.3, vol=25% → large E[DD], lots of time in DD → low score."""
        s = _score_drawdown_resilience(0.3, 0.25)
        assert s < 40.0


class TestScoreAlpha:
    def test_dsr_zero(self):
        assert _score_alpha(0.0) == pytest.approx(0.0)

    def test_dsr_half(self):
        """DSR=0.5 → score=50."""
        assert _score_alpha(0.5) == pytest.approx(50.0)

    def test_dsr_one(self):
        """DSR=1.0 → score=100."""
        assert _score_alpha(1.0) == pytest.approx(100.0)

    def test_dsr_over_one_capped(self):
        """DSR > 1.0 capped at 100."""
        assert _score_alpha(1.5) == 100.0

    def test_negative_dsr_floors(self):
        assert _score_alpha(-0.5) == 0.0


class TestScoreDiversification:
    def test_zero_correlation(self):
        """avg_corr=0 → perfectly diversified → score=100."""
        assert _score_diversification(0.0) == pytest.approx(100.0)

    def test_full_correlation(self):
        """avg_corr=1.0 → no diversification → score=0."""
        assert _score_diversification(1.0) == pytest.approx(0.0)

    def test_moderate_correlation(self):
        """avg_corr=0.3 → score=70."""
        assert _score_diversification(0.3) == pytest.approx(70.0)

    def test_half_correlation(self):
        """avg_corr=0.5 → score=50."""
        assert _score_diversification(0.5) == pytest.approx(50.0)


class TestScoreTailRisk:
    def test_zero_var_returns_default(self):
        assert _score_tail_risk(0.0, 0.0) == 50.0

    def test_thin_tails(self):
        """CVaR/VaR ≈ 1.0 with small absolute VaR → high score."""
        # VaR=0.5%, CVaR=0.55% → ratio ≈ 1.1
        s = _score_tail_risk(0.005, 0.0055)
        assert s > 70.0

    def test_fat_tails(self):
        """CVaR/VaR = 2.5 with large absolute VaR → low score."""
        # VaR=3%, CVaR=7.5% → ratio=2.5
        s = _score_tail_risk(0.03, 0.075)
        assert s < 30.0

    def test_ratio_component(self):
        """Ratio of 1.0 should give ratio_score ≈ 100, ratio of 2.0 gives ≈ 40."""
        # ratio=1.0: 100 - (1-1)*60 = 100
        # ratio=2.0: 100 - (2-1)*60 = 40
        # These are the ratio_score components only (60% weight)
        s_thin = _score_tail_risk(0.001, 0.001)  # ratio=1, small abs
        s_fat = _score_tail_risk(0.001, 0.002)   # ratio=2, small abs
        assert s_thin > s_fat


class TestScoreExposureBalance:
    def test_perfectly_hedged(self):
        """Net beta=0, L/S ratio=1.0, gross=1.5 → high score."""
        s = _score_exposure_balance(0.0, 1.5, 1.0)
        # beta_score=100, ls_score=100, gross_score=80
        # 100*0.45 + 100*0.35 + 80*0.20 = 45 + 35 + 16 = 96
        assert s == pytest.approx(96.0)

    def test_high_net_beta(self):
        """Net beta=0.5 → beta_score=0."""
        s = _score_exposure_balance(0.5, 1.5, 1.0)
        # beta_score=0, ls_score=100, gross_score=80
        # 0*0.45 + 100*0.35 + 80*0.20 = 0 + 35 + 16 = 51
        assert s == pytest.approx(51.0)

    def test_no_shorts_infinite_ratio(self):
        """All-long portfolio → L/S ratio=inf → ls_score=0."""
        s = _score_exposure_balance(0.3, 1.0, float("inf"))
        # beta_score = max(0, 100 - 0.3*200) = 40
        # ls_score = 0 (inf)
        # gross_score = 80 (1.0 in range)
        # 40*0.45 + 0*0.35 + 80*0.20 = 18 + 0 + 16 = 34
        assert s == pytest.approx(34.0)

    def test_over_levered(self):
        """Gross > 2.5 → gross_score=20."""
        s = _score_exposure_balance(0.0, 3.0, 1.0)
        # beta_score=100, ls_score=100, gross_score=20
        # 100*0.45 + 100*0.35 + 20*0.20 = 45 + 35 + 4 = 84
        assert s == pytest.approx(84.0)


# ---------------------------------------------------------------------------
# Letter grade mapping
# ---------------------------------------------------------------------------


class TestLetterGrade:
    @pytest.mark.parametrize("score,expected", [
        (95.0, "A+"),
        (93.0, "A+"),
        (90.0, "A"),
        (87.0, "A"),
        (85.0, "A\u2212"),  # A−
        (83.0, "A\u2212"),
        (80.0, "B+"),
        (78.0, "B+"),
        (75.0, "B"),
        (73.0, "B"),
        (70.0, "B\u2212"),
        (68.0, "B\u2212"),
        (65.0, "C+"),
        (63.0, "C+"),
        (60.0, "C"),
        (58.0, "C"),
        (55.0, "C\u2212"),
        (53.0, "C\u2212"),
        (50.0, "D"),
        (45.0, "D"),
        (40.0, "F"),
        (0.0, "F"),
    ])
    def test_grade_boundaries(self, score: float, expected: str):
        assert _letter_grade(score) == expected

    def test_monotonically_improving(self):
        """Higher score should never produce a worse grade."""
        grade_order = ["F", "D", "C\u2212", "C", "C+", "B\u2212", "B", "B+",
                       "A\u2212", "A", "A+"]
        prev_idx = -1
        for score in range(0, 101):
            grade = _letter_grade(float(score))
            idx = grade_order.index(grade)
            assert idx >= prev_idx, f"Grade went backwards at score={score}"
            prev_idx = idx


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------


class TestDefaultWeights:
    def test_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_weights_positive(self):
        assert all(w > 0 for w in DEFAULT_WEIGHTS.values())

    def test_six_dimensions(self):
        assert len(DEFAULT_WEIGHTS) == 6


# ---------------------------------------------------------------------------
# Full compute_quality_score integration
# ---------------------------------------------------------------------------


def _make_portfolio() -> Portfolio:
    """A balanced L/S portfolio for quality scoring."""
    positions = [
        Position(ticker="AAPL", side="LONG", shares=1000, entry_price=150.0,
                 current_price=160.0, sector="Technology"),
        Position(ticker="MSFT", side="LONG", shares=500, entry_price=300.0,
                 current_price=310.0, sector="Technology"),
        Position(ticker="JNJ", side="LONG", shares=800, entry_price=170.0,
                 current_price=165.0, sector="Health Care"),
        Position(ticker="XOM", side="SHORT", shares=600, entry_price=100.0,
                 current_price=95.0, sector="Energy"),
        Position(ticker="TSLA", side="SHORT", shares=200, entry_price=200.0,
                 current_price=210.0, sector="Technology"),
    ]
    return Portfolio(positions=positions, nav=1_000_000)


def _make_returns_df(n: int = 252) -> pl.DataFrame:
    """Synthetic daily returns for 5 stocks with realistic characteristics."""
    rng = np.random.default_rng(42)
    factor = rng.normal(0.0003, 0.01, n)  # market factor
    return pl.DataFrame({
        "date": list(range(n)),
        "AAPL": (factor * 1.2 + rng.normal(0, 0.005, n)).tolist(),
        "MSFT": (factor * 1.1 + rng.normal(0, 0.005, n)).tolist(),
        "JNJ": (factor * 0.6 + rng.normal(0, 0.004, n)).tolist(),
        "XOM": (factor * 0.9 + rng.normal(0, 0.006, n)).tolist(),
        "TSLA": (factor * 1.5 + rng.normal(0, 0.01, n)).tolist(),
    })


class TestComputeQualityScore:
    def test_returns_valid_score_without_returns(self):
        """Even without returns data, should produce a valid score (exposure-only)."""
        p = _make_portfolio()
        result = compute_quality_score(p)
        assert isinstance(result, PortfolioQualityScore)
        assert 0 <= result.total_score <= 100
        assert result.grade in ["A+", "A", "A\u2212", "B+", "B", "B\u2212",
                                "C+", "C", "C\u2212", "D", "F"]

    def test_returns_valid_score_with_returns(self):
        """With returns data, all 6 dimensions should be populated."""
        p = _make_portfolio()
        df = _make_returns_df()
        betas = {"AAPL": 1.2, "MSFT": 1.1, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.5}
        result = compute_quality_score(p, returns_df=df, betas=betas)
        assert len(result.dimensions) == 6
        for dim in result.dimensions:
            assert 0 <= dim.score <= 100
            assert dim.weight > 0

    def test_six_dimensions_always(self):
        """Should always produce exactly 6 dimensions."""
        p = _make_portfolio()
        result = compute_quality_score(p)
        assert len(result.dimensions) == 6

    def test_weighted_sum_equals_total(self):
        """Total score = Σ(dim.score × dim.weight), clamped to [0, 100]."""
        p = _make_portfolio()
        df = _make_returns_df()
        betas = {"AAPL": 1.2, "MSFT": 1.1, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.5}
        result = compute_quality_score(p, returns_df=df, betas=betas)
        expected = sum(d.score * d.weight for d in result.dimensions)
        expected = min(100.0, max(0.0, expected))
        assert result.total_score == pytest.approx(expected)

    def test_grade_matches_total(self):
        """Grade should correspond to total_score."""
        p = _make_portfolio()
        df = _make_returns_df()
        betas = {"AAPL": 1.2, "MSFT": 1.1, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.5}
        result = compute_quality_score(p, returns_df=df, betas=betas)
        assert result.grade == _letter_grade(result.total_score)

    def test_custom_weights(self):
        """Custom weights should override defaults."""
        p = _make_portfolio()
        df = _make_returns_df()
        betas = {"AAPL": 1.2, "MSFT": 1.1, "JNJ": 0.6, "XOM": 0.9, "TSLA": 1.5}
        # All weight on exposure balance
        custom = {
            "risk_adjusted_return": 0.0,
            "drawdown_resilience": 0.0,
            "alpha_generation": 0.0,
            "diversification": 0.0,
            "tail_risk": 0.0,
            "exposure_balance": 1.0,
        }
        result = compute_quality_score(p, returns_df=df, betas=betas, weights=custom)
        # Total should equal just the exposure balance score
        exp_dim = next(d for d in result.dimensions if d.name == "Exposure Balance")
        assert result.total_score == pytest.approx(exp_dim.score, abs=0.01)

    def test_insufficient_returns_uses_defaults(self):
        """< 30 observations → has_returns=False, uses defaults."""
        p = _make_portfolio()
        short_df = _make_returns_df(n=20)
        result = compute_quality_score(p, returns_df=short_df)
        # Sharpe should be 0 (insufficient data)
        sharpe_dim = next(d for d in result.dimensions if d.name == "Risk-Adj Return")
        # With sharpe=0, _score_sharpe(0) = 10
        assert sharpe_dim.score == pytest.approx(10.0)

    def test_dimension_names(self):
        """Check all expected dimension names are present."""
        p = _make_portfolio()
        result = compute_quality_score(p)
        names = {d.name for d in result.dimensions}
        expected_names = {
            "Risk-Adj Return", "DD Resilience", "Alpha Quality",
            "Diversification", "Tail Risk", "Exposure Balance",
        }
        assert names == expected_names

    def test_dimension_weights_match_defaults(self):
        """With no custom weights, dimension weights should match DEFAULT_WEIGHTS."""
        p = _make_portfolio()
        result = compute_quality_score(p)
        weight_sum = sum(d.weight for d in result.dimensions)
        assert weight_sum == pytest.approx(1.0)
