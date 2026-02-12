"""Portfolio Quality Score — composite metric of portfolio health.

Combines multiple dimensions of portfolio quality into a single 0–100 score.
Not based on P&L — measures the structural quality of the portfolio and its
risk/return characteristics.

Dimensions (equal-weighted by default):
    1. Risk-Adjusted Return  — Sharpe ratio quality
    2. Drawdown Resilience   — recovery speed + time in drawdown
    3. Alpha Generation      — CAPM alpha significance (DSR)
    4. Diversification       — avg pairwise correlation (lower = better)
    5. Tail Risk Management  — CVaR/VaR ratio (lower = thinner tails)
    6. Exposure Balance      — how well hedged (net beta near zero, L/S ratio)

Each dimension scores 0–100, then combined via weighted average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from core.metrics import (
    correlation_metrics,
    return_metrics,
    risk_metrics,
)
from core.metrics.drawdown_analytics import expected_drawdown_pct, time_in_drawdown_pct
from core.portfolio import Portfolio


@dataclass
class QualityDimension:
    """A single dimension of the quality score."""

    name: str
    score: float  # 0–100
    weight: float  # fraction of total (sums to 1.0)
    detail: str  # human-readable explanation


@dataclass
class PortfolioQualityScore:
    """Composite portfolio quality assessment."""

    total_score: float  # 0–100 weighted composite
    grade: str  # A+ through F
    dimensions: list[QualityDimension] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dimension scoring helpers (all return 0–100)
# ---------------------------------------------------------------------------


def _score_sharpe(sharpe: float) -> float:
    """Score Sharpe ratio quality.

    0.0 → 20,  0.5 → 40,  1.0 → 60,  1.5 → 75,  2.0 → 90,  3.0+ → 100
    Negative Sharpe floors at 0.
    """
    if sharpe <= 0:
        return max(0.0, 10 + sharpe * 10)  # slight credit for near-zero
    # Sigmoid-like mapping: 100 * (1 - exp(-0.8 * SR))
    raw = 100.0 * (1.0 - math.exp(-0.8 * sharpe))
    return min(100.0, max(0.0, raw))


def _score_drawdown_resilience(sharpe: float, ann_vol: float) -> float:
    """Score drawdown resilience using Bailey & Lopez de Prado analytics.

    Low expected drawdown + low time in drawdown → high score.
    """
    if sharpe <= 0 or ann_vol <= 0:
        return 10.0

    # Expected drawdown as % of NAV
    edd = expected_drawdown_pct(sharpe, ann_vol)
    # Time in drawdown as %
    tidd = time_in_drawdown_pct(sharpe)

    # Score: lower edd and tidd are better
    # edd < 2% → excellent, > 10% → poor
    edd_score = max(0.0, 100.0 - (abs(edd) * 100) * 8)  # 2% → 84, 5% → 60, 10% → 20
    # tidd < 20% → excellent, > 60% → poor
    tidd_score = max(0.0, 100.0 - tidd * 1.5)  # 20% → 70, 40% → 40, 60% → 10

    return (edd_score + tidd_score) / 2.0


def _score_alpha(dsr: float) -> float:
    """Score alpha generation quality using Deflated Sharpe Ratio.

    DSR > 0.95 → strong evidence of real alpha.
    DSR < 0.50 → likely random.
    """
    # Linear mapping: DSR 0→10, 0.5→35, 0.8→65, 0.95→88, 1.0→100
    return min(100.0, max(0.0, dsr * 100.0))


def _score_diversification(avg_pairwise_corr: float) -> float:
    """Score diversification quality from average pairwise correlation.

    Lower correlation = more diversified = higher score.
    avg_corr 0.0 → 100,  0.3 → 70,  0.5 → 50,  0.8 → 20,  1.0 → 0
    """
    return max(0.0, min(100.0, 100.0 * (1.0 - avg_pairwise_corr)))


def _score_tail_risk(var_95: float, cvar_95: float) -> float:
    """Score tail risk management from CVaR/VaR ratio.

    CVaR/VaR ≈ 1.0 → thin tails (great), ratio > 2.0 → fat tails (dangerous).
    Also penalizes absolute VaR > 3%.
    """
    if var_95 <= 0:
        return 50.0  # no data

    ratio = cvar_95 / var_95 if var_95 > 0 else 1.0

    # Ratio score: 1.0 → 90, 1.5 → 60, 2.0 → 30, 3.0 → 0
    ratio_score = max(0.0, 100.0 - (ratio - 1.0) * 60.0)

    # Absolute VaR score: < 1% → 90, 2% → 60, 3% → 30, 5% → 0
    abs_score = max(0.0, 100.0 - var_95 * 100 * 20)

    return (ratio_score * 0.6 + abs_score * 0.4)


def _score_exposure_balance(
    net_beta: float,
    gross_exposure: float,
    ls_ratio: float,
) -> float:
    """Score how well-balanced the portfolio exposures are.

    Net beta near 0 = hedged well.
    L/S ratio near 1.0 = balanced books.
    Gross exposure 1.0–1.8 = optimal range.
    """
    # Beta score: 0.0 → 100, ±0.1 → 80, ±0.3 → 40, ±0.5 → 0
    beta_score = max(0.0, 100.0 - abs(net_beta) * 200)

    # L/S ratio score: 1.0 → 100, 1.5 → 60, 2.0 → 30
    if math.isinf(ls_ratio):
        ls_score = 0.0
    else:
        ls_score = max(0.0, 100.0 - abs(ls_ratio - 1.0) * 70)

    # Gross exposure: 1.0–1.8 optimal, penalize outside
    if 0.8 <= gross_exposure <= 2.0:
        gross_score = 80.0
    elif gross_exposure < 0.5:
        gross_score = 30.0  # under-invested
    elif gross_exposure > 2.5:
        gross_score = 20.0  # over-levered
    else:
        gross_score = 60.0

    return beta_score * 0.45 + ls_score * 0.35 + gross_score * 0.20


def _letter_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 93:
        return "A+"
    if score >= 87:
        return "A"
    if score >= 83:
        return "A−"
    if score >= 78:
        return "B+"
    if score >= 73:
        return "B"
    if score >= 68:
        return "B−"
    if score >= 63:
        return "C+"
    if score >= 58:
        return "C"
    if score >= 53:
        return "C−"
    if score >= 45:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

# Default weights — each dimension contributes equally
DEFAULT_WEIGHTS = {
    "risk_adjusted_return": 0.20,
    "drawdown_resilience": 0.18,
    "alpha_generation": 0.18,
    "diversification": 0.16,
    "tail_risk": 0.14,
    "exposure_balance": 0.14,
}


def compute_quality_score(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
    weights: dict[str, float] | None = None,
) -> PortfolioQualityScore:
    """
    Compute composite Portfolio Quality Score (0–100).

    Args:
        portfolio: current portfolio state
        returns_df: daily returns DataFrame (date + per-ticker columns)
        betas: per-ticker betas
        risk_free_rate: annualized risk-free rate
        weights: optional custom dimension weights (must sum to ~1.0)

    Returns:
        PortfolioQualityScore with total score, grade, and per-dimension breakdown.
    """
    w = weights or DEFAULT_WEIGHTS
    dimensions: list[QualityDimension] = []

    port_weights = portfolio.weight_vector()
    has_returns = returns_df is not None and returns_df.height > 30

    # --- 1. Risk-Adjusted Return (Sharpe) ---
    sharpe = 0.0
    ann_vol = 0.0
    if has_returns:
        port_rets = return_metrics.portfolio_daily_returns(port_weights, returns_df)
        if port_rets.len() > 20:
            sharpe = return_metrics.sharpe_ratio(port_rets, risk_free_rate)
            ann_vol = return_metrics.annualized_volatility(port_rets)

    s1 = _score_sharpe(sharpe)
    dimensions.append(QualityDimension(
        name="Risk-Adj Return",
        score=s1,
        weight=w["risk_adjusted_return"],
        detail=f"Sharpe {sharpe:.2f}",
    ))

    # --- 2. Drawdown Resilience ---
    s2 = _score_drawdown_resilience(sharpe, ann_vol)
    dimensions.append(QualityDimension(
        name="DD Resilience",
        score=s2,
        weight=w["drawdown_resilience"],
        detail=f"E[DD] {expected_drawdown_pct(sharpe, ann_vol):.1%}" if sharpe > 0 and ann_vol > 0 else "—",
    ))

    # --- 3. Alpha Generation (DSR) ---
    dsr = 0.0
    if has_returns:
        port_rets = return_metrics.portfolio_daily_returns(port_weights, returns_df)
        if port_rets.len() > 30:
            dsr = return_metrics.deflated_sharpe_ratio(port_rets, risk_free_rate)

    s3 = _score_alpha(dsr)
    dimensions.append(QualityDimension(
        name="Alpha Quality",
        score=s3,
        weight=w["alpha_generation"],
        detail=f"DSR {dsr:.1%}",
    ))

    # --- 4. Diversification ---
    avg_corr = 0.5  # default
    if has_returns:
        tickers = [t for t in portfolio.tickers if t in returns_df.columns]
        if len(tickers) >= 3:
            avg_corr = correlation_metrics.average_pairwise_correlation(returns_df, tickers)

    s4 = _score_diversification(avg_corr)
    dimensions.append(QualityDimension(
        name="Diversification",
        score=s4,
        weight=w["diversification"],
        detail=f"Avg ρ {avg_corr:.3f}",
    ))

    # --- 5. Tail Risk ---
    var_95 = 0.0
    cvar_95 = 0.0
    if has_returns:
        port_rets = return_metrics.portfolio_daily_returns(port_weights, returns_df)
        if port_rets.len() > 20:
            var_95 = risk_metrics.var_historical(port_rets, 0.95)
            cvar_95 = risk_metrics.cvar_historical(port_rets, 0.95)

    s5 = _score_tail_risk(var_95, cvar_95)
    dimensions.append(QualityDimension(
        name="Tail Risk",
        score=s5,
        weight=w["tail_risk"],
        detail=f"VaR {var_95:.2%} / CVaR {cvar_95:.2%}",
    ))

    # --- 6. Exposure Balance ---
    net_beta = 0.0
    if betas:
        from core.metrics.exposure_metrics import net_beta_exposure
        net_beta = net_beta_exposure(portfolio, betas)

    s6 = _score_exposure_balance(
        net_beta=net_beta,
        gross_exposure=portfolio.gross_exposure,
        ls_ratio=portfolio.long_short_ratio,
    )
    dimensions.append(QualityDimension(
        name="Exposure Balance",
        score=s6,
        weight=w["exposure_balance"],
        detail=f"β {net_beta:+.3f}, L/S {portfolio.long_short_ratio:.2f}",
    ))

    # --- Composite ---
    total = sum(d.score * d.weight for d in dimensions)
    total = min(100.0, max(0.0, total))

    return PortfolioQualityScore(
        total_score=total,
        grade=_letter_grade(total),
        dimensions=dimensions,
    )
