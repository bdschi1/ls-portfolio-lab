"""Portfolio rebalancer — constrained optimization to hit risk targets.

Given a portfolio, target net beta and/or target annualized volatility,
computes new position weights that satisfy the constraints while minimizing
turnover (deviation from current weights).

Uses scipy SLSQP, same pattern as mock_portfolio._optimize_weights().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize

from core.metrics import exposure_metrics, risk_metrics
from core.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RebalanceRequest:
    """Input parameters for a rebalance operation."""

    target_net_beta: float | None = None  # e.g. 0.08 for 8%
    target_ann_vol: float | None = None  # e.g. 0.06 for 6%
    max_sector_net: float = 0.50  # max absolute sector net exposure
    min_weight: float = 0.005  # 0.5% floor per position
    max_weight: float = 0.08  # 8% cap per position


@dataclass
class RebalanceResult:
    """Output of a rebalance computation."""

    portfolio_before: Portfolio
    portfolio_after: Portfolio
    weight_changes: dict[str, tuple[float, float]]  # {ticker: (old_wt, new_wt)}
    trades_needed: list[dict]  # [{ticker, action, old_shares, new_shares, delta_notional}]
    converged: bool
    target_met: dict[str, bool]  # {"net_beta": True, "ann_vol": False}
    warnings: list[str] = field(default_factory=list)
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core rebalance function
# ---------------------------------------------------------------------------


def compute_rebalance(
    portfolio: Portfolio,
    request: RebalanceRequest,
    returns_df: pl.DataFrame | None,
    betas: dict[str, float],
    current_prices: dict[str, float],
) -> RebalanceResult:
    """Compute rebalanced portfolio to meet risk targets.

    Minimizes turnover (squared deviation from current weights) subject to:
    - Net beta == target (if set)
    - Annualized vol <= target (if set)
    - |Sector net exposure| <= max_sector_net per sector
    - Each position weight in [min_weight, max_weight]

    Options are excluded from optimization and keep current weights.
    Returns a new portfolio — original is never mutated.
    """
    warnings: list[str] = []

    # --- Validate inputs ---
    if request.target_net_beta is None and request.target_ann_vol is None:
        return RebalanceResult(
            portfolio_before=portfolio,
            portfolio_after=portfolio,
            weight_changes={},
            trades_needed=[],
            converged=True,
            target_met={},
            warnings=["No targets set. Nothing to rebalance."],
        )

    # Separate optimizable positions (equity + ETF) from options
    opt_positions = [p for p in portfolio.positions if p.asset_type != "OPTION"]
    option_positions = [p for p in portfolio.positions if p.asset_type == "OPTION"]

    if len(opt_positions) < 2:
        return RebalanceResult(
            portfolio_before=portfolio,
            portfolio_after=portfolio,
            weight_changes={},
            trades_needed=[],
            converged=False,
            target_met={},
            warnings=["Need at least 2 non-option positions to rebalance."],
        )

    # Vol target requires returns data
    has_returns = returns_df is not None and returns_df.height > 20
    if request.target_ann_vol is not None and not has_returns:
        warnings.append(
            "No returns data available — cannot target volatility. "
            "Only beta constraint will be applied."
        )

    # --- Build optimizer inputs ---
    tickers = [p.ticker for p in opt_positions]
    n = len(tickers)

    signs = np.array([p.direction for p in opt_positions], dtype=float)
    beta_vec = np.array([betas.get(t, 1.0) for t in tickers])
    sectors = [p.sector or "Unknown" for p in opt_positions]

    # Current absolute weights (x0)
    current_abs_weights = np.array(
        [p.abs_weight_in(portfolio.nav) for p in opt_positions]
    )
    # Clamp x0 to bounds so optimizer starts feasible
    x0 = np.clip(current_abs_weights, request.min_weight, request.max_weight)

    # Covariance matrix (only if vol target set and returns available)
    cov_matrix = None
    if request.target_ann_vol is not None and has_returns:
        available_tickers = [t for t in tickers if t in returns_df.columns]
        if len(available_tickers) >= 2:
            cov_matrix = risk_metrics.covariance_matrix(returns_df, tickers)
            if cov_matrix.size == 0:
                cov_matrix = None
                warnings.append("Could not build covariance matrix — skipping vol constraint.")

    # --- Objective: minimize turnover from current weights ---
    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - x0) ** 2))

    # --- Constraints ---
    constraints: list[dict] = []

    # 1. Net beta constraint (equality)
    if request.target_net_beta is not None:
        target_beta = request.target_net_beta

        def beta_constraint(x: np.ndarray) -> float:
            signed_w = signs * x
            return float(np.dot(signed_w, beta_vec) - target_beta)

        constraints.append({"type": "eq", "fun": beta_constraint})

    # 2. Annualized vol constraint (inequality: vol <= target)
    if request.target_ann_vol is not None and cov_matrix is not None:
        target_vol = request.target_ann_vol
        trading_days = 252.0

        def vol_constraint(x: np.ndarray) -> float:
            signed_w = signs * x
            port_var = float(signed_w @ cov_matrix @ signed_w)
            if port_var < 0:
                port_var = 0.0
            ann_vol = float(np.sqrt(port_var) * np.sqrt(trading_days))
            return target_vol - ann_vol  # must be >= 0

        constraints.append({"type": "ineq", "fun": vol_constraint})

    # 3. Sector net exposure constraints (inequality per sector)
    unique_sectors = list(set(sectors))
    for sector in unique_sectors:
        mask = np.array([1.0 if s == sector else 0.0 for s in sectors])

        def make_sector_constraint(mask_=mask):
            def constraint(x: np.ndarray) -> float:
                signed_w = signs * x
                sector_net = float(np.dot(signed_w, mask_))
                return request.max_sector_net - abs(sector_net)
            return constraint

        constraints.append({"type": "ineq", "fun": make_sector_constraint()})

    # --- Bounds ---
    bounds = [(request.min_weight, request.max_weight)] * n

    # --- Run optimizer ---
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    converged = result.success
    if not converged:
        warnings.append(
            f"Optimizer did not fully converge: {result.message}. "
            "Using best available weights."
        )

    new_abs_weights = result.x

    # --- Build new portfolio ---
    new_positions = []
    weight_changes: dict[str, tuple[float, float]] = {}
    trades_needed: list[dict] = []

    for i, pos in enumerate(opt_positions):
        old_abs_wt = float(current_abs_weights[i])
        new_abs_wt = float(new_abs_weights[i])
        old_signed_wt = old_abs_wt * pos.direction
        new_signed_wt = new_abs_wt * pos.direction

        weight_changes[pos.ticker] = (old_signed_wt, new_signed_wt)

        # Convert weight to shares
        price = current_prices.get(pos.ticker, pos.current_price)
        if price <= 0:
            price = pos.current_price
        new_notional = new_abs_wt * portfolio.nav
        new_shares = new_notional / price if price > 0 else pos.shares

        # Record trade if weight changed materially (>0.05%)
        delta_wt = new_abs_wt - old_abs_wt
        if abs(delta_wt) > 0.0005:
            if pos.side == "LONG":
                action = "ADD" if delta_wt > 0 else "REDUCE"
            else:
                action = "SHORT" if delta_wt > 0 else "COVER"

            trades_needed.append({
                "ticker": pos.ticker,
                "action": action,
                "old_shares": round(pos.shares, 2),
                "new_shares": round(new_shares, 2),
                "delta_notional": round(delta_wt * portfolio.nav, 0),
                "old_weight": round(old_signed_wt * 100, 2),
                "new_weight": round(new_signed_wt * 100, 2),
            })

        new_positions.append(pos.model_copy(update={"shares": max(new_shares, 0.01)}))

    # Keep option positions unchanged
    for pos in option_positions:
        new_positions.append(pos)
        old_wt = pos.weight_in(portfolio.nav)
        weight_changes[pos.ticker] = (old_wt, old_wt)

    # Rebuild portfolio
    total_invested = sum(p.notional for p in new_positions)
    new_cash = portfolio.nav - total_invested

    if new_cash < 0:
        warnings.append(
            f"Rebalance results in negative cash (${new_cash:,.0f}). "
            "Consider reducing gross exposure."
        )

    portfolio_after = portfolio.model_copy(update={
        "positions": new_positions,
        "cash": new_cash,
    })

    # --- Check if targets were met ---
    target_met: dict[str, bool] = {}

    new_weights = portfolio_after.weight_vector()

    if request.target_net_beta is not None:
        actual_beta = risk_metrics.portfolio_beta(new_weights, betas)
        target_met["net_beta"] = abs(actual_beta - request.target_net_beta) < 0.005

    if request.target_ann_vol is not None and has_returns:
        actual_vol = risk_metrics.portfolio_volatility(new_weights, returns_df)
        target_met["ann_vol"] = actual_vol <= request.target_ann_vol + 0.002

    # --- Compute before/after summary metrics ---
    old_weights = portfolio.weight_vector()

    metrics_before: dict[str, float] = {
        "net_beta": risk_metrics.portfolio_beta(old_weights, betas),
        "gross_exposure": portfolio.gross_exposure,
        "net_exposure": portfolio.net_exposure,
    }
    metrics_after: dict[str, float] = {
        "net_beta": risk_metrics.portfolio_beta(new_weights, betas),
        "gross_exposure": portfolio_after.gross_exposure,
        "net_exposure": portfolio_after.net_exposure,
    }

    if has_returns:
        metrics_before["ann_vol"] = risk_metrics.portfolio_volatility(old_weights, returns_df)
        metrics_after["ann_vol"] = risk_metrics.portfolio_volatility(new_weights, returns_df)

    # Sort trades by absolute delta
    trades_needed.sort(key=lambda t: abs(t["delta_notional"]), reverse=True)

    return RebalanceResult(
        portfolio_before=portfolio,
        portfolio_after=portfolio_after,
        weight_changes=weight_changes,
        trades_needed=trades_needed,
        converged=converged,
        target_met=target_met,
        warnings=warnings,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )
