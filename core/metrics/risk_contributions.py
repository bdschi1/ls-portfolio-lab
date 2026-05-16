"""Per-name risk decomposition and sizing diagnostics.

Implements marginal (MCTR) and component (CCTR) contributions to portfolio
total risk and to portfolio idiosyncratic-only risk, plus an IR-style sizing
yardstick (alpha / idio_vol).

Pure functions — Polars in, numpy in the linear-algebra core, plain dataclasses
out. No Streamlit, no pandas.

Method summary:

- Build the realised daily covariance matrix Sigma from the per-name return
  panel. Portfolio variance is sigma_p^2 = w' Sigma w with w the signed weight
  vector (longs positive, shorts negative).
- MCTR_i = (Sigma w)_i / sigma_p — the partial derivative of portfolio vol
  w.r.t. weight i.
- CCTR_i = w_i * MCTR_i. By Euler's theorem on the homogeneous-degree-1
  portfolio-vol function, sum_i CCTR_i = sigma_p.
- Annualize sigma_p and CCTRs by sqrt(252).
- For the idio-only view, regress each name's returns on the supplied factor
  set (market, SMB, HML, MOM at minimum), keep residuals, rebuild covariance
  on residuals, and repeat the MCTR/CCTR calc. Sum of idio CCTRs equals
  portfolio idio vol.
- alpha_estimate = mean(daily return) * 252.
- alpha_over_idio = alpha_estimate / idio_vol_ann, with idio_vol_ann == 0
  returning 0.0 (chosen over NaN so downstream sorting/aggregation works
  without filtering; see DIV_BY_ZERO_POLICY below).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.portfolio import Portfolio

TRADING_DAYS_PER_YEAR = 252

# Divide-by-zero policy for alpha_over_idio: when idio_vol_ann is at or below
# IDIO_VOL_FLOOR (effectively zero — e.g., a name whose returns are perfectly
# explained by the factor set, or a degenerate constant-return series), we
# return 0.0 rather than NaN or a huge spurious ratio. Rationale: this metric
# is most often consumed by sort / rank / aggregate operations in the
# Streamlit UI; NaN would force every consumer to filter or fillna, and a
# 1e16-scale ratio from a 1e-16 residual would dominate any ranking. A zero
# explicitly says "no information about IR, do not size on this." Callers that
# want NaN semantics can re-check idio_vol_ann themselves.
# Floor is set well above float64 noise (~1e-15 daily → ~1e-14 annualised)
# but well below any economically meaningful idio vol (~0.01 = 1% ann).
IDIO_VOL_FLOOR = 1e-10
DIV_BY_ZERO_POLICY = f"alpha_over_idio returns 0.0 when idio_vol_ann <= {IDIO_VOL_FLOOR}"


@dataclass
class NameRiskContrib:
    """Per-name risk decomposition for one portfolio position."""

    ticker: str
    side: str  # LONG | SHORT
    shares: float
    notional: float  # signed dollar position (+ long, - short)
    weight: float  # signed weight, fraction of NAV
    beta_market: float
    idio_vol_ann: float  # annualized idio vol
    factor_vol_ann: float  # annualized factor-driven vol of the name
    total_vol_ann: float  # annualized total vol of the name (realised)
    mctr_total: float  # marginal contribution to total portfolio risk
    cctr_total: float  # component contribution to total portfolio risk
    mctr_idio: float  # marginal contribution to idio-only portfolio risk
    cctr_idio: float  # component contribution to idio portfolio risk
    dollar_vol_contrib: float  # |cctr_total| * NAV — $ vol per year contributed
    alpha_estimate: float  # mean daily return * 252
    alpha_over_idio: float  # alpha_estimate / idio_vol_ann (IR-style)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aligned_return_matrix(
    returns_df: pl.DataFrame,
    tickers: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return (n_obs x n_tickers) numpy matrix and the list of tickers used.

    Tickers absent from returns_df are silently dropped (their slot in the
    output ordering is skipped). Drops rows where any kept column is null.
    """
    available = [t for t in tickers if t in returns_df.columns]
    if not available:
        return np.empty((0, 0)), []
    sub = returns_df.select(available).drop_nulls()
    return sub.to_numpy(), available


def _portfolio_returns(
    weights_by_ticker: dict[str, float],
    returns_df: pl.DataFrame,
) -> pl.Series:
    """Daily portfolio returns = sum_i w_i * r_i, restricted to tickers present."""
    tickers = [t for t in weights_by_ticker if t in returns_df.columns]
    if not tickers:
        return pl.Series("portfolio_returns", [], dtype=pl.Float64)
    mat, _ = _aligned_return_matrix(returns_df, tickers)
    if mat.size == 0:
        return pl.Series("portfolio_returns", [], dtype=pl.Float64)
    w = np.array([weights_by_ticker[t] for t in tickers], dtype=float)
    port = mat @ w
    return pl.Series("portfolio_returns", port)


def _factor_matrix(
    factor_returns: dict[str, pl.Series],
    n_obs: int,
) -> tuple[np.ndarray, list[str]]:
    """Stack factor series into an (n_obs x k) matrix.

    All factor series are tail-aligned to n_obs (matches the convention in
    core/factor_model.py). Factors with fewer than n_obs observations are
    truncated by tail() — caller is responsible for passing series long enough.
    """
    names = list(factor_returns.keys())
    if not names:
        return np.empty((n_obs, 0)), []
    cols = [factor_returns[name].tail(n_obs).to_numpy() for name in names]
    return np.column_stack(cols), names


def _residuals_after_factor_regression(
    returns_matrix: np.ndarray,
    factor_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS-residualise each column of returns_matrix on factor_matrix + intercept.

    Returns (residuals_matrix, betas_matrix). betas_matrix is (k_factors x n_names)
    excluding the intercept row, in factor order.
    """
    n_obs, n_names = returns_matrix.shape
    if factor_matrix.size == 0 or n_obs == 0:
        return returns_matrix.copy(), np.zeros((0, n_names))

    x_with_const = np.column_stack([np.ones(n_obs), factor_matrix])
    # Multi-column OLS: betas is (1 + k_factors) x n_names.
    betas, *_ = np.linalg.lstsq(x_with_const, returns_matrix, rcond=None)
    fitted = x_with_const @ betas
    residuals = returns_matrix - fitted
    return residuals, betas[1:, :]


def _mctr_cctr(
    weights: np.ndarray,
    cov_daily: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute MCTR, CCTR and portfolio vol from a daily covariance matrix.

    Returns are all in *annualised* units (sqrt(252) applied). MCTR and CCTR
    are zero-vectors when portfolio variance is non-positive (degenerate case,
    e.g. all-zero weights).
    """
    port_var_daily = float(weights @ cov_daily @ weights)
    if port_var_daily <= 0.0:
        zeros = np.zeros_like(weights)
        return zeros, zeros, 0.0
    sigma_p_daily = np.sqrt(port_var_daily)
    sigma_w = cov_daily @ weights
    mctr_daily = sigma_w / sigma_p_daily
    cctr_daily = weights * mctr_daily
    ann = float(np.sqrt(TRADING_DAYS_PER_YEAR))
    return mctr_daily * ann, cctr_daily * ann, sigma_p_daily * ann


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def per_name_risk_contributions(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
) -> list[NameRiskContrib]:
    """Per-name MCTR/CCTR for total and idio risk, plus alpha / IR proxies.

    Positions whose ticker is absent from returns_df are silently skipped
    (consistent with risk_metrics.portfolio_volatility's behaviour).
    """
    weights_by_ticker = portfolio.weight_vector()
    tickers_in_order = [p.ticker for p in portfolio.positions if p.ticker in returns_df.columns]
    if not tickers_in_order:
        return []

    returns_matrix, used = _aligned_return_matrix(returns_df, tickers_in_order)
    if returns_matrix.size == 0:
        return []
    # _aligned_return_matrix preserves the input order, so `used == tickers_in_order`
    # whenever the matrix is non-empty.
    n_obs = returns_matrix.shape[0]

    weights = np.array([weights_by_ticker[t] for t in used], dtype=float)

    # --- Total-risk decomposition -----------------------------------------
    cov_total_daily = np.cov(returns_matrix, rowvar=False, ddof=1)
    # np.cov returns a 0-d array for a single column; promote to 2-d for uniform
    # downstream indexing.
    cov_total_daily = np.atleast_2d(cov_total_daily)
    mctr_total_ann, cctr_total_ann, _sigma_p_total_ann = _mctr_cctr(weights, cov_total_daily)

    # --- Idio decomposition (residuals after factor regression) -----------
    factor_matrix, _factor_names = _factor_matrix(factor_returns, n_obs)
    residuals, factor_betas = _residuals_after_factor_regression(returns_matrix, factor_matrix)
    cov_idio_daily = np.atleast_2d(np.cov(residuals, rowvar=False, ddof=1))
    mctr_idio_ann, cctr_idio_ann, _sigma_p_idio_ann = _mctr_cctr(weights, cov_idio_daily)

    # --- Per-name stats ----------------------------------------------------
    name_stats_total_vol = np.std(returns_matrix, axis=0, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    name_stats_idio_vol = np.std(residuals, axis=0, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Factor-driven vol per name = sqrt(max(total_var - idio_var, 0)). Done in
    # variance space then annualised.
    total_var_daily = np.var(returns_matrix, axis=0, ddof=1)
    idio_var_daily = np.var(residuals, axis=0, ddof=1)
    factor_var_daily = np.clip(total_var_daily - idio_var_daily, a_min=0.0, a_max=None)
    name_stats_factor_vol = np.sqrt(factor_var_daily) * np.sqrt(TRADING_DAYS_PER_YEAR)

    mean_daily_ret = returns_matrix.mean(axis=0)
    alpha_estimates = mean_daily_ret * TRADING_DAYS_PER_YEAR

    # Market beta is the first factor row if present; else 0.0.
    if factor_betas.size > 0 and factor_betas.shape[0] >= 1:
        market_betas = factor_betas[0, :]
    else:
        market_betas = np.zeros(len(used))

    nav = portfolio.nav
    position_by_ticker = {p.ticker: p for p in portfolio.positions}

    out: list[NameRiskContrib] = []
    for i, ticker in enumerate(used):
        pos = position_by_ticker[ticker]
        idio_vol = float(name_stats_idio_vol[i])
        alpha_est = float(alpha_estimates[i])
        # Divide-by-zero policy: see module docstring + DIV_BY_ZERO_POLICY.
        alpha_over_idio = alpha_est / idio_vol if idio_vol > IDIO_VOL_FLOOR else 0.0
        out.append(
            NameRiskContrib(
                ticker=ticker,
                side=pos.side,
                shares=float(pos.shares),
                notional=float(pos.signed_notional),
                weight=float(weights[i]),
                beta_market=float(market_betas[i]),
                idio_vol_ann=idio_vol,
                factor_vol_ann=float(name_stats_factor_vol[i]),
                total_vol_ann=float(name_stats_total_vol[i]),
                mctr_total=float(mctr_total_ann[i]),
                cctr_total=float(cctr_total_ann[i]),
                mctr_idio=float(mctr_idio_ann[i]),
                cctr_idio=float(cctr_idio_ann[i]),
                dollar_vol_contrib=abs(float(cctr_total_ann[i])) * nav,
                alpha_estimate=alpha_est,
                alpha_over_idio=alpha_over_idio,
            )
        )
    return out


def portfolio_total_vol(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
) -> float:
    """Annualized portfolio vol from realised returns (sqrt(w' Sigma w) * sqrt(252))."""
    weights_by_ticker = portfolio.weight_vector()
    tickers = [t for t in weights_by_ticker if t in returns_df.columns]
    if not tickers:
        return 0.0
    returns_matrix, used = _aligned_return_matrix(returns_df, tickers)
    if returns_matrix.size == 0:
        return 0.0
    weights = np.array([weights_by_ticker[t] for t in used], dtype=float)
    cov_daily = np.atleast_2d(np.cov(returns_matrix, rowvar=False, ddof=1))
    port_var_daily = float(weights @ cov_daily @ weights)
    if port_var_daily <= 0.0:
        return 0.0
    return float(np.sqrt(port_var_daily) * np.sqrt(TRADING_DAYS_PER_YEAR))


def portfolio_idio_vol(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
) -> float:
    """Annualized vol of residuals after regressing portfolio returns on factors.

    Uses a single regression on the *portfolio* return series: the idio-only
    vol of the portfolio is the vol of the residual stream after factor
    exposure is hedged out at the portfolio level.
    """
    port_rets = _portfolio_returns(portfolio.weight_vector(), returns_df)
    if port_rets.len() < 2:
        return 0.0
    n_obs = port_rets.len()
    factor_matrix, _ = _factor_matrix(factor_returns, n_obs)
    y = port_rets.to_numpy()
    if factor_matrix.size == 0:
        return float(np.std(y, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    x_with_const = np.column_stack([np.ones(n_obs), factor_matrix])
    betas, *_ = np.linalg.lstsq(x_with_const, y, rcond=None)
    residuals = y - x_with_const @ betas
    if residuals.size < 2:
        return 0.0
    return float(np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
