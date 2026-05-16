"""Factor-decomposed P&L attribution.

Decomposes realized P&L over a horizon (1D, WTD, MTD, YTD) into:
- Market loading ($)
- Style factor loadings ($, per factor)
- Sector loading ($, optional)
- Idiosyncratic / stock-specific ($)

Method
------
1. For each position, estimate factor betas on the FULL return history via OLS
   (delegates to `core.factor_model.multi_factor_regression` when applicable;
   falls back to a direct OLS for sector-only or arbitrary style sets).
2. For each period in the horizon slice, decompose return as:
       r_i,t = sum_k beta_i,k * f_k,t + epsilon_i,t
   The $ contribution of factor k for position i over the horizon is
       beta_i,k * (sum_t f_k,t) * notional_i
   (signed by side: SHORT positions earn the negative of the underlying return).
3. Idiosyncratic $ = realized $ P&L over the horizon minus all factor $.
4. Portfolio aggregates are simple sums. We assert that
       market + sum(style) + sector + idio  ~=  realized_total
   within a small tolerance and log a warning if not.

Notes
-----
- Polars only. No pandas.
- `returns_df` must contain a `date` column (Date or Datetime) and one column
  per ticker holding daily simple returns.
- Position-start notional is approximated by `position.notional` (current
  notional). The horizon is short relative to estimation history so this is
  acceptable; documented for callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import polars as pl

from core.portfolio import Portfolio

logger = logging.getLogger(__name__)

Horizon = Literal["1D", "WTD", "MTD", "YTD"]

# Tolerance for reconciliation between (market+style+sector+idio) and realized P&L.
# Set as a fraction of gross notional — exact equality is expected by construction,
# but we leave a small floating-point cushion.
_RECON_REL_TOL = 1e-6
_RECON_ABS_TOL = 1e-3  # $ — absolute floor for tiny portfolios


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NameContrib:
    """Per-position factor P&L breakdown for the chosen horizon."""

    ticker: str
    side: str  # "LONG" | "SHORT"
    pnl_total: float
    pnl_market: float
    pnl_style: dict[str, float] = field(default_factory=dict)
    pnl_sector: float = 0.0
    pnl_idio: float = 0.0


@dataclass
class FactorAttribution:
    """Portfolio-level factor P&L attribution over a horizon."""

    horizon: Horizon
    total_pnl: float
    market_pnl: float
    style_pnl: dict[str, float]
    sector_pnl: float
    idio_pnl: float
    name_contribs: list[NameContrib]
    top_idio_winners: list[NameContrib] = field(default_factory=list)
    top_idio_losers: list[NameContrib] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Horizon slicing
# ---------------------------------------------------------------------------


def _ensure_date_col(returns_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure 'date' column exists and is Date dtype."""
    if "date" not in returns_df.columns:
        msg = "returns_df must contain a 'date' column"
        raise ValueError(msg)
    col = returns_df["date"]
    if col.dtype == pl.Datetime:
        return returns_df.with_columns(pl.col("date").cast(pl.Date))
    if col.dtype != pl.Date:
        return returns_df.with_columns(pl.col("date").cast(pl.Date))
    return returns_df


def _slice_horizon(returns_df: pl.DataFrame, horizon: Horizon) -> pl.DataFrame:
    """Return the subset of rows that fall inside the requested horizon.

    Horizon is measured against the LAST date in `returns_df` (treated as
    "today" for the purpose of WTD/MTD/YTD). 1D returns the final row only.
    """
    if returns_df.height == 0:
        return returns_df

    df = _ensure_date_col(returns_df).sort("date")
    last_date_val = df["date"].max()
    if last_date_val is None:
        return df

    last_date: date = last_date_val  # type: ignore[assignment]
    if isinstance(last_date, datetime):
        last_date = last_date.date()

    if horizon == "1D":
        return df.tail(1)
    if horizon == "WTD":
        # ISO week: Monday is the start. WTD includes Monday..last_date inclusive.
        iso = last_date.isocalendar()
        monday = last_date - timedelta(days=iso.weekday - 1)
        return df.filter(pl.col("date") >= monday)
    if horizon == "MTD":
        month_start = last_date.replace(day=1)
        return df.filter(pl.col("date") >= month_start)
    if horizon == "YTD":
        year_start = last_date.replace(month=1, day=1)
        return df.filter(pl.col("date") >= year_start)
    msg = f"Unknown horizon: {horizon}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------


def _fit_betas(
    y: np.ndarray,
    factor_matrix: np.ndarray,
) -> np.ndarray:
    """OLS with intercept; return factor betas (excluding intercept).

    Shape: factor_matrix is (T, K); returns (K,).
    """
    if y.size < 2 or factor_matrix.shape[0] != y.size:
        return np.zeros(factor_matrix.shape[1] if factor_matrix.ndim == 2 else 0)
    x_with_const = np.column_stack([np.ones(y.size), factor_matrix])
    try:
        betas, *_ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(factor_matrix.shape[1])
    return betas[1:]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def factor_pnl_attribution(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
    sector_returns: dict[str, pl.Series] | None = None,
    horizon: Horizon = "1D",
) -> FactorAttribution:
    """Decompose realized portfolio P&L over `horizon` into market, style,
    sector, and idiosyncratic components.

    Parameters
    ----------
    portfolio: current portfolio (positions carry notional via current_price).
    returns_df: daily simple returns. Must include a 'date' column and one
        column per ticker held in `portfolio`.
    factor_returns: must include key 'market'. Additional keys are treated as
        style factors. Each Series must align row-for-row with `returns_df`.
    sector_returns: optional {sector_name: daily_returns_series}. If a
        position's `sector` matches a key, its sector exposure is fit alongside
        the factor block. If empty/None, sector_pnl is 0.
    horizon: '1D' | 'WTD' | 'MTD' | 'YTD'.

    Returns
    -------
    FactorAttribution with per-name and portfolio-level dollar breakdowns.
    """
    if "market" not in factor_returns:
        msg = "factor_returns must contain a 'market' key"
        raise ValueError(msg)

    df = _ensure_date_col(returns_df).sort("date")
    horizon_df = _slice_horizon(df, horizon)

    style_factor_names = [k for k in factor_returns if k != "market"]
    sector_map = sector_returns or {}

    # Pre-extract full-history factor arrays for beta fitting.
    n_full = df.height
    market_full = factor_returns["market"].to_numpy()
    style_full = {k: factor_returns[k].to_numpy() for k in style_factor_names}
    sector_full = {k: v.to_numpy() for k, v in sector_map.items()}

    # Align factor arrays to df length (truncate from the head if longer).
    def _tail(arr: np.ndarray) -> np.ndarray:
        if arr.size > n_full:
            return arr[-n_full:]
        return arr

    market_full = _tail(market_full)
    style_full = {k: _tail(v) for k, v in style_full.items()}
    sector_full = {k: _tail(v) for k, v in sector_full.items()}

    # Pre-extract horizon-window factor sums (used for $-decomposition).
    # Build boolean mask via date comparison to avoid is_in() deprecation warning.
    horizon_dates_set = set(horizon_df["date"].to_list())
    horizon_idx_mask = np.array([d in horizon_dates_set for d in df["date"].to_list()], dtype=bool)
    market_horizon_sum = float(market_full[horizon_idx_mask].sum())
    style_horizon_sums = {k: float(v[horizon_idx_mask].sum()) for k, v in style_full.items()}
    sector_horizon_sums = {k: float(v[horizon_idx_mask].sum()) for k, v in sector_full.items()}

    name_contribs: list[NameContrib] = []
    realized_total_pnl = 0.0

    for pos in portfolio.positions:
        ticker = pos.ticker
        if ticker not in df.columns:
            # No data → skip but emit empty contrib so caller can see the gap.
            name_contribs.append(
                NameContrib(
                    ticker=ticker,
                    side=pos.side,
                    pnl_total=0.0,
                    pnl_market=0.0,
                    pnl_style={k: 0.0 for k in style_factor_names},
                    pnl_sector=0.0,
                    pnl_idio=0.0,
                )
            )
            continue

        r_full = df[ticker].to_numpy()
        # Build factor matrix for this name (market + styles + optional own sector)
        col_blocks = [market_full]
        block_names: list[str] = ["market"]
        for k in style_factor_names:
            col_blocks.append(style_full[k])
            block_names.append(f"style:{k}")
        own_sector_key: str | None = None
        if pos.sector and pos.sector in sector_full:
            own_sector_key = pos.sector
            col_blocks.append(sector_full[own_sector_key])
            block_names.append(f"sector:{own_sector_key}")

        factor_matrix = np.column_stack(col_blocks)
        # Drop any rows with NaN in y or X to keep OLS stable.
        valid = ~np.isnan(r_full) & ~np.isnan(factor_matrix).any(axis=1)
        betas = _fit_betas(r_full[valid], factor_matrix[valid])

        # Direction: SHORT positions realize the negative of underlying return.
        direction = pos.direction
        notional = pos.notional  # approximation: period-start ≈ current

        # Per-factor $ contributions
        beta_market = float(betas[0]) if betas.size > 0 else 0.0
        pnl_market = direction * beta_market * market_horizon_sum * notional

        pnl_style: dict[str, float] = {}
        idx = 1
        for k in style_factor_names:
            b_k = float(betas[idx]) if idx < betas.size else 0.0
            pnl_style[k] = direction * b_k * style_horizon_sums[k] * notional
            idx += 1

        pnl_sector = 0.0
        if own_sector_key is not None:
            b_sec = float(betas[idx]) if idx < betas.size else 0.0
            pnl_sector = direction * b_sec * sector_horizon_sums[own_sector_key] * notional

        # Realized $ P&L over the horizon for this name
        r_horizon = r_full[horizon_idx_mask]
        # geometric compounding over the window
        if r_horizon.size > 0:
            realized_ret = float(np.prod(1.0 + r_horizon) - 1.0)
        else:
            realized_ret = 0.0
        pnl_total = direction * realized_ret * notional

        pnl_idio = pnl_total - pnl_market - sum(pnl_style.values()) - pnl_sector

        realized_total_pnl += pnl_total

        name_contribs.append(
            NameContrib(
                ticker=ticker,
                side=pos.side,
                pnl_total=pnl_total,
                pnl_market=pnl_market,
                pnl_style=pnl_style,
                pnl_sector=pnl_sector,
                pnl_idio=pnl_idio,
            )
        )

    # Aggregate
    market_pnl = sum(c.pnl_market for c in name_contribs)
    sector_pnl = sum(c.pnl_sector for c in name_contribs)
    idio_pnl = sum(c.pnl_idio for c in name_contribs)
    style_pnl: dict[str, float] = {k: 0.0 for k in style_factor_names}
    for c in name_contribs:
        for k, v in c.pnl_style.items():
            style_pnl[k] = style_pnl.get(k, 0.0) + v
    total_pnl = market_pnl + sum(style_pnl.values()) + sector_pnl + idio_pnl

    # Reconciliation check
    gross = sum(abs(p.notional) for p in portfolio.positions) or 1.0
    diff = abs(total_pnl - realized_total_pnl)
    if diff > max(_RECON_ABS_TOL, _RECON_REL_TOL * gross):
        logger.warning(
            "Attribution reconciliation gap: components=%.4f, realized=%.4f, diff=%.4f",
            total_pnl,
            realized_total_pnl,
            diff,
        )

    # Ranked idio winners / losers (top 10 each end, sorted)
    by_idio = sorted(name_contribs, key=lambda c: c.pnl_idio, reverse=True)
    top_idio_winners = by_idio[:10]
    top_idio_losers = sorted(name_contribs, key=lambda c: c.pnl_idio)[:10]

    return FactorAttribution(
        horizon=horizon,
        total_pnl=total_pnl,
        market_pnl=market_pnl,
        style_pnl=style_pnl,
        sector_pnl=sector_pnl,
        idio_pnl=idio_pnl,
        name_contribs=name_contribs,
        top_idio_winners=top_idio_winners,
        top_idio_losers=top_idio_losers,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def hit_rate(positions: list[NameContrib]) -> float:
    """Fraction of positions with positive idiosyncratic P&L."""
    if not positions:
        return 0.0
    wins = sum(1 for p in positions if p.pnl_idio > 0)
    return wins / len(positions)


def slugging(positions: list[NameContrib]) -> float:
    """Average winning idio $ / average losing idio $ (absolute).

    Returns NaN when there are no losers (undefined ratio).
    """
    if not positions:
        return float("nan")
    wins = [p.pnl_idio for p in positions if p.pnl_idio > 0]
    losses = [abs(p.pnl_idio) for p in positions if p.pnl_idio < 0]
    if not losses:
        return float("nan")
    if not wins:
        return 0.0
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    return avg_win / avg_loss
