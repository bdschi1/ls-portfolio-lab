"""Portfolio variance decomposition into market / style / sector / idiosyncratic.

Decomposes annualized portfolio variance into market, style, sector, and
idiosyncratic components.

Workflow per position:
    1. Regress r_i on the supplied factor set (market + style) using the
       existing CAPM/FF helpers in core/factor_model.py — do NOT re-implement
       OLS here.
    2. If sector ETF returns are supplied, run a SECOND regression of the
       first-pass residuals on the sector returns to extract sector loadings.
    3. Combine name-level betas into portfolio loadings with weights
       w_i = signed_notional_i / long_gross.

Component variances (annualized, x252):
    market_var  = beta_p_mkt^2 * var(mkt)
    style_var_k = beta_p_k^2  * var(factor_k)
    sector_var  = sum over sectors of (beta_p_sector^2 * var(sector))
    idio_var    = sum_i (w_i^2 * var(eps_i))   [assumes independent residuals]

total_var = market_var + sum(style_var) + sector_var + idio_var
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from core.factor_model import multi_factor_regression
from core.portfolio import Portfolio

TRADING_DAYS_PER_YEAR = 252


# Mapping from user-facing factor keys to the keys expected by
# core.factor_model.multi_factor_regression. The factor model uses
# lowercase canonical names ("market", "smb", "hml", "momentum").
_FACTOR_KEY_ALIASES: dict[str, str] = {
    "MKT": "market",
    "MARKET": "market",
    "SMB": "smb",
    "HML": "hml",
    "MOM": "momentum",
    "UMD": "momentum",
    "MOMENTUM": "momentum",
}


@dataclass
class VarianceDecomposition:
    """Annualized variance decomposition for an L/S portfolio."""

    total_var: float
    market_var: float
    style_var: dict[str, float]
    sector_var: float
    idio_var: float
    market_pct: float
    style_pct: dict[str, float] = field(default_factory=dict)
    sector_pct: float = 0.0
    idio_pct: float = 0.0


def _canonical_factor_key(name: str) -> str:
    """Map a user-supplied factor label to the regression-helper key."""
    return _FACTOR_KEY_ALIASES.get(name.upper(), name.lower())


def _align_to_length(series: pl.Series, n: int) -> np.ndarray:
    """Take the tail of length n and return as a 1-D numpy array."""
    return series.tail(n).to_numpy()


def _portfolio_weights(portfolio: Portfolio) -> dict[str, float]:
    """Signed weights normalized by long-side gross.

    Long positions contribute positive weights, shorts negative. Per the
    decomposition spec: w_i = signed_notional_i / sum_long(|notional|).
    """
    long_gross = sum(p.notional for p in portfolio.long_positions)
    if long_gross <= 0:
        return {p.ticker: 0.0 for p in portfolio.positions}
    return {p.ticker: p.signed_notional / long_gross for p in portfolio.positions}


def variance_decomposition(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
    sector_returns: dict[str, pl.Series] | None = None,
) -> VarianceDecomposition:
    """Decompose annualized portfolio variance into market/style/sector/idio.

    Args:
        portfolio: Pydantic Portfolio with one or more positions.
        returns_df: Daily simple returns; must contain a column per ticker.
        factor_returns: {factor_name: daily return series}. Must include a
            market factor under key "MKT" or "market".
        sector_returns: Optional {sector_name: daily ETF return series}. When
            supplied, sector exposures are estimated via a second regression
            of the first-pass residuals on the sector series.

    Returns:
        VarianceDecomposition with annualized variance components and their
        shares of total variance.
    """
    # --- Canonicalize factor keys for the regression helper ---
    canonical_factors: dict[str, pl.Series] = {}
    user_to_canonical: dict[str, str] = {}
    for user_name, series in factor_returns.items():
        canon = _canonical_factor_key(user_name)
        canonical_factors[canon] = series
        user_to_canonical[user_name] = canon

    if "market" not in canonical_factors:
        msg = "factor_returns must include a market factor (key 'MKT' or 'market')."
        raise ValueError(msg)

    # --- Common length across all relevant series ---
    series_lengths: list[int] = [returns_df.height]
    series_lengths.extend(s.len() for s in canonical_factors.values())
    if sector_returns:
        series_lengths.extend(s.len() for s in sector_returns.values())
    n = min(series_lengths)
    if n < 30:
        # Insufficient data: return a zero-everything result with idio_pct=1.0
        zero_style = {
            uname: 0.0 for uname in factor_returns if _canonical_factor_key(uname) != "market"
        }
        return VarianceDecomposition(
            total_var=0.0,
            market_var=0.0,
            style_var=zero_style,
            sector_var=0.0,
            idio_var=0.0,
            market_pct=0.0,
            style_pct=dict(zero_style),
            sector_pct=0.0,
            idio_pct=1.0,
        )

    weights = _portfolio_weights(portfolio)

    # --- Per-name regression (factor model helpers) ---
    # Map each user-facing factor label to its portfolio loading
    style_user_names = [u for u in factor_returns if _canonical_factor_key(u) != "market"]
    portfolio_beta_mkt = 0.0
    portfolio_beta_style: dict[str, float] = {u: 0.0 for u in style_user_names}
    # Per-name residual series after market+style regression, for sector pass + idio var
    name_residuals: dict[str, np.ndarray] = {}
    name_idio_vars: dict[str, float] = {}

    # Pre-extract aligned factor arrays for residual computation
    aligned_factors: dict[str, np.ndarray] = {
        canon: _align_to_length(series, n) for canon, series in canonical_factors.items()
    }

    for position in portfolio.positions:
        ticker = position.ticker
        if ticker not in returns_df.columns:
            continue
        w = weights.get(ticker, 0.0)
        ret_series = returns_df[ticker]
        if ret_series.len() < 30:
            continue

        decomp = multi_factor_regression(ret_series, canonical_factors, risk_free_rate=0.0)

        portfolio_beta_mkt += w * decomp.beta_market
        for user_name in style_user_names:
            canon = user_to_canonical[user_name]
            beta_k = None
            if canon == "smb":
                beta_k = decomp.beta_size
            elif canon == "hml":
                beta_k = decomp.beta_value
            elif canon == "momentum":
                beta_k = decomp.beta_momentum
            if beta_k is None:
                beta_k = 0.0
            portfolio_beta_style[user_name] += w * beta_k

        # Reconstruct residuals for sector pass / idio variance
        r_i = _align_to_length(ret_series, n)
        fitted = np.full(n, decomp.alpha / TRADING_DAYS_PER_YEAR)  # de-annualize intercept
        fitted = fitted + decomp.beta_market * aligned_factors["market"]
        if decomp.beta_size is not None and "smb" in aligned_factors:
            fitted = fitted + decomp.beta_size * aligned_factors["smb"]
        if decomp.beta_value is not None and "hml" in aligned_factors:
            fitted = fitted + decomp.beta_value * aligned_factors["hml"]
        if decomp.beta_momentum is not None and "momentum" in aligned_factors:
            fitted = fitted + decomp.beta_momentum * aligned_factors["momentum"]
        residuals = r_i - fitted
        name_residuals[ticker] = residuals

        # idio variance from residual (daily); annualize later. Assumes
        # independence across names (see module docstring).
        name_idio_vars[ticker] = float(np.var(residuals, ddof=1)) if residuals.size > 1 else 0.0

    # --- Market & style variance contributions (annualized) ---
    var_mkt_daily = float(np.var(aligned_factors["market"], ddof=1))
    market_var = portfolio_beta_mkt**2 * var_mkt_daily * TRADING_DAYS_PER_YEAR

    style_var: dict[str, float] = {}
    for user_name in style_user_names:
        canon = user_to_canonical[user_name]
        var_k_daily = float(np.var(aligned_factors[canon], ddof=1))
        style_var[user_name] = (
            portfolio_beta_style[user_name] ** 2 * var_k_daily * TRADING_DAYS_PER_YEAR
        )

    # --- Sector variance (second-pass regression on residuals) ---
    sector_var = 0.0
    if sector_returns:
        aligned_sectors: dict[str, np.ndarray] = {
            name: _align_to_length(series, n) for name, series in sector_returns.items()
        }
        sector_names = list(aligned_sectors.keys())
        if sector_names:
            X_sector = np.column_stack([aligned_sectors[s] for s in sector_names])
            X_sector_const = np.column_stack([np.ones(n), X_sector])
            portfolio_beta_sector: dict[str, float] = {s: 0.0 for s in sector_names}

            for position in portfolio.positions:
                ticker = position.ticker
                if ticker not in name_residuals:
                    continue
                w = weights.get(ticker, 0.0)
                y = name_residuals[ticker]
                try:
                    betas, *_ = np.linalg.lstsq(X_sector_const, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                sector_betas = betas[1:]
                # Update residuals so they exclude sector contribution for
                # the idio_var step below.
                fitted_sector = X_sector_const @ betas
                new_resid = y - fitted_sector
                name_residuals[ticker] = new_resid
                name_idio_vars[ticker] = (
                    float(np.var(new_resid, ddof=1)) if new_resid.size > 1 else 0.0
                )
                for i, sname in enumerate(sector_names):
                    portfolio_beta_sector[sname] += w * float(sector_betas[i])

            for sname in sector_names:
                var_s_daily = float(np.var(aligned_sectors[sname], ddof=1))
                sector_var += (
                    portfolio_beta_sector[sname] ** 2 * var_s_daily * TRADING_DAYS_PER_YEAR
                )

    # --- Idiosyncratic variance (independent-residual assumption) ---
    idio_var_daily = 0.0
    for ticker, idio_var_i in name_idio_vars.items():
        w_i = weights.get(ticker, 0.0)
        idio_var_daily += w_i**2 * idio_var_i
    idio_var = idio_var_daily * TRADING_DAYS_PER_YEAR

    total_var = market_var + sum(style_var.values()) + sector_var + idio_var

    if total_var <= 0:
        market_pct = 0.0
        style_pct = {k: 0.0 for k in style_var}
        sector_pct = 0.0
        idio_pct = 0.0
    else:
        market_pct = market_var / total_var
        style_pct = {k: v / total_var for k, v in style_var.items()}
        sector_pct = sector_var / total_var
        idio_pct = idio_var / total_var

    return VarianceDecomposition(
        total_var=total_var,
        market_var=market_var,
        style_var=style_var,
        sector_var=sector_var,
        idio_var=idio_var,
        market_pct=market_pct,
        style_pct=style_pct,
        sector_pct=sector_pct,
        idio_pct=idio_pct,
    )
