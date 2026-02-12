"""Factor model: CAPM, Fama-French 3-factor, 4-factor (with momentum).

Decomposes returns into systematic (factor-driven) and idiosyncratic components.
Used for:
- Understanding what's driving portfolio returns
- Separating alpha from beta
- Measuring % systematic vs idiosyncratic risk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class FactorDecomposition:
    """Result of a factor regression for a single position or portfolio."""

    alpha: float  # annualized Jensen's alpha
    beta_market: float
    beta_size: float | None = None  # SMB (Fama-French)
    beta_value: float | None = None  # HML (Fama-French)
    beta_momentum: float | None = None  # UMD / MOM (Carhart)
    r_squared: float = 0.0  # % of variance explained by factors
    residual_vol: float = 0.0  # annualized idiosyncratic vol
    systematic_pct: float = 0.0  # R² as percentage
    idiosyncratic_pct: float = 0.0  # 1 - R²

    @property
    def factor_exposures(self) -> dict[str, float]:
        exposures = {"market": self.beta_market}
        if self.beta_size is not None:
            exposures["size"] = self.beta_size
        if self.beta_value is not None:
            exposures["value"] = self.beta_value
        if self.beta_momentum is not None:
            exposures["momentum"] = self.beta_momentum
        return exposures


TRADING_DAYS_PER_YEAR = 252


def capm_regression(
    returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.05,
) -> FactorDecomposition:
    """
    Single-factor CAPM regression.

    Ri - Rf = α + β(Rm - Rf) + ε
    """
    min_len = min(returns.len(), market_returns.len())
    if min_len < 30:
        return FactorDecomposition(
            alpha=0.0,
            beta_market=1.0,
            r_squared=0.0,
            residual_vol=0.0,
            systematic_pct=0.0,
            idiosyncratic_pct=100.0,
        )

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    ri = returns.tail(min_len).to_numpy() - daily_rf
    rm = market_returns.tail(min_len).to_numpy() - daily_rf

    # OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(rm, ri)

    r_squared = r_value**2
    residuals = ri - (intercept + slope * rm)
    resid_vol = float(np.std(residuals, ddof=2) * np.sqrt(TRADING_DAYS_PER_YEAR))
    alpha_annual = float(intercept * TRADING_DAYS_PER_YEAR)

    return FactorDecomposition(
        alpha=alpha_annual,
        beta_market=float(slope),
        r_squared=float(r_squared),
        residual_vol=resid_vol,
        systematic_pct=float(r_squared * 100),
        idiosyncratic_pct=float((1 - r_squared) * 100),
    )


def multi_factor_regression(
    returns: pl.Series,
    factor_returns: dict[str, pl.Series],
    risk_free_rate: float = 0.05,
) -> FactorDecomposition:
    """
    Multi-factor regression (Fama-French 3 or 4 factor).

    Ri - Rf = α + β_mkt(Rm-Rf) + β_smb(SMB) + β_hml(HML) [+ β_mom(MOM)] + ε

    factor_returns should contain keys: "market", and optionally "smb", "hml", "momentum"
    """
    if "market" not in factor_returns:
        msg = "factor_returns must contain 'market' key"
        raise ValueError(msg)

    # Determine minimum common length
    min_len = returns.len()
    for fr in factor_returns.values():
        min_len = min(min_len, fr.len())

    if min_len < 30:
        return FactorDecomposition(
            alpha=0.0, beta_market=1.0,
            r_squared=0.0, residual_vol=0.0,
            systematic_pct=0.0, idiosyncratic_pct=100.0,
        )

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    y = returns.tail(min_len).to_numpy() - daily_rf

    # Build factor matrix
    factor_names = ["market"]
    X_cols = [factor_returns["market"].tail(min_len).to_numpy() - daily_rf]

    for name in ["smb", "hml", "momentum"]:
        if name in factor_returns:
            factor_names.append(name)
            X_cols.append(factor_returns[name].tail(min_len).to_numpy())

    X = np.column_stack(X_cols)
    # Add intercept
    X_with_const = np.column_stack([np.ones(min_len), X])

    # OLS via least squares
    try:
        betas, residuals_ss, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return FactorDecomposition(
            alpha=0.0, beta_market=1.0,
            r_squared=0.0, residual_vol=0.0,
            systematic_pct=0.0, idiosyncratic_pct=100.0,
        )

    intercept = betas[0]
    factor_betas = betas[1:]

    # Compute R²
    y_hat = X_with_const @ betas
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_squared = max(0.0, r_squared)  # clamp

    residuals = y - y_hat
    resid_vol = float(np.std(residuals, ddof=len(betas)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    alpha_annual = float(intercept * TRADING_DAYS_PER_YEAR)

    result = FactorDecomposition(
        alpha=alpha_annual,
        beta_market=float(factor_betas[0]),
        r_squared=float(r_squared),
        residual_vol=resid_vol,
        systematic_pct=float(r_squared * 100),
        idiosyncratic_pct=float((1 - r_squared) * 100),
    )

    # Map remaining factor betas
    for i, name in enumerate(factor_names[1:], start=1):
        if name == "smb":
            result.beta_size = float(factor_betas[i])
        elif name == "hml":
            result.beta_value = float(factor_betas[i])
        elif name == "momentum":
            result.beta_momentum = float(factor_betas[i])

    return result


def portfolio_factor_decomposition(
    portfolio_returns: pl.Series,
    market_returns: pl.Series,
    risk_free_rate: float = 0.05,
    factor_returns: dict[str, pl.Series] | None = None,
) -> FactorDecomposition:
    """
    Factor decomposition for the overall portfolio.

    Uses multi-factor if factor_returns provided, otherwise CAPM.
    """
    if factor_returns is not None:
        return multi_factor_regression(portfolio_returns, factor_returns, risk_free_rate)
    return capm_regression(portfolio_returns, market_returns, risk_free_rate)


def build_proxy_factors(
    returns_df: pl.DataFrame,
    market_ticker: str = "SPY",
    size_long: str = "IWM",
    size_short: str = "SPY",
    value_long: str = "IWD",
    value_short: str = "IWF",
    momentum_long: str = "MTUM",
) -> dict[str, pl.Series]:
    """
    Build proxy factor returns from ETFs.

    This is an approximation — the proper approach uses Fama-French factor data
    from Kenneth French's website. But for a practical tool, ETF proxies work:

    - Market: SPY returns
    - Size (SMB proxy): IWM - SPY (small minus large)
    - Value (HML proxy): IWD - IWF (value minus growth)
    - Momentum: MTUM returns (or MTUM - SPY)

    All return series should be in the returns_df with daily simple returns.
    """
    factors: dict[str, pl.Series] = {}

    if market_ticker in returns_df.columns:
        factors["market"] = returns_df[market_ticker]

    if size_long in returns_df.columns and size_short in returns_df.columns:
        factors["smb"] = returns_df[size_long] - returns_df[size_short]

    if value_long in returns_df.columns and value_short in returns_df.columns:
        factors["hml"] = returns_df[value_long] - returns_df[value_short]

    if momentum_long in returns_df.columns:
        if market_ticker in returns_df.columns:
            factors["momentum"] = returns_df[momentum_long] - returns_df[market_ticker]
        else:
            factors["momentum"] = returns_df[momentum_long]

    return factors
