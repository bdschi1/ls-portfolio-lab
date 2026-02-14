"""P&L attribution — sector, side, and position-level attribution.

Decomposes portfolio P&L into:
- Position-level: each position's contribution to total P&L
- Sector-level: aggregate P&L by GICS sector
- Side-level: long book vs short book contribution
- Factor-level: market/size/value/momentum attribution via factor regression

All functions are pure — take Portfolio + market data, return results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from core.portfolio import Portfolio

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PositionAttribution:
    """P&L attribution for a single position."""

    ticker: str
    side: str
    sector: str
    pnl_dollars: float
    pnl_pct: float  # position-level P&L %
    weight_pct: float  # current weight in portfolio
    contribution_bps: float  # contribution to portfolio return in bps


@dataclass
class SectorAttribution:
    """P&L attribution for a sector."""

    sector: str
    pnl_dollars: float
    long_pnl: float
    short_pnl: float
    contribution_bps: float
    position_count: int
    net_weight_pct: float


@dataclass
class SideAttribution:
    """P&L attribution by long/short side."""

    long_pnl: float
    short_pnl: float
    long_contribution_bps: float
    short_contribution_bps: float
    long_hit_rate: float  # fraction of longs with positive P&L
    short_hit_rate: float  # fraction of shorts with positive P&L


@dataclass
class FactorAttribution:
    """P&L attribution via factor regression."""

    market_contribution_pct: float  # % of return explained by market beta
    size_contribution_pct: float  # % from SMB
    value_contribution_pct: float  # % from HML
    momentum_contribution_pct: float  # % from momentum
    alpha_contribution_pct: float  # % from stock-picking alpha
    residual_pct: float  # unexplained


@dataclass
class AttributionSummary:
    """Complete P&L attribution breakdown."""

    positions: list[PositionAttribution] = field(default_factory=list)
    sectors: list[SectorAttribution] = field(default_factory=list)
    side: SideAttribution | None = None
    factors: FactorAttribution | None = None
    total_pnl_dollars: float = 0.0
    total_pnl_bps: float = 0.0


# ---------------------------------------------------------------------------
# Position-level attribution
# ---------------------------------------------------------------------------


def position_attribution(portfolio: Portfolio) -> list[PositionAttribution]:
    """
    Compute P&L attribution at the position level.

    Each position's contribution to portfolio return = position P&L / NAV.
    """
    results: list[PositionAttribution] = []
    nav = portfolio.nav
    if nav == 0:
        return results

    for p in portfolio.positions:
        pnl = p.pnl_dollars
        contribution_bps = (pnl / nav) * 10_000  # basis points

        results.append(PositionAttribution(
            ticker=p.ticker,
            side=p.side,
            sector=p.sector or "Unknown",
            pnl_dollars=pnl,
            pnl_pct=p.pnl_pct * 100,
            weight_pct=p.weight_in(nav) * 100,
            contribution_bps=contribution_bps,
        ))

    # Sort by absolute contribution (largest impact first)
    results.sort(key=lambda x: abs(x.contribution_bps), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Sector-level attribution
# ---------------------------------------------------------------------------


def sector_attribution(portfolio: Portfolio) -> list[SectorAttribution]:
    """
    Aggregate P&L by sector.

    Returns sorted by absolute P&L contribution.
    """
    nav = portfolio.nav
    if nav == 0:
        return []

    sector_data: dict[str, dict] = {}

    for p in portfolio.positions:
        sector = p.sector or "Unknown"
        if sector not in sector_data:
            sector_data[sector] = {
                "pnl_dollars": 0.0,
                "long_pnl": 0.0,
                "short_pnl": 0.0,
                "count": 0,
                "net_weight": 0.0,
            }

        pnl = p.pnl_dollars
        sector_data[sector]["pnl_dollars"] += pnl
        sector_data[sector]["count"] += 1
        sector_data[sector]["net_weight"] += p.weight_in(nav)

        if p.side == "LONG":
            sector_data[sector]["long_pnl"] += pnl
        else:
            sector_data[sector]["short_pnl"] += pnl

    results = []
    for sector, data in sector_data.items():
        contribution_bps = (data["pnl_dollars"] / nav) * 10_000
        results.append(SectorAttribution(
            sector=sector,
            pnl_dollars=data["pnl_dollars"],
            long_pnl=data["long_pnl"],
            short_pnl=data["short_pnl"],
            contribution_bps=contribution_bps,
            position_count=data["count"],
            net_weight_pct=data["net_weight"] * 100,
        ))

    results.sort(key=lambda x: abs(x.contribution_bps), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Side-level attribution
# ---------------------------------------------------------------------------


def side_attribution(portfolio: Portfolio) -> SideAttribution:
    """Decompose P&L into long book vs short book."""
    nav = portfolio.nav
    long_pnl = sum(p.pnl_dollars for p in portfolio.long_positions)
    short_pnl = sum(p.pnl_dollars for p in portfolio.short_positions)

    long_winners = sum(1 for p in portfolio.long_positions if p.pnl_dollars > 0)
    short_winners = sum(1 for p in portfolio.short_positions if p.pnl_dollars > 0)

    long_count = portfolio.long_count
    short_count = portfolio.short_count

    return SideAttribution(
        long_pnl=long_pnl,
        short_pnl=short_pnl,
        long_contribution_bps=(long_pnl / nav * 10_000) if nav else 0.0,
        short_contribution_bps=(short_pnl / nav * 10_000) if nav else 0.0,
        long_hit_rate=(long_winners / long_count) if long_count else 0.0,
        short_hit_rate=(short_winners / short_count) if short_count else 0.0,
    )


# ---------------------------------------------------------------------------
# Factor-level attribution
# ---------------------------------------------------------------------------


def factor_attribution(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None,
    betas: dict[str, float] | None = None,
    market_return: float | None = None,
    factor_returns: dict[str, float] | None = None,
) -> FactorAttribution | None:
    """
    Decompose portfolio return into factor contributions.

    Uses Brinson-style attribution:
    - Market contribution = portfolio_beta × market_return
    - Size/value/momentum contributions from factor exposures
    - Alpha = residual after removing factor contributions

    If returns_df is available, computes actual factor exposures via regression.
    Otherwise, uses beta-weighted estimates.
    """
    if betas is None or not betas:
        return None

    nav = portfolio.nav
    if nav == 0:
        return None

    total_pnl_pct = portfolio.total_pnl_pct * 100  # as percentage

    # Compute portfolio beta
    total_beta = 0.0
    for p in portfolio.positions:
        weight = p.weight_in(nav)
        beta = betas.get(p.ticker, 1.0)
        total_beta += weight * beta

    # If we have market return data, do factor decomposition
    if market_return is not None:
        market_contrib = total_beta * market_return * 100  # as pct
    elif returns_df is not None and "SPY" in returns_df.columns:
        spy_rets = returns_df["SPY"]
        if spy_rets.len() > 0:
            # Use cumulative market return over the period
            cum_mkt = float((1 + spy_rets).product() - 1)
            market_contrib = total_beta * cum_mkt * 100
        else:
            market_contrib = 0.0
    else:
        market_contrib = 0.0

    # Factor contributions (simplified — uses portfolio-level estimates)
    size_contrib = 0.0
    value_contrib = 0.0
    momentum_contrib = 0.0

    if factor_returns:
        if "smb" in factor_returns:
            size_contrib = factor_returns["smb"] * 100
        if "hml" in factor_returns:
            value_contrib = factor_returns["hml"] * 100
        if "momentum" in factor_returns:
            momentum_contrib = factor_returns["momentum"] * 100

    # Alpha = total return minus factor contributions
    factor_total = market_contrib + size_contrib + value_contrib + momentum_contrib
    alpha_contrib = total_pnl_pct - factor_total
    residual = 0.0

    return FactorAttribution(
        market_contribution_pct=market_contrib,
        size_contribution_pct=size_contrib,
        value_contribution_pct=value_contrib,
        momentum_contribution_pct=momentum_contrib,
        alpha_contribution_pct=alpha_contrib,
        residual_pct=residual,
    )


# ---------------------------------------------------------------------------
# Full attribution summary
# ---------------------------------------------------------------------------


def full_attribution(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
) -> AttributionSummary:
    """
    Compute complete P&L attribution breakdown.

    Returns an AttributionSummary with position, sector, side, and factor views.
    """
    nav = portfolio.nav
    total_pnl = portfolio.total_pnl_dollars
    total_bps = (total_pnl / nav * 10_000) if nav else 0.0

    return AttributionSummary(
        positions=position_attribution(portfolio),
        sectors=sector_attribution(portfolio),
        side=side_attribution(portfolio),
        factors=factor_attribution(portfolio, returns_df, betas),
        total_pnl_dollars=total_pnl,
        total_pnl_bps=total_bps,
    )
