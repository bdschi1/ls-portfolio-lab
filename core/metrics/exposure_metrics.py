"""Exposure metrics: gross, net, sector, concentration.

Portfolio exposure analysis — how much capital is deployed, where, and how concentrated.
"""

from __future__ import annotations

from core.portfolio import Portfolio


def gross_exposure(portfolio: Portfolio) -> float:
    """Gross exposure as fraction of NAV: (long_notional + short_notional) / NAV."""
    return portfolio.gross_exposure


def net_exposure(portfolio: Portfolio) -> float:
    """Net exposure as fraction of NAV: (long_notional - short_notional) / NAV."""
    return portfolio.net_exposure


def long_short_ratio(portfolio: Portfolio) -> float:
    """Long notional / short notional."""
    return portfolio.long_short_ratio


def net_beta_exposure(
    portfolio: Portfolio,
    betas: dict[str, float],
) -> float:
    """
    Net beta-adjusted exposure.

    Σ(signed_weight_i × beta_i) for all positions.
    A portfolio with +6% net beta means it behaves like being
    6% net long the market on a beta-adjusted basis.
    """
    total = 0.0
    for p in portfolio.positions:
        weight = p.weight_in(portfolio.nav)
        beta = betas.get(p.ticker, 1.0)
        total += weight * beta
    return total


def sector_net_exposure(portfolio: Portfolio) -> dict[str, dict[str, float]]:
    """
    Net, long, short exposure per sector.

    Returns: {sector: {"long": float, "short": float, "net": float}}
    All values as fractions of NAV.
    """
    return portfolio.sector_exposure()


def subsector_net_exposure(portfolio: Portfolio) -> dict[str, dict[str, float]]:
    """
    Net, long, short exposure per subsector.

    Returns: {subsector: {"long": float, "short": float, "net": float}}
    """
    return portfolio.subsector_exposure()


def concentration_hhi(portfolio: Portfolio) -> float:
    """
    Herfindahl-Hirschman Index of portfolio concentration.

    HHI = Σ(weight_i²) where weights are absolute weights.
    Lower = more diversified. 1.0 = single position.
    """
    total = 0.0
    for p in portfolio.positions:
        w = p.abs_weight_in(portfolio.nav)
        total += w * w
    return total


def top_n_concentration(portfolio: Portfolio, n: int = 5) -> float:
    """Sum of the top N absolute position weights."""
    weights = sorted(
        [p.abs_weight_in(portfolio.nav) for p in portfolio.positions],
        reverse=True,
    )
    return sum(weights[:n])


def position_count_summary(portfolio: Portfolio) -> dict[str, int]:
    """Count summary of positions."""
    return {
        "total": portfolio.total_count,
        "long": portfolio.long_count,
        "short": portfolio.short_count,
    }


def exposure_summary(
    portfolio: Portfolio,
    betas: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Comprehensive exposure summary.

    Returns a flat dict of all key exposure metrics.
    """
    summary = {
        "gross_exposure": portfolio.gross_exposure,
        "net_exposure": portfolio.net_exposure,
        "long_notional": portfolio.long_notional,
        "short_notional": portfolio.short_notional,
        "long_short_ratio": portfolio.long_short_ratio,
        "long_count": float(portfolio.long_count),
        "short_count": float(portfolio.short_count),
        "total_count": float(portfolio.total_count),
        "hhi": concentration_hhi(portfolio),
        "top_5_concentration": top_n_concentration(portfolio, 5),
        "top_10_concentration": top_n_concentration(portfolio, 10),
    }

    if betas:
        summary["net_beta_exposure"] = net_beta_exposure(portfolio, betas)

    return summary


def check_exposure_limits(
    portfolio: Portfolio,
    betas: dict[str, float] | None = None,
    max_sector_net: float = 0.50,
    max_subsector_net: float = 0.50,
    max_single_position: float = 0.10,
    max_net_beta: float = 0.30,
    min_net_beta: float = -0.10,
    max_gross: float = 2.50,
) -> list[str]:
    """
    Check portfolio against exposure limits.

    Returns list of violation warnings (empty if all within limits).
    """
    warnings: list[str] = []

    # Gross exposure
    if portfolio.gross_exposure > max_gross:
        warnings.append(
            f"Gross exposure {portfolio.gross_exposure:.1%} exceeds limit {max_gross:.1%}"
        )

    # Single position concentration
    for p in portfolio.positions:
        w = p.abs_weight_in(portfolio.nav)
        if w > max_single_position:
            warnings.append(
                f"{p.ticker} weight {w:.1%} exceeds single-position limit {max_single_position:.1%}"
            )

    # Sector net exposure
    for sector, exposures in portfolio.sector_exposure().items():
        if abs(exposures["net"]) > max_sector_net:
            warnings.append(
                f"Sector '{sector}' net exposure {exposures['net']:.1%} "
                f"exceeds limit ±{max_sector_net:.1%}"
            )

    # Subsector net exposure
    for subsector, exposures in portfolio.subsector_exposure().items():
        if abs(exposures["net"]) > max_subsector_net:
            warnings.append(
                f"Subsector '{subsector}' net exposure {exposures['net']:.1%} "
                f"exceeds limit ±{max_subsector_net:.1%}"
            )

    # Beta limits
    if betas:
        nb = net_beta_exposure(portfolio, betas)
        if nb > max_net_beta:
            warnings.append(
                f"Net beta {nb:.3f} exceeds max limit {max_net_beta:.3f}"
            )
        if nb < min_net_beta:
            warnings.append(
                f"Net beta {nb:.3f} below min limit {min_net_beta:.3f}"
            )

    return warnings
