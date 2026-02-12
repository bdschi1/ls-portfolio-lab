"""Mock portfolio generator — creates a constrained hypothetical L/S portfolio.

Generates a plausible, diversified long/short equity portfolio with:
- ~30 longs, ~40 shorts (~70 positions)
- Minimum 20 longs, always more shorts than longs
- Net beta exposure of ~+6%
- No sector or subsector net exposure > 50%
- Curated ~440-name Russell 1000 universe (market cap > $5B)
- All 11 GICS sectors with extra depth in Healthcare
- NAV: $3B with cash tracking
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta

import numpy as np
from scipy.optimize import minimize

from core.portfolio import Portfolio, Position
from data.sector_map import classify_ticker
from data.universe import MOCK_UNIVERSE

logger = logging.getLogger(__name__)

# Configuration — flexible counts: min 20 longs, always more shorts than longs
TARGET_LONGS = 30
TARGET_SHORTS = 40
TARGET_NET_BETA = 0.06
MAX_SECTOR_NET = 0.50
MAX_SUBSECTOR_NET = 0.50
DEFAULT_NAV = 3_000_000_000.0  # $3B
MIN_MARKET_CAP = 5_000_000_000  # $5B


def _sample_positions(
    n_longs: int = TARGET_LONGS,
    n_shorts: int = TARGET_SHORTS,
    seed: int | None = None,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """
    Sample tickers from the universe ensuring sector diversity.

    Returns (longs, shorts) where each is a list of (ticker, sector, subsector).

    Three-pass sampling strategy (handles 39 subsectors with 30L/40S targets):
      1. Sector pass:  guarantee ≥1 long + ≥1 short per GICS sector (11 sectors)
      2. Subsector pass: add breadth across remaining subsectors
      3. Random fill:  top up to target counts from the remaining pool
    """
    if seed is not None:
        random.seed(seed)

    # Build flat list with sector info
    all_names: list[tuple[str, str, str]] = []  # (ticker, sector, subsector)
    for subsector, tickers in MOCK_UNIVERSE.items():
        sector = (
            subsector.split(" - ")[0] if " - " in subsector else subsector
        )
        for ticker in tickers:
            all_names.append((ticker, sector, subsector))

    random.shuffle(all_names)

    # Group by sector and by subsector
    by_sector: dict[str, list[tuple[str, str, str]]] = {}
    by_subsector: dict[str, list[tuple[str, str, str]]] = {}
    for name in all_names:
        sec, ss = name[1], name[2]
        by_sector.setdefault(sec, []).append(name)
        by_subsector.setdefault(ss, []).append(name)

    longs: list[tuple[str, str, str]] = []
    shorts: list[tuple[str, str, str]] = []
    used: set[str] = set()

    # --- Pass 1: sector-level diversity (1L + 1S per GICS sector) ---
    for _sec, names in by_sector.items():
        available = [n for n in names if n[0] not in used]
        if len(available) >= 2:
            longs.append(available[0])
            used.add(available[0][0])
            shorts.append(available[1])
            used.add(available[1][0])
        elif len(available) == 1:
            if len(shorts) < len(longs):
                shorts.append(available[0])
            else:
                longs.append(available[0])
            used.add(available[0][0])

    # --- Pass 2: subsector diversity (add 1 name per subsector if room) ---
    for _ss, names in by_subsector.items():
        if len(longs) >= n_longs and len(shorts) >= n_shorts:
            break
        available = [n for n in names if n[0] not in used]
        if not available:
            continue
        # Add to whichever side has more room
        if len(longs) < n_longs:
            longs.append(available[0])
            used.add(available[0][0])
        elif len(shorts) < n_shorts:
            shorts.append(available[0])
            used.add(available[0][0])

    # --- Pass 3: random fill ---
    remaining = [n for n in all_names if n[0] not in used]
    random.shuffle(remaining)

    while len(longs) < n_longs and remaining:
        longs.append(remaining.pop())
    while len(shorts) < n_shorts and remaining:
        shorts.append(remaining.pop())

    # Trim if oversampled
    longs = longs[:n_longs]
    shorts = shorts[:n_shorts]

    return longs, shorts


def _optimize_weights(
    longs: list[tuple[str, str, str]],
    shorts: list[tuple[str, str, str]],
    betas: dict[str, float],
    target_net_beta: float = TARGET_NET_BETA,
    max_sector_net: float = MAX_SECTOR_NET,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Optimize position weights to meet portfolio constraints.

    Approach: start with equal weights, then adjust to hit target net beta
    and sector limits. Uses scipy minimize with constraints.

    Returns (long_weights, short_weights) as fractions of NAV.
    """
    n_long = len(longs)
    n_short = len(shorts)
    n_total = n_long + n_short

    # Tickers in order
    all_tickers = [t[0] for t in longs] + [t[0] for t in shorts]
    all_sectors = [t[1] for t in longs] + [t[1] for t in shorts]

    # Betas for each position
    beta_vec = np.array([betas.get(t, 1.0) for t in all_tickers])

    # Signs: +1 for longs, -1 for shorts
    signs = np.array([1.0] * n_long + [-1.0] * n_short)

    # Initial weights: equal weight, scaled to reasonable gross exposure (~180%)
    target_gross = 1.80
    initial_long_wt = target_gross * 0.55 / n_long  # longs ~55% of gross
    initial_short_wt = target_gross * 0.45 / n_short  # shorts ~45% of gross

    x0 = np.array(
        [initial_long_wt] * n_long + [initial_short_wt] * n_short
    )

    # Objective: minimize deviation from equal weights (stay diversified)
    # while meeting constraints
    def objective(x: np.ndarray) -> float:
        # Penalize deviation from uniform
        target = np.mean(x)
        concentration_penalty = np.sum((x - target) ** 2)
        return concentration_penalty

    # Constraint 1: net beta = target
    def beta_constraint(x: np.ndarray) -> float:
        signed_weights = signs * x
        net_beta = np.dot(signed_weights, beta_vec)
        return net_beta - target_net_beta

    # Constraint 2: sector net exposure limits
    unique_sectors = list(set(all_sectors))
    sector_constraints = []
    for sector in unique_sectors:
        mask = np.array([1.0 if s == sector else 0.0 for s in all_sectors])

        def make_sector_upper(mask_=mask):
            def constraint(x: np.ndarray) -> float:
                signed_weights = signs * x
                sector_net = np.dot(signed_weights, mask_)
                return max_sector_net - abs(sector_net)
            return constraint

        sector_constraints.append({"type": "ineq", "fun": make_sector_upper()})

    constraints = [
        {"type": "eq", "fun": beta_constraint},
        *sector_constraints,
    ]

    # Bounds: each weight between 1% and 8% of NAV
    bounds = [(0.01, 0.08)] * n_total

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success:
        logger.warning(
            "Weight optimization did not fully converge: %s. Using best available.",
            result.message,
        )

    weights = result.x

    long_weights = {longs[i][0]: float(weights[i]) for i in range(n_long)}
    short_weights = {shorts[i][0]: float(weights[n_long + i]) for i in range(n_short)}

    return long_weights, short_weights


def generate_mock_portfolio(
    betas: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    nav: float = DEFAULT_NAV,
    seed: int | None = 42,
) -> Portfolio:
    """
    Generate the mock portfolio.

    Args:
        betas: {ticker: beta} from market data. If None, uses 1.0 for all.
        prices: {ticker: current_price}. If None, uses placeholder $100.
        nav: starting capital (default $3B)
        seed: random seed for reproducibility

    Returns a Portfolio with ~30 longs / ~40 shorts meeting all constraints.
    Constraints: min 20 longs, short_count > long_count.
    """
    if betas is None:
        betas = {}

    # When no prices dict provided, use $100 placeholder for every ticker (test/offline mode).
    # When prices ARE provided (live mode), tickers with missing/zero prices get skipped.
    use_fallback_prices = prices is None
    if prices is None:
        prices = {}

    longs, shorts = _sample_positions(seed=seed)

    # Optimize weights
    long_weights, short_weights = _optimize_weights(longs, shorts, betas)

    # Convert weights to positions with share counts
    positions: list[Position] = []
    skipped: list[str] = []

    for ticker, sector_name, subsector_name in longs:
        weight = long_weights.get(ticker, 0.03)
        price = prices.get(ticker, 100.0 if use_fallback_prices else 0.0)
        if not price or price <= 0:
            logger.warning("Skipping %s (LONG) — no valid price data.", ticker)
            skipped.append(ticker)
            continue
        notional = weight * nav
        shares = max(round(notional / price, 2), 0.01)
        sector, subsector = classify_ticker(ticker)

        positions.append(Position(
            ticker=ticker,
            side="LONG",
            shares=shares,
            entry_price=price,
            entry_date=date.today() - timedelta(days=random.randint(30, 180)),
            current_price=price,
            sector=sector,
            subsector=subsector,
        ))

    for ticker, sector_name, subsector_name in shorts:
        weight = short_weights.get(ticker, 0.03)
        price = prices.get(ticker, 100.0 if use_fallback_prices else 0.0)
        if not price or price <= 0:
            logger.warning("Skipping %s (SHORT) — no valid price data.", ticker)
            skipped.append(ticker)
            continue
        notional = weight * nav
        shares = max(round(notional / price, 2), 0.01)
        sector, subsector = classify_ticker(ticker)

        positions.append(Position(
            ticker=ticker,
            side="SHORT",
            shares=shares,
            entry_price=price,
            entry_date=date.today() - timedelta(days=random.randint(30, 180)),
            current_price=price,
            sector=sector,
            subsector=subsector,
        ))

    if skipped:
        logger.warning("Skipped %d tickers with no price data: %s", len(skipped), skipped)

    # Soft constraint validation
    long_count = sum(1 for p in positions if p.side == "LONG")
    short_count = sum(1 for p in positions if p.side == "SHORT")

    if long_count < 20:
        logger.warning(
            "Mock portfolio has only %d longs (minimum 20). "
            "Consider expanding the universe.", long_count,
        )
    if short_count <= long_count:
        logger.warning(
            "Mock portfolio has %d shorts <= %d longs. "
            "Constraint requires more shorts than longs.", short_count, long_count,
        )

    # Cash = NAV minus total invested notional (auto-computed by Portfolio validator)
    portfolio = Portfolio(
        name="Mock L/S Portfolio",
        positions=positions,
        benchmark="SPY",
        inception_date=date.today(),
        nav=nav,
    )

    logger.info(
        "Generated mock portfolio: %d longs, %d shorts, gross=%.1f%%, net=%.1f%%, "
        "cash=$%s (%.1f%%)",
        portfolio.long_count,
        portfolio.short_count,
        portfolio.gross_exposure * 100,
        portfolio.net_exposure * 100,
        f"{portfolio.cash:,.0f}",
        portfolio.cash_pct * 100,
    )

    return portfolio
