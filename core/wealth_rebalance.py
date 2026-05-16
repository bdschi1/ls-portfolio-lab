"""Wealth-management rebalancer — target-weights schema → TradeBasket.

The plugin's `portfolio-rebalance` skill speaks in *target weights with drift
bands*. This module translates that schema into a `TradeBasket` consumable by
`core/trade_impact.py`. It does NOT run SLSQP — that's `core/rebalancer.py`'s
job for net-beta/vol-target L/S rebalances. The two engines run side by side.

Signed weights: positive = long target, negative = short target. A target of
zero means "exit this position." Tickers in the portfolio but not in the
target dict are left alone.

Bands gate trading: a name only trades if |drift| > band. Default band is
50bps (0.005). The plugin permits 3–5% bands for client portfolios; that
maps to 0.03–0.05 here.

If the absolute drift basket would exceed `TradeBasket.max_length=10`, we
prioritize the largest absolute drifts and drop the rest with a warning.

Side flips (e.g. target moves a position from LONG +3% to SHORT -2%) are
rejected — they need an explicit two-step rebalance to keep the trade-cost
accounting honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from core.portfolio import Portfolio, ProposedTrade, TradeBasket

logger = logging.getLogger(__name__)

_DEFAULT_BAND = 0.005  # 50bps
_MAX_BASKET = 10  # mirrors TradeBasket.max_length


class TargetWeight(BaseModel):
    """A single target allocation entry from the rebalance skill."""

    ticker: str
    target_weight: float = Field(
        description="Signed weight; positive=long, negative=short, zero=exit",
    )
    band: float = Field(
        default=_DEFAULT_BAND,
        ge=0.0,
        description="Drift tolerance; trade fires only if |drift| > band",
    )


@dataclass
class WealthRebalanceRequest:
    """Input for a target-weights rebalance."""

    targets: list[TargetWeight]
    description: str = "Wealth-management target-weights rebalance"


@dataclass
class WealthRebalanceResult:
    """Output: the TradeBasket plus the drift table that produced it."""

    basket: TradeBasket
    drift_rows: list[dict]
    warnings: list[str] = field(default_factory=list)


def compute_drift_rows(portfolio: Portfolio, targets: list[TargetWeight]) -> list[dict]:
    """Return a drift row per target ticker. Does not generate trades."""
    current_weights = portfolio.weight_vector()  # signed
    rows: list[dict] = []
    for t in targets:
        cur = current_weights.get(t.ticker, 0.0)
        drift = cur - t.target_weight
        rows.append({
            "ticker": t.ticker,
            "target_weight": t.target_weight,
            "current_weight": cur,
            "drift": drift,
            "band": t.band,
            "in_band": abs(drift) <= t.band,
        })
    return rows


def build_basket(portfolio: Portfolio, request: WealthRebalanceRequest) -> WealthRebalanceResult:
    """Translate target-weights + bands into a TradeBasket.

    Trade selection:
      1. Compute drift = current_weight - target_weight (both signed).
      2. Skip names where |drift| <= band.
      3. For each out-of-band name, emit one trade. Dollar amount =
         |drift| * NAV. Action depends on sign and position state.
      4. If >10 trades qualify, keep the 10 largest |drift| trades.
    """
    warnings: list[str] = []
    drift_rows = compute_drift_rows(portfolio, request.targets)

    out_of_band = [r for r in drift_rows if not r["in_band"]]
    out_of_band.sort(key=lambda r: abs(r["drift"]), reverse=True)

    if len(out_of_band) > _MAX_BASKET:
        dropped = [r["ticker"] for r in out_of_band[_MAX_BASKET:]]
        warnings.append(
            f"Trade basket capped at {_MAX_BASKET}; deferred {len(dropped)} smaller "
            f"drift trades: {', '.join(dropped)}"
        )
        out_of_band = out_of_band[:_MAX_BASKET]

    trades: list[ProposedTrade] = []
    for row in out_of_band:
        ticker = row["ticker"]
        target_w = row["target_weight"]
        current_w = row["current_weight"]
        drift = row["drift"]
        action, dollar_amount = _resolve_trade(
            ticker=ticker,
            target_w=target_w,
            current_w=current_w,
            drift=drift,
            nav=portfolio.nav,
            portfolio=portfolio,
            warnings=warnings,
        )
        if action is None:
            continue
        trades.append(ProposedTrade(
            ticker=ticker,
            action=action,
            dollar_amount=dollar_amount,
            notes=f"wealth-rebalance: drift {drift * 100:.2f}% vs target {target_w * 100:.2f}%",
        ))

    basket = TradeBasket(trades=trades, description=request.description)
    return WealthRebalanceResult(basket=basket, drift_rows=drift_rows, warnings=warnings)


def _resolve_trade(
    *,
    ticker: str,
    target_w: float,
    current_w: float,
    drift: float,
    nav: float,
    portfolio: Portfolio,
    warnings: list[str],
) -> tuple[str | None, float | None]:
    """Pick the action + dollar size for one out-of-band name.

    Side-flip cases (target sign differs from current sign and both nonzero)
    are rejected; user must run two separate rebalances to keep cost
    accounting clean. We do allow current=0 → new entry, and any → 0 exit.
    """
    dollar_gap = abs(drift) * nav
    existing = portfolio.get_position(ticker)

    if target_w == 0.0:
        if existing is None:
            return None, None
        return "EXIT", None

    # New entry
    if existing is None:
        if target_w > 0:
            return "BUY", abs(target_w) * nav
        return "SHORT", abs(target_w) * nav

    # Sign-flip detection — read side directly off the position
    is_long = existing.side == "LONG"
    target_is_long = target_w > 0
    if is_long != target_is_long:
        warnings.append(
            f"{ticker}: side flip from {existing.side} to "
            f"{'LONG' if target_is_long else 'SHORT'} requires a manual two-step rebalance."
        )
        return None, None

    # Same side — ADD vs REDUCE
    if is_long:
        # current_w is positive; target_w positive. drift = current - target.
        # drift > 0 means we're over target → REDUCE; drift < 0 means under → ADD.
        return ("REDUCE" if drift > 0 else "ADD"), dollar_gap

    # Short side — current_w is negative; target_w negative.
    # drift = current_w - target_w. If current = -0.04, target = -0.02, drift = -0.02 →
    # we're more short than target → COVER (reduce short). If drift > 0 → already less
    # short than target → ADD to short (SHORT more).
    return ("COVER" if drift < 0 else "SHORT"), dollar_gap
