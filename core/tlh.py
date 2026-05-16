"""Tax-loss harvesting scanner with L/S short-leg awareness.

Long-leg TLH follows the plugin's standard playbook: scan positions with
unrealized losses, prioritize by holding period and loss magnitude, and
flag wash-sale conflicts.

Short-leg TLH is the part the wealth-management plugin doesn't cover.
Two L/S-specific rules apply at scan time:

- **Wash sale on short legs.** Covering a short at a loss and re-shorting
  "substantially identical" stock within 30 days triggers wash-sale
  treatment. We scan the JSONL trade log for the same ticker across a
  ±30-day window and block re-entry proposals that fall in it.

- **Wash sale on long legs.** Same mechanism — recent same-ticker activity
  in the trade log blocks the TLH candidate until the window clears.

Note on IRC §1259 short-against-the-box: `core.portfolio.Portfolio` rejects
duplicate tickers, so a single portfolio cannot hold both a long and a
short on the same name. The constructive-sale gate would only apply at a
multi-portfolio or proposal-time layer (not implemented here). Cross-book
checks should be added when multi-portfolio support lands.

Cost basis = `Position.entry_price`. Holding period uses `Position.entry_date`.
Unrealized loss is signed (negative) per `Position.pnl_dollars`.

Scope: this module identifies candidates and applies the L/S-specific
gates. Replacement-security suggestions are out of scope — the plugin's
skill handles that conversationally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from core.portfolio import Portfolio

logger = logging.getLogger(__name__)

_WASH_SALE_DAYS = 30
_SHORT_TERM_DAYS = 365


@dataclass
class TLHCandidate:
    """One position eligible (or blocked) for tax-loss harvesting."""

    ticker: str
    side: str  # LONG or SHORT
    shares: float
    cost_basis: float
    current_price: float
    unrealized_loss_dollars: float  # negative for a loss; zero or positive = not a candidate
    holding_days: int
    is_short_term: bool
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    wash_sale_window_end: date | None = None

    @property
    def actionable(self) -> bool:
        return not self.blocked


@dataclass
class TLHScan:
    """Output of a TLH scan."""

    candidates: list[TLHCandidate]
    warnings: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> list[TLHCandidate]:
        return [c for c in self.candidates if c.actionable]

    @property
    def blocked(self) -> list[TLHCandidate]:
        return [c for c in self.candidates if c.blocked]

    @property
    def total_loss_actionable(self) -> float:
        return sum(c.unrealized_loss_dollars for c in self.actionable)


def scan(
    portfolio: Portfolio,
    trade_log_records: list | None = None,
    *,
    today: date | None = None,
    min_loss_dollars: float = 100.0,
) -> TLHScan:
    """Scan portfolio for TLH candidates with L/S-aware gating.

    `trade_log_records` is a list of `history.trade_log.TradeRecord` (or
    None to skip wash-sale lookback — useful in tests). We accept the
    list directly rather than importing TradeLog to keep core/ free of
    history/ dependencies (one-directional layering).

    Returns a TLHScan with candidates flagged and gated.
    """
    today = today or date.today()
    records = trade_log_records or []

    candidates: list[TLHCandidate] = []
    warnings: list[str] = []

    for pos in portfolio.positions:
        loss = pos.pnl_dollars
        if loss >= -min_loss_dollars:
            continue  # not enough loss to bother

        holding_days = (today - pos.entry_date).days
        cand = TLHCandidate(
            ticker=pos.ticker,
            side=pos.side,
            shares=pos.shares,
            cost_basis=pos.entry_price,
            current_price=pos.current_price,
            unrealized_loss_dollars=loss,
            holding_days=holding_days,
            is_short_term=holding_days <= _SHORT_TERM_DAYS,
        )

        # Wash-sale lookback: any same-ticker trade in the last 30 days?
        ws_end = _wash_sale_window_end(pos.ticker, records, today)
        if ws_end is not None:
            cand.blocked = True
            cand.wash_sale_window_end = ws_end
            cand.block_reasons.append(
                f"wash-sale window open until {ws_end} "
                f"(same-ticker trade within {_WASH_SALE_DAYS}d)"
            )

        candidates.append(cand)

    # Sort: actionable first by absolute loss desc, then blocked
    candidates.sort(key=lambda c: (c.blocked, -abs(c.unrealized_loss_dollars)))

    return TLHScan(candidates=candidates, warnings=warnings)


def check_reentry_blocked(
    ticker: str,
    trade_log_records: list,
    *,
    today: date | None = None,
) -> tuple[bool, date | None]:
    """Return (blocked, window_end_date) for a proposed re-entry on `ticker`.

    Use this to gate a re-shorting (or re-buying) trade against the
    wash-sale window after a loss-realizing close.
    """
    today = today or date.today()
    end = _wash_sale_window_end(ticker, trade_log_records, today)
    return (end is not None, end)


def _wash_sale_window_end(
    ticker: str,
    records: list,
    today: date,
) -> date | None:
    """Find the latest closing trade on `ticker` within the last 30 days.

    If any matching trade is found within [today - 30d, today], return the
    date 30 days after it. Otherwise return None.

    We treat any trade on the ticker in the lookback window as a wash-sale
    trigger — even profitable closes can chain through subsequent loss
    realizations. The IRS rule is about the substantially-identical
    *acquisition*, but in practice scanning all same-ticker activity is
    the conservative call.
    """
    if not records:
        return None
    lookback_start = today - timedelta(days=_WASH_SALE_DAYS)
    latest: date | None = None
    for r in records:
        if r.ticker != ticker:
            continue
        ts = r.timestamp.date() if isinstance(r.timestamp, datetime) else r.timestamp
        if lookback_start <= ts <= today:
            if latest is None or ts > latest:
                latest = ts
    if latest is None:
        return None
    return latest + timedelta(days=_WASH_SALE_DAYS)
