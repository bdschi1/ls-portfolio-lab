"""Append-only trade journal for paper portfolio mode.

Every trade is recorded with full context. This is the source of truth
for PM performance analytics. The log is immutable — trades can be added
but never modified or deleted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TradeRecord(BaseModel):
    """A single trade execution record."""

    timestamp: datetime = Field(default_factory=datetime.now)
    ticker: str
    action: str  # BUY, SELL, SHORT, COVER, ADD, REDUCE, EXIT
    shares: float
    price: float  # execution price at time of trade
    notional: float  # shares × price
    asset_type: str = "EQUITY"  # EQUITY, ETF, or OPTION
    side_after: str | None = None  # LONG, SHORT, or None if exited
    shares_after: float = 0.0  # total shares in position after trade
    portfolio_nav_after: float = 0.0
    sector: str = ""
    subsector: str = ""
    notes: str = ""

    @property
    def is_entry(self) -> bool:
        """True if this trade opens a new position."""
        return self.action in ("BUY", "SHORT")

    @property
    def is_exit(self) -> bool:
        """True if this trade closes a position."""
        return self.action == "EXIT" or self.shares_after == 0

    @property
    def is_add(self) -> bool:
        """True if this trade adds to an existing position."""
        return self.action in ("ADD",)


class TradeLog:
    """
    Append-only trade journal backed by JSONL file.

    Each line in the file is a JSON-serialized TradeRecord.
    """

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.log_path.exists():
            self.log_path.touch()

    def append(self, record: TradeRecord) -> None:
        """Append a trade record to the log."""
        with self.log_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        logger.info(
            "Trade logged: %s %s %.0f shares of %s @ %.2f",
            record.action,
            record.ticker,
            record.shares,
            record.ticker,
            record.price,
        )

    def append_batch(self, records: list[TradeRecord]) -> None:
        """Append multiple trade records."""
        with self.log_path.open("a") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[TradeRecord]:
        """Read all trade records from the log."""
        records: list[TradeRecord] = []
        if not self.log_path.exists():
            return records

        with self.log_path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(TradeRecord(**data))
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Skipping malformed trade record at line %d", line_num)

        return records

    def read_for_ticker(self, ticker: str) -> list[TradeRecord]:
        """Read all trade records for a specific ticker."""
        return [r for r in self.read_all() if r.ticker == ticker]

    def read_date_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[TradeRecord]:
        """Read trade records within a date range."""
        records = self.read_all()
        if start:
            records = [r for r in records if r.timestamp >= start]
        if end:
            records = [r for r in records if r.timestamp <= end]
        return records

    @property
    def trade_count(self) -> int:
        """Total number of trades in the log."""
        if not self.log_path.exists():
            return 0
        with self.log_path.open() as f:
            return sum(1 for line in f if line.strip())

    def identify_closed_trades(self) -> list[dict]:
        """
        Match entries with exits to identify completed (closed) trades.

        Returns list of dicts with: ticker, side, entry_date, exit_date,
        entry_price, exit_price, shares, pnl_dollars, pnl_pct, holding_days
        """
        records = self.read_all()

        # Track open positions and match with exits
        # Simple FIFO matching
        open_positions: dict[str, list[TradeRecord]] = {}
        closed_trades: list[dict] = []

        for record in records:
            ticker = record.ticker

            if record.action in ("BUY", "SHORT", "ADD"):
                if ticker not in open_positions:
                    open_positions[ticker] = []
                open_positions[ticker].append(record)

            elif record.action in ("SELL", "COVER", "REDUCE", "EXIT"):
                if ticker not in open_positions or not open_positions[ticker]:
                    continue

                # FIFO: match against earliest entry
                remaining_to_close = record.shares
                while remaining_to_close > 0 and open_positions.get(ticker):
                    entry = open_positions[ticker][0]
                    close_shares = min(remaining_to_close, entry.shares)

                    # Determine side from entry
                    side = "LONG" if entry.action in ("BUY", "ADD") else "SHORT"
                    direction = 1 if side == "LONG" else -1

                    pnl = direction * close_shares * (record.price - entry.price)
                    pnl_pct = direction * (record.price - entry.price) / entry.price

                    holding_days = (record.timestamp - entry.timestamp).days

                    closed_trades.append({
                        "ticker": ticker,
                        "side": side,
                        "sector": entry.sector,
                        "subsector": entry.subsector,
                        "entry_date": entry.timestamp.date(),
                        "exit_date": record.timestamp.date(),
                        "entry_price": entry.price,
                        "exit_price": record.price,
                        "shares": close_shares,
                        "pnl_dollars": pnl,
                        "pnl_pct": pnl_pct,
                        "holding_days": holding_days,
                    })

                    remaining_to_close -= close_shares
                    if close_shares >= entry.shares:
                        open_positions[ticker].pop(0)
                    else:
                        # Partially closed — reduce the entry
                        updated = entry.model_copy(
                            update={"shares": entry.shares - close_shares}
                        )
                        open_positions[ticker][0] = updated

        return closed_trades
