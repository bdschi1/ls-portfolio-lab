"""Daily portfolio snapshots for paper mode time-series tracking.

Captures the full portfolio state at end of each day.
Used for NAV curve construction, rolling metric calculation,
and drawdown analysis over the paper trading period.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PositionSnapshot(BaseModel):
    """Snapshot of a single position at a point in time."""

    ticker: str
    side: str
    shares: float
    price: float
    notional: float
    weight: float  # as fraction of NAV, signed
    pnl_dollars: float
    pnl_pct: float
    beta: float = 1.0
    sector: str = ""
    subsector: str = ""
    asset_type: str = "EQUITY"  # EQUITY, ETF, or OPTION


class DailySnapshot(BaseModel):
    """Complete portfolio state at end of day."""

    date: date
    timestamp: datetime = Field(default_factory=datetime.now)
    nav: float
    cash: float = 0.0  # cash balance at snapshot time
    gross_exposure: float
    net_exposure: float
    net_beta: float = 0.0
    portfolio_vol: float = 0.0
    sharpe: float = 0.0
    long_count: int = 0
    short_count: int = 0
    total_pnl_dollars: float = 0.0
    total_pnl_pct: float = 0.0
    positions: list[PositionSnapshot] = Field(default_factory=list)
    sector_exposures: dict[str, float] = Field(default_factory=dict)


class SnapshotStore:
    """
    JSONL-backed store for daily portfolio snapshots.

    One snapshot per day. If multiple snapshots on the same day,
    the latest one wins.
    """

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.touch()

    def save_snapshot(self, snapshot: DailySnapshot) -> None:
        """Append a daily snapshot. Overwrites if same date already exists."""
        # Read all, replace if exists, append if new
        existing = self.read_all()
        replaced = False
        new_list: list[DailySnapshot] = []
        for s in existing:
            if s.date == snapshot.date:
                new_list.append(snapshot)
                replaced = True
            else:
                new_list.append(s)

        if not replaced:
            new_list.append(snapshot)

        # Rewrite file
        self._write_all(new_list)

        logger.info(
            "Snapshot saved for %s: NAV=$%.0f, gross=%.1f%%, net=%.1f%%",
            snapshot.date,
            snapshot.nav,
            snapshot.gross_exposure * 100,
            snapshot.net_exposure * 100,
        )

    def read_all(self) -> list[DailySnapshot]:
        """Read all snapshots, sorted by date."""
        snapshots: list[DailySnapshot] = []
        if not self.store_path.exists():
            return snapshots

        with self.store_path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    snapshots.append(DailySnapshot(**data))
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Skipping malformed snapshot at line %d", line_num)

        return sorted(snapshots, key=lambda s: s.date)

    def read_date_range(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[DailySnapshot]:
        """Read snapshots within a date range."""
        snapshots = self.read_all()
        if start:
            snapshots = [s for s in snapshots if s.date >= start]
        if end:
            snapshots = [s for s in snapshots if s.date <= end]
        return snapshots

    def get_snapshot(self, target_date: date) -> DailySnapshot | None:
        """Get a specific day's snapshot."""
        for s in self.read_all():
            if s.date == target_date:
                return s
        return None

    def nav_series(self) -> list[tuple[date, float]]:
        """Extract NAV time series from snapshots."""
        return [(s.date, s.nav) for s in self.read_all()]

    def _write_all(self, snapshots: list[DailySnapshot]) -> None:
        """Rewrite the entire file with the given snapshots."""
        with self.store_path.open("w") as f:
            for s in sorted(snapshots, key=lambda s: s.date):
                f.write(s.model_dump_json() + "\n")

    @property
    def snapshot_count(self) -> int:
        """Total number of snapshots stored."""
        if not self.store_path.exists():
            return 0
        with self.store_path.open() as f:
            return sum(1 for line in f if line.strip())
