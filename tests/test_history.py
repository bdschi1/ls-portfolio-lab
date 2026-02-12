"""Tests for history/ modules — trade log, snapshots, performance."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from history.trade_log import TradeLog, TradeRecord
from history.snapshot import DailySnapshot, PositionSnapshot, SnapshotStore


class TestTradeLog:
    def _make_log(self, tmp_path: Path) -> TradeLog:
        return TradeLog(tmp_path / "test_trades.jsonl")

    def test_append_and_read(self, tmp_path):
        log = self._make_log(tmp_path)
        record = TradeRecord(
            ticker="AAPL", action="BUY", shares=100, price=150.0,
            notional=15000.0, side_after="LONG", shares_after=100,
        )
        log.append(record)
        records = log.read_all()
        assert len(records) == 1
        assert records[0].ticker == "AAPL"

    def test_multiple_appends(self, tmp_path):
        log = self._make_log(tmp_path)
        for i in range(5):
            record = TradeRecord(
                ticker=f"T{i}", action="BUY", shares=10, price=100.0,
                notional=1000.0,
            )
            log.append(record)
        assert log.trade_count == 5

    def test_read_for_ticker(self, tmp_path):
        log = self._make_log(tmp_path)
        log.append(TradeRecord(ticker="AAPL", action="BUY", shares=100, price=150.0, notional=15000.0))
        log.append(TradeRecord(ticker="MSFT", action="BUY", shares=50, price=300.0, notional=15000.0))
        log.append(TradeRecord(ticker="AAPL", action="ADD", shares=50, price=155.0, notional=7750.0))

        aapl_trades = log.read_for_ticker("AAPL")
        assert len(aapl_trades) == 2

    def test_identify_closed_trades(self, tmp_path):
        log = self._make_log(tmp_path)
        # Open and close AAPL
        log.append(TradeRecord(
            ticker="AAPL", action="BUY", shares=100, price=150.0,
            notional=15000.0, side_after="LONG", shares_after=100,
            sector="Tech",
        ))
        log.append(TradeRecord(
            ticker="AAPL", action="SELL", shares=100, price=170.0,
            notional=17000.0, side_after=None, shares_after=0,
            sector="Tech",
        ))

        closed = log.identify_closed_trades()
        assert len(closed) == 1
        assert closed[0]["ticker"] == "AAPL"
        assert closed[0]["pnl_dollars"] == pytest.approx(2000.0)  # 100 * (170-150)

    def test_partial_close(self, tmp_path):
        log = self._make_log(tmp_path)
        log.append(TradeRecord(
            ticker="AAPL", action="BUY", shares=100, price=150.0,
            notional=15000.0, sector="Tech",
        ))
        log.append(TradeRecord(
            ticker="AAPL", action="SELL", shares=50, price=170.0,
            notional=8500.0, sector="Tech",
        ))

        closed = log.identify_closed_trades()
        assert len(closed) == 1
        assert closed[0]["shares"] == 50  # only 50 closed
        assert closed[0]["pnl_dollars"] == pytest.approx(1000.0)  # 50 * 20

    def test_short_trade_pnl(self, tmp_path):
        log = self._make_log(tmp_path)
        log.append(TradeRecord(
            ticker="UNH", action="SHORT", shares=40, price=500.0,
            notional=20000.0, sector="Healthcare",
        ))
        log.append(TradeRecord(
            ticker="UNH", action="COVER", shares=40, price=460.0,
            notional=18400.0, sector="Healthcare",
        ))

        closed = log.identify_closed_trades()
        assert len(closed) == 1
        # Short: sold at 500, bought back at 460 → profit = 40 * 40 = 1600
        assert closed[0]["pnl_dollars"] == pytest.approx(1600.0)


class TestSnapshotStore:
    def _make_store(self, tmp_path: Path) -> SnapshotStore:
        return SnapshotStore(tmp_path / "test_snapshots.jsonl")

    def test_save_and_read(self, tmp_path):
        store = self._make_store(tmp_path)
        snapshot = DailySnapshot(
            date=date(2024, 6, 15),
            nav=10_000_000.0,
            gross_exposure=1.85,
            net_exposure=0.12,
        )
        store.save_snapshot(snapshot)
        all_snaps = store.read_all()
        assert len(all_snaps) == 1
        assert all_snaps[0].nav == 10_000_000.0

    def test_overwrites_same_date(self, tmp_path):
        store = self._make_store(tmp_path)
        snap1 = DailySnapshot(date=date(2024, 6, 15), nav=10_000_000.0, gross_exposure=1.85, net_exposure=0.12)
        snap2 = DailySnapshot(date=date(2024, 6, 15), nav=10_100_000.0, gross_exposure=1.90, net_exposure=0.15)
        store.save_snapshot(snap1)
        store.save_snapshot(snap2)
        all_snaps = store.read_all()
        assert len(all_snaps) == 1
        assert all_snaps[0].nav == 10_100_000.0

    def test_nav_series(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(5):
            store.save_snapshot(DailySnapshot(
                date=date(2024, 6, 15 + i),
                nav=10_000_000.0 + i * 50_000,
                gross_exposure=1.85,
                net_exposure=0.12,
            ))
        nav = store.nav_series()
        assert len(nav) == 5
        assert nav[0][1] == 10_000_000.0
        assert nav[4][1] == 10_200_000.0
