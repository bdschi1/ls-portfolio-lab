"""Tests for core.tlh — L/S tax-loss harvesting scanner."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from core.portfolio import Portfolio, Position
from core.tlh import check_reentry_blocked, scan
from history.trade_log import TradeRecord


def _today() -> date:
    return date(2026, 5, 1)


def _long_at_loss() -> Position:
    return Position(
        ticker="AAPL",
        side="LONG",
        shares=100,
        entry_price=200,
        current_price=150,  # $50 loss/share = $5,000 unrealized loss
        entry_date=date(2025, 6, 1),  # ~11 months ago → short-term
        sector="Tech",
    )


def _long_at_gain() -> Position:
    return Position(
        ticker="MSFT",
        side="LONG",
        shares=50,
        entry_price=300,
        current_price=400,
        entry_date=date(2024, 1, 1),
        sector="Tech",
    )


def _short_at_loss() -> Position:
    # Short entered at $100, now $120 → loss of $20/share × 30 = -$600
    return Position(
        ticker="TSLA",
        side="SHORT",
        shares=30,
        entry_price=100,
        current_price=120,
        entry_date=date(2024, 1, 1),  # long-term
        sector="Consumer Disc",
    )


def _pf(positions: list[Position]) -> Portfolio:
    return Portfolio(name="t", nav=1_000_000.0, positions=positions)


def test_no_candidates_when_all_positions_at_gain():
    pf = _pf([_long_at_gain()])
    s = scan(pf, today=_today())
    assert s.candidates == []
    assert s.actionable == []


def test_long_loss_appears_as_actionable_candidate():
    pf = _pf([_long_at_loss(), _long_at_gain()])
    s = scan(pf, today=_today())
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert c.ticker == "AAPL"
    assert c.side == "LONG"
    assert c.unrealized_loss_dollars == pytest.approx(-5000.0)
    assert c.actionable is True


def test_short_loss_appears_as_candidate_long_term():
    pf = _pf([_short_at_loss()])
    s = scan(pf, today=_today())
    assert len(s.candidates) == 1
    c = s.candidates[0]
    assert c.side == "SHORT"
    assert c.unrealized_loss_dollars == pytest.approx(-600.0)
    assert c.is_short_term is False  # entered 2024-01-01, today 2026-05-01 → >365 days


def test_min_loss_threshold_filters_small_losses():
    tiny_loss = Position(
        ticker="XYZ",
        side="LONG",
        shares=10,
        entry_price=100,
        current_price=99,  # $10 loss total
        entry_date=date(2025, 6, 1),
    )
    pf = _pf([tiny_loss])
    s = scan(pf, today=_today(), min_loss_dollars=100.0)
    assert s.candidates == []


def test_wash_sale_blocks_recent_same_ticker_trade():
    pf = _pf([_long_at_loss()])
    recent = datetime(2026, 4, 15, 10, 0)  # 16 days before today=2026-05-01
    records = [
        TradeRecord(
            timestamp=recent,
            ticker="AAPL",
            action="SELL",
            shares=50,
            price=155,
            notional=7750,
        )
    ]
    s = scan(pf, trade_log_records=records, today=_today())
    c = s.candidates[0]
    assert c.blocked is True
    assert any("wash-sale" in r for r in c.block_reasons)
    # window end = recent_date + 30 = 2026-04-15 + 30 = 2026-05-15
    assert c.wash_sale_window_end == date(2026, 5, 15)


def test_old_trades_do_not_trigger_wash_sale():
    pf = _pf([_long_at_loss()])
    old = datetime(2026, 1, 1, 10, 0)  # 4 months ago
    records = [
        TradeRecord(timestamp=old, ticker="AAPL", action="SELL", shares=20, price=180, notional=3600)
    ]
    s = scan(pf, trade_log_records=records, today=_today())
    assert s.candidates[0].blocked is False


def test_actionable_total_loss_sums_only_unblocked():
    long_loss = _long_at_loss()  # -$5000, unblocked
    short_loss = _short_at_loss()  # -$600, unblocked
    pf = _pf([long_loss, short_loss])
    s = scan(pf, today=_today())
    assert s.total_loss_actionable == pytest.approx(-5600.0)


def test_candidates_sorted_actionable_first_then_by_loss_magnitude():
    big_loss_blocked = Position(
        ticker="BIG", side="LONG", shares=100, entry_price=500, current_price=100,
        entry_date=date(2025, 1, 1),
    )
    small_loss = Position(
        ticker="SMALL", side="LONG", shares=10, entry_price=100, current_price=80,
        entry_date=date(2025, 1, 1),
    )
    pf = _pf([big_loss_blocked, small_loss])
    # Block BIG via wash-sale: recent same-ticker trade
    records = [
        TradeRecord(
            timestamp=datetime(2026, 4, 20, 9, 0),
            ticker="BIG",
            action="SELL",
            shares=10,
            price=100,
            notional=1000,
        )
    ]
    s = scan(pf, trade_log_records=records, today=_today())
    # SMALL (actionable, -200) should come before BIG (blocked, -40000)
    assert s.candidates[0].ticker == "SMALL"
    assert s.candidates[-1].ticker == "BIG"


def test_check_reentry_blocked_returns_window_for_recent_close():
    records = [
        TradeRecord(
            timestamp=datetime(2026, 4, 20, 9, 0),
            ticker="META",
            action="COVER",
            shares=50,
            price=400,
            notional=20000,
        )
    ]
    blocked, end = check_reentry_blocked("META", records, today=_today())
    assert blocked is True
    assert end == date(2026, 5, 20)


def test_check_reentry_blocked_ignores_other_tickers():
    records = [
        TradeRecord(
            timestamp=datetime(2026, 4, 20, 9, 0),
            ticker="GOOG",
            action="SELL",
            shares=10,
            price=150,
            notional=1500,
        )
    ]
    blocked, end = check_reentry_blocked("META", records, today=_today())
    assert blocked is False
    assert end is None
