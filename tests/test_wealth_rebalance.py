"""Tests for core.wealth_rebalance — target-weights → TradeBasket translator."""

from __future__ import annotations

import pytest

from core.portfolio import Portfolio, Position
from core.wealth_rebalance import (
    TargetWeight,
    WealthRebalanceRequest,
    build_basket,
    compute_drift_rows,
)


def _pf() -> Portfolio:
    return Portfolio(
        name="t",
        nav=1_000_000.0,
        positions=[
            Position(ticker="AAPL", side="LONG", shares=100, entry_price=180, current_price=200, sector="Tech"),
            Position(ticker="MSFT", side="LONG", shares=50, entry_price=400, current_price=400, sector="Tech"),
            Position(ticker="GOOG", side="SHORT", shares=30, entry_price=140, current_price=140, sector="Comm"),
        ],
    )


def test_compute_drift_rows_flags_in_and_out_of_band():
    pf = _pf()
    aapl_cur = pf.get_position("AAPL").weight_in(pf.nav)
    rows = compute_drift_rows(pf, [
        TargetWeight(ticker="AAPL", target_weight=aapl_cur, band=0.005),  # exactly at target
        TargetWeight(ticker="MSFT", target_weight=0.50, band=0.005),  # huge drift
    ])
    assert rows[0]["in_band"] is True
    assert rows[1]["in_band"] is False
    assert rows[1]["drift"] == pytest.approx(rows[1]["current_weight"] - 0.50)


def test_build_basket_emits_no_trades_when_all_in_band():
    pf = _pf()
    weights = pf.weight_vector()
    targets = [TargetWeight(ticker=t, target_weight=w, band=0.005) for t, w in weights.items()]
    result = build_basket(pf, WealthRebalanceRequest(targets=targets))
    assert result.basket.trades == []
    assert result.warnings == []


def test_target_zero_on_existing_long_emits_exit():
    pf = _pf()
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="AAPL", target_weight=0.0, band=0.005),
    ]))
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].ticker == "AAPL"
    assert result.basket.trades[0].action == "EXIT"


def test_target_zero_on_missing_position_emits_nothing():
    pf = _pf()
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="NVDA", target_weight=0.0, band=0.005),
    ]))
    assert result.basket.trades == []


def test_new_long_entry_emits_buy():
    pf = _pf()
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="NVDA", target_weight=0.03, band=0.005),
    ]))
    assert len(result.basket.trades) == 1
    t = result.basket.trades[0]
    assert t.action == "BUY"
    assert t.dollar_amount == pytest.approx(0.03 * pf.nav)


def test_new_short_entry_emits_short():
    pf = _pf()
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="TSLA", target_weight=-0.02, band=0.005),
    ]))
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].action == "SHORT"


def test_under_target_long_emits_add():
    pf = _pf()
    aapl_cur = pf.get_position("AAPL").weight_in(pf.nav)
    # bump target above current → drift negative → ADD
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="AAPL", target_weight=aapl_cur + 0.02, band=0.005),
    ]))
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].action == "ADD"


def test_over_target_long_emits_reduce():
    pf = _pf()
    aapl_cur = pf.get_position("AAPL").weight_in(pf.nav)
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="AAPL", target_weight=aapl_cur - 0.01, band=0.001),
    ]))
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].action == "REDUCE"


def test_under_target_short_emits_short_more():
    """GOOG short ~ -0.42%; target -2.42% (more short). drift positive → SHORT more."""
    pf = _pf()
    goog_cur = pf.get_position("GOOG").weight_in(pf.nav)
    assert goog_cur < 0
    target = goog_cur - 0.02  # more negative = larger short
    result = build_basket(
        pf,
        WealthRebalanceRequest(targets=[
            TargetWeight(ticker="GOOG", target_weight=target, band=0.001),
        ]),
    )
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].action == "SHORT"


def test_over_target_short_emits_cover():
    """GOOG short ~ -0.42%; target -0.05% (less short, still short). drift negative → COVER."""
    pf = _pf()
    goog_cur = pf.get_position("GOOG").weight_in(pf.nav)
    target = goog_cur / 10.0  # still negative, but much smaller magnitude
    assert target < 0
    result = build_basket(
        pf,
        WealthRebalanceRequest(targets=[
            TargetWeight(ticker="GOOG", target_weight=target, band=0.0001),
        ]),
    )
    assert len(result.basket.trades) == 1
    assert result.basket.trades[0].action == "COVER"


def test_side_flip_rejected_with_warning():
    pf = _pf()
    # AAPL is LONG; ask to flip to SHORT
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="AAPL", target_weight=-0.02, band=0.001),
    ]))
    assert result.basket.trades == []
    assert any("side flip" in w.lower() for w in result.warnings)


def test_basket_cap_at_ten_keeps_largest_drifts():
    pf = _pf()
    targets = [TargetWeight(ticker=f"X{i:02d}", target_weight=0.01 * (i + 1), band=0.0) for i in range(12)]
    result = build_basket(pf, WealthRebalanceRequest(targets=targets))
    assert len(result.basket.trades) == 10
    # smallest two should be dropped: X00 (drift 0.01) and X01 (drift 0.02)
    assert any("deferred 2 smaller" in w for w in result.warnings)


def test_basket_consumable_by_trade_impact_engine():
    """End-to-end: basket built by wealth_rebalance can feed trade_impact."""
    from core.trade_impact import apply_trades

    pf = _pf()
    result = build_basket(pf, WealthRebalanceRequest(targets=[
        TargetWeight(ticker="NVDA", target_weight=0.03, band=0.005),
    ]))
    prices = {p.ticker: p.current_price for p in pf.positions}
    prices["NVDA"] = 600.0
    new_pf = apply_trades(pf, result.basket, prices)
    assert new_pf.get_position("NVDA") is not None
    assert new_pf.get_position("NVDA").side == "LONG"
