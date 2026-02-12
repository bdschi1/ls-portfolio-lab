"""Tests for core/trade_impact.py — trade impact simulator."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from core.portfolio import Portfolio, Position, ProposedTrade, TradeBasket
from core.trade_impact import apply_trades, simulate_impact


def _make_portfolio() -> Portfolio:
    return Portfolio(
        positions=[
            Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0,
                     current_price=160.0, sector="Tech", subsector="Hardware"),
            Position(ticker="MSFT", side="LONG", shares=50, entry_price=300.0,
                     current_price=310.0, sector="Tech", subsector="Software"),
            Position(ticker="UNH", side="SHORT", shares=40, entry_price=500.0,
                     current_price=490.0, sector="Healthcare", subsector="HMO"),
        ],
        nav=100000.0,
    )


def _make_prices() -> dict[str, float]:
    return {"AAPL": 160.0, "MSFT": 310.0, "UNH": 490.0, "GOOG": 175.0, "HCA": 320.0}


class TestApplyTrades:
    def test_buy_new_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.total_count == 4
        assert p2.get_position("GOOG") is not None
        assert p2.get_position("GOOG").side == "LONG"

    def test_short_new_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="HCA", action="SHORT", shares=30),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("HCA").side == "SHORT"

    def test_add_to_existing(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="AAPL", action="ADD", shares=50),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("AAPL").shares == 150

    def test_reduce_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="AAPL", action="REDUCE", shares=30),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("AAPL").shares == 70

    def test_exit_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="UNH", action="EXIT"),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("UNH") is None
        assert p2.total_count == 2

    def test_cover_short(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="UNH", action="COVER", shares=20),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("UNH").shares == 20

    def test_cover_full_removes(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="UNH", action="COVER", shares=40),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("UNH") is None

    def test_sell_full_removes(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="AAPL", action="SELL", shares=100),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.get_position("AAPL") is None

    def test_cannot_buy_short_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="UNH", action="BUY", shares=10),
        ])
        with pytest.raises(ValueError, match="SHORT"):
            apply_trades(p, basket, _make_prices())

    def test_cannot_sell_short_position(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="UNH", action="SELL", shares=10),
        ])
        with pytest.raises(ValueError, match="SHORT"):
            apply_trades(p, basket, _make_prices())

    def test_multiple_trades(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
            ProposedTrade(ticker="UNH", action="EXIT"),
            ProposedTrade(ticker="AAPL", action="REDUCE", shares=30),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p2.total_count == 3  # +1 (GOOG), -1 (UNH), 0 (AAPL still there)
        assert p2.get_position("GOOG") is not None
        assert p2.get_position("UNH") is None
        assert p2.get_position("AAPL").shares == 70

    def test_no_price_raises(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="ZZZZ", action="BUY", shares=10),
        ])
        with pytest.raises(ValueError, match="No price"):
            apply_trades(p, basket, {})

    def test_immutability(self):
        """Original portfolio should not be modified."""
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        assert p.total_count == 3
        assert p2.total_count == 4


class TestSimulateImpact:
    def test_basic_simulation(self):
        p = _make_portfolio()
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
        ])
        # Create minimal returns_df
        rng = np.random.default_rng(42)
        returns_df = pl.DataFrame({
            "date": list(range(100)),
            "AAPL": rng.normal(0, 0.02, 100).tolist(),
            "MSFT": rng.normal(0, 0.015, 100).tolist(),
            "UNH": rng.normal(0, 0.018, 100).tolist(),
            "GOOG": rng.normal(0, 0.02, 100).tolist(),
            "SPY": rng.normal(0, 0.01, 100).tolist(),
        })

        result = simulate_impact(
            portfolio=p,
            basket=basket,
            current_prices=_make_prices(),
            returns_df=returns_df,
            betas={"AAPL": 1.2, "MSFT": 1.0, "UNH": 0.8, "GOOG": 1.1},
        )

        assert result.portfolio_after.total_count == 4
        assert "GOOG" in result.new_positions
        assert len(result.metric_diffs) > 0

        # Check that diffs have before/after values
        for diff in result.metric_diffs:
            assert isinstance(diff.before, float)
            assert isinstance(diff.after, float)


class TestOptionTrades:
    """Test option trade handling through the trade impact engine."""

    def test_buy_option(self):
        """Should create a new OPTION position."""
        p = _make_portfolio()
        prices = {**_make_prices(), "AAPL": 5.0}  # option premium = $5
        basket = TradeBasket(trades=[
            ProposedTrade(
                ticker="AAPL_C190",
                action="BUY",
                shares=10,
                asset_type="OPTION",
                strike=190.0,
                expiry=date(2025, 6, 20),
                option_type="CALL",
                delta=0.55,
            ),
        ])
        prices["AAPL_C190"] = 5.0
        p2 = apply_trades(p, basket, prices)
        opt = p2.get_position("AAPL_C190")
        assert opt is not None
        assert opt.asset_type == "OPTION"
        assert opt.strike == 190.0
        assert opt.option_type == "CALL"
        assert opt.side == "LONG"

    def test_exit_option(self):
        """Should be able to EXIT an option position."""
        p = Portfolio(
            positions=[
                Position(
                    ticker="AAPL_C190",
                    side="LONG",
                    shares=10,
                    entry_price=5.0,
                    current_price=6.0,
                    asset_type="OPTION",
                    strike=190.0,
                    expiry=date(2025, 6, 20),
                    option_type="CALL",
                ),
            ],
            nav=100000.0,
        )
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="AAPL_C190", action="EXIT"),
        ])
        p2 = apply_trades(p, basket, {"AAPL_C190": 6.0})
        assert p2.get_position("AAPL_C190") is None

    def test_short_option_raises(self):
        """SHORT action should not be allowed for options in the trade engine."""
        # The ProposedTrade validator already blocks this
        with pytest.raises(ValueError, match="SHORT"):
            ProposedTrade(
                ticker="AAPL_P180",
                action="SHORT",
                shares=10,
                asset_type="OPTION",
                strike=180.0,
                expiry=date(2025, 6, 20),
                option_type="PUT",
            )

    def test_etf_trade(self):
        """Should create ETF positions correctly."""
        p = _make_portfolio()
        prices = {**_make_prices(), "SPY": 500.0}
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="SPY", action="BUY", shares=20, asset_type="ETF"),
        ])
        p2 = apply_trades(p, basket, prices)
        spy = p2.get_position("SPY")
        assert spy is not None
        assert spy.asset_type == "ETF"
        assert spy.side == "LONG"
        assert spy.shares == 20


class TestCashTracking:
    """Test that cash is properly tracked through trades."""

    def test_buy_reduces_cash(self):
        p = _make_portfolio()
        initial_cash = p.cash
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        # Buying GOOG at $175, 50 shares = $8750
        assert p2.cash < initial_cash

    def test_exit_returns_cash(self):
        p = _make_portfolio()
        initial_cash = p.cash
        basket = TradeBasket(trades=[
            ProposedTrade(ticker="AAPL", action="EXIT"),
        ])
        p2 = apply_trades(p, basket, _make_prices())
        # Exiting AAPL returns its notional (100 * 160 = 16000)
        assert p2.cash > initial_cash

    def test_cash_survives_round_trip(self):
        """Cash should be consistent after add and remove."""
        p = _make_portfolio()
        initial_cash = p.cash

        # Buy GOOG
        basket1 = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="BUY", shares=50),
        ])
        p2 = apply_trades(p, basket1, _make_prices())

        # Then exit GOOG
        basket2 = TradeBasket(trades=[
            ProposedTrade(ticker="GOOG", action="EXIT"),
        ])
        p3 = apply_trades(p2, basket2, _make_prices())

        # Cash should be back (approximately — entry and exit at same price)
        assert p3.cash == pytest.approx(initial_cash, rel=0.01)
