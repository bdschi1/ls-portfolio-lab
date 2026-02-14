"""Tests for core/portfolio.py — Portfolio and Position data models."""

from datetime import date

import pytest

from core.portfolio import Portfolio, Position, ProposedTrade, TradeBasket

# --- Position tests ---


class TestPosition:
    def test_long_direction(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0)
        assert p.direction == 1

    def test_short_direction(self):
        p = Position(ticker="AAPL", side="SHORT", shares=100, entry_price=150.0)
        assert p.direction == -1

    def test_notional(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=155.0)
        assert p.notional == 15500.0

    def test_signed_notional_long(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=155.0)
        assert p.signed_notional == 15500.0

    def test_signed_notional_short(self):
        p = Position(ticker="AAPL", side="SHORT", shares=100, entry_price=150.0, current_price=155.0)
        assert p.signed_notional == -15500.0

    def test_pnl_long_profit(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=160.0)
        assert p.pnl_dollars == 1000.0
        assert p.pnl_pct == pytest.approx(1000.0 / (100 * 150.0))

    def test_pnl_short_profit(self):
        p = Position(ticker="AAPL", side="SHORT", shares=100, entry_price=150.0, current_price=140.0)
        assert p.pnl_dollars == 1000.0
        assert p.pnl_pct == pytest.approx(10.0 / 150.0)

    def test_pnl_long_loss(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=140.0)
        assert p.pnl_dollars == -1000.0

    def test_pnl_short_loss(self):
        p = Position(ticker="AAPL", side="SHORT", shares=100, entry_price=150.0, current_price=160.0)
        assert p.pnl_dollars == -1000.0

    def test_weight_in_nav(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=100.0)
        assert p.weight_in(100000.0) == pytest.approx(0.10)

    def test_weight_short_negative(self):
        p = Position(ticker="AAPL", side="SHORT", shares=100, entry_price=150.0, current_price=100.0)
        assert p.weight_in(100000.0) == pytest.approx(-0.10)


# --- Portfolio tests ---


class TestPortfolio:
    def _make_portfolio(self) -> Portfolio:
        return Portfolio(
            name="Test",
            positions=[
                Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=160.0, sector="Tech"),
                Position(ticker="MSFT", side="LONG", shares=50, entry_price=300.0, current_price=310.0, sector="Tech"),
                Position(ticker="AMZN", side="SHORT", shares=30, entry_price=180.0, current_price=170.0, sector="Tech"),
            ],
            nav=100000.0,
        )

    def test_counts(self):
        p = self._make_portfolio()
        assert p.long_count == 2
        assert p.short_count == 1
        assert p.total_count == 3

    def test_long_notional(self):
        p = self._make_portfolio()
        # AAPL: 100 * 160 = 16000, MSFT: 50 * 310 = 15500
        assert p.long_notional == 31500.0

    def test_short_notional(self):
        p = self._make_portfolio()
        # AMZN: 30 * 170 = 5100
        assert p.short_notional == 5100.0

    def test_gross_exposure(self):
        p = self._make_portfolio()
        assert p.gross_exposure == pytest.approx((31500 + 5100) / 100000)

    def test_net_exposure(self):
        p = self._make_portfolio()
        assert p.net_exposure == pytest.approx((31500 - 5100) / 100000)

    def test_no_duplicate_tickers(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Portfolio(
                positions=[
                    Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0),
                    Position(ticker="AAPL", side="LONG", shares=50, entry_price=155.0),
                ],
            )

    def test_get_position(self):
        p = self._make_portfolio()
        pos = p.get_position("MSFT")
        assert pos is not None
        assert pos.shares == 50

    def test_get_position_missing(self):
        p = self._make_portfolio()
        assert p.get_position("GOOG") is None

    def test_add_position(self):
        p = self._make_portfolio()
        new_pos = Position(ticker="GOOG", side="LONG", shares=20, entry_price=140.0)
        p2 = p.add_position(new_pos)
        assert p2.total_count == 4
        assert p.total_count == 3  # original unchanged

    def test_add_duplicate_raises(self):
        p = self._make_portfolio()
        with pytest.raises(ValueError, match="already exists"):
            p.add_position(Position(ticker="AAPL", side="LONG", shares=10, entry_price=150.0))

    def test_remove_position(self):
        p = self._make_portfolio()
        p2 = p.remove_position("AMZN")
        assert p2.total_count == 2
        assert p2.short_count == 0
        assert p.total_count == 3  # original unchanged

    def test_remove_missing_raises(self):
        p = self._make_portfolio()
        with pytest.raises(ValueError, match="not found"):
            p.remove_position("GOOG")

    def test_update_position(self):
        p = self._make_portfolio()
        p2 = p.update_position("AAPL", shares=200)
        assert p2.get_position("AAPL").shares == 200
        assert p.get_position("AAPL").shares == 100  # original unchanged

    def test_update_prices(self):
        p = self._make_portfolio()
        p2 = p.update_prices({"AAPL": 170.0, "MSFT": 320.0})
        assert p2.get_position("AAPL").current_price == 170.0
        assert p2.get_position("MSFT").current_price == 320.0
        assert p.get_position("AAPL").current_price == 160.0  # original unchanged

    def test_sector_exposure(self):
        p = self._make_portfolio()
        exp = p.sector_exposure()
        assert "Tech" in exp
        assert exp["Tech"]["long"] > 0
        assert exp["Tech"]["short"] > 0

    def test_weight_vector(self):
        p = self._make_portfolio()
        wv = p.weight_vector()
        assert wv["AAPL"] > 0  # long = positive
        assert wv["AMZN"] < 0  # short = negative

    def test_total_pnl(self):
        p = self._make_portfolio()
        # AAPL: +1000, MSFT: +500, AMZN(short): +300
        expected = 100 * 10 + 50 * 10 + (-1) * 30 * (170 - 180)
        assert p.total_pnl_dollars == expected

    def test_max_80_positions(self):
        positions = [
            Position(ticker=f"T{i:03d}", side="LONG", shares=10, entry_price=100.0)
            for i in range(80)
        ]
        p = Portfolio(positions=positions)
        assert p.total_count == 80

        with pytest.raises(ValueError, match="max 80"):
            p.add_position(Position(ticker="T999", side="LONG", shares=10, entry_price=100.0))


# --- ProposedTrade tests ---


class TestProposedTrade:
    def test_valid_trade(self):
        t = ProposedTrade(ticker="AAPL", action="BUY", shares=100)
        assert t.ticker == "AAPL"

    def test_exit_no_sizing_ok(self):
        t = ProposedTrade(ticker="AAPL", action="EXIT")
        assert t.action == "EXIT"

    def test_non_exit_needs_sizing(self):
        with pytest.raises(ValueError, match="shares or dollar_amount"):
            ProposedTrade(ticker="AAPL", action="BUY")

    def test_dollar_amount_sizing(self):
        t = ProposedTrade(ticker="AAPL", action="BUY", dollar_amount=50000.0)
        assert t.dollar_amount == 50000.0


class TestTradeBasket:
    def test_max_10_trades(self):
        trades = [
            ProposedTrade(ticker=f"T{i}", action="BUY", shares=10)
            for i in range(10)
        ]
        basket = TradeBasket(trades=trades)
        assert len(basket.trades) == 10


# --- Option Position tests ---


class TestPositionOptions:
    def test_option_creation(self):
        p = Position(
            ticker="AAPL",
            side="LONG",
            shares=10,
            entry_price=5.0,
            current_price=6.0,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
            contract_multiplier=100,
            delta=0.55,
            underlying_price=195.0,
        )
        assert p.asset_type == "OPTION"
        assert p.strike == 190.0
        assert p.option_type == "CALL"

    def test_option_requires_fields(self):
        """Option without strike/expiry/option_type should raise."""
        with pytest.raises(ValueError, match="Option positions require"):
            Position(
                ticker="AAPL",
                side="LONG",
                shares=10,
                entry_price=5.0,
                asset_type="OPTION",
            )

    def test_option_notional(self):
        """Option notional = shares * multiplier * current_price."""
        p = Position(
            ticker="AAPL",
            side="LONG",
            shares=10,
            entry_price=5.0,
            current_price=6.0,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
            contract_multiplier=100,
        )
        assert p.notional == 10 * 100 * 6.0  # 6000

    def test_option_entry_notional(self):
        p = Position(
            ticker="AAPL",
            side="LONG",
            shares=10,
            entry_price=5.0,
            current_price=6.0,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
        )
        assert p.entry_notional == 10 * 100 * 5.0  # 5000

    def test_option_pnl(self):
        """Option P&L uses multiplier."""
        p = Position(
            ticker="AAPL",
            side="LONG",
            shares=10,
            entry_price=5.0,
            current_price=7.0,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
        )
        # direction(1) * shares(10) * multiplier(100) * (7 - 5) = 2000
        assert p.pnl_dollars == 2000.0

    def test_option_delta_adjusted(self):
        p = Position(
            ticker="AAPL",
            side="LONG",
            shares=10,
            entry_price=5.0,
            current_price=6.0,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
            delta=0.50,
            underlying_price=200.0,
        )
        # shares(10) * multiplier(100) * delta(0.50) * underlying(200) = 100000
        assert p.delta_adjusted_notional == 100000.0

    def test_equity_delta_adjusted_is_none(self):
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0)
        assert p.delta_adjusted_notional is None

    def test_equity_notional_no_multiplier(self):
        """Equity notional should NOT use contract_multiplier."""
        p = Position(ticker="AAPL", side="LONG", shares=100, entry_price=150.0, current_price=160.0)
        assert p.notional == 16000.0  # 100 * 160, not 100 * 100 * 160


# --- Cash management tests ---


def _pos(
    ticker: str, side: str, shares: int,
    entry: float, current: float = 0.0, **kw: object,
) -> Position:
    """Helper to create Positions concisely in tests."""
    return Position(
        ticker=ticker, side=side, shares=shares,
        entry_price=entry, current_price=current, **kw,
    )


class TestPortfolioCash:
    def test_cash_auto_init(self):
        """Cash should be auto-computed from NAV - invested."""
        p = Portfolio(
            positions=[_pos("AAPL", "LONG", 100, 150.0, 160.0)],
            nav=100000.0,
        )
        # NAV = 100000, invested = 100*160 = 16000, cash = 84000
        assert p.cash == pytest.approx(84000.0)

    def test_cash_empty_portfolio(self):
        """Empty portfolio should have cash = NAV."""
        p = Portfolio(nav=100000.0)
        assert p.cash == 100000.0

    def test_cash_pct(self):
        p = Portfolio(
            positions=[_pos("AAPL", "LONG", 100, 150.0, 160.0)],
            nav=100000.0,
        )
        assert p.cash_pct == pytest.approx(84000.0 / 100000.0)

    def test_investable_cash(self):
        """Investable cash = cash - (nav * min_cash_pct)."""
        p = Portfolio(
            positions=[_pos("AAPL", "LONG", 100, 150.0, 160.0)],
            nav=100000.0,
            min_cash_pct=0.05,
        )
        # cash = 84000, floor = 5000, investable = 79000
        assert p.investable_cash == pytest.approx(79000.0)

    def test_add_position_deducts_cash(self):
        p = Portfolio(
            positions=[_pos("AAPL", "LONG", 100, 150.0, 160.0)],
            nav=100000.0,
        )
        initial_cash = p.cash
        new_pos = _pos("MSFT", "LONG", 10, 300.0, 310.0)
        p2 = p.add_position(new_pos)
        # Cash decreases by notional (10 * 310 = 3100)
        assert p2.cash == pytest.approx(initial_cash - 3100.0)

    def test_remove_position_returns_cash(self):
        p = Portfolio(
            positions=[
                _pos("AAPL", "LONG", 100, 150.0, 160.0),
                _pos("MSFT", "LONG", 10, 300.0, 310.0),
            ],
            nav=100000.0,
        )
        initial_cash = p.cash
        p2 = p.remove_position("MSFT")
        # Cash increases by notional (10 * 310 = 3100)
        assert p2.cash == pytest.approx(initial_cash + 3100.0)

    def test_default_nav_3b(self):
        p = Portfolio()
        assert p.nav == 3_000_000_000.0

    def test_cash_warning_does_not_block(self):
        """Below 5% cash warns but does not block."""
        p = Portfolio(
            positions=[
                _pos("AAPL", "LONG", 9000, 10.0, 10.0),
            ],
            nav=100000.0,
        )
        # cash = 10000 (10% of NAV). Adding $6000 → $4000 (4%)
        big_pos = _pos("MSFT", "LONG", 60, 100.0, 100.0)
        p2 = p.add_position(big_pos)  # Should not raise
        assert p2.cash == pytest.approx(4000.0)
        assert p2.cash_pct < p2.min_cash_pct


# --- ETF constraint tests ---


def _etf(ticker: str, side: str, shares: int, price: float) -> Position:
    """Helper to create ETF positions concisely."""
    return Position(
        ticker=ticker, side=side, shares=shares,
        entry_price=price, asset_type="ETF",
    )


class TestPortfolioETF:
    def test_etf_count(self):
        p = Portfolio(
            positions=[
                _etf("SPY", "LONG", 100, 450.0),
                Position(ticker="AAPL", side="LONG", shares=50, entry_price=150.0),
            ],
            nav=100000.0,
        )
        assert p.etf_count == 1

    def test_max_etf_positions(self):
        """Should raise if too many ETFs."""
        with pytest.raises(ValueError, match="Too many ETF"):
            Portfolio(
                positions=[
                    _etf("SPY", "LONG", 10, 450.0),
                    _etf("QQQ", "LONG", 10, 400.0),
                    _etf("IWM", "SHORT", 10, 200.0),
                    _etf("XLF", "SHORT", 10, 40.0),
                ],
                nav=100000.0,
                max_etf_positions=3,
            )

    def test_add_etf_exceeds_max(self):
        """Adding a 4th ETF should raise."""
        p = Portfolio(
            positions=[
                _etf("SPY", "LONG", 10, 450.0),
                _etf("QQQ", "LONG", 10, 400.0),
                _etf("IWM", "SHORT", 10, 200.0),
            ],
            nav=100000.0,
            max_etf_positions=3,
        )
        assert p.etf_count == 3
        new_etf = _etf("XLF", "SHORT", 10, 40.0)
        with pytest.raises(ValueError, match="max.*ETF"):
            p.add_position(new_etf)

    def test_equity_positions_accessor(self):
        p = Portfolio(
            positions=[
                _etf("SPY", "LONG", 10, 450.0),
                Position(ticker="AAPL", side="LONG", shares=50, entry_price=150.0),
                Position(ticker="MSFT", side="SHORT", shares=30, entry_price=300.0),
            ],
            nav=100000.0,
        )
        assert len(p.equity_positions) == 2
        assert len(p.etf_positions) == 1
        assert len(p.option_positions) == 0


# --- ProposedTrade option tests ---


class TestProposedTradeOptions:
    def test_option_trade_valid(self):
        t = ProposedTrade(
            ticker="AAPL",
            action="BUY",
            shares=10,
            asset_type="OPTION",
            strike=190.0,
            expiry=date(2025, 6, 20),
            option_type="CALL",
        )
        assert t.asset_type == "OPTION"

    def test_option_trade_no_short(self):
        """Options cannot be shorted."""
        with pytest.raises(ValueError, match="SHORT"):
            ProposedTrade(
                ticker="AAPL",
                action="SHORT",
                shares=10,
                asset_type="OPTION",
                strike=190.0,
                expiry=date(2025, 6, 20),
                option_type="CALL",
            )

    def test_option_trade_no_cover(self):
        """Options cannot be covered."""
        with pytest.raises(ValueError, match="COVER"):
            ProposedTrade(
                ticker="AAPL",
                action="COVER",
                shares=10,
                asset_type="OPTION",
                strike=190.0,
                expiry=date(2025, 6, 20),
                option_type="CALL",
            )

    def test_option_trade_requires_fields(self):
        """Option trades must have strike, expiry, option_type."""
        with pytest.raises(ValueError, match="Option trades require"):
            ProposedTrade(
                ticker="AAPL",
                action="BUY",
                shares=10,
                asset_type="OPTION",
            )
