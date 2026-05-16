"""Tests for the extended ingest surface: equity/position/$ amount aliases,
bare 'EQUITY' Bloomberg suffix, RSI/beta passthrough, header normalization."""

import pytest

from data.ingest import _clean_ticker, _normalize_header, load_from_csv_string


class TestBareEquitySuffix:
    """Bloomberg cleaner accepts both 'TICKER XX EQUITY' and bare 'TICKER EQUITY'."""

    def test_bare_equity_stripped(self):
        assert _clean_ticker("AAPL EQUITY") == "AAPL"

    def test_bare_equity_lowercase(self):
        assert _clean_ticker("aapl equity") == "AAPL"

    def test_us_equity_still_works(self):
        assert _clean_ticker("AAPL US EQUITY") == "AAPL"

    def test_bare_equity_in_ingest(self):
        csv = "ticker,shares\nAAPL EQUITY,100\nMSFT EQUITY,-50\n"
        portfolio = load_from_csv_string(csv)
        assert set(portfolio.tickers) == {"AAPL", "MSFT"}


class TestHeaderNormalization:
    """_normalize_header strips $, lowercases, collapses non-alphanumerics."""

    def test_dollar_amount(self):
        assert _normalize_header("$ Amount") == "amount"

    def test_dollar_amount_no_space(self):
        assert _normalize_header("$Amount") == "amount"

    def test_dollar_amount_two_words(self):
        assert _normalize_header("Dollar Amount") == "dollar_amount"

    def test_parenthesized(self):
        assert _normalize_header("Ticker (BBG)") == "ticker_bbg"

    def test_multiple_spaces(self):
        assert _normalize_header("Market   Value") == "market_value"


class TestTickerColumnAliases:
    """Column A: ticker / stock / equity all accepted."""

    def test_stock_header(self):
        csv = "stock,shares\nAAPL,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL") is not None

    def test_equity_header(self):
        csv = "equity,shares\nAAPL,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL") is not None

    def test_security_header(self):
        csv = "security,shares\nAAPL,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL") is not None


class TestSizeColumnAliases:
    """Column B: shares / position / $ amount all accepted."""

    def test_position_header_treated_as_shares(self):
        csv = "ticker,position\nAAPL,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").shares == 100.0

    def test_dollar_amount_converted_via_entry_price(self):
        # $50,000 / $100 = 500 shares
        csv = "ticker,$ amount,entry_price\nAAPL,50000,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").shares == 500.0

    def test_dollar_amount_signed_infers_short(self):
        csv = "ticker,$ amount,entry_price\nAAPL,-50000,100\n"
        portfolio = load_from_csv_string(csv)
        pos = portfolio.get_position("AAPL")
        assert pos.side == "SHORT"
        assert pos.shares == 500.0

    def test_notional_alias(self):
        csv = "ticker,notional,entry_price\nAAPL,10000,50\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").shares == 200.0

    def test_market_value_alias(self):
        csv = "ticker,market_value,entry_price\nAAPL,10000,50\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").shares == 200.0

    def test_dollar_amount_no_entry_price_skips(self):
        csv = "ticker,$ amount\nAAPL,50000\n"
        with pytest.raises(ValueError, match="No valid positions"):
            load_from_csv_string(csv)

    def test_shares_takes_priority_over_notional(self):
        csv = "ticker,shares,notional,entry_price\nAAPL,42,10000,50\n"
        portfolio = load_from_csv_string(csv)
        # Explicit shares wins over $-conversion
        assert portfolio.get_position("AAPL").shares == 42.0


class TestRsiBetaPassthrough:
    """Imported RSI and beta land on Position as floats."""

    def test_rsi_and_beta_stored(self):
        csv = "ticker,shares,rsi,beta\nAAPL,100,55.5,1.2\n"
        portfolio = load_from_csv_string(csv)
        pos = portfolio.get_position("AAPL")
        assert pos.rsi == 55.5
        assert pos.beta == 1.2

    def test_rsi_beta_default_none_when_missing(self):
        csv = "ticker,shares\nAAPL,100\n"
        portfolio = load_from_csv_string(csv)
        pos = portfolio.get_position("AAPL")
        assert pos.rsi is None
        assert pos.beta is None

    def test_beta_to_spy_alias(self):
        csv = "ticker,shares,beta_to_spy\nAAPL,100,0.85\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").beta == 0.85

    def test_relative_strength_alias(self):
        csv = "ticker,shares,relative_strength\nAAPL,100,70\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").rsi == 70.0


class TestEverythingTogether:
    """End-to-end: Equity column + $ Amount + Bloomberg ticker + RSI + Beta."""

    def test_full_combo(self):
        csv = (
            "equity,$ amount,entry_price,rsi,beta\n"
            "AAPL US EQUITY,50000,100,55.5,1.2\n"
            "VOD LN EQUITY,-25000,50,40.0,0.9\n"
        )
        portfolio = load_from_csv_string(csv)
        assert set(portfolio.tickers) == {"AAPL", "VOD"}
        aapl = portfolio.get_position("AAPL")
        vod = portfolio.get_position("VOD")
        assert aapl.side == "LONG"
        assert aapl.shares == 500.0
        assert aapl.rsi == 55.5
        assert aapl.beta == 1.2
        assert vod.side == "SHORT"
        assert vod.shares == 500.0
