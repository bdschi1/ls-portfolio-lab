"""Tests for data/ingest.py — portfolio import, Bloomberg ticker cleaning, side inference."""

import pytest

from data.ingest import _clean_ticker, load_from_csv_string

# ---------------------------------------------------------------------------
# Bloomberg ticker cleaning
# ---------------------------------------------------------------------------


class TestCleanTicker:
    """Unit tests for _clean_ticker() Bloomberg suffix stripping."""

    def test_plain_ticker_unchanged(self):
        assert _clean_ticker("AAPL") == "AAPL"

    def test_lowercase_uppercased(self):
        assert _clean_ticker("aapl") == "AAPL"

    def test_whitespace_trimmed(self):
        assert _clean_ticker("  AAPL  ") == "AAPL"

    def test_us_equity_stripped(self):
        assert _clean_ticker("AAPL US EQUITY") == "AAPL"

    def test_ln_equity_stripped(self):
        assert _clean_ticker("VOD LN EQUITY") == "VOD"

    def test_jp_equity_stripped(self):
        assert _clean_ticker("7203 JP EQUITY") == "7203"

    def test_gr_equity_stripped(self):
        assert _clean_ticker("SAP GR EQUITY") == "SAP"

    def test_fp_equity_stripped(self):
        assert _clean_ticker("MC FP EQUITY") == "MC"

    def test_cn_equity_stripped(self):
        assert _clean_ticker("RY CN EQUITY") == "RY"

    def test_au_equity_stripped(self):
        assert _clean_ticker("BHP AU EQUITY") == "BHP"

    def test_hyphenated_ticker(self):
        assert _clean_ticker("BRK-B US EQUITY") == "BRK-B"

    def test_dotted_ticker(self):
        assert _clean_ticker("BRK.B US EQUITY") == "BRK.B"

    def test_lowercase_bloomberg(self):
        assert _clean_ticker("aapl us equity") == "AAPL"

    def test_no_equity_suffix_passthrough(self):
        """'AAPL US' without EQUITY should not be stripped."""
        assert _clean_ticker("AAPL US") == "AAPL US"

    def test_equity_alone_passthrough(self):
        assert _clean_ticker("EQUITY") == "EQUITY"

    def test_single_letter_exchange_no_match(self):
        assert _clean_ticker("AAPL U EQUITY") == "AAPL U EQUITY"

    def test_three_letter_exchange_no_match(self):
        assert _clean_ticker("AAPL USA EQUITY") == "AAPL USA EQUITY"


class TestIngestBloombergTickers:
    """Integration: Bloomberg tickers cleaned during CSV ingest."""

    def test_bloomberg_tickers_cleaned(self):
        csv = "ticker,side,shares\nAAPL US EQUITY,LONG,100\nVOD LN EQUITY,SHORT,50\n"
        portfolio = load_from_csv_string(csv)
        assert "AAPL" in portfolio.tickers
        assert "VOD" in portfolio.tickers
        assert "AAPL US EQUITY" not in portfolio.tickers

    def test_mixed_plain_and_bloomberg(self):
        csv = "ticker,side,shares\nAAPL,LONG,100\nVOD LN EQUITY,SHORT,50\n"
        portfolio = load_from_csv_string(csv)
        assert set(portfolio.tickers) == {"AAPL", "VOD"}


# ---------------------------------------------------------------------------
# Side inference from signed shares
# ---------------------------------------------------------------------------


class TestSideInference:
    """Infer LONG/SHORT from sign of shares when no side column."""

    def test_positive_shares_inferred_long(self):
        csv = "ticker,shares\nAAPL,100\nMSFT,200\n"
        portfolio = load_from_csv_string(csv)
        for p in portfolio.positions:
            assert p.side == "LONG"
            assert p.shares > 0

    def test_negative_shares_inferred_short(self):
        csv = "ticker,shares\nAAPL,-100\nMSFT,-200\n"
        portfolio = load_from_csv_string(csv)
        for p in portfolio.positions:
            assert p.side == "SHORT"
            assert p.shares > 0

    def test_mixed_signed_shares(self):
        csv = "ticker,shares\nAAPL,100\nMSFT,-50\n"
        portfolio = load_from_csv_string(csv)
        aapl = portfolio.get_position("AAPL")
        msft = portfolio.get_position("MSFT")
        assert aapl.side == "LONG"
        assert aapl.shares == 100.0
        assert msft.side == "SHORT"
        assert msft.shares == 50.0

    def test_explicit_side_still_works(self):
        csv = "ticker,side,shares\nAAPL,LONG,100\nMSFT,SHORT,50\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").side == "LONG"
        assert portfolio.get_position("MSFT").side == "SHORT"

    def test_explicit_side_takes_priority(self):
        """When side column exists, it controls side regardless of share sign."""
        csv = "ticker,side,shares\nAAPL,SHORT,100\n"
        portfolio = load_from_csv_string(csv)
        aapl = portfolio.get_position("AAPL")
        assert aapl.side == "SHORT"
        assert aapl.shares == 100.0

    def test_no_side_no_shares_raises(self):
        csv = "ticker,weight\nAAPL,0.05\n"
        with pytest.raises(ValueError, match="side"):
            load_from_csv_string(csv)

    def test_bloomberg_plus_signed_shares(self):
        """Both features together: Bloomberg tickers + signed shares."""
        csv = "ticker,shares\nAAPL US EQUITY,1000\nTSLA US EQUITY,-500\n"
        portfolio = load_from_csv_string(csv)
        aapl = portfolio.get_position("AAPL")
        tsla = portfolio.get_position("TSLA")
        assert aapl.side == "LONG"
        assert tsla.side == "SHORT"
        assert tsla.shares == 500.0


# ---------------------------------------------------------------------------
# Basic ingest coverage
# ---------------------------------------------------------------------------


class TestIngestBasic:
    """Basic ingest functionality."""

    def test_missing_ticker_raises(self):
        csv = "side,shares\nLONG,100\n"
        with pytest.raises(ValueError, match="ticker"):
            load_from_csv_string(csv)

    def test_missing_shares_and_weight_raises(self):
        csv = "ticker,side\nAAPL,LONG\n"
        with pytest.raises(ValueError, match="shares.*weight"):
            load_from_csv_string(csv)

    def test_basic_csv_import(self):
        csv = "ticker,side,shares,entry_price\nAAPL,LONG,100,150.0\nMSFT,SHORT,50,300.0\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.total_count == 2
        assert portfolio.long_count == 1
        assert portfolio.short_count == 1

    def test_side_aliases(self):
        csv = "ticker,side,shares\nAAPL,buy,100\nMSFT,sell,50\nGOOG,l,30\nAMZN,s,20\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").side == "LONG"
        assert portfolio.get_position("MSFT").side == "SHORT"
        assert portfolio.get_position("GOOG").side == "LONG"
        assert portfolio.get_position("AMZN").side == "SHORT"

    def test_column_aliases(self):
        """Alternative column names should be recognized."""
        csv = "symbol,direction,qty\nAAPL,long,100\n"
        portfolio = load_from_csv_string(csv)
        assert portfolio.get_position("AAPL").side == "LONG"
        assert portfolio.get_position("AAPL").shares == 100.0
