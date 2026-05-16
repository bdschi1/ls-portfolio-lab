"""Tests for multi-tab Excel discovery, header-row detection, and unit-marker scaling."""

from pathlib import Path

import pytest

from data.ingest import (
    _extract_unit_scale,
    _find_portfolio_sheet_and_header,
    load_from_excel,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestUnitScaleExtraction:
    """Headers carrying ($M), ($MM), ($K), etc. have the scale stripped."""

    def test_no_marker(self):
        assert _extract_unit_scale("Shares") == ("Shares", 1.0)

    def test_dollar_m(self):
        assert _extract_unit_scale("Position Size ($M)") == ("Position Size", 1e6)

    def test_dollar_mm(self):
        assert _extract_unit_scale("Notional ($MM)") == ("Notional", 1e6)

    def test_dollar_k(self):
        assert _extract_unit_scale("Value ($K)") == ("Value", 1e3)

    def test_dollar_billions(self):
        assert _extract_unit_scale("NAV ($B)") == ("NAV", 1e9)

    def test_no_dollar_just_m(self):
        assert _extract_unit_scale("Shares (000s)") == ("Shares", 1e3)

    def test_millions_word(self):
        h, s = _extract_unit_scale("Amount (Millions)")
        assert h == "Amount"
        assert s == 1e6


class TestMultiTabDiscovery:
    """load_from_excel auto-finds the right tab and header row."""

    def test_finds_holdings_tab_with_offset_header(self):
        portfolio = load_from_excel(FIXTURES / "multi_tab_header_row.xlsx")
        assert portfolio.total_count == 2
        # Headers are at row 4 in the 'Holdings' tab; '$ Amount' col with entry_price
        aapl = portfolio.get_position("AAPL")
        msft = portfolio.get_position("MSFT")
        assert aapl is not None
        assert msft is not None
        # 50000 / 100 = 500 shares
        assert aapl.shares == 500.0
        assert aapl.side == "LONG"
        # 30000 / 200 = 150 shares
        assert msft.shares == 150.0
        assert msft.side == "SHORT"

    def test_trivial_two_column_on_second_tab(self):
        """Even a bare ticker+shares table on a non-first tab should be discovered."""
        portfolio = load_from_excel(FIXTURES / "two_col_trivial.xlsx")
        assert portfolio.total_count == 2
        assert portfolio.get_position("AAPL").shares == 100.0
        assert portfolio.get_position("AAPL").side == "LONG"
        assert portfolio.get_position("MSFT").shares == 50.0
        assert portfolio.get_position("MSFT").side == "SHORT"

    def test_dollar_m_with_price_fetcher(self):
        """Master Portfolio style: Position Size ($M), no entry_price, side column,
        prices supplied by fetcher."""
        prices = {"NVDA": 175.0, "INTC": 30.0}

        portfolio = load_from_excel(
            FIXTURES / "master_portfolio_dollarm.xlsx",
            price_fetcher=lambda tickers: {t: prices.get(t, 0.0) for t in tickers},
        )
        # Footer row is silently skipped → 2 positions
        assert portfolio.total_count == 2
        nvda = portfolio.get_position("NVDA")
        intc = portfolio.get_position("INTC")
        # $40M / $175 = 228,571.43 shares
        assert nvda.shares == pytest.approx(40_000_000 / 175.0)
        assert nvda.side == "LONG"
        assert nvda.beta == 1.75
        # $35M / $30 = 1,166,666.67 shares
        assert intc.shares == pytest.approx(35_000_000 / 30.0)
        assert intc.side == "SHORT"

    def test_dollar_m_preserves_fetched_price_as_entry_price(self):
        """Notional → shares conversion must store the fetched price on Position so
        downstream exposure (shares × entry_price) reproduces the original $ notional.
        Regression: prior to fix, entry_price fell back to the $1 placeholder and
        gross exposure collapsed by a factor of price."""
        prices = {"NVDA": 175.0, "INTC": 30.0}

        portfolio = load_from_excel(
            FIXTURES / "master_portfolio_dollarm.xlsx",
            price_fetcher=lambda tickers: {t: prices.get(t, 0.0) for t in tickers},
        )
        nvda = portfolio.get_position("NVDA")
        intc = portfolio.get_position("INTC")
        assert nvda.entry_price == pytest.approx(175.0)
        assert intc.entry_price == pytest.approx(30.0)
        # shares × entry_price reproduces the original $40M / $35M notionals
        assert nvda.shares * nvda.entry_price == pytest.approx(40_000_000)
        assert intc.shares * intc.entry_price == pytest.approx(35_000_000)

    def test_dollar_m_without_price_fetcher_skips(self):
        """Without a price source and no entry_price, $-notional rows skip and ingest fails."""
        with pytest.raises(ValueError, match="No valid positions"):
            load_from_excel(FIXTURES / "master_portfolio_dollarm.xlsx")


class TestSheetDiscoveryHelper:
    """Lower-level: _find_portfolio_sheet_and_header picks the right sheet/row."""

    def test_picks_holdings_over_dashboard(self):
        import openpyxl

        wb = openpyxl.load_workbook(
            FIXTURES / "multi_tab_header_row.xlsx", data_only=True, read_only=True
        )
        try:
            found = _find_portfolio_sheet_and_header(wb)
        finally:
            wb.close()
        assert found is not None
        sheet, row = found
        assert sheet == "Holdings"
        assert row == 4

    def test_picks_master_portfolio_over_summary(self):
        import openpyxl

        wb = openpyxl.load_workbook(
            FIXTURES / "master_portfolio_dollarm.xlsx", data_only=True, read_only=True
        )
        try:
            found = _find_portfolio_sheet_and_header(wb)
        finally:
            wb.close()
        assert found is not None
        sheet, row = found
        assert sheet == "Master Portfolio"
        assert row == 4
