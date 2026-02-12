"""Tests for core/mock_portfolio.py"""

from core.mock_portfolio import _sample_positions, generate_mock_portfolio


class TestSamplePositions:
    def test_counts(self):
        longs, shorts = _sample_positions(30, 40, seed=42)
        assert len(longs) == 30
        assert len(shorts) == 40

    def test_no_overlap(self):
        longs, shorts = _sample_positions(seed=42)
        long_tickers = {t[0] for t in longs}
        short_tickers = {t[0] for t in shorts}
        assert long_tickers.isdisjoint(short_tickers)

    def test_sector_diversity(self):
        longs, shorts = _sample_positions(seed=42)
        long_sectors = {t[1] for t in longs}
        short_sectors = {t[1] for t in shorts}
        # Should have many sectors on each side (11 GICS sectors)
        assert len(long_sectors) >= 5
        assert len(short_sectors) >= 5

    def test_all_gics_sectors_represented(self):
        """Both sides should have all 11 GICS sectors."""
        longs, shorts = _sample_positions(seed=42)
        long_sectors = {t[1] for t in longs}
        short_sectors = {t[1] for t in shorts}
        expected = {
            "Communication Services",
            "Consumer Discretionary",
            "Consumer Staples",
            "Energy",
            "Financials",
            "Healthcare",
            "Industrials",
            "Materials",
            "Real Estate",
            "Technology",
            "Utilities",
        }
        assert long_sectors == expected
        assert short_sectors == expected


class TestGenerateMockPortfolio:
    def test_basic_generation(self):
        portfolio = generate_mock_portfolio(seed=42)
        assert portfolio.long_count >= 20  # minimum 20 longs
        assert portfolio.short_count > portfolio.long_count
        assert portfolio.total_count == (
            portfolio.long_count + portfolio.short_count
        )
        # With defaults ~30L/~40S = ~70 total
        assert 60 <= portfolio.total_count <= 80

    def test_no_duplicates(self):
        portfolio = generate_mock_portfolio(seed=42)
        tickers = portfolio.tickers
        assert len(tickers) == len(set(tickers))

    def test_has_sectors(self):
        portfolio = generate_mock_portfolio(seed=42)
        sectors = {p.sector for p in portfolio.positions}
        assert len(sectors) >= 5

    def test_with_betas(self):
        """Should work with provided betas."""
        betas = {"AAPL": 1.2, "MSFT": 1.0, "UNH": 0.8}
        portfolio = generate_mock_portfolio(betas=betas, seed=42)
        assert portfolio.long_count >= 20
        assert portfolio.short_count > portfolio.long_count

    def test_with_prices(self):
        """Should size positions correctly when prices provided."""
        prices = {"AAPL": 190.0, "MSFT": 420.0, "UNH": 560.0}
        portfolio = generate_mock_portfolio(prices=prices, seed=42)
        # Positions with known prices should use those prices
        for p in portfolio.positions:
            if p.ticker in prices:
                assert p.entry_price == prices[p.ticker]

    def test_reproducible(self):
        """Same seed should produce same portfolio."""
        p1 = generate_mock_portfolio(seed=123)
        p2 = generate_mock_portfolio(seed=123)
        assert p1.tickers == p2.tickers

    def test_nav_default_3b(self):
        """NAV should default to $3B."""
        portfolio = generate_mock_portfolio(seed=42)
        assert portfolio.nav == 3_000_000_000.0

    def test_cash_initialized(self):
        """Cash should be auto-computed after portfolio generation."""
        portfolio = generate_mock_portfolio(seed=42)
        assert portfolio.cash is not None
        assert portfolio.cash > 0
        assert portfolio.cash < portfolio.nav  # some capital is invested

    def test_universe_size(self):
        """Universe should have ~440+ tickers (Russell 1000 curated)."""
        from data.universe import flat_universe

        assert len(flat_universe()) >= 400

    def test_universe_sector_map_sync(self):
        """Universe and sector_map must contain the same tickers."""
        from data.sector_map import all_tickers_in_universe
        from data.universe import flat_universe

        assert set(flat_universe()) == set(all_tickers_in_universe())

    def test_all_gics_sectors_in_universe(self):
        """Universe should cover all 11 GICS sectors."""
        from data.universe import MOCK_UNIVERSE

        sectors = set()
        for subsector_key in MOCK_UNIVERSE:
            sector = subsector_key.split(" - ")[0]
            sectors.add(sector)
        assert len(sectors) == 11

    def test_min_longs_constraint(self):
        """Must have at least 20 longs."""
        portfolio = generate_mock_portfolio(seed=42)
        assert portfolio.long_count >= 20

    def test_more_shorts_than_longs(self):
        """Must have more shorts than longs."""
        portfolio = generate_mock_portfolio(seed=42)
        assert portfolio.short_count > portfolio.long_count
