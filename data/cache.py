"""Local SQLite cache for market data.

Caches daily prices and ticker info to avoid repeatedly hitting the upstream
data provider.  Works with **any** DataProvider — Yahoo Finance (default),
Bloomberg, or Interactive Brokers.  The active provider is injected at
construction time; swap it via ``set_provider()`` to switch data sources
without clearing cached prices.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from data.provider import DataProvider

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_cache")
DEFAULT_DB_NAME = "prices.db"


class DataCache:
    """SQLite-backed cache wrapping any DataProvider."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        price_staleness_hours: int = 18,
        info_staleness_days: int = 7,
        max_history_years: int = 3,
        provider: DataProvider | None = None,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / DEFAULT_DB_NAME
        self.price_staleness_hours = price_staleness_hours
        self.info_staleness_days = info_staleness_days
        self.max_history_years = max_history_years

        if provider is not None:
            self.provider = provider
        else:
            # Default to Yahoo Finance (always available, no extra deps)
            from data.yahoo_provider import YahooProvider
            self.provider = YahooProvider()

        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_info (
                    ticker TEXT PRIMARY KEY,
                    market_cap REAL,
                    sector TEXT,
                    industry TEXT,
                    beta REAL,
                    shares_outstanding INTEGER,
                    avg_volume INTEGER,
                    dividend_yield REAL,
                    short_pct_of_float REAL,
                    fetched_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
                ON daily_prices (ticker, date)
            """)

    # --- Price data ---

    def get_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """
        Get daily prices, using cache when fresh enough, fetching otherwise.

        Returns long-format polars DataFrame with columns:
        date, ticker, open, high, low, close, adj_close, volume
        """
        if not tickers:
            return self._empty_price_df()

        # Check which tickers need refreshing
        tickers_to_fetch: list[str] = []
        for ticker in tickers:
            if self._prices_stale(ticker, end):
                tickers_to_fetch.append(ticker)

        # Fetch missing data
        if tickers_to_fetch:
            logger.info("Fetching fresh prices for: %s", tickers_to_fetch)
            fresh = self.provider.fetch_daily_prices(tickers_to_fetch, start, end)
            if fresh.height > 0:
                self._store_prices(fresh)

        # Read from cache
        return self._read_cached_prices(tickers, start, end)

    def _prices_stale(self, ticker: str, end_date: date) -> bool:
        """Check if cached prices for a ticker are stale."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(fetched_at) FROM daily_prices WHERE ticker = ?",
                (ticker,),
            ).fetchone()

        if row is None or row[0] is None:
            return True

        fetched_at = datetime.fromisoformat(row[0])
        age = datetime.now() - fetched_at
        return age.total_seconds() > self.price_staleness_hours * 3600

    def _store_prices(self, df: pl.DataFrame) -> None:
        """Upsert price data into the cache."""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for row in df.iter_rows(named=True):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_prices
                    (ticker, date, open, high, low, close, adj_close, volume, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("ticker", ""),
                        str(row.get("date", "")),
                        row.get("open"),
                        row.get("high"),
                        row.get("low"),
                        row.get("close"),
                        row.get("adj_close"),
                        row.get("volume"),
                        now,
                    ),
                )

    def _read_cached_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Read cached price data for given tickers and date range."""
        placeholders = ",".join(["?"] * len(tickers))
        query = f"""
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM daily_prices
            WHERE ticker IN ({placeholders})
            AND date >= ? AND date <= ?
            ORDER BY ticker, date
        """
        params = [*tickers, str(start), str(end)]

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return self._empty_price_df()

        return pl.DataFrame(
            rows,
            schema={
                "ticker": pl.Utf8,
                "date": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "adj_close": pl.Float64,
                "volume": pl.Float64,
            },
            orient="row",
        ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

    def _empty_price_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "ticker": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "adj_close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    # --- Ticker info ---

    def get_ticker_info(self, ticker: str) -> dict:
        """Get ticker info, using cache when fresh enough."""
        cached = self._read_cached_info(ticker)
        if cached is not None:
            return cached

        logger.info("Fetching fresh info for: %s", ticker)
        info = self.provider.fetch_ticker_info(ticker)
        self._store_info(ticker, info)
        return info

    def get_batch_info(self, tickers: list[str]) -> dict[str, dict]:
        """Get info for multiple tickers."""
        return {t: self.get_ticker_info(t) for t in tickers}

    def _read_cached_info(self, ticker: str) -> dict | None:
        """Read cached info if fresh enough."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM ticker_info WHERE ticker = ?", (ticker,)
            ).fetchone()

        if row is None:
            return None

        fetched_at = datetime.fromisoformat(row[9])  # last column
        age = datetime.now() - fetched_at
        if age > timedelta(days=self.info_staleness_days):
            return None

        return {
            "market_cap": row[1] or 0,
            "sector": row[2] or "",
            "industry": row[3] or "",
            "beta": row[4] or 1.0,
            "shares_outstanding": row[5] or 0,
            "avg_volume": row[6] or 0,
            "dividend_yield": row[7] or 0.0,
            "short_pct_of_float": row[8] or 0.0,
        }

    def _store_info(self, ticker: str, info: dict) -> None:
        """Upsert ticker info into the cache."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ticker_info
                (ticker, market_cap, sector, industry, beta, shares_outstanding,
                 avg_volume, dividend_yield, short_pct_of_float, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    info.get("market_cap", 0),
                    info.get("sector", ""),
                    info.get("industry", ""),
                    info.get("beta", 1.0),
                    info.get("shares_outstanding", 0),
                    info.get("avg_volume", 0),
                    info.get("dividend_yield", 0.0),
                    info.get("short_pct_of_float", 0.0),
                    now,
                ),
            )

    # --- Current prices ---

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest prices (always live — not cached for staleness reasons)."""
        return self.provider.fetch_current_prices(tickers)

    # --- Risk-free rate ---

    def get_risk_free_rate(self) -> float:
        """Fetch current risk-free rate."""
        return self.provider.fetch_risk_free_rate()

    # --- Provider management ---

    def set_provider(self, provider: DataProvider) -> None:
        """Switch the upstream data provider.

        Cached data is retained — only future fetches use the new provider.
        Call ``clear_cache()`` first if you want a clean slate.
        """
        old_name = getattr(self.provider, "name", type(self.provider).__name__)
        new_name = getattr(provider, "name", type(provider).__name__)
        self.provider = provider
        logger.info("Data provider switched: %s → %s", old_name, new_name)

    @property
    def provider_name(self) -> str:
        """Human-readable name of the active provider."""
        return getattr(self.provider, "name", type(self.provider).__name__)

    # --- Utility ---

    def default_start_date(self) -> date:
        """Default start date for price history (max_history_years ago)."""
        return date.today() - timedelta(days=self.max_history_years * 365)

    def clear_cache(self) -> None:
        """Delete all cached data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM daily_prices")
            conn.execute("DELETE FROM ticker_info")
        logger.info("Cache cleared")
