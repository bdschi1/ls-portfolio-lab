"""Abstract data provider interface.

Defines the contract for fetching market data.
Concrete implementations: yahoo_provider.py, bloomberg_provider.py,
ib_provider.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class DataProvider(ABC):
    """Abstract interface for market data providers."""

    @property
    def name(self) -> str:
        """Human-readable provider name (override in subclasses)."""
        return type(self).__name__

    @abstractmethod
    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """
        Fetch daily adjusted close prices for a list of tickers.

        Returns a DataFrame with columns: date, ticker, open, high, low, close, adj_close, volume
        """
        ...

    @abstractmethod
    def fetch_ticker_info(self, ticker: str) -> dict:
        """
        Fetch fundamental info for a single ticker.

        Returns dict with keys: market_cap, sector, industry, beta, shares_outstanding,
                                avg_volume, dividend_yield, short_pct_of_float
        """
        ...

    @abstractmethod
    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """
        Fetch latest closing price for a list of tickers.

        Returns: {ticker: price}
        """
        ...

    @abstractmethod
    def fetch_risk_free_rate(self) -> float:
        """
        Fetch current annualized risk-free rate.

        Uses 13-week T-bill rate (^IRX) by default.
        Returns as decimal (e.g., 0.05 for 5%).
        """
        ...
