"""Yahoo Finance data provider — concrete implementation of DataProvider.

Uses yfinance for all market data. Free, no API key required.
All data is EOD (end of day) — suitable for daily analytics, not real-time.

Resilience: all yfinance calls are wrapped with tenacity retry (exponential
backoff: 2s → 4s → 8s, 3 attempts) and a per-call timeout (60s on Unix via
SIGALRM, thread-based fallback on Windows/macOS).
"""

from __future__ import annotations

import logging
import platform
import signal
import threading
from datetime import date, timedelta
from functools import wraps
from typing import Any, Callable, TypeVar

import polars as pl
import yfinance as yf
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data.provider import DataProvider

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Timeout helper — SIGALRM on Linux, thread-based elsewhere
# ---------------------------------------------------------------------------

_USE_SIGALRM = platform.system() == "Linux"


class _YFinanceTimeout(Exception):
    """Raised when a yfinance call exceeds the timeout."""


def _timeout(seconds: int = 60) -> Callable[[F], F]:
    """Decorator that aborts a function after *seconds*.

    On Linux: uses SIGALRM (reliable, no thread overhead).
    Everywhere else: spawns a daemon thread and joins with timeout.
    """
    def decorator(func: F) -> F:
        if _USE_SIGALRM:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                def _handler(signum: int, frame: Any) -> None:
                    raise _YFinanceTimeout(
                        f"{func.__name__} timed out after {seconds}s"
                    )
                old = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
        else:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result: list[Any] = []
                exc: list[BaseException] = []

                def _target() -> None:
                    try:
                        result.append(func(*args, **kwargs))
                    except BaseException as e:
                        exc.append(e)

                t = threading.Thread(target=_target, daemon=True)
                t.start()
                t.join(timeout=seconds)
                if t.is_alive():
                    raise _YFinanceTimeout(
                        f"{func.__name__} timed out after {seconds}s"
                    )
                if exc:
                    raise exc[0]
                return result[0]
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Retry decorator — exponential backoff 2s→4s→8s, 3 attempts
# ---------------------------------------------------------------------------

_yf_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class YahooProvider(DataProvider):
    """Fetch market data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """
        Fetch daily OHLCV + adjusted close for multiple tickers.

        Returns long-format DataFrame: date, ticker, open, high, low, close, adj_close, volume
        """
        if not tickers:
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

        # yfinance download: returns pandas DataFrame with MultiIndex columns for multiple tickers
        # Add a buffer day to start to ensure we get the full range
        start_str = (start - timedelta(days=5)).isoformat()
        end_str = (end + timedelta(days=1)).isoformat()

        logger.info("Fetching prices for %d tickers: %s to %s", len(tickers), start_str, end_str)

        try:
            raw = self._download_prices(tickers, start_str, end_str)
        except Exception:
            logger.exception("yfinance download failed after retries")
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

        if raw.empty:
            logger.warning("yfinance returned empty data for %s", tickers)
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

        # Convert from wide pandas format to long polars format
        frames = []

        if len(tickers) == 1:
            # Single ticker: columns are just Open, High, Low, Close, Adj Close, Volume
            ticker = tickers[0]
            pdf = raw.reset_index()
            pdf.columns = [c.lower().replace(" ", "_") for c in pdf.columns]
            df = pl.from_pandas(pdf)
            # Rename columns
            rename_map = {}
            for col in df.columns:
                if "date" in col.lower():
                    rename_map[col] = "date"
            df = df.rename(rename_map) if rename_map else df
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            frames.append(df)
        else:
            # Multiple tickers: MultiIndex columns like (Open, AAPL), (Open, MSFT), ...
            for ticker in tickers:
                try:
                    if hasattr(raw.columns, "levels"):
                        ticker_data = raw.xs(ticker, axis=1, level=1)
                    else:
                        ticker_data = raw
                    pdf = ticker_data.reset_index()
                    pdf.columns = [c.lower().replace(" ", "_") for c in pdf.columns]
                    df = pl.from_pandas(pdf)
                    rename_map = {}
                    for col in df.columns:
                        if "date" in col.lower():
                            rename_map[col] = "date"
                    df = df.rename(rename_map) if rename_map else df
                    df = df.with_columns(pl.lit(ticker).alias("ticker"))
                    df = df.drop_nulls(subset=["close"])
                    frames.append(df)
                except (KeyError, ValueError):
                    logger.warning("No data for ticker %s, skipping", ticker)

        if not frames:
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

        result = pl.concat(frames, how="diagonal")

        # Standardize column names
        col_mapping = {
            "adj_close": "adj_close",
            "adjclose": "adj_close",
            "adjusted_close": "adj_close",
        }
        for old, new in col_mapping.items():
            if old in result.columns and old != new:
                result = result.rename({old: new})

        # Ensure date column is Date type
        if "date" in result.columns:
            result = result.with_columns(pl.col("date").cast(pl.Date))

        # Filter to requested date range
        result = result.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

        # Select and order columns
        available_cols = result.columns
        desired = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        final_cols = [c for c in desired if c in available_cols]

        return result.select(final_cols).sort(["ticker", "date"])

    @staticmethod
    @_yf_retry
    @_timeout(60)
    def _download_prices(tickers: list[str], start_str: str, end_str: str):
        """Resilient yfinance download with retry + timeout."""
        return yf.download(
            tickers=tickers,
            start=start_str,
            end=end_str,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    @staticmethod
    @_yf_retry
    @_timeout(30)
    def _fetch_info_raw(ticker: str) -> dict:
        """Resilient yfinance Ticker.info with retry + timeout."""
        t = yf.Ticker(ticker)
        return t.info

    def fetch_ticker_info(self, ticker: str) -> dict:
        """
        Fetch fundamental info for a single ticker.

        Returns dict with standardized keys.
        """
        try:
            info = self._fetch_info_raw(ticker)
        except Exception:
            logger.exception("Failed to fetch info for %s after retries", ticker)
            return {
                "market_cap": 0.0,
                "sector": "",
                "industry": "",
                "beta": 1.0,
                "shares_outstanding": 0,
                "avg_volume": 0,
                "dividend_yield": 0.0,
                "short_pct_of_float": 0.0,
            }

        return {
            "market_cap": info.get("marketCap", 0) or 0,
            "sector": info.get("sector", "") or "",
            "industry": info.get("industry", "") or "",
            "beta": info.get("beta", 1.0) or 1.0,
            "shares_outstanding": info.get("sharesOutstanding", 0) or 0,
            "avg_volume": info.get("averageVolume", 0) or 0,
            "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
            "short_pct_of_float": info.get("shortPercentOfFloat", 0.0) or 0.0,
        }

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest closing price for a list of tickers."""
        prices: dict[str, float] = {}

        if not tickers:
            return prices

        try:
            raw = self._download_current_prices(tickers)
        except Exception:
            logger.exception("Failed to fetch current prices after retries")
            return prices

        if raw.empty:
            return prices

        if len(tickers) == 1:
            last_close = raw["Close"].dropna().iloc[-1] if not raw["Close"].dropna().empty else 0.0
            prices[tickers[0]] = float(last_close)
        else:
            for ticker in tickers:
                try:
                    close = raw["Close"]
                    col = close[ticker] if hasattr(close, "__getitem__") else close
                    last = col.dropna().iloc[-1] if not col.dropna().empty else 0.0
                    prices[ticker] = float(last)
                except (KeyError, IndexError):
                    logger.warning("No price data for %s", ticker)

        return prices

    @staticmethod
    @_yf_retry
    @_timeout(60)
    def _download_current_prices(tickers: list[str]):
        """Resilient yfinance download for current prices with retry + timeout."""
        return yf.download(
            tickers=tickers,
            period="5d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    @staticmethod
    @_yf_retry
    @_timeout(30)
    def _fetch_risk_free_raw() -> float | None:
        """Resilient yfinance fetch for ^IRX with retry + timeout."""
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            rate = hist["Close"].dropna().iloc[-1]
            return float(rate) / 100.0
        return None

    def fetch_risk_free_rate(self) -> float:
        """
        Fetch current annualized risk-free rate from 13-week T-bill (^IRX).

        Returns as decimal (e.g., 0.05 for 5%).
        Falls back to 0.05 if unavailable.
        """
        try:
            rate = self._fetch_risk_free_raw()
            if rate is not None:
                return rate
        except Exception:
            logger.warning("Failed to fetch risk-free rate after retries, using 0.05 default")

        return 0.05
