"""Interactive Brokers data provider via ib_insync.

Requires:
    - IB account (paper or live)
    - TWS or IB Gateway running locally
    - ib_insync: pip install ib_insync

Connection: localhost:7497 (TWS paper), 7496 (TWS live),
            4002 (Gateway paper), 4001 (Gateway live).

Supports:
    - Historical daily bars (up to 1 year for free, longer with market data sub)
    - Real-time last price snapshots
    - Fundamental reference data (requires market data subscription)

All methods return the same schema as YahooProvider.

Usage:
    from data.ib_provider import IBProvider
    provider = IBProvider(port=7497)  # paper trading
    df = provider.fetch_daily_prices(["AAPL"], start, end)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from data.provider import DataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — ib_insync may not be installed
# ---------------------------------------------------------------------------

try:
    from ib_insync import IB, Stock, util

    _HAS_IB = True
except ImportError:
    IB = None  # type: ignore[assignment, misc]
    Stock = None  # type: ignore[assignment, misc]
    util = None  # type: ignore[assignment]
    _HAS_IB = False


def is_available() -> bool:
    """Return True if ib_insync is importable."""
    return _HAS_IB


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7497  # TWS paper trading
_DEFAULT_CLIENT_ID = 10  # avoid conflicts with other IB apps


class _IBConnection:
    """Managed IB connection with auto-connect and reconnect."""

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        client_id: int = _DEFAULT_CLIENT_ID,
        timeout: int = 15,
        readonly: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self._ib: Any = None

        if not _HAS_IB:
            msg = (
                "ib_insync is not installed. Install with: "
                "pip install ib_insync"
            )
            raise ImportError(msg)

    def connect(self) -> Any:
        """Return a connected IB instance, reconnecting if needed."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        ib = IB()
        try:
            ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.readonly,
            )
        except Exception as exc:
            msg = (
                f"Cannot connect to IB on {self.host}:{self.port}. "
                f"Ensure TWS/Gateway is running. Error: {exc}"
            )
            raise ConnectionError(msg) from exc

        logger.info(
            "IB connected to %s:%d (clientId=%d, readonly=%s)",
            self.host, self.port, self.client_id, self.readonly,
        )
        self._ib = ib
        return ib

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def __del__(self) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contract(ticker: str) -> Any:
    """Create an IB Stock contract for a US equity ticker."""
    return Stock(ticker, "SMART", "USD")


def _ib_duration(start: date, end: date) -> str:
    """Convert a date range into IB's durationStr format.

    IB uses strings like "1 Y", "6 M", "30 D".
    """
    days = (end - start).days + 1
    if days > 365:
        years = days // 365
        return f"{min(years, 5)} Y"
    if days > 30:
        months = days // 30
        return f"{months} M"
    return f"{days} D"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class IBProvider(DataProvider):
    """Fetch market data from Interactive Brokers via ib_insync.

    Requires TWS or IB Gateway running locally.
    Connects in readonly mode by default (no order capability).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        client_id: int = _DEFAULT_CLIENT_ID,
        timeout: int = 15,
    ):
        self._conn = _IBConnection(
            host=host, port=port, client_id=client_id,
            timeout=timeout, readonly=True,
        )

    @property
    def name(self) -> str:
        return "Interactive Brokers"

    # ------------------------------------------------------------------
    # Daily prices via reqHistoricalData
    # ------------------------------------------------------------------

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV bars from IB.

        Returns long-format DataFrame: date, ticker, open, high, low, close,
        adj_close, volume.

        Note: IB historical data pacing rules apply (~60 requests / 10 min).
        For large universes, use the cache layer to avoid re-fetching.
        """
        empty = self._empty_price_df()
        if not tickers:
            return empty

        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("IB connection failed: %s", exc)
            return empty

        frames: list[dict] = []
        duration = _ib_duration(start, end)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)

        for ticker in tickers:
            contract = _make_contract(ticker)

            try:
                ib.qualifyContracts(contract)
            except Exception:
                logger.warning("IB cannot qualify contract for %s", ticker)
                continue

            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,       # regular trading hours only
                    formatDate=1,      # yyyyMMdd format
                    keepUpToDate=False,
                )
            except Exception as exc:
                logger.warning("IB historical data failed for %s: %s", ticker, exc)
                continue

            if not bars:
                logger.warning("No IB data returned for %s", ticker)
                continue

            for bar in bars:
                bar_date = bar.date
                if isinstance(bar_date, str):
                    bar_date = datetime.strptime(bar_date, "%Y%m%d").date()
                elif isinstance(bar_date, datetime):
                    bar_date = bar_date.date()

                # Filter to requested range
                if bar_date < start or bar_date > end:
                    continue

                frames.append({
                    "date": bar_date,
                    "ticker": ticker,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "adj_close": float(bar.close),  # ADJUSTED_LAST is already adjusted
                    "volume": float(bar.volume),
                })

            # IB pacing: small sleep between requests to avoid throttling
            ib.sleep(0.5)

        if not frames:
            return empty

        return pl.DataFrame(frames).sort(["ticker", "date"])

    # ------------------------------------------------------------------
    # Ticker info via reqFundamentalData / contract details
    # ------------------------------------------------------------------

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamental data from IB contract details + fundamentals.

        IB provides limited fundamental data compared to Bloomberg.
        Sector/industry come from contract details; beta requires market
        data subscription.
        """
        defaults = {
            "market_cap": 0.0,
            "sector": "",
            "industry": "",
            "beta": 1.0,
            "shares_outstanding": 0,
            "avg_volume": 0,
            "dividend_yield": 0.0,
            "short_pct_of_float": 0.0,
        }

        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("IB connection failed: %s", exc)
            return defaults

        contract = _make_contract(ticker)

        try:
            ib.qualifyContracts(contract)
        except Exception:
            logger.warning("IB cannot qualify contract for %s", ticker)
            return defaults

        # Contract details give us sector/industry
        try:
            details_list = ib.reqContractDetails(contract)
            if details_list:
                details = details_list[0]
                defaults["sector"] = getattr(details, "category", "") or ""
                defaults["industry"] = getattr(details, "industry", "") or ""
        except Exception as exc:
            logger.warning("IB contract details failed for %s: %s", ticker, exc)

        # Fundamental data (requires subscription) — parse XML
        try:
            fundamentals = ib.reqFundamentalData(contract, "ReportSnapshot")
            if fundamentals:
                defaults = _parse_ib_fundamentals(fundamentals, defaults)
        except Exception as exc:
            logger.debug(
                "IB fundamental data unavailable for %s (may need subscription): %s",
                ticker, exc,
            )

        return defaults

    # ------------------------------------------------------------------
    # Current prices — snapshot via reqMktData
    # ------------------------------------------------------------------

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest price for each ticker using IB market data snapshot."""
        prices: dict[str, float] = {}
        if not tickers:
            return prices

        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("IB connection failed: %s", exc)
            return prices

        for ticker in tickers:
            contract = _make_contract(ticker)

            try:
                ib.qualifyContracts(contract)
            except Exception:
                continue

            try:
                # Request a snapshot (no streaming)
                ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
                ib.sleep(1.0)  # wait for data

                ticker_obj = ib.ticker(contract)
                if ticker_obj and ticker_obj.last and ticker_obj.last > 0:
                    prices[ticker] = float(ticker_obj.last)
                elif ticker_obj and ticker_obj.close and ticker_obj.close > 0:
                    prices[ticker] = float(ticker_obj.close)

                ib.cancelMktData(contract)
            except Exception as exc:
                logger.warning("IB price snapshot failed for %s: %s", ticker, exc)

        return prices

    # ------------------------------------------------------------------
    # Risk-free rate — US 3-month T-bill via IB
    # ------------------------------------------------------------------

    def fetch_risk_free_rate(self) -> float:
        """Fetch 3-month T-bill rate from IB.

        Falls back to 0.05 if unavailable (IB doesn't always have
        government bond data without specific subscriptions).
        """
        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError):
            return 0.05

        # Try to get 3-month T-bill via historical data
        # IB ticker for 13-week T-bill: use IRX index as proxy
        try:
            from ib_insync import Index
            contract = Index("IRX", "CBOE")
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if bars:
                last_close = float(bars[-1].close)
                return last_close / 100.0  # IRX quotes in percentage points
        except Exception as exc:
            logger.debug("IB risk-free rate fetch failed: %s", exc)

        return 0.05

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_price_df() -> pl.DataFrame:
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

    def disconnect(self) -> None:
        """Disconnect from IB."""
        self._conn.disconnect()

    def close(self) -> None:
        """Alias for disconnect."""
        self.disconnect()


# ---------------------------------------------------------------------------
# XML parser for IB fundamental data (ReportSnapshot)
# ---------------------------------------------------------------------------


def _parse_ib_fundamentals(xml_str: str, defaults: dict) -> dict:
    """Parse IB's ReportSnapshot XML into our standard dict.

    IB returns XML like:
    <ReportSnapshot>
      <CoIDs><CoID Type="Ticker">AAPL</CoID></CoIDs>
      <Ratios>
        <Group><Ratio FieldName="MKTCAP">2800000</Ratio>...</Group>
      </Ratios>
    </ReportSnapshot>
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_str)

        # Extract market cap, beta, dividend yield from ratios
        for ratio in root.iter("Ratio"):
            field = ratio.get("FieldName", "")
            val = ratio.text

            if not val:
                continue

            try:
                if field == "MKTCAP":
                    defaults["market_cap"] = float(val) * 1e6
                elif field == "BETA":
                    defaults["beta"] = float(val)
                elif field == "YIELD":
                    defaults["dividend_yield"] = float(val) / 100.0
                elif field == "SHARESOUT":
                    defaults["shares_outstanding"] = int(float(val) * 1e6)
                elif field == "AVOLUME":
                    defaults["avg_volume"] = int(float(val))
            except (ValueError, TypeError):
                continue

    except Exception as exc:
        logger.debug("Failed to parse IB fundamentals XML: %s", exc)

    return defaults
