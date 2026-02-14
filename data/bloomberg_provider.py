"""Bloomberg Professional API data provider.

Requires:
    - Bloomberg Terminal or B-PIPE running locally
    - blpapi Python SDK: pip install blpapi
    - DAPI entitlement on the Terminal

Connection: connects to localhost:8194 (default DAPI port).
Streams reference data from //blp/refdata and market data from //blp/mktdata.

All methods return the same schema as YahooProvider, so the rest of
the application works identically regardless of data source.

Usage:
    from data.bloomberg_provider import BloombergProvider
    provider = BloombergProvider()
    df = provider.fetch_daily_prices(["AAPL US Equity"], start, end)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl

from data.provider import DataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — blpapi may not be installed
# ---------------------------------------------------------------------------

try:
    import blpapi

    _HAS_BLPAPI = True
except ImportError:
    blpapi = None  # type: ignore[assignment]
    _HAS_BLPAPI = False


def is_available() -> bool:
    """Return True if blpapi is importable."""
    return _HAS_BLPAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_OPTIONS_DEFAULTS = {
    "host": "localhost",
    "port": 8194,
}

# Map Yahoo-style tickers to Bloomberg tickers
# "AAPL" → "AAPL US Equity"
def _to_bbg_ticker(ticker: str) -> str:
    """Convert a plain ticker to Bloomberg security identifier."""
    if " " in ticker:
        return ticker  # already in BBG format
    return f"{ticker} US Equity"


def _from_bbg_ticker(bbg_ticker: str) -> str:
    """Strip Bloomberg suffix back to plain ticker."""
    parts = bbg_ticker.split()
    return parts[0] if parts else bbg_ticker


class _BloombergSession:
    """Managed Bloomberg API session with auto-connect."""

    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host = host
        self.port = port
        self._session: Any = None

        if not _HAS_BLPAPI:
            msg = (
                "blpapi is not installed. Install with: "
                "pip install blpapi  "
                "(requires Bloomberg C++ SDK headers)"
            )
            raise ImportError(msg)

    def connect(self) -> Any:
        """Open a blpapi session, reusing if already connected."""
        if self._session is not None:
            return self._session

        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)

        session = blpapi.Session(opts)
        if not session.start():
            msg = f"Failed to start Bloomberg session on {self.host}:{self.port}"
            raise ConnectionError(msg)
        if not session.openService("//blp/refdata"):
            session.stop()
            msg = "Failed to open //blp/refdata service"
            raise ConnectionError(msg)

        logger.info("Bloomberg session connected to %s:%d", self.host, self.port)
        self._session = session
        return session

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except Exception:
                pass
            self._session = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class BloombergProvider(DataProvider):
    """Fetch market data from Bloomberg Professional API (DAPI/B-PIPE).

    Requires a running Bloomberg Terminal or B-PIPE connection.
    Falls back gracefully with clear error messages when unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
    ):
        self._host = host
        self._port = port
        self._bbg = _BloombergSession(host, port)

    @property
    def name(self) -> str:
        return "Bloomberg"

    # ------------------------------------------------------------------
    # Daily prices via //blp/refdata HistoricalDataRequest
    # ------------------------------------------------------------------

    def fetch_daily_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV for multiple tickers.

        Returns long-format DataFrame matching the standard schema:
        date, ticker, open, high, low, close, adj_close, volume
        """
        empty = self._empty_price_df()
        if not tickers:
            return empty

        try:
            session = self._bbg.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("Bloomberg connection failed: %s", exc)
            return empty

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("HistoricalDataRequest")

        for t in tickers:
            request.append("securities", _to_bbg_ticker(t))

        request.append("fields", "PX_OPEN")
        request.append("fields", "PX_HIGH")
        request.append("fields", "PX_LOW")
        request.append("fields", "PX_LAST")
        request.append("fields", "EQY_WEIGHTED_AVG_PX")  # VWAP as adj_close proxy
        request.append("fields", "VOLUME")

        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("adjustmentNormal", True)
        request.set("adjustmentAbnormal", True)
        request.set("adjustmentSplit", True)

        session.sendRequest(request)

        frames: list[dict] = []
        done = False

        while not done:
            event = session.nextEvent(5000)

            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    security_data = msg.getElement("securityData")
                    bbg_ticker = security_data.getElementAsString("security")
                    ticker = _from_bbg_ticker(bbg_ticker)

                    if security_data.hasElement("securityError"):
                        logger.warning("Bloomberg error for %s: %s", ticker,
                                       security_data.getElement("securityError"))
                        continue

                    field_data = security_data.getElement("fieldData")
                    for i in range(field_data.numValues()):
                        bar = field_data.getValueAsElement(i)
                        bar_date = bar.getElementAsDatetime("date")
                        frames.append({
                            "date": date(bar_date.year, bar_date.month, bar_date.day),
                            "ticker": ticker,
                            "open": _safe_float(bar, "PX_OPEN"),
                            "high": _safe_float(bar, "PX_HIGH"),
                            "low": _safe_float(bar, "PX_LOW"),
                            "close": _safe_float(bar, "PX_LAST"),
                            "adj_close": _safe_float(bar, "PX_LAST"),  # BBG auto-adjusts
                            "volume": _safe_float(bar, "VOLUME"),
                        })

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        if not frames:
            return empty

        return pl.DataFrame(frames).sort(["ticker", "date"])

    # ------------------------------------------------------------------
    # Ticker info via //blp/refdata ReferenceDataRequest
    # ------------------------------------------------------------------

    def fetch_ticker_info(self, ticker: str) -> dict:
        """Fetch fundamental reference data for a single ticker."""
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
            session = self._bbg.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("Bloomberg connection failed: %s", exc)
            return defaults

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("ReferenceDataRequest")
        request.append("securities", _to_bbg_ticker(ticker))

        fields = [
            "CUR_MKT_CAP",
            "GICS_SECTOR_NAME",
            "GICS_INDUSTRY_NAME",
            "BETA_ADJ_OVERRIDABLE",
            "EQY_SH_OUT",
            "VOLUME_AVG_30D",
            "EQY_DVD_YLD_IND",
            "SHORT_INT_PCT",
        ]
        for f in fields:
            request.append("fields", f)

        session.sendRequest(request)

        done = False
        while not done:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data_arr = msg.getElement("securityData")
                    if sec_data_arr.numValues() == 0:
                        continue
                    sec_data = sec_data_arr.getValueAsElement(0)

                    if sec_data.hasElement("securityError"):
                        logger.warning("Bloomberg error for %s", ticker)
                        continue

                    fd = sec_data.getElement("fieldData")
                    defaults["market_cap"] = _safe_ref_float(fd, "CUR_MKT_CAP") * 1e6
                    defaults["sector"] = _safe_ref_str(fd, "GICS_SECTOR_NAME")
                    defaults["industry"] = _safe_ref_str(fd, "GICS_INDUSTRY_NAME")
                    defaults["beta"] = _safe_ref_float(fd, "BETA_ADJ_OVERRIDABLE", 1.0)
                    defaults["shares_outstanding"] = int(
                        _safe_ref_float(fd, "EQY_SH_OUT") * 1e6
                    )
                    defaults["avg_volume"] = int(_safe_ref_float(fd, "VOLUME_AVG_30D"))
                    defaults["dividend_yield"] = (
                        _safe_ref_float(fd, "EQY_DVD_YLD_IND") / 100.0
                    )
                    defaults["short_pct_of_float"] = (
                        _safe_ref_float(fd, "SHORT_INT_PCT") / 100.0
                    )

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        return defaults

    # ------------------------------------------------------------------
    # Current prices — last price from reference data
    # ------------------------------------------------------------------

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest price for each ticker."""
        prices: dict[str, float] = {}
        if not tickers:
            return prices

        try:
            session = self._bbg.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("Bloomberg connection failed: %s", exc)
            return prices

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("ReferenceDataRequest")
        for t in tickers:
            request.append("securities", _to_bbg_ticker(t))
        request.append("fields", "PX_LAST")

        session.sendRequest(request)

        done = False
        while not done:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data_arr = msg.getElement("securityData")
                    for i in range(sec_data_arr.numValues()):
                        sec = sec_data_arr.getValueAsElement(i)
                        bbg_t = sec.getElementAsString("security")
                        ticker = _from_bbg_ticker(bbg_t)
                        if sec.hasElement("securityError"):
                            continue
                        fd = sec.getElement("fieldData")
                        px = _safe_ref_float(fd, "PX_LAST")
                        if px > 0:
                            prices[ticker] = px

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        return prices

    # ------------------------------------------------------------------
    # Risk-free rate — US 3-month T-bill
    # ------------------------------------------------------------------

    def fetch_risk_free_rate(self) -> float:
        """Fetch 3-month T-bill rate from Bloomberg."""
        try:
            session = self._bbg.connect()
        except (ConnectionError, ImportError):
            logger.warning("Bloomberg unavailable for risk-free rate, using 0.05")
            return 0.05

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("ReferenceDataRequest")
        request.append("securities", "GB3 Govt")  # 3-month T-bill
        request.append("fields", "PX_LAST")

        session.sendRequest(request)

        rate = 0.05
        done = False
        while not done:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sec_data_arr = msg.getElement("securityData")
                    if sec_data_arr.numValues() > 0:
                        sec = sec_data_arr.getValueAsElement(0)
                        if not sec.hasElement("securityError"):
                            fd = sec.getElement("fieldData")
                            val = _safe_ref_float(fd, "PX_LAST")
                            if val > 0:
                                rate = val / 100.0  # BBG quotes in pct

            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        return rate

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

    def close(self) -> None:
        """Shut down the Bloomberg session."""
        self._bbg.close()


# ---------------------------------------------------------------------------
# Safe element extraction (BBG elements can be missing/null)
# ---------------------------------------------------------------------------


def _safe_float(element: Any, field: str, default: float = 0.0) -> float:
    """Safely extract a float from a blpapi Element."""
    try:
        if element.hasElement(field):
            return float(element.getElementAsFloat64(field))
    except Exception:
        pass
    return default


def _safe_ref_float(element: Any, field: str, default: float = 0.0) -> float:
    """Safely extract a float from a ReferenceData fieldData Element."""
    try:
        if element.hasElement(field):
            return float(element.getElementAsFloat64(field))
    except Exception:
        pass
    return default


def _safe_ref_str(element: Any, field: str, default: str = "") -> str:
    """Safely extract a string from a ReferenceData fieldData Element."""
    try:
        if element.hasElement(field):
            return str(element.getElementAsString(field))
    except Exception:
        pass
    return default
