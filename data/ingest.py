"""Portfolio import from Excel, CSV, and PDF files.

Reads user-provided portfolio files and converts them into a Portfolio model.
Handles various column naming conventions and fills in missing data.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from core.portfolio import Portfolio, Position
from data.sector_map import classify_ticker

logger = logging.getLogger(__name__)

# Common column name aliases — map to our standard names
COLUMN_ALIASES: dict[str, list[str]] = {
    "ticker": ["ticker", "symbol", "sym", "stock", "name", "security"],
    "side": ["side", "direction", "long_short", "l/s", "ls", "position_type"],
    "shares": ["shares", "quantity", "qty", "size", "amount", "units"],
    "entry_price": ["entry_price", "entry", "avg_price", "avg_cost", "cost", "price", "avg"],
    "entry_date": ["entry_date", "date", "trade_date", "open_date", "start_date"],
    "sector": ["sector", "sector_override", "gics_sector"],
    "subsector": ["subsector", "industry", "sub_sector", "subsector_override"],
    "weight": ["weight", "wt", "pct", "allocation", "weight_pct"],
    "asset_type": ["asset_type", "type", "instrument", "instrument_type", "product_type"],
}

# Valid side labels → standardized
SIDE_ALIASES: dict[str, str] = {
    "long": "LONG",
    "l": "LONG",
    "buy": "LONG",
    "1": "LONG",
    "+1": "LONG",
    "short": "SHORT",
    "s": "SHORT",
    "sell": "SHORT",
    "-1": "SHORT",
}


def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to standard names using alias lookup."""
    rename_map: dict[str, str] = {}
    lower_cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols and standard not in rename_map.values():
                rename_map[lower_cols[alias]] = standard
                break

    return df.rename(rename_map)


def _standardize_side(side_val: str) -> str:
    """Convert various side labels to LONG or SHORT."""
    cleaned = str(side_val).strip().lower()
    result = SIDE_ALIASES.get(cleaned)
    if result is None:
        msg = f"Cannot interpret side value: '{side_val}'. Expected LONG/SHORT/L/S/BUY/SELL."
        raise ValueError(msg)
    return result


def load_from_excel(
    file_path: str | Path,
    nav: float = 3_000_000_000.0,
    sheet_name: str | None = None,
) -> Portfolio:
    """
    Load a portfolio from an Excel (.xlsx) or CSV (.csv) file.

    Minimum required columns: ticker, side, shares (or weight)
    Optional: entry_price, entry_date, sector, subsector

    If weight is provided instead of shares, positions are sized using NAV.
    If entry_price is missing, it defaults to 0 (updated when prices are fetched).
    """
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        df = pl.read_csv(path, infer_schema_length=1000)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pl.read_excel(path, sheet_name=sheet_name or 0)
    else:
        msg = f"Unsupported file type: {path.suffix}. Use .csv, .xlsx, or .xls."
        raise ValueError(msg)

    return _parse_portfolio_df(df, nav=nav, source=str(path))


def load_from_csv_string(csv_string: str, nav: float = 3_000_000_000.0) -> Portfolio:
    """Load a portfolio from a CSV string (for testing or direct input)."""
    import io

    df = pl.read_csv(io.StringIO(csv_string), infer_schema_length=1000)
    return _parse_portfolio_df(df, nav=nav, source="csv_string")


def _parse_portfolio_df(
    df: pl.DataFrame,
    nav: float = 3_000_000_000.0,
    source: str = "unknown",
) -> Portfolio:
    """Parse a polars DataFrame into a Portfolio."""
    df = _standardize_columns(df)

    # Validate required columns
    has_ticker = "ticker" in df.columns
    has_side = "side" in df.columns
    has_shares = "shares" in df.columns
    has_weight = "weight" in df.columns

    if not has_ticker:
        msg = f"Missing required column 'ticker' in {source}. Found: {df.columns}"
        raise ValueError(msg)
    if not has_side:
        msg = f"Missing required column 'side' in {source}. Found: {df.columns}"
        raise ValueError(msg)
    if not has_shares and not has_weight:
        msg = f"Need either 'shares' or 'weight' column in {source}. Found: {df.columns}"
        raise ValueError(msg)

    positions: list[Position] = []

    for row in df.iter_rows(named=True):
        ticker = str(row["ticker"]).strip().upper()
        side = _standardize_side(str(row["side"]))

        # Sizing: prefer shares, fall back to weight-based
        if has_shares and row.get("shares") is not None:
            shares = float(row["shares"])
        elif has_weight and row.get("weight") is not None:
            # Weight is a fraction of NAV; need price to compute shares
            # For now, store weight and resolve later when prices are available
            weight = float(row["weight"])
            # Placeholder: assume $100 price, will be corrected on price update
            shares = abs(weight) * nav / 100.0
        else:
            logger.warning("No shares or weight for %s, skipping", ticker)
            continue

        # Entry price (optional)
        entry_price = float(row.get("entry_price", 0) or 0)

        # Entry date (optional)
        entry_date_raw = row.get("entry_date")
        if entry_date_raw is not None and entry_date_raw != "":
            if isinstance(entry_date_raw, date):
                entry_date = entry_date_raw
            else:
                try:
                    entry_date = date.fromisoformat(str(entry_date_raw))
                except ValueError:
                    entry_date = date.today()
        else:
            entry_date = date.today()

        # Sector classification
        sector_override = str(row.get("sector", "") or "")
        subsector_override = str(row.get("subsector", "") or "")
        sector, subsector = classify_ticker(ticker)
        if sector_override:
            sector = sector_override
        if subsector_override:
            subsector = subsector_override

        # Asset type (optional — defaults to EQUITY)
        asset_type_raw = str(row.get("asset_type", "") or "").strip().upper()
        asset_type = asset_type_raw if asset_type_raw in ("EQUITY", "ETF", "OPTION") else "EQUITY"

        positions.append(
            Position(
                ticker=ticker,
                side=side,
                shares=shares,
                entry_price=entry_price if entry_price > 0 else 1.0,  # placeholder
                entry_date=entry_date,
                sector=sector,
                subsector=subsector,
                asset_type=asset_type,
            )
        )

    if not positions:
        msg = f"No valid positions found in {source}"
        raise ValueError(msg)

    return Portfolio(
        name=Path(source).stem if source != "csv_string" else "Imported",
        positions=positions,
        nav=nav,
    )


def load_from_pdf(file_path: str | Path, nav: float = 3_000_000_000.0) -> Portfolio:
    """
    Best-effort portfolio extraction from PDF.

    Uses pdfplumber to find tables with ticker/position data.
    Falls back to text extraction if table extraction fails.
    """
    import pdfplumber

    path = Path(file_path)
    all_tables: list[list[list[str]]] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)

    if not all_tables:
        msg = f"No tables found in PDF: {path}. Please use Excel/CSV format instead."
        raise ValueError(msg)

    # Try to find a table that looks like a portfolio (has ticker-like data)
    for table in all_tables:
        if len(table) < 2:
            continue

        # Use first row as headers, rest as data
        headers = [str(h).strip() for h in table[0]]
        rows = table[1:]

        try:
            col_data = {
                headers[i]: [
                    row[i] if i < len(row) else None for row in rows
                ]
                for i in range(len(headers))
            }
            df = pl.DataFrame(col_data)
            return _parse_portfolio_df(df, nav=nav, source=str(path))
        except (ValueError, KeyError):
            continue

    msg = f"Could not find portfolio data in any PDF table from {path}"
    raise ValueError(msg)
