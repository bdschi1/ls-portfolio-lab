"""Ticker universe definitions for portfolio generation.

Curated Russell 1000 universe (~500 liquid names, market cap > $5B) across
all 11 GICS sectors.  Healthcare has extra depth covering every GICS
sub-industry (Life Sciences Tools, Health Care Distributors, Managed Care,
Biotech, Pharma, MedTech, Hospitals, etc.).

Groups tickers by subsector for use in the constrained portfolio generator.
Each group represents a pool from which positions can be drawn.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe organized by subsector — used by mock_portfolio.py to sample from.
# All names should be liquid US-listed equities with market cap > $5B.
# This is the CANDIDATE pool; the generator selects a subset (~70 names).
# ---------------------------------------------------------------------------

MOCK_UNIVERSE: dict[str, list[str]] = {
    # ===================================================================
    # HEALTHCARE  (~80 names, 8 subsectors)
    # ===================================================================
    "Healthcare - Managed Care": [
        "UNH", "ELV", "CI", "HUM", "CNC", "MOH",
        "ALHC", "OSCR",
    ],
    "Healthcare - Hospitals / Facilities": [
        "HCA", "THC", "UHS", "CYH", "SEM",
        "ENSG", "ACHC", "SGRY",
    ],
    "Healthcare - MedTech / Devices": [
        "ISRG", "SYK", "MDT", "BSX", "EW", "ZBH", "HOLX", "ALGN",
        "DXCM", "PODD", "BDX", "BAX", "GEHC",
        "PEN", "GKOS", "INSP",
    ],
    "Healthcare - Life Sci Tools": [
        "TMO", "DHR", "A", "ILMN", "MTD", "WAT", "RVTY", "BIO",
        "CRL", "BRKR", "TECH", "TXG", "AZTA", "IDXX",
    ],
    "Healthcare - Biotech": [
        "AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA", "BMRN",
        "ALNY", "INCY", "PCVX", "IONS", "NBIX", "SRPT",
        "HALO", "EXAS", "RARE", "LEGN",
    ],
    "Healthcare - Pharma": [
        "LLY", "JNJ", "MRK", "ABBV", "PFE", "BMY", "ABT",
        "ZTS", "VTRS", "OGN",
    ],
    "Healthcare - Distributors": [
        "MCK", "COR", "CAH", "HSIC",
    ],
    "Healthcare - Services / Other": [
        "CVS", "VEEV", "RVMD", "NVCR", "DOCS",
        "GH", "INVA",
    ],
    # ===================================================================
    # TECHNOLOGY  (~65 names, 4 subsectors)
    # ===================================================================
    "Technology - Software": [
        "MSFT", "CRM", "NOW", "ADBE", "SNPS", "CDNS", "PANW", "FTNT",
        "WDAY", "HUBS", "DDOG", "NET", "ZS", "CRWD", "MDB",
        "TEAM", "MNDY", "BILL", "PCOR", "TOST",
        "INTU", "ADSK", "PLTR", "APP",
    ],
    "Technology - Semiconductors": [
        "NVDA", "AMD", "AVGO", "MRVL", "KLAC", "LRCX", "AMAT", "ON",
        "TER", "ENTG", "MPWR", "SWKS", "QCOM", "MU",
        "TXN", "ADI", "NXPI", "MCHP", "GFS",
    ],
    "Technology - Hardware": [
        "AAPL", "DELL", "HPQ", "HPE", "KEYS", "ZBRA",
        "SMCI", "ANET", "CSCO",
    ],
    "Technology - IT Services": [
        "ACN", "IBM", "FISV", "ADP", "PAYX", "IT",
        "CTSH", "GPN", "BR", "WEX", "DXC",
    ],
    # ===================================================================
    # FINANCIALS  (~55 names, 4 subsectors)
    # ===================================================================
    "Financials - Banks": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
        "TFC", "SCHW", "FITB", "CFG", "KEY", "MTB", "HBAN",
        "RF", "ZION",
    ],
    "Financials - Insurance": [
        "BRK-B", "PGR", "AIG", "MET", "PRU", "AFL", "TRV",
        "ALL", "CB", "HIG", "CINF", "GL", "AON", "MMC",
    ],
    "Financials - Capital Markets": [
        "BLK", "SPGI", "ICE", "CME", "MCO", "MSCI", "NDAQ",
        "MKTX", "COIN", "RJF", "LPLA", "HOOD",
    ],
    "Financials - Specialty Finance": [
        "AXP", "V", "MA", "COF", "SYF", "ALLY",
        "PYPL", "SOFI", "FIS",
    ],
    # ===================================================================
    # CONSUMER DISCRETIONARY  (~50 names, 4 subsectors)
    # ===================================================================
    "Consumer Discretionary - Retail": [
        "AMZN", "HD", "LOW", "TJX", "ROST", "TGT",
        "DG", "DLTR", "ORLY", "AZO", "BURL", "FIVE",
        "ULTA", "BBY",
    ],
    "Consumer Discretionary - Homebuilders / Housing": [
        "DHI", "LEN", "NVR", "PHM", "TOL",
        "WSM", "RH", "W", "ETSY",
    ],
    "Consumer Discretionary - Restaurants / Leisure": [
        "MCD", "SBUX", "CMG", "YUM", "DPZ", "DRI",
        "MAR", "HLT", "RCL", "NCLH", "CCL", "LVS",
        "WYNN", "BKNG",
    ],
    "Consumer Discretionary - Autos / Transport": [
        "TSLA", "F", "GM", "RIVN",
        "FDX", "UPS", "JBHT", "XPO", "ODFL",
        "DAL", "UAL", "LUV",
    ],
    # ===================================================================
    # INDUSTRIALS  (~50 names, 4 subsectors)
    # ===================================================================
    "Industrials - Aerospace / Defense": [
        "BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG",
        "HWM", "HEI", "AXON",
    ],
    "Industrials - Machinery / Equipment": [
        "HON", "CAT", "DE", "EMR", "ETN", "ROK", "AME",
        "PH", "CMI", "DOV", "IR",
    ],
    "Industrials - Engineering / Construction": [
        "GE", "GEV", "JCI", "CARR", "OTIS", "TT",
        "WM", "RSG", "VRSK", "PWR",
    ],
    "Industrials - Business Services": [
        "UNP", "CSX", "NSC", "WAB", "GWW", "FAST",
        "CTAS", "CPRT", "POOL", "UBER",
        "ABNB", "DASH",
    ],
    # ===================================================================
    # COMMUNICATION SERVICES  (~22 names, 3 subsectors)
    # ===================================================================
    "Communication Services - Interactive Media": [
        "GOOGL", "META", "SNAP", "PINS", "MTCH",
        "ZG", "TTD", "ROKU",
    ],
    "Communication Services - Media / Entertainment": [
        "DIS", "NFLX", "CMCSA", "WBD", "FOX",
        "LYV", "RBLX", "EA", "TTWO",
    ],
    "Communication Services - Telecom": [
        "T", "VZ", "TMUS", "LUMN",
    ],
    # ===================================================================
    # CONSUMER STAPLES  (~28 names, 3 subsectors)
    # ===================================================================
    "Consumer Staples - Food / Beverage": [
        "PEP", "KO", "MDLZ", "GIS", "KHC", "HSY", "SJM",
        "ADM", "BG", "STZ", "SAM", "MNST",
    ],
    "Consumer Staples - Household / Personal": [
        "PG", "CL", "EL", "CLX", "CHD",
        "KMB", "KVUE", "SWK",
    ],
    "Consumer Staples - Retail Staples": [
        "WMT", "COST", "KR", "SYY", "USFD",
        "PM", "MO",
    ],
    # ===================================================================
    # ENERGY  (~28 names, 3 subsectors)
    # ===================================================================
    "Energy - E&P": [
        "XOM", "CVX", "COP", "EOG", "DVN", "FANG",
        "OVV", "EQT", "AR",
    ],
    "Energy - Oilfield Services": [
        "SLB", "HAL", "BKR", "FTI", "NOV",
        "WFRD", "HP",
    ],
    "Energy - Integrated / Midstream": [
        "WMB", "KMI", "OKE", "ET", "TRGP", "PSX",
        "VLO", "MPC", "LNG",
    ],
    # ===================================================================
    # MATERIALS  (~22 names, 3 subsectors)
    # ===================================================================
    "Materials - Chemicals": [
        "LIN", "APD", "SHW", "ECL", "DD", "DOW",
        "PPG", "CE", "ALB", "FMC",
    ],
    "Materials - Metals / Mining": [
        "NEM", "FCX", "NUE", "STLD", "CLF", "AA",
        "RS",
    ],
    "Materials - Packaging / Specialty": [
        "BALL", "PKG", "AVY", "IP",
        "AMCR", "SEE",
    ],
    # ===================================================================
    # UTILITIES  (~18 names, 2 subsectors)
    # ===================================================================
    "Utilities - Electric": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
        "XEL", "WEC", "ED",
    ],
    "Utilities - Gas / Multi / Water": [
        "PCG", "EIX", "CMS", "DTE", "AES",
        "AWK", "WTRG", "NI",
    ],
    # ===================================================================
    # REAL ESTATE  (~22 names, 1 subsector)
    # ===================================================================
    "Real Estate - REITs": [
        "PLD", "AMT", "EQIX", "PSA", "O", "SPG", "WELL", "DLR",
        "AVB", "EQR", "VICI", "INVH",
        "ARE", "MAA", "UDR", "CPT", "ESS", "KIM",
        "IRM", "CCI", "EXR", "DOC",
    ],
}


def flat_universe() -> list[str]:
    """All tickers in the universe as a flat list."""
    tickers: list[str] = []
    for group in MOCK_UNIVERSE.values():
        tickers.extend(group)
    return tickers


def subsector_for_ticker(ticker: str) -> str:
    """Look up the subsector for a ticker in the mock universe."""
    for subsector, tickers in MOCK_UNIVERSE.items():
        if ticker in tickers:
            return subsector
    return "Unknown"


def sector_for_ticker(ticker: str) -> str:
    """Look up the top-level sector for a ticker in the mock universe."""
    subsector = subsector_for_ticker(ticker)
    if " - " in subsector:
        return subsector.split(" - ")[0]
    return subsector
