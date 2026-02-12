"""Sector and subsector classification mapping.

Maps tickers to GICS-like sector and subsector labels.
Uses manual overrides for the curated Russell 1000 universe (~440 names),
with yfinance industry data as fallback for tickers not in the universe.
"""

from __future__ import annotations

# Manual subsector overrides — these take priority over yfinance classification.
# The keys are the subsector labels we care about for portfolio analytics.
# This is the source of truth for the curated Russell 1000 universe.

SUBSECTOR_OVERRIDES: dict[str, tuple[str, str]] = {
    # (sector, subsector)
    # =================================================================
    # Healthcare
    # =================================================================
    # Healthcare - Managed Care
    "UNH": ("Healthcare", "Managed Care"),
    "ELV": ("Healthcare", "Managed Care"),
    "CI": ("Healthcare", "Managed Care"),
    "HUM": ("Healthcare", "Managed Care"),
    "CNC": ("Healthcare", "Managed Care"),
    "MOH": ("Healthcare", "Managed Care"),
    "ALHC": ("Healthcare", "Managed Care"),
    "OSCR": ("Healthcare", "Managed Care"),
    # Healthcare - Hospitals / Facilities
    "HCA": ("Healthcare", "Hospitals / Facilities"),
    "THC": ("Healthcare", "Hospitals / Facilities"),
    "UHS": ("Healthcare", "Hospitals / Facilities"),
    "CYH": ("Healthcare", "Hospitals / Facilities"),
    "SEM": ("Healthcare", "Hospitals / Facilities"),
    "ENSG": ("Healthcare", "Hospitals / Facilities"),
    "ACHC": ("Healthcare", "Hospitals / Facilities"),
    "SGRY": ("Healthcare", "Hospitals / Facilities"),
    # Healthcare - MedTech / Devices
    "ISRG": ("Healthcare", "MedTech / Devices"),
    "SYK": ("Healthcare", "MedTech / Devices"),
    "MDT": ("Healthcare", "MedTech / Devices"),
    "BSX": ("Healthcare", "MedTech / Devices"),
    "EW": ("Healthcare", "MedTech / Devices"),
    "ZBH": ("Healthcare", "MedTech / Devices"),
    "HOLX": ("Healthcare", "MedTech / Devices"),
    "ALGN": ("Healthcare", "MedTech / Devices"),
    "DXCM": ("Healthcare", "MedTech / Devices"),
    "PODD": ("Healthcare", "MedTech / Devices"),
    "BDX": ("Healthcare", "MedTech / Devices"),
    "BAX": ("Healthcare", "MedTech / Devices"),
    "GEHC": ("Healthcare", "MedTech / Devices"),
    "PEN": ("Healthcare", "MedTech / Devices"),
    "GKOS": ("Healthcare", "MedTech / Devices"),
    "INSP": ("Healthcare", "MedTech / Devices"),
    # Healthcare - Life Sci Tools
    "TMO": ("Healthcare", "Life Sci Tools"),
    "DHR": ("Healthcare", "Life Sci Tools"),
    "A": ("Healthcare", "Life Sci Tools"),
    "ILMN": ("Healthcare", "Life Sci Tools"),
    "MTD": ("Healthcare", "Life Sci Tools"),
    "WAT": ("Healthcare", "Life Sci Tools"),
    "RVTY": ("Healthcare", "Life Sci Tools"),
    "BIO": ("Healthcare", "Life Sci Tools"),
    "CRL": ("Healthcare", "Life Sci Tools"),
    "BRKR": ("Healthcare", "Life Sci Tools"),
    "TECH": ("Healthcare", "Life Sci Tools"),
    "TXG": ("Healthcare", "Life Sci Tools"),
    "AZTA": ("Healthcare", "Life Sci Tools"),
    "IDXX": ("Healthcare", "Life Sci Tools"),
    # Healthcare - Biotech
    "AMGN": ("Healthcare", "Biotech"),
    "GILD": ("Healthcare", "Biotech"),
    "VRTX": ("Healthcare", "Biotech"),
    "REGN": ("Healthcare", "Biotech"),
    "BIIB": ("Healthcare", "Biotech"),
    "MRNA": ("Healthcare", "Biotech"),
    "BMRN": ("Healthcare", "Biotech"),
    "ALNY": ("Healthcare", "Biotech"),
    "INCY": ("Healthcare", "Biotech"),
    "PCVX": ("Healthcare", "Biotech"),
    "IONS": ("Healthcare", "Biotech"),
    "NBIX": ("Healthcare", "Biotech"),
    "SRPT": ("Healthcare", "Biotech"),
    "HALO": ("Healthcare", "Biotech"),
    "EXAS": ("Healthcare", "Biotech"),
    "RARE": ("Healthcare", "Biotech"),
    "LEGN": ("Healthcare", "Biotech"),
    # Healthcare - Pharma
    "LLY": ("Healthcare", "Pharma"),
    "JNJ": ("Healthcare", "Pharma"),
    "MRK": ("Healthcare", "Pharma"),
    "ABBV": ("Healthcare", "Pharma"),
    "PFE": ("Healthcare", "Pharma"),
    "BMY": ("Healthcare", "Pharma"),
    "ABT": ("Healthcare", "Pharma"),
    "ZTS": ("Healthcare", "Pharma"),
    "VTRS": ("Healthcare", "Pharma"),
    "OGN": ("Healthcare", "Pharma"),
    # Healthcare - Distributors
    "MCK": ("Healthcare", "Distributors"),
    "COR": ("Healthcare", "Distributors"),
    "CAH": ("Healthcare", "Distributors"),
    "HSIC": ("Healthcare", "Distributors"),
    # Healthcare - Services / Other
    "CVS": ("Healthcare", "Services / Other"),
    "VEEV": ("Healthcare", "Services / Other"),
    "RVMD": ("Healthcare", "Services / Other"),
    "NVCR": ("Healthcare", "Services / Other"),
    "DOCS": ("Healthcare", "Services / Other"),
    "GH": ("Healthcare", "Services / Other"),
    "INVA": ("Healthcare", "Services / Other"),
    # =================================================================
    # Technology
    # =================================================================
    # Technology - Software
    "MSFT": ("Technology", "Software"),
    "CRM": ("Technology", "Software"),
    "NOW": ("Technology", "Software"),
    "ADBE": ("Technology", "Software"),
    "SNPS": ("Technology", "Software"),
    "CDNS": ("Technology", "Software"),
    "PANW": ("Technology", "Software"),
    "FTNT": ("Technology", "Software"),
    "WDAY": ("Technology", "Software"),
    "HUBS": ("Technology", "Software"),
    "DDOG": ("Technology", "Software"),
    "NET": ("Technology", "Software"),
    "ZS": ("Technology", "Software"),
    "CRWD": ("Technology", "Software"),
    "MDB": ("Technology", "Software"),
    "TEAM": ("Technology", "Software"),
    "MNDY": ("Technology", "Software"),
    "BILL": ("Technology", "Software"),
    "PCOR": ("Technology", "Software"),
    "TOST": ("Technology", "Software"),
    "INTU": ("Technology", "Software"),
    "ADSK": ("Technology", "Software"),
    "PLTR": ("Technology", "Software"),
    "APP": ("Technology", "Software"),
    # Technology - Semiconductors
    "NVDA": ("Technology", "Semiconductors"),
    "AMD": ("Technology", "Semiconductors"),
    "AVGO": ("Technology", "Semiconductors"),
    "MRVL": ("Technology", "Semiconductors"),
    "KLAC": ("Technology", "Semiconductors"),
    "LRCX": ("Technology", "Semiconductors"),
    "AMAT": ("Technology", "Semiconductors"),
    "ON": ("Technology", "Semiconductors"),
    "TER": ("Technology", "Semiconductors"),
    "ENTG": ("Technology", "Semiconductors"),
    "MPWR": ("Technology", "Semiconductors"),
    "SWKS": ("Technology", "Semiconductors"),
    "QCOM": ("Technology", "Semiconductors"),
    "MU": ("Technology", "Semiconductors"),
    "TXN": ("Technology", "Semiconductors"),
    "ADI": ("Technology", "Semiconductors"),
    "NXPI": ("Technology", "Semiconductors"),
    "MCHP": ("Technology", "Semiconductors"),
    "GFS": ("Technology", "Semiconductors"),
    # Technology - Hardware
    "AAPL": ("Technology", "Hardware"),
    "DELL": ("Technology", "Hardware"),
    "HPQ": ("Technology", "Hardware"),
    "HPE": ("Technology", "Hardware"),
    "KEYS": ("Technology", "Hardware"),
    "ZBRA": ("Technology", "Hardware"),
    "SMCI": ("Technology", "Hardware"),
    "ANET": ("Technology", "Hardware"),
    "CSCO": ("Technology", "Hardware"),
    # Technology - IT Services
    "ACN": ("Technology", "IT Services"),
    "IBM": ("Technology", "IT Services"),
    "FISV": ("Technology", "IT Services"),
    "ADP": ("Technology", "IT Services"),
    "PAYX": ("Technology", "IT Services"),
    "IT": ("Technology", "IT Services"),
    "CTSH": ("Technology", "IT Services"),
    "GPN": ("Technology", "IT Services"),
    "BR": ("Technology", "IT Services"),
    "WEX": ("Technology", "IT Services"),
    "DXC": ("Technology", "IT Services"),
    # =================================================================
    # Financials
    # =================================================================
    # Financials - Banks
    "JPM": ("Financials", "Banks"),
    "BAC": ("Financials", "Banks"),
    "WFC": ("Financials", "Banks"),
    "GS": ("Financials", "Banks"),
    "MS": ("Financials", "Banks"),
    "C": ("Financials", "Banks"),
    "USB": ("Financials", "Banks"),
    "PNC": ("Financials", "Banks"),
    "TFC": ("Financials", "Banks"),
    "SCHW": ("Financials", "Banks"),
    "FITB": ("Financials", "Banks"),
    "CFG": ("Financials", "Banks"),
    "KEY": ("Financials", "Banks"),
    "MTB": ("Financials", "Banks"),
    "HBAN": ("Financials", "Banks"),
    "RF": ("Financials", "Banks"),
    "ZION": ("Financials", "Banks"),
    # Financials - Insurance
    "BRK-B": ("Financials", "Insurance"),
    "PGR": ("Financials", "Insurance"),
    "AIG": ("Financials", "Insurance"),
    "MET": ("Financials", "Insurance"),
    "PRU": ("Financials", "Insurance"),
    "AFL": ("Financials", "Insurance"),
    "TRV": ("Financials", "Insurance"),
    "ALL": ("Financials", "Insurance"),
    "CB": ("Financials", "Insurance"),
    "HIG": ("Financials", "Insurance"),
    "CINF": ("Financials", "Insurance"),
    "GL": ("Financials", "Insurance"),
    "AON": ("Financials", "Insurance"),
    "MMC": ("Financials", "Insurance"),
    # Financials - Capital Markets
    "BLK": ("Financials", "Capital Markets"),
    "SPGI": ("Financials", "Capital Markets"),
    "ICE": ("Financials", "Capital Markets"),
    "CME": ("Financials", "Capital Markets"),
    "MCO": ("Financials", "Capital Markets"),
    "MSCI": ("Financials", "Capital Markets"),
    "NDAQ": ("Financials", "Capital Markets"),
    "MKTX": ("Financials", "Capital Markets"),
    "COIN": ("Financials", "Capital Markets"),
    "RJF": ("Financials", "Capital Markets"),
    "LPLA": ("Financials", "Capital Markets"),
    "HOOD": ("Financials", "Capital Markets"),
    # Financials - Specialty Finance
    "AXP": ("Financials", "Specialty Finance"),
    "V": ("Financials", "Specialty Finance"),
    "MA": ("Financials", "Specialty Finance"),
    "COF": ("Financials", "Specialty Finance"),
    "SYF": ("Financials", "Specialty Finance"),
    "ALLY": ("Financials", "Specialty Finance"),
    "PYPL": ("Financials", "Specialty Finance"),
    "SOFI": ("Financials", "Specialty Finance"),
    "FIS": ("Financials", "Specialty Finance"),
    # =================================================================
    # Consumer Discretionary
    # =================================================================
    # Consumer Discretionary - Retail
    "AMZN": ("Consumer Discretionary", "Retail"),
    "HD": ("Consumer Discretionary", "Retail"),
    "LOW": ("Consumer Discretionary", "Retail"),
    "TJX": ("Consumer Discretionary", "Retail"),
    "ROST": ("Consumer Discretionary", "Retail"),
    "TGT": ("Consumer Discretionary", "Retail"),
    "DG": ("Consumer Discretionary", "Retail"),
    "DLTR": ("Consumer Discretionary", "Retail"),
    "ORLY": ("Consumer Discretionary", "Retail"),
    "AZO": ("Consumer Discretionary", "Retail"),
    "BURL": ("Consumer Discretionary", "Retail"),
    "FIVE": ("Consumer Discretionary", "Retail"),
    "ULTA": ("Consumer Discretionary", "Retail"),
    "BBY": ("Consumer Discretionary", "Retail"),
    # Consumer Discretionary - Homebuilders / Housing
    "DHI": ("Consumer Discretionary", "Homebuilders / Housing"),
    "LEN": ("Consumer Discretionary", "Homebuilders / Housing"),
    "NVR": ("Consumer Discretionary", "Homebuilders / Housing"),
    "PHM": ("Consumer Discretionary", "Homebuilders / Housing"),
    "TOL": ("Consumer Discretionary", "Homebuilders / Housing"),
    "WSM": ("Consumer Discretionary", "Homebuilders / Housing"),
    "RH": ("Consumer Discretionary", "Homebuilders / Housing"),
    "W": ("Consumer Discretionary", "Homebuilders / Housing"),
    "ETSY": ("Consumer Discretionary", "Homebuilders / Housing"),
    # Consumer Discretionary - Restaurants / Leisure
    "MCD": ("Consumer Discretionary", "Restaurants / Leisure"),
    "SBUX": ("Consumer Discretionary", "Restaurants / Leisure"),
    "CMG": ("Consumer Discretionary", "Restaurants / Leisure"),
    "YUM": ("Consumer Discretionary", "Restaurants / Leisure"),
    "DPZ": ("Consumer Discretionary", "Restaurants / Leisure"),
    "DRI": ("Consumer Discretionary", "Restaurants / Leisure"),
    "MAR": ("Consumer Discretionary", "Restaurants / Leisure"),
    "HLT": ("Consumer Discretionary", "Restaurants / Leisure"),
    "RCL": ("Consumer Discretionary", "Restaurants / Leisure"),
    "NCLH": ("Consumer Discretionary", "Restaurants / Leisure"),
    "CCL": ("Consumer Discretionary", "Restaurants / Leisure"),
    "LVS": ("Consumer Discretionary", "Restaurants / Leisure"),
    "WYNN": ("Consumer Discretionary", "Restaurants / Leisure"),
    "BKNG": ("Consumer Discretionary", "Restaurants / Leisure"),
    # Consumer Discretionary - Autos / Transport
    "TSLA": ("Consumer Discretionary", "Autos / Transport"),
    "F": ("Consumer Discretionary", "Autos / Transport"),
    "GM": ("Consumer Discretionary", "Autos / Transport"),
    "RIVN": ("Consumer Discretionary", "Autos / Transport"),
    "FDX": ("Consumer Discretionary", "Autos / Transport"),
    "UPS": ("Consumer Discretionary", "Autos / Transport"),
    "JBHT": ("Consumer Discretionary", "Autos / Transport"),
    "XPO": ("Consumer Discretionary", "Autos / Transport"),
    "ODFL": ("Consumer Discretionary", "Autos / Transport"),
    "DAL": ("Consumer Discretionary", "Autos / Transport"),
    "UAL": ("Consumer Discretionary", "Autos / Transport"),
    "LUV": ("Consumer Discretionary", "Autos / Transport"),
    # =================================================================
    # Industrials
    # =================================================================
    # Industrials - Aerospace / Defense
    "BA": ("Industrials", "Aerospace / Defense"),
    "LMT": ("Industrials", "Aerospace / Defense"),
    "RTX": ("Industrials", "Aerospace / Defense"),
    "NOC": ("Industrials", "Aerospace / Defense"),
    "GD": ("Industrials", "Aerospace / Defense"),
    "LHX": ("Industrials", "Aerospace / Defense"),
    "TDG": ("Industrials", "Aerospace / Defense"),
    "HWM": ("Industrials", "Aerospace / Defense"),
    "HEI": ("Industrials", "Aerospace / Defense"),
    "AXON": ("Industrials", "Aerospace / Defense"),
    # Industrials - Machinery / Equipment
    "HON": ("Industrials", "Machinery / Equipment"),
    "CAT": ("Industrials", "Machinery / Equipment"),
    "DE": ("Industrials", "Machinery / Equipment"),
    "EMR": ("Industrials", "Machinery / Equipment"),
    "ETN": ("Industrials", "Machinery / Equipment"),
    "ROK": ("Industrials", "Machinery / Equipment"),
    "AME": ("Industrials", "Machinery / Equipment"),
    "PH": ("Industrials", "Machinery / Equipment"),
    "CMI": ("Industrials", "Machinery / Equipment"),
    "DOV": ("Industrials", "Machinery / Equipment"),
    "IR": ("Industrials", "Machinery / Equipment"),
    # Industrials - Engineering / Construction
    "GE": ("Industrials", "Engineering / Construction"),
    "GEV": ("Industrials", "Engineering / Construction"),
    "JCI": ("Industrials", "Engineering / Construction"),
    "CARR": ("Industrials", "Engineering / Construction"),
    "OTIS": ("Industrials", "Engineering / Construction"),
    "TT": ("Industrials", "Engineering / Construction"),
    "WM": ("Industrials", "Engineering / Construction"),
    "RSG": ("Industrials", "Engineering / Construction"),
    "VRSK": ("Industrials", "Engineering / Construction"),
    "PWR": ("Industrials", "Engineering / Construction"),
    # Industrials - Business Services
    "UNP": ("Industrials", "Business Services"),
    "CSX": ("Industrials", "Business Services"),
    "NSC": ("Industrials", "Business Services"),
    "WAB": ("Industrials", "Business Services"),
    "GWW": ("Industrials", "Business Services"),
    "FAST": ("Industrials", "Business Services"),
    "CTAS": ("Industrials", "Business Services"),
    "CPRT": ("Industrials", "Business Services"),
    "POOL": ("Industrials", "Business Services"),
    "UBER": ("Industrials", "Business Services"),
    "ABNB": ("Industrials", "Business Services"),
    "DASH": ("Industrials", "Business Services"),
    # =================================================================
    # Communication Services
    # =================================================================
    # Communication Services - Interactive Media
    "GOOGL": ("Communication Services", "Interactive Media"),
    "META": ("Communication Services", "Interactive Media"),
    "SNAP": ("Communication Services", "Interactive Media"),
    "PINS": ("Communication Services", "Interactive Media"),
    "MTCH": ("Communication Services", "Interactive Media"),
    "ZG": ("Communication Services", "Interactive Media"),
    "TTD": ("Communication Services", "Interactive Media"),
    "ROKU": ("Communication Services", "Interactive Media"),
    # Communication Services - Media / Entertainment
    "DIS": ("Communication Services", "Media / Entertainment"),
    "NFLX": ("Communication Services", "Media / Entertainment"),
    "CMCSA": ("Communication Services", "Media / Entertainment"),
    "WBD": ("Communication Services", "Media / Entertainment"),
    "FOX": ("Communication Services", "Media / Entertainment"),
    "LYV": ("Communication Services", "Media / Entertainment"),
    "RBLX": ("Communication Services", "Media / Entertainment"),
    "EA": ("Communication Services", "Media / Entertainment"),
    "TTWO": ("Communication Services", "Media / Entertainment"),
    # Communication Services - Telecom
    "T": ("Communication Services", "Telecom"),
    "VZ": ("Communication Services", "Telecom"),
    "TMUS": ("Communication Services", "Telecom"),
    "LUMN": ("Communication Services", "Telecom"),
    # =================================================================
    # Consumer Staples
    # =================================================================
    # Consumer Staples - Food / Beverage
    "PEP": ("Consumer Staples", "Food / Beverage"),
    "KO": ("Consumer Staples", "Food / Beverage"),
    "MDLZ": ("Consumer Staples", "Food / Beverage"),
    "GIS": ("Consumer Staples", "Food / Beverage"),
    "KHC": ("Consumer Staples", "Food / Beverage"),
    "HSY": ("Consumer Staples", "Food / Beverage"),
    "SJM": ("Consumer Staples", "Food / Beverage"),
    "ADM": ("Consumer Staples", "Food / Beverage"),
    "BG": ("Consumer Staples", "Food / Beverage"),
    "STZ": ("Consumer Staples", "Food / Beverage"),
    "SAM": ("Consumer Staples", "Food / Beverage"),
    "MNST": ("Consumer Staples", "Food / Beverage"),
    # Consumer Staples - Household / Personal
    "PG": ("Consumer Staples", "Household / Personal"),
    "CL": ("Consumer Staples", "Household / Personal"),
    "EL": ("Consumer Staples", "Household / Personal"),
    "CLX": ("Consumer Staples", "Household / Personal"),
    "CHD": ("Consumer Staples", "Household / Personal"),
    "KMB": ("Consumer Staples", "Household / Personal"),
    "KVUE": ("Consumer Staples", "Household / Personal"),
    "SWK": ("Consumer Staples", "Household / Personal"),
    # Consumer Staples - Retail Staples
    "WMT": ("Consumer Staples", "Retail Staples"),
    "COST": ("Consumer Staples", "Retail Staples"),
    "KR": ("Consumer Staples", "Retail Staples"),
    "SYY": ("Consumer Staples", "Retail Staples"),
    "USFD": ("Consumer Staples", "Retail Staples"),
    "PM": ("Consumer Staples", "Retail Staples"),
    "MO": ("Consumer Staples", "Retail Staples"),
    # =================================================================
    # Energy
    # =================================================================
    # Energy - E&P
    "XOM": ("Energy", "E&P"),
    "CVX": ("Energy", "E&P"),
    "COP": ("Energy", "E&P"),
    "EOG": ("Energy", "E&P"),
    "DVN": ("Energy", "E&P"),
    "FANG": ("Energy", "E&P"),
    "OVV": ("Energy", "E&P"),
    "EQT": ("Energy", "E&P"),
    "AR": ("Energy", "E&P"),
    # Energy - Oilfield Services
    "SLB": ("Energy", "Oilfield Services"),
    "HAL": ("Energy", "Oilfield Services"),
    "BKR": ("Energy", "Oilfield Services"),
    "FTI": ("Energy", "Oilfield Services"),
    "NOV": ("Energy", "Oilfield Services"),
    "WFRD": ("Energy", "Oilfield Services"),
    "HP": ("Energy", "Oilfield Services"),
    # Energy - Integrated / Midstream
    "WMB": ("Energy", "Integrated / Midstream"),
    "KMI": ("Energy", "Integrated / Midstream"),
    "OKE": ("Energy", "Integrated / Midstream"),
    "ET": ("Energy", "Integrated / Midstream"),
    "TRGP": ("Energy", "Integrated / Midstream"),
    "PSX": ("Energy", "Integrated / Midstream"),
    "VLO": ("Energy", "Integrated / Midstream"),
    "MPC": ("Energy", "Integrated / Midstream"),
    "LNG": ("Energy", "Integrated / Midstream"),
    # =================================================================
    # Materials
    # =================================================================
    # Materials - Chemicals
    "LIN": ("Materials", "Chemicals"),
    "APD": ("Materials", "Chemicals"),
    "SHW": ("Materials", "Chemicals"),
    "ECL": ("Materials", "Chemicals"),
    "DD": ("Materials", "Chemicals"),
    "DOW": ("Materials", "Chemicals"),
    "PPG": ("Materials", "Chemicals"),
    "CE": ("Materials", "Chemicals"),
    "ALB": ("Materials", "Chemicals"),
    "FMC": ("Materials", "Chemicals"),
    # Materials - Metals / Mining
    "NEM": ("Materials", "Metals / Mining"),
    "FCX": ("Materials", "Metals / Mining"),
    "NUE": ("Materials", "Metals / Mining"),
    "STLD": ("Materials", "Metals / Mining"),
    "CLF": ("Materials", "Metals / Mining"),
    "AA": ("Materials", "Metals / Mining"),
    "RS": ("Materials", "Metals / Mining"),
    # Materials - Packaging / Specialty
    "BALL": ("Materials", "Packaging / Specialty"),
    "PKG": ("Materials", "Packaging / Specialty"),
    "AVY": ("Materials", "Packaging / Specialty"),
    "IP": ("Materials", "Packaging / Specialty"),
    "AMCR": ("Materials", "Packaging / Specialty"),
    "SEE": ("Materials", "Packaging / Specialty"),
    # =================================================================
    # Utilities
    # =================================================================
    # Utilities - Electric
    "NEE": ("Utilities", "Electric"),
    "DUK": ("Utilities", "Electric"),
    "SO": ("Utilities", "Electric"),
    "D": ("Utilities", "Electric"),
    "AEP": ("Utilities", "Electric"),
    "EXC": ("Utilities", "Electric"),
    "SRE": ("Utilities", "Electric"),
    "XEL": ("Utilities", "Electric"),
    "WEC": ("Utilities", "Electric"),
    "ED": ("Utilities", "Electric"),
    # Utilities - Gas / Multi / Water
    "PCG": ("Utilities", "Gas / Multi / Water"),
    "EIX": ("Utilities", "Gas / Multi / Water"),
    "CMS": ("Utilities", "Gas / Multi / Water"),
    "DTE": ("Utilities", "Gas / Multi / Water"),
    "AES": ("Utilities", "Gas / Multi / Water"),
    "AWK": ("Utilities", "Gas / Multi / Water"),
    "WTRG": ("Utilities", "Gas / Multi / Water"),
    "NI": ("Utilities", "Gas / Multi / Water"),
    # =================================================================
    # Real Estate
    # =================================================================
    # Real Estate - REITs
    "PLD": ("Real Estate", "REITs"),
    "AMT": ("Real Estate", "REITs"),
    "EQIX": ("Real Estate", "REITs"),
    "PSA": ("Real Estate", "REITs"),
    "O": ("Real Estate", "REITs"),
    "SPG": ("Real Estate", "REITs"),
    "WELL": ("Real Estate", "REITs"),
    "DLR": ("Real Estate", "REITs"),
    "AVB": ("Real Estate", "REITs"),
    "EQR": ("Real Estate", "REITs"),
    "VICI": ("Real Estate", "REITs"),
    "INVH": ("Real Estate", "REITs"),
    "ARE": ("Real Estate", "REITs"),
    "MAA": ("Real Estate", "REITs"),
    "UDR": ("Real Estate", "REITs"),
    "CPT": ("Real Estate", "REITs"),
    "ESS": ("Real Estate", "REITs"),
    "KIM": ("Real Estate", "REITs"),
    "IRM": ("Real Estate", "REITs"),
    "CCI": ("Real Estate", "REITs"),
    "EXR": ("Real Estate", "REITs"),
    "DOC": ("Real Estate", "REITs"),
}

# Reverse lookup: sector -> list of subsectors
SECTOR_SUBSECTORS: dict[str, list[str]] = {}
for _ticker, (_sector, _subsector) in SUBSECTOR_OVERRIDES.items():
    if _sector not in SECTOR_SUBSECTORS:
        SECTOR_SUBSECTORS[_sector] = []
    if _subsector not in SECTOR_SUBSECTORS[_sector]:
        SECTOR_SUBSECTORS[_sector].append(_subsector)


def classify_ticker(
    ticker: str, yf_sector: str = "", yf_industry: str = "",
) -> tuple[str, str]:
    """
    Classify a ticker into (sector, subsector).

    Priority:
    1. Manual override (SUBSECTOR_OVERRIDES)
    2. yfinance sector/industry (if provided)
    3. ("Unknown", "Unknown")
    """
    if ticker in SUBSECTOR_OVERRIDES:
        return SUBSECTOR_OVERRIDES[ticker]

    if yf_sector:
        # Use yfinance sector as sector, industry as subsector
        subsector = yf_industry if yf_industry else yf_sector
        return (yf_sector, subsector)

    return ("Unknown", "Unknown")


def all_tickers_in_universe() -> list[str]:
    """Return all tickers in the manual classification universe."""
    return list(SUBSECTOR_OVERRIDES.keys())


def tickers_by_sector(sector: str) -> list[str]:
    """Return all tickers in a given sector."""
    return [t for t, (s, _) in SUBSECTOR_OVERRIDES.items() if s == sector]


def tickers_by_subsector(subsector: str) -> list[str]:
    """Return all tickers in a given subsector."""
    return [
        t for t, (_, ss) in SUBSECTOR_OVERRIDES.items() if ss == subsector
    ]
