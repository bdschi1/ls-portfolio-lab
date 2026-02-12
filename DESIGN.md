# LS Portfolio Lab — Design Document

## Long/Short Equity Portfolio Risk Workbench

**Purpose:** A portfolio risk management and analytics workbench for long/short equity PMs.
Not an alpha generator — a **risk cockpit** + **paper P&L tracker** + **what-if simulator**.

Given a portfolio (uploaded or generated), calculate and display all risk/return metrics.
When a trade is proposed, show exactly how every metric changes. In paper-portfolio mode,
track all changes over time and produce PM performance analytics.

---

## 1. CORE CONCEPTS

### 1.1 What This App Does

1. **Holds a portfolio** — up to 80 positions, long and short, with weights/shares/notional
2. **Calculates everything** — Sharpe, vol, beta, idiosyncratic risk, RSI, correlation, factor exposures, drawdown, etc.
3. **Simulates trades** — introduce/add/reduce/exit positions → see all metrics change in real-time
4. **Tracks history** (paper mode) — every trade timestamped, P&L tracked, PM performance scored
5. **Sources all data** — daily prices, fundamentals, sector classifications from free/reliable APIs

### 1.2 What This App Does NOT Do

- Does not pick stocks or generate alpha signals
- Does not execute real trades
- Does not require real-time data (EOD is fine)
- Does not manage actual capital

---

## 2. ARCHITECTURE

```
ls-portfolio-lab/
├── README.md
├── DESIGN.md                          # This file
├── pyproject.toml                     # Python 3.12+, Polars, Ruff
├── .env.example                       # API keys template
│
├── app/                               # Streamlit application
│   ├── __init__.py
│   ├── main.py                        # App entry point, layout orchestration
│   ├── pages/
│   │   ├── portfolio_view.py          # Main portfolio + metrics dashboard
│   │   ├── trade_simulator.py         # What-if trade entry + impact preview
│   │   ├── paper_portfolio.py         # History mode: tracked changes + P&L
│   │   └── pm_scorecard.py            # PM performance analytics
│   ├── components/
│   │   ├── portfolio_table.py         # Editable portfolio grid
│   │   ├── metrics_panel.py           # Right-side metrics display
│   │   ├── trade_input.py             # Trade entry form (batch up to 10)
│   │   ├── impact_diff.py             # Before/after metrics comparison
│   │   ├── chart_gallery.py           # Plotly visualizations
│   │   └── settings_panel.py          # User customization controls
│   └── state/
│       ├── session.py                 # Streamlit session state management
│       └── persistence.py             # Save/load portfolio snapshots
│
├── core/                              # Pure computation — no UI dependencies
│   ├── __init__.py
│   ├── portfolio.py                   # Portfolio data model (Pydantic + Polars)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── return_metrics.py          # Sharpe, Sortino, Calmar, information ratio
│   │   ├── risk_metrics.py            # Vol, beta, idiosyncratic vol, tracking error
│   │   ├── exposure_metrics.py        # Gross, net, sector, factor exposures
│   │   ├── drawdown_metrics.py        # Max DD, current DD, time to recovery
│   │   ├── technical_metrics.py       # RSI, momentum measures
│   │   ├── correlation_metrics.py     # Pairwise, portfolio avg, crowding score
│   │   └── pm_performance.py          # Hit rate, slugging %, sector attribution
│   ├── factor_model.py               # Beta decomposition, factor regression
│   ├── trade_impact.py               # Simulate adding/removing positions
│   └── mock_portfolio.py             # Generate the base hypothetical portfolio
│
├── data/                              # Data acquisition and storage
│   ├── __init__.py
│   ├── provider.py                    # Abstract data provider interface
│   ├── yahoo_provider.py             # yfinance-based implementation
│   ├── cache.py                       # Local SQLite cache for price/fundamental data
│   ├── sector_map.py                 # GICS sector/subsector classification
│   ├── universe.py                    # Ticker universe for mock portfolio generation
│   └── ingest.py                      # Excel/PDF portfolio import
│
├── history/                           # Paper portfolio tracking
│   ├── __init__.py
│   ├── trade_log.py                   # Append-only trade journal
│   ├── snapshot.py                    # Daily portfolio state snapshots
│   └── performance.py                 # Time-weighted return calculation
│
├── tests/
│   ├── test_portfolio.py
│   ├── test_metrics/
│   │   ├── test_return_metrics.py
│   │   ├── test_risk_metrics.py
│   │   ├── test_exposure_metrics.py
│   │   ├── test_drawdown_metrics.py
│   │   ├── test_technical_metrics.py
│   │   ├── test_correlation_metrics.py
│   │   └── test_pm_performance.py
│   ├── test_factor_model.py
│   ├── test_trade_impact.py
│   ├── test_mock_portfolio.py
│   ├── test_data_provider.py
│   └── test_history.py
│
└── data_cache/                        # .gitignored — local SQLite + parquet files
    ├── prices.db
    └── snapshots/
```

---

## 3. DATA LAYER

### 3.1 Data Source: yfinance (Primary)

**Why yfinance:**
- Free, no API key required for basic usage
- Daily OHLCV, market cap, sector/industry, beta, shares outstanding
- Sufficient for EOD workflow (not real-time)
- Well-maintained, widely used

**Fallback consideration:** If yfinance rate-limits become an issue, can add
`openbb` or `financialdatapy` as secondary providers behind the same interface.

### 3.2 Data We Need Per Ticker

| Data Point | Source | Frequency |
|-----------|--------|-----------|
| Daily adjusted close | yfinance `.history()` | Daily (cached) |
| Daily OHLCV | yfinance `.history()` | Daily (cached) |
| Market cap | yfinance `.info` | Weekly refresh |
| Sector / Industry (GICS-like) | yfinance `.info` + manual mapping | Static + overrides |
| Beta (vs SPY) | Calculated from returns | On demand |
| Shares outstanding | yfinance `.info` | Weekly refresh |
| Short interest (if available) | yfinance `.info` | When available |
| Dividend yield | yfinance `.info` | Weekly refresh |

### 3.3 Benchmark Data

- **SPY** — market beta reference
- **Sector ETFs** — XLV (healthcare), XLK (tech), XLRE (real estate), etc.
- **Risk-free rate** — ^IRX (13-week T-bill) or ^TNX (10Y) via yfinance

### 3.4 Cache Strategy

SQLite database in `data_cache/prices.db`:
- `daily_prices` table: ticker, date, open, high, low, close, adj_close, volume
- `ticker_info` table: ticker, market_cap, sector, industry, beta, last_updated
- On app startup: check staleness, fetch only missing/stale data
- Max lookback: 3 years of daily data per ticker (sufficient for all calculations)
- Cache invalidation: daily for prices, weekly for fundamentals

### 3.5 Portfolio Import

**Excel (.xlsx/.csv):**
```
| Ticker | Side   | Shares | Entry Price | Entry Date  | Sector Override |
|--------|--------|--------|-------------|-------------|-----------------|
| AMZN   | LONG   | 500    | 185.50      | 2024-06-15  |                 |
| ISRG   | LONG   | 200    | 410.00      | 2024-07-01  |                 |
| HCA    | SHORT  | 300    | 320.00      | 2024-08-01  |                 |
```

Minimum required columns: Ticker, Side (LONG/SHORT), Shares or Weight.
Optional: Entry Price, Entry Date, Sector Override.
If no entry price → use current price. If no entry date → use today.

**PDF:** Extract tables via `pdfplumber` (pattern from existing repos). Best-effort
parsing with user confirmation before loading.

---

## 4. PORTFOLIO DATA MODEL

```python
from pydantic import BaseModel, Field
from datetime import date
from typing import Literal
import polars as pl

class Position(BaseModel):
    ticker: str
    side: Literal["LONG", "SHORT"]
    shares: float
    entry_price: float
    entry_date: date
    current_price: float = 0.0
    sector: str = ""
    subsector: str = ""
    market_cap: float = 0.0

    @property
    def notional(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        direction = 1 if self.side == "LONG" else -1
        return direction * self.shares * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price * (
            1 if self.side == "LONG" else -1
        )


class Portfolio(BaseModel):
    name: str = "Untitled"
    positions: list[Position] = Field(default_factory=list, max_length=80)
    benchmark: str = "SPY"
    inception_date: date = Field(default_factory=date.today)
    nav: float = 10_000_000.0  # $10M notional for mock

    @property
    def long_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "LONG"]

    @property
    def short_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "SHORT"]

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.notional) for p in self.positions)

    @property
    def net_exposure(self) -> float:
        return sum(
            p.notional * (1 if p.side == "LONG" else -1)
            for p in self.positions
        )

    @property
    def long_count(self) -> int:
        return len(self.long_positions)

    @property
    def short_count(self) -> int:
        return len(self.short_positions)
```

The heavy lifting (returns matrices, covariance calculations, factor regressions)
happens in Polars DataFrames, not in the Pydantic model. The model is for
validation and serialization. Metrics modules take a `Portfolio` + price data
and return results.

---

## 5. METRICS — COMPREHENSIVE CATALOG

All metrics are calculated in `core/metrics/`. Each module is pure functions
that take Polars DataFrames and return results. No side effects.

### 5.1 Position-Level Metrics

| Metric | Formula / Method | Module |
|--------|-----------------|--------|
| **Annualized Volatility** | σ = std(daily returns) × √252 | `risk_metrics.py` |
| **Beta (vs SPY)** | cov(Ri, Rm) / var(Rm), rolling or full-period | `risk_metrics.py` |
| **Idiosyncratic Volatility** | σ_idio = std(residuals from CAPM regression) | `risk_metrics.py` |
| **Sharpe Ratio** | (annualized return - Rf) / annualized vol | `return_metrics.py` |
| **Sortino Ratio** | (annualized return - Rf) / downside deviation | `return_metrics.py` |
| **RSI (14-day default)** | 100 - (100 / (1 + avg_gain/avg_loss)) | `technical_metrics.py` |
| **RSI (configurable period)** | Same formula, user-selectable lookback | `technical_metrics.py` |
| **Max Drawdown** | max peak-to-trough decline | `drawdown_metrics.py` |
| **Current Drawdown** | current distance from peak | `drawdown_metrics.py` |
| **Average Daily Volume** | mean(volume) over lookback | `risk_metrics.py` |
| **Days to Liquidate** | position_shares / avg_daily_volume | `risk_metrics.py` |
| **Return (various periods)** | 1D, 1W, 1M, 3M, 6M, YTD, 1Y, custom | `return_metrics.py` |
| **P&L ($, %)** | current vs entry, or vs any reference date | `return_metrics.py` |
| **Contribution to Portfolio Vol** | marginal contribution via covariance | `risk_metrics.py` |
| **Contribution to Portfolio Beta** | weight × position_beta | `risk_metrics.py` |

### 5.2 Portfolio-Level Metrics

| Metric | Formula / Method | Module |
|--------|-----------------|--------|
| **Portfolio Sharpe** | portfolio return / portfolio vol (annualized) | `return_metrics.py` |
| **Portfolio Sortino** | portfolio return / portfolio downside dev | `return_metrics.py` |
| **Calmar Ratio** | annualized return / max drawdown | `return_metrics.py` |
| **Information Ratio** | alpha / tracking error (vs benchmark) | `return_metrics.py` |
| **Portfolio Volatility** | √(w'Σw), annualized | `risk_metrics.py` |
| **Portfolio Beta** | Σ(wi × βi), net of longs and shorts | `risk_metrics.py` |
| **Net Beta Exposure** | net_notional / gross_notional × portfolio_beta | `exposure_metrics.py` |
| **Gross Exposure** | Σ|notional_i| / NAV | `exposure_metrics.py` |
| **Net Exposure** | (long_notional - short_notional) / NAV | `exposure_metrics.py` |
| **Long/Short Ratio** | long_notional / short_notional | `exposure_metrics.py` |
| **Number of Positions** | total, long, short | `exposure_metrics.py` |
| **Sector Net Exposure** | per GICS sector: Σ(long_weight) - Σ(short_weight) | `exposure_metrics.py` |
| **Subsector Net Exposure** | per subsector (e.g., HMOs vs MedTech) | `exposure_metrics.py` |
| **Tracking Error** | std(portfolio_return - benchmark_return) × √252 | `risk_metrics.py` |
| **Portfolio Max Drawdown** | max peak-to-trough on portfolio NAV curve | `drawdown_metrics.py` |
| **Current Portfolio Drawdown** | current NAV vs peak NAV | `drawdown_metrics.py` |
| **Drawdown Duration** | days from peak to current (if in drawdown) | `drawdown_metrics.py` |
| **Average Pairwise Correlation** | mean of all position-pair correlations | `correlation_metrics.py` |
| **Long-Short Correlation** | correlation between long book and short book | `correlation_metrics.py` |
| **Concentration (HHI)** | Σ(weight_i²) — Herfindahl index | `exposure_metrics.py` |
| **Top 5/10 Concentration** | weight of top N positions | `exposure_metrics.py` |
| **Factor Exposures** | from multi-factor regression (market, size, value, momentum) | `factor_model.py` |
| **Residual (Idiosyncratic) Risk** | portfolio-level unexplained variance | `factor_model.py` |
| **% Systematic vs Idiosyncratic** | R² from factor model | `factor_model.py` |
| **VaR (95%, 99%)** | historical or parametric | `risk_metrics.py` |
| **CVaR / Expected Shortfall** | mean of returns below VaR threshold | `risk_metrics.py` |

### 5.3 Technical / Signal Metrics (Per Position)

| Metric | Method | Module |
|--------|--------|--------|
| **RSI** | Wilder's smoothed, configurable period (default 14) | `technical_metrics.py` |
| **RSI Divergence** | price making new high/low while RSI isn't | `technical_metrics.py` |
| **Price vs 50/200 SMA** | % above/below moving average | `technical_metrics.py` |
| **52-Week High/Low %** | distance from 52-week extremes | `technical_metrics.py` |
| **Momentum (various)** | 1M, 3M, 6M, 12M total returns | `technical_metrics.py` |

### 5.4 PM Performance Metrics (Paper Portfolio Mode Only)

| Metric | Formula / Method | Module |
|--------|-----------------|--------|
| **Hit Rate** | # winning trades / # total closed trades | `pm_performance.py` |
| **Slugging Percentage** | avg_win_size / avg_loss_size | `pm_performance.py` |
| **Win/Loss Ratio** | total_wins_$ / total_losses_$ | `pm_performance.py` |
| **Average Holding Period** | mean days from entry to exit | `pm_performance.py` |
| **Best/Worst Trade** | max/min P&L on closed positions | `pm_performance.py` |
| **Sector Hit Rate** | hit rate broken down by sector | `pm_performance.py` |
| **Sector Slugging** | slugging % broken down by sector | `pm_performance.py` |
| **Long Hit Rate** | hit rate on long book only | `pm_performance.py` |
| **Short Hit Rate** | hit rate on short book only | `pm_performance.py` |
| **Long Slugging** | slugging on long book | `pm_performance.py` |
| **Short Slugging** | slugging on short book | `pm_performance.py` |
| **Alpha (vs benchmark)** | Jensen's alpha from CAPM regression | `pm_performance.py` |
| **Idiosyncratic Return** | return not explained by factor model | `pm_performance.py` |
| **Portfolio Turnover** | total traded notional / avg NAV over period | `pm_performance.py` |
| **Drawdown Response** | how quickly PM reduces exposure after drawdown | `pm_performance.py` |
| **Drawdown Recovery Time** | days from trough to previous peak | `pm_performance.py` |
| **Max Drawdown Depth** | deepest peak-to-trough on paper portfolio | `pm_performance.py` |
| **Time-Weighted Return** | geometric chain of sub-period returns | `pm_performance.py` |
| **Risk-Adjusted Return** | return / realized vol | `pm_performance.py` |
| **Batting Avg by Market Cap** | hit rate bucketed by position market cap | `pm_performance.py` |
| **Batting Avg by Holding Period** | hit rate bucketed by hold duration | `pm_performance.py` |

---

## 6. TRADE IMPACT SIMULATOR

The core "what-if" engine. Takes the current portfolio + a proposed set of trades
(up to 10 at a time) and shows a full before/after comparison.

### 6.1 Trade Input Format

```python
class ProposedTrade(BaseModel):
    ticker: str
    action: Literal["BUY", "SELL", "SHORT", "COVER", "ADD", "REDUCE", "EXIT"]
    shares: float | None = None       # if None, use dollar_amount
    dollar_amount: float | None = None # alternative sizing
    # at least one of shares or dollar_amount required

class TradeBasket(BaseModel):
    trades: list[ProposedTrade] = Field(max_length=10)
    description: str = ""  # optional note
```

### 6.2 Impact Calculation Flow

```
Current Portfolio State (all metrics calculated)
        ↓
Apply proposed trades to create "hypothetical portfolio"
        ↓
Recalculate ALL metrics on hypothetical portfolio
        ↓
Diff: show Δ for every metric
        ↓
User decides: ACCEPT (apply to portfolio) or REJECT (discard)
```

### 6.3 Impact Display

For each metric, show:
- **Before** value
- **After** value
- **Change** (absolute and percentage)
- **Color coding**: green = improvement, red = degradation, yellow = neutral/mixed

Key impact highlights:
- Net beta exposure change
- Sector concentration changes (flag if any sector goes above user-defined limit)
- Portfolio vol change
- Gross/net exposure change
- Correlation impact (does the new position diversify or concentrate?)
- Marginal contribution to risk of new position(s)
- Days to liquidate for any illiquid additions

---

## 7. MOCK PORTFOLIO GENERATOR

### 7.1 Constraints (Per Spec)

- **20 longs, 25 shorts** (45 total positions)
- **Net beta exposure: +6%** (slightly net long bias)
- **No sector/subsector net exposure > 50%**
- **Liquid, mid-to-large cap names**
- **Sectors:** Healthcare (HMOs, hospitals, MedTech, lifesci tools), Tech, Real Estate/REITs

### 7.2 Universe Definition

```python
MOCK_UNIVERSE = {
    "Healthcare - HMOs/Managed Care": [
        "UNH", "HUM", "ELV", "CI", "CNC", "MOH",
    ],
    "Healthcare - Hospitals": [
        "HCA", "THC", "UHS", "CYH", "SEM",
    ],
    "Healthcare - MedTech": [
        "ISRG", "SYK", "MDT", "BSX", "EW", "ZBH", "HOLX", "ALGN",
        "DXCM", "PODD", "RVMD",
    ],
    "Healthcare - Life Sci Tools": [
        "TMO", "DHR", "A", "ILMN", "MTD", "WAT", "PKI", "BIO",
        "TXG", "AZTA", "CRL",
    ],
    "Healthcare - Biotech/Pharma": [
        "AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA", "BMRN", "ALNY",
        "SGEN", "INCY",
    ],
    "Technology - Software": [
        "MSFT", "CRM", "NOW", "ADBE", "SNPS", "CDNS", "PANW", "FTNT",
        "WDAY", "HUBS", "DDOG", "NET", "ZS", "CRWD", "MDB",
    ],
    "Technology - Semis": [
        "NVDA", "AMD", "AVGO", "MRVL", "KLAC", "LRCX", "AMAT", "ON",
        "TER", "ENTG",
    ],
    "Technology - Hardware/Internet": [
        "AAPL", "META", "GOOGL", "AMZN", "NFLX", "SHOP", "UBER",
        "ABNB", "DASH", "SNAP",
    ],
    "Real Estate / REITs": [
        "PLD", "AMT", "EQIX", "PSA", "O", "SPG", "WELL", "DLR",
        "AVB", "EQR", "VICI", "INVH",
    ],
}
```

### 7.3 Generation Algorithm

1. Randomly select from universe ensuring sector/subsector diversity
2. Fetch current betas for all selected names
3. Assign initial weights using optimization:
   - Objective: net beta = +0.06 (of NAV)
   - Constraint: no sector net > 50% of gross
   - Constraint: no subsector net > 50% of gross
   - Constraint: all names are liquid (avg volume > $5M/day)
   - Constraint: market cap > $2B (mid-cap floor)
4. If constraints not satisfiable, swap names and retry
5. Result: a plausible, diversified L/S equity book

---

## 8. USER CUSTOMIZATION

### 8.1 Timeframe Controls

- **Return calculation period:** 1D, 1W, 1M, 3M, 6M, YTD, 1Y, 2Y, 3Y, Custom range
- **Lookback for vol/beta/correlation:** 60D, 90D, 126D (6M), 252D (1Y), custom
- **RSI period:** 7, 14, 21, custom
- **Moving average periods:** 20, 50, 100, 200, custom
- **Rolling window for metrics:** user-selectable

### 8.2 Risk Parameter Controls

- **Benchmark selection:** SPY (default), QQQ, IWM, sector ETF, custom ticker
- **Risk-free rate:** auto (T-bill), or manual override
- **VaR confidence level:** 95%, 99%, custom
- **Factor model:** CAPM (1-factor), Fama-French 3, FF+Momentum 4, custom
- **Sector classification:** use yfinance default, or manual override per position
- **Exposure limits:** configurable thresholds for alerts (sector max, beta range, etc.)

### 8.3 Display Controls

- **Sort portfolio by:** any column (ticker, weight, P&L, beta, vol, sector, etc.)
- **Filter by:** sector, side (long/short), P&L status, beta range
- **Metrics to display:** user selects which metrics appear in the panel
- **Chart preferences:** which visualizations to show

### 8.4 Portfolio Settings

- **NAV:** configurable starting capital (default $10M)
- **Max positions:** configurable (default 80)
- **Rebalance frequency for paper mode:** daily, weekly, monthly
- **Currency:** USD (single currency for simplicity)

---

## 9. PAPER PORTFOLIO MODE (HISTORY TRACKING)

### 9.1 How It Works

When "Paper Portfolio Mode" is toggled ON:
- Every trade is logged to an append-only trade journal
- Daily snapshots of portfolio state are saved
- All metrics are tracked over time
- Nothing is destructive — full history preserved

### 9.2 Trade Journal Schema

```python
class TradeRecord(BaseModel):
    timestamp: datetime
    ticker: str
    action: str           # BUY, SELL, SHORT, COVER, ADD, REDUCE, EXIT
    shares: float
    price: float           # execution price (current market price at time of trade)
    notional: float
    side_after: str | None # LONG, SHORT, or None (if exited)
    shares_after: float    # total shares in position after trade
    portfolio_nav_after: float
    notes: str = ""
```

### 9.3 Daily Snapshot Schema

```python
class DailySnapshot(BaseModel):
    date: date
    nav: float
    gross_exposure: float
    net_exposure: float
    net_beta: float
    portfolio_vol: float
    portfolio_sharpe: float
    long_count: int
    short_count: int
    positions: list[PositionSnapshot]  # ticker, weight, pnl, beta per position
    sector_exposures: dict[str, float]
```

### 9.4 Performance Analytics Available

From the trade journal + daily snapshots, calculate:

**Trading Stats:**
- Total trades executed
- Trades per week/month
- Average trade size (notional)
- Turnover rate

**P&L Stats:**
- Cumulative P&L ($, %)
- Daily P&L distribution
- Best/worst day
- Consecutive winning/losing days
- P&L by sector
- P&L by long vs short

**Risk Evolution:**
- Rolling Sharpe (30D, 60D, 90D)
- Rolling vol
- Rolling beta
- Exposure drift over time

**PM Scorecard (the important stuff):**
- Hit rate (overall, by sector, by side, by cap bucket)
- **Slugging percentage** (overall, by sector, by side) — *the key metric*
- Expected value per trade (hit_rate × avg_win - miss_rate × avg_loss)
- Drawdown analysis:
  - Number of drawdowns > X%
  - Average depth
  - Average recovery time
  - PM behavior during drawdown (did they cut exposure? add? freeze?)
- Alpha decomposition over paper period
- Sector skill: which sectors does PM add/destroy alpha
- Timing skill: does PM entry timing add value vs random entry

---

## 10. UI LAYOUT

### 10.1 Main Dashboard (Portfolio View)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [Upload] [Generate Mock] [Save] [Load]    [Paper Mode: ON/OFF]       │
│  ─────────────────────────────────────────────────────────────────────  │
│  Settings: Lookback [▼252D] Benchmark [▼SPY] RSI Period [▼14]        │
├────────────────────────────────┬────────────────────────────────────────┤
│                                │                                        │
│   PORTFOLIO TABLE              │   METRICS PANEL                        │
│   ─────────────────────        │   ─────────────────                    │
│   Ticker | Side | Shrs | Wt%  │   Portfolio Summary                    │
│   | Price | P&L% | Beta | Vol │   ├ NAV: $10,245,000                   │
│   | RSI | Sector | Days2Liq   │   ├ Gross: 187%                        │
│                                │   ├ Net: +12%                          │
│   UNH  L  500  4.2%  +3.1%   │   ├ Net Beta: +0.06                    │
│   ISRG L  200  3.8%  -1.2%   │   ├ Portfolio Vol: 14.2%               │
│   HCA  S  300  2.9%  +5.4%   │   ├ Sharpe: 1.34                       │
│   ...                          │   ├ Sortino: 1.81                      │
│                                │   ├ Max DD: -8.3%                      │
│   [20 more rows...]           │   ├ VaR(95%): -1.8%                    │
│                                │                                        │
│                                │   Sector Exposures                     │
│                                │   ├ HC-HMO: +8% / -12% = -4%          │
│                                │   ├ HC-MedTech: +15% / -8% = +7%      │
│                                │   ├ Tech-SW: +12% / -18% = -6%        │
│                                │   ├ ...                                │
│                                │                                        │
│                                │   Risk Decomposition                   │
│                                │   ├ Systematic: 45%                    │
│                                │   ├ Idiosyncratic: 55%                 │
│                                │   ├ Tracking Error: 11.2%              │
│                                │                                        │
├────────────────────────────────┴────────────────────────────────────────┤
│  CHARTS: [Sector Exposure ▼] [Correlation Heatmap ▼] [NAV Curve ▼]   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Plotly Chart Area                             │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Trade Simulator Page

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PROPOSE TRADES (max 10)                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Ticker: [____] Action: [▼BUY] Shares: [____] or $: [____]     │   │
│  │ Ticker: [____] Action: [▼SHORT] Shares: [____] or $: [____]   │   │
│  │ [+ Add another trade]                                           │   │
│  │                                          [Preview Impact]       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  IMPACT PREVIEW                                                         │
│  ┌──────────────────────┬──────────┬──────────┬──────────┐             │
│  │ Metric               │ Before   │ After    │ Δ Change │             │
│  ├──────────────────────┼──────────┼──────────┼──────────┤             │
│  │ Net Beta             │ +0.060   │ +0.072   │ +0.012 ▲│             │
│  │ Portfolio Vol        │ 14.2%    │ 14.8%    │ +0.6% ▲ │             │
│  │ Sharpe (hist.)       │ 1.34     │ 1.29     │ -0.05 ▼ │             │
│  │ HC-HMO Net           │ -4%      │ +2%      │ +6% ▲   │             │
│  │ Gross Exposure       │ 187%     │ 193%     │ +6% ▲   │             │
│  │ Positions            │ 45       │ 47       │ +2      │             │
│  │ Avg Pair Correlation │ 0.31     │ 0.33     │ +0.02   │             │
│  │ VaR(95%)             │ -1.8%    │ -2.0%    │ -0.2% ▼ │             │
│  │ ...                  │          │          │          │             │
│  └──────────────────────┴──────────┴──────────┴──────────┘             │
│                                                                         │
│  ⚠ WARNING: HC-HMO subsector net exposure would reach +48%            │
│                                                                         │
│  [Apply Trades] [Discard]                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 PM Scorecard Page (Paper Mode)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PM SCORECARD — Paper Portfolio Performance                             │
│  Period: 2024-06-15 to 2025-01-15 (214 trading days)                  │
│                                                                         │
│  ┌─────────────────────┐  ┌──────────────────────┐                     │
│  │ HEADLINE NUMBERS    │  │ RISK PROFILE          │                     │
│  │ TWR: +14.2%         │  │ Realized Vol: 12.8%   │                     │
│  │ Sharpe: 1.42        │  │ Max DD: -6.1%         │                     │
│  │ Alpha: +5.8%        │  │ DD Recovery: 18 days  │                     │
│  │ Hit Rate: 58%       │  │ Avg DD: -3.2%         │                     │
│  │ Slugging: 2.3x  ★  │  │ Calmar: 2.33          │                     │
│  │ EV/Trade: +$18,200  │  │                        │                     │
│  └─────────────────────┘  └──────────────────────┘                     │
│                                                                         │
│  SECTOR SKILL                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sector          │ Hit Rate │ Slugging │ Alpha │ # Trades        │   │
│  │ HC-MedTech      │ 67%      │ 3.1x     │ +8.2% │ 12             │   │
│  │ Tech-SW         │ 55%      │ 2.0x     │ +3.1% │ 18             │   │
│  │ HC-HMO          │ 50%      │ 1.2x     │ -0.5% │ 8              │   │
│  │ REITs           │ 43%      │ 0.8x     │ -2.1% │ 7              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  DRAWDOWN ANALYSIS                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  [NAV curve with drawdown shading]                               │  │
│  │  [Exposure overlay during drawdowns]                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  TRADE LOG (last 20)                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Date     │ Ticker │ Action │ Shares │ Price  │ P&L    │ Notes  │  │
│  │ 01/14    │ ISRG   │ ADD    │ +50    │ 425.20 │        │        │  │
│  │ 01/10    │ SNAP   │ COVER  │ 200    │ 11.50  │ +$800  │ target │  │
│  │ ...                                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. CHARTS / VISUALIZATIONS (Plotly)

All charts rendered via Plotly for interactivity.

1. **Sector Exposure Bar Chart** — stacked long/short by sector, net line overlay
2. **Correlation Heatmap** — all positions pairwise (clustered)
3. **Beta Scatter** — each position: x=beta, y=weight, color=sector, size=notional
4. **NAV Curve** — cumulative portfolio return over time (paper mode)
5. **Drawdown Chart** — underwater plot showing drawdown depth over time
6. **Risk Contribution Waterfall** — which positions contribute most to portfolio vol
7. **P&L Attribution** — sector-level P&L contribution over time
8. **Rolling Metrics** — Sharpe, vol, beta over rolling windows
9. **RSI Dashboard** — all positions RSI in a heatmap or sorted bar chart
10. **Position P&L Scatter** — x=holding period, y=P&L%, color=side, size=notional
11. **Factor Exposure Radar** — market, size, value, momentum factor loadings
12. **Exposure Time Series** — gross/net/beta exposure evolution (paper mode)

---

## 12. TECH STACK

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Language** | Python 3.12+ | Consistent with existing repos |
| **Data Frames** | Polars | Fast, consistent with quant setup preferences |
| **Validation** | Pydantic v2 | Consistent with arena/redflag repos |
| **UI** | Streamlit | Consistent with existing repos, rapid iteration |
| **Charts** | Plotly | Interactive, works well with Streamlit |
| **Data Source** | yfinance | Free, no API key, sufficient for EOD |
| **Cache** | SQLite (via polars or sqlite3) | Simple, file-based, no server needed |
| **Optimization** | scipy.optimize | For mock portfolio weight optimization |
| **Statistics** | numpy, scipy.stats | For regressions, VaR calculations |
| **PDF Import** | pdfplumber | Consistent with existing repos |
| **Excel Import** | polars (read_excel) or openpyxl | Native support |
| **Testing** | pytest + pytest-cov | Consistent with existing repos |
| **Linting** | Ruff (E, F, W, I) | Consistent with existing repos |
| **Persistence** | JSON + Parquet | JSON for config, Parquet for time series |

---

## 13. BUILD ORDER (Implementation Phases)

### Phase 1: Foundation (Data + Portfolio Model)
1. Project scaffolding (pyproject.toml, directory structure, .env.example)
2. `data/yahoo_provider.py` — fetch daily prices, info, sector data
3. `data/cache.py` — SQLite caching layer
4. `data/sector_map.py` — GICS sector/subsector mapping with manual overrides
5. `core/portfolio.py` — Pydantic portfolio model
6. `data/ingest.py` — Excel/CSV import
7. Tests for all above

### Phase 2: Metrics Engine
8. `core/metrics/return_metrics.py` — Sharpe, Sortino, Calmar, IR, period returns
9. `core/metrics/risk_metrics.py` — Vol, beta, idiosyncratic vol, VaR, CVaR
10. `core/metrics/exposure_metrics.py` — Gross, net, sector, concentration
11. `core/metrics/drawdown_metrics.py` — Max DD, current DD, recovery time
12. `core/metrics/technical_metrics.py` — RSI, momentum, SMA proximity
13. `core/metrics/correlation_metrics.py` — Pairwise, avg, long-short
14. `core/factor_model.py` — CAPM + multi-factor regression
15. Tests for all metrics (comprehensive, with known-answer tests)

### Phase 3: Trade Impact + Mock Portfolio
16. `core/trade_impact.py` — Apply hypothetical trades, diff all metrics
17. `core/mock_portfolio.py` — Generate constrained mock portfolio
18. Tests for trade impact and mock generation

### Phase 4: Streamlit UI
19. `app/main.py` — Layout, navigation, session state
20. `app/pages/portfolio_view.py` — Main dashboard
21. `app/components/portfolio_table.py` — Sortable, filterable table
22. `app/components/metrics_panel.py` — Right-side metrics display
23. `app/components/chart_gallery.py` — All Plotly charts
24. `app/components/settings_panel.py` — Customization controls
25. `app/pages/trade_simulator.py` — Trade input + impact preview

### Phase 5: Paper Portfolio Mode
26. `history/trade_log.py` — Append-only trade journal
27. `history/snapshot.py` — Daily snapshots
28. `history/performance.py` — TWR, P&L calculations
29. `core/metrics/pm_performance.py` — Hit rate, slugging, sector skill, DD analysis
30. `app/pages/paper_portfolio.py` — History view
31. `app/pages/pm_scorecard.py` — PM analytics dashboard
32. Tests for all history/performance modules

### Phase 6: Polish
33. PDF import for portfolio files
34. Export functionality (Excel, PDF report)
35. Edge case handling (delistings, stock splits, missing data)
36. Performance optimization (lazy loading, incremental metric updates)
37. Documentation and README

---

## 14. KEY CALCULATIONS — IMPLEMENTATION NOTES

### 14.1 Portfolio Volatility (the right way)

```python
# w = weight vector (n,)
# Σ = covariance matrix of daily returns (n, n)
# annualize by √252

import numpy as np

def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualized portfolio volatility from weight vector and covariance matrix."""
    port_var = weights @ cov_matrix @ weights
    return float(np.sqrt(port_var) * np.sqrt(252))
```

Note: weights for shorts are NEGATIVE. This correctly captures the hedging
benefit (negative covariance contribution) of short positions.

### 14.2 Beta Calculation

```python
def position_beta(
    returns: pl.Series,   # daily returns for position
    market_returns: pl.Series,  # daily returns for SPY
) -> float:
    """OLS beta: cov(Ri, Rm) / var(Rm)."""
    cov = returns.to_numpy() @ market_returns.to_numpy() / len(returns)
    var_m = market_returns.var()
    return float(cov / var_m)
```

Portfolio beta = Σ(wi × βi), where wi is signed (+long, -short).

### 14.3 Idiosyncratic Volatility

```python
def idiosyncratic_vol(
    returns: pl.Series,
    market_returns: pl.Series,
) -> float:
    """Residual vol from CAPM regression."""
    beta = position_beta(returns, market_returns)
    residuals = returns.to_numpy() - beta * market_returns.to_numpy()
    return float(np.std(residuals) * np.sqrt(252))
```

### 14.4 RSI (Wilder's Smoothed)

```python
def rsi(prices: pl.Series, period: int = 14) -> pl.Series:
    """RSI using Wilder's exponential smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    # Wilder's smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm_mean(span=period * 2 - 1, adjust=False)
    avg_loss = loss.ewm_mean(span=period * 2 - 1, adjust=False)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
```

### 14.5 Slugging Percentage

```python
def slugging_pct(closed_trades: list[TradeRecord]) -> float:
    """
    Slugging % = avg_win_magnitude / avg_loss_magnitude

    This is the PM's most important metric. A PM with 45% hit rate
    but 3.0x slugging is far better than 60% hit rate with 1.0x slugging.

    Slugging > 2.0x with hit rate > 45% = elite PM performance.
    """
    wins = [t.pnl for t in closed_trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in closed_trades if t.pnl < 0]

    if not wins or not losses:
        return 0.0

    return (sum(wins) / len(wins)) / (sum(losses) / len(losses))
```

### 14.6 Drawdown Recovery Analysis

```python
def drawdown_analysis(nav_series: pl.Series) -> dict:
    """
    Comprehensive drawdown analysis:
    - Identify all drawdown episodes
    - Measure depth, duration, recovery time
    - Classify PM behavior during drawdown
    """
    peak = nav_series.cum_max()
    drawdown = (nav_series - peak) / peak

    # Find drawdown episodes (contiguous periods below peak)
    # For each episode: start_date, trough_date, recovery_date,
    #                   depth, duration_to_trough, duration_to_recovery
    ...
```

---

## 15. CONFIGURATION FILE

```yaml
# config.yaml — user-editable defaults
portfolio:
  max_positions: 80
  default_nav: 10_000_000
  benchmark: "SPY"

metrics:
  lookback_days: 252
  rsi_period: 14
  var_confidence: 0.95
  risk_free_rate: "auto"  # or manual float like 0.05
  factor_model: "CAPM"    # CAPM, FF3, FF4

alerts:
  max_sector_net_exposure: 0.50
  max_subsector_net_exposure: 0.50
  max_single_position_weight: 0.10
  max_net_beta: 0.30
  min_net_beta: -0.10
  max_gross_exposure: 2.50

display:
  default_return_period: "3M"
  show_charts: ["sector_exposure", "correlation_heatmap", "nav_curve"]
  decimal_places: 2

cache:
  price_staleness_hours: 18  # refetch if older than this
  info_staleness_days: 7
  max_history_years: 3
```

---

## 16. ERROR HANDLING AND EDGE CASES

- **Delisted tickers:** Detect via yfinance (empty history), warn user, keep last known price
- **Stock splits:** Use adjusted close for all return calculations
- **Missing data days:** Forward-fill prices for up to 5 days, then flag
- **Thin trading:** Flag positions where avg volume < $1M/day
- **Corporate actions:** Dividends handled via adjusted close; M&A flagged manually
- **Concurrent positions:** Can't be long AND short same ticker (enforced in model)
- **Zero-division:** Guard all ratios (Sharpe with zero vol, slugging with zero losses, etc.)
- **API failures:** Graceful degradation — show cached data with staleness warning

---

## 17. WHAT THIS IS NOT

To be very explicit about scope boundaries:

- **Not a trading system** — no order management, no broker connectivity
- **Not real-time** — EOD prices only, no streaming
- **Not an optimizer** — doesn't tell you what to buy/sell (except mock generation)
- **Not a backtester** — doesn't replay historical strategies
- **Not multi-asset** — equities only (no options, futures, FX)
- **Not multi-currency** — USD only
- **Not a factor model research tool** — uses factors for decomposition, not discovery

It IS: a **risk workbench** that answers "if I do this trade, what happens to my portfolio's risk profile?" and tracks PM performance when running a paper book.
