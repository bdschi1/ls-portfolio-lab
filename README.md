# LS Portfolio Lab — v1.0

**Long/Short Equity Portfolio Risk Workbench**

Portfolio dashboard for long/short equity portfolio managers. Monitor, stress-test, and track portfolio risk/return profile with institutional-grade analytics.

Not an alpha generator — a **risk management tool** that answers: *"What happens to my risk if I add this trade?"*

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/bdschi1/ls-portfolio-lab.git && cd ls-portfolio-lab
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run
streamlit run app/main.py
```

Open `http://localhost:8501`. Generate a mock portfolio or upload your own (CSV/Excel).

### Optional Provider Install

```bash
pip install -e ".[bloomberg]"    # Bloomberg Professional API
pip install -e ".[ib]"           # Interactive Brokers
pip install -e ".[dev,bloomberg,ib]"  # Everything
```

---

## Pages

### 1. Portfolio Dashboard
- **Top metrics bar:** Portfolio Vol, Net Beta, Sharpe, weighted RSI (long/short books), Gross/Net exposure, Cash %, Time in Drawdown, Quality Score (0-100)
- **Detail grid:** Summary (NAV, L/S ratio, HHI), Risk (Sortino, VaR, CVaR, DSR), Drawdown (max DD, current DD, Calmar, expected DD, time in DD — Bailey & Lopez de Prado), Factors (CAPM/FF3/FF4 alpha, beta, systematic %, factor tilts), Correlation (pairwise, long book, short book, L/S Corr, Idio %)
- **Position table:** Filterable by side/sector, 20+ columns: annualized vol, idiosyncratic vol, beta, RSI, ADV$ (20-day avg dollar volume), alpha 30d/1yr, P&L, weight. A+/A- font controls
- **Charts:** Sector exposure (diverging bar with annotations), beta scatter, risk contribution, RSI heatmap, correlation matrix (with most/least correlated pair tables), NAV curve, drawdown, P&L waterfall, sector P&L (long/short stacked), dispersion, quality score radar, rolling metrics
- **Rebalancer:** SLSQP constrained optimizer targeting net beta and annualized volatility

### 2. Trade Simulator
- Model up to 10 trades per basket (BUY, SHORT, ADD, REDUCE, SELL, COVER, EXIT)
- Supports equities, ETFs, and options with delta adjustment
- Full before/after metric comparison with limit warnings — then apply or discard

### 3. Paper Portfolio
- Toggle Paper Mode ON in the sidebar to start tracking
- Immutable JSONL trade journal — every applied trade logged with timestamp
- Daily NAV snapshots with positions, exposures, and P&L
- NAV curve, exposure evolution, and beta history charts
- Closed trade summary with hit rate and slugging %
- Persists to disk — survives app restarts

### 4. PM Scorecard
- Hit rate, slugging %, expected value per trade
- Long vs. short breakdown with separate hit rate and slugging
- Sector skill table (hit rate, slugging, total P&L per sector)
- NAV curve with drawdown shading, drawdown behavior metrics
- Key takeaways generated from trade patterns

---

## Repository Structure

```
ls-portfolio-lab/
│
├── app/                              # Streamlit application layer
│   ├── __init__.py
│   ├── main.py                       # Entry point — sidebar, navigation, data source selector
│   │
│   ├── pages/                        # One module per page
│   │   ├── __init__.py
│   │   ├── portfolio_view.py         # Portfolio Dashboard — data refresh, metrics, charts
│   │   ├── trade_simulator.py        # Trade Simulator — trade entry, impact preview, apply
│   │   ├── paper_portfolio.py        # Paper Portfolio — trade journal, snapshots, NAV curve
│   │   └── pm_scorecard.py           # PM Scorecard — hit rate, slugging, sector skill
│   │
│   ├── components/                   # Reusable UI components
│   │   ├── __init__.py
│   │   ├── metrics_panel.py          # Top metrics bar, detail grid (HTML), sector exposure chart
│   │   ├── portfolio_table.py        # Position table — 20+ columns, filters, A+/A- font
│   │   └── chart_gallery.py          # 12+ Plotly charts — sector, beta, correlation, P&L, etc.
│   │
│   └── state/                        # Session state & persistence
│       ├── __init__.py
│       ├── session.py                # Session state init — cache, settings, alerts, returns
│       └── persistence.py            # Portfolio save/load (JSON to disk)
│
├── core/                             # Pure business logic (no Streamlit imports)
│   ├── __init__.py
│   ├── portfolio.py                  # Pydantic models — Portfolio, Position, ProposedTrade, TradeBasket
│   ├── mock_portfolio.py             # Mock portfolio generator (~30L/~40S, 11 GICS sectors, $3B NAV)
│   ├── rebalancer.py                 # SLSQP constrained optimizer (net beta, vol targets)
│   ├── trade_impact.py               # Trade simulation engine — apply trades, compute metric diffs
│   ├── factor_model.py               # CAPM, FF3, FF4 regressions (ETF proxies: IWM, IWD, IWF, MTUM)
│   │
│   └── metrics/                      # All analytics — pure functions, Polars DataFrames
│       ├── __init__.py
│       ├── return_metrics.py         # Sharpe, Sortino, Calmar, Deflated Sharpe Ratio (DSR)
│       ├── risk_metrics.py           # Portfolio vol, VaR 95%, CVaR 95%, beta, idio vol, MCR
│       ├── drawdown_analytics.py     # Bailey & Lopez de Prado — E[DD], P(DD≥b), time in DD
│       ├── drawdown_metrics.py       # Empirical drawdown — max DD, current DD, recovery
│       ├── exposure_metrics.py       # Gross/net exposure, net beta, HHI, top-5 concentration
│       ├── correlation_metrics.py    # Pairwise, long book, short book, L/S correlation
│       ├── technical_metrics.py      # RSI (Wilder's), SMA, momentum, 52-week high/low
│       ├── pm_performance.py         # Hit rate, slugging %, EV per trade, sector attribution
│       ├── quality_score.py          # Composite 0-100 score — 6 weighted dimensions (A+ to F)
│       └── attribution.py            # P&L attribution — position, sector, side, factor decomposition
│
├── data/                             # Market data providers & caching
│   ├── __init__.py
│   ├── provider.py                   # Abstract DataProvider base class (4 methods)
│   ├── yahoo_provider.py             # Yahoo Finance — yfinance wrapper, tenacity retry + timeout
│   ├── bloomberg_provider.py         # Bloomberg Professional API — DAPI, blpapi session management
│   ├── ib_provider.py                # Interactive Brokers — ib_insync, TWS/Gateway connection
│   ├── provider_factory.py           # Provider registry, auto-discovery, get_provider_safe()
│   ├── cache.py                      # SQLite cache — 18hr price staleness, 7d info staleness
│   ├── universe.py                   # ~440 Russell 1000 tickers (market cap > $5B)
│   ├── sector_map.py                 # GICS sector/subsector classification + ETF detection
│   └── ingest.py                     # Portfolio parser — CSV, Excel (.xlsx/.xls), PDF extraction
│
├── history/                          # Paper portfolio persistence (append-only)
│   ├── __init__.py
│   ├── trade_log.py                  # JSONL trade journal — immutable, timestamped records
│   ├── snapshot.py                   # Daily snapshots — NAV, positions, sector exposures
│   └── performance.py                # Time-weighted return, PM scorecard generation
│
├── tests/                            # 277 tests (pytest)
│   ├── __init__.py
│   ├── test_portfolio.py             # Portfolio, Position, ProposedTrade, TradeBasket models
│   ├── test_trade_impact.py          # Trade application, simulation, cash tracking
│   ├── test_rebalancer.py            # Optimizer constraints, convergence, side preservation
│   ├── test_mock_portfolio.py        # Mock generation, sector diversity, universe sync
│   ├── test_drawdown_analytics.py    # Bailey & Lopez de Prado analytical formulas
│   ├── test_factor_model.py          # CAPM, FF3, FF4 regression accuracy
│   ├── test_history.py               # Trade log, snapshot store, round-trip persistence
│   │
│   └── test_metrics/                 # Per-module metric tests
│       ├── __init__.py
│       ├── test_return_metrics.py    # Sharpe, Sortino, Calmar, DSR
│       ├── test_risk_metrics.py      # Vol, VaR, CVaR, beta, idio vol
│       ├── test_drawdown_metrics.py  # Empirical drawdown computation
│       ├── test_exposure_metrics.py  # Gross/net, HHI, sector limits
│       ├── test_technical_metrics.py # RSI, momentum, 52-week high/low
│       └── test_pm_performance.py    # Hit rate, slugging, sector skill
│
├── config.yaml                       # Default configuration (cache, metrics, alerts, portfolio)
├── pyproject.toml                    # Project metadata — v1.0.0, dependencies, ruff, pytest
├── requirements.txt                  # Pinned dependencies (pip freeze)
├── Makefile                          # Dev shortcuts — run, test, coverage, lint, fmt, install
├── .env.example                      # Environment variable template (data + AI providers)
├── .gitignore                        # Excludes: venv, .env, data_cache, __pycache__, AI tool dirs
├── LICENSE                           # MIT License (2025-2026 BDS)
├── DESIGN.md                         # Detailed design document
├── REFERENCES.md                     # Academic citations with implemented equations
├── CITATION.cff                      # GitHub citation metadata (v1.0.0, 2026-02-12)
└── CONTRIBUTING.md                   # Contribution guidelines
```

---

## Data Providers

| Provider | Status | Requirements | Data |
|----------|--------|-------------|------|
| **Yahoo Finance** | Default | None (free) | EOD adjusted close, ~18hr delay |
| **Bloomberg** | Optional | Terminal + `pip install blpapi` | Real-time, institutional reference data |
| **Interactive Brokers** | Optional | TWS/Gateway + `pip install ib_insync` | Real-time quotes, execution-ready |

Switch providers in the sidebar under **Data Source**. The system auto-detects which providers are available based on installed packages. Falls back to Yahoo Finance if a requested provider is unavailable.

**Provider architecture:** Abstract `DataProvider` base class with 4 methods (`fetch_daily_prices`, `fetch_ticker_info`, `fetch_current_prices`, `fetch_risk_free_rate`). Each provider is a standalone module importable from any Python program.

---

## Environment Variables

Copy `.env.example` to `.env` and uncomment as needed. No API keys required for basic operation.

| Category | Variables |
|----------|-----------|
| **Config** | `LS_CONFIG_PATH` |
| **Bloomberg** | `BLOOMBERG_HOST`, `BLOOMBERG_PORT` |
| **Interactive Brokers** | `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID` |
| **Anthropic** | `ANTHROPIC_API_KEY` |
| **OpenAI** | `OPENAI_API_KEY` |
| **Google / Gemini** | `GOOGLE_API_KEY`, `GOOGLE_PROJECT_ID` |
| **Ollama** | `OLLAMA_HOST` |
| **DeepSeek** | `DEEPSEEK_API_KEY` |
| **GitHub Copilot** | `GITHUB_TOKEN` |

---

## Analytics

| Category | Metrics |
|----------|---------|
| **Return** | Sharpe, Sortino, Calmar, Deflated Sharpe Ratio (DSR) |
| **Risk** | Portfolio vol, VaR 95%, CVaR 95%, tracking error, idiosyncratic vol |
| **Drawdown** | Max DD, current DD, E[DD], P(DD≥10%), time in DD (Bailey & Lopez de Prado) |
| **Exposure** | Gross/net, net beta, HHI, top-5 concentration, L/S ratio, cash % |
| **Factors** | CAPM, Fama-French 3-factor, Carhart 4-factor (ETF proxies) |
| **Correlation** | Avg pairwise, long book, short book, L/S book, most/least correlated pairs |
| **Quality** | Composite 0-100 score — 6 dimensions: risk-adj return, DD resilience, alpha quality, diversification, tail risk, exposure balance |
| **Technical** | RSI (Wilder's, configurable 7/14/21d), ADV$ (20-day avg dollar volume) |
| **Attribution** | Position P&L, sector P&L (long/short stacked), factor decomposition (market/size/value/momentum/alpha) |
| **PM** | Hit rate, slugging %, EV per trade, long/short breakdown, sector skill |

---

## Page Workflow

```
Portfolio Dashboard  →  Trade Simulator  →  Paper Portfolio  →  PM Scorecard
   (load/generate)      (propose trades)    (toggle ON first)   (needs trades)
                         (preview impact)    (journal + NAV)     (hit rate, slugging)
                         (apply or discard)  (daily snapshots)   (sector skill)
```

**Paper Portfolio** and **PM Scorecard** require:
1. Paper Mode toggled ON in the sidebar
2. Trades applied through the Trade Simulator
3. Daily snapshots taken on the Paper Portfolio page

---

## Sidebar Controls

| Control | Options | Default |
|---------|---------|---------|
| **Lookback** | 3M, 6M, 1Y, 2Y, 3Y | 1Y (252d) |
| **Benchmark** | Any ticker | SPY |
| **RSI Period** | 7, 14, 21 days | 14 |
| **Factor Model** | CAPM, FF3, FF4 | CAPM |
| **Risk-Free Rate** | Auto (T-bill) or manual | Auto |
| **Data Source** | Yahoo, Bloomberg, IB | Yahoo Finance |
| **Alert Thresholds** | Max sector net, beta bounds, gross cap | Configurable |
| **Paper Mode** | ON / OFF | OFF |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.54 |
| **Data** | Polars (DataFrames), Pydantic (models) |
| **Charts** | Plotly |
| **Market Data** | yfinance, blpapi, ib_insync |
| **Optimization** | SciPy (SLSQP) |
| **Statistics** | NumPy, SciPy |
| **Persistence** | SQLite (cache), JSONL (trade log, snapshots) |
| **Testing** | pytest (277 tests), pytest-cov |
| **Linting** | Ruff |
| **Python** | 3.12+ |

---

## References

- Bailey, D.H. & Lopez de Prado, M. (2014). *Algorithmic Finance*, 3(1-2), 99-109. [DOI](https://doi.org/10.3233/AF-140035)
- Fama, E.F. & French, K.R. (1993). *Journal of Financial Economics*, 33(1), 3-56.
- Carhart, M.M. (1997). *The Journal of Finance*, 52(1), 57-82.

See [REFERENCES.md](REFERENCES.md) for full citations and implemented equations. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Testing

```bash
make test                         # Run all 277 tests
make coverage                     # With coverage report
make lint                         # Ruff linting
make fmt                          # Auto-format
```

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| **v1.0** | 2026-02-12 | Initial release — full dashboard, trade simulator, paper portfolio, PM scorecard, 3 data providers, 277 tests |

---

## License

MIT — see [LICENSE](LICENSE).

---

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat&logo=yahoo&logoColor=white)
![Bloomberg](https://img.shields.io/badge/Bloomberg-000000?style=flat&logo=bloomberg&logoColor=white)
![Interactive Brokers](https://img.shields.io/badge/Interactive_Brokers-D71920?style=flat)
