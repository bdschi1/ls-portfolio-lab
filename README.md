# LS Portfolio Lab

![Python](https://img.shields.io/badge/python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)

**Long/Short Equity Portfolio Risk Workbench**

A Streamlit-based dashboard for monitoring and stress-testing long/short equity portfolios. It computes risk, return, and exposure metrics across a portfolio of long and short stock positions, lets you simulate proposed trades and see their impact before executing, and tracks paper portfolio performance over time.

This is a risk management tool, not a signal generator. It answers the question: *"What happens to my risk profile if I add this trade?"*

This is a continually developed project. Features, interfaces, and test coverage expand over time as new research ideas and workflow needs arise.

---

## Quickstart

```bash
git clone https://github.com/bdschi1/ls-portfolio-lab.git && cd ls-portfolio-lab
./run.sh            # setup + launch Streamlit app
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app/main.py
```

Run `./run.sh help` for all commands (`setup`, `app`, `test`).

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
- **Top metrics bar:** Portfolio Vol, Net Beta, Sharpe, weighted RSI, Gross/Net exposure, Cash %, Time in Drawdown, Quality Score (0-100)
- **Detail grid:** Summary (NAV, L/S ratio, HHI), Risk (Sortino, VaR, CVaR, DSR), Drawdown (max DD, current DD, Calmar, expected DD, time in DD), Factors (CAPM/FF3/FF4 alpha, beta, systematic %, factor tilts), Correlation (pairwise, long book, short book, L/S correlation, idiosyncratic %)
- **Position table:** Filterable by side/sector, 20+ columns including annualized vol, beta, RSI, ADV$ (20-day average dollar volume), alpha, P&L, weight
- **Charts:** Sector exposure, beta scatter, risk contribution, RSI heatmap, correlation matrix, NAV curve, drawdown, P&L waterfall, sector P&L, dispersion, quality score radar, rolling metrics
- **Rebalancer:** Constrained optimizer targeting net beta and annualized volatility

*In plain language: the dashboard shows where your risk is concentrated, how diversified the portfolio is, and whether the long and short books are behaving as expected.*

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

*Hit rate is the fraction of trades that made money. Slugging % is the average winner divided by the average loser — it measures how much you make when you're right relative to how much you lose when you're wrong.*

---

## Repository Structure

```
ls-portfolio-lab/
│
├── app/                              # Streamlit application layer
│   ├── main.py                       # Entry point — sidebar, navigation, data source selector
│   ├── pages/                        # One module per page
│   │   ├── portfolio_view.py         # Portfolio Dashboard
│   │   ├── trade_simulator.py        # Trade Simulator
│   │   ├── paper_portfolio.py        # Paper Portfolio
│   │   └── pm_scorecard.py           # PM Scorecard
│   ├── components/                   # Reusable UI components
│   │   ├── metrics_panel.py          # Top metrics bar, detail grid, sector exposure chart
│   │   ├── portfolio_table.py        # Position table — 20+ columns, filters
│   │   └── chart_gallery.py          # 12+ Plotly charts
│   └── state/                        # Session state & persistence
│       ├── session.py                # Session state init — cache, settings, alerts
│       └── persistence.py            # Portfolio save/load (JSON to disk)
│
├── core/                             # Pure business logic (no Streamlit imports)
│   ├── portfolio.py                  # Pydantic models — Portfolio, Position, ProposedTrade, TradeBasket
│   ├── mock_portfolio.py             # Mock portfolio generator (~30L/~40S, 11 GICS sectors, $3B NAV)
│   ├── rebalancer.py                 # SLSQP constrained optimizer (net beta, vol targets)
│   ├── trade_impact.py               # Trade simulation engine — apply trades, compute metric diffs
│   ├── factor_model.py               # CAPM, FF3, FF4 regressions (ETF proxies)
│   └── metrics/                      # All analytics — pure functions on Polars DataFrames
│       ├── return_metrics.py         # Sharpe, Sortino, Calmar, DSR, PSR, Sharpe CI, MinTRL
│       ├── sharpe_inference.py       # Full Sharpe inference: PSR, MinTRL, critical SR, power, FDR, FWER
│       ├── risk_metrics.py           # Portfolio vol, VaR 95%, CVaR 95%, beta, idio vol, MCR
│       ├── drawdown_analytics.py     # Bailey & Lopez de Prado — E[DD], P(DD≥b), time in DD
│       ├── drawdown_metrics.py       # Empirical drawdown — max DD, current DD, recovery
│       ├── exposure_metrics.py       # Gross/net exposure, net beta, HHI, top-5 concentration
│       ├── correlation_metrics.py    # Pairwise, partial, long book, short book, L/S correlation
│       ├── technical_metrics.py      # RSI (Wilder's), SMA, momentum, 52-week high/low
│       ├── pm_performance.py         # Hit rate, slugging %, EV per trade, sector attribution
│       ├── quality_score.py          # Composite 0-100 score — 6 weighted dimensions (A+ to F)
│       └── attribution.py            # P&L attribution — position, sector, side, factor decomposition
│
├── data/                             # Market data providers & caching
│   ├── provider.py                   # Abstract DataProvider base class (4 methods)
│   ├── yahoo_provider.py             # Yahoo Finance — default, free, EOD data
│   ├── bloomberg_provider.py         # Bloomberg Professional API (optional)
│   ├── ib_provider.py                # Interactive Brokers (optional)
│   ├── provider_factory.py           # Provider registry, auto-discovery, fallback
│   ├── cache.py                      # SQLite cache — 18hr price staleness, 7d info staleness
│   ├── universe.py                   # ~440 Russell 1000 tickers (market cap > $5B)
│   ├── sector_map.py                 # GICS sector/subsector classification + ETF detection
│   └── ingest.py                     # Portfolio parser — CSV, Excel, PDF extraction
│
├── history/                          # Paper portfolio persistence (append-only)
│   ├── trade_log.py                  # JSONL trade journal — immutable, timestamped records
│   ├── snapshot.py                   # Daily snapshots — NAV, positions, sector exposures
│   └── performance.py                # Time-weighted return, PM scorecard generation
│
├── tests/                            # 450 tests (pytest)
│   ├── test_portfolio.py             # Portfolio, Position, ProposedTrade, TradeBasket models
│   ├── test_trade_impact.py          # Trade application, simulation, cash tracking
│   ├── test_rebalancer.py            # Optimizer constraints, convergence, side preservation
│   ├── test_mock_portfolio.py         # Mock generation, sector diversity, universe sync
│   ├── test_drawdown_analytics.py    # Bailey & Lopez de Prado analytical formulas
│   ├── test_factor_model.py          # CAPM, FF3, FF4 regression accuracy
│   ├── test_history.py               # Trade log, snapshot store, round-trip persistence
│   └── test_metrics/                 # Per-module metric tests
│       ├── test_return_metrics.py    # Sharpe, Sortino, Calmar, DSR formula validation
│       ├── test_risk_metrics.py      # Vol, VaR, CVaR, beta, MCR Euler decomposition
│       ├── test_sharpe_inference.py  # PSR, MinTRL, FWER corrections
│       ├── test_correlation_metrics.py # Pairwise, partial (3-asset analytic), L/S book
│       ├── test_drawdown_metrics.py  # Empirical drawdown computation
│       ├── test_exposure_metrics.py  # Gross/net, HHI, sector limits
│       ├── test_technical_metrics.py # RSI, momentum, 52-week high/low
│       ├── test_pm_performance.py    # Hit rate, slugging, sector skill
│       ├── test_attribution.py       # Position/sector/side/factor attribution, sum invariants
│       └── test_quality_score.py     # Subscores, grade boundaries, composite pipeline
│
├── config.yaml                       # Default configuration (cache, metrics, alerts)
├── pyproject.toml                    # Project metadata, dependencies, ruff, pytest
├── requirements.txt                  # Pinned dependencies
├── Makefile                          # Dev shortcuts — run, test, coverage, lint, fmt, install
├── .env.example                      # Environment variable template
├── DESIGN.md                         # Detailed design document with formula derivations
├── REFERENCES.md                     # Academic citations with implemented equations
├── CITATION.cff                      # GitHub citation metadata
├── CONTRIBUTING.md                   # Contribution guidelines
└── LICENSE                           # MIT License
```

---

## Data Providers

| Provider | Status | Requirements | Data |
|----------|--------|-------------|------|
| **Yahoo Finance** | Default | None (free) | EOD adjusted close, ~18hr delay |
| **Bloomberg** | Optional | Terminal + `pip install blpapi` | Real-time, institutional reference data |
| **Interactive Brokers** | Optional | TWS/Gateway + `pip install ib_insync` | Real-time quotes |

Switch providers in the sidebar under **Data Source**. The system auto-detects installed packages and falls back to Yahoo Finance if a requested provider is unavailable.

The provider layer uses an abstract base class with 4 methods (`fetch_daily_prices`, `fetch_ticker_info`, `fetch_current_prices`, `fetch_risk_free_rate`). Each provider is a standalone module. A SQLite caching layer sits between the provider and the app, avoiding redundant API calls (18-hour staleness for prices, 7-day for fundamentals).

---

## Analytics

| Category | Metrics |
|----------|---------|
| **Return** | Sharpe, Sortino, Calmar, Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), Sharpe CI, Minimum Track Record Length (MinTRL) |
| **Risk** | Portfolio vol, VaR 95%, CVaR 95%, tracking error, idiosyncratic vol, marginal contribution to risk |
| **Drawdown** | Max DD, current DD, E[DD], P(DD≥b), time in DD (Bailey & Lopez de Prado analytical framework) |
| **Exposure** | Gross/net, net beta, HHI concentration, top-5 concentration, L/S ratio, cash % |
| **Factors** | CAPM, Fama-French 3-factor, Carhart 4-factor via ETF proxies |
| **Correlation** | Avg pairwise, partial (precision matrix), long book, short book, L/S book, most/least correlated pairs |
| **Quality** | Composite 0-100 score across 6 dimensions: risk-adjusted return, drawdown resilience, alpha quality, diversification, tail risk, exposure balance |
| **Technical** | RSI (Wilder's, configurable 7/14/21d), ADV$ (20-day avg dollar volume) |
| **Attribution** | Position P&L, sector P&L (long/short), side, factor decomposition (market/size/value/momentum/alpha) |
| **PM** | Hit rate, slugging %, EV per trade, long/short breakdown, sector skill |

*DSR adjusts the Sharpe ratio for non-normality and multiple testing — it answers whether a reported Sharpe ratio is statistically distinguishable from luck. PSR gives the probability that the true Sharpe exceeds a benchmark. MinTRL is the minimum number of observations needed to trust the result.*

---

## Page Workflow

```
┌─ PAGE 1 ── Portfolio Dashboard ──────────────────────────┐
│  Load or generate portfolio                               │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌─ PAGE 2 ── Trade Simulator ──────────────────────────────┐
│  Propose trades, preview impact, apply or discard         │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌─ PAGE 3 ── Paper Portfolio ──────────────────────────────┐
│  Toggle ON first · trade journal + NAV · daily snapshots  │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌─ PAGE 4 ── PM Scorecard ────────────────────────────────┐
│  Hit rate, slugging %, sector skill (needs trades)        │
└──────────────────────────────────────────────────────────┘
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

## Environment Variables

Copy `.env.example` to `.env` and uncomment as needed. No API keys required for basic operation.

| Category | Variables |
|----------|-----------|
| **Config** | `LS_CONFIG_PATH` |
| **Bloomberg** | `BLOOMBERG_HOST`, `BLOOMBERG_PORT` |
| **Interactive Brokers** | `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID` |

---

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v               # Run all 450 tests
pytest tests/ -v --cov=core    # With coverage report
make lint                      # Ruff linting
make fmt                       # Auto-format
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **Data** | Polars (DataFrames), Pydantic v2 (models) |
| **Charts** | Plotly |
| **Market Data** | yfinance, blpapi, ib_insync |
| **Optimization** | SciPy (SLSQP) |
| **Statistics** | NumPy, SciPy |
| **Persistence** | SQLite (cache), JSONL (trade log, snapshots) |
| **Testing** | pytest (450 tests), pytest-cov |
| **Linting** | Ruff |
| **Python** | 3.12+ |

---

## References

- Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107.
- Bailey, D.H. & Lopez de Prado, M. (2014). The Sharpe Ratio Efficient Frontier. *Algorithmic Finance*, 3(1-2), 99-109. [DOI](https://doi.org/10.3233/AF-140035)
- Lo, A. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36-52.
- Paleologo, G. (2024). *The Elements of Quantitative Investing*. Insight 4.2: Precision Matrix and Partial Correlations.
- Fama, E.F. & French, K.R. (1993). *Journal of Financial Economics*, 33(1), 3-56.
- Carhart, M.M. (1997). *The Journal of Finance*, 52(1), 57-82.

See [REFERENCES.md](REFERENCES.md) for full citations and implemented equations. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Related Work

- **AlphaAgents** (Zhao et al., 2025) — Multi-agent framework for equity portfolio construction using Fundamental, Sentiment, and Valuation agents with risk tolerance conditioning. [arXiv:2508.11152](https://arxiv.org/abs/2508.11152)
- **XAI for SME Investment** (Babaei & Giudici, 2025) — Dual-component XAI framework using XGBoost + SHAP for credit risk and expected return estimation. [Expert Systems with Applications, 2025]
- **Interpretable ML for Corporate Financialization** (Wang et al., 2025) — SHAP-enhanced framework revealing non-linear relationships between financial variables. [Mathematics, 2025]

---

## Contributing

Contributions welcome. Areas for improvement:
- Additional analytics and risk metrics
- New data provider integrations
- Enhanced chart types and dashboard views
- Extended paper portfolio tracking features

## Status

This project is under active, ongoing development. Core analytics, trade simulation, and paper portfolio tracking are stable. New metrics, chart types, and provider integrations are added as workflow needs evolve.

## License

MIT — see [LICENSE](LICENSE).
