# LS Portfolio Lab — v1.0

**Long/Short Equity Portfolio Risk Workbench**

A local-first risk cockpit for long/short equity portfolio managers. Monitor, stress-test, and track your portfolio's risk/return profile with institutional-grade analytics.

Not an alpha generator — a **risk management tool** that answers: *"What happens to my risk if I add this trade?"*

---

## Quickstart

```bash
# Clone and install
git clone <repo-url> && cd ls-portfolio-lab
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser. Generate a mock portfolio or upload your own (CSV/Excel).

---

## Pages

### 1. Portfolio Dashboard
- **Top metrics bar:** Portfolio Vol, Net Beta, Sharpe, weighted RSI (long/short books), Gross/Net exposure, Cash %, Time in Drawdown, Quality Score
- **Detail grid:** Summary, Risk (Sortino, VaR, CVaR, DSR), Drawdown (Bailey & Lopez de Prado analytical framework), Factors (CAPM/FF3/FF4), Correlation (L/S Corr, Idio %), Sector exposure
- **Position table:** Filterable by side/sector, per-name annualized vol, idiosyncratic vol, beta, RSI, ADV$ (20-day avg dollar volume), alpha 30d/1yr. A+/A- font controls
- **Charts:** Sector exposure, beta scatter, risk contribution, RSI heatmap, correlation matrix (with most/least correlated tables), NAV curve, drawdown, P&L waterfall, sector P&L, dispersion, quality score radar, rolling metrics

### 2. Trade Simulator
- Model up to 10 trades per basket (BUY, SHORT, ADD, REDUCE, SELL, COVER, EXIT)
- Supports equities, ETFs, and options with delta adjustment
- Full before/after metric comparison — then apply or discard

### 3. Paper Portfolio
- Toggle Paper Mode in the sidebar to start tracking
- Immutable JSONL trade journal with daily NAV snapshots
- Tracks every trade with entry/exit prices and realized P&L
- Persists to disk — survives app restarts

### 4. PM Scorecard
- Hit rate, slugging %, expected value per trade
- Long vs. short breakdown, sector attribution
- NAV curve, drawdown behavior, turnover analysis

---

## Architecture

```
ls-portfolio-lab/
├── app/                          # Streamlit application
│   ├── main.py                   # Entry point, sidebar, navigation
│   ├── pages/
│   │   ├── portfolio_view.py     # Main dashboard
│   │   ├── trade_simulator.py    # What-if trade entry
│   │   ├── paper_portfolio.py    # Trade history & snapshots
│   │   └── pm_scorecard.py       # PM performance analytics
│   ├── components/
│   │   ├── metrics_panel.py      # Top bar, detail grid, sector chart
│   │   ├── portfolio_table.py    # Position table with A+/A- controls
│   │   └── chart_gallery.py      # Interactive Plotly charts
│   └── state/
│       ├── session.py            # Session state initialization
│       └── persistence.py        # Portfolio save/load
│
├── core/                         # Pure business logic (no Streamlit)
│   ├── portfolio.py              # Pydantic models (Portfolio, Position)
│   ├── mock_portfolio.py         # Constrained mock portfolio generator
│   ├── rebalancer.py             # SLSQP portfolio optimizer
│   ├── trade_impact.py           # Trade simulation engine
│   ├── factor_model.py           # CAPM, FF3, FF4 regressions
│   └── metrics/
│       ├── return_metrics.py     # Sharpe, Sortino, Calmar, DSR
│       ├── risk_metrics.py       # Vol, VaR, CVaR, beta, MCR
│       ├── drawdown_analytics.py # Bailey & Lopez de Prado framework
│       ├── drawdown_metrics.py   # Empirical drawdown analysis
│       ├── exposure_metrics.py   # Gross/net, HHI, sector limits
│       ├── correlation_metrics.py# Pairwise, L/S book correlation
│       ├── technical_metrics.py  # RSI, SMA, momentum
│       ├── pm_performance.py     # Hit rate, slugging, sector skill
│       ├── quality_score.py      # Composite portfolio quality (0-100)
│       └── attribution.py        # P&L attribution (position, sector, factor)
│
├── data/                         # Market data layer
│   ├── provider.py               # Abstract DataProvider interface
│   ├── yahoo_provider.py         # Yahoo Finance (free, default)
│   ├── bloomberg_provider.py     # Bloomberg Professional API (DAPI)
│   ├── ib_provider.py            # Interactive Brokers (TWS/Gateway)
│   ├── provider_factory.py       # Provider registry & auto-discovery
│   ├── cache.py                  # SQLite cache (18hr prices, 7d info)
│   ├── universe.py               # ~440 Russell 1000 names (>$5B mcap)
│   ├── sector_map.py             # GICS sector/subsector classification
│   └── ingest.py                 # CSV/Excel/PDF portfolio parser
│
├── history/                      # Paper portfolio persistence
│   ├── trade_log.py              # Append-only JSONL trade journal
│   ├── snapshot.py               # Daily NAV snapshot store
│   └── performance.py            # TWR, scorecard generation
│
├── tests/                        # 277 tests
├── config.yaml                   # Default configuration
├── pyproject.toml                # Project metadata & tool config
├── Makefile                      # Dev shortcuts (run, test, lint, fmt)
├── DESIGN.md                     # Detailed design document
├── REFERENCES.md                 # Academic citations with formulas
├── CITATION.cff                  # GitHub citation metadata
└── CONTRIBUTING.md               # Contribution guidelines
```

---

## Data Providers

| Provider | Status | Requirements |
|----------|--------|-------------|
| **Yahoo Finance** | Default | None (free, no API key) |
| **Bloomberg** | Optional | Bloomberg Terminal + `pip install blpapi` |
| **Interactive Brokers** | Optional | TWS/Gateway running + `pip install ib_insync` |

Switch providers in the sidebar under **🔌 Data Source**. The system auto-detects which providers are available based on installed packages.

---

## Analytics

| Category | Metrics |
|----------|---------|
| **Return** | Sharpe, Sortino, Calmar, Deflated Sharpe Ratio (DSR) |
| **Risk** | Portfolio vol, VaR 95%, CVaR 95%, tracking error |
| **Drawdown** | Max DD, current DD, E[DD], P(DD≥10%), time in DD (Bailey & Lopez de Prado) |
| **Exposure** | Gross/net, net beta, HHI, top-5 concentration, L/S ratio |
| **Factors** | CAPM, Fama-French 3-factor, Carhart 4-factor (ETF proxies) |
| **Correlation** | Avg pairwise, long book, short book, L/S book, most/least correlated pairs |
| **Quality** | Composite 0-100 score (risk-adj return, DD resilience, alpha, diversification, tail risk, exposure balance) |
| **Technical** | RSI (Wilder's, configurable period), ADV$ (20-day avg dollar volume) |
| **Attribution** | Position P&L, sector P&L (long/short breakdown), factor decomposition |
| **PM** | Hit rate, slugging %, EV per trade, sector attribution |

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
