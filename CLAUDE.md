# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## What This Is
Streamlit-based portfolio risk workbench for long/short equity portfolio managers. Provides institutional-grade analytics, trade simulation with before/after impact preview, paper portfolio tracking, and PM performance scoring. Not an alpha generator -- a risk management tool.

## Commands
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run
streamlit run app/main.py

# Tests
pytest tests/ -v
# or: make test

# Lint
ruff check .
ruff format .
# or: make lint / make fmt

# Coverage
make coverage
```

## Architecture
- `app/` -- Streamlit application layer
  - `main.py` -- Entry point, sidebar, navigation, data source selector
  - `pages/` -- One module per page: portfolio_view, trade_simulator, paper_portfolio, pm_scorecard
  - `components/` -- Reusable UI: metrics_panel, portfolio_table, chart_gallery (12+ Plotly charts)
  - `state/` -- Session state init and portfolio save/load (JSON persistence)
- `core/` -- Pure business logic (no Streamlit imports)
  - `portfolio.py` -- Pydantic models: Portfolio, Position, ProposedTrade, TradeBasket
  - `mock_portfolio.py` -- Generator (~30L/~40S, 11 GICS sectors, $3B NAV)
  - `rebalancer.py` -- SLSQP constrained optimizer (net beta, vol targets)
  - `trade_impact.py` -- Trade simulation engine, apply trades, compute metric diffs
  - `factor_model.py` -- CAPM, FF3, FF4 regressions using ETF proxies
  - `metrics/` -- All analytics as pure functions on Polars DataFrames (return, risk, drawdown, exposure, correlation, technical, PM performance, quality score, attribution, Sharpe inference — PSR, MinTRL, CI, FDR, FWER corrections — Bailey & Lopez de Prado)
- `data/` -- Market data providers and caching
  - `provider.py` -- Abstract DataProvider base class (4 methods)
  - `yahoo_provider.py`, `bloomberg_provider.py`, `ib_provider.py` -- Concrete providers
  - `provider_factory.py` -- Registry, auto-discovery, `get_provider_safe()`
  - `cache.py` -- SQLite cache with staleness thresholds
  - `ingest.py` -- Portfolio parser (CSV, Excel, PDF)
- `history/` -- Paper portfolio persistence (append-only JSONL trade log, daily snapshots, TWR performance)

## Key Patterns
- Strict separation: `core/` has zero Streamlit imports; `app/` handles UI only
- All metrics are pure functions taking Polars DataFrames and returning typed results
- Provider abstraction: sidebar auto-detects installed providers (Yahoo default, Bloomberg, IB optional)
- SQLite caching sits between providers and the app; cache keys are (ticker, date_range, field)
- Paper portfolio uses immutable JSONL for trade journal -- append-only, never edited
- Page workflow: Dashboard -> Trade Simulator -> Paper Portfolio -> PM Scorecard

## Testing Conventions
- 481 tests in `tests/` with a `test_metrics/` subdirectory for per-module metric tests
- Tests use mock/synthetic data -- no live API calls
- Run with `pytest tests/ -v` or `make test`
- `pyproject.toml` sets `pythonpath = ["."]` so imports resolve without install
- `tests/` E501 (line length) is suppressed via ruff per-file-ignores
