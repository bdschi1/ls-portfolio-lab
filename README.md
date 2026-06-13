<!-- ls-portfolio-lab/README.md | Last updated: 2026-06-13 -->

# LSLab — Long/Short Equity Risk Workbench

![Python](https://img.shields.io/badge/python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![tests](https://img.shields.io/badge/tests-601%20passing-brightgreen?style=flat)

Streamlit dashboard for monitoring and stress-testing long/short equity portfolios. Computes risk, return, and exposure metrics across long + short books, simulates proposed trades with before/after impact, and tracks paper-portfolio performance over time.

**Plain English:** A risk-management tool, not a signal generator. It answers: what happens to my risk profile if I add this trade, and how concentrated am I across sectors, factors, and individual names?

## Install

```
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[bloomberg]"   # or .[ib] (optional)
```

Copy `.env.example` to `.env` and fill in only what you need; no keys required for basic operation.

## Usage

```
streamlit run app/main.py --server.port=8516
./run.sh                  # setup + launch (run.sh help for all commands)
```

## What it does

- 5 pages: Portfolio Dashboard, Risk Analytics, Trade Simulator, Paper Portfolio, PM Scorecard
- Paper Mode starts an immutable JSONL trade journal + daily snapshots
- Provider auto-detect with Yahoo fallback; SQLite cache (18h prices / 7d info)
- `core/` is Streamlit-free — all metrics are pure Polars-DataFrame functions
- Sharpe inference suite (DSR, PSR, MinTRL, FDR/FWER) following Bailey & López de Prado

## Tests

```
pytest tests/ -v
make lint
```

## License

MIT
