# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0] - 2026-02-12

### Added
- Portfolio Dashboard with top metrics bar (vol, beta, Sharpe, RSI, gross/net exposure, cash %, time in DD, quality score), detail grid, position table (20+ columns), and 12+ Plotly charts
- Trade Simulator supporting up to 10 trades per basket (BUY, SHORT, ADD, REDUCE, SELL, COVER, EXIT) with equities, ETFs, and options (delta-adjusted), plus before/after metric comparison
- Paper Portfolio with immutable JSONL trade journal, daily NAV snapshots, and persistence across app restarts
- PM Scorecard with hit rate, slugging %, EV per trade, long/short breakdown, and sector skill table
- Data provider abstraction (ABC) with Yahoo Finance (default), Bloomberg, and Interactive Brokers providers; auto-detection and sidebar switching
- SQLite caching layer with 18hr price staleness and 7d info staleness
- Pydantic models for Portfolio, Position, ProposedTrade, and TradeBasket
- Mock portfolio generator (~30L/~40S positions, 11 GICS sectors, $3B NAV)
- SLSQP constrained rebalancer targeting net beta and annualized volatility
- Analytics suite: Sharpe, Sortino, Calmar, DSR, VaR 95%, CVaR 95%, portfolio vol, idiosyncratic vol, beta, HHI, RSI, quality score (0-100), factor models (CAPM/FF3/FF4), correlation analysis, P&L attribution, drawdown analytics (Bailey & Lopez de Prado)
- Russell 1000 universe (~440 tickers, market cap > $5B)
- Portfolio ingestion from CSV, Excel (.xlsx/.xls), and PDF
- 277 tests across 8 test files and a test_metrics subdirectory
