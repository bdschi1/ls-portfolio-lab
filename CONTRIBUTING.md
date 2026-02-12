# Contributing to LS Portfolio Lab

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

```bash
# Run the app
make run

# Run tests
make test

# Lint & format
make lint
make fmt
```

## Code Standards

- Python 3.12+
- Ruff for linting and formatting (line length: 100)
- Pydantic for all data models
- Polars for DataFrames (not Pandas)
- Pure functions in `core/metrics/` (no side effects)
- All new features must have tests

## Architecture

- `app/` — Streamlit UI only (no business logic)
- `core/` — Pure computation (no UI dependencies)
- `data/` — Data acquisition and caching
- `history/` — Paper portfolio persistence
- `tests/` — Mirrors source structure

## Testing

All tests must pass before merging:

```bash
pytest tests/ -v
```

Target: 100% of metric functions covered.
