"""Portfolio save/load persistence.

Saves and loads portfolio state to/from JSON files.
Supports multiple saved portfolios for comparison.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from core.portfolio import Portfolio

logger = logging.getLogger(__name__)

SAVE_DIR = Path("data_cache") / "saved_portfolios"


def save_portfolio(portfolio: Portfolio, filename: str | None = None) -> Path:
    """
    Save a portfolio to a JSON file.

    Returns the path to the saved file.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{portfolio.name}_{timestamp}.json"

    filepath = SAVE_DIR / filename
    with filepath.open("w") as f:
        f.write(portfolio.model_dump_json(indent=2))

    logger.info("Portfolio saved to %s", filepath)
    return filepath


def load_portfolio(filepath: str | Path) -> Portfolio:
    """Load a portfolio from a JSON file."""
    path = Path(filepath)
    with path.open() as f:
        data = json.load(f)
    return Portfolio(**data)


def list_saved_portfolios() -> list[dict[str, str]]:
    """
    List all saved portfolio files.

    Returns list of {name, filepath, modified} dicts.
    """
    if not SAVE_DIR.exists():
        return []

    result = []
    for f in sorted(SAVE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            portfolio = load_portfolio(f)
            result.append({
                "name": portfolio.name,
                "filepath": str(f),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "positions": str(portfolio.total_count),
            })
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping invalid file: %s", f)

    return result


def delete_saved_portfolio(filepath: str | Path) -> None:
    """Delete a saved portfolio file."""
    path = Path(filepath)
    if path.exists():
        path.unlink()
        logger.info("Deleted saved portfolio: %s", path)
