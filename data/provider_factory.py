"""Provider factory — discover, register, and switch data providers.

Auto-detects which providers are available based on installed packages:
    - Yahoo Finance (yfinance) — always available, default
    - Bloomberg (blpapi)       — optional, requires Terminal/B-PIPE
    - Interactive Brokers (ib_insync) — optional, requires TWS/Gateway

Usage:
    from data.provider_factory import get_provider, available_providers

    # List what's available
    for name in available_providers():
        print(name)

    # Get a specific provider
    provider = get_provider("Bloomberg")

    # Get default (Yahoo)
    provider = get_provider()

Standalone usage (importable from any program):
    import sys
    sys.path.insert(0, "/path/to/ls-portfolio-lab")
    from data.provider_factory import get_provider
    bbg = get_provider("Bloomberg")
    df = bbg.fetch_daily_prices(["AAPL"], start, end)
"""

from __future__ import annotations

import logging
from typing import Any

from data.provider import DataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# Each entry: (display_name, module_path, class_name, is_available_func)
_PROVIDER_REGISTRY: list[tuple[str, str, str, str]] = [
    ("Yahoo Finance", "data.yahoo_provider", "YahooProvider", None),
    (
        "Bloomberg", "data.bloomberg_provider",
        "BloombergProvider", "data.bloomberg_provider.is_available",
    ),
    (
        "Interactive Brokers", "data.ib_provider",
        "IBProvider", "data.ib_provider.is_available",
    ),
]

# Cache of instantiated providers (singleton per session)
_provider_cache: dict[str, DataProvider] = {}


def available_providers() -> list[str]:
    """Return names of all providers whose dependencies are installed.

    Always includes Yahoo Finance. Bloomberg and IB are included only
    if their respective Python packages (blpapi, ib_insync) can be imported.
    """
    available: list[str] = []

    for display_name, module_path, class_name, avail_func_path in _PROVIDER_REGISTRY:
        if avail_func_path is None:
            # Always available (Yahoo)
            available.append(display_name)
            continue

        try:
            mod_path, func_name = avail_func_path.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(mod_path)
            is_avail = getattr(mod, func_name)
            if is_avail():
                available.append(display_name)
        except Exception:
            # Module can't be imported or function fails — skip
            pass

    return available


def get_provider(
    name: str = "Yahoo Finance",
    **kwargs: Any,
) -> DataProvider:
    """Get a data provider instance by name.

    Args:
        name: Provider display name ("Yahoo Finance", "Bloomberg",
              "Interactive Brokers")
        **kwargs: Passed to the provider constructor (e.g., host, port)

    Returns:
        DataProvider instance

    Raises:
        ValueError: if provider name is not recognized
        ImportError: if required package is not installed
        ConnectionError: if provider cannot connect (Bloomberg/IB)
    """
    # Return cached instance if no kwargs override
    if name in _provider_cache and not kwargs:
        return _provider_cache[name]

    for display_name, module_path, class_name, _ in _PROVIDER_REGISTRY:
        if display_name == name:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls(**kwargs)
            _provider_cache[name] = instance
            logger.info("Data provider initialized: %s", name)
            return instance

    available = available_providers()
    msg = f"Unknown provider '{name}'. Available: {available}"
    raise ValueError(msg)


def get_provider_safe(
    name: str = "Yahoo Finance",
    **kwargs: Any,
) -> DataProvider:
    """Get a provider, falling back to Yahoo Finance on any error.

    Use this in the UI — it never raises, always returns a working provider.
    """
    try:
        return get_provider(name, **kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to initialize %s provider (%s), falling back to Yahoo Finance",
            name, exc,
        )
        return get_provider("Yahoo Finance")


def clear_cache() -> None:
    """Clear the provider cache (useful for reconnecting)."""
    for name, provider in _provider_cache.items():
        if hasattr(provider, "close"):
            try:
                provider.close()
            except Exception:
                pass
    _provider_cache.clear()


# ---------------------------------------------------------------------------
# Convenience: provider descriptions for UI
# ---------------------------------------------------------------------------

PROVIDER_DESCRIPTIONS: dict[str, str] = {
    "Yahoo Finance": "Free, no account required. EOD data, ~18hr delay.",
    "Bloomberg": "Professional terminal. Real-time, institutional-grade reference data.",
    "Interactive Brokers": "Brokerage account. Real-time quotes, execution-ready.",
}
