"""App-layer import smoke tests.

Catches the class of bug where a page module is renamed but `app/main.py`
or sibling pages still reference the old name (e.g. `paleologo_dashboard`
→ `risk_analytics`). Streamlit needs an AppContext to actually render,
so these tests only verify imports + that `render` is callable per page.
"""

from __future__ import annotations

import importlib

import pytest

_PAGE_MODULES = [
    "app.pages.portfolio_view",
    "app.pages.risk_analytics",
    "app.pages.trade_simulator",
    "app.pages.paper_portfolio",
    "app.pages.pm_scorecard",
]

_COMPONENT_MODULES = [
    "app.components.chart_gallery",
    "app.components.metrics_panel",
    "app.components.portfolio_table",
]

_STATE_MODULES = [
    "app.state.session",
    "app.state.persistence",
]


def test_app_main_imports():
    # catches rename drift in main.py imports
    importlib.import_module("app.main")


@pytest.mark.parametrize("module_name", _PAGE_MODULES)
def test_page_module_imports_and_exports_render(module_name):
    module = importlib.import_module(module_name)
    assert hasattr(module, "render"), f"{module_name} missing render()"
    assert callable(module.render), f"{module_name}.render not callable"


@pytest.mark.parametrize("module_name", _COMPONENT_MODULES)
def test_component_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _STATE_MODULES)
def test_state_module_imports(module_name):
    importlib.import_module(module_name)


def test_main_wires_all_five_pages():
    import app.main as main_module

    src = open(main_module.__file__).read()
    for page in ["portfolio_view", "risk_analytics", "trade_simulator", "paper_portfolio", "pm_scorecard"]:
        assert page in src, f"app/main.py does not reference page module {page}"
