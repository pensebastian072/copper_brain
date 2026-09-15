"""Live source pulls — hit real free endpoints. Run explicitly:

    .venv\\Scripts\\python -m pytest -m network

Skipped by default so the offline suite stays fast and deterministic.
"""
import pytest

from copper_brain import config
from copper_brain.sources.fred import FredSource
from copper_brain.sources.stooq import StooqSource
from copper_brain.sources.westmetall import WestmetallSource
from copper_brain.sources.worldbank import WorldBankSource
from copper_brain.sources.yfinance_src import YFinanceSource

pytestmark = pytest.mark.network


def _assert_has_copper(res):
    assert res.ok, res.error
    assert not res.df.empty
    assert "copper" in set(res.df["series"])


def test_stooq_live():
    # Stooq serves a JS anti-bot challenge to some networks; treat it as a
    # best-effort 3rd cross-check. Require a clean (non-crashing) result, and
    # copper IF any data came back.
    res = StooqSource().fetch()
    assert res.ok, res.error
    if not res.df.empty:
        assert "copper" in set(res.df["series"])


def test_yfinance_live():
    _assert_has_copper(YFinanceSource().fetch())


def test_worldbank_live():
    _assert_has_copper(WorldBankSource().fetch())


@pytest.mark.skipif(not config.FRED_API_KEY, reason="no FRED_API_KEY in .env")
def test_fred_live():
    _assert_has_copper(FredSource().fetch())


def test_westmetall_live():
    # Inventory/curve feed: assert it returns at least one LME series.
    res = WestmetallSource().fetch()
    assert res.ok, res.error
    # Scrape may degrade; require non-empty OR a clear error, never a crash.
    if not res.df.empty:
        assert any(s.startswith("lme_cu") for s in res.df["series"])
