import pandas as pd

from copper_brain import config
from copper_brain.crosscheck import crosscheck_copper


def _series(source, start_val, end_val, days=40):
    # Anchor the latest point to "today" so the cross-check freshness filter
    # (CROSSCHECK_MAX_AGE_DAYS) keeps the series in scope.
    end = pd.Timestamp.now().normalize()
    d0 = end - pd.Timedelta(days=days)
    return pd.DataFrame({
        "date": [d0, end],
        "series": ["copper", "copper"],
        "value": [start_val, end_val],
        "source": [source, source],
    })


def test_agreeing_sources_pass():
    # all +10% over the window, different unit levels
    df = pd.concat([
        _series("fred", 9000, 9900),
        _series("stooq", 4.0, 4.4),
        _series("yfinance", 4.0, 4.4),
    ], ignore_index=True)
    cc = crosscheck_copper(df)
    assert cc["verdict"] == "ok"
    assert cc["diverging"] == []


def test_one_diverging_source_flagged():
    df = pd.concat([
        _series("fred", 9000, 9900),     # +10%
        _series("stooq", 4.0, 4.4),      # +10%
        _series("yfinance", 4.0, 4.0),   # 0% -> diverges
    ], ignore_index=True)
    cc = crosscheck_copper(df)
    assert cc["verdict"] == "divergent"
    assert "yfinance" in cc["diverging"]


def test_insufficient_when_one_source():
    cc = crosscheck_copper(_series("fred", 9000, 9900))
    assert cc["verdict"] == "insufficient"


def test_empty_input():
    cc = crosscheck_copper(pd.DataFrame(columns=["date", "series", "value", "source"]))
    assert cc["verdict"] == "insufficient"
    assert cc["tolerance"] == config.CROSSCHECK_PCT_TOLERANCE
