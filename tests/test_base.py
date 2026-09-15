import pandas as pd

from copper_brain.sources.base import FetchResult, Source


def test_tidy_drops_nan_dedupes_and_sorts():
    raw = pd.DataFrame({
        "date": ["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-03"],
        "series": ["copper", "copper", "copper", "copper"],
        "value": [9.0, 8.0, 9.5, None],  # dup date -> keep last; NaN -> dropped
    })
    out = Source.tidy(raw)
    assert list(out["value"]) == [8.0, 9.5]  # sorted by date, dup last kept, NaN gone
    assert out["date"].is_monotonic_increasing


def test_tidy_empty_returns_schema():
    out = Source.tidy(None)
    assert list(out.columns) == ["date", "series", "value"]
    assert out.empty


def _mk(df, ok=True):
    return FetchResult("x", Source.tidy(df), ok=ok)


def test_fetchresult_health_ok_and_stale():
    today = pd.Timestamp.now().normalize()
    fresh = pd.DataFrame({"date": [today], "series": ["copper"], "value": [9.0]})
    assert _mk(fresh).health() == "ok"

    old = today - pd.Timedelta(days=30)
    stale = pd.DataFrame({"date": [old], "series": ["copper"], "value": [9.0]})
    assert _mk(stale).health().startswith("stale_")


def test_fetchresult_error_and_empty_health():
    assert _mk(None, ok=False).health() == "error"
    assert _mk(None, ok=True).health() == "empty"
