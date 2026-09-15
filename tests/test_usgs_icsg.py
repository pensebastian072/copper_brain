import pandas as pd

from copper_brain import config
from copper_brain.sources.usgs_icsg import UsgsIcsgSource


def test_manual_csv_loaded(tmp_path, monkeypatch):
    csv = config.MANUAL_DIR / "_test_icsg.csv"
    pd.DataFrame({
        "date": ["2026-01-01", "2026-02-01"],
        "series": ["mine_production_kt", "refined_usage_kt"],
        "value": [1850, 1900],
    }).to_csv(csv, index=False)
    try:
        res = UsgsIcsgSource().fetch()
        assert res.ok
        assert set(res.df["series"]) >= {"mine_production_kt", "refined_usage_kt"}
    finally:
        csv.unlink(missing_ok=True)


def test_missing_columns_skipped(tmp_path):
    bad = config.MANUAL_DIR / "_test_bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
    try:
        res = UsgsIcsgSource().fetch()
        # No valid manual rows from the bad file -> empty but not an error.
        assert res.ok
    finally:
        bad.unlink(missing_ok=True)
