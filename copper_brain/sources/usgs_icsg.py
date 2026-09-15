"""USGS / ICSG manual loader — production & refined-usage drops.

No free clean API for global copper mine production / refined balance. The user
drops CSVs into data/manual/ and this loader tidies them. Expected CSV schema
(header row, case-insensitive):

    date, series, value
    2026-01-01, mine_production_kt, 1850
    2026-01-01, refined_usage_kt, 1900
    2026-01-01, refined_balance_kt, -50

`series` is free-form but prefix with the metric. Anything unparseable is
skipped with no crash — manual data is optional, the model degrades to price/
inventory/macro features when it's absent.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from .base import Source


class UsgsIcsgSource(Source):
    source_id = "usgs_icsg"
    needs_key = False

    def _fetch_raw(self) -> pd.DataFrame:
        frames = []
        for fp in sorted(config.MANUAL_DIR.glob("*.csv")):
            try:
                df = pd.read_csv(fp)
            except Exception:  # noqa: BLE001
                continue
            cols = {c.lower().strip(): c for c in df.columns}
            if not {"date", "series", "value"} <= set(cols):
                continue
            sub = pd.DataFrame({
                "date": df[cols["date"]],
                "series": df[cols["series"]].astype(str),
                "value": df[cols["value"]],
            })
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return pd.concat(frames, ignore_index=True)
