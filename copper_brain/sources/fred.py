"""FRED source — keyed, free. Copper price + USD/rates/VIX macro series.

Hits the FRED JSON observations API directly (no pandas-datareader — that lib is
unmaintained against pandas 3.0). One request per series id; missing values come
back as "." and are dropped by Source.tidy. Series ids from config.FRED_SERIES.
"""
from __future__ import annotations

import pandas as pd
import requests

from .. import config
from .base import Source

_API = "https://api.stlouisfed.org/fred/series/observations"


class FredSource(Source):
    source_id = "fred"
    needs_key = True

    def key_available(self) -> bool:
        return bool(config.FRED_API_KEY)

    def _fetch_series(self, series_id: str) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
        }
        resp = requests.get(_API, params=params, timeout=60)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        return pd.DataFrame(obs)  # columns: date, value, realtime_start/end

    def _fetch_raw(self) -> pd.DataFrame:
        frames = []
        for series, series_id in config.FRED_SERIES.items():
            df = self._fetch_series(series_id)
            if df.empty:
                continue
            out = pd.DataFrame({
                "date": df["date"],
                "series": series,
                "value": df["value"],  # "." -> NaN via tidy's to_numeric
            })
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return pd.concat(frames, ignore_index=True)
