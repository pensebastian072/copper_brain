"""Westmetall source — free scrape. The ONLY free LME inventory + curve feed.

One Westmetall page returns a 4-column table:
    date | LME Copper Cash-Settlement | LME Copper 3-month | LME Copper stock

We emit four tidy series:
    lme_cu_cash, lme_cu_3m, lme_cu_stock, lme_cu_backwardation (cash - 3m)

Positive backwardation (cash > 3m) = tight physical market = bullish inventory
signal. Layout drift degrades the source to empty (never crashes ingest). Always
sends a browser-like User-Agent.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .. import config
from .base import Source

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
# Positional column -> tidy series name (col 0 is the date).
_COLS = {1: "lme_cu_cash", 2: "lme_cu_3m", 3: "lme_cu_stock"}


def _num(col: pd.Series) -> pd.Series:
    cleaned = (col.astype(str)
                  .str.replace(",", "", regex=False)        # thousands sep
                  .str.replace(r"[^\d.\-]", "", regex=True))  # strip units/junk
    return pd.to_numeric(cleaned, errors="coerce")


class WestmetallSource(Source):
    source_id = "westmetall"
    needs_key = False

    def _fetch_raw(self) -> pd.DataFrame:
        resp = requests.get(config.WESTMETALL_URL, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))  # StringIO required pandas 3.0
        tbl = next((t for t in tables if t.shape[1] >= 4 and len(t) >= 5), None)
        if tbl is None:
            return pd.DataFrame(columns=["date", "series", "value"])

        dates = pd.to_datetime(tbl.iloc[:, 0].astype(str).str.strip(),
                               format=config.WESTMETALL_DATE_FORMAT, errors="coerce")
        frames = []
        per_series: dict[str, pd.Series] = {}
        for idx, series in _COLS.items():
            vals = _num(tbl.iloc[:, idx])
            good = dates.notna() & vals.notna()
            if good.sum() == 0:
                continue
            frames.append(pd.DataFrame({
                "date": dates[good], "series": series, "value": vals[good],
            }))
            per_series[series] = pd.Series(vals[good].values, index=dates[good])

        if "lme_cu_cash" in per_series and "lme_cu_3m" in per_series:
            spread = (per_series["lme_cu_cash"] - per_series["lme_cu_3m"]).dropna()
            if not spread.empty:
                frames.append(pd.DataFrame({
                    "date": spread.index,
                    "series": "lme_cu_backwardation",
                    "value": spread.values,
                }))
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return pd.concat(frames, ignore_index=True)
