"""World Bank "Pink Sheet" — free, no key. Monthly copper price cross-check.

The Pink Sheet monthly workbook has a multi-row banner before the data. We scan
for the header row that contains "Copper", then parse the date column ("1960M01"
style) and the copper column. Coarse monthly series — used only to sanity-check
the daily feeds, never as the primary price.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

from .. import config
from .base import Source

_SHEET = "Monthly Prices"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# Match the monthly-history workbook link anywhere on the CMO landing page.
_LINK_RE = re.compile(
    r'https://[^"\'\s]*CMO-Historical-Data-Monthly\.xlsx', re.IGNORECASE)


def _discover_url() -> str:
    """Find the current Pink Sheet xlsx link; fall back to the known URL."""
    try:
        resp = requests.get(config.WORLDBANK_CMO_PAGE, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        m = _LINK_RE.search(resp.text)
        if m:
            return m.group(0)
    except Exception:  # noqa: BLE001
        pass
    return config.WORLDBANK_PINKSHEET_URL


def _parse_wb_date(val) -> pd.Timestamp | None:
    """World Bank months look like '1960M01'. Return month-start timestamp."""
    s = str(val).strip()
    if "M" not in s:
        return None
    try:
        year, month = s.split("M")
        return pd.Timestamp(int(year), int(month), 1)
    except (ValueError, TypeError):
        return None


class WorldBankSource(Source):
    source_id = "worldbank"
    needs_key = False

    def _fetch_raw(self) -> pd.DataFrame:
        resp = requests.get(_discover_url(), headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        book = pd.read_excel(io.BytesIO(resp.content), sheet_name=_SHEET,
                             header=None, engine="openpyxl")
        # Find the header row holding "Copper" and the units row beneath it.
        header_row = None
        for i in range(min(12, len(book))):
            row = book.iloc[i].astype(str).str.lower()
            if row.str.contains("copper").any():
                header_row = i
                break
        if header_row is None:
            raise ValueError("Pink Sheet: no 'Copper' header row found")
        headers = book.iloc[header_row].astype(str)
        copper_col = next(c for c in headers.index
                          if "copper" in str(headers[c]).lower())
        # Data starts a couple rows below the header (skip the units line).
        data = book.iloc[header_row + 1:]
        dates = data.iloc[:, 0].map(_parse_wb_date)
        out = pd.DataFrame({
            "date": dates,
            "series": "copper",
            "value": pd.to_numeric(data[copper_col], errors="coerce"),
        }).dropna(subset=["date", "value"])
        return out
