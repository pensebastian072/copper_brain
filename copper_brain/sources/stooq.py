"""Stooq source — free, no key. Daily continuous copper future + macro proxies.

Hits Stooq's CSV download endpoint directly (no pandas-datareader). One request
per symbol; keeps the daily Close as the series value. Symbols from
config.STOOQ_SYMBOLS (e.g. HG.F copper, DX.F dollar index, SPY.US).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .. import config

from .base import Source

_CSV = "https://stooq.com/q/d/l/"
# Stooq serves a JS anti-bot challenge to non-browser UAs; a normal browser UA
# usually gets the raw CSV. If it still challenges, the source degrades to empty
# and the cross-check falls back to the other feeds.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


class StooqSource(Source):
    source_id = "stooq"
    needs_key = False

    def _fetch_symbol(self, symbol: str) -> pd.DataFrame:
        params = {"s": symbol.lower(), "i": "d"}  # daily
        resp = requests.get(_CSV, params=params, headers=_HEADERS, timeout=60)
        resp.raise_for_status()
        text = resp.text.strip()
        # Stooq returns the literal "No data" / "Exceeded the daily hits" on miss.
        if not text or not text.lower().startswith("date"):
            return pd.DataFrame(columns=["Date", "Close"])
        return pd.read_csv(io.StringIO(text))

    def _fetch_raw(self) -> pd.DataFrame:
        frames = []
        for series, symbol in config.STOOQ_SYMBOLS.items():
            df = self._fetch_symbol(symbol)
            if df.empty or "Close" not in df or "Date" not in df:
                continue
            out = pd.DataFrame({
                "date": df["Date"],
                "series": series,
                "value": df["Close"],
            })
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return pd.concat(frames, ignore_index=True)
