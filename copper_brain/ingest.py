"""Ingest orchestrator — pull every source, write clean parquet + freshness.

Runs each Source.fetch() (which never raises), concatenates the tidy frames into
one long table data/clean/observations.parquet, runs the cross-check, and writes
journal/freshness.json. Returns the combined frame + the freshness records so
callers (update.py, tests) can act on health without re-reading disk.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .crosscheck import crosscheck_copper
from .sources.base import FetchResult, write_freshness
from .sources.fred import FredSource
from .sources.stooq import StooqSource
from .sources.usgs_icsg import UsgsIcsgSource
from .sources.westmetall import WestmetallSource
from .sources.worldbank import WorldBankSource
from .sources.yfinance_src import YFinanceSource

CLEAN_FILE = config.CLEAN_DIR / "observations.parquet"

# Order matters only for readability; all run regardless of each other's result.
SOURCES = [
    FredSource(),
    StooqSource(),
    YFinanceSource(),
    WestmetallSource(),
    WorldBankSource(),
    UsgsIcsgSource(),
]


def run_ingest(verbose: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch all sources, persist clean table + freshness manifest."""
    results: list[FetchResult] = []
    frames: list[pd.DataFrame] = []
    for src in SOURCES:
        res = src.fetch()
        results.append(res)
        if res.df is not None and not res.df.empty:
            tagged = res.df.copy()
            tagged["source"] = src.source_id
            frames.append(tagged)
        if verbose:
            print(f"[{src.source_id:10}] {res.health():>10}  "
                  f"rows={0 if res.df is None else len(res.df):>5}  "
                  f"through={res.data_through}  err={res.error or ''}")

    combined = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["date", "series", "value", "source"]))
    if not combined.empty:
        combined.to_parquet(CLEAN_FILE, index=False)

    # Cross-check copper price agreement across the daily/monthly sources.
    cc = crosscheck_copper(combined)

    records = [r.freshness_record() for r in results]
    # Attach the cross-check verdict as a synthetic record so the manifest is
    # self-contained for the publisher's source_health block.
    records.append({"source": "_crosscheck", **cc})
    write_freshness(records)

    if verbose:
        print(f"\ncrosscheck: {cc['verdict']}  "
              f"diverging={cc.get('diverging') or 'none'}")
    return combined, records


def load_clean() -> pd.DataFrame:
    """Load the last clean observations table (empty frame if none yet)."""
    if CLEAN_FILE.exists():
        return pd.read_parquet(CLEAN_FILE)
    return pd.DataFrame(columns=["date", "series", "value", "source"])
