"""Source ABC — common contract for every free data feed.

A Source.fetch() returns a tidy long-format DataFrame:

    columns = ["date", "series", "value"]   (date = tz-naive datetime64, UTC day)

`series` is a stable semantic key (e.g. "copper", "usd_broad") so downstream
features never care which raw vendor symbol produced it. The base class handles
raw caching to data/raw/<id>.parquet and a freshness record, so subclasses only
implement `_fetch_raw()`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .. import config

TIDY_COLUMNS = ["date", "series", "value"]


@dataclass
class FetchResult:
    """Outcome of one source pull — tidy data plus a freshness record."""
    source_id: str
    df: pd.DataFrame                     # tidy: date, series, value
    ok: bool
    error: str | None = None
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def data_through(self) -> str | None:
        if self.df is None or self.df.empty:
            return None
        return pd.to_datetime(self.df["date"]).max().date().isoformat()

    @property
    def age_days(self) -> int | None:
        dt = self.data_through
        if dt is None:
            return None
        last = datetime.fromisoformat(dt).date()
        return (datetime.now(timezone.utc).date() - last).days

    @property
    def stale(self) -> bool:
        age = self.age_days
        return age is None or age > config.SOURCE_STALE_DAYS

    def health(self) -> str:
        if not self.ok:
            return "error"
        if self.df is None or self.df.empty:
            return "empty"
        if self.stale:
            return f"stale_{self.age_days}d"
        return "ok"

    def freshness_record(self) -> dict:
        return {
            "source": self.source_id,
            "ok": self.ok,
            "health": self.health(),
            "data_through": self.data_through,
            "age_days": self.age_days,
            "rows": 0 if self.df is None else int(len(self.df)),
            "series": [] if self.df is None or self.df.empty
                      else sorted(self.df["series"].unique().tolist()),
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


class Source(ABC):
    """Base class for a free data feed."""

    #: stable id used for cache filename + freshness key
    source_id: str = "base"
    #: True if the source needs an API key (skipped with a clear msg if missing)
    needs_key: bool = False

    @abstractmethod
    def _fetch_raw(self) -> pd.DataFrame:
        """Return a tidy DataFrame [date, series, value]. May raise on failure."""
        raise NotImplementedError

    def key_available(self) -> bool:
        return True  # overridden by keyed sources

    def _cache_path(self):
        return config.RAW_DIR / f"{self.source_id}.parquet"

    @staticmethod
    def tidy(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize: enforce columns, drop NaN values, sort, dedupe."""
        if df is None or df.empty:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        out = df[TIDY_COLUMNS].copy()
        # utc=True normalizes mixed naive (FRED/Stooq) + tz-aware (yfinance)
        # inputs to one UTC basis; tz_localize(None) then drops tz for a clean
        # naive day index. (Plain tz_localize(None) raises on already-naive.)
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None).dt.normalize()
        out["series"] = out["series"].astype(str)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        out = (out.drop_duplicates(subset=["date", "series"], keep="last")
                  .sort_values(["series", "date"])
                  .reset_index(drop=True))
        return out

    def fetch(self) -> FetchResult:
        """Pull, tidy, cache raw to parquet. Never raises — errors -> FetchResult."""
        if self.needs_key and not self.key_available():
            return FetchResult(self.source_id, self.tidy(None), ok=False,
                               error="missing API key")
        try:
            raw = self._fetch_raw()
            df = self.tidy(raw)
            if not df.empty:
                df.to_parquet(self._cache_path(), index=False)
            return FetchResult(self.source_id, df, ok=True)
        except Exception as exc:  # noqa: BLE001 — log-and-degrade, never crash ingest
            cached = self._load_cache()
            return FetchResult(self.source_id, cached, ok=False,
                               error=f"{type(exc).__name__}: {exc}")

    def _load_cache(self) -> pd.DataFrame:
        p = self._cache_path()
        if p.exists():
            try:
                return self.tidy(pd.read_parquet(p))
            except Exception:  # noqa: BLE001
                return self.tidy(None)
        return self.tidy(None)


def write_freshness(records: list[dict]) -> None:
    """Write the per-source freshness manifest atomically."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {r["source"]: r for r in records},
    }
    tmp = config.FRESHNESS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(config.FRESHNESS_FILE)
