"""Cross-check copper price agreement across independent free sources.

Sources quote copper in different units (FRED $/mt, yfinance $/lb, World Bank
$/mt) AND at different frequencies (daily vs monthly), so comparing levels — or
even raw trailing-window % changes — is wrong: a 35-calendar-day daily move is
not comparable to a month-over-month monthly move.

So we put every source on a common monthly grid: resample copper to month-end
(last observation in the month), take month-over-month % change, and compare all
sources on the LATEST month they share. Monthly sources lag by design, so that
shared month is the most recent fully-settled one (the partial current month is
naturally excluded). A source whose MoM move diverges from the cross-source
median by more than config.CROSSCHECK_PCT_TOLERANCE is flagged in freshness.json,
never silently fed to the score.
"""
from __future__ import annotations

import pandas as pd

from . import config


def _monthly_mom(sub: pd.DataFrame) -> pd.Series:
    """Month-over-month % change of a source's copper series (indexed by month)."""
    s = (sub.set_index("date").sort_index()["value"]
            .resample("ME").last().dropna())
    return s.pct_change().dropna()


def crosscheck_copper(combined: pd.DataFrame) -> dict:
    """Compare aligned monthly copper %-moves across config.CROSSCHECK_SOURCES.

    verdict ∈ {ok, divergent, insufficient}. Reports the common month compared,
    each source's MoM move on it, the median, divergence, and any sources
    excluded for staleness.
    """
    base = {"verdict": "insufficient", "per_source": {}, "median_pct": None,
            "diverging": [], "excluded_stale": [], "compared_month": None,
            "tolerance": config.CROSSCHECK_PCT_TOLERANCE}
    if combined is None or combined.empty:
        return base

    copper = combined[combined["series"] == "copper"]
    today = pd.Timestamp.now().normalize()

    mom: dict[str, pd.Series] = {}
    excluded_stale: list[str] = []
    for src in config.CROSSCHECK_SOURCES:
        sub = copper[copper["source"] == src]
        if sub.empty:
            continue
        if (today - pd.to_datetime(sub["date"]).max()).days > config.CROSSCHECK_MAX_AGE_DAYS:
            excluded_stale.append(src)
            continue
        series = _monthly_mom(sub)
        if not series.empty:
            mom[src] = series
    base["excluded_stale"] = excluded_stale

    if len(mom) < 2:
        base["per_source"] = {k: round(float(v.iloc[-1]), 5) for k, v in mom.items()}
        return base

    # Latest month present in ALL kept sources.
    common = set.intersection(*(set(s.index) for s in mom.values()))
    if not common:
        base["per_source"] = {k: round(float(v.iloc[-1]), 5) for k, v in mom.items()}
        return base
    month = max(common)

    per_source = {src: round(float(s.loc[month]), 5) for src, s in mom.items()}
    moves = pd.Series(per_source)
    median = float(moves.median())
    tol = config.CROSSCHECK_PCT_TOLERANCE
    diverging = sorted(moves.index[(moves - median).abs() > tol].tolist())

    return {
        "verdict": "divergent" if diverging else "ok",
        "per_source": per_source,
        "median_pct": round(median, 5),
        "diverging": diverging,
        "excluded_stale": excluded_stale,
        "compared_month": month.date().isoformat(),
        "tolerance": tol,
    }
