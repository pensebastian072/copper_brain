import numpy as np
import pandas as pd

from copper_brain.features import compute_feature_frame


def _clean(series_map, n=200, start="2025-01-01"):
    """Build a long clean df. series_map: {(series, source): array-like length n}."""
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for (series, source), vals in series_map.items():
        frames.append(pd.DataFrame({
            "date": dates, "series": series, "source": source, "value": vals,
        }))
    return pd.concat(frames, ignore_index=True)


def test_bullish_inputs_yield_positive_components_and_score():
    n = 200
    rising = np.linspace(4.0, 5.0, n)               # copper up 25%
    stock_down = np.linspace(300000, 250000, n)     # inventory drawing
    backward_pos = np.full(n, 60.0)                 # cash > 3m => tight
    aud_up = np.linspace(0.62, 0.70, n)
    clean = _clean({
        ("copper", "yfinance"): rising,
        ("lme_cu_stock", "westmetall"): stock_down,
        ("lme_cu_backwardation", "westmetall"): backward_pos,
        ("audusd", "yfinance"): aud_up,
        ("usd_broad", "fred"): np.full(n, 120.0),
        ("real_10y", "fred"): np.full(n, 2.0),
        ("vix", "fred"): np.full(n, 14.0),          # < 18 center => risk-on
    }, n=n)
    feats, missing = compute_feature_frame(clean)
    last = feats.iloc[-1]
    assert last["c_trend"] > 0
    assert last["c_inventory"] > 0
    assert last["c_demand"] > 0
    assert last["c_risk"] > 0
    assert "fxi" in missing and "news" in missing


def test_missing_sources_degrade_to_zero_not_crash():
    n = 120
    clean = _clean({("copper", "yfinance"): np.linspace(4.0, 4.2, n)}, n=n)
    feats, missing = compute_feature_frame(clean)
    last = feats.iloc[-1]
    # Only copper present: trend computed, all other components neutral 0.
    assert last["c_inventory"] == 0.0
    assert last["c_demand"] == 0.0
    assert last["c_usd_rates"] == 0.0
    assert last["c_news"] == 0.0
    for need in ("lme_cu_stock", "audusd", "usd_broad", "vix"):
        assert need in missing


def test_empty_clean_returns_empty():
    feats, missing = compute_feature_frame(pd.DataFrame(
        columns=["date", "series", "value", "source"]))
    assert feats.empty
    assert missing == ["all"]


def test_components_bounded():
    n = 200
    # extreme moves should still squash into [-1, 1]
    spike = np.concatenate([np.full(n - 1, 4.0), [40.0]])
    clean = _clean({("copper", "yfinance"): spike}, n=n)
    feats, _ = compute_feature_frame(clean)
    assert feats["c_trend"].abs().max() <= 1.0
