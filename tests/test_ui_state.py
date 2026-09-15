import numpy as np
import pandas as pd

from copper_brain.ui_state import build_state


def _clean(n=400):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n)
    px = 4.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    rows = {
        ("copper", "yfinance"): px,
        ("usd_broad", "fred"): 120 + np.cumsum(rng.normal(0, 0.1, n)),
        ("real_10y", "fred"): 1.8 + np.cumsum(rng.normal(0, 0.01, n)),
        ("vix", "fred"): np.clip(16 + rng.normal(0, 3, n), 9, 50),
        ("audusd", "yfinance"): 0.66 + np.cumsum(rng.normal(0, 0.001, n)),
        ("lme_cu_stock", "westmetall"): np.linspace(300000, 270000, n),
        ("lme_cu_backwardation", "westmetall"): np.full(n, 30.0),
    }
    return pd.concat(
        [pd.DataFrame({"date": dates, "series": s, "source": src, "value": v})
         for (s, src), v in rows.items()], ignore_index=True)


def test_build_state_shape():
    s = build_state(_clean())
    for k in ("regime", "components", "hitrate", "v2", "sources",
              "price", "inventory", "findings", "projection", "history"):
        assert k in s
    assert len(s["components"]) == 6
    assert {c["name"] for c in s["components"]} == \
        {"trend", "inventory", "demand", "usd_rates", "risk", "news"}


def test_price_downsampled_and_aligned():
    s = build_state(_clean(n=1200))
    p = s["price"]
    assert 0 < len(p["dates"]) <= 500
    assert len(p["dates"]) == len(p["copper"]) == len(p["sma50"])


def test_inventory_present_when_westmetall_supplied():
    s = build_state(_clean())
    assert "backwardation" in s["inventory"]
    assert "stock" in s["inventory"]


def test_findings_and_projection_nonempty():
    s = build_state(_clean())
    assert len(s["findings"]) >= 1
    assert s["projection"]["v1"]["lean"] in ("bullish", "bearish", "neutral")
    assert s["projection"]["v2"]["status"] in ("shadow", "live")


def test_empty_clean_degrades_gracefully():
    s = build_state(pd.DataFrame(columns=["date", "series", "value", "source"]))
    assert s["components"] == []
    assert s["findings"]  # has a "no data" note
