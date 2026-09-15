import json

import numpy as np
import pandas as pd

from copper_brain import config
from copper_brain.publish import build_regime, publish, read_regime


def _scored(last_date, score):
    idx = pd.bdate_range(end=last_date, periods=5)
    return pd.DataFrame({
        "copper_px": np.linspace(4.0, 4.5, 5),
        "c_trend": 0.5, "c_inventory": 0.0, "c_demand": 0.0,
        "c_usd_rates": 0.0, "c_risk": 0.0, "c_news": 0.0,
        "score": [0, 0, 0, 0, score],
    }, index=idx)


def test_fresh_bullish_publishes_regime():
    today = pd.Timestamp.now().normalize()
    reg = build_regime(_scored(today, 70), missing_inputs=["news"])
    assert reg["copper_regime"] == "bullish"
    assert reg["stale"] is False
    assert reg["score"] == 70


def test_stale_data_forces_neutral():
    old = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    reg = build_regime(_scored(old, 80), missing_inputs=[])
    assert reg["stale"] is True
    assert reg["copper_regime"] == "neutral"  # fail-safe overrides bullish score


def test_publish_and_read_roundtrip(monkeypatch, tmp_path):
    # Isolate to a temp file so the test never clobbers the real flag-file.
    monkeypatch.setattr(config, "REGIME_FILE", tmp_path / "copper_regime.json")
    today = pd.Timestamp.now().normalize()
    reg = build_regime(_scored(today, -65), missing_inputs=[])
    path = publish(reg)
    on_disk = json.loads(open(path).read())
    assert on_disk["copper_regime"] == "bearish"
    back = read_regime()
    assert back["copper_regime"] == "bearish"


def test_read_regime_missing_file_is_neutral(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REGIME_FILE", tmp_path / "nope.json")
    out = read_regime()
    assert out["copper_regime"] == "neutral"
    assert out["veto"] is False


def test_empty_scored_is_neutral_fallback():
    reg = build_regime(pd.DataFrame(), missing_inputs=[])
    assert reg["copper_regime"] == "neutral"
    assert reg["stale"] is True
