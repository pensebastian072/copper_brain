import numpy as np
import pandas as pd

from copper_brain import config
from copper_brain.score import (classify, main_drivers, score_frame,
                                walk_forward_hitrate)


def _feat_row(**comps):
    base = {f"c_{c}": 0.0 for c in config.SCORE_WEIGHTS}
    base.update({f"c_{k}": v for k, v in comps.items()})
    return base


def test_classify_thresholds():
    assert classify(config.SCORE_BULLISH_THRESHOLD) == "bullish"
    assert classify(config.SCORE_BEARISH_THRESHOLD) == "bearish"
    assert classify(0) == "neutral"
    assert classify(59) == "neutral"


def test_score_frame_weighted_sum():
    # all components +1 -> score = sum(weights) = 100
    df = pd.DataFrame([_feat_row(trend=1, inventory=1, demand=1,
                                 usd_rates=1, risk=1, news=1)])
    out = score_frame(df)
    assert out["score"].iloc[0] == 100

    # only trend +1 -> score = trend weight (30)
    df2 = pd.DataFrame([_feat_row(trend=1)])
    assert score_frame(df2)["score"].iloc[0] == config.SCORE_WEIGHTS["trend"]


def test_score_clipped():
    df = pd.DataFrame([_feat_row(trend=5, inventory=5, demand=5,
                                 usd_rates=5, risk=5, news=5)])
    assert score_frame(df)["score"].iloc[0] == 100


def test_main_drivers_signed_and_ranked():
    row = pd.Series(_feat_row(trend=0.9, usd_rates=-0.8, inventory=0.1))
    drivers = main_drivers(row, top_n=2)
    assert drivers[0] == "trend_positive"      # biggest |contribution|
    assert "usd_rates_headwind" in drivers


def test_walk_forward_hitrate_perfect_trend():
    # monotonically rising copper + always-positive score -> ~100% hit-rate
    n = 120
    px = pd.Series(np.linspace(4.0, 6.0, n),
                   index=pd.bdate_range("2025-01-01", periods=n))
    scored = pd.DataFrame({"copper_px": px, "score": np.full(n, 50.0)})
    hr = walk_forward_hitrate(scored)
    assert hr["hit_rate_5d"] == 1.0
    assert hr["n_5d"] > 0
