import numpy as np
import pandas as pd

from copper_brain.model import build_model_features, train_and_validate


def _clean_copper(n=900, start="2022-01-01", trend=0.0003, seed=0):
    """Synthetic clean df: copper random walk + macro context columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(trend, 0.012, n)
    px = 4.0 * np.exp(np.cumsum(rets))
    rows = {
        ("copper", "yfinance"): px,
        ("usd_broad", "fred"): 120 + np.cumsum(rng.normal(0, 0.1, n)),
        ("real_10y", "fred"): 1.5 + np.cumsum(rng.normal(0, 0.01, n)),
        ("vix", "fred"): np.clip(18 + rng.normal(0, 3, n), 9, 60),
        ("audusd", "yfinance"): 0.66 + np.cumsum(rng.normal(0, 0.001, n)),
    }
    frames = [pd.DataFrame({"date": dates, "series": s, "source": src, "value": v})
              for (s, src), v in rows.items()]
    return pd.concat(frames, ignore_index=True)


def test_build_model_features_no_core_nan():
    X, copper = build_model_features(_clean_copper())
    core = ["copper_ret5", "copper_ret20", "copper_ret60",
            "px_vs_sma20", "px_vs_sma50", "px_vs_sma200", "copper_vol20"]
    assert not X[core].isna().any().any()
    assert len(X) == len(copper)
    assert X.index.equals(copper.index)


def test_train_skipped_when_too_few_rows():
    out = train_and_validate(_clean_copper(n=300))
    assert out["status"] == "skipped"
    assert out["promoted"] is False


def test_train_runs_and_gates_random_walk():
    # A near-random walk should NOT clear the overfit gate -> stays shadow.
    out = train_and_validate(_clean_copper(n=900, trend=0.0))
    assert out["status"] == "trained"
    assert out["promoted"] is False           # no real edge -> not promoted
    assert out["model_version"] == "v1-score"
    assert set(out["horizons"]) == {"5d", "21d"}
    for g in out["horizons"].values():
        assert "passes" in g and "pbo" in g
