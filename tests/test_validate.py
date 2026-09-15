import numpy as np

from copper_brain import config
from copper_brain.validate import (deflated_sharpe, evaluate_gate, profit_factor,
                                   purged_kfold, pbo_cscv, sharpe,
                                   walk_forward_splits)


def test_profit_factor_and_sharpe():
    assert profit_factor([1, 1, -1]) == 2.0
    assert profit_factor([1, 1]) == float("inf")
    assert sharpe([1, 1, 1, 1]) is None  # zero variance
    assert sharpe([1.0]) is None


def test_pbo_needs_enough_data():
    assert pbo_cscv([1, -1] * 5) is None  # < n_groups*4
    pbo = pbo_cscv(list(np.random.default_rng(0).normal(0, 1, 200)))
    assert pbo is None or 0.0 <= pbo <= 1.0


def test_deflated_sharpe_positive_vs_noise():
    rng = np.random.default_rng(1)
    strong = rng.normal(0.5, 1.0, 300)      # clear positive edge
    noise = rng.normal(0.0, 1.0, 300)       # no edge
    ds_strong = deflated_sharpe(strong, n_trials=1)
    ds_noise = deflated_sharpe(noise, n_trials=1)
    assert ds_strong["ratio"] > 0 and ds_strong["prob"] > 0.9
    assert ds_noise["ratio"] < ds_strong["ratio"]


def test_deflated_sharpe_more_trials_lowers_ratio():
    # More candidate trials => higher expected-max under H0 => stricter.
    rng = np.random.default_rng(7)
    series = rng.normal(0.5, 1.0, 300)
    assert deflated_sharpe(series, n_trials=10)["ratio"] < \
        deflated_sharpe(series, n_trials=1)["ratio"]


def test_deflated_sharpe_insufficient():
    assert deflated_sharpe([1, 2, 3]) is None


def test_purged_kfold_no_leakage():
    n, h, n_splits = 100, 5, 5
    embargo_pct = 0.01
    embargo = int(n * embargo_pct)
    for tr, te in purged_kfold(n, n_splits, label_horizon=h, embargo_pct=embargo_pct):
        ts, ted = te.min(), te.max() + 1
        # no training index has a label window reaching the test block,
        # and none falls inside the embargo band after it
        assert ((tr < ts - h) | (tr >= ted + embargo)).all()


def test_walk_forward_gap_and_expanding():
    folds = list(walk_forward_splits(120, n_folds=4, label_horizon=5))
    assert folds
    prev_train = 0
    for tr, te in folds:
        assert tr.min() == 0                 # expanding from the start
        assert te.min() - tr.max() >= 5      # >= horizon gap
        assert tr.max() >= prev_train        # train window grows
        prev_train = tr.max()


def test_gate_passes_strong_fails_losing():
    rng = np.random.default_rng(2)
    good = rng.normal(0.4, 1.0, 200)
    bad = rng.normal(-0.3, 1.0, 200)
    assert evaluate_gate(good, n_trials=1)["passes"] is True
    assert evaluate_gate(bad, n_trials=1)["passes"] is False


def test_gate_uses_config_thresholds():
    g = evaluate_gate(list(np.random.default_rng(3).normal(0, 1, 100)), n_trials=2)
    assert g["thresholds"]["pbo_max"] == config.PBO_MAX
    assert g["thresholds"]["deflated_sharpe_min"] == config.DEFLATED_SHARPE_MIN
