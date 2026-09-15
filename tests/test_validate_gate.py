"""Calibration tests for the overfit gate. These exist because it was once wrong.

Until 2026-07-30 `deflated_sharpe` subtracted the dimensionless BLdP bracket directly
from a per-observation Sharpe and then scaled the difference by sqrt(T-1)/denom,
inflating the penalty ~sqrt(T-1)-fold. It was a CONSTANT FAIL: prob 0.0000 on
best-of-N pure noise, and 0.000 pass rate even at a planted true per-observation SR of
0.30 (~4.8 annualised).

That matters here specifically: copper v2's recorded verdict was "DSR ratio -18/-10,
OOS PF 0.96/0.77 -> stayed shadow", and it was cited across this box as an example of
a correct gate failing a first-pass model. The PF part of that still stands. The DSR
part carries no information and must be re-run rather than quoted.

The bug hid because the n_trials=1 path is correct and the tests used it, so these
deliberately exercise n_trials > 1 and check the property that DEFINES deflation:
it removes exactly the advantage of having searched N times, so the best of N
pure-noise trials must score ~0.5, not ~0.0.

The same tests live in macro_gpu_lab and hq-trading-system, which hold the other two
copies of this math.
"""
from __future__ import annotations

import numpy as np
import pytest

from copper_brain import config
from copper_brain.validate import (
    deflated_sharpe,
    expected_max_sr_units,
    pbo_cscv,
)

T = 210


def _best_of_n_noise(rng, n_trials, t=T, planted_sr=0.0):
    trials = rng.standard_normal((n_trials, t)) * 0.01
    if planted_sr:
        trials[0] += planted_sr * 0.01
    srs = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
    return trials[int(np.argmax(srs))]


@pytest.mark.parametrize("n_trials", [2, 8, 42])
def test_deflated_sharpe_null_calibration(n_trials):
    """THE test. Best-of-N on pure noise must average ~0.5, never ~0.0."""
    rng = np.random.default_rng(11)
    probs = [deflated_sharpe(_best_of_n_noise(rng, n_trials), n_trials=n_trials)["prob"]
             for _ in range(400)]
    mean = float(np.mean(probs))
    assert abs(mean - 0.5) < 0.10, (
        f"n_trials={n_trials}: mean DSR probability {mean:.4f} on PURE NOISE. A correct "
        "deflated Sharpe returns ~0.50 here. Near 0.0 means the penalty is inflated "
        "(the pre-2026-07-30 bug); near 1.0 means it is missing."
    )


def test_deflated_sharpe_has_power():
    rng = np.random.default_rng(17)
    passes = [deflated_sharpe(_best_of_n_noise(rng, 8, planted_sr=0.30),
                              n_trials=8)["ratio"] > 0
              for _ in range(300)]
    assert float(np.mean(passes)) > 0.90


def test_power_rises_with_the_true_edge():
    rng = np.random.default_rng(23)
    rates = []
    for planted in (0.0, 0.15, 0.30):
        passes = [deflated_sharpe(_best_of_n_noise(rng, 8, planted_sr=planted),
                                  n_trials=8)["ratio"] > 0
                  for _ in range(200)]
        rates.append(float(np.mean(passes)))
    assert rates == sorted(rates), rates


def test_bracket_is_dimensionless_and_monotonic():
    assert expected_max_sr_units(1) == 0.0
    vals = [expected_max_sr_units(n) for n in (2, 8, 42, 132)]
    assert vals == sorted(vals)
    assert all(0 < v < 4 for v in vals)


def test_ratio_equals_psr_z_minus_bracket():
    rng = np.random.default_rng(9)
    pnl = rng.standard_normal(250) * 0.01 + 0.0006
    for n in (2, 8, 42):
        d = deflated_sharpe(pnl, n_trials=n)
        assert d["ratio"] == pytest.approx(d["psr_z"] - d["bracket"], abs=2e-4)


def test_degenerate_inputs_return_none():
    assert deflated_sharpe([0.1] * 4, n_trials=2) is None
    # constant series: std is float noise (~1.4e-17), which used to give sr ~ 7e15
    assert deflated_sharpe([0.1] * 20, n_trials=2) is None
    assert deflated_sharpe([0.1] * 19 + [float("nan")], n_trials=2) is None


def test_gate_threshold_is_the_significance_bar():
    """Raised 0.0 -> 1.645 on 2026-07-30: ratio>0 is a median test, not a test."""
    assert config.DEFLATED_SHARPE_MIN == pytest.approx(1.645)
    assert config.PBO_MAX == 0.5


def test_pbo_is_unchanged_by_the_dsr_fix():
    rng = np.random.default_rng(31)
    p = pbo_cscv(rng.standard_normal(400) * 0.01)
    assert p is None or 0.0 <= p <= 1.0
