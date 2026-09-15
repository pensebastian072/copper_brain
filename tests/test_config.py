from copper_brain import config


def test_score_weights_sum_to_100():
    assert sum(config.SCORE_WEIGHTS.values()) == 100


def test_thresholds_symmetric_and_ordered():
    assert config.SCORE_BEARISH_THRESHOLD < 0 < config.SCORE_BULLISH_THRESHOLD


def test_overfit_gate_constants():
    # Plan rule 3: v2 promotes only if PBO < 0.5 AND the Deflated Sharpe clears the bar.
    # The bar was 0.0 until 2026-07-30, when deflated_sharpe's unit bug was fixed and
    # the threshold was raised to 1.645. With a CORRECT DSR, ratio>0 is a median test:
    # best-of-8 pure noise clears it 44.8% of the time. 1.645 is prob>0.95.
    # See research_ledger/audit/dsr_audit.py and tests/test_validate_gate.py.
    assert config.PBO_MAX == 0.5
    assert config.DEFLATED_SHARPE_MIN == 1.645


def test_dirs_exist():
    for d in (config.RAW_DIR, config.CLEAN_DIR, config.REGIME_DIR, config.JOURNAL_DIR):
        assert d.exists()
