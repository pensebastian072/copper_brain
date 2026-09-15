from copper_brain.webhook_gate import HQ_READER_STUB, evaluate_alert


def _regime(regime="bullish", score=70, stale=False):
    return {"copper_regime": regime, "score": score, "stale": stale,
            "model_version": "v1-score"}


def test_non_copper_alert_passes_through():
    d = evaluate_alert({"asset": "GOLD", "direction": "long"}, regime=_regime())
    assert d["applies"] is False
    assert d["confirm"] is True  # don't block non-copper


def test_long_confirmed_when_bullish_and_score_clears():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "long", "min_copper_score": 60},
        regime=_regime("bullish", 72))
    assert d["confirm"] is True
    assert d["applies"] is True


def test_long_rejected_when_score_below_threshold():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "long", "min_copper_score": 60},
        regime=_regime("bullish", 40))
    assert d["confirm"] is False
    assert "not met" in d["reason"]


def test_long_rejected_when_regime_wrong():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "long", "min_copper_score": 0},
        regime=_regime("bearish", -70))
    assert d["confirm"] is False
    assert "!= required bullish" in d["reason"]


def test_short_confirmed_when_bearish_and_score_negative_enough():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "short", "min_copper_score": 60},
        regime=_regime("bearish", -75))
    assert d["confirm"] is True


def test_short_rejected_when_score_not_negative_enough():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "short", "min_copper_score": 60},
        regime=_regime("bearish", -40))
    assert d["confirm"] is False


def test_stale_regime_never_confirms():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "long", "min_copper_score": 0},
        regime=_regime("bullish", 90, stale=True))
    assert d["confirm"] is False
    assert "stale" in d["reason"]


def test_explicit_required_regime_overrides_direction():
    d = evaluate_alert(
        {"asset": "COPPER", "direction": "long",
         "required_model_regime": "bearish", "min_copper_score": 60},
        regime=_regime("bearish", -80))
    assert d["confirm"] is True  # required=bearish + short-style score check


def test_reader_stub_is_self_contained_text():
    # The pasteable stub must not reference copper_brain (separate-repo safe).
    assert "import copper_brain" not in HQ_READER_STUB
    assert "read_copper_regime" in HQ_READER_STUB
