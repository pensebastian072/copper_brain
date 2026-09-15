"""v2 model — RandomForest copper-direction classifier, SHADOW until it clears
the overfit gate. v1 keeps publishing meanwhile (Plan rule 3).

Workflow (mirrors hq-trading-system/analytics/meta_label_model.py, adapted for a
price-direction task with overlapping forward-return labels):

    raw features  ->  forward-direction labels (per horizon)
    ->  expanding walk-forward (gap = horizon) producing OUT-OF-SAMPLE proba
    ->  non-overlapping OOS trade PnL (position = sign(p-0.5) * forward return)
    ->  validate.evaluate_gate  (Deflated Sharpe ratio > 0 AND PBO < 0.5)
    ->  promote to v2 only if EVERY horizon clears; else stay shadow

The final model (fit on all rows) is persisted with its scorecard regardless, so
a shadow model is still tracked. predict_latest() returns p_up per horizon for
publish to surface once promoted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .features import build_wide, _pick_copper
from .labels import forward_returns
from .validate import evaluate_gate, walk_forward_splits

MODEL_DIR = config.JOURNAL_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Need a meaningful daily history before training a price model at all.
MIN_ROWS_FOR_TRAIN = 750          # ~3y of business days
N_WALK_FORWARD_FOLDS = 6
# Deflation trial count: we evaluate one RF spec across len(HORIZONS) horizons.
N_TRIALS_DEFLATION = len(config.HORIZONS_DAYS)
RF_KWARGS = dict(n_estimators=300, max_depth=5, min_samples_leaf=50,
                 class_weight="balanced", random_state=42, n_jobs=-1)


# ────────────────────────────────────────────────────────────────
# Feature matrix (raw, leakage-free — all backward-looking)
# ────────────────────────────────────────────────────────────────

def build_model_features(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, copper_px). X columns are backward-looking transforms only."""
    wide = build_wide(clean)
    if wide.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    copper = _pick_copper(wide)
    X = pd.DataFrame(index=wide.index)

    # copper trend / momentum / vol (mandatory core)
    ret1 = copper.pct_change()
    X["copper_ret5"] = copper.pct_change(5)
    X["copper_ret20"] = copper.pct_change(20)
    X["copper_ret60"] = copper.pct_change(60)
    X["px_vs_sma20"] = copper / copper.rolling(20, min_periods=20).mean() - 1
    X["px_vs_sma50"] = copper / copper.rolling(50, min_periods=30).mean() - 1
    X["px_vs_sma200"] = copper / copper.rolling(200, min_periods=120).mean() - 1
    X["copper_vol20"] = ret1.rolling(20, min_periods=20).std()
    core = list(X.columns)  # rows missing any core feature are dropped

    def add(col_src: str, name: str, fn):
        if col_src in wide and wide[col_src].notna().any():
            X[name] = fn(wide[col_src])
        else:
            X[name] = 0.0  # absent feed -> neutral, not a dropped row

    add("usd_broad__fred", "usd_ret20", lambda s: s.pct_change(20))
    add("usd_broad__fred", "usd_ret60", lambda s: s.pct_change(60))
    add("real_10y__fred", "real10y_chg20", lambda s: s.diff(20))
    add("breakeven_10y__fred", "breakeven_chg20", lambda s: s.diff(20))
    add("vix__fred", "vix_level", lambda s: s)
    add("vix__fred", "vix_chg20", lambda s: s.diff(20))
    add("audusd__yfinance", "audusd_ret20", lambda s: s.pct_change(20))
    add("lme_cu_backwardation__westmetall", "backwardation", lambda s: s)
    add("lme_cu_stock__westmetall", "stock_chg30", lambda s: s.pct_change(30))

    # Fill aux NaN (leading) with 0 = neutral; drop rows missing CORE features.
    X[[c for c in X.columns if c not in core]] = \
        X[[c for c in X.columns if c not in core]].fillna(0.0)
    X = X.dropna(subset=core)
    copper = copper.reindex(X.index)
    return X, copper


# ────────────────────────────────────────────────────────────────
# Per-horizon walk-forward OOS + gate
# ────────────────────────────────────────────────────────────────

def _oos_pnls(X: pd.DataFrame, fwd_ret: pd.Series, horizon: int) -> list[float]:
    """Expanding walk-forward; non-overlapping OOS trade PnL for the gate."""
    from sklearn.ensemble import RandomForestClassifier

    y = (fwd_ret > 0).astype(int).to_numpy()
    Xv = X.to_numpy()
    fr = fwd_ret.to_numpy()
    n = len(X)
    pnls: list[float] = []
    for tr, te in walk_forward_splits(n, N_WALK_FORWARD_FOLDS, horizon):
        ytr = y[tr]
        if np.unique(ytr).size < 2:   # need both classes to fit
            continue
        m = RandomForestClassifier(**RF_KWARGS)
        m.fit(Xv[tr], ytr)
        proba = m.predict_proba(Xv[te])[:, 1]
        # Walk through the test block in non-overlapping `horizon` steps so the
        # PnL observations don't share forward windows (needed for honest DSR).
        for j in range(0, len(te), horizon):
            p = proba[j]
            r = fr[te[j]]
            if np.isnan(r):
                continue
            pos = 1.0 if p > 0.5 else -1.0
            pnls.append(float(pos * r))
    return pnls


def train_and_validate(clean: pd.DataFrame) -> dict:
    """Train + gate v2 across all horizons. Returns a verdict dict.

    promoted=True only if EVERY horizon clears the overfit gate. Persists the
    final model + scorecard regardless (shadow tracking).
    """
    from sklearn.ensemble import RandomForestClassifier

    X, copper = build_model_features(clean)
    if len(X) < MIN_ROWS_FOR_TRAIN:
        return {"status": "skipped", "promoted": False, "n_rows": len(X),
                "reason": f"need >={MIN_ROWS_FOR_TRAIN} rows, have {len(X)}"}

    fr_all = forward_returns(copper)
    horizons: dict[str, dict] = {}
    p_up: dict[str, float] = {}
    final_models = {}
    all_pass = True

    for h in config.HORIZONS_DAYS:
        fwd = fr_all[f"fwd_ret_{h}d"].reindex(X.index)
        valid = fwd.notna()
        Xh, fwdh = X[valid], fwd[valid]
        pnls = _oos_pnls(Xh, fwdh, h)
        gate = evaluate_gate(pnls, n_trials=N_TRIALS_DEFLATION)
        horizons[f"{h}d"] = gate
        all_pass = all_pass and gate["passes"]

        # Final model on all labelled rows -> latest p_up.
        y = (fwdh > 0).astype(int).to_numpy()
        if np.unique(y).size == 2:
            m = RandomForestClassifier(**RF_KWARGS)
            m.fit(Xh.to_numpy(), y)
            final_models[h] = m
            p_up[f"{h}d"] = round(float(m.predict_proba(X.iloc[[-1]].to_numpy())[0, 1]), 4)
        else:
            p_up[f"{h}d"] = None

    verdict = {
        "status": "trained",
        "promoted": bool(all_pass),
        "model_version": config.MODEL_VERSION_V2 if all_pass else config.MODEL_VERSION_V1,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "horizons": horizons,
        "p_up": p_up,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    _persist(final_models, verdict)
    return verdict


# ────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────

def _persist(models: dict, verdict: dict) -> None:
    try:
        import joblib
        for h, m in models.items():
            joblib.dump(m, MODEL_DIR / f"copper_v2_{h}d.joblib")
    except Exception:  # noqa: BLE001
        pass
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scorecard = config.SCORECARD_DIR / f"copper_v2_{today}.json"
    scorecard.write_text(json.dumps(verdict, indent=2, default=str))
    (MODEL_DIR / "copper_v2_latest.json").write_text(
        json.dumps(verdict, indent=2, default=str))


def load_latest_verdict() -> dict | None:
    p = MODEL_DIR / "copper_v2_latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
