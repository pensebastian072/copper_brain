"""Self-update entrypoint — idempotent. Run any time or on a schedule.

    python -m copper_brain.update            # ingest -> features -> score -> publish
    python -m copper_brain.update --retrain  # (P3: also retrain v2 behind gate)

P1 wired the data layer; P2 wires features + v1 score + flag-file publish. The v2
retrain gate (P3) attaches at --retrain. Single cron/Task-Scheduler entrypoint.
"""
from __future__ import annotations

import argparse
import json

from . import config
from .features import compute_feature_frame
from .ingest import load_clean, run_ingest
from .model import load_latest_verdict
from .publish import build_regime, publish
from .score import score_frame, walk_forward_hitrate


def _load_freshness() -> dict | None:
    try:
        return json.loads(config.FRESHNESS_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None


def _write_run_record(regime: dict, v2: dict | None, retrain: bool) -> None:
    """Append a per-run summary to journal/runs/ for an audit trail."""
    from datetime import datetime, timezone
    rec = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "retrain": retrain,
        "regime": regime.get("copper_regime"),
        "score": regime.get("score"),
        "stale": regime.get("stale"),
        "model_version": regime.get("model_version"),
        "v2_promoted": bool(v2 and v2.get("promoted")),
        "data_through": regime.get("data_through"),
        "crosscheck": regime.get("crosscheck"),
        "missing_inputs": regime.get("missing_inputs"),
        "source_health": regime.get("source_health"),
    }
    stamp = rec["ran_at"].replace(":", "").replace("-", "").split(".")[0]
    try:
        (config.RUNS_DIR / f"run_{stamp}.json").write_text(
            json.dumps(rec, indent=2, default=str))
    except Exception:  # noqa: BLE001
        pass


def run_update(retrain: bool = False, verbose: bool = True,
               skip_ingest: bool = False) -> dict:
    """Full pipeline: ingest -> features -> score -> [retrain v2] -> publish."""
    if skip_ingest:
        clean = load_clean()
    else:
        clean, _records = run_ingest(verbose=verbose)

    features, missing = compute_feature_frame(clean)
    scored = score_frame(features)
    freshness = _load_freshness()

    # v2 (P3): retrain + gate on demand; promote ONLY if it clears the overfit
    # gate, else stay v1 (shadow). A previously-promoted model also lifts later
    # runs without retraining.
    v2 = _train_v2(clean, verbose) if retrain else load_latest_verdict()
    promoted = bool(v2 and v2.get("promoted"))
    model_version = config.MODEL_VERSION_V2 if promoted else config.MODEL_VERSION_V1

    regime = build_regime(scored, missing, freshness=freshness,
                          model_version=model_version)
    if promoted and v2.get("p_up"):
        regime["p_up_5d"] = v2["p_up"].get("5d")
        regime["p_up_21d"] = v2["p_up"].get("21d")
    path = publish(regime)
    _write_run_record(regime, v2, retrain)

    if verbose:
        hr = walk_forward_hitrate(scored)
        print(f"\nregime: {regime['copper_regime'].upper()}  "
              f"score={regime['score']}  stale={regime['stale']}  "
              f"drivers={regime['main_drivers']}")
        print(f"missing_inputs: {missing}")
        if hr:
            print(f"v1 walk-forward hit-rate: "
                  f"5d={hr.get('hit_rate_5d')} (n={hr.get('n_5d')})  "
                  f"21d={hr.get('hit_rate_21d')} (n={hr.get('n_21d')})")
        print(f"model_version published: {model_version} "
              f"(v2 promoted={promoted})")
        print(f"published -> {path}")
    return regime


def _train_v2(clean, verbose: bool) -> dict:
    """Train + gate the v2 model; print the verdict."""
    from .model import train_and_validate
    v2 = train_and_validate(clean)
    if verbose:
        if v2.get("status") == "skipped":
            print(f"\nv2 retrain SKIPPED: {v2.get('reason')}")
        else:
            print(f"\nv2 retrain: promoted={v2['promoted']} "
                  f"(version stays {v2['model_version']})")
            for h, g in v2.get("horizons", {}).items():
                dsr = g.get("deflated_sharpe") or {}
                print(f"  {h}: gate_passes={g['passes']} "
                      f"dsr_ratio={dsr.get('ratio')} pbo={g['pbo']} "
                      f"n={g['n_trades']} pf={g['profit_factor']} "
                      f"-> {g['reasons']}")
    return v2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="copper_brain.update")
    ap.add_argument("--retrain", action="store_true",
                    help="(P3) retrain v2 walk-forward behind the PBO/DSR gate")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse the last clean parquet (no network pull)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    regime = run_update(retrain=args.retrain, verbose=not args.quiet,
                        skip_ingest=args.skip_ingest)
    return 0 if regime.get("copper_regime") else 1


if __name__ == "__main__":
    raise SystemExit(main())
