# Copper Brain — agent guide

Layer-2 Dalio-style **fundamental regime model for copper**. Pulls free data,
builds features, writes **one advisory flag-file** `data/regime/copper_regime.json`.
The `hq-trading-system` webhook reads that file **off the hot path** (same
pattern as that repo's LLM `veto_flag.json`). Copper Brain **never sends orders**.

Layer 1 (TradingView/Pine intraday execution) is stubbed and deferred until
Layer-2 findings exist. This is a fresh build — NOT the old copper v1.

Full build plan + data contract: **`PLAN.md`** (read it first).

Skills: load `paper-trading-guardrails` before editing publish/webhook-contract/UI
code, `quant-research-gate` before touching the v2 gate or any model, `win-quant-env`
before installs/git/PS. Before committing guarded-path changes, run
`node ~/.claude/hooks/quant-review-scope.js copper_brain` and, on `DECISION: REVIEW`,
the `quant-reviewer` subagent.

## Hard rules (non-negotiable — mirror hq-trading-system)

1. **Data + model quality first.** No execution wiring until the score's
   walk-forward directional hit-rate is measured.
2. **Paper / advisory only.** Copper Brain writes a JSON file. Never routes an order.
3. **Overfit-strict.** v1 weights are fixed a priori. v2 ML stays *shadow* until
   it clears PBO < 0.5 AND Deflated Sharpe > 0 under purged CV + walk-forward.
4. **Fail-safe.** Stale or missing flag-file => regime = `neutral`. The webhook
   only reads a local file; it never blocks on a network call.
5. **Free data only.** 4 sources cross-check each other; divergence above
   tolerance is flagged in `freshness.json`, not silently used.

## Verification commands

- Tests (no network): `.venv\Scripts\python -m pytest`
- Tests incl. live sources: `.venv\Scripts\python -m pytest -m network`
- Pull data + rebuild + publish: `.venv\Scripts\python -m copper_brain.update`
- Retrain v2 behind overfit gate: `.venv\Scripts\python -m copper_brain.update --retrain`
- Self-update runner (logs to journal/runs): `powershell -ExecutionPolicy Bypass -File scripts\run_update.ps1`
- Register weekly (Sun 09:00) Task Scheduler job: `powershell -ExecutionPolicy Bypass -File scripts\register_weekly_update.ps1`
  (the `CopperBrainWeeklyUpdate` task is registered **elevated** — changing it needs a
  UAC relaunch; see `win-quant-env`)
- On the user saying "update": run `python -m copper_brain.update` (add `--retrain` to rerun the v2 gate).
- Dashboard (127.0.0.1 only, never expose): `.venv\Scripts\python -m copper_brain.cli ui` then open http://127.0.0.1:8077, or `powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1` (opens an Edge app-window).

## Reuse from hq-trading-system (do not reinvent)

- PBO (CSCV) + Deflated Sharpe + walk-forward PF: `analytics/research_scorecard.py`
- Flag-file + fail-safe pattern: `analytics/veto_reconcile.py` / `webhook_receiver.py`
- RandomForest meta-label workflow: `analytics/meta_label_model.py`
- Task Scheduler registration: `analytics/register_autostart.ps1`

## Environment

- Python 3.11 in `.venv` (created from the uv-managed base interpreter).
- Windows 10. Local-timezone day boundaries. No POSIX-only syscalls.
- Behind TLS interception: `pip install` may need `--use-feature=truststore`;
  retry `git add` if AV transiently locks `.git/objects`.
