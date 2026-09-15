# Copper Brain — Build Plan (Layer 2)

Standalone Dalio-style fundamental regime model for copper. Outputs **one regime
flag-file**. The `hq-trading-system` webhook reads it off the hot path (same
pattern as the existing LLM `veto_flag.json`). Copper Brain **never sends
orders** — advisory regime only.

Layer 1 (TradingView/Pine intraday execution) is stubbed and deferred until
Layer-2 findings exist. Does **not** build on the old copper v1 — fresh build.

Status: **PLANNED** (2026-06-21). Build starts P0 + P1.

---

## The question the model answers

> "Based on supply, demand, inventories, macro, and trend, should copper be
> bullish, bearish, or neutral over the next 5–21 trading days?"

---

## Core principles (non-negotiable — mirror hq-trading-system hard rules)

1. **Data + model quality first.** No execution wiring until the score's
   out-of-sample (walk-forward) directional hit-rate is measured.
2. **Paper / advisory only.** Copper Brain writes a JSON file. Nothing else.
   Never routes an order.
3. **Overfit-strict.** v1 weights are fixed a priori (NOT tuned on the same
   data). v2 ML stays *shadow* until it clears PBO < 0.5 AND Deflated Sharpe > 0
   under purged CV + walk-forward.
4. **Fail-safe.** Stale or missing flag-file => regime = `neutral`, never
   confirms a trade. The webhook never blocks on a hot-path network call — it
   only reads a local file.
5. **Free data only.** 4 sources cross-check each other; divergence above
   tolerance is flagged, not silently used.

---

## Project layout

```
copper_brain/
  pyproject.toml  requirements.txt  .env.example(FRED_API_KEY)  CLAUDE.md  README.md
  copper_brain/
    config.py            # paths, score weights, thresholds, source registry
    sources/
      base.py            # Source ABC: fetch()->tidy df, cache, date-stamp
      fred.py stooq.py westmetall.py yfinance_src.py worldbank.py
      usgs_icsg.py       # loader for manual PDF/CSV drops in data/manual/
    ingest.py            # orchestrate sources -> data/clean parquet + freshness manifest
    crosscheck.py        # reconcile overlapping copper price series, flag divergence
    features.py          # trend/momentum, inventory tightness, curve(backwardation),
                         #   demand proxies, USD/rates, risk appetite, news-score stub
    labels.py            # forward 5d / 21d direction labels
    score.py             # v1 transparent weighted Copper Score -> regime
    model.py             # v2 RF/XGB classifier (p_up_5d, p_up_21d), walk-forward
    validate.py          # PBO (CSCV), Deflated Sharpe, purged+embargo CV, walk-forward PF
    publish.py           # atomic write data/regime/copper_regime.json + stale guard
    update.py            # entrypoint: ingest->features->score->[retrain]->publish
    cli.py
  data/ raw/ clean/ manual/ regime/copper_regime.json
  journal/ runs/ scorecards/ freshness.json
  scripts/register_weekly_update.ps1   # Task Scheduler, mirrors hq register_autostart.ps1
  tests/  (one per module)
```

---

## Data feeds (all 4 cross-check each other)

| Source | Key? | Gives |
|---|---|---|
| **FRED** | free key | copper price (PCOPPUSDM), broad USD (DTWEXBGS), real 10y (DFII10), breakeven (T10YIE), VIX (VIXCLS) |
| **Stooq** | none | daily continuous copper futures HG.F, dollar index DX.F, SPY/EEM/FXI |
| **Westmetall** | none (scrape) | LME copper warehouse stock + cash vs 3-month spread = backwardation / inventory signal (only free inventory+curve source) |
| **World Bank Pink Sheet** | none | monthly commodity price cross-check |
| **yfinance** | none (gray ToS) | HG=F, AUDUSD=X, ^TNX — fallback + 4th cross-check |
| USGS / ICSG | manual | production/usage drops into `data/manual/`, loaded by `usgs_icsg.py` |

`crosscheck.py`: align copper price across FRED-monthly / Stooq-daily /
Westmetall-LME / yfinance, flag any series diverging beyond tolerance into a
quality alarm in `freshness.json`.

---

## Copper Score (v1, transparent, fixed weights)

```
Copper Score =
    30% trend / momentum
  + 20% inventory tightness
  + 20% industrial demand
  + 15% USD / rates macro
  + 10% risk appetite
  +  5% news / disruption score
```

Classify:

- `score >= +60`  => bullish regime
- `score <= -60`  => bearish regime
- otherwise       => neutral / no-trade regime

---

## Regime flag-file contract (what the webhook reads)

`data/regime/copper_regime.json`:

```json
{
  "asset": "COPPER", "as_of": "2026-06-21T13:00:00Z", "data_through": "2026-06-20",
  "copper_regime": "bullish|bearish|neutral", "score": 72,
  "p_up_5d": 0.61, "p_up_21d": 0.66,
  "main_drivers": ["trend_positive","inventories_tight","dxy_not_confirming_downside"],
  "veto": false, "stale": false,
  "model_version": "v1-score|v2-rf",
  "source_health": {"fred":"ok","stooq":"ok","westmetall":"stale_3d","yfinance":"ok"}
}
```

Webhook gate example (matches existing hq architecture):

```json
{
  "asset": "COPPER", "timeframe": "15m", "setup": "macro_confirmed_breakout",
  "direction": "long", "tv_signal": true,
  "required_model_regime": "bullish", "min_copper_score": 60
}
```

---

## Self-update mechanism

- `python -m copper_brain.update` — incremental, **idempotent** (dedupe by
  date), rebuilds features, recomputes v1 score, republishes flag-file. Safe to
  run any time.
- `python -m copper_brain.update --retrain` — also retrains v2 walk-forward and
  reruns the PBO / Deflated-Sharpe gate. **Promotes the new model only if it
  clears the gate**; otherwise keeps the prior model.
- **Weekly:** `scripts/register_weekly_update.ps1` registers a Sunday Windows
  Task Scheduler job.
- **On command:** "update" / "pull newest data and rerun" => run the same
  entrypoint. `journal/freshness.json` tracks last pull per source.

---

## Phases (quality-ordered)

- **P0** — Scaffold, `config.py`, Source ABC, test harness, `CLAUDE.md` with
  hard rules + paper map.
- **P1 — DATA LAYER FIRST.** 4 sources + `ingest` + `crosscheck` + freshness
  manifest. Verify every source pulls; prove copper price agrees across all 4.
- **P2** — Features + v1 transparent weighted score + publish flag-file.
  End-to-end regime output. Measure the score's walk-forward 5d/21d directional
  hit-rate (no fitting — fixed weights).
- **P3** — Labels + v2 RF/XGB classifier. Purged K-fold + embargo (de Prado),
  walk-forward, PBO + Deflated Sharpe gate. **Shadow** until it clears; v1 stays
  published meanwhile.
- **P4** — Self-update automation (weekly Task Scheduler + manual + retrain
  path).
- **P5** — Webhook integration in hq-trading-system: copper alert reads
  `copper_regime.json`, fail-safe `neutral`. (Touches Layer 1 => deferred; ship
  the contract + reader stub only.)
- **P6 (deferred)** — news/disruption LLM sidecar (reuse `news_poller`); RL
  sizing. Explicitly later, per the RL survey paper.

---

## Why not jump straight to RL

Supervised learning predicts price error; RL can optimize a financial objective
directly but adds friction modeling, non-stationarity, and risk-management
complexity. Order: rule-based score -> feature model -> RandomForest/XGBoost
classifier -> walk-forward validation -> only then test RL for sizing / trailing
exits.

## The main danger: overfitting

Testing many variations on the same history breeds strategies that look great
in-sample and fail out-of-sample; standard holdout may not catch it. Controls:
walk-forward testing; a small set of core hypotheses; Deflated Sharpe /
Probability of Backtest Overfitting checks; separate "model score" testing from
"execution strategy" testing; paper trading before any real execution.

---

## Papers used (already in `C:\Users\<your-user>\Downloads`)

- `2_The_Backtest_Overfitting_Demonstration.pdf` => `validate.py` PBO /
  Deflated-Sharpe (reuse the math already in
  `hq-trading-system/analytics/research_scorecard.py`).
- `Learning_low_frequency_temporal_patterns.pdf` (RandomForest trading agent)
  => `model.py` workflow: preprocess -> time-series segment -> dim-reduce -> RF
  -> eval; features = OHLC + fundamentals + TA + directional-change + seasonality.
- `Reinforcement_Learning_in_Quantitative_T.pdf` => documents *why RL is
  deferred*.

**Open gap:** the "commodity-momentum returns linked to low-inventory states"
cite has no matching PDF in Downloads yet. The `inventory_tightness x momentum`
interaction is encoded as a core hypothesis regardless. Locate that paper to
wire it into the feature doc.

---

## Reused from hq-trading-system (do not reinvent)

- PBO (CSCV) + Deflated Sharpe + walk-forward PF math: `analytics/research_scorecard.py`
- Flag-file + fail-safe pattern: `analytics/veto_reconcile.py` / `webhook_receiver.py` veto flag
- RandomForest meta-label workflow reference: `analytics/meta_label_model.py`
- Task Scheduler registration pattern: `analytics/register_autostart.ps1`

## Environment

- Python 3.11 (venv per project, like the other two trading repos).
- Windows 10, local-timezone day boundaries, no POSIX-only syscalls.
- GitHub: `github.com/pensebastian072/copper_brain`.
