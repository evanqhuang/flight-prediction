# Flight Pricing LoRA on TimesFM — Design

**Date:** 2026-05-03
**Status:** Draft, pending implementation plan

## Goal

Train a LoRA adapter on top of TimesFM-2.5 using historical flight pricing data from `unfare_db` to produce a forecast model whose output drives a **buy-now vs. wait** signal for the existing alert system. Research first, then graduate the trained adapter into a batch scorer that writes predictions back to Postgres.

## Background

The `unfare` system scrapes flight prices into a TimescaleDB hypertable (`flight_prices`) at roughly hourly cadence per (route, departure). Existing alerting is rule-based and reactive. A learned forecast over the next 7 days enables proactive "wait — likely to drop" recommendations. TimesFM is a strong forecasting foundation model with first-class HuggingFace + PEFT support, making LoRA fine-tuning the natural path.

## Data overview (as of 2026-05-03)

- **Rows:** 55.97M in `flight_prices`, 3-month coverage (2026-02-03 → 2026-05-03).
- **Routes:** 246 actively scraped (of 406 total active in `routes`).
- **Forward booking horizon:** ~75 days (departures up to 2026-07-19).
- **Per-(route, departure) cadence:** 400–1000 snapshots per week, covering ~42–46 distinct hours → effectively hourly.
- **Storage:** TimescaleDB hypertable; chunked by `scraped_at`.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction target | Buy/wait signal derived from price-trajectory forecast | Forecast first, threshold downstream; clean separation |
| Series unit | Cheapest economy quote per `(route_id, departure_datetime)` | Matches "should I buy this fare" framing |
| Resampling cadence | Hourly | Preserves intraday volatility / drop signal |
| Forecast horizon | 168 hours (7 days) | Window where buy/wait has a meaningful answer |
| Hardware | Single 24GB consumer GPU | Sufficient for LoRA on TimesFM-2.5-200M |
| Covariates | `days_to_departure`, `day_of_week` | Known causal drivers, fully knowable into the future |
| Deployment | Research → batch scorer (no real-time API in v1) | Don't build serving until model proves out |
| Success criteria | Forecast metrics + decision metrics, both gated | Vibes-free promotion criteria |
| LoRA strategy | Single global adapter, staged training | Cheap-to-debug, attribution per stage |
| Code/runtime split | Code on Mac, synced via Mutagen, training on server | Best dev UX without shipping data over the network |
| Price normalization | Log-transform during preprocessing | Equal gradient contribution across fare scales |

## Architecture

Five-unit research repo at `~/flight-prediction/`:

```
flight-prediction/
├── data/                      # cached parquet datasets (gitignored, server-only)
├── artifacts/                 # checkpoints, LoRA adapters, eval reports (server-only)
├── src/flight_lora/
│   ├── db.py                  # Postgres reader (read-only, parameterized)
│   ├── dataset.py             # raw rows → log-transformed hourly series + splits
│   ├── covariates.py          # builds days_to_departure, day_of_week aligned to series
│   ├── train.py               # the staged training runs
│   ├── eval.py                # forecast + decision-quality metrics
│   └── infer.py               # batch scorer (v2)
├── notebooks/                 # exploration, sanity checks, eval plots
├── scripts/
│   ├── build_dataset.py       # one-shot: pull from db → write parquet
│   ├── run_stage.py           # CLI: --stage zero_shot|lora_uni|lora_cov
│   ├── run_batch_scorer.py    # v2 batch inference
│   └── remote_train.sh        # ssh + tmux wrapper around run_stage.py
├── tests/                     # unit tests
├── Makefile                   # train-uni, train-cov, pull-artifacts, etc.
├── .env.local                 # DB_HOST=192.168.93.107  (Mac side)
├── .env.server                # DB_HOST=localhost        (server side, mutagen-excluded)
├── .mutagen.yml               # sync config
├── pyproject.toml
└── uv.lock
```

**Module boundaries:**

- **`db.py`** — given a config + window, returns a DataFrame of raw `flight_prices` rows. Postgres is its only dependency; trivially mockable.
- **`dataset.py`** — pure transform from raw rows to (context, target, covariates) tensors at a regular hourly grid. Owns the route-level train/val/test split. Applies `log(price)` before yielding.
- **`covariates.py`** — pure function: given a series, emits aligned `days_to_departure` and `day_of_week` arrays of length `context + horizon`.
- **`train.py`** — given a stage config, loads model + LoRA, trains, saves adapter. Knows nothing about the DB.
- **`eval.py`** — given predictions and ground truth, computes forecast metrics in dollar-space (post-`exp`) and decision metrics. Reusable across all stages.

**Data flow:**

```
Postgres → db.py → raw DataFrame
                       ↓
               build_dataset.py → cached parquet (per-series, log-transformed)
                       ↓
                   dataset.py → (context, target, covariates) tensors
                       ↓
                    train.py → LoRA adapter checkpoint
                       ↓
                     eval.py → metrics + gate decision
```

## Remote development setup (Mutagen)

Code repo lives on the Mac, mirrored to the server (which holds the DB and GPU) via a continuous Mutagen sync session.

- **In sync:** `src/`, `scripts/`, `tests/`, `notebooks/`, `Makefile`, `pyproject.toml`, `uv.lock`.
- **Server-only (excluded from Mac):** `data/` (parquet cache), `artifacts/` (checkpoints), `.env.server`, `.venv/`.
- **Mac-only (excluded from server):** `.env.local`.

Training is invoked with `make train-uni` / `make train-cov`, which `ssh`s to the server, starts (or attaches to) a per-stage tmux session, and runs `scripts/run_stage.py` inside it. The training script knows nothing about being driven remotely. `make pull-artifacts` rsyncs eval reports back to the Mac for inspection.

## Data pipeline

**Series definition:**

- One series = `(route_id, departure_datetime)`, value = `min(price)` per hourly bucket over economy snapshots.
- Row-level filters: `cabin_class = 'economy'`; drop rows with `scraped_at >= departure_datetime`. `is_mistake_fare` rows are kept and treated as ordinary observations (the flag is not meaningfully populated).

**Hourly grid:**

- Build a regular hourly index per series from `min(scraped_at)` to `min(max(scraped_at), departure_datetime)`.
- Bucket aggregation: `min(price)` per hour.
- Missing hours: short gaps (≤6h) → forward-fill. Longer gaps → split the series at the gap (TimesFM expects continuous input).

**Length filter:**

- A training pair needs `context_len + horizon_len = 512 + 168 = 680` contiguous hours (~28 days).
- Series shorter than that are skipped.

**Normalization:**

- Apply `np.log(price)` after hourly aggregation, before yielding tensors. All training is in log-space.
- TimesFM's built-in `normalize_inputs=True` runs additionally for per-context-window standardization.
- Forecasts are `exp`-transformed back to dollar-space before computing reportable metrics and decision thresholds.

**Covariate alignment** (`covariates.py`):

- `days_to_departure` — float, dynamic numerical, decreases monotonically over the series, defined for `context + horizon` steps.
- `day_of_week` — int 0–6, dynamic categorical, defined for `context + horizon` steps.
- Both are calendar-derivable, so they are knowable into the forecast horizon — exactly what `forecast_with_covariates` requires.

**Splits:**

- **Primary split: by `route_id`**, 70/15/15 → ~172/37/37 routes. Each route lives in exactly one of train/val/test. This measures generalization to unseen routes.
- **Secondary temporal hold-out** within train routes: hold out the most recent 7 days of scrape data for early-stopping signal during training.
- **Random split of (route, departure) pairs is forbidden** — same route in train and test leaks identity.

**Training-pair sampling:**

- Random window sampling per the official TimesFM `finetune_lora.py` recipe: pick a qualifying series, pick a random valid start, take 512 hours context + 168 hours target.
- ~100k sampled windows per epoch (tunable; range 50k–500k).

**Caching:**

- `scripts/build_dataset.py` runs once on the server → writes `data/series.parquet` (one row per series with the hourly array + metadata) and `data/splits.json` (route_id → split label).
- Idempotent given a `--cutoff` timestamp. ~hundreds of MB on disk; loadable into RAM.

## Training (staged)

Three sequential runs, each gated on the previous. All use `google/timesfm-2.5-200m-pytorch` as the base. All log to `artifacts/runs/<stage>/`.

**Stage 0 — Zero-shot baseline (no training):**

- Load base TimesFM, run inference on val and test sets.
- Records: per-series MAPE/SMAPE at 168h horizon, plus naive-last-value and seasonal-naive(24h, 168h) baselines.
- Output: `artifacts/runs/zero_shot/metrics.json`. No gate — establishes the bar.

**Stage 1 — Univariate LoRA:**

- `TimesFm2_5ModelForPrediction` + `peft.LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.05, bias="none")`.
- ~0.5% of params trainable (~1M params). Fits in 24GB with bf16 and reasonable batch size.
- Loss: pinball loss against quantile head outputs (q ∈ {0.1, 0.5, 0.9}). Falls back to MSE if quantile head is not used.
- Optimizer: AdamW, lr 5e-5, cosine schedule, 10 epochs over ~100k sampled windows. Early stop on val SMAPE plateau.
- **Gate:** test SMAPE ≥15% lower than Stage 0. If not met → halt; investigate.

**Stage 2 — LoRA + covariates:**

- Same LoRA config. Training uses `forecast_with_covariates` API with `days_to_departure` (dynamic numerical) and `day_of_week` (dynamic categorical).
- `xreg_mode="xreg + timesfm"` per the official recipe.
- All other hyperparameters held constant from Stage 1 to isolate the covariate contribution.
- **Gate:** test SMAPE ≥5% lower than Stage 1 **AND** recall@80%-precision ≥10pp higher than Stage 0. If not met → ship Stage 1 adapter instead.

**Reproducibility/safety:**

- Fixed seeds (numpy, torch, sampler).
- Each run writes `artifacts/runs/<stage>/{adapter/, metrics.json, config.yaml, train_log.jsonl, eval_report.md, summary.json}`.
- Config is the only file the training script reads — no hidden CLI overrides. Any run is replayable from its config.

## Evaluation

Two metric families, both reported.

**Forecast metrics** (model selection + stage gates):

- **MAPE** and **SMAPE** at horizon 168h, computed in dollar-space after `exp`-transforming predictions. Primary metric: SMAPE.
- **Pinball loss** at quantiles {0.1, 0.5, 0.9} (when using the continuous quantile head). Calibrates the forecast distribution that the buy/wait decision depends on.
- Reported per-series and aggregated (mean, median, P90).

**Decision metrics** (deployment gate):

- Frame buy/wait as binary classifier: `predicted_min_price_in_next_7d < current_price * (1 - 0.10)` → "wait."
- Ground truth: actual realized minimum price over the next 7 days.
- Sweep decision threshold; report the precision-recall curve and the **recall at 80% precision** operating point.

**Baselines (always reported alongside the model):**

1. Naive last-value (forecast = current price for all 168 steps).
2. Seasonal-naive(24h) and seasonal-naive(168h).
3. Zero-shot TimesFM-2.5 (the gate the LoRA must beat).

**Eval splits:**

- Val (37 routes): training-time signal for early stopping and hyperparam tuning.
- Test (37 routes): touched **once per stage** at end. Test metrics decide gates.
- **Departure-time stratification:** report broken down by `days_to_departure` bucket (0–7d, 7–30d, 30+d).

**Reporting:**

- `eval.py` writes `artifacts/runs/<stage>/eval_report.md`:
  - Aggregate forecast metrics (table: model × metric × baseline).
  - Per-bucket forecast metrics (table: bucket × metric).
  - PR curve plot + 80%-precision operating point (PNG).
  - Top-20 worst-performing series by per-series SMAPE for sanity-checking.
- `summary.json` — minimal JSON the gate logic in `run_stage.py` reads to decide pass/fail automatically.

## Environment

Server-side, managed by `uv`:

- Python 3.11
- Core: `timesfm`, `transformers`, `peft`, `accelerate`, `torch` (CUDA build)
- Data: `polars`, `pyarrow`, `psycopg[binary]`
- Eval/plots: `numpy`, `scikit-learn`, `matplotlib`
- Dev: `pytest`, `ruff`
- All pinned in `pyproject.toml` + `uv.lock`.

## Deployment (v2 batch scorer)

Built only after Stage 2 passes its gates. New script `scripts/run_batch_scorer.py`, invoked by cron or systemd timer on the server.

Each run:

1. Pulls all active `(route, departure)` series from `flight_prices` where `departure_datetime > now()` and ≥512h of contiguous history exists.
2. Runs forecast inference (Stage 2 adapter) for all qualifying series in batched bf16 — single 24GB GPU handles thousands per minute.
3. For each series computes: `predicted_min_p10/p50/p90 over next 168h` (dollar-space), `current_min_price`, derived `wait_score`, and `wait_decision` at the calibrated 80%-precision threshold.
4. Writes results to a new `flight_predictions` table (`route_id, departure_datetime, scored_at, p10, p50, p90, wait_score, wait_decision`).

The existing `unfare` alert pipeline reads `flight_predictions` to decide whether to fire alerts — no changes to alert logic itself, only a new column to consult.

**Cadence:** every 6 hours initially. Tunable.

**Schema migration:** new `flight_predictions` table only. No changes to existing tables. The `unfare` repo (separate) gets a small reader change, out of scope here.

## Non-goals (v1)

- No real-time inference API.
- No per-route or per-cluster LoRA adapters; single global adapter only.
- No multi-horizon training; 168h only.
- No retraining automation; manual `make train-*` re-runs are sufficient until a multi-version history exists.
- No model serving infrastructure (TF Serving, TorchServe, Triton).
- No changes to scrapers, alert logic, or the `unfare` web app.

## Risks

- **Forward-fill of short gaps** could mask scraper outages as legitimate price persistence. Mitigated by splitting series at gaps >6h. Watch eval reports for anomalous low-error series — could indicate filled-flat data.
- **37-route test set** is statistically thin. If gate decisions sit near the threshold, fall back to 5-fold cross-validation on the route split.
- **TimesFM 2.5 is recent**; APIs may shift. Pin exact versions in `uv.lock` and avoid relying on undocumented internals.
- **Log-transform interaction with TimesFM's per-window normalization** is layered preprocessing the foundation model wasn't pretrained on. The LoRA should adapt to it, but if Stage 1 fails to beat zero-shot this is the first thing to test (re-run without the log-transform).
