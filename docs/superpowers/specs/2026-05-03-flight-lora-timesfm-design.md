# Flight Pricing LoRA on TimesFM — Design

**Date:** 2026-05-03
**Status:** Draft, pending implementation plan

## Goal

Train a LoRA adapter on top of TimesFM-2.5 using historical flight pricing data from `unfare_db` to produce a forecast model whose output drives a **buy-now vs. wait** signal for the existing alert system. Research first, then graduate the trained adapter into a batch scorer that writes predictions back to Postgres.

## Background

The `unfare` system scrapes flight prices into a TimescaleDB hypertable (`flight_prices`) at roughly hourly cadence per (route, departure). Existing alerting is rule-based and reactive. A learned forecast over the next 7 days enables proactive "wait — likely to drop" recommendations. TimesFM is a strong forecasting foundation model with first-class HuggingFace + PEFT support, making LoRA fine-tuning the natural path.

## Data overview (as of 2026-05-03)

- **Rows:** 55.97M in `flight_prices`, 3-month coverage (2026-02-03 → 2026-05-03).
- **Routes:** 246 actively scraped (of 406 total active in `routes`); all are domestic (`is_international=0`).
- **Forward booking horizon:** ~75 days (departures up to 2026-07-19).
- **Per-(route, departure) cadence:** 400–1000 snapshots per week, covering ~42–46 distinct hours → effectively hourly.
- **Storage:** TimescaleDB hypertable; chunked by `scraped_at`.
- **Currency:** All prices assumed USD (consistent with all-domestic route set). To be verified before training.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction target | Buy/wait signal derived from price-trajectory forecast | Forecast first, threshold downstream; clean separation |
| Series unit | Cheapest economy quote per `(route_id, departure_datetime)` | Matches "should I buy this fare" framing |
| Resampling cadence | Hourly | Preserves intraday volatility / drop signal |
| Forecast horizon | 168 hours (7 days) | Window where buy/wait has a meaningful answer |
| Hardware | Single 24GB consumer GPU | Sufficient for LoRA on TimesFM-2.5-200M |
| Covariates | **None in v1** (univariate only); XReg overlay deferred to v1.5 | TimesFM-2.5 has no covariate-aware training path; see Risks |
| Deployment | Research → batch scorer (no real-time API in v1) | Don't build serving until model proves out |
| Success criteria | Forecast metrics + decision metrics, both gated | Vibes-free promotion criteria |
| LoRA strategy | Single global adapter, baseline-gated training | Cheap-to-debug; one zero-shot baseline + one LoRA training run |
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
│   ├── train.py               # baseline + LoRA training runs
│   ├── eval.py                # forecast + decision-quality metrics
│   └── infer.py               # inference library: load adapter, score series (used by v2)
├── notebooks/                 # exploration, sanity checks, eval plots
├── scripts/
│   ├── build_dataset.py       # one-shot: pull from db → write parquet
│   ├── run_stage.py           # CLI: --stage zero_shot|lora_uni|xreg_eval
│   ├── run_batch_scorer.py    # v2 batch inference
│   └── remote_train.sh        # ssh + tmux wrapper around run_stage.py
├── tests/                     # unit tests
├── Makefile                   # baseline, train-uni, eval-xreg, pull-artifacts, etc.
├── .env.local                 # DB_HOST=192.168.93.107  (Mac side)
├── .env.server                # DB_HOST=localhost        (server side, mutagen-excluded)
├── .mutagen.yml               # sync config
├── pyproject.toml
└── uv.lock
```

**Module boundaries:**

- **`db.py`** — given a config + window, returns a DataFrame of raw `flight_prices` rows. Postgres is its only dependency; trivially mockable.
- **`dataset.py`** — pure transform from raw rows to `(context, target)` tensors at a regular hourly grid. Owns the route-level train/val/test split. Applies `log(price)` before yielding.
- **`train.py`** — given a stage config, loads model + LoRA, trains, saves adapter. Knows nothing about the DB.
- **`eval.py`** — given predictions and ground truth, computes forecast metrics in dollar-space (post-`exp`) and decision metrics. Reusable across all stages.
- **`infer.py`** — given a saved adapter and a series, returns quantile forecasts. Pure inference, no orchestration. Imported by `scripts/run_batch_scorer.py` for the v2 deployment.

**Data flow:**

```
Postgres → db.py → raw DataFrame
                       ↓
               build_dataset.py → cached parquet (per-series, log-transformed)
                       ↓
                   dataset.py → (context, target) tensors
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

Training is invoked with `make baseline` / `make train-uni` (and optional `make eval-xreg`), each of which `ssh`s to the server, starts (or attaches to) a per-stage tmux session, and runs `scripts/run_stage.py` inside it. The training script knows nothing about being driven remotely. `make pull-artifacts` rsyncs eval reports back to the Mac for inspection.

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

**Covariates:** None in v1. TimesFM-2.5 provides no covariate-aware fine-tuning path — see Risks. The model is expected to implicitly capture weekly seasonality (168-hour periodicity is regular in the data) and proximity-to-departure (every series ends at the departure timestamp, so position-from-end encodes it). XReg-style overlay is deferred to v1.5; see "Future work."

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

## Training (baseline + LoRA)

Two sequential runs, gated. All use `google/timesfm-2.5-200m-pytorch` as the base. All log to `artifacts/runs/<stage>/`. A third optional inference-time evaluation (Stage 1.5) is described after.

**Stage 0 — Zero-shot baseline (no training):**

- Load base TimesFM, run inference on val and test sets.
- Records: per-series MAPE/SMAPE at 168h horizon, plus naive-last-value and seasonal-naive(24h, 168h) baselines.
- Output: standard run directory `artifacts/runs/zero_shot/{metrics.json, eval_report.md, summary.json, config.yaml}` (no `adapter/` since nothing is trained). No self-gate — Stage 0 metrics establish the bar that Stage 1's gate references.

**Stage 1 — Univariate LoRA:**

- `TimesFm2_5ModelForPrediction` + `peft.LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.05, bias="none")`. Mirrors the official `finetune_lora.py` recipe verbatim apart from data and hyperparameters.
- ~0.5% of params trainable (~1M params). Fits in 24GB with bf16 and reasonable batch size.
- Training forward pass: `model(past_values=context, future_values=target, forecast_context_len=context_len)` — `outputs.loss` is the model's internal loss (verified against current source; we don't compute our own).
- Optimizer: AdamW, lr 5e-5, cosine schedule, 10 epochs over ~100k sampled windows. Gradient clip at 1.0. Early stop on val SMAPE plateau.
- **Ship gate:** test SMAPE ≥15% lower than Stage 0 **AND** recall@80%-precision ≥10pp higher than Stage 0. If both met → ship adapter to v2 batch scorer. If not → halt; investigate.

**Stage 1.5 — XReg overlay at inference (optional, evaluation only):**

- No additional training. After Stage 1 ships, evaluate the adapter again with `forecast_with_covariates(xreg_mode="xreg + timesfm")` passing `days_to_departure` (dynamic numerical) and `day_of_week` (dynamic categorical).
- XReg here is a per-call sklearn linear regression on the context — it learns no parameters during our training. Adds zero training cost.
- **Promote-to-deployment gate:** Stage 1.5 metrics must beat Stage 1 metrics by **≥3% SMAPE** on the test set to justify enabling XReg in the batch scorer (XReg adds ~per-series sklearn fit cost and runtime dependency on `scikit-learn`).
- Limitation: linear-only fit on covariates; nonlinear effects (e.g., V-shaped fare-curve in last 14 days) will not be captured. If Stage 1.5 fails its gate, ship Stage 1 alone.

**Reproducibility/safety:**

- Fixed seeds (numpy, torch, sampler).
- Each run writes `artifacts/runs/<stage>/{adapter/, metrics.json, config.yaml, train_log.jsonl, eval_report.md, summary.json}` (no `adapter/` for Stage 0 or Stage 1.5).
- Config is the only file the training/eval script reads — no hidden CLI overrides. Any run is replayable from its config.

## Evaluation

Two metric families, both reported.

**Forecast metrics** (model selection + stage gates):

- **MAPE** and **SMAPE** at horizon 168h, computed in dollar-space after `exp`-transforming predictions. Primary metric: SMAPE.
- **Pinball loss** at quantiles {0.1, 0.5, 0.9} (when using the continuous quantile head). Calibrates the forecast distribution that the buy/wait decision depends on.
- Reported per-series and aggregated (mean, median, P90).

**Decision metrics** (deployment gate):

- Frame buy/wait as binary classifier: positive class = "actual minimum price over next 7d is at least 10% below current price" (the *truth* label).
- The classifier's score is `wait_score = P(predicted_min_p_quantile < current_price * 0.9)`, derived from the forecast quantile distribution (using the q=0.1 lower-tail estimate as the conservative "drop possible" probability).
- Sweep the **`wait_score` threshold** (0 → 1); report the precision-recall curve and the **recall at 80% precision** operating point. The 10% drop cutoff is the *positive-class definition* and is held fixed across the sweep.

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

Built only after Stage 1 passes its ship gate. New script `scripts/run_batch_scorer.py`, invoked by cron or systemd timer on the server.

Each run:

1. Pulls all active `(route, departure)` series from `flight_prices` where `departure_datetime > now()` and ≥512h of contiguous history exists.
2. Runs forecast inference (Stage 1 adapter) for all qualifying series in batched bf16 — single 24GB GPU handles thousands per minute. If Stage 1.5 has also passed its gate, inference uses `forecast_with_covariates` with `xreg_mode="xreg + timesfm"`; otherwise plain `forecast`.
3. For each series computes: `predicted_min_p10/p50/p90 over next 168h` (dollar-space), `current_min_price`, derived `wait_score`, and `wait_decision` at the calibrated 80%-precision threshold.
4. Writes results to a new `flight_predictions` table (`route_id, departure_datetime, scored_at, p10, p50, p90, wait_score, wait_decision`).

The existing `unfare` alert pipeline reads `flight_predictions` to decide whether to fire alerts — no changes to alert logic itself, only a new column to consult.

**Cadence:** every 6 hours initially. Tunable.

**Schema migration:** new `flight_predictions` table only. No changes to existing tables. The `unfare` repo (separate) gets a small reader change, out of scope here.

## Non-goals (v1)

- No real-time inference API.
- No per-route or per-cluster LoRA adapters; single global adapter only.
- No multi-horizon training; 168h only.
- **No covariate-aware fine-tuning.** TimesFM-2.5's `PatchedTimeSeriesDecoder` (the LoRA target) has no covariate-aware training path; covariates only enter inference via the post-hoc XReg linear regression. v1 uses pure univariate. See "Future work."
- No retraining automation; manual `make train-*` re-runs are sufficient until a multi-version history exists.
- No model serving infrastructure (TF Serving, TorchServe, Triton).
- No changes to scrapers, alert logic, or the `unfare` web app.

## Risks & known constraints

- **TimesFM-2.5 has no covariate-aware fine-tuning.** Verified against (a) the `finetune_lora.py` source on `master` (univariate `(context, target)` only), (b) GitHub issue #257, (c) the `PatchedTimeSeriesDecoder` source which has no covariate forward path, and (d) third-party tutorials confirming covariate support was removed between TimesFM-2.0 and 2.5. The XReg API at inference time uses a per-call sklearn linear fit on the context — it is *not* trained jointly with the model. Implications for our v1: pure univariate LoRA; calendar/seasonality signals come purely from the foundation model's pattern-matching plus the regular hourly cadence in the data.
- **Forward-fill of short gaps** could mask scraper outages as legitimate price persistence. Mitigated by splitting series at gaps >6h. Watch eval reports for anomalous low-error series — could indicate filled-flat data.
- **37-route test set** is statistically thin. If gate decisions sit near the threshold, fall back to 5-fold cross-validation on the route split.
- **TimesFM 2.5 is recent**; APIs may shift. Pin exact versions in `uv.lock` and avoid relying on undocumented internals.
- **Log-transform interaction with TimesFM's internal RevIN normalization** is layered preprocessing the foundation model wasn't pretrained on. The LoRA should adapt to it, but if Stage 1 fails to beat zero-shot this is the first thing to test (re-run without the log-transform).

## Future work (v1.5+)

- **Stage 1.5 XReg overlay** (already specified above) — first thing to evaluate after Stage 1 ships.
- **Calendar-effect pre-subtraction**: fit a global per-(day_of_week, days_to_departure_bucket) median multiplier; subtract from log-prices before feeding to TimesFM; LoRA learns to forecast residuals; add the calendar effect back at inference. Lets the foundation model focus on dynamics; gives covariate-like benefits without needing TimesFM to ingest covariates. Worth exploring if Stage 1.5 underperforms.
- **Per-route-cluster LoRAs**: cluster routes by haul length / volatility; train one adapter per cluster. Only worth doing if global LoRA plateaus well below per-route potential.
- **Multi-horizon training**: add 24h/72h horizons alongside 168h.
- **Retraining automation** once we have ≥3 successful versions and a clear cadence.
