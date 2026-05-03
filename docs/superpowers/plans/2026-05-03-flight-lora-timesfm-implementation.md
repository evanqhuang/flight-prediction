# Flight Pricing LoRA on TimesFM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and gate a univariate LoRA adapter on TimesFM-2.5 against `unfare_db` flight pricing data. Ships an adapter + automated gate decision; v2 batch scorer is out of scope.

**Architecture:** Five-module Python package (`db`, `dataset`, `eval`, `infer`, `train`) plus three orchestration scripts. Pure-transform modules (`dataset`, `eval`) are TDD'd against synthetic fixtures on the Mac; GPU-bound modules (`infer`, `train`) get smoke-test coverage on the Mac and are validated end-to-end via Stage 0 / Stage 1 runs on the remote GPU server. Mutagen continuously syncs source code to the server; `make` targets ssh in and run training inside `tmux`.

**Tech Stack:** Python 3.11, `uv` for env management, `polars` + `pyarrow` for data, `psycopg[binary]` for Postgres, `transformers` + `peft` + `accelerate` + `torch` (CUDA) for the model, `pytest` + `ruff` for tests/lint.

**Spec:** `docs/superpowers/specs/2026-05-03-flight-lora-timesfm-design.md`

---

## Prerequisites (operational, not tasks)

- **Server access:** ssh key auth to the GPU box (assumed reachable as `gpu-host` in `~/.ssh/config`; substitute `192.168.93.107` if no alias is configured).
- **Server software:** Python 3.11, CUDA toolkit matching the installed GPU driver, `uv` installed, `tmux`.
- **Postgres reachable from server at `localhost:5433`** (the unfare_db is on the same host as the GPU).
- **Mutagen** installed on both Mac and server (`brew install mutagen-io/mutagen/mutagen`).
- **Database credentials:** the implementer has `unfare_user` / `unfare_pass` for the read-only role.

---

## Task 1: Bootstrap the project

**Files:**
- Create: `pyproject.toml`
- Create: `src/flight_lora/__init__.py`
- Create: `tests/__init__.py`
- Create: `Makefile`
- Create: `.env.example`
- Create: `.mutagen.yml`
- Modify: `.gitignore` (already exists, append entries if missing)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "flight-lora"
version = "0.1.0"
description = "TimesFM LoRA adapter for unfare flight pricing forecasts"
requires-python = "==3.11.*"
dependencies = [
    "torch>=2.4.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "timesfm[xreg]>=2.5.0",
    "polars>=1.10.0",
    "pyarrow>=18.0.0",
    "psycopg[binary]>=3.2.0",
    "scikit-learn>=1.5.0",
    "matplotlib>=3.9.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/flight_lora"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = ["gpu: tests that require a CUDA GPU (skipped on Mac)"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
```

- [ ] **Step 2: Create the empty package init files**

```python
# src/flight_lora/__init__.py
"""Flight pricing LoRA adapter for TimesFM."""

__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 3: Write the `Makefile`**

```makefile
SHELL := /bin/bash
SERVER ?= gpu-host
REMOTE_DIR ?= ~/flight-prediction

.PHONY: install lint test test-gpu sync baseline train-uni eval-xreg pull-artifacts

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -m "not gpu"

test-gpu:
	uv run pytest -m gpu

sync:
	mutagen sync create --name=flight-lora . $(SERVER):$(REMOTE_DIR) || \
	  mutagen sync resume flight-lora
	mutagen sync flush flight-lora

baseline: sync
	bash scripts/remote_train.sh zero_shot

train-uni: sync
	bash scripts/remote_train.sh lora_uni

eval-xreg: sync
	bash scripts/remote_train.sh xreg_eval

pull-artifacts:
	rsync -av --exclude='*.pt' --exclude='adapter_model.safetensors' \
	  $(SERVER):$(REMOTE_DIR)/artifacts/ ./artifacts/
```

- [ ] **Step 4: Write `.env.example`**

```bash
# Database connection (Mac points at LAN IP; server uses localhost)
DB_HOST=192.168.93.107
DB_PORT=5433
DB_USER=unfare_user
DB_PASSWORD=
DB_NAME=unfare_db

# Optional: cutoff timestamp for dataset build (ISO 8601 UTC)
DATASET_CUTOFF=2026-05-03T00:00:00Z
```

- [ ] **Step 5: Write `.mutagen.yml`**

```yaml
sync:
  defaults:
    mode: two-way-resolved
    ignore:
      vcs: true
      paths:
        - "/data"
        - "/artifacts"
        - "/.venv"
        - "/.env"
        - "/.env.local"
        - "/.env.server"
        - "*.pyc"
        - "__pycache__"
        - ".pytest_cache"
        - ".ruff_cache"
        - "*.parquet"
        - "*.safetensors"
```

- [ ] **Step 6: Verify `.gitignore` covers env files and parquet/artifacts**

Confirm these lines exist (add any missing):

```
.env
.env.local
.env.server
data/
artifacts/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 7: Run `uv sync` and verify install**

Run: `uv sync --all-extras`
Expected: completes without error; `uv run python -c "import flight_lora; print(flight_lora.__version__)"` prints `0.1.0`.

- [ ] **Step 8: Verify lint passes (no source files yet, so this is a no-op smoke test)**

Run: `uv run ruff check src tests scripts || true`
Expected: no errors (scripts/ doesn't exist yet — `|| true` swallows that one warning).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/flight_lora/__init__.py tests/__init__.py \
        Makefile .env.example .mutagen.yml .gitignore
git commit -m "scaffold project: package layout, deps, Makefile, mutagen config"
```

---

## Task 2: db.py — Postgres reader

**Files:**
- Create: `src/flight_lora/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
"""Tests for db.py — focus on SQL construction & parameter handling.
End-to-end DB connectivity is covered by Task 3's smoke test."""

import os
from unittest.mock import MagicMock, patch

import pytest

from flight_lora.db import DBConfig, fetch_flight_prices, fetch_route_ids


def test_dbconfig_from_env(monkeypatch):
    monkeypatch.setenv("DB_HOST", "192.168.1.10")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("DB_NAME", "n")
    cfg = DBConfig.from_env()
    assert cfg.host == "192.168.1.10"
    assert cfg.port == 5433
    assert cfg.user == "u"
    assert cfg.password == "p"
    assert cfg.dbname == "n"


def test_dbconfig_missing_password_raises(monkeypatch):
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="DB_PASSWORD"):
        DBConfig.from_env()


def test_fetch_flight_prices_builds_correct_sql(mocker):
    cfg = DBConfig(host="h", port=5432, user="u", password="p", dbname="d")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.description = [
        ("scraped_at",), ("route_id",), ("departure_datetime",), ("price",),
        ("airline",), ("stops",), ("cabin_class",), ("is_mistake_fare",),
    ]
    mocker.patch("flight_lora.db.psycopg.connect", return_value=mock_conn)

    fetch_flight_prices(cfg, route_ids=[5, 21], cutoff="2026-05-03T00:00:00Z")

    sql, params = mock_cursor.execute.call_args[0]
    assert "FROM flight_prices" in sql
    assert "cabin_class = 'economy'" in sql
    assert "scraped_at < %(cutoff)s" in sql
    assert "scraped_at < departure_datetime" in sql
    assert "route_id = ANY(%(route_ids)s)" in sql
    assert params == {"cutoff": "2026-05-03T00:00:00Z", "route_ids": [5, 21]}


def test_fetch_route_ids_returns_active_only(mocker):
    cfg = DBConfig(host="h", port=5432, user="u", password="p", dbname="d")
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(5,), (21,), (123,)]
    mocker.patch("flight_lora.db.psycopg.connect", return_value=mock_conn)

    ids = fetch_route_ids(cfg)
    assert ids == [5, 21, 123]
    sql = mock_cursor.execute.call_args[0][0]
    assert "is_active" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: 4 failures with `ImportError` / `ModuleNotFoundError` for `flight_lora.db`.

- [ ] **Step 3: Implement `db.py`**

```python
# src/flight_lora/db.py
"""Read-only Postgres access to unfare_db.

Only path that touches Postgres directly. End-to-end correctness is verified
by build_dataset.py against the real DB (Task 3's smoke test); unit tests
here cover SQL shape and config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import polars as pl
import psycopg


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str

    @classmethod
    def from_env(cls) -> "DBConfig":
        password = os.environ.get("DB_PASSWORD")
        if not password:
            raise ValueError("DB_PASSWORD env var is required")
        return cls(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", "5432")),
            user=os.environ.get("DB_USER", "unfare_user"),
            password=password,
            dbname=os.environ.get("DB_NAME", "unfare_db"),
        )

    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname}"
        )


_FLIGHT_PRICES_SQL = """
SELECT
    scraped_at,
    route_id,
    departure_datetime,
    price::float8 AS price,
    airline,
    stops,
    cabin_class,
    is_mistake_fare
FROM flight_prices
WHERE cabin_class = 'economy'
  AND scraped_at < departure_datetime
  AND scraped_at < %(cutoff)s
  AND route_id = ANY(%(route_ids)s)
ORDER BY route_id, departure_datetime, scraped_at
"""

_ROUTE_IDS_SQL = """
SELECT id FROM routes
WHERE is_active = true
ORDER BY id
"""


def fetch_route_ids(cfg: DBConfig) -> list[int]:
    """Return the list of currently-active route ids."""
    with psycopg.connect(cfg.conninfo()) as conn, conn.cursor() as cur:
        cur.execute(_ROUTE_IDS_SQL)
        return [row[0] for row in cur.fetchall()]


def fetch_flight_prices(
    cfg: DBConfig,
    route_ids: Iterable[int],
    cutoff: str,
) -> pl.DataFrame:
    """Return economy quotes for the given routes, scraped before `cutoff`."""
    route_list = list(route_ids)
    with psycopg.connect(cfg.conninfo()) as conn, conn.cursor() as cur:
        cur.execute(
            _FLIGHT_PRICES_SQL,
            {"cutoff": cutoff, "route_ids": route_list},
        )
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pl.from_records(rows, schema=cols, orient="row")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/db.py tests/test_db.py
git commit -m "add db.py: read-only Postgres access for flight_prices and routes"
```

---

## Task 3: db.py smoke test against real unfare_db

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/check_db.py`
- Create: `.env.local` (NOT committed; gitignored)

- [ ] **Step 1: Create `.env.local`** (gitignored)

```bash
DB_HOST=192.168.93.107
DB_PORT=5433
DB_USER=unfare_user
DB_PASSWORD=unfare_pass
DB_NAME=unfare_db
DATASET_CUTOFF=2026-05-03T00:00:00Z
```

- [ ] **Step 2: Write `scripts/check_db.py`**

```python
# scripts/check_db.py
"""One-shot smoke test: connect to the real DB, fetch a tiny slice, print stats."""

from dotenv import load_dotenv

from flight_lora.db import DBConfig, fetch_flight_prices, fetch_route_ids


def main() -> None:
    load_dotenv(".env.local")
    cfg = DBConfig.from_env()
    print(f"Connecting to {cfg.host}:{cfg.port}/{cfg.dbname} as {cfg.user} ...")

    routes = fetch_route_ids(cfg)
    print(f"Active routes: {len(routes)}")
    assert len(routes) > 0, "expected at least one active route"

    sample = routes[:3]
    df = fetch_flight_prices(cfg, route_ids=sample, cutoff="2026-05-03T00:00:00Z")
    print(f"Sample rows for routes {sample}: {df.height:,}")
    print(df.head())
    print(df.describe())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/__init__.py` (empty)**

```python
# scripts/__init__.py
```

- [ ] **Step 4: Run the smoke test**

Run: `uv run python scripts/check_db.py`
Expected: prints "Active routes: 246" (or similar), prints a sample DataFrame with columns `scraped_at, route_id, departure_datetime, price, airline, stops, cabin_class, is_mistake_fare`, and a `describe()` table.

If it errors with auth: confirm `.env.local` contents. If it errors with timeout: confirm LAN reachability with `nc -zv 192.168.93.107 5433`.

- [ ] **Step 5: Commit (the script, not the .env.local)**

```bash
git add scripts/__init__.py scripts/check_db.py
git commit -m "add scripts/check_db.py: smoke test for db.py against real unfare_db"
```

---

## Task 4: dataset.py — hourly resampling

**Files:**
- Create: `src/flight_lora/dataset.py`
- Create: `tests/test_dataset.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write a shared fixture**

```python
# tests/conftest.py
"""Shared synthetic fixtures for dataset/eval tests."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest


@pytest.fixture
def synthetic_raw_rows() -> pl.DataFrame:
    """Two routes x two departures x 720 hourly snapshots each (30 days)."""
    rows = []
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for route_id in (5, 21):
        for dep_offset_days in (45, 60):
            departure = base + timedelta(days=dep_offset_days)
            for hour in range(720):
                scraped = base + timedelta(hours=hour)
                price = 200.0 + 50.0 * ((hour % 168) - 84) / 84
                rows.append({
                    "scraped_at": scraped,
                    "route_id": route_id,
                    "departure_datetime": departure,
                    "price": price,
                    "airline": "AA",
                    "stops": 0,
                    "cabin_class": "economy",
                    "is_mistake_fare": False,
                })
    return pl.from_dicts(rows)
```

- [ ] **Step 2: Write the failing test for `build_hourly_series`**

```python
# tests/test_dataset.py
"""Tests for dataset.py — pure transforms on synthetic input."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from flight_lora.dataset import build_hourly_series


def test_hourly_series_is_one_per_route_and_departure(synthetic_raw_rows):
    series = build_hourly_series(synthetic_raw_rows)
    keys = {(s.route_id, s.departure_datetime) for s in series}
    assert len(keys) == 4  # 2 routes x 2 departures


def test_hourly_series_uses_min_price_per_bucket():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    departure = base + timedelta(days=10)
    rows = pl.from_dicts([
        {"scraped_at": base, "route_id": 1, "departure_datetime": departure,
         "price": 300.0, "airline": "AA", "stops": 0, "cabin_class": "economy",
         "is_mistake_fare": False},
        {"scraped_at": base + timedelta(minutes=30), "route_id": 1,
         "departure_datetime": departure, "price": 250.0, "airline": "AA",
         "stops": 0, "cabin_class": "economy", "is_mistake_fare": False},
        {"scraped_at": base + timedelta(hours=1), "route_id": 1,
         "departure_datetime": departure, "price": 280.0, "airline": "AA",
         "stops": 0, "cabin_class": "economy", "is_mistake_fare": False},
    ])
    series = build_hourly_series(rows)
    assert len(series) == 1
    prices = series[0].prices
    assert prices[0] == 250.0
    assert prices[1] == 280.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: failures with `ModuleNotFoundError: flight_lora.dataset`.

- [ ] **Step 4: Implement `build_hourly_series` in `dataset.py`**

```python
# src/flight_lora/dataset.py
"""Pure transforms: raw rows -> hourly series -> (context, target) tensors.

Owns route-level train/val/test split. Applies log(price). All time-series
work happens here; downstream modules see only numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import polars as pl


@dataclass
class HourlySeries:
    """One time series for a (route, departure) pair on a regular hourly grid."""
    route_id: int
    departure_datetime: datetime
    start_hour: datetime  # first hour bucket (UTC, hour-aligned)
    prices: np.ndarray   # 1-D float64 array of dollar prices, hourly cadence


def build_hourly_series(raw: pl.DataFrame) -> list[HourlySeries]:
    """Group raw rows by (route_id, departure_datetime); aggregate to hourly min(price)."""
    if raw.is_empty():
        return []

    df = (
        raw.with_columns(
            pl.col("scraped_at").dt.truncate("1h").alias("hour_bucket")
        )
        .group_by(["route_id", "departure_datetime", "hour_bucket"])
        .agg(pl.col("price").min().alias("price"))
        .sort(["route_id", "departure_datetime", "hour_bucket"])
    )

    out: list[HourlySeries] = []
    for (route_id, dep), grp in df.group_by(["route_id", "departure_datetime"], maintain_order=True):
        hours = grp["hour_bucket"].to_list()
        prices = grp["price"].to_numpy()
        if not hours:
            continue
        out.append(HourlySeries(
            route_id=int(route_id),
            departure_datetime=dep,
            start_hour=hours[0],
            prices=prices.astype(np.float64),
        ))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/flight_lora/dataset.py tests/test_dataset.py tests/conftest.py
git commit -m "add dataset.build_hourly_series: per-(route,departure) hourly min(price)"
```

---

## Task 5: dataset.py — gap splitting + length filter + log transform

**Files:**
- Modify: `src/flight_lora/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_dataset.py`:

```python
import pytest

from flight_lora.dataset import (
    HourlySeries,
    forward_fill_short_gaps,
    log_transform,
    split_at_long_gaps,
    filter_by_min_length,
)


def _make_series_with_gap(prices_before: int, gap_hours: int, prices_after: int) -> HourlySeries:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    total = prices_before + gap_hours + prices_after
    arr = np.full(total, np.nan, dtype=np.float64)
    arr[:prices_before] = 100.0
    arr[prices_before + gap_hours:] = 200.0
    return HourlySeries(
        route_id=1,
        departure_datetime=base + timedelta(days=60),
        start_hour=base,
        prices=arr,
    )


def test_forward_fill_short_gaps_fills_up_to_six_hours():
    s = _make_series_with_gap(prices_before=10, gap_hours=4, prices_after=10)
    filled = forward_fill_short_gaps(s, max_gap=6)
    assert not np.isnan(filled.prices).any()
    assert filled.prices[10] == 100.0
    assert filled.prices[13] == 100.0


def test_forward_fill_does_not_fill_long_gaps():
    s = _make_series_with_gap(prices_before=10, gap_hours=12, prices_after=10)
    filled = forward_fill_short_gaps(s, max_gap=6)
    assert np.isnan(filled.prices[10:22]).all()


def test_split_at_long_gaps_produces_two_series():
    s = _make_series_with_gap(prices_before=10, gap_hours=12, prices_after=10)
    parts = split_at_long_gaps(s, max_gap=6)
    assert len(parts) == 2
    assert len(parts[0].prices) == 10
    assert len(parts[1].prices) == 10
    assert parts[1].start_hour == s.start_hour + timedelta(hours=22)


def test_filter_by_min_length_drops_short():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    short = HourlySeries(1, base, base, np.ones(100))
    long_series = HourlySeries(2, base, base, np.ones(700))
    kept = filter_by_min_length([short, long_series], min_length=680)
    assert len(kept) == 1
    assert kept[0].route_id == 2


def test_log_transform_applied():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    s = HourlySeries(1, base, base, np.array([100.0, 200.0, 400.0]))
    out = log_transform(s)
    np.testing.assert_allclose(out.prices, np.log([100.0, 200.0, 400.0]))


def test_log_transform_rejects_non_positive():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    s = HourlySeries(1, base, base, np.array([100.0, 0.0, 200.0]))
    with pytest.raises(ValueError, match="non-positive"):
        log_transform(s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 6 new failures (`ImportError` for the new functions).

- [ ] **Step 3: Add the new functions to `dataset.py`**

Append to `src/flight_lora/dataset.py`:

```python
from datetime import timedelta


def forward_fill_short_gaps(s: HourlySeries, max_gap: int = 6) -> HourlySeries:
    """Forward-fill NaN runs of length <= max_gap. Leaves longer runs as NaN."""
    arr = s.prices.copy()
    n = len(arr)
    i = 0
    while i < n:
        if np.isnan(arr[i]):
            j = i
            while j < n and np.isnan(arr[j]):
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0:
                arr[i:j] = arr[i - 1]
            i = j
        else:
            i += 1
    return HourlySeries(s.route_id, s.departure_datetime, s.start_hour, arr)


def split_at_long_gaps(s: HourlySeries, max_gap: int = 6) -> list[HourlySeries]:
    """Split a series into contiguous segments wherever a NaN run > max_gap exists.

    Assumes forward_fill_short_gaps has already been applied; remaining NaN runs
    are by definition longer than max_gap and therefore real splits.
    """
    arr = s.prices
    n = len(arr)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        while i < n and np.isnan(arr[i]):
            i += 1
        if i >= n:
            break
        seg_start = i
        while i < n and not np.isnan(arr[i]):
            i += 1
        segments.append((seg_start, i))
    return [
        HourlySeries(
            s.route_id,
            s.departure_datetime,
            s.start_hour + timedelta(hours=start),
            arr[start:end].copy(),
        )
        for start, end in segments
    ]


def filter_by_min_length(series: Iterable[HourlySeries], min_length: int) -> list[HourlySeries]:
    """Keep only series with at least `min_length` observations."""
    return [s for s in series if len(s.prices) >= min_length]


def log_transform(s: HourlySeries) -> HourlySeries:
    """Apply log(price). Rejects non-positive values."""
    if (s.prices <= 0).any():
        raise ValueError("non-positive price encountered; log_transform requires positive prices")
    return HourlySeries(
        s.route_id, s.departure_datetime, s.start_hour, np.log(s.prices)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: all dataset tests PASS (8 total).

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/dataset.py tests/test_dataset.py
git commit -m "add gap fill/split, length filter, log transform to dataset.py"
```

---

## Task 6: dataset.py — route-level splits and training-pair sampler

**Files:**
- Modify: `src/flight_lora/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
from flight_lora.dataset import (
    sample_training_windows,
    split_routes,
)


def test_split_routes_is_deterministic():
    routes = list(range(246))
    a = split_routes(routes, train_frac=0.7, val_frac=0.15, seed=42)
    b = split_routes(routes, train_frac=0.7, val_frac=0.15, seed=42)
    assert a == b


def test_split_routes_partitions_disjointly():
    routes = list(range(246))
    splits = split_routes(routes, train_frac=0.7, val_frac=0.15, seed=42)
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(routes)


def test_split_routes_proportions_within_one():
    routes = list(range(246))
    splits = split_routes(routes, train_frac=0.7, val_frac=0.15, seed=42)
    assert abs(len(splits["train"]) - 172) <= 1
    assert abs(len(splits["val"]) - 37) <= 1
    assert abs(len(splits["test"]) - 37) <= 1


def test_sample_training_windows_returns_correct_shape():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    series = [
        HourlySeries(1, base, base, np.arange(1000.0)),
        HourlySeries(2, base, base, np.arange(1000.0)),
    ]
    samples = sample_training_windows(
        series, context_len=512, horizon_len=168, num_samples=10, seed=42
    )
    assert len(samples) == 10
    for ctx, tgt in samples:
        assert ctx.shape == (512,)
        assert tgt.shape == (168,)


def test_sample_training_windows_skips_too_short():
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    series = [
        HourlySeries(1, base, base, np.arange(500.0)),
        HourlySeries(2, base, base, np.arange(1000.0)),
    ]
    samples = sample_training_windows(
        series, context_len=512, horizon_len=168, num_samples=20, seed=42
    )
    for ctx, tgt in samples:
        assert ctx.max() < 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 5 failures.

- [ ] **Step 3: Add the new functions to `dataset.py`**

Append to `src/flight_lora/dataset.py`:

```python
def split_routes(
    route_ids: Iterable[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Deterministic route-level split using a seeded permutation."""
    routes = sorted(set(route_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(routes))
    shuffled = [routes[i] for i in perm]

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def sample_training_windows(
    series: list[HourlySeries],
    context_len: int,
    horizon_len: int,
    num_samples: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random-window sampler matching the official TimesFM finetune_lora.py recipe."""
    rng = np.random.default_rng(seed)
    min_len = context_len + horizon_len
    eligible = [s for s in series if len(s.prices) >= min_len]
    if not eligible:
        raise ValueError(
            f"No series with length >= {min_len}; got max length "
            f"{max((len(s.prices) for s in series), default=0)}"
        )

    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_samples):
        s = eligible[int(rng.integers(0, len(eligible)))]
        start = int(rng.integers(0, len(s.prices) - min_len + 1))
        ctx = s.prices[start : start + context_len].astype(np.float32)
        tgt = s.prices[start + context_len : start + min_len].astype(np.float32)
        samples.append((ctx, tgt))
    return samples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: all dataset tests PASS (13 total).

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/dataset.py tests/test_dataset.py
git commit -m "add route-level split and training-window sampler to dataset.py"
```

---

## Task 7: scripts/build_dataset.py — wire it all together

**Files:**
- Create: `scripts/build_dataset.py`
- Create: `tests/test_build_dataset.py`

- [ ] **Step 1: Write a small end-to-end test on synthetic data**

```python
# tests/test_build_dataset.py
"""Tests for the dataset-building pipeline as a whole."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from flight_lora.dataset import (
    build_hourly_series,
    filter_by_min_length,
    forward_fill_short_gaps,
    log_transform,
    split_at_long_gaps,
)


def test_full_pipeline_yields_log_transformed_long_series():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    departure = base + timedelta(days=60)
    for h in range(800):
        rows.append({
            "scraped_at": base + timedelta(hours=h),
            "route_id": 1,
            "departure_datetime": departure,
            "price": 250.0 + 30.0 * np.sin(h * 2 * np.pi / 168),
            "airline": "AA", "stops": 0, "cabin_class": "economy",
            "is_mistake_fare": False,
        })
    raw = pl.from_dicts(rows)

    series = build_hourly_series(raw)
    series = [forward_fill_short_gaps(s) for s in series]
    series = [seg for s in series for seg in split_at_long_gaps(s)]
    series = filter_by_min_length(series, min_length=680)
    series = [log_transform(s) for s in series]

    assert len(series) == 1
    assert len(series[0].prices) == 800
    assert series[0].prices.min() > np.log(200.0)
    assert series[0].prices.max() < np.log(300.0)
```

- [ ] **Step 2: Run the test to verify it passes** (it should — all components already exist)

Run: `uv run pytest tests/test_build_dataset.py -v`
Expected: 1 test PASS.

- [ ] **Step 3: Implement `scripts/build_dataset.py`**

```python
# scripts/build_dataset.py
"""Pull raw rows from unfare_db, transform to hourly series, persist parquet.

Idempotent given a --cutoff timestamp. Default cutoff comes from
DATASET_CUTOFF in .env.local / .env.server.

Output:
    data/series.parquet  -- one row per (route, departure, segment)
    data/splits.json     -- route_id -> "train"|"val"|"test"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from flight_lora.db import DBConfig, fetch_flight_prices, fetch_route_ids
from flight_lora.dataset import (
    build_hourly_series,
    filter_by_min_length,
    forward_fill_short_gaps,
    log_transform,
    split_at_long_gaps,
    split_routes,
)


def main() -> None:
    load_dotenv(".env.server" if os.environ.get("FLIGHT_LORA_ENV") == "server" else ".env.local")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=os.environ.get("DATASET_CUTOFF"))
    parser.add_argument("--min-length", type=int, default=680)
    parser.add_argument("--max-gap", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if not args.cutoff:
        raise SystemExit("--cutoff required (or set DATASET_CUTOFF in env)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DBConfig.from_env()
    print(f"Fetching active routes from {cfg.host}:{cfg.port}/{cfg.dbname} ...")
    route_ids = fetch_route_ids(cfg)
    print(f"  {len(route_ids)} active routes")

    print(f"Fetching flight_prices (cutoff={args.cutoff}) ...")
    raw = fetch_flight_prices(cfg, route_ids=route_ids, cutoff=args.cutoff)
    print(f"  {raw.height:,} raw rows")

    print("Building hourly series ...")
    series = build_hourly_series(raw)
    print(f"  {len(series)} (route, departure) series")

    print("Forward-filling short gaps and splitting at long gaps ...")
    series = [forward_fill_short_gaps(s, max_gap=args.max_gap) for s in series]
    series = [seg for s in series for seg in split_at_long_gaps(s, max_gap=args.max_gap)]
    print(f"  {len(series)} segments after gap handling")

    print(f"Filtering segments shorter than {args.min_length} hours ...")
    series = filter_by_min_length(series, min_length=args.min_length)
    print(f"  {len(series)} segments retained")

    print("Applying log transform ...")
    series = [log_transform(s) for s in series]

    print(f"Writing parquet to {args.out_dir / 'series.parquet'} ...")
    rows = [
        {
            "route_id": s.route_id,
            "departure_datetime": s.departure_datetime,
            "start_hour": s.start_hour,
            "log_prices": s.prices.tolist(),
            "n": len(s.prices),
        }
        for s in series
    ]
    pl.from_dicts(rows).write_parquet(args.out_dir / "series.parquet")

    print("Computing route splits ...")
    used_routes = sorted({s.route_id for s in series})
    splits = split_routes(used_routes, train_frac=0.7, val_frac=0.15, seed=42)
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    (args.out_dir / "splits.json").write_text(json.dumps(splits, indent=2, default=str))

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_dataset.py tests/test_build_dataset.py
git commit -m "add scripts/build_dataset.py: db -> hourly series -> parquet pipeline"
```

---

## Task 8: eval.py — point forecast metrics + pinball loss

**Files:**
- Create: `src/flight_lora/eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval.py
"""Tests for eval.py — pure functions on synthetic predictions."""

import numpy as np
import pytest

from flight_lora.eval import (
    mape,
    pinball_loss,
    smape,
)


def test_smape_zero_for_perfect_forecast():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert smape(y_true, y_pred) == 0.0


def test_smape_symmetric():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 220.0])
    assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))


def test_smape_known_value():
    y_true = np.array([100.0])
    y_pred = np.array([90.0])
    assert smape(y_true, y_pred) == pytest.approx(0.10526315789, rel=1e-6)


def test_mape_zero_for_perfect_forecast():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([100.0, 200.0])
    assert mape(y_true, y_pred) == 0.0


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    assert mape(y_true, y_pred) == pytest.approx(0.1)


def test_pinball_loss_at_median_equals_half_l1():
    y_true = np.array([100.0, 200.0])
    y_pred_quantile = np.array([110.0, 180.0])
    expected = np.mean(0.5 * np.abs(y_true - y_pred_quantile))
    assert pinball_loss(y_true, y_pred_quantile, q=0.5) == pytest.approx(expected)


def test_pinball_loss_penalizes_under_prediction_more_at_high_quantile():
    y_true = np.array([100.0])
    pred_low = np.array([90.0])
    pred_high = np.array([110.0])
    assert pinball_loss(y_true, pred_low, q=0.9) > pinball_loss(y_true, pred_high, q=0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 7 failures (`ModuleNotFoundError: flight_lora.eval`).

- [ ] **Step 3: Implement the metrics**

```python
# src/flight_lora/eval.py
"""Forecast and decision-quality metrics, baselines, and report generation."""

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — bounded in [0, 2]; preferred for low-magnitude prices."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(diff[mask] / denom[mask]))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, computed where y_true != 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not mask.any():
        raise ValueError("MAPE undefined when all y_true are zero")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def pinball_loss(y_true: np.ndarray, y_quantile_pred: np.ndarray, q: float) -> float:
    """Quantile / pinball loss at a single quantile q in (0, 1)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_quantile_pred = np.asarray(y_quantile_pred, dtype=np.float64)
    diff = y_true - y_quantile_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/eval.py tests/test_eval.py
git commit -m "add eval point metrics: smape, mape, pinball_loss"
```

---

## Task 9: eval.py — baselines (naive last value, seasonal naive)

**Files:**
- Modify: `src/flight_lora/eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from flight_lora.eval import naive_last_value, seasonal_naive


def test_naive_last_value_replicates_last_context_step():
    context = np.array([100.0, 110.0, 120.0])
    pred = naive_last_value(context, horizon=4)
    np.testing.assert_array_equal(pred, np.full(4, 120.0))


def test_seasonal_naive_24h_repeats_last_24_hours():
    rng = np.random.default_rng(0)
    context = rng.normal(size=200)
    pred = seasonal_naive(context, horizon=48, season=24)
    np.testing.assert_array_equal(pred[:24], context[-24:])
    np.testing.assert_array_equal(pred[24:], context[-24:])


def test_seasonal_naive_168h_partial_horizon():
    context = np.arange(200.0)
    pred = seasonal_naive(context, horizon=10, season=168)
    np.testing.assert_array_equal(pred, context[-168:-158])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 3 new failures.

- [ ] **Step 3: Implement the baselines**

Append to `src/flight_lora/eval.py`:

```python
def naive_last_value(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast = last observed value, repeated."""
    if len(context) == 0:
        raise ValueError("context must be non-empty")
    return np.full(horizon, context[-1], dtype=np.float64)


def seasonal_naive(context: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Forecast = repeat the last `season` observations as needed to fill `horizon`."""
    if len(context) < season:
        raise ValueError(f"context length {len(context)} shorter than season {season}")
    last_season = np.asarray(context[-season:], dtype=np.float64)
    reps = (horizon + season - 1) // season
    return np.tile(last_season, reps)[:horizon]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: all 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/eval.py tests/test_eval.py
git commit -m "add naive baselines: naive_last_value and seasonal_naive"
```

---

## Task 10: eval.py — wait_score and decision metrics

**Files:**
- Modify: `src/flight_lora/eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
from flight_lora.eval import (
    decision_labels,
    pr_curve,
    recall_at_precision,
    wait_score_from_quantile,
)


def test_wait_score_high_when_lower_quantile_below_threshold():
    q10_forecast = np.array([85.0, 88.0, 92.0])
    score = wait_score_from_quantile(q10_forecast, current_price=100.0, drop_pct=0.10)
    assert score == pytest.approx(2.0 / 3.0)


def test_wait_score_zero_when_quantile_above_threshold():
    q10_forecast = np.array([95.0, 96.0, 97.0])
    assert wait_score_from_quantile(q10_forecast, current_price=100.0, drop_pct=0.10) == 0.0


def test_decision_labels_match_realized_drops():
    actual_min = np.array([85.0, 95.0, 80.0])
    current = np.array([100.0, 100.0, 100.0])
    labels = decision_labels(actual_min, current_price=current, drop_pct=0.10)
    np.testing.assert_array_equal(labels, np.array([True, False, True]))


def test_pr_curve_returns_monotone_recall():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
    labels = np.array([False, False, True, True, True])
    precision, recall, thresholds = pr_curve(scores, labels)
    assert all(recall[i] >= recall[i + 1] for i in range(len(recall) - 1))


def test_recall_at_precision_finds_max_recall_meeting_constraint():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
    labels = np.array([False, False, True, True, True])
    r = recall_at_precision(scores, labels, target_precision=0.8)
    assert 0.0 <= r <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 5 new failures.

- [ ] **Step 3: Implement the decision-metric functions**

Append to `src/flight_lora/eval.py`:

```python
from sklearn.metrics import precision_recall_curve


def wait_score_from_quantile(
    q_low_forecast: np.ndarray,
    current_price: float,
    drop_pct: float = 0.10,
) -> float:
    """Fraction of horizon steps where the lower-quantile prediction is below
    the drop threshold. Maps a forecast distribution to a [0, 1] "wait" score."""
    threshold = current_price * (1.0 - drop_pct)
    return float(np.mean(np.asarray(q_low_forecast) < threshold))


def decision_labels(
    actual_min: np.ndarray,
    current_price: np.ndarray,
    drop_pct: float = 0.10,
) -> np.ndarray:
    """Ground-truth labels: True if actual minimum over the horizon is at least
    `drop_pct` below the current price."""
    actual_min = np.asarray(actual_min, dtype=np.float64)
    current_price = np.asarray(current_price, dtype=np.float64)
    return actual_min < (current_price * (1.0 - drop_pct))


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wraps sklearn's precision_recall_curve for our return convention."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision, recall, thresholds


def recall_at_precision(
    scores: np.ndarray, labels: np.ndarray, target_precision: float
) -> float:
    """Maximum recall achievable while precision >= target_precision."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    mask = precision[:-1] >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[:-1][mask].max())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: all 15 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/eval.py tests/test_eval.py
git commit -m "add decision metrics: wait_score, decision_labels, pr_curve, recall_at_precision"
```

---

## Task 11: eval.py — report writer

**Files:**
- Modify: `src/flight_lora/eval.py`
- Modify: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval.py`:

```python
import json
from pathlib import Path

from flight_lora.eval import StageMetrics, write_report


def test_write_report_creates_markdown_and_summary(tmp_path: Path):
    metrics = StageMetrics(
        stage_name="zero_shot",
        smape=0.18,
        mape=0.21,
        pinball={0.1: 0.07, 0.5: 0.10, 0.9: 0.08},
        recall_at_p80=0.42,
        baselines={
            "naive_last": {"smape": 0.30, "mape": 0.35},
            "seasonal_24h": {"smape": 0.25, "mape": 0.27},
            "seasonal_168h": {"smape": 0.22, "mape": 0.24},
        },
        per_bucket_smape={"0-7d": 0.12, "7-30d": 0.18, "30+d": 0.22},
    )
    write_report(metrics, out_dir=tmp_path)

    report_md = (tmp_path / "eval_report.md").read_text()
    assert "zero_shot" in report_md
    assert "0.18" in report_md
    assert "0.30" in report_md

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["smape"] == 0.18
    assert summary["recall_at_p80"] == 0.42
    assert summary["baselines"]["naive_last"]["smape"] == 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 1 failure (`ImportError`).

- [ ] **Step 3: Add `StageMetrics` and `write_report`**

Append to `src/flight_lora/eval.py`:

```python
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageMetrics:
    """Bundle of all metrics produced for a single stage run."""
    stage_name: str
    smape: float
    mape: float
    pinball: dict[float, float] = field(default_factory=dict)
    recall_at_p80: float = 0.0
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    per_bucket_smape: dict[str, float] = field(default_factory=dict)


def write_report(metrics: StageMetrics, out_dir: Path) -> None:
    """Write `eval_report.md` and `summary.json` into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Eval report — {metrics.stage_name}")
    lines.append("")
    lines.append("## Headline forecast metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| SMAPE | {metrics.smape:.4f} |")
    lines.append(f"| MAPE  | {metrics.mape:.4f} |")
    for q, v in sorted(metrics.pinball.items()):
        lines.append(f"| Pinball@{q} | {v:.4f} |")
    lines.append(f"| Recall@P80 | {metrics.recall_at_p80:.4f} |")

    if metrics.baselines:
        lines.append("")
        lines.append("## Baselines")
        lines.append("")
        lines.append("| Model | SMAPE | MAPE |")
        lines.append("|---|---|---|")
        for name, b in metrics.baselines.items():
            lines.append(
                f"| {name} | {b.get('smape', float('nan')):.4f} | "
                f"{b.get('mape', float('nan')):.4f} |"
            )

    if metrics.per_bucket_smape:
        lines.append("")
        lines.append("## SMAPE by days-to-departure bucket")
        lines.append("")
        lines.append("| Bucket | SMAPE |")
        lines.append("|---|---|")
        for bucket, v in metrics.per_bucket_smape.items():
            lines.append(f"| {bucket} | {v:.4f} |")

    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n")

    summary = {
        "stage_name": metrics.stage_name,
        "smape": metrics.smape,
        "mape": metrics.mape,
        "pinball": {str(k): v for k, v in metrics.pinball.items()},
        "recall_at_p80": metrics.recall_at_p80,
        "baselines": metrics.baselines,
        "per_bucket_smape": metrics.per_bucket_smape,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: all 16 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/eval.py tests/test_eval.py
git commit -m "add StageMetrics + write_report for eval artifacts"
```

---

## Task 12: infer.py — model loader and forecast wrapper

**Files:**
- Create: `src/flight_lora/infer.py`
- Create: `tests/test_infer.py`

The unit test uses a fake model that satisfies the same interface; the real-model path is exercised by Stage 0 / Stage 1 runs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer.py
"""Tests for infer.py — uses a fake model to exercise the wrapper logic."""

import numpy as np
import pytest
import torch

from flight_lora.infer import ForecastResult, forecast_series


class _FakeModel:
    """Minimal stand-in for TimesFM: forecast = mean(context) for all horizon steps,
    quantiles = mean +/- 1.28 * std(context)."""

    def __init__(self) -> None:
        self.config = type("cfg", (), {"context_length": 512})()

    def to_eval_mode(self):
        return self

    @torch.no_grad()
    def __call__(self, past_values, forecast_context_len, **kwargs):
        ctx = past_values.float()
        mean = ctx.mean(dim=1, keepdim=True)
        std = ctx.std(dim=1, keepdim=True)
        horizon = kwargs.get("horizon", 168)
        point = mean.expand(-1, horizon)
        quantiles = torch.stack([
            point - 1.28 * std.expand(-1, horizon),
            point,
            point + 1.28 * std.expand(-1, horizon),
        ], dim=-1)
        out = type("Out", (), {})()
        out.point_forecast = point
        out.quantile_forecast = quantiles
        return out


def test_forecast_returns_log_space_arrays_of_correct_shape():
    fake = _FakeModel()
    log_context = np.log(np.linspace(100.0, 200.0, 600))
    result = forecast_series(fake, log_context, horizon=168, context_len=512)

    assert isinstance(result, ForecastResult)
    assert result.point.shape == (168,)
    assert result.q10.shape == (168,)
    assert result.q50.shape == (168,)
    assert result.q90.shape == (168,)


def test_forecast_dollar_space_via_exp():
    fake = _FakeModel()
    log_context = np.log(np.linspace(100.0, 200.0, 600))
    result = forecast_series(fake, log_context, horizon=4, context_len=512)
    dollar = result.to_dollar_space()
    assert (dollar.point > 0).all()
    np.testing.assert_allclose(
        dollar.point[0], np.exp(log_context.mean()), rtol=1e-5
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_infer.py -v`
Expected: 2 failures (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `infer.py`**

```python
# src/flight_lora/infer.py
"""Inference wrapper around a TimesFM (PEFT-adapted or zero-shot) model.

All public functions take and return numpy arrays; the model object itself
can be the HF transformers `TimesFm2_5ModelForPrediction` (with optional PEFT
adapter), or a fake satisfying the same interface (for tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ForecastResult:
    """Quantile forecasts in whatever space the model was trained in (log for us)."""
    point: np.ndarray  # shape (horizon,)
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray

    def to_dollar_space(self) -> "ForecastResult":
        return ForecastResult(
            point=np.exp(self.point),
            q10=np.exp(self.q10),
            q50=np.exp(self.q50),
            q90=np.exp(self.q90),
        )


def load_model(model_id: str = "google/timesfm-2.5-200m-pytorch", adapter_dir: Path | None = None):
    """Load TimesFM (HF transformers) and optionally apply a PEFT adapter."""
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = _set_eval_mode(model)
    return model


def _set_eval_mode(model):
    """Switch model to evaluation mode (PyTorch) — wrapped so the test fake
    can satisfy this without inheriting nn.Module."""
    fn = getattr(model, "to_eval_mode", None) or getattr(model, "eval", None)
    if fn is None:
        raise AttributeError("model has no eval/to_eval_mode method")
    return fn() or model


@torch.no_grad()
def forecast_series(
    model,
    log_context: np.ndarray,
    horizon: int,
    context_len: int,
) -> ForecastResult:
    """Forecast `horizon` steps from a single log-price context."""
    ctx = torch.tensor(log_context[-context_len:], dtype=torch.float32).unsqueeze(0)
    out = model(past_values=ctx, forecast_context_len=context_len, horizon=horizon)
    point = out.point_forecast.squeeze(0).cpu().numpy()
    q = out.quantile_forecast.squeeze(0).cpu().numpy()  # shape (horizon, 3)
    return ForecastResult(
        point=point,
        q10=q[:, 0],
        q50=q[:, 1],
        q90=q[:, 2],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_infer.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/infer.py tests/test_infer.py
git commit -m "add infer.py: load model + adapter, forecast wrapper returning quantiles"
```

---

## Task 13: train.py — config + LoRA setup

**Files:**
- Create: `src/flight_lora/train.py`
- Create: `tests/test_train.py`
- Create: `configs/zero_shot.yaml`
- Create: `configs/lora_uni.yaml`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train.py
"""Tests for train.py — focus on config loading and LoRA wiring."""

from pathlib import Path

import pytest

from flight_lora.train import StageConfig, load_config


def test_load_config_parses_yaml(tmp_path: Path):
    cfg_text = """
stage: lora_uni
model_id: google/timesfm-2.5-200m-pytorch
context_len: 512
horizon_len: 168
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lr: 5.0e-5
epochs: 10
batch_size: 8
num_samples_per_epoch: 100000
seed: 42
data_dir: data
out_dir: artifacts/runs/lora_uni
"""
    cfg_path = tmp_path / "lora_uni.yaml"
    cfg_path.write_text(cfg_text)
    cfg = load_config(cfg_path)
    assert cfg.stage == "lora_uni"
    assert cfg.context_len == 512
    assert cfg.horizon_len == 168
    assert cfg.lora_r == 8
    assert cfg.lr == pytest.approx(5e-5)


def test_stage_config_validates_stage_name():
    with pytest.raises(ValueError, match="stage"):
        StageConfig(
            stage="bogus", model_id="m", context_len=512, horizon_len=168,
            lora_r=8, lora_alpha=16, lora_dropout=0.05, lr=1e-4,
            epochs=1, batch_size=1, num_samples_per_epoch=10, seed=1,
            data_dir=Path("data"), out_dir=Path("out"),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: 2 failures.

- [ ] **Step 3: Implement `train.py` (config bits only)**

```python
# src/flight_lora/train.py
"""Training driver for Stage 0 (zero-shot eval) and Stage 1 (LoRA fine-tune).

Each stage reads exactly one YAML config and writes a standard run directory.
The training loop mirrors the official TimesFM finetune_lora.py recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_STAGES = {"zero_shot", "lora_uni", "xreg_eval"}


@dataclass
class StageConfig:
    stage: str
    model_id: str
    context_len: int
    horizon_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lr: float
    epochs: int
    batch_size: int
    num_samples_per_epoch: int
    seed: int
    data_dir: Path
    out_dir: Path

    def __post_init__(self) -> None:
        if self.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}; got {self.stage!r}")
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)


def load_config(path: Path) -> StageConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return StageConfig(**raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Write the two YAML configs**

```yaml
# configs/zero_shot.yaml
stage: zero_shot
model_id: google/timesfm-2.5-200m-pytorch
context_len: 512
horizon_len: 168
lora_r: 0
lora_alpha: 0
lora_dropout: 0.0
lr: 0.0
epochs: 0
batch_size: 8
num_samples_per_epoch: 0
seed: 42
data_dir: data
out_dir: artifacts/runs/zero_shot
```

```yaml
# configs/lora_uni.yaml
stage: lora_uni
model_id: google/timesfm-2.5-200m-pytorch
context_len: 512
horizon_len: 168
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lr: 5.0e-5
epochs: 10
batch_size: 8
num_samples_per_epoch: 100000
seed: 42
data_dir: data
out_dir: artifacts/runs/lora_uni
```

- [ ] **Step 6: Commit**

```bash
git add src/flight_lora/train.py tests/test_train.py configs/zero_shot.yaml configs/lora_uni.yaml
git commit -m "add train.StageConfig + load_config + zero_shot/lora_uni configs"
```

---

## Task 14: train.py — training loop and stage runners

**Files:**
- Modify: `src/flight_lora/train.py`

This task adds GPU-bound code that we can only smoke-test on the server (Tasks 18 & 19). We add no unit test for the loop itself; we add an importability test to catch obvious wiring breaks.

- [ ] **Step 1: Add the importability test**

Append to `tests/test_train.py`:

```python
def test_run_stage_zero_shot_function_exists():
    """Importing run_stage_zero_shot must not fail (smoke test for module wiring)."""
    from flight_lora.train import run_stage_zero_shot, run_stage_lora_uni
    assert callable(run_stage_zero_shot)
    assert callable(run_stage_lora_uni)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: 1 failure (`ImportError`).

- [ ] **Step 3: Implement the stage runners and training loop**

Append to `src/flight_lora/train.py`:

```python
import json
import logging
from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from flight_lora.dataset import (
    HourlySeries,
    sample_training_windows,
)
from flight_lora.eval import (
    StageMetrics,
    mape,
    naive_last_value,
    pinball_loss,
    recall_at_precision,
    seasonal_naive,
    smape,
    wait_score_from_quantile,
    write_report,
)

logger = logging.getLogger(__name__)


def _load_series(data_dir: Path, split: str) -> list[HourlySeries]:
    """Load parquet, return only series whose route_id is in the named split."""
    splits = json.loads((data_dir / "splits.json").read_text())
    keep = set(splits[split])
    rows = pl.read_parquet(data_dir / "series.parquet")
    out: list[HourlySeries] = []
    for row in rows.iter_rows(named=True):
        if row["route_id"] not in keep:
            continue
        out.append(HourlySeries(
            route_id=row["route_id"],
            departure_datetime=row["departure_datetime"],
            start_hour=row["start_hour"],
            prices=np.asarray(row["log_prices"], dtype=np.float64),
        ))
    return out


class _RandomWindowDataset(Dataset):
    def __init__(self, series: list[HourlySeries], context_len: int, horizon_len: int,
                 num_samples: int, seed: int) -> None:
        self.samples = sample_training_windows(
            series, context_len=context_len, horizon_len=horizon_len,
            num_samples=num_samples, seed=seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        ctx, tgt = self.samples[i]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def _evaluate_on_split(
    model, series: list[HourlySeries], cfg: StageConfig, stage_name: str,
) -> StageMetrics:
    """Run forecasts on the last window of each series; collect metrics."""
    from flight_lora.infer import forecast_series

    smape_vals: list[float] = []
    mape_vals: list[float] = []
    pinball_vals: dict[float, list[float]] = {0.1: [], 0.5: [], 0.9: []}
    baseline_smape = {"naive_last": [], "seasonal_24h": [], "seasonal_168h": []}
    baseline_mape = {"naive_last": [], "seasonal_24h": [], "seasonal_168h": []}
    wait_scores: list[float] = []
    truth_labels: list[bool] = []
    per_bucket: dict[str, list[float]] = {"0-7d": [], "7-30d": [], "30+d": []}

    for s in series:
        if len(s.prices) < cfg.context_len + cfg.horizon_len:
            continue
        log_context = s.prices[-cfg.context_len - cfg.horizon_len : -cfg.horizon_len]
        log_target = s.prices[-cfg.horizon_len:]

        result = forecast_series(model, log_context, horizon=cfg.horizon_len,
                                  context_len=cfg.context_len)
        dollar = result.to_dollar_space()
        actual = np.exp(log_target)

        smape_vals.append(smape(actual, dollar.point))
        mape_vals.append(mape(actual, dollar.point))
        pinball_vals[0.1].append(pinball_loss(actual, dollar.q10, q=0.1))
        pinball_vals[0.5].append(pinball_loss(actual, dollar.q50, q=0.5))
        pinball_vals[0.9].append(pinball_loss(actual, dollar.q90, q=0.9))

        ctx_dollar = np.exp(log_context)
        for name, fn in (
            ("naive_last", lambda c: naive_last_value(c, cfg.horizon_len)),
            ("seasonal_24h", lambda c: seasonal_naive(c, cfg.horizon_len, season=24)),
            ("seasonal_168h", lambda c: seasonal_naive(c, cfg.horizon_len, season=168)),
        ):
            pred = fn(ctx_dollar)
            baseline_smape[name].append(smape(actual, pred))
            baseline_mape[name].append(mape(actual, pred))

        current_price = float(ctx_dollar[-1])
        wait_scores.append(wait_score_from_quantile(dollar.q10, current_price=current_price))
        truth_labels.append(bool(actual.min() < current_price * 0.9))

        first_forecast_hour = s.start_hour + np.timedelta64(len(s.prices) - cfg.horizon_len, "h")
        days_to_dep = (s.departure_datetime - first_forecast_hour).total_seconds() / 86400.0
        bucket = "0-7d" if days_to_dep <= 7 else "7-30d" if days_to_dep <= 30 else "30+d"
        per_bucket[bucket].append(smape_vals[-1])

    metrics = StageMetrics(
        stage_name=stage_name,
        smape=float(np.mean(smape_vals)),
        mape=float(np.mean(mape_vals)),
        pinball={q: float(np.mean(v)) for q, v in pinball_vals.items()},
        recall_at_p80=recall_at_precision(
            np.asarray(wait_scores), np.asarray(truth_labels), target_precision=0.8
        ),
        baselines={
            "naive_last": {
                "smape": float(np.mean(baseline_smape["naive_last"])),
                "mape": float(np.mean(baseline_mape["naive_last"])),
            },
            "seasonal_24h": {
                "smape": float(np.mean(baseline_smape["seasonal_24h"])),
                "mape": float(np.mean(baseline_mape["seasonal_24h"])),
            },
            "seasonal_168h": {
                "smape": float(np.mean(baseline_smape["seasonal_168h"])),
                "mape": float(np.mean(baseline_mape["seasonal_168h"])),
            },
        },
        per_bucket_smape={
            k: float(np.mean(v)) if v else float("nan") for k, v in per_bucket.items()
        },
    )
    return metrics


def run_stage_zero_shot(cfg: StageConfig) -> StageMetrics:
    """Evaluate the base TimesFM model (no adapter) on val + test."""
    from flight_lora.infer import load_model

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.yaml").write_text(yaml.safe_dump({
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
    }))

    logger.info("Loading base model %s ...", cfg.model_id)
    model = load_model(cfg.model_id)

    test_series = _load_series(cfg.data_dir, "test")
    logger.info("Evaluating zero-shot on %d test series ...", len(test_series))
    metrics = _evaluate_on_split(model, test_series, cfg, stage_name="zero_shot")
    write_report(metrics, cfg.out_dir)
    return metrics


def run_stage_lora_uni(cfg: StageConfig) -> StageMetrics:
    """Train a univariate LoRA adapter, then evaluate on test."""
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.out_dir / "adapter"
    log_path = cfg.out_dir / "train_log.jsonl"
    (cfg.out_dir / "config.yaml").write_text(yaml.safe_dump({
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
    }))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Stage 1 (lora_uni) requires CUDA")

    logger.info("Loading model %s ...", cfg.model_id)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    context_len = min(cfg.context_len, model.config.context_length)

    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules="all-linear", lora_dropout=cfg.lora_dropout, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_series = _load_series(cfg.data_dir, "train")
    val_series = _load_series(cfg.data_dir, "val")
    logger.info("Train series: %d  Val series: %d", len(train_series), len(val_series))

    train_ds = _RandomWindowDataset(
        train_series, context_len=context_len, horizon_len=cfg.horizon_len,
        num_samples=cfg.num_samples_per_epoch, seed=cfg.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader)
    )

    best_val_smape = float("inf")
    patience_left = 2
    log_f = log_path.open("w")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            outputs = model(
                past_values=context,
                future_values=target,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            running += float(loss.item())
            n_batches += 1

        avg_train_loss = running / max(n_batches, 1)

        model.eval()
        val_metrics = _evaluate_on_split(model, val_series, cfg, stage_name="lora_uni_val")

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_smape=%.4f",
            epoch, cfg.epochs, avg_train_loss, val_metrics.smape,
        )
        log_f.write(json.dumps({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_smape": val_metrics.smape,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        log_f.flush()

        if val_metrics.smape < best_val_smape - 1e-4:
            best_val_smape = val_metrics.smape
            patience_left = 2
            model.save_pretrained(str(adapter_dir))
            logger.info("  saved adapter (new best val_smape=%.4f)", best_val_smape)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("  early stopping at epoch %d", epoch)
                break

    log_f.close()

    logger.info("Reloading best adapter for test eval ...")
    from peft import PeftModel
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    test_series = _load_series(cfg.data_dir, "test")
    test_metrics = _evaluate_on_split(model, test_series, cfg, stage_name="lora_uni")
    write_report(test_metrics, cfg.out_dir)
    return test_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flight_lora/train.py tests/test_train.py
git commit -m "add training loop and stage runners (zero_shot + lora_uni)"
```

---

## Task 15: scripts/run_stage.py — CLI orchestrator

**Files:**
- Create: `scripts/run_stage.py`

- [ ] **Step 1: Implement the CLI**

```python
# scripts/run_stage.py
"""CLI dispatcher for stage runs.

Reads a single YAML config and calls the matching runner. The runner is
responsible for reading data, writing artifacts, and computing the gate
result; this script only logs the outcome and exits with a non-zero code
if a stage's ship gate fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from flight_lora.train import (
    StageConfig,
    load_config,
    run_stage_lora_uni,
    run_stage_zero_shot,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(".env.server" if os.environ.get("FLIGHT_LORA_ENV") == "server" else ".env.local")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg: StageConfig = load_config(args.config)

    if cfg.stage == "zero_shot":
        metrics = run_stage_zero_shot(cfg)
        print(json.dumps({"stage": "zero_shot", "smape": metrics.smape}))
        return 0

    if cfg.stage == "lora_uni":
        metrics = run_stage_lora_uni(cfg)
        zs_path = Path("artifacts/runs/zero_shot/summary.json")
        if not zs_path.exists():
            print(f"WARN: zero-shot summary missing at {zs_path}; skipping gate")
            return 0
        zs = json.loads(zs_path.read_text())
        smape_drop = (zs["smape"] - metrics.smape) / zs["smape"]
        recall_gain = metrics.recall_at_p80 - zs["recall_at_p80"]
        gate = smape_drop >= 0.15 and recall_gain >= 0.10
        result = {
            "stage": "lora_uni",
            "test_smape": metrics.smape,
            "zero_shot_smape": zs["smape"],
            "smape_drop_frac": smape_drop,
            "recall_at_p80": metrics.recall_at_p80,
            "zero_shot_recall_at_p80": zs["recall_at_p80"],
            "recall_gain_pp": recall_gain * 100.0,
            "gate_passed": gate,
        }
        print(json.dumps(result, indent=2))
        return 0 if gate else 2

    print(f"unsupported stage: {cfg.stage}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify imports work locally**

Run: `uv run python -c "import scripts.run_stage; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_stage.py
git commit -m "add scripts/run_stage.py: CLI dispatcher with ship-gate exit code"
```

---

## Task 16: scripts/remote_train.sh

**Files:**
- Create: `scripts/remote_train.sh`

- [ ] **Step 1: Write the remote runner**

```bash
#!/usr/bin/env bash
# scripts/remote_train.sh — ssh to the GPU server, start (or attach to)
# a per-stage tmux session, and run the corresponding stage.
#
# Usage: bash scripts/remote_train.sh <zero_shot|lora_uni|xreg_eval>

set -euo pipefail

STAGE="${1:?usage: $0 <stage>}"
SERVER="${SERVER:-gpu-host}"
REMOTE_DIR="${REMOTE_DIR:-~/flight-prediction}"
SESSION="flight-lora-${STAGE}"
CONFIG_PATH="configs/${STAGE}.yaml"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
cd ${REMOTE_DIR}
export FLIGHT_LORA_ENV=server
mkdir -p artifacts/runs
if ! tmux has-session -t ${SESSION} 2>/dev/null; then
  tmux new-session -d -s ${SESSION} \
    "uv run python scripts/run_stage.py --config ${CONFIG_PATH} 2>&1 | tee artifacts/runs/${STAGE}.log"
  echo "Started tmux session ${SESSION}; tail with: tmux attach -t ${SESSION}"
else
  echo "Session ${SESSION} already exists; attach with: tmux attach -t ${SESSION}"
fi
EOF

ssh "${SERVER}" "${REMOTE_CMD}"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/remote_train.sh`

- [ ] **Step 3: Lint & sanity check the script**

Run: `bash -n scripts/remote_train.sh`
Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/remote_train.sh
git commit -m "add scripts/remote_train.sh: ssh+tmux wrapper for per-stage runs"
```

---

## Task 17: Server-side environment + Mutagen sync

This task is operational, not code. Success is verified by `make sync` succeeding.

- [ ] **Step 1: SSH to the server and create the project dir**

Run (locally): `ssh gpu-host "mkdir -p ~/flight-prediction"`

- [ ] **Step 2: Install `uv` on the server if not present**

Run: `ssh gpu-host "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh"`

- [ ] **Step 3: Start the mutagen sync session**

Run (locally, from the project root): `make sync`
Expected: prints sync session creation; `mutagen sync list` shows `flight-lora` as `Watching for changes`.

- [ ] **Step 4: Create `.env.server` on the server (NOT synced)**

```bash
ssh gpu-host "cat > ~/flight-prediction/.env.server" <<'EOF'
DB_HOST=localhost
DB_PORT=5433
DB_USER=unfare_user
DB_PASSWORD=unfare_pass
DB_NAME=unfare_db
DATASET_CUTOFF=2026-05-03T00:00:00Z
EOF
```

- [ ] **Step 5: Install Python deps on the server**

Run: `ssh gpu-host "cd ~/flight-prediction && uv sync --all-extras"`
Expected: completes; CUDA-enabled torch is installed.

- [ ] **Step 6: Verify CUDA is visible**

Run: `ssh gpu-host "cd ~/flight-prediction && uv run python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"`
Expected: `True <gpu-name>` (e.g., `True NVIDIA GeForce RTX 4090`).

- [ ] **Step 7: Run the dataset build on the server**

Run: `ssh gpu-host "cd ~/flight-prediction && FLIGHT_LORA_ENV=server uv run python scripts/build_dataset.py"`
Expected: prints series counts; `data/series.parquet` and `data/splits.json` exist on the server.

- [ ] **Step 8: No commit** (this task changes server state, not files in the repo)

---

## Task 18: Run Stage 0 — Zero-shot baseline

- [ ] **Step 1: Trigger the run**

Run: `make baseline`
Expected: prints `Started tmux session flight-lora-zero_shot; tail with: tmux attach -t flight-lora-zero_shot`.

- [ ] **Step 2: Watch logs**

Run: `ssh gpu-host "tmux attach -t flight-lora-zero_shot"`
Expected: training script logs progress; eventually prints a JSON line with `{"stage": "zero_shot", "smape": <number>}`.
(Detach with `Ctrl-b d`.)

- [ ] **Step 3: Pull artifacts back to the Mac**

Run: `make pull-artifacts`
Expected: `artifacts/runs/zero_shot/{eval_report.md, summary.json, config.yaml}` appear locally.

- [ ] **Step 4: Inspect the report**

Run: `cat artifacts/runs/zero_shot/eval_report.md`
Expected: a markdown table with SMAPE, MAPE, Pinball@{0.1, 0.5, 0.9}, recall@P80, baselines, and per-bucket SMAPE.

- [ ] **Step 5: Commit the eval results into the repo**

```bash
git add artifacts/runs/zero_shot/eval_report.md \
        artifacts/runs/zero_shot/summary.json \
        artifacts/runs/zero_shot/config.yaml
git commit -m "record Stage 0 (zero-shot) baseline results"
```

(Reasoning: we keep eval reports and summaries in git as a permanent record of each stage's outcome. We do NOT commit raw checkpoints / parquet — those stay server-only via `.gitignore`.)

---

## Task 19: Run Stage 1 — Univariate LoRA + decide gate

- [ ] **Step 1: Trigger the run**

Run: `make train-uni`
Expected: prints `Started tmux session flight-lora-lora_uni; tail with: tmux attach -t flight-lora-lora_uni`.

- [ ] **Step 2: Monitor training**

Run: `ssh gpu-host "tmux attach -t flight-lora-lora_uni"`
Expected: per-epoch lines like `Epoch 1/10  train_loss=0.0432  val_smape=0.1521`.
(Detach with `Ctrl-b d`. Training should take ~2-6 hours on a 24GB consumer GPU.)

- [ ] **Step 3: Check exit status (after the tmux session ends)**

Run: `ssh gpu-host "cd ~/flight-prediction && cat artifacts/runs/lora_uni.log | tail -40"`
Expected: a final JSON object containing `gate_passed: true` or `false`.

- [ ] **Step 4: Pull artifacts**

Run: `make pull-artifacts`
Expected: `artifacts/runs/lora_uni/{eval_report.md, summary.json, train_log.jsonl, config.yaml}` locally. The `adapter/` directory stays server-side (excluded from rsync via the `--exclude='adapter_model.safetensors'` rule).

- [ ] **Step 5: Inspect**

Run: `cat artifacts/runs/lora_uni/eval_report.md`
Expected: SMAPE, MAPE, etc. — compare against `artifacts/runs/zero_shot/eval_report.md`.

- [ ] **Step 6: Decision point**

Read the gate result line from Step 3:

- **If `gate_passed: true`**: the v1 univariate LoRA is shippable. Commit the eval artifacts; v2 batch scorer is the next plan.

```bash
git add artifacts/runs/lora_uni/eval_report.md \
        artifacts/runs/lora_uni/summary.json \
        artifacts/runs/lora_uni/train_log.jsonl \
        artifacts/runs/lora_uni/config.yaml
git commit -m "record Stage 1 (lora_uni) gate-pass results"
```

- **If `gate_passed: false`**: do NOT ship. Commit the eval artifacts as a record of the failed run, then diagnose. Common failure modes (in order of likelihood):
  1. **Log-transform interaction with TimesFM's RevIN** (per spec Risks). Sanity-check: re-run with raw prices (skip `log_transform`) by adding a `--no-log` flag to `build_dataset.py`. If that beats zero-shot but log-transform doesn't, the layered normalization is fighting itself.
  2. **Insufficient train data**. Bump `num_samples_per_epoch` in `configs/lora_uni.yaml` to 250000 and re-run.
  3. **LoRA rank too low**. Try `lora_r: 16, lora_alpha: 32`.
  4. **Wrong target modules**. Inspect what `target_modules="all-linear"` actually adapted via `model.print_trainable_parameters()` in the training log.

```bash
git add artifacts/runs/lora_uni/eval_report.md \
        artifacts/runs/lora_uni/summary.json \
        artifacts/runs/lora_uni/train_log.jsonl \
        artifacts/runs/lora_uni/config.yaml
git commit -m "record Stage 1 (lora_uni) gate-fail results — see eval_report.md"
```

---

## Plan complete

After Task 19 succeeds (or after a deliberate choice to halt on failure), the v1 implementation is done. Next steps (out of scope for this plan, captured here for handoff):

- **Stage 1.5 XReg overlay evaluation** (spec Future Work): add a `run_stage_xreg_eval(cfg)` function in `train.py` that loads the Stage 1 adapter, runs `forecast_with_covariates(xreg_mode="xreg + timesfm")` over the test set, writes an eval report under `artifacts/runs/xreg_eval/`, and gates on >=3% SMAPE improvement over Stage 1.
- **v2 batch scorer**: a separate plan; reuses `infer.py` and adds `scripts/run_batch_scorer.py` plus a `flight_predictions` table migration.
