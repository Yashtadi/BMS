# ΔZ (deltaZ) Feature Table

This document explains the **ΔZ (deltaZ) baseline and feature table** — the bridge between the raw
labeled spectra (`data_exploration.md`, Stage 0) and the physics-residual model (Stage 2 README). It
covers what ΔZ is, how the baseline is chosen, the two output tables and every column in them, the real
findings from running this on the actual 228-cell dataset, and how to run the pipeline yourself.

Code: `src/features/deltaz.py` (core logic) and `src/features/build_and_analyze_deltaz.py` (runnable
script with diagnostics).

---

## Part 0 — Data source

**Input:** `data/interim/eis_labeled.parquet`.

This is **not** rebuilt by anything in this document — it is the finished Stage 0 output, produced on
the **`ECM-v2` branch**, and is the exact same file described column-by-column in `data_exploration.md`
Part 7. Everything here treats it as a fixed, upstream input: 228 cells, 39,368 spectra, 1,102,304
rows, `soh_cap` present on 100% of rows.

If your local `eis_labeled.parquet` doesn't match those counts, **stop and re-pull/rebuild it** before
running anything below — a mismatch means your local data or environment is out of sync with the rest
of the team, and building ΔZ on top of it will just produce numbers nobody else can reproduce.

---

## Part 1 — What ΔZ is

For every frequency point, every check-up, and every cell:

```
deltaZ(f) = Z_baseline(f) − Z_now(f)
```

`Z_baseline(f)` is **that cell's own healthy reference spectrum** — not a population average, not a
manufacturer spec value. This matches the convention locked in `data_exploration.md` §6.2, chosen so
it lines up with how the dataset's own `z_ref_init` is defined.

**Important: the baseline is frequency-specific, not one number per cell.** `Z_baseline` is a full
28-point spectrum, and the subtraction happens frequency-by-frequency — the 0.05 Hz reading is only
ever compared against the 0.05 Hz baseline, never mixed with any other frequency.

---

## Part 2 — How the baseline is chosen

The baseline for each `(cell_id, soc_nom, is_rt, freq_Hz)` combination is: **the earliest check-up at
which that exact combination was measured and flagged `valid`.**

This is *not* a blind "always use check-up 0" rule. If check-up 0 was invalid at a given frequency, the
next valid check-up for that same `(cell, SoC, pass, freq)` is used instead. This fallback happens
automatically as a side effect of sorting by `cu_index` and taking the first valid row per group — no
separate retry logic needed.

### Real numbers from the 228-cell dataset

- **Total baselines: 63,840** = 228 cells × 5 SoC levels × 2 passes × 28 frequencies. Each cell has its
  own 280 baseline values (5 × 2 × 28), not a single reference number.
- **62,496 (97.9%)** came from check-up 0, as expected.
- **1,344 (2.1%)** had to fall back to check-up 1 — none needed to fall back further, meaning the
  invalid check-up-0 readings are scattered, not clustered into cells with multiple consecutive bad
  early check-ups.
- Of the full dataset's 1,102,304 rows, only **63,840 (5.8%)** are baseline rows themselves (flagged
  via `is_baseline_row`); the remaining **1,038,464 (94.2%)** are the actual "how has this cell changed
  since baseline" measurements ΔZ is built to capture.

---

## Part 3 — Output tables

### `deltaz_long.parquet` — full resolution

One row per frequency point (same shape as `eis_labeled.parquet`: 1,102,304 rows), with four columns
added:

| Column | Meaning |
|---|---|
| `baseline_cu_index` | Which check-up supplied the baseline for this row's `(cell, SoC, pass, freq)` |
| `Z_baseline_mOhm` | The healthy-reference Z″ value at this exact frequency |
| `deltaZ_imag_mOhm` | `Z_baseline_mOhm − Z_imag_mOhm`. **NaN** if the current row's `valid == 0** — never computed from a flagged-bad reading |
| `is_baseline_row` | `True` if this row *is* the baseline itself (`cu_index == baseline_cu_index`) — its own deltaZ is 0 by construction and isn't a real "aged" data point |

### `deltaz_features.parquet` — one row per spectrum

Collapses `deltaz_long` down to 39,368 rows (one per spectrum), with three candidate summary
statistics plus quality-tracking columns:

| Column | Meaning |
|---|---|
| `deltaz_mean_all` | Mean `deltaZ_imag_mOhm` across **all 28** frequency points |
| `deltaz_mean_lowfreq` | Mean `deltaZ_imag_mOhm` restricted to the **1–5 Hz** band (the aging-sensitive region `z_ref_init` is anchored to) |
| `deltaz_integrated_abs` | Sum of **\|deltaZ\|** across all 28 points — absolute value taken first so positive/negative swings across the spectrum don't cancel out |
| `n_valid_points` | How many of the 28 frequency points were flagged `valid` for this spectrum |
| `n_total_points` | Always 28 — included so `n_valid_points / n_total_points` gives a completeness ratio at a glance |
| `baseline_cu_index` | Which check-up the baseline for this spectrum's SoC/pass came from |

These three summary columns exist specifically to feed the Stage 2 README's Part 3 decision: fit each
against `soh_cap` with the three candidate functional forms (exponential, power law, linear) and pick
whichever (statistic, form) pair gives the best R² — that becomes `f_physics`.

---

## Part 4 — How to run this

**Prerequisites:** the `bms` conda environment active, with `data/interim/eis_labeled.parquet` already
present (see Part 0 — this is the `ECM-v2` branch's output, not something this pipeline builds itself).

```powershell
conda activate bms
cd BMS\src\features
python build_and_analyze_deltaz.py ..\..\data\interim\eis_labeled.parquet
```

This runs end-to-end and:
1. Loads `eis_labeled.parquet` and checks it against the expected shape (Part 0).
2. Builds the baseline table (Part 2) and reports how many baselines needed the check-up-0-invalid
   fallback.
3. Builds `deltaz_long` (full resolution) and `deltaz_features` (per-spectrum summary), reporting the
   `n_valid_points` distribution.
4. Flags any spectra with zero valid points.
5. Prints a raw correlation check of each candidate summary statistic against `soh_cap` (sanity check
   only — not the real Part 3 model-selection comparison in the Stage 2 README).
6. Saves both output files **next to the input file** — i.e. into `data/interim/`, alongside
   `eis_labeled.parquet` — regardless of which folder you ran the script from.

**Expected output shapes**, if your `eis_labeled.parquet` matches Part 0's counts:

| File | Rows | Notes |
|---|---|---|
| `deltaz_long.parquet` | 1,102,304 | Same row count as the input |
| `deltaz_features.parquet` | 39,368 | One row per spectrum |

If your numbers differ from these, or from the specific findings in Part 5 below, stop and compare
against a teammate's run before using the output — same rule as Stage 0's verification step.

To use the module directly instead of the script (e.g. inside a notebook):
```python
import pandas as pd
from src.features.deltaz import build_deltaz_table

df = pd.read_parquet("data/interim/eis_labeled.parquet")
deltaz_long, deltaz_features = build_deltaz_table(df)
```

---

## Part 5 — Findings from the real data run

### 75 spectra have zero valid frequency points
All 28 points flagged invalid — every summary statistic is `NaN` for these. Concentrated at
`soc_nom == 10, is_rt == 0` (44 of 75; another 10 at `soc_nom == 70, is_rt == 0`). 69 distinct cells
affected. This lines up with the KIT paper's own documented test-bench interruptions during some
cells' first EIS procedure — a real gap in the source data, not a bug in the builder.

> **Open decision:** filter policy for these 75 spectra before modeling (drop, or impute, or exclude
> only from `f_physics` fitting but keep for other diagnostics). Not yet decided.

### Raw correlation against `soh_cap` (sanity check only — not the real Part 3 comparison)

| Statistic | r |
|---|---|
| `deltaz_mean_all` | **+0.715** |
| `deltaz_integrated_abs` | **−0.727** |
| `deltaz_mean_lowfreq` | +0.617 |

The full-spectrum statistics both outperform the low-frequency-only restriction in raw linear
correlation. This doesn't decide `f_physics`'s form (that needs the actual curve-fit comparison,
Stage 2 README Part 3) but is a useful early signal that restricting to 1–5 Hz may be discarding
usable signal rather than isolating it.

---

## Part 6 — Decisions log

| # | Decision | Why | Reversible? |
|---|---|---|---|
| 1 | Baseline = earliest **valid** check-up per `(cell, SoC, pass, freq)`, not blindly check-up 0 | 2.1% of check-up-0 points are invalid; blind cu_index==0 use would silently bake in bad readings | yes |
| 2 | `deltaZ` is `NaN` (not computed) when the current reading is invalid | Avoids producing a garbage number that looks like a real value | yes |
| 3 | `deltaz_integrated_abs` uses `abs()` before summing | Prevents positive/negative swings across the spectrum from canceling out | yes |
| 4 | Merge uses `validate="many_to_one"` | Same guardrail pattern that caught the silent duplication bug in the Stage 0 recipe-metadata join | — |
| — | Filter policy for the 75 zero-valid-point spectra | **Not yet decided** | — |
| — | Which (statistic, functional form) becomes `f_physics` | **Not yet decided** — needs the Part 3 curve-fit comparison | — |

---

## Part 7 — What's next

Run the Stage 2 README Part 3 comparison: fit exponential / power-law / linear forms against each of
the three summary statistics here, using `scipy.optimize.curve_fit`, and compare R² to lock in
`f_physics`.