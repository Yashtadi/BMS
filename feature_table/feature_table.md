# Feature Table — build, filter, and orthogonalize

This document covers the three scripts that turn your raw ECM and deltaZ outputs into the final,
orthogonalized feature table used by downstream models (e.g. `fit_monotonic_gbm.py`, documented
separately in `gbm.md`). Read `physics_residual_model.md` first for the concepts behind `deltaZ`,
the ECM parameters, and why orthogonalization is needed at all (Part 2).

All commands below use forward slashes in paths, which work on Windows, macOS, and Linux alike, and
are kept on a single line so no shell-specific line-continuation character is needed.

```
ecm_features_v2.parquet  ---.
deltaz_features.parquet  ---+--> build_feature_table.py --> stage2_features.parquet
eis_labeled.parquet      ---'
                                        |
                                        v
                            filter_stage2_features.py
                                        |
                                        v
                        stage2_features_filtered.parquet
                                        |
                                        v
                          orthogonalize_features.py
                                        |
                                        v
              stage2_features_filtered_orthogonalized.parquet
                                   (final output --
                              ready for any downstream model)
```

---

## Part 1 — build_feature_table.py

Merges three separate sources into one table, one row per spectrum:

- **`ecm_features_v2.parquet`** — fitted ECM circuit parameters (`R_ohm`, `R_1arc`, `sigma_1arc`, etc.)
  from `fit_ecm.py`.
- **`deltaz_features.parquet`** — aggregate deltaZ summary statistics (`deltaz_mean_all`,
  `deltaz_mean_lowfreq`, `deltaz_integrated_abs`) from `deltaz.py`.
- **`eis_labeled.parquet`** — used only to pull `soh_cap` (the target) and `aging_type`/`temp_degC`
  back onto each spectrum, since those live in the original labeled table, not the ECM/deltaZ outputs.

```python
soh = (
    labeled.groupby(SPECTRUM_KEYS)[["soh_cap", "aging_type", "temp_degC"]]
    .first()
    .reset_index()
)
merged = ecm.merge(deltaz, on=SPECTRUM_KEYS, how="inner", suffixes=("", "_deltaz"))
merged = merged.merge(soh, on=SPECTRUM_KEYS, how="left", validate="one_to_one")
```

**`validate="one_to_one"`** is a guardrail: it stops the script immediately if a spectrum somehow
matches more than one row (e.g. a bug in the `groupby` collapsing step, or an unexpected duplicate
key), rather than silently producing duplicated rows with possibly conflicting `soh_cap` values
downstream. Fail loudly at the merge, not quietly three steps later.

**Optional `--deltaz-wide`:** merges in 28 individual per-frequency `deltaz_f_*` columns (from
`build_deltaz_wide.py`). Only needed if you specifically want to compare individual frequencies
against the aggregate summaries as candidate model inputs — not required for normal use.

### How to run

```
python build_feature_table.py --ecm path/to/ecm_features_v2.parquet --deltaz path/to/deltaz_features.parquet --labeled path/to/eis_labeled.parquet --out path/to/stage2_features.parquet
```

**Output:** `stage2_features.parquet` — prints row counts, `fit_succeeded` count, and how many rows are
missing `soh_cap`/`deltaz_mean_all` so you can sanity-check the merge before moving on.

---

## Part 2 — filter_stage2_features.py

Drops rows where the underlying measurement or circuit fit can't be trusted. Two independent checks,
either one is enough to drop a row:

**1. Kramers-Kronig validation failure (`kk_valid == False`)** — the raw measured spectrum itself
violated the linearity/causality/stability assumptions EIS requires (e.g. the cell drifted mid-sweep).
Computed upstream in `fit_ecm.py`; this script just acts on the flag.

**2. Implausible one-arc fit** — any of:
```python
(df["R_ohm_1arc"] < 1.0)        # near-zero ohmic resistance -- degenerate fit
| (df["R_1arc"] > 150.0)        # absurdly large mid-frequency resistance
| (df["n_1arc"] >= 0.999)       # CPE exponent pinned at its upper bound
| (df["n_1arc"] <= 0.001)       # CPE exponent pinned at its lower bound
```
A real 18650 cell never has near-zero ohmic resistance, and `R_1arc` above ~150 mOhm is far beyond even
severely aged behavior — these indicate the optimizer converged to a degenerate solution rather than a
physically meaningful one. A CPE exponent pinned exactly at 0 or 1 means the optimizer hit a hard bound
rather than finding a genuine value.

### The drop-concentration check — read this before trusting the output

Before saving anything, the script prints **where the dropped rows concentrate**, broken down by life
stage (`cu_index` bucketed into early/mid/late) and by SoC:

```
=== Where the dropped rows concentrate ===
By life stage (drop rate):
By SoC (drop rate):
```

**This matters.** If drops cluster heavily at *late* check-ups, filtering would systematically remove
your most-aged, most-informative spectra — biasing any downstream model toward only fitting the easy,
healthy end of the aging curve. If that happens, consider downweighting those rows (e.g. via
`curve_fit`'s `sigma` parameter) rather than dropping them outright.

**Real result on the full dataset:** 95.7% retention (37,693 of 39,368 rows), **zero cells removed
entirely**, `soh_cap` range fully preserved (-35.11 to 98.04, identical before and after), and drop
rates *higher* in early life than late — the opposite of the risky pattern. Filtering was safe here.

### How to run

```
python filter_stage2_features.py --file path/to/stage2_features.parquet
```
Or with an explicit output path:
```
python filter_stage2_features.py --file path/to/stage2_features.parquet --out path/to/stage2_features_filtered.parquet
```

**Output:** `stage2_features_filtered.parquet` (defaults to `<input>_filtered.parquet` next to the
input file).

---

## Part 3 — orthogonalize_features.py

Runs a VIF (variance inflation factor) check, then orthogonalizes the ECM parameters against your
chosen `deltaZ` column — regressing each ECM parameter on `deltaZ` and keeping only the *residual*
(the part `deltaZ` doesn't already explain). This exists because ECM parameters and `deltaZ` are
derived from the same underlying spectrum and are often strongly collinear — without this step, a
downstream model has no principled way to know which feature deserves credit for a given prediction.

**This replaces `fit_f_physics.py`'s VIF/orthogonalization step**, without also fitting or comparing
`f_physics` candidate curve forms — use this when your downstream model doesn't use `f_physics` at all
(e.g. a GBM that takes `deltaZ` and the ECM features together, rather than a separate physics term).

```python
def orthogonalize(df, target_col, against_col):
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return residual
```
For each ECM parameter (`R_ohm`, `R_1arc`, `sigma_1arc`), this fits a simple straight line predicting
it from `--deltaz-col`, then keeps only the leftover — the `_orth` columns (`R_ohm_orth`,
`R_1arc_orth`, `sigma_1arc_orth`) added to the output.

**`--deltaz-col` matters — it must match whatever deltaZ column your downstream model actually treats
as primary.** Orthogonalizing against the wrong column leaves the ECM features still partly collinear
with the one your model actually uses, undermining the point of the step. Default is
`deltaz_integrated_abs` (matches `fit_monotonic_gbm.py`'s `DELTAZ_COL`) — change it explicitly if your
downstream model uses a different summary statistic.

### How to run

```
python orthogonalize_features.py --file path/to/stage2_features_filtered.parquet
```
To orthogonalize against a different deltaZ column:
```
python orthogonalize_features.py --file path/to/stage2_features_filtered.parquet --deltaz-col deltaz_mean_all
```

**Output:** `stage2_features_filtered_orthogonalized.parquet` — prints the VIF table before and after
orthogonalization (the "after" VIF for each `_orth` column vs. your chosen `deltaz-col` should land
close to 1, confirming the collinearity was actually removed).

---

## Part 4 — Running these on different systems

All three scripts are plain Python + pandas — nothing OS-specific in the code itself. What varies:
- **How you write the path** — forward slashes (`data/interim/file.parquet`) work everywhere Python
  runs; backslashes only matter inside Windows shell syntax, not inside the Python argument itself.
- **How you activate your environment** — `conda activate bms` (or your environment's name) works the
  same on Windows, macOS, and Linux once conda is installed.
- **Line continuation for long commands** — differs by shell (`` ` `` in PowerShell, `\` in bash/zsh,
  `^` in `cmd.exe`). Commands here are single-line specifically to avoid this.

**Dependencies:** `pandas`, `numpy`, `pyarrow` (for parquet), and `statsmodels` (for the VIF check in
Part 3).

---

