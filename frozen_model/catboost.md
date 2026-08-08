# tune_earlylife_catboost.py

Searches CatBoost hyperparameters to minimise error on the **early-life** data
(`cu_index <= --early-life-max-cu`), using Optuna over the standard boosting
knobs. Trains only on early-life, selects on an early-life validation split, and
then reports the winning config's error on `test_early` **and** later-life so you
can see how the tuned model generalises past the training window.

## What it does

1. Loads the orthogonalized feature table, keeps `fit_succeeded` rows, and takes
   the early-life pool (`cu_index <= 5` by default).
2. Splits it into train / val / test_early with a **stratified, recipe-balanced,
   cell-grouped** three-way split (`StratifiedGroupKFold`) — every aging recipe
   (Calendar / Cyclic / Profile) appears in each split, and no cell leaks across
   splits.
3. Runs an Optuna search over `depth`, `learning_rate`, `l2_leaf_reg`,
   `subsample`, `random_strength`, `min_data_in_leaf`, selecting the trial with
   the lowest **validation** score.
4. Retrains the winning config and a fixed baseline config, and prints MAE / RMSE
   for both across all splits (train / val / test_early / later_life / all).
5. Saves the best params, the tuned model, per-row predictions for the whole
   dataset, and a per-check-up MAE table.

## Why it optimises on VAL, not later-life

The search only ever sees early-life data. Tuning against later-life would let
the model peek at the drift regime a frozen early-life model is meant to be blind
to, which would defeat the purpose of the frozen-model framing. The later-life
numbers are reported **after** the search, purely to show how the tuned model
generalises — they never influence which trial is chosen.

> Empirically, tuning against early-life validation improved in-window error but
> did **not** improve (and slightly worsened) later-life error, because the drift
> regime is invisible to an early-life search. Keep this in mind when deciding
> whether to deploy the tuned params or the conservative baseline.

## Important: loss function vs. selection metric

CatBoost's `loss_function="MAE"` **breaks badly when combined with the monotone
constraint** on `deltaz_integrated_abs` (MAE ~18 instead of ~4) — the constant-
sign MAE gradient interacts poorly with the monotone projection. The script
therefore always **trains with `loss_function="RMSE"`** (which is well-behaved)
while the Optuna objective **selects trials by validation MAE**. You optimise the
metric you care about without triggering the broken optimiser. Do not switch the
training loss to MAE.

The monotone constraint on `deltaz_integrated_abs` (`-1`) is fixed across every
trial — the search tunes the boosting knobs, not the physical prior.

## Inputs

- `--file` — orthogonalized feature table
  (default `data/interim/stage2_features_filtered_orthogonalized.parquet`).
- `--early-life-max-cu` — early-life cutoff (default 5).
- `--val-size`, `--test-size` — split fractions (default 0.2 each).
- `--n-trials` — number of Optuna trials (default 50).
- `--max-iterations` — cap on boosting rounds per trial (default 1500); early
  stopping usually finishes sooner. Lower it to speed up the search.
- `--seed` — split + sampler seed (default 0).
- `--task-type` — `CPU` or `GPU`.
- `--output-dir` — where outputs are written (defaults next to `--file`).

Feature columns used: `deltaz_integrated_abs` + `R_ohm_orth`, `R_1arc_orth`,
`sigma_1arc_orth` + `soc_nom`, `temp_degC`, `is_rt`.

## Outputs

Written to `--output-dir`, prefixed with the input file's stem:

- `..._catboost_best_params.json` — the winning hyperparameters, best val score,
  early-life cutoff, seed.
- `..._catboost_tuned.cbm` — the trained tuned model.
- `..._catboost_predictions.parquet` — per-row predictions for the **entire**
  dataset (cu 0–28): `cell_id`, `cu_index`, `soc_nom`, `is_rt`, `temp_degC`,
  `aging_type`, `split`, `soh_cap`, `pred_tuned`, `pred_baseline`,
  `abserr_tuned`, `abserr_baseline`.
- `..._catboost_mae_per_checkup.csv` / `.parquet` — per-check-up MAE / RMSE for
  tuned and baseline.

The predictions parquet is the input consumed by the downstream diagnosis and
SHML scripts.

## Console output

- The early-life split sizes and recipe counts per split.
- Every Optuna trial: trial number, its val MAE, best-so-far, the full parameter
  set, and a `*NEW*` marker when a trial becomes the best.
- The best val MAE and params.
- A tuned-vs-baseline MAE / RMSE table by split.
- The tuned model's later-life MAE per check-up, with the baseline alongside.

## Usage

The script lives in the `frozen_model/` folder. Run from the repo root.

```bash
python frozen_model/tune_earlylife_catboost.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --n-trials 50 --output-dir frozen_model/results
```

Faster search (fewer trials, lower iteration cap):

```bash
python frozen_model/tune_earlylife_catboost.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --n-trials 20 --max-iterations 600 --output-dir frozen_model/results
```

## Dependencies

`pandas`, `numpy`, `pyarrow`, `scikit-learn`, `catboost`, `optuna`.