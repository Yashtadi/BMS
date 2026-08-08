# Frozen-model SoH diagnosis & SHML

This branch contains the model-selection, tuning, and self-healing (SHML)
workflow built on top of the Stage-2 feature table. The pipeline trains a
State-of-Health (SoH) model on **early-life** battery data, freezes it, watches
how its error drifts as cells age, attributes that drift to physical (ECM)
mechanisms, and proposes how the model should adapt.

The input to everything here is the orthogonalized feature table produced by the
feature-table pipeline (`build_feature_table.py` → `filter_stage2_features.py` →
`orthogonalize_features.py`):
`data/interim/stage2_features_filtered_orthogonalized.parquet`.

## Workflow overview

```
      raw EIS + capacity data
                |
                v
         feature_table/
   build -> filter -> orthogonalize
                |
                v
  stage2_features_filtered_orthogonalized.parquet
                                     |
        ┌────────────────────────────┼────────────────────────────┐
        v                            v                             v
  model_choice/               frozen_model/                      shml/
  pick the model                tune it                          diagnose drift
                                                                 + adapt (SHML)
```

### 1. `feature_table/` — build the feature table

Turns the raw ECM fits, deltaZ features, and SoH labels into the single
orthogonalized table every downstream stage consumes.

- `build_feature_table.py` — merges `ecm_features_v2` + `deltaz_features` +
  `soh_cap` (and optionally the per-frequency `deltaz_wide`) into one row per
  spectrum.
- `filter_stage2_features.py` — drops KK-validation failures and implausible
  one-arc fits, and reports where the dropped rows concentrate.
- `orthogonalize_features.py` — VIF check, then orthogonalizes the ECM
  parameters against `deltaz_integrated_abs` so downstream models can attribute
  credit cleanly.
- `README_build_feature_table.md`, `README_filter_stage2_features.md`,
  `README_orthogonalize_features.md` — details, including every deltaZ attribute.

Output: `data/interim/stage2_features_filtered_orthogonalized.parquet` — the
input to every stage below.

### 2. `model_choice/` — pick the base model

Compares LightGBM, XGBoost, CatBoost, and their equal-weight ensemble on the
same early-life split, and documents why **CatBoost** was chosen (lowest
later-life MAE/RMSE — the regime that matters for a frozen model).

- `fit_earlylife_ensemble.py` — trains all four on a stratified, recipe-balanced,
  cell-grouped split and reports per-split MAE/RMSE.
- `choice.md` — the decision and its justification, from the actual numbers.

### 3. `frozen_model/` — tune the chosen model

Searches CatBoost hyperparameters on the early-life data.

- `tune_earlylife_catboost.py` — Optuna search over the boosting knobs, selecting
  on validation MAE while training with RMSE loss (MAE loss breaks under the
  monotone constraint). Saves the best params, the tuned model, per-row
  predictions for the whole dataset, and a per-check-up MAE table.
- `README_tune_earlylife_catboost.md` — details.

Key output consumed downstream:
`..._catboost_predictions.parquet` and `..._catboost_best_params.json`.

### 4. `shml/` — diagnose drift and adapt (self-healing)

Watches the frozen model's error drift across later check-ups, attributes it to
ECM mechanisms, and (as future work) adapts the model.

- `shml_from_artifacts.py` — **diagnosis + recommendation**, trains nothing.
  Reads the predictions + ECM values + params, reports per-check-up MAE drift and
  each mechanism's contribution, and produces a contribution-weighted
  hyperparameter recommendation (with ohmic resistance on a separate track).
- `shml_stage5_adapt_test.py` — **the adaptation loop** (Monitor → Diagnose →
  Adapt → Test), running a diagnosis-conditioned policy against a reason-agnostic
  control, with a backtest gate that only keeps an adapted model if it beats the
  frozen one on the next unseen window. Requires a cold-start model from
  `catboost_coldstart.py`.
- `README_shml_from_artifacts.md`, `README_shml_stage5_adapt_test.md` — details.

## End-to-end run order

```bash
# 1. build the feature table (merge -> filter -> orthogonalize)
python feature_table/build_feature_table.py \
    --ecm data/interim/ecm_features_v2.parquet \
    --deltaz data/interim/deltaz_features.parquet \
    --labeled data/interim/eis_labeled.parquet \
    --out data/interim/stage2_features.parquet
python feature_table/filter_stage2_features.py \
    --file data/interim/stage2_features.parquet
python feature_table/orthogonalize_features.py \
    --file data/interim/stage2_features_filtered.parquet
# -> data/interim/stage2_features_filtered_orthogonalized.parquet

# 2. pick the model
python model_choice/fit_earlylife_ensemble.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --output-dir model_choice/results

# 3. tune CatBoost (produces predictions + best_params consumed by SHML)
python frozen_model/tune_earlylife_catboost.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --n-trials 50 --output-dir frozen_model/results

# 4. SHML diagnosis + recommendation (trains nothing)
python shml/shml_from_artifacts.py \
    --pred-file  frozen_model/results/stage2_features_filtered_orthogonalized_catboost_predictions.parquet \
    --ecm-file   data/interim/stage2_features_filtered_orthogonalized.parquet \
    --params-file frozen_model/results/stage2_features_filtered_orthogonalized_catboost_best_params.json \
    --abserr-col abserr_tuned --error-threshold 3.6 --exclude-ohmic \
    --output-dir shml/results
```

(The `shml_stage5_adapt_test.py` adaptation loop is the future-work step; see its
README for the cold-start prerequisite and how to run it.)

## What is validated vs. what is future work

- **Validated:** the model comparison (CatBoost wins later-life), the tuning, and
  the drift diagnosis (which mechanisms the error tracks).
- **Future work:** the *self-healing* claim — that adapting the model actually
  reduces later-life error. `shml_from_artifacts.py` only *recommends*;
  `shml_stage5_adapt_test.py` prototypes the adaptation but its gains need
  multi-seed, multi-window (ideally multi-dataset) validation before any healing
  claim is made. Report the SHML result as an efficiency argument
  (comparable deployment success with fewer expensive retrains), not a blanket
  accuracy win.

## Dependencies

See `environment.yml`. Beyond the scientific-Python base, this workflow needs
`catboost`, `optuna`, `xgboost`, `lightgbm`, `scikit-learn`, `statsmodels`,
`scipy`, and `pyarrow`.