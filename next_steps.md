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
                  stage2_features_filtered_orthogonalized.parquet
                                     |
        ┌────────────────────────────┼────────────────────────────┐
        v                            v                             v
  model_choice/                 catboost/                        shml/
  pick the model                tune it                          diagnose drift
                                                                 + adapt (SHML)
```

### 1. `model_choice/` — pick the base model

Compares LightGBM, XGBoost, CatBoost, and their equal-weight ensemble on the
same early-life split, and documents why **CatBoost** was chosen (lowest
later-life MAE/RMSE — the regime that matters for a frozen model).

- `fit_earlylife_ensemble.py` — trains all four on a stratified, recipe-balanced,
  cell-grouped split and reports per-split MAE/RMSE.
- `choice.md` — the decision and its justification, from the actual numbers.

### 2. `catboost/` — tune the chosen model

Searches CatBoost hyperparameters on the early-life data.

- `tune_earlylife_catboost.py` — Optuna search over the boosting knobs, selecting
  on validation MAE while training with RMSE loss (MAE loss breaks under the
  monotone constraint). Saves the best params, the tuned model, per-row
  predictions for the whole dataset, and a per-check-up MAE table.
- `README_tune_earlylife_catboost.md` — details.

Key output consumed downstream:
`..._catboost_predictions.parquet` and `..._catboost_best_params.json`.

### 3. `shml/` — diagnose drift and adapt (self-healing)

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
# 0. (prerequisite) build the feature table -> stage2_features_filtered_orthogonalized.parquet

# 1. pick the model
python model_choice/fit_earlylife_ensemble.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --output-dir model_choice/results

# 2. tune CatBoost (produces predictions + best_params consumed by SHML)
python catboost/tune_earlylife_catboost.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --n-trials 50 --output-dir catboost/results

# 3. SHML diagnosis + recommendation (trains nothing)
python shml/shml_from_artifacts.py \
    --pred-file  catboost/results/stage2_features_filtered_orthogonalized_catboost_predictions.parquet \
    --ecm-file   data/interim/stage2_features_filtered_orthogonalized.parquet \
    --params-file catboost/results/stage2_features_filtered_orthogonalized_catboost_best_params.json \
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