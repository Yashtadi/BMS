# shml_from_artifacts.py

Builds the SHML **recommendation** from existing artifacts only — it trains
nothing. It reads the frozen model's predictions, the ECM mechanism values, and
the model's current hyperparameters, then reports how the model's error drifts
across later check-ups, attributes that drift to individual ECM parameters, and
produces a contribution-weighted hyperparameter recommendation for what a future
adaptive scheme should try.

This is the **diagnosis-and-recommendation** half of SHML. Actually applying the
recommendation — retraining/adapting the model and validating that it heals
later-life error — is deliberately left as future work (see the final section).

## What it does

1. **Join** the per-row predictions with the ECM mechanism values on
   `cell_id, cu_index, soc_nom, is_rt`. deltaZ is pulled in if present.
2. **Baseline** — from early-life rows (`cu_index <= --baseline-max-cu`), compute
   each mechanism's mean and standard deviation (the healthy reference).
3. **MAE drift + contribution per check-up** — for each later check-up, report
   the model's MAE and, for each factor, a contribution score:

   ```
   contribution = deviation-from-baseline (in SD) × |Spearman(deviation, error)|
   ```

   Multiplying deviation by its correlation with error means a factor is only
   "blamed" if it *both* drifted *and* that drift tracks the model's error —
   separating "drifts but doesn't matter" from "drifts and drives the error."
4. **Two contribution views**:
   - **Full view** — all factors including ohmic resistance (`R_ohm_1arc`).
   - **Aging-mechanism view** (`--exclude-ohmic`) — ohmic factored out and the
     rest renormalized, with the rationale stated (ohmic/series resistance often
     reflects contact/electrolyte effects rather than electrode-level aging).
     deltaZ (imaginary-part, reactive/diffusive) stays; only real-part ohmic is
     set aside. Both views print — ohmic is set aside transparently, not hidden.
5. **Contribution-weighted recommendation** — one merged hyperparameter set from
   two tracks:
   - **Aging track** — each aging mechanism nudges its knob, scaled by its
     contribution share (bigger contributor → bigger nudge):
     `n_1arc→learning_rate`, `sigma_1arc→subsample`, `R_1arc→min_data_in_leaf`.
   - **Ohmic track** — `R_ohm_1arc` handled separately, scaled by its own share:
     `→ l2_leaf_reg, depth`.

   Every knob change is recorded with **provenance** (which mechanism moved it,
   aging vs ohmic). deltaZ is the aging *signal*, diagnosed via the ECM params —
   it is the monotone-constrained feature, not a tunable knob, so it has no rule
   of its own.

## Inputs

- `--pred-file` (required) — per-row predictions parquet with an `abserr_*`
  column and `cell_id/cu_index/soc_nom/is_rt`
  (e.g. `..._catboost_predictions.parquet` from `tune_earlylife_catboost.py`).
- `--ecm-file` (required) — parquet with the raw ECM mechanisms
  (`R_ohm_1arc, R_1arc, n_1arc, sigma_1arc`) and, for the deltaZ contribution,
  `deltaz_integrated_abs`.
  **Use the orthogonalized feature table here** — it contains all four
  mechanisms *and* deltaZ. `ecm_features_v2.parquet` has the mechanisms but not
  deltaZ, so deltaZ would be silently skipped.
- `--params-file` (required) — the model's current hyperparameters JSON
  (`..._catboost_best_params.json`).
- `--abserr-col` — which error column to diagnose (default `abserr_tuned`;
  `abserr_baseline` also available).
- `--error-threshold` — MAE above which the model is "drifting" (default 3.6).
- `--baseline-max-cu` — early-life cutoff (default 5).
- `--exclude-ohmic` — also print the aging-mechanism view (ohmic factored out).
- `--output-dir` — where outputs are written (defaults next to `--pred-file`).

## Outputs

- `..._shml_recommendation.json` — the full/aging contributions, current +
  recommended params, and per-knob `change_provenance`.
- `..._shml_per_checkup.csv` — per-check-up MAE, over-threshold flag, dominant
  factor, and each factor's contribution.

## How to run

The script lives in the `shml/` folder. Run from the repo root.

```bash
python shml/shml_from_artifacts.py \
    --pred-file  frozen_model/results/stage2_features_filtered_orthogonalized_catboost_predictions.parquet \
    --ecm-file   data/interim/stage2_features_filtered_orthogonalized.parquet \
    --params-file frozen_model/results/stage2_features_filtered_orthogonalized_catboost_best_params.json \
    --abserr-col abserr_tuned \
    --error-threshold 3.6 \
    --exclude-ohmic \
    --output-dir frozen_model/results
```

**Check when it runs:** the "Drift factors" line near the top should list five
items ending in `deltaz_integrated_abs`. If it shows only four, `--ecm-file` is
pointing at a file without deltaZ (use the orthogonalized table). The join line
should report roughly the full row count; a much lower number means the join
keys didn't match.

## What this is — and is not

Every knob rule (e.g. "spectral drift → lower learning rate") is a
**stated-in-advance heuristic**, and the contribution-scaling is a design choice.
The script trains and validates **nothing** — it prints a NOTE saying so. The
output is a principled, drift-informed *suggestion*, not a demonstrated fix. This
project's own tuning experiments showed that hyperparameters chosen from
early-life data alone did not reliably improve later-life error, which is exactly
why the recommendation is framed as a direction rather than a guarantee.

## Dependencies

`pandas`, `numpy`, `scipy` (Spearman), `pyarrow`.

## Future work — implementing SHML

This script stops at the recommendation. The remaining SHML work — actually
adapting the model and validating that it heals later-life error — is future
work, partially prototyped in `shml_stage5_adapt_test.py`. The pieces a full
implementation needs:

1. **Apply the recommendation.** Feed the recommended hyperparameters into an
   adaptation step that retrains/warm-starts the model when the monitored MAE
   crosses the threshold, rather than only printing them.
2. **Close the monitor→adapt→test loop.** Trigger on drift, adapt, and gate the
   adapted model on the *next* unseen window (keep it only if it beats the frozen
   model). `shml_stage5_adapt_test.py` prototypes this loop with a backtest gate.
3. **Validate honestly.** Because the expected gains are small and possibly
   within noise on a single dataset, validation must span **multiple seeds** and
   **multiple windows**, reporting confidence intervals — and ideally a **second
   dataset** — before any healing claim is made.
4. **Report efficiency, not just accuracy.** Compare diagnosis-conditioned
   adaptation against a reason-agnostic control (always full retrain); the
   defensible claim is comparable deployment success with fewer expensive
   retrains, not a blanket accuracy win.

Until those are done, treat the output of this script as a diagnosis and a
future-work proposal, not a validated adaptive fix.