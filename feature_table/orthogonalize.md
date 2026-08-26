# orthogonalize_features.py

Runs a variance-inflation-factor (VIF) collinearity check, then orthogonalizes
the ECM parameters against a chosen `deltaZ` column — replacing each ECM
parameter with the part of it that `deltaZ` does **not** already explain. This
decorrelates the ECM features from `deltaZ` so a downstream model can attribute
predictive credit cleanly instead of splitting it between collinear inputs.

## Why this step exists

The ECM parameters (`R_ohm`, `R_1arc`, `sigma_1arc`) and the `deltaZ` summary are
derived from the same underlying impedance spectrum, so they are often strongly
collinear. If both are fed to a model as-is, the model has no principled way to
decide which feature deserves credit for a prediction, and feature-importance
readings become unreliable. Orthogonalizing the ECM parameters against `deltaZ`
removes the shared component, leaving each `_orth` feature carrying only the
information `deltaZ` lacks.



## How orthogonalization works

For each ECM parameter, the script fits a simple straight line predicting that
parameter from the `deltaZ` column, then keeps only the **residual** — the
vertical distance from the fitted line:

```python
slope, intercept = np.polyfit(x, y, 1)   # x = deltaZ, y = ECM parameter
residual = y - (slope * x + intercept)   # the part deltaZ doesn't explain
```

The residual becomes a new `_orth` column (`R_ohm_orth`, `R_1arc_orth`,
`sigma_1arc_orth`). Rows with `NaN` in either the parameter or the `deltaZ`
column keep `NaN` in the output; if fewer than three usable rows exist, the whole
`_orth` column is `NaN`.

## `--deltaz-col` matters — match it to your downstream model

The ECM parameters must be orthogonalized against **whichever `deltaZ` column the
downstream model actually treats as primary.** Orthogonalizing against the wrong
column leaves the `_orth` features still partly collinear with the `deltaZ`
feature the model really uses, defeating the point.



## What the VIF numbers mean

The script prints a VIF table before and after orthogonalization:

- **VIF > 5** — moderate collinearity; **VIF > 10** — severe.
- **Before:** the raw ECM parameters vs. `deltaZ` — expected to be high, which is
  what motivates the step.
- **After:** each `_orth` column vs. `deltaZ` should land **close to 1**,
  confirming the collinearity was actually removed. If an `_orth` VIF is still
  high, the orthogonalization didn't take (usually a sign `--deltaz-col` points
  at the wrong column).

## Inputs

- `--file` — the filtered Stage-2 table (from `filter_stage2_features.py`);
  default `data/interim/stage2_features_filtered.parquet`. Only rows with
  `fit_succeeded == True` are kept before orthogonalizing.
- `--deltaz-col` — the `deltaZ` column to orthogonalize against
  (default `deltaz_integrated_abs`).

The columns orthogonalized are fixed in `ECM_PARAM_COLS = ["R_ohm", "R_1arc",
"sigma_1arc"]`.

## Outputs

`<file>_orthogonalized.parquet` next to the input — the input table plus the
three `_orth` columns. For the standard filtered input this produces
`stage2_features_filtered_orthogonalized.parquet`, the final feature table ready
for any downstream model.

## Usage

```bash
python orthogonalize_features.py --file data/interim/stage2_features_filtered.parquet
```

Orthogonalize against a different `deltaZ` column:

```bash
python orthogonalize_features.py \
    --file data/interim/stage2_features_filtered.parquet \
    --deltaz-col deltaz_mean_all
```

## Dependencies

`pandas`, `numpy`, `pyarrow` (parquet), and `statsmodels` (for the VIF check).

## Where it sits in the pipeline

```
build_feature_table.py  ->  filter_stage2_features.py  ->  orthogonalize_features.py
   (merge sources)            (drop bad spectra)            (decorrelate ECM vs deltaZ)
                                                                     |
                                                                     v
                                        stage2_features_filtered_orthogonalized.parquet
                                                  (final feature table)
```