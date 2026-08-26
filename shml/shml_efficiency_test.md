# shml_efficiency_test.py

A five-policy **deployment-efficiency** comparison for the SHML adaptation stage.
Where `shml_reweight_loop_v2.py` asks the narrow accuracy question ("does
diagnosis-conditioned boosting beat generic adaptation?" — answer: no), this
script reframes the comparison around **cost**: how much data, and how much
compute, each policy needs to keep the frozen model healthy.

It exists to test one claim:

> "Warm-start adaptation on recent data reaches deployment accuracy comparable to
> (or better than) full retraining, while using far less data / compute."

The script does **not** assume that claim — it measures whether it holds on your
data and prints an explicit verdict. If it doesn't hold, the honest negative
result stands.

## The five policies

| policy | on drift, it… | cost profile |
|--------|---------------|--------------|
| `frozen` | does nothing | 0 data, 0 compute (the floor) |
| `full_retrain` | retrains from scratch on **all** data seen so far | expensive: most data |
| `generic_adapt` | warm-starts from the frozen model on the **recent** window only | cheap: little data |
| `diagnosis_single` | generic adapt **+** boosts the **one** dominant diagnosed feature | cheap: little data |
| `diagnosis_shared` | generic adapt **+** boosts **all** features by contribution share | cheap: little data |

The point of having all five is to separate two independent questions:

1. **Efficiency:** is warm-start adaptation (`generic_adapt`) as good as
   `full_retrain` at lower cost? — compares adaptation vs full retrain.
2. **Does the diagnosis add value?** — compares **both** diagnosis policies
   (`diagnosis_single`, `diagnosis_shared`) against `generic_adapt`. If they're
   equal, the diagnosis contributes nothing and any efficiency win belongs to
   warm-start adaptation itself, not the diagnosis.

Having both diagnosis policies means the negative-on-diagnosis finding covers
**both** boosting styles in a single run — matching the completeness of
`shml_reweight_loop_v2.py` (which also has single + shared).

## The two boost styles

Both use CatBoost's `feature_weights` (a train-time multiplier on a feature's
split-gain), differing only in how the weight is distributed:

- **single** — one feature (the dominant diagnosed mechanism) gets the full
  boost `b`; all others stay at 1.0.
  `w_f = b if f = f*  else 1`
- **shared** — every feature is boosted in proportion to its diagnosed
  contribution share; a 100%-share feature gets the full boost, smaller
  contributors get proportionally less.
  `w_f = 1 + (b − 1) · share_f`
  With `--exclude-ohmic`, the ohmic feature is left unboosted so only genuine
  aging mechanisms are emphasised.

## What it measures (per seed × window)

For every policy at every window it logs:

- `gate_mae` — accuracy on the **next, unseen** window (the honest test)
- `adapted` — whether an adaptation event fired (drift was detected)
- `rows_used` — training-set size for that adaptation (data cost)
- `train_seconds` — wall-clock training time (compute proxy)

and then aggregates per policy: mean gate MAE (± 95% CI), number of adaptation
events, rows per adaptation, seconds per adaptation, and total compute.

## The monitor → adapt → gate loop

Same honest structure as `shml_reweight_loop_v2.py`:

1. **Monitor** — is the frozen model's error on the adapt window above threshold?
2. **Adapt** — if drifted, each adaptive policy retrains (full / generic /
   diagnosis_single / diagnosis_shared).
3. **Gate** — score the adapted model on the **next unseen** window; keep it only
   if it beats the frozen model there. Testing on unseen data is what stops an
   "improvement" from being mere overfitting to the window it trained on.

If a window has **not** drifted, no adaptive policy fires — so `adapt_events`
also captures *how often* each policy needs to retrain, which is itself an
efficiency axis.

## Inputs

Mirror `shml_reweight_loop_v2.py`, so the same artifacts work:

| flag | meaning |
|------|---------|
| `--file` | orthogonalized feature table (parquet) |
| `--rec-file` | `_shml_recommendation.json` — the diagnosis (dominant feature + contribution shares) |
| `--params-file` | `_catboost_best_params.json` — frozen hyperparameters |
| `--auto-start-from-mae` | `_catboost_mae_per_checkup.csv/.parquet` — finds the drift point |
| `--error-threshold` | drift threshold (default 3.6) |
| `--window-size` | checkups per window (default 4) |
| `--baseline-max-cu` | last early-life checkup (default 5) |
| `--seeds` | comma-separated seeds (default `0,1,2,3,4`) |
| `--boost` | **single** boost strength used by both diagnosis policies (default 1.5) |
| `--exclude-ohmic` | boost the dominant *aging* mechanism (single) and leave ohmic unboosted (shared) |
| `--max-iter` | CatBoost iterations (default 1000) |
| `--output-dir` | where results are written |

Note: `--boost` is a single value here (not a sweep) — the boost sweep already
lives in `shml_reweight_loop_v2.py`. This script fixes one boost and focuses on
the multi-policy cost comparison.

## Outputs

Written to `--output-dir`:

- `shml_efficiency_per_window.csv` — every `(seed, window, policy)` record, with
  `gate_mae`, `adapted`, `rows_used`, `train_seconds`.
- `shml_efficiency_summary.csv` — the per-policy aggregate table (the paper
  table): `gate_mae`, `gate_mae_ci95`, `adapt_events`, `rows_per_adapt`,
  `sec_per_adapt`, `total_train_sec`.

Console prints the per-policy table plus a two-part verdict:
1. warm-start adaptation vs full retrain (the efficiency axis), and
2. each diagnosis policy vs generic (does the diagnosis add value).

## How to run

From the repo root:

```bash
python shml/shml_efficiency_test.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --rec-file shml/results/stage2_features_filtered_orthogonalized_catboost_shml_recommendation.json \
    --params-file frozen_model/results/stage2_features_filtered_orthogonalized_catboost_best_params.json \
    --auto-start-from-mae frozen_model/results/stage2_features_filtered_orthogonalized_catboost_mae_per_checkup.csv \
    --error-threshold 3.6 --window-size 4 --seeds 0,1,2,3,4 --boost 1.5 \
    --exclude-ohmic --max-iter 1000 --output-dir shml/results/run_efficiency
```

On Windows PowerShell, put it all on **one line** (no `\`), and run from the
repo root, not from inside `shml/`. The five-policy run takes noticeably longer
than a four-policy one (an extra retrain per drifted window).

## Interpreting the result (what our run showed)

On KIT (5 seeds × 4 windows), all five policies:

| policy | gate MAE ± 95% CI | rows/adapt | total compute |
|--------|-------------------|------------|---------------|
| frozen | 5.206 ± 0.090 | 0 | 0 s |
| full_retrain | 3.441 ± 0.037 | 27,656 | 490 s |
| generic_adapt | 3.090 ± 0.030 | 4,717 | 493 s |
| diagnosis_single | 3.091 ± 0.029 | 4,717 | 461 s |
| diagnosis_shared | 3.094 ± 0.029 | 4,717 | 486 s |

Two findings, reported honestly and separately:

1. **Warm-start adaptation beats full retraining** (3.09 vs 3.44, CIs don't
   overlap) using **~6× less data**. This is a genuine positive, data-efficiency
   result: adapting on the recent window sharpens to the drift, while full
   retraining dilutes it across all the old data.
2. **Neither diagnosis policy adds value** — single and shared boost both land
   within noise of generic adaptation. The efficiency win belongs to warm-start
   adaptation itself, not to steering toward the diagnosed mechanism.

**Caveat on compute:** in our final run the warm-start policies used about the
same wall-clock as full retraining (~490 s for both) despite training on ~6×
less data — feature-weighting and init-from-frozen overhead offset the smaller
training set. So the honest claim is **"better accuracy on far less DATA, at
comparable compute,"** not "faster." Report the data-efficiency axis, not speed.

## Scope and honesty

- One dataset, one frozen-model family — the same two structural limits as the
  rest of the project. This probes whether the efficiency framing is worth
  pursuing; it is **not** by itself a conference result.
- `train_seconds` is wall-clock and machine-dependent — a **relative** proxy,
  compared within a single run, not an absolute cost.
- "Full retrain from all data so far" is the honest expensive baseline. If your
  deployment reality differs (e.g. a warm-started full retrain, or a fixed
  retrain budget), adjust the `full_retrain` branch accordingly — a reviewer
  will probe whether the baseline is fair.

## Dependencies

`pandas`, `numpy`, `scipy`, `catboost`, `scikit-learn`, `pyarrow`.