# SHML: diagnosing and (attempting to) heal frozen-model drift 

This document explains the complete self-healing-ML (SHML) experiment end to end:
the motivation, the pipeline that feeds it, every design decision inside the
adaptation loop, and the full results across all configurations. 

---

## 0. summary

We train a battery State-of-Health (SoH) model on early-life data, freeze it,
and watch its error grow as cells age. We diagnose *which physical mechanism*
drives that error, then test whether steering the model toward that mechanism
(by reweighting its features during retraining) heals the error better than
simply retraining on recent data. Across 5 seeds, 3 targeting strategies, 5
boost strengths, and 2 frozen-model hyperparameter sets, the answer is
consistent: **retraining on recent data heals the drift substantially, but
diagnosis-conditioned reweighting never beats plain retraining — at best it
breaks even, and stronger boosting monotonically worsens it.** This is a
rigorous negative result. The project's positive contribution is the *diagnosis*
(a mechanism-transition story and a real-part/imaginary-part feature blind spot),
not the adaptation.

---

## 1. Background and motivation

### The setting
Batteries degrade. Electrochemical Impedance Spectroscopy (EIS) measures a cell's
impedance across frequencies; from each spectrum we fit an equivalent-circuit
model (ECM) and compute features that predict SoH (capacity health, a
percentage).

### The frozen-model premise
In deployment you often train a model once, early in life, and cannot keep
retraining it. So we deliberately **freeze** an early-life model and study how it
fails as cells age — and whether it can heal itself when it detects it is
failing. That self-healing idea is "SHML."

### Why this matters
If a diagnosis of *why* a model is drifting could tell it *how* to adapt, you'd
have a principled, physically-grounded self-healing scheme — more trustworthy
than blind retraining. The whole experiment tests whether that promise holds.

---

## 2. The data and features

- **Dataset:** KIT battery aging dataset — LG INR18650HG2 cells, EIS spectra at
  check-up indices `cu_index` 0–28, across aging recipes (Calendar / Cyclic /
  Profile), SoC and temperature conditions.
- **One row per spectrum**, keyed by `cell_id, cu_index, soc_nom, is_rt`.
- **Target:** `soh_cap` (State of Health, %).

### The model's features
- `deltaz_integrated_abs` — the integrated **imaginary-part** change of impedance
  vs. beginning-of-life. The model's dominant aging signal. **Being
  imaginary-only, it is largely blind to real-part ohmic-resistance growth** —
  this fact becomes the project's key diagnostic insight.
- `R_ohm_orth`, `R_1arc_orth`, `sigma_1arc_orth` — ECM parameters (ohmic,
  interfacial, diffusion), **orthogonalized** against deltaZ so each carries
  information deltaZ does not (removing collinearity so the model can attribute
  credit cleanly).
- `soc_nom`, `temp_degC`, `is_rt` — measurement conditions.

### The pipeline that produces the feature table
```
build_feature_table.py   merge ECM + deltaZ + SoH label into one row per spectrum
        |
filter_stage2_features.py  drop KK-validation failures and implausible one-arc fits
        |
orthogonalize_features.py  VIF check + orthogonalize ECM params against deltaZ
        |
   stage2_features_filtered_orthogonalized.parquet   (the feature table used below)
```

---

## 3. The model choice and the frozen model

### Why CatBoost
Four candidates (LightGBM, XGBoost, CatBoost, equal-weight ensemble) were trained
on the same early-life split and scored on later life. CatBoost had the lowest
**later-life** MAE (4.74) — the regime that matters for a frozen model — beating
XGBoost (5.38), the ensemble (5.13), and LightGBM (6.11). XGBoost fit the early
window tighter but lost later life; the ensemble did not beat CatBoost. So
CatBoost is the base model. (Full justification in `choice.md`.)

### Tuning and the RMSE-loss detail
`tune_earlylife_catboost.py` runs an Optuna search on early-life data and saves
`_catboost_best_params.json`. A critical implementation detail: CatBoost's
`loss_function="MAE"` **breaks under the monotone constraint** on deltaZ (MAE
~18 instead of ~4), because MAE's constant-sign gradient interacts badly with the
monotone projection. The fix, used throughout: **train with `loss_function=
"RMSE"`, select trials by validation MAE.**

### The monotone prior
Every model enforces a monotone constraint (`-1`) on `deltaz_integrated_abs`:
SoH can only decrease as deltaZ grows. This physical prior is held fixed across
all models and all adaptation policies.

### What "the frozen model" is in the loop
The adaptation loop does **not** load a saved `.cbm`. It **rebuilds** the frozen
model each run: train CatBoost on early life (`cu_index <= 5`) using the tuned
hyperparameters from `_catboost_best_params.json`. This is deliberate — the loop
needs a frozen model built on each seed's split so the comparison against adapted
models is fair within that seed.

---

## 4. The diagnosis (the project's positive contribution)

Before adaptation, we diagnose *why* the frozen model errs in later life
(`shml_from_artifacts.py`, which trains nothing — it reads existing artifacts).

### How contribution is computed
For each later check-up and each mechanism:
```
contribution = |deviation from early-life baseline, in SDs| × |Spearman(deviation, model error)|
```
A mechanism is "blamed" only if it **both** drifted **and** its drift tracks the
model's error. This separates "drifts but doesn't matter" from "drifts and drives
the error."

### What the diagnosis found
- Early later-life (cu 6–9): diffusion/spectral lead.
- From cu 10 on: **ohmic resistance dominates and climbs monotonically** (0.12 →
  0.32 of the error contribution).
- deltaZ (imaginary) tracks alongside but is second.

### The key mechanistic insight
The model's main feature (deltaZ) is **imaginary-part only**, so it is
structurally blind to **real-part ohmic growth** — which is exactly the mechanism
that dominates later-life error. The model fails late in life partly because its
primary feature cannot see the thing that is changing. This is the most novel,
defensible result in the project.

### Two views
- **Full view** — all factors including ohmic.
- **Aging view** (`--exclude-ohmic`) — ohmic factored out (ohmic/series resistance
  can reflect contact/electrolyte/cabling effects rather than electrode aging),
  renormalized, so the electrode-level aging mechanisms are visible. Both are
  reported; ohmic is set aside transparently, never hidden.

---

## 5. The adaptation experiment (`shml_reweight_loop_v2.py`)

### The hypothesis
When the frozen model drifts, retrain it **while boosting the feature matching
the diagnosed drifting mechanism** (via CatBoost `feature_weights`, which
multiplies how often a feature is chosen for splits). If the diagnosis is useful,
this should beat plain retraining.

### Four policies, same windows
- **frozen** — never adapts (do-nothing baseline).
- **control** — warm-start retrain, **uniform** weights (retraining, no diagnosis).
- **single_boost** — retrain, boost the ONE dominant diagnosed feature.
- **shared_boost** — retrain, boost ALL mechanism features scaled by contribution
  share.

The decisive comparison is **single/shared vs. control** — both retrain on the
same data, so any difference is due purely to the diagnosis-driven weighting.

### The monitor → adapt → gate loop, per window
- **adapt window** = `[start, start+window_size)` — recent data to adapt on.
- **gate window** = `[start+window_size, start+2*window_size)` — the *next*,
  unseen cycles, used only to judge the adapted model.
- **MONITOR:** frozen MAE on the adapt window; "drifted" if over `--error-
  threshold` (3.6). Only drifted windows adapt.
- **ADAPT:** warm-start retrain (`init_model=frozen`) with the policy's weights.
- **GATE:** keep the adapted model only if its gate-window MAE beats frozen's,
  else fall back to frozen. The separate gate window forces improvements to
  *generalize*, not just fit the adaptation data — the core anti-overfitting
  safeguard.

### Data-driven window placement
`--auto-start-from-mae` reads the per-check-up MAE file and starts windows at the
**first later-life cu whose MAE exceeds the threshold** — on this data, **cu=8**,
giving starts `[8, 12, 16, 20]`. This makes the intervention point principled
(where the model actually drifted, by its own criterion) rather than hand-picked,
which pre-empts a "why start there?" reviewer objection.

### What v2 adds (to interrogate v1's negative result)
1. `--boost-list` — sweeps several boost strengths in one run.
2. `--exclude-ohmic` — boost a genuine aging mechanism instead of ohmic.
3. `--boosted-l2-mult` — extra L2 on boosted policies, to counter overfitting.
4. A real paired t-test (`scipy.stats.ttest_rel`) alongside a 2·SE screen.

### On seeds (why they're here even with a fixed model)
The tuned **hyperparameters** are fixed, but training still has randomness: the
early-life train/val split and CatBoost's Bernoulli row sampling, both controlled
by the seed. Different seeds build different frozen models from the same settings.
Running seeds 0–4 tests whether the finding survives that randomness — essential
because the effects are small. The seed never re-tunes hyperparameters.

### Where inputs come from
- Frozen hyperparameters: `--params-file` → `_catboost_best_params.json`
  (`best_params` key). Falls back to the rec-file's `current_params` if omitted.
- Diagnosis (which feature to boost, shares): `--rec-file` →
  `_shml_recommendation.json`. (CatBoost stores no editable per-feature weights,
  so weights are constructed in the loop, not read from a model.)

---

## 6. Results — the complete picture

5 seeds, windows auto-started at cu=8, frozen model from `best_params.json`.
**Control MAE = 3.090** in every run (constant — control does not depend on
boost). Numbers are **how much WORSE boosting is than control** (positive =
worse; negative would mean boosting helped):

| boost | ohmic single | aging single | aging shared |
|-------|--------------|--------------|--------------|
| 1.3   | +0.019       | **+0.0009**  | +0.008       |
| 1.5   | +0.039       | +0.001       | +0.007       |
| 2.0   | +0.092       | +0.014       | +0.026       |
| 3.0   | +0.187       | +0.035       | +0.067       |
| 5.0   | +0.305       | +0.074       | +0.147       |

Frozen model MAE on these windows is ~5.34, so **control (3.090) heals ~2.25 MAE
by retraining** — adaptation clearly works. But:

- **Boosting never beats control** — every cell is positive, in every
  configuration, at every strength.
- **Best case is break-even** — aging mechanism at boost 1.3 gives +0.0009,
  statistically indistinguishable from control. It stops hurting; it does not help.
- **Clear dose-response** — harm grows monotonically with boost strength, the
  signature of overfitting to the boosted feature.
- **Aging targets beat ohmic targets** (smaller harm), consistent with ohmic
  being a poorer target — but still no gain.
- **Regularization does not rescue it** (the aging+L2 configuration, run earlier,
  remained worse than control).

### Robustness
The negative result holds across: **5 seeds × 3 targeting strategies × 5 boost
strengths × 2 frozen-model hyperparameter sets** (rec-file params and best_params
gave near-identical numbers). It is not an artifact of one split, one target, one
strength, or one hyperparameter choice.

### The honest conclusion
> Diagnosis-conditioned feature reweighting does not improve on plain retraining.
> The gain from adaptation comes from retraining on fresh data, not from steering
> the model's feature attention. This holds robustly across every configuration
> tested.

### Why this happens (the mechanism)
The diagnosed feature is **already in the model**, and retraining already lets the
model use it as much as the data justifies. Forcing extra weight onto it cannot
add information the data does not contain — it only distorts the fit away from the
data-optimal weighting, which is why boosting hurts and hurts more with strength.

---

## 7. Statistical cautions

- **Multiple comparisons:** up to 10 t-tests per run (boosts × policies). At
  p<0.05 that's a ~40% chance of one spurious "significant" cell. Trust the
  *consistent trend across the sweep*, not any single cell; for a single claim use
  a Bonferroni threshold (p < 0.005).
- **Zero-diff rows:** non-drifted windows copy frozen's number to all policies
  (difference exactly 0), which can inflate `n` and over-tighten the t-test. Here
  nearly all windows (cu ≥ 8) drift, so the effect is minor; for a clean test,
  filter to `drifted==True` before the t-test.


---

## 8. How to reproduce

```bash
# 1. build the feature table
python feature_table/build_feature_table.py --ecm ... --deltaz ... --labeled ... --out ...
python feature_table/filter_stage2_features.py --file ...
python feature_table/orthogonalize_features.py --file ...

# 2. pick the model
python model_choice/fit_earlylife_ensemble.py --file ...orthogonalized.parquet --output-dir model_choice/results

# 3. tune CatBoost (saves best_params.json + predictions + mae_per_checkup)
python frozen_model/tune_earlylife_catboost.py --file ...orthogonalized.parquet --output-dir frozen_model/results

# 4. diagnosis + recommendation (trains nothing)
python shml/shml_from_artifacts.py --pred-file ...predictions.parquet \
    --ecm-file ...orthogonalized.parquet --params-file ...best_params.json \
    --error-threshold 3.6 --exclude-ohmic --output-dir shml/results

# 5. the adaptation experiment (per configuration, into its own output dir)
python shml/shml_reweight_loop_v2.py --file ...orthogonalized.parquet \
    --rec-file ...shml_recommendation.json --params-file ...best_params.json \
    --auto-start-from-mae ...mae_per_checkup.parquet \
    --error-threshold 3.6 --window-size 4 --seeds 0,1,2,3,4 \
    --boost-list 1.3,1.5,2.0,3.0,5.0 --output-dir shml/results/run_ohmic
# add --exclude-ohmic for the aging target; add --boosted-l2-mult 2.0 for the L2 test
```

Each run writes `shml_reweight_sweep_summary.csv` and
`shml_reweight_sweep_per_window.csv`. Use a distinct `--output-dir` per
configuration (fixed filenames overwrite otherwise)


---

## 9. What this experiment establishes, and what remains

**Established (validated):**
- CatBoost is the best base model for later-life SoH here.
- The frozen model's later-life error is driven by an identifiable, age-dependent
  mechanism transition, with ohmic resistance dominating late — a mechanism the
  imaginary-part deltaZ feature is structurally blind to.
- Retraining on recent data heals the drift substantially.
- Diagnosis-conditioned feature reweighting does **not** beat plain retraining —
  a robust negative result.

**Not addressed (the ceilings):**
- A single dataset (KIT). Generalization is untested.
- No benchmark against published EIS-SoH methods, so absolute model quality is
  unknown.

**The one path where the diagnosis could add value** (untested, and different
from reweighting): use the diagnosis to **supply information the model lacks** —
e.g. add a real-part ohmic feature to address the imaginary-part blind spot —
rather than reweighting features the model already has. Reweighting cannot add
information; a missing feature can. This is the promising direction that builds on
the diagnosis rather than re-confirming the negative reweighting result.

---

## 10. File map

- `feature_table/` — build → filter → orthogonalize (+ READMEs)
- `model_choice/` — `fit_earlylife_ensemble.py`, `choice.md`
- `frozen_model/` — `tune_earlylife_catboost.py`, apply/diagnose scripts (+ READMEs)
- `shml/` — `shml_from_artifacts.py` (diagnosis), `shml_reweight_loop_v2.py`
  (adaptation experiment), `shml_stage5_adapt_test.py` (earlier full loop), READMEs

## Dependencies
`pandas`, `numpy`, `scipy`, `scikit-learn`, `catboost`, `optuna`, `xgboost`,
`lightgbm`, `statsmodels`, `pyarrow`.