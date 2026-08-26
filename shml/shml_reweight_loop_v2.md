# shml_reweight_loop_v2.py — full guide

A detailed walkthrough of what this script does, the concepts behind it, how to
run it, and what our experiments actually found. Written to be readable without
prior context.

---

## 1. The question this script answers

We have a battery State-of-Health (SoH) model — a CatBoost regressor — trained
on **early-life** data and then **frozen** (never updated). As batteries age,
the model's predictions drift and its error (MAE) grows.

A diagnosis step (`shml_from_artifacts.py`) tells us *which physical mechanism*
is driving that error — for example, ohmic resistance growth. The natural next
idea, and the one this script tests:

> When the model has drifted, should we **retrain it while forcing it to pay
> more attention to the drifting mechanism's feature** — and does that heal the
> error better than simply retraining normally?

This is "diagnosis-conditioned feature reweighting." The script tests it
rigorously and answers yes or no with statistics.

---

## 2. Key concepts (read this if any term is unfamiliar)

### SoH and MAE
**SoH** = State of Health, a percentage (100% = new). The model predicts it.
**MAE** = Mean Absolute Error — on average, how many SoH points the prediction
is off by. Lower is better. MAE is the right metric because SoH is a continuous
number (a regression), not a category — so classification metrics like accuracy,
precision, recall, and F1 do **not** apply here.

### Frozen model
A model trained once on early life and never changed afterward. We freeze it on
purpose, to study how a fixed model degrades as conditions drift — and whether
we can heal it.

### Drift
As cells age, the relationship between the measured features and true SoH
shifts. The frozen model, trained on young cells, gets this increasingly wrong.
We call a check-up "drifted" when the frozen model's MAE there exceeds a
threshold (default 3.6).

### deltaZ and ECM parameters (the features)
The model's inputs describe each impedance spectrum:
- **deltaZ** (`deltaz_integrated_abs`) — the change in a spectrum's **imaginary**
  impedance vs. beginning-of-life. It's the model's main aging signal. Being
  imaginary-part only, it captures reactive/diffusive change and is largely
  **blind to real-part ohmic-resistance growth**.
- **ECM parameters** — fitted equivalent-circuit values, orthogonalized against
  deltaZ so they carry independent information: `R_ohm_orth` (ohmic/series
  resistance), `R_1arc_orth` (interfacial), `sigma_1arc_orth` (diffusion).

### feature_weights (the mechanism this script uses)
CatBoost is a tree ensemble; it has no single editable "weight" per feature.
But `feature_weights` is a training parameter that multiplies how likely each
feature is to be chosen when trees split. Setting `R_ohm_orth`'s weight to 3.0
makes the model build ~3x more splits on ohmic resistance — i.e. "pay more
attention" to it. This only takes effect during **training**, which is why
applying it means retraining.

### Warm-start (init_model)
Instead of training a fresh model, CatBoost can **continue** from an existing
one (`init_model=frozen`), adding more trees. This is how we adapt the frozen
model on new data rather than starting over. (It prints a harmless warning:
"Model shrinkage in combination with learning continuation is not implemented
yet. Reset model_shrink_rate to 0." — ignore it.)

### What a "seed" is  (you asked about this)
Computers generate *pseudo*-random numbers: they look random but come from a
deterministic algorithm with a starting point called the **seed**. Same seed →
identical "random" numbers every run. Different seed → a different draw.

In this script the seed controls two random things, **even though the
hyperparameters are fixed**:
1. the train/validation split of cells for the frozen model, and
2. CatBoost's internal row sampling (`bootstrap_type="Bernoulli"`).

So seed 0 and seed 1 produce *different* frozen models from the *same* settings.

**Why run several seeds?** A single run is one random draw. If an effect is
small, one draw can't tell a real effect from luck. Running seeds 0-4 gives five
independent answers; if the finding holds across all five (small spread), it's
robust — if it flips between seeds, it's noise. Reporting mean +/- std across
seeds is how you show a result is trustworthy rather than a fluke.

**Note on hyperparameters vs. seed:** the seed does **not** re-tune
hyperparameters. This script always uses the frozen model's already-tuned
hyperparameters (read from the recommendation JSON's `current_params`). The seed
only varies the data split and row sampling *around* those fixed settings. So
"using the frozen model's parameters" and "running multiple seeds" are
compatible — the params are fixed; the seed tests robustness to randomness.
Running a single seed is fine for a quick look; multiple seeds are what make the
final reported numbers defensible.

---

## 3. The four policies compared

Every window, four versions are built and scored the same way:

| policy | what it does |
|---|---|
| **frozen** | never adapts — the do-nothing baseline |
| **control** | warm-start retrain on recent data, **uniform** feature weights (retraining, no diagnosis) |
| **single_boost** | retrain, boost the ONE dominant diagnosed mechanism's feature |
| **shared_boost** | retrain, boost ALL mechanism features scaled by their diagnosis contribution shares |

The comparison that matters: **single/shared vs. control.** Both retrain on the
same data; the only difference is the feature weighting. So if boosting beats
control, the *diagnosis-driven reweighting* added value. If not, plain
retraining is doing all the work. `frozen` is only there to confirm adaptation
helps at all.

---

## 4. How the loop works, step by step

For each seed, the frozen model is trained once on early life
(`cu_index <= --baseline-max-cu`, default 5). Then for each window start:

- **adapt window** = `[start, start+window_size)` — recent drifted data to adapt on.
- **gate window** = `[start+window_size, start+2*window_size)` — the *next*,
  unseen cycles, used only to judge the adapted model. Never used for fitting.
- **MONITOR:** compute frozen MAE on the adapt window; it's "drifted" if over
  `--error-threshold`. Only drifted windows are adapted.
- **ADAPT:** warm-start retrain (control / single_boost / shared_boost).
- **GATE:** keep the adapted model only if its gate-window MAE beats frozen's;
  else fall back to frozen's number. (Deploy = adapted model was kept.)

The **separate gate window is the honesty mechanism**: an improvement must
generalize to cycles the model didn't adapt on, not just fit the adaptation
window.

### Auto-starting windows at the drift point
With `--auto-start-from-mae <mae_per_checkup file>`, the script reads the frozen
model's per-check-up MAE, finds the **first later-life cu whose MAE exceeds the
threshold**, and tiles windows from there. This makes the window placement
data-driven — you start adapting exactly where the model has actually drifted,
by its own criterion, rather than at an arbitrary hand-picked cu. **The start
point is dynamic:** it depends on the dataset, the `--error-threshold`, and which
frozen model was used, so it can differ between runs. On our data with threshold
3.6 and the tuned frozen model it resolves to around cu=8 (window starts near
`[8, 12, 16, 20]`), but read the actual value from the `AUTO-START:` line the
script prints at the top of each run rather than assuming it.

---

## 5. What v2 adds over v1

v1 found boosting hurt at boost=3.0, worse the harder the boost. v2 interrogates
that three ways:

1. **`--boost-list`** — sweeps several boost strengths in one run (reusing the
   per-seed frozen model), to find whether a *gentler* boost helps or breaks even.
2. **`--exclude-ohmic`** — selects the dominant mechanism from the *aging view*
   (drops `R_ohm_1arc`, renormalizes), testing whether boosting a genuine aging
   mechanism works better than boosting ohmic resistance (which the diagnosis
   flags as possibly contact/cabling noise, not electrode aging).
3. **`--boosted-l2-mult`** — extra L2 regularization on the boosted policies
   only, to counter overfitting-to-one-feature.
4. **A real paired t-test** (`scipy.stats.ttest_rel`) alongside a rough 2*SE
   screen.

---

## 6. Inputs

- `--file` (required) — the orthogonalized feature table.
- `--rec-file` (required) — the `_shml_recommendation.json`; supplies
  `current_params` (frozen hyperparameters), `overall_dominant` (mechanism to
  boost), `contribution_pct_full` (shares for shared_boost).
- `--auto-start-from-mae` — the `_catboost_mae_per_checkup` file; auto-sets
  window starts to the drift point. Omit to use `--window-starts` instead.
- `--window-starts` (default `10,14,18,22`) — used only without auto-start.
- `--window-size` (default 4), `--baseline-max-cu` (default 5),
  `--error-threshold` (default 3.6).
- `--seeds` (default `0,1,2,3,4`) — comma-separated; one value = a single run.
- `--boost-list` (default `3.0`) — boost strengths to sweep.
- `--exclude-ohmic` — boost the top aging mechanism instead of ohmic.
- `--boosted-l2-mult` (default 1.0) — extra L2 on boosted policies.
- `--output-dir`.

## 7. Outputs

- `shml_reweight_sweep_summary.csv` — per boost: control / single / shared MAE
  and the differences. **This is the paper table.**
- `shml_reweight_sweep_per_window.csv` — every (seed, window, boost) record.
- Console: the sweep table, paired t-tests per boost x policy, and deploy rates.

**These filenames are fixed**, so a second run to the same `--output-dir`
overwrites them. Use a different `--output-dir` per configuration (e.g.
`results/run1_ohmic`, `results/run2_aging`) and capture the console with
`> run.txt 2>&1` so nothing is lost.

## 8. Usage

```bash
# boost ohmic (overall-dominant mechanism), auto-started windows
python shml_reweight_loop_v2.py \
    --file data/interim/stage2_features_filtered_orthogonalized.parquet \
    --rec-file shml/results/..._shml_recommendation.json \
    --auto-start-from-mae frozen_model/results/..._catboost_mae_per_checkup.parquet \
    --error-threshold 3.6 --window-size 4 --seeds 0,1,2,3,4 \
    --boost-list 1.3,1.5,2.0,3.0,5.0 \
    --output-dir shml/results/run1_ohmic

# boost an aging mechanism instead
python shml_reweight_loop_v2.py ... --exclude-ohmic \
    --output-dir shml/results/run2_aging

# aging mechanism + extra regularization
python shml_reweight_loop_v2.py ... --exclude-ohmic --boosted-l2-mult 2.0 \
    --boost-list 1.5,3.0 --output-dir shml/results/run3_aging_l2
```

---

## 9. What our experiment found

We ran three configurations, **5 seeds (0-4)**, windows auto-started at the
dynamic drift point (around cu=8 for our threshold and frozen model — check the
`AUTO-START:` line for the exact value, which may shift slightly with the
threshold or frozen-model choice). In every run **control MAE ≈ 3.09** (constant
within a run — control doesn't depend on boost). Numbers below are **how much
WORSE boosting is than control** (positive = worse; a negative number would mean
boosting helped). Exact values may vary slightly between runs because the drift
trigger and per-seed splits are dynamic, but the pattern is stable:

| boost | Run 1 ohmic (single) | Run 2 aging (single) | Run 3 aging+L2 (single) |
|-------|----------------------|----------------------|-------------------------|
| 1.3   | +0.019               | +0.0009              | —                       |
| 1.5   | +0.039               | +0.001               | +0.030                  |
| 2.0   | +0.092               | +0.014               | —                       |
| 3.0   | +0.187               | +0.035               | +0.066                  |
| 5.0   | +0.305               | +0.074               | —                       |

### How to read this

- **Adaptation itself helps a lot.** control (3.090) beats the frozen model
  (~5.34 on these windows) — retraining on recent data substantially cuts
  later-life error. That part works.
- **Reweighting never beats control.** Every cell above is **positive** —
  boosting is worse than plain retraining in all three configurations, at every
  boost strength. There is no setting where steering feature attention helps.
- **The best case is break-even.** Boosting a genuine aging mechanism (Run 2) at
  minimal strength (1.3) gives +0.0009 — essentially zero, statistically
  indistinguishable from control. It stops *hurting* but does not *help*.
- **Clear dose-response:** the harder the boost, the worse the result. This is
  the signature of overfitting to the boosted feature — forcing more weight onto
  it than the data supports distorts the fit.
- **Regularization doesn't rescue it** (Run 3 still worse), so the harm isn't
  simple overfitting that L2 fixes.

### The honest conclusion

> Diagnosis-conditioned feature reweighting does not improve on plain retraining.
> Across three targeting strategies (ohmic, aging-mechanism, aging+regularization)
> and five boost strengths, boosting was worse than plain retraining in every
> configuration; the best case merely reached break-even. The gain from
> adaptation comes from retraining on fresh data, not from steering the model's
> feature attention.

This is a **legitimate, publishable negative result** — and a thorough one
(swept strength, tested the better-motivated target, added regularization, used a
paired t-test). It tells the field something useful: for this problem, don't
bother reweighting toward the diagnosed mechanism — just retrain.

### Two statistical cautions when reading the output

- **Multiple comparisons:** the script runs up to 10 t-tests (boosts x policies).
  At p<0.05 that's a ~40% chance of one spurious "significant" cell. Trust the
  *consistent trend across the sweep*, not any single cell. For a single claim,
  use a Bonferroni threshold (p < 0.005).
- **These are 5-seed results**, which is what makes the finding defensible: the
  pattern holds averaged across five independent random splits and model
  initializations, so it isn't a fluke of one draw. Reporting more seeds (e.g.
  10) would strengthen it further but is unlikely to change the consistent
  picture.

### On not cherry-picking
There is no configuration where boosting beats control — the best is break-even
at +0.0009. So there is no favorable outlier to build a paper around; reporting
one boost value while hiding the ones that showed harm would be selection bias a
reviewer would catch (and the dose-response pattern is itself the fingerprint of
*no* real effect). The honest, defensible story is the whole sweep: reweighting
doesn't beat retraining.

---

## 10. Dependencies

`pandas`, `numpy`, `scipy`, `catboost`, `scikit-learn`, `pyarrow`.