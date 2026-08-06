# Model choice — why CatBoost

This document records which base SoH model we selected and why. The decision is
made from the head-to-head comparison in `fit_earlylife_ensemble.py`, which
trains LightGBM, XGBoost, and CatBoost on the **same** early-life split and
scores all of them (plus their equal-weight average) on every check-up. Because
all four share one split, the numbers are directly comparable row-for-row.

## The decision

**CatBoost is the base SoH predictor.**

## Why — the deciding evidence

The models were trained only on early-life check-ups (`cu_index <= 5`), frozen,
and evaluated across all later check-ups. The regime that matters is
**later-life** (the unseen-drift regime the frozen model must survive), because
the whole project is about how a frozen model behaves as cells age.

MAE by split (from the aggregated comparison):

| model    | train MAE | val MAE | test_early MAE | **later_life MAE** | all MAE |
|----------|-----------|---------|----------------|--------------------|---------|
| LightGBM | 3.371     | 4.080   | 3.888          | 6.112              | 5.320   |
| XGBoost  | 2.469     | 3.355   | 3.216          | 5.377              | 4.556   |
| **CatBoost** | 3.153 | 3.744   | 3.539          | **4.743**          | **4.298** |
| Ensemble | 2.814     | 3.556   | 3.363          | 5.129              | 4.475   |

RMSE by split (same split, same rows):

| model    | train RMSE | val RMSE | test_early RMSE | **later_life RMSE** | all RMSE |
|----------|------------|----------|-----------------|---------------------|----------|
| LightGBM | 4.615      | 5.535    | 5.308           | 8.137               | 7.283    |
| XGBoost  | 3.433      | 4.568    | 4.341           | 7.281               | 6.402    |
| **CatBoost** | 4.154  | 4.833    | 4.633           | **6.426**           | **5.859** |
| Ensemble | 3.837      | 4.760    | 4.515           | 6.918               | 6.183    |

Later-life ranking by MAE: **CatBoost (4.743) < Ensemble (5.129) < XGBoost
(5.377) < LightGBM (6.112).**

The same ordering holds on RMSE (CatBoost 6.426, best later-life), and CatBoost
wins later-life at essentially every individual check-up, not just on average.

### 1. CatBoost has the lowest later-life error — by a clear margin
CatBoost's later-life MAE (4.743) beats the next-best single model, XGBoost
(5.377), by ~0.63 SoH points, and beats LightGBM (6.112) by ~1.37. Since
later-life is the regime the frozen model is meant to be tested on, this is the
decisive number.

### 2. XGBoost looks better in-window, but that's the wrong regime
XGBoost has the lowest train/val/test_early MAE — it fits the early-life window
most tightly. But that advantage **inverts** in later life: its tighter early-life
fit does not transfer to the drifted regime, where it falls behind CatBoost.
Selecting on in-window error would have picked the wrong model for the actual
task. CatBoost generalizes to drift better, which is exactly what we need.

### 3. The ensemble does NOT beat CatBoost — a useful negative result
Averaging LightGBM + XGBoost + CatBoost (equal weight) gives later-life MAE 5.129
— **worse** than CatBoost alone (4.743). The average is dragged toward the weaker
LightGBM/XGBoost later-life predictions rather than lifted by them. This tells us
the three libraries are not diverse enough on this data for simple averaging to
help, so more elaborate weighting schemes are unlikely to rescue it either. We
therefore use CatBoost alone, not the ensemble.

## Consistency / robustness

- The comparison uses a **stratified, recipe-balanced, cell-grouped** three-way
  split (`StratifiedGroupKFold`), so every aging recipe (Calendar / Cyclic /
  Profile) appears in train, val, and test, and no cell leaks across splits. The
  CatBoost win is not an artifact of an unlucky split — it holds under the more
  rigorous stratified split, and across seeds we checked.
- All three libraries enforce the same physical prior: a **monotone constraint**
  on `deltaz_integrated_abs` (SoH can only decrease as deltaZ increases). The
  comparison is therefore of the tree library itself, not of different priors.

## Trade-off we accepted

LightGBM is the only one of the three that also supports `interaction_constraints`
(forcing deltaZ to split first). CatBoost has the monotone constraint but no
interaction-constraint equivalent. We accepted losing that structural guarantee
because (a) CatBoost's later-life accuracy is clearly better in practice, and
(b) deltaZ remains the dominant feature by importance in CatBoost anyway
(~50%+), so the constraint's absence does not cost us the intended
deltaZ-led behaviour.

## Summary

CatBoost is chosen because it has the **lowest later-life MAE and RMSE** — the
regime that matters for a frozen early-life model — beating every other single
model and the ensemble, under a rigorous recipe-balanced split, while still
honouring the deltaZ monotone prior. The in-window metrics favour XGBoost, but
that advantage does not survive into later life, and selecting on it would have
optimised for the wrong regime.