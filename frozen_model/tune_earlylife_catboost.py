"""
tune_earlylife_catboost.py — search CatBoost hyperparameters to minimize MAE
on the early-life (first few cycles) data.

Trains on cu_index <= --early-life-max-cu, using the same stratified,
recipe-balanced, cell-grouped 3-way split as the other early-life scripts.
Optuna searches over depth / learning_rate / l2_leaf_reg / subsample /
random_strength, selecting the config with the lowest VAL MAE.

WHY VAL, NOT LATER-LIFE:
The search optimizes MAE on the val split (unseen cells inside the training
window), never on later-life data. Tuning against later-life would let the
model peek at the drift regime it's supposed to be blind to -- the whole
point of the early-life framing. After the search, the winning config's
error is reported on test_early AND later-life so you can see how the tuned
model generalizes, but those numbers never influence the search.

The monotonic constraint on deltaZ is kept fixed across all trials (same as
every other model in this project) -- the search tunes the boosting knobs,
not the physical prior.

Usage:
    python tune_earlylife_catboost.py --file path/to/stage2_features_filtered_orthogonalized.parquet --n-trials 50 --output-dir results
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedGroupKFold

DELTAZ_COL = "deltaz_integrated_abs"
ECM_FEATURE_COLS = ["R_ohm_orth", "R_1arc_orth", "sigma_1arc_orth"]
CONDITION_COLS = ["soc_nom", "temp_degC", "is_rt"]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def stratified_group_3way(df, group_col, strat_col, test_size, val_size, seed):
    """Three-way split keeping every cell in one split AND every recipe in each split."""
    n_splits_test = max(2, round(1 / test_size))
    sgkf1 = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf1.split(df, df[strat_col], groups=df[group_col]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    n_splits_val = max(2, round(1 / val_size))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf2.split(trainval, trainval[strat_col], groups=trainval[group_col]))
    return trainval.iloc[tr_idx], trainval.iloc[va_idx], test


def build_monotone(feature_cols):
    idx = feature_cols.index(DELTAZ_COL)
    return [-1 if i == idx else 0 for i in range(len(feature_cols))]


def train_catboost(train, val, feature_cols, params, task_type="CPU", max_iterations=1500):
    monotone = build_monotone(feature_cols)
    model = CatBoostRegressor(
        iterations=max_iterations,
        loss_function="RMSE",
        eval_metric="MAE",
        monotone_constraints=monotone,
        bootstrap_type="Bernoulli",
        random_seed=0,
        early_stopping_rounds=100,
        verbose=False,
        task_type=task_type,
        **params,
    )
    model.fit(Pool(train[feature_cols], label=train["soh_cap"]),
              eval_set=Pool(val[feature_cols], label=val["soh_cap"]),
              use_best_model=True, verbose=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/interim/stage2_features_filtered_orthogonalized.parquet")
    parser.add_argument("--early-life-max-cu", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=1500,
                         help="upper cap on boosting rounds per trial; early stopping usually "
                              "finishes well before this. Lower it to make the search faster.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df = df[df["fit_succeeded"]]

    feature_cols = [DELTAZ_COL] + ECM_FEATURE_COLS + CONDITION_COLS
    data = df.dropna(subset=feature_cols + ["soh_cap"]).copy()
    early = data[data["cu_index"] <= args.early_life_max_cu]
    print(f"Early-life training pool (cu_index <= {args.early_life_max_cu}): "
          f"{len(early)} rows, {early['cell_id'].nunique()} cells")

    train, val, test_early = stratified_group_3way(
        early, "cell_id", "aging_type", args.test_size, args.val_size, args.seed)
    print(f"  train: {len(train)} rows / {train['cell_id'].nunique()} cells,  "
          f"val: {len(val)} rows / {val['cell_id'].nunique()} cells,  "
          f"test_early: {len(test_early)} rows / {test_early['cell_id'].nunique()} cells")
    for nm, sp in [("train", train), ("val", val), ("test_early", test_early)]:
        print(f"    {nm} recipes: {sp.groupby('aging_type')['cell_id'].nunique().to_dict()}")
    print()

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        }
        model = train_catboost(train, val, feature_cols, params, args.task_type, args.max_iterations)
        return mae(model.predict(val[feature_cols]), val["soh_cap"])

    print(f"Running Optuna search ({args.n_trials} trials, optimizing val MAE)...")
    print(f"{'trial':>5}  {'val_MAE':>8}  {'best':>8}  params")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def print_trial(study, trial):
        best = study.best_value
        is_best = "  *NEW*" if abs(trial.value - best) < 1e-12 else ""
        pstr = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in trial.params.items()
        )
        print(f"{trial.number:>5}  {trial.value:>8.4f}  {best:>8.4f}  {pstr}{is_best}")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False,
                   callbacks=[print_trial])

    best_params = study.best_params
    print(f"\nBest val MAE: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print()

    best_model = train_catboost(train, val, feature_cols, best_params, args.task_type, args.max_iterations)
    default_params = {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 3.0,
                       "subsample": 0.8, "min_data_in_leaf": 50}
    baseline_model = train_catboost(train, val, feature_cols, default_params, args.task_type, args.max_iterations)

    data["pred_tuned"] = best_model.predict(data[feature_cols])
    data["pred_baseline"] = baseline_model.predict(data[feature_cols])
    data["abserr_tuned"] = (data["soh_cap"] - data["pred_tuned"]).abs()
    data["abserr_baseline"] = (data["soh_cap"] - data["pred_baseline"]).abs()
    data["split"] = "later_life"
    data.loc[train.index, "split"] = "train"
    data.loc[val.index, "split"] = "val"
    data.loc[test_early.index, "split"] = "test_early"

    print("=" * 70)
    print("MAE / RMSE: tuned vs. baseline CatBoost, by split")
    print("=" * 70)
    print(f"{'split':>12}  {'tuned_MAE':>10}  {'base_MAE':>9}  {'tuned_RMSE':>11}  {'base_RMSE':>10}")
    for split_name in ["train", "val", "test_early", "later_life", "all"]:
        d = data if split_name == "all" else data[data["split"] == split_name]
        print(f"{split_name:>12}  "
              f"{mae(d['pred_tuned'], d['soh_cap']):>10.3f}  "
              f"{mae(d['pred_baseline'], d['soh_cap']):>9.3f}  "
              f"{rmse(d['pred_tuned'], d['soh_cap']):>11.3f}  "
              f"{rmse(d['pred_baseline'], d['soh_cap']):>10.3f}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.file))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(args.file).replace(".parquet", "")

    params_path = os.path.join(output_dir, f"{base}_catboost_best_params.json")
    with open(params_path, "w") as f:
        json.dump({"best_params": best_params, "best_val_mae": study.best_value,
                   "early_life_max_cu": args.early_life_max_cu, "seed": args.seed}, f, indent=2)
    print(f"\nSaved best params: {params_path}")

    model_path = os.path.join(output_dir, f"{base}_catboost_tuned.cbm")
    best_model.save_model(model_path)
    print(f"Saved tuned model: {model_path}")

    # --- NEW: save per-row predictions + abs error for the ENTIRE dataset ---
    pred_cols = [c for c in ["cell_id", "cu_index", "soc_nom", "is_rt", "temp_degC",
                              "aging_type", "split", "soh_cap",
                              "pred_tuned", "pred_baseline",
                              "abserr_tuned", "abserr_baseline"] if c in data.columns]
    preds_path = os.path.join(output_dir, f"{base}_catboost_predictions.parquet")
    data[pred_cols].to_parquet(preds_path, index=False)
    print(f"Saved per-row predictions (entire dataset): {preds_path}")

    # --- NEW: save per-check-up MAE/RMSE for the ENTIRE dataset ---
    per_cu_rows = []
    for cu in sorted(data["cu_index"].unique()):
        g = data[data["cu_index"] == cu]
        per_cu_rows.append({
            "cu_index": int(cu), "split": g["split"].mode().iloc[0], "n_rows": len(g),
            "mae_tuned": mae(g["pred_tuned"], g["soh_cap"]),
            "rmse_tuned": rmse(g["pred_tuned"], g["soh_cap"]),
            "mae_baseline": mae(g["pred_baseline"], g["soh_cap"]),
            "rmse_baseline": rmse(g["pred_baseline"], g["soh_cap"]),
        })
    mae_df = pd.DataFrame(per_cu_rows)
    mae_csv = os.path.join(output_dir, f"{base}_catboost_mae_per_checkup.csv")
    mae_df.to_csv(mae_csv, index=False)
    mae_pq = os.path.join(output_dir, f"{base}_catboost_mae_per_checkup.parquet")
    mae_df.to_parquet(mae_pq, index=False)
    print(f"Saved per-check-up MAE (csv):     {mae_csv}")
    print(f"Saved per-check-up MAE (parquet): {mae_pq}")

    print("\n=== Tuned model: later-life MAE per check-up ===")
    ll = data[data["split"] == "later_life"]
    for cu, g in ll.groupby("cu_index"):
        print(f"  cu_index={int(cu):>2}: MAE={mae(g['pred_tuned'], g['soh_cap']):.3f}  "
              f"(baseline {mae(g['pred_baseline'], g['soh_cap']):.3f})")


if __name__ == "__main__":
    main()