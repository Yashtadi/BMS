"""
fit_earlylife_ensemble.py — average predictions from LightGBM, XGBoost, and
CatBoost trained on the same early-life pool.

The reasoning: each library uses a slightly different tree-building
mechanism (LightGBM's leaf-wise growth + interaction constraints,
XGBoost's level-wise growth, CatBoost's oblivious trees + ordered
boosting), so their errors on unseen data are only partially correlated.
Averaging can cancel out that uncorrelated error component while keeping
the shared signal.

The equal-weight mean is deliberately used here rather than an optimized
weighting -- it's the honest baseline for "does combining help at all".
If simple averaging already beats the best single model, the more elaborate
weight-learning schemes are worth exploring; if it doesn't, they probably
aren't going to save it, because the models aren't diverse enough on this
data.

Same three-way grouped split as the individual scripts (train/val/
test_early), so ensemble numbers are directly comparable to what each
individual script reported.

Usage:
    python fit_earlylife_ensemble.py --file path/to/stage2_features_filtered_orthogonalized.parquet --output-dir results
"""

import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold


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

DELTAZ_COL = "deltaz_integrated_abs"
ECM_FEATURE_COLS = ["R_ohm_orth", "R_1arc_orth", "sigma_1arc_orth"]
CONDITION_COLS = ["soc_nom", "temp_degC", "is_rt"]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def train_lgb(train, val, feature_cols):
    deltaz_idx = feature_cols.index(DELTAZ_COL)
    monotone = [0] * len(feature_cols)
    monotone[deltaz_idx] = -1
    other = [i for i in range(len(feature_cols)) if i != deltaz_idx]
    ic = [[deltaz_idx]] + [[deltaz_idx, i] for i in other]

    params = {
        "objective": "regression", "metric": "rmse",
        "monotone_constraints": monotone, "monotone_constraints_method": "advanced",
        "interaction_constraints": ic,
        "num_leaves": 15, "learning_rate": 0.03, "min_data_in_leaf": 50,
        "verbosity": -1, "seed": 0,
    }
    d_train = lgb.Dataset(train[feature_cols], label=train["soh_cap"])
    d_val = lgb.Dataset(val[feature_cols], label=val["soh_cap"], reference=d_train)
    return lgb.train(params, d_train, num_boost_round=2000, valid_sets=[d_val],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def train_xgb(train, val, feature_cols):
    deltaz_idx = feature_cols.index(DELTAZ_COL)
    monotone = tuple(-1 if i == deltaz_idx else 0 for i in range(len(feature_cols)))
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "monotone_constraints": str(monotone),
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 50,
        "verbosity": 0, "seed": 0,
    }
    d_train = xgb.DMatrix(train[feature_cols].to_numpy(), label=train["soh_cap"].to_numpy(),
                           feature_names=feature_cols)
    d_val = xgb.DMatrix(val[feature_cols].to_numpy(), label=val["soh_cap"].to_numpy(),
                         feature_names=feature_cols)
    return xgb.train(params, d_train, num_boost_round=2000, evals=[(d_val, "val")],
                     early_stopping_rounds=100, verbose_eval=False)


def train_cat(train, val, feature_cols, task_type="CPU"):
    deltaz_idx = feature_cols.index(DELTAZ_COL)
    monotone = [-1 if i == deltaz_idx else 0 for i in range(len(feature_cols))]
    model = CatBoostRegressor(
        iterations=2000, loss_function="RMSE", eval_metric="RMSE",
        monotone_constraints=monotone, bootstrap_type="Bernoulli",
        depth=4, learning_rate=0.03, min_data_in_leaf=50,
        random_seed=0, early_stopping_rounds=100, verbose=False, task_type=task_type,
    )
    model.fit(Pool(train[feature_cols], label=train["soh_cap"]),
              eval_set=Pool(val[feature_cols], label=val["soh_cap"]),
              use_best_model=True, verbose=False)
    return model


def predict_lgb(model, X, feature_cols):
    return model.predict(X[feature_cols])


def predict_xgb(model, X, feature_cols):
    dmat = xgb.DMatrix(X[feature_cols].to_numpy(), feature_names=feature_cols)
    return model.predict(dmat)


def predict_cat(model, X, feature_cols):
    return model.predict(X[feature_cols])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/interim/stage2_features_filtered_orthogonalized.parquet")
    parser.add_argument("--early-life-max-cu", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--error-threshold", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df = df[df["fit_succeeded"]]

    feature_cols = [DELTAZ_COL] + ECM_FEATURE_COLS + CONDITION_COLS
    data = df.dropna(subset=feature_cols + ["soh_cap"]).copy()
    print(f"Usable rows total: {len(data)}")

    early = data[data["cu_index"] <= args.early_life_max_cu]
    print(f"Early-life training pool (cu_index <= {args.early_life_max_cu}): "
          f"{len(early)} rows, {early['cell_id'].nunique()} cells")

    train, val, test_early = stratified_group_3way(
        early, group_col="cell_id", strat_col="aging_type",
        test_size=args.test_size, val_size=args.val_size, seed=args.seed)

    print(f"  train:      {len(train):5d} rows, {train['cell_id'].nunique():3d} cells")
    print(f"  val:        {len(val):5d} rows, {val['cell_id'].nunique():3d} cells")
    print(f"  test_early: {len(test_early):5d} rows, {test_early['cell_id'].nunique():3d} cells")
    for nm, sp in [("train", train), ("val", val), ("test_early", test_early)]:
        rc = sp.groupby("aging_type")["cell_id"].nunique().to_dict()
        print(f"    {nm} recipes: {rc}")
    print()

    # --- Train all three libraries on the same train/val split ---
    print("Training LightGBM...")
    lgb_model = train_lgb(train, val, feature_cols)
    print("Training XGBoost...")
    xgb_model = train_xgb(train, val, feature_cols)
    print("Training CatBoost...")
    cat_model = train_cat(train, val, feature_cols, args.task_type)
    print()

    # --- Score every row of the full dataset with each model ---
    data["pred_lgb"] = predict_lgb(lgb_model, data, feature_cols)
    data["pred_xgb"] = predict_xgb(xgb_model, data, feature_cols)
    data["pred_cat"] = predict_cat(cat_model, data, feature_cols)
    data["soh_cap_predicted"] = (data["pred_lgb"] + data["pred_xgb"] + data["pred_cat"]) / 3.0
    data["residual"] = data["soh_cap"] - data["soh_cap_predicted"]
    data["abs_residual"] = data["residual"].abs()
    data["error_too_big"] = (data["abs_residual"] > args.error_threshold).astype(int)

    # Tag splits so the per-check-up report can break errors down honestly
    data["split"] = "later_life"
    data.loc[train.index, "split"] = "train"
    data.loc[val.index, "split"] = "val"
    data.loc[test_early.index, "split"] = "test_early"

    print("=== Ensemble RMSE on early-life splits ===")
    print(f"  train RMSE:      {rmse(train[feature_cols].pipe(lambda X: (predict_lgb(lgb_model, X, feature_cols) + predict_xgb(xgb_model, X, feature_cols) + predict_cat(cat_model, X, feature_cols)) / 3.0), train['soh_cap']):.3f}")
    print(f"  val RMSE:        {rmse(val[feature_cols].pipe(lambda X: (predict_lgb(lgb_model, X, feature_cols) + predict_xgb(xgb_model, X, feature_cols) + predict_cat(cat_model, X, feature_cols)) / 3.0), val['soh_cap']):.3f}")
    print(f"  test_early RMSE: {rmse(test_early[feature_cols].pipe(lambda X: (predict_lgb(lgb_model, X, feature_cols) + predict_xgb(xgb_model, X, feature_cols) + predict_cat(cat_model, X, feature_cols)) / 3.0), test_early['soh_cap']):.3f}  <- honest test on unseen cells within training window")
    print()

    # --- Per-check-up performance across full life, broken down by split ---
    # Also includes the individual-model MAEs alongside the ensemble MAE so
    # you can see per check-up whether the ensemble is actually beating its
    # components or just landing between them.
    print("=" * 120)
    print(f"Model performance per check-up (error threshold = {args.error_threshold} SoH points)")
    print(f"Ensemble MAE reported alongside the three individual libraries for direct comparison")
    print("=" * 120)
    print(f"{'cu_index':>8}  {'split':>11}  {'n':>5}  "
          f"{'MAE_lgb':>7}  {'MAE_xgb':>7}  {'MAE_cat':>7}  {'MAE_ens':>7}  "
          f"{'RMSE_ens':>8}  {'frac_big':>8}")
    print("-" * 120)
    summary_rows = []
    for cu in sorted(data["cu_index"].unique()):
        cu_data = data[data["cu_index"] == cu]
        for split_name in ["train", "val", "test_early", "later_life"]:
            group = cu_data[cu_data["split"] == split_name]
            if len(group) == 0:
                continue
            mae_l = mae(group["pred_lgb"], group["soh_cap"])
            mae_x = mae(group["pred_xgb"], group["soh_cap"])
            mae_c = mae(group["pred_cat"], group["soh_cap"])
            mae_e = mae(group["soh_cap_predicted"], group["soh_cap"])
            rmse_e = rmse(group["soh_cap_predicted"], group["soh_cap"])
            frac = group["error_too_big"].mean()
            print(f"{cu:>8}  {split_name:>11}  {len(group):>5}  "
                  f"{mae_l:>7.3f}  {mae_x:>7.3f}  {mae_c:>7.3f}  {mae_e:>7.3f}  "
                  f"{rmse_e:>8.3f}  {frac:>8.3f}")
            summary_rows.append({
                "cu_index": int(cu), "split": split_name, "n_rows": len(group),
                "mae_lgb": mae_l, "mae_xgb": mae_x, "mae_cat": mae_c,
                "mae_ensemble": mae_e, "rmse_ensemble": rmse_e,
                "frac_error_too_big": frac,
            })
        print()

    # --- Save outputs ---
    input_dir = os.path.dirname(os.path.abspath(args.file))
    output_dir = args.output_dir if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(args.file).replace(".parquet", "")

    out_rows = os.path.join(output_dir, f"{base}_earlylife_ensemble_predictions.csv")
    data.to_csv(out_rows, index=False)
    print()
    print(f"Saved per-row predictions + labels: {out_rows}")

    out_summary = os.path.join(output_dir, f"{base}_earlylife_ensemble_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False)
    print(f"Saved per-check-up summary: {out_summary}")

    # --- Combined parquet: every model's prediction + soh_cap + cell/check-up
    # details, one row per spectrum. All four predictions come from models
    # trained on the same split, so the columns are directly comparable
    # row-for-row -- you can reproduce any model-vs-model comparison straight
    # from this file without re-running anything.
    id_cols = ["cell_id", "cu_index", "soc_nom", "is_rt", "temp_degC", "aging_type", "split"]
    combined = pd.DataFrame()
    for col in id_cols:
        if col in data.columns:
            combined[col] = data[col].values
    combined["soh_cap"] = data["soh_cap"].values          # ground truth

    combined["pred_lightgbm"] = data["pred_lgb"].values
    combined["pred_xgboost"] = data["pred_xgb"].values
    combined["pred_catboost"] = data["pred_cat"].values
    combined["pred_ensemble"] = data["soh_cap_predicted"].values   # equal-weight mean

    # Per-model residual (soh_cap - prediction) for downstream error analysis
    combined["residual_lightgbm"] = combined["soh_cap"] - combined["pred_lightgbm"]
    combined["residual_xgboost"] = combined["soh_cap"] - combined["pred_xgboost"]
    combined["residual_catboost"] = combined["soh_cap"] - combined["pred_catboost"]
    combined["residual_ensemble"] = combined["soh_cap"] - combined["pred_ensemble"]

    # Per-row absolute error and squared error for each model. With these in
    # the file, MAE/RMSE for ANY slice come straight out of this one parquet:
    #   df.groupby("cu_index")["abserr_catboost"].mean()          -> MAE per check-up
    #   df.groupby("split")["sqerr_catboost"].mean() ** 0.5       -> RMSE per split
    # No separate metrics file needed -- the numbers live with the predictions.
    for name in ["lightgbm", "xgboost", "catboost", "ensemble"]:
        combined[f"abserr_{name}"] = combined[f"residual_{name}"].abs()
        combined[f"sqerr_{name}"] = combined[f"residual_{name}"] ** 2

    combined = combined.sort_values(["cell_id", "cu_index"]).reset_index(drop=True)

    # --- Add MAE/RMSE columns directly into this same table ---
    # For each model, every row gets its CHECK-UP's MAE and RMSE (computed over all
    # rows sharing that cu_index) broadcast onto it. So a row at cu_index=20
    # carries mae_catboost / rmse_catboost = the error of the CatBoost model
    # across all cu_index=20 rows. This puts the aggregate error numbers in the
    # same file as the predictions, at check-up granularity.
    for name, pred_col in [("lightgbm", "pred_lightgbm"), ("xgboost", "pred_xgboost"),
                            ("catboost", "pred_catboost"), ("ensemble", "pred_ensemble")]:
        # per-check-up MAE = mean absolute error within each cu_index
        mae_by_cu = combined.groupby("cu_index")[f"abserr_{name}"].transform("mean")
        # per-check-up RMSE = sqrt(mean squared error within each cu_index)
        rmse_by_cu = np.sqrt(combined.groupby("cu_index")[f"sqerr_{name}"].transform("mean"))
        combined[f"mae_{name}"] = mae_by_cu
        combined[f"rmse_{name}"] = rmse_by_cu

    out_parquet = os.path.join(output_dir, f"{base}_earlylife_all_models_predictions.parquet")
    combined.to_parquet(out_parquet, index=False)
    print(f"Saved combined all-models parquet: {out_parquet}")
    print(f"  {len(combined)} rows, {combined['cell_id'].nunique()} cells, columns: {combined.columns.tolist()}")

    # --- Per-model MAE/RMSE summary, broken down by split ---
    # One row per (model, split): the paper-ready head-to-head comparison.
    # "later_life" is the row that matters most -- it's the unseen-drift regime.
    model_pred_cols = {
        "LightGBM": "pred_lgb",
        "XGBoost": "pred_xgb",
        "CatBoost": "pred_cat",
        "Ensemble": "soh_cap_predicted",
    }
    metric_rows = []
    for split_name in ["train", "val", "test_early", "later_life"]:
        split_data = data[data["split"] == split_name]
        if len(split_data) == 0:
            continue
        for model_name, pred_col in model_pred_cols.items():
            metric_rows.append({
                "split": split_name,
                "cu_index": -1,       # -1 = aggregated over all check-ups in this split
                "model": model_name,
                "n_rows": len(split_data),
                "mae": mae(split_data[pred_col], split_data["soh_cap"]),
                "rmse": rmse(split_data[pred_col], split_data["soh_cap"]),
            })
    # Also an overall "all rows" block, for a single headline number per model
    for model_name, pred_col in model_pred_cols.items():
        metric_rows.append({
            "split": "all",
            "cu_index": -1,          # -1 = aggregated over all check-ups in this split
            "model": model_name,
            "n_rows": len(data),
            "mae": mae(data[pred_col], data["soh_cap"]),
            "rmse": rmse(data[pred_col], data["soh_cap"]),
        })

    # Per-check-up MAE/RMSE per model -- captures the later-cycle error growth
    # at cu_index granularity (cu_index 0-28), tagged with the split each
    # check-up's rows belong to. This is the fine-grained later-cycle result.
    for cu in sorted(data["cu_index"].unique()):
        cu_data = data[data["cu_index"] == cu]
        # a check-up's rows can span multiple splits (train/val/test_early in
        # the early window); report the dominant split label for reference
        split_label = cu_data["split"].mode().iloc[0]
        for model_name, pred_col in model_pred_cols.items():
            metric_rows.append({
                "split": split_label,
                "cu_index": int(cu),
                "model": model_name,
                "n_rows": len(cu_data),
                "mae": mae(cu_data[pred_col], cu_data["soh_cap"]),
                "rmse": rmse(cu_data[pred_col], cu_data["soh_cap"]),
            })

    metrics_df = pd.DataFrame(metric_rows)

    out_metrics = os.path.join(output_dir, f"{base}_earlylife_model_metrics.csv")
    metrics_df.to_csv(out_metrics, index=False)
    print(f"Saved per-model MAE/RMSE summary (CSV):     {out_metrics}")

    out_metrics_pq = os.path.join(output_dir, f"{base}_earlylife_model_metrics.parquet")
    metrics_df.to_parquet(out_metrics_pq, index=False)
    print(f"Saved per-model MAE/RMSE summary (parquet): {out_metrics_pq}")
    print()
    print("=== Per-model MAE / RMSE by split ===")
    for split_name in ["train", "val", "test_early", "later_life", "all"]:
        block = metrics_df[(metrics_df["split"] == split_name) & (metrics_df["cu_index"] == -1)]
        if len(block) == 0:
            continue
        print(f"\n  [{split_name}]  (n={block.iloc[0]['n_rows']})")
        print(f"    {'model':>10}  {'MAE':>7}  {'RMSE':>7}")
        for _, r in block.iterrows():
            print(f"    {r['model']:>10}  {r['mae']:>7.3f}  {r['rmse']:>7.3f}")

    # --- Plot: ensemble MAE per split, and later-life comparison against individuals ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # LEFT: ensemble MAE broken down by split (matches the individual scripts' plot)
    split_colors = {
        "train":      ("#1f77b4", "o", "train"),
        "val":        ("#ff7f0e", "s", "val"),
        "test_early": ("#2ca02c", "D", "test (early-life, unseen cells)"),
        "later_life": ("#d62728", "^", "later-life (unseen check-ups)"),
    }
    for split_name, (color, marker, label) in split_colors.items():
        rows = summary_df[summary_df["split"] == split_name].sort_values("cu_index")
        if len(rows) == 0:
            continue
        axes[0].plot(rows["cu_index"], rows["mae_ensemble"], marker=marker, color=color,
                     label=label, linestyle="-")
    boundary = args.early_life_max_cu + 0.5
    axes[0].axvline(boundary, color="black", linestyle=":", linewidth=1.2,
                     label=f"training boundary\n(cu_index <= {args.early_life_max_cu})")
    axes[0].set_xlabel("cu_index (check-up number)")
    axes[0].set_ylabel("MAE (SoH points)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", framealpha=0.9)
    axes[0].set_title("Ensemble MAE per check-up, broken down by split")

    # RIGHT: later-life MAE, all four models on the same axis, so you can see
    # whether averaging beats or just averages
    later = summary_df[summary_df["split"] == "later_life"].sort_values("cu_index")
    axes[1].plot(later["cu_index"], later["mae_lgb"], marker="o", label="LightGBM", color="#1f77b4")
    axes[1].plot(later["cu_index"], later["mae_xgb"], marker="s", label="XGBoost", color="#d62728")
    axes[1].plot(later["cu_index"], later["mae_cat"], marker="D", label="CatBoost", color="#2ca02c")
    axes[1].plot(later["cu_index"], later["mae_ensemble"], marker="*", markersize=10,
                 label="Ensemble (mean of 3)", color="black", linewidth=2)
    axes[1].set_xlabel("cu_index (check-up number)")
    axes[1].set_ylabel("MAE (SoH points)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", framealpha=0.9)
    axes[1].set_title("Later-life MAE comparison: ensemble vs. individuals")

    plt.suptitle(f"Ensemble of LightGBM + XGBoost + CatBoost (equal-weight average)\n"
                  f"trained on cu_index <= {args.early_life_max_cu}, error threshold = "
                  f"{args.error_threshold} SoH points", fontsize=13)
    plt.tight_layout()

    out_plot = os.path.join(output_dir, f"{base}_earlylife_ensemble_error_trend.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved error-trend plot: {out_plot}")


if __name__ == "__main__":
    main()