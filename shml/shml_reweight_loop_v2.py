"""
shml_reweight_loop_v2.py — SHML adaptation by FEATURE REWEIGHTING.
Extends shml_reweight_loop.py with the fixes suggested by the v1 result
(control beat frozen; single_boost/shared_boost were WORSE than control
at boost=3.0, worse in proportion to how concentrated the boost was):

  1. --boost-list sweeps MULTIPLE boost strengths in ONE run, reusing the
     same per-seed frozen model across all boost values (no wasted retrains).
  2. --exclude-ohmic switches the dominant-mechanism selection to the
     AGING view (drops R_ohm_1arc, renormalizes) -- lets you test whether
     boosting a genuinely age-related mechanism (interfacial/diffusion/
     spectral) works better than boosting ohmic resistance, which the
     original diagnosis script itself flags as possibly contact/cabling
     noise rather than electrode aging.
  3. --boosted-l2-mult applies EXTRA L2 regularization only to the boosted
     policies (single_boost / shared_boost), to counter the overfitting-
     to-one-feature failure mode the dose-response pattern in v1 suggested
     (single_boost, the more concentrated boost, hurt more than shared_boost).
  4. Adds a real paired t-test (scipy) alongside the original 2*SE screen,
     since n=20 with small effect sizes needs more than a rough screen.

Usage (boost sweep):
    python shml_reweight_loop_v2.py \
        --file data/interim/stage2_features_filtered_orthogonalized.parquet \
        --rec-file shml/results/..._shml_recommendation.json \
        --error-threshold 3.6 --window-starts 10,14,18,22 --window-size 4 \
        --seeds 0,1,2,3,4 --boost-list 1.3,1.5,2.0,3.0,5.0 \
        --output-dir shml/results

Usage (exclude ohmic + regularize boosted variants):
    python shml_reweight_loop_v2.py \
        --file data/interim/stage2_features_filtered_orthogonalized.parquet \
        --rec-file shml/results/..._shml_recommendation.json \
        --error-threshold 3.6 --window-starts 10,14,18,22 --window-size 4 \
        --seeds 0,1,2,3,4 --boost-list 1.5,3.0 --exclude-ohmic \
        --boosted-l2-mult 2.0 --output-dir shml/results
"""

import argparse
import json
import os
import sys
import contextlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupShuffleSplit
from scipy import stats


# CatBoost prints this harmless C++ warning once per warm-start retrain:
#   "Model shrinkage in combination with learning continuation is not
#    implemented yet. Reset model_shrink_rate to 0."
# It is expected (we intentionally warm-start from the frozen model) and just
# clutters the console. This context manager filters ONLY that line from stderr,
# letting every other message through unchanged.
_SUPPRESS = "Model shrinkage in combination with learning continuation"


class _FilterStderr:
    def __init__(self, real):
        self.real = real
    def write(self, msg):
        if _SUPPRESS not in msg:
            self.real.write(msg)
    def flush(self):
        self.real.flush()


@contextlib.contextmanager
def quiet_shrinkage_warning():
    old = sys.stderr
    sys.stderr = _FilterStderr(old)
    try:
        yield
    finally:
        sys.stderr = old

DELTAZ_COL = "deltaz_integrated_abs"
ECM_ORTH = ["R_ohm_orth", "R_1arc_orth", "sigma_1arc_orth"]
CONDITIONS = ["soc_nom", "temp_degC", "is_rt"]
FEATURES = [DELTAZ_COL] + ECM_ORTH + CONDITIONS
OHMIC = "R_ohm_1arc"

CAUSE_TO_FEATURE = {
    "R_ohm_1arc": "R_ohm_orth",
    "R_1arc": "R_1arc_orth",
    "sigma_1arc": "sigma_1arc_orth",
    "n_1arc": DELTAZ_COL,
    "deltaz_integrated_abs": DELTAZ_COL,
}


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def build_monotone(fc):
    i = fc.index(DELTAZ_COL)
    return [-1 if k == i else 0 for k in range(len(fc))]


def train(train_df, params, seed, feature_weights=None, init_model=None,
          max_iter=250, l2_mult=1.0):
    p = dict(params)
    if l2_mult != 1.0 and "l2_leaf_reg" in p:
        p["l2_leaf_reg"] = float(p["l2_leaf_reg"]) * l2_mult
    kw = {}
    if feature_weights is not None:
        kw["feature_weights"] = feature_weights
    model = CatBoostRegressor(
        iterations=max_iter, loss_function="RMSE", eval_metric="MAE",
        monotone_constraints=build_monotone(FEATURES), bootstrap_type="Bernoulli",
        random_seed=seed, verbose=False, **p, **kw)
    fit_kw = {}
    if init_model is not None:
        fit_kw["init_model"] = init_model
    with quiet_shrinkage_warning():
        model.fit(Pool(train_df[FEATURES], label=train_df["soh_cap"]),
                  verbose=False, **fit_kw)
    return model


def train_frozen(data, baseline_max_cu, params, seed, max_iter=250):
    early = data[data["cu_index"] <= baseline_max_cu]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, _ = next(gss.split(early, groups=early["cell_id"]))
    return train(early.iloc[tr_idx], params, seed, max_iter=max_iter)


def weights_single(dominant_feature, boost):
    return [boost if f == dominant_feature else 1.0 for f in FEATURES]


def weights_shared(contrib_pct, boost):
    w = []
    for f in FEATURES:
        share = 0.0
        for cause, feat in CAUSE_TO_FEATURE.items():
            if feat == f and cause in contrib_pct:
                share = max(share, contrib_pct[cause] / 100.0)
        w.append(1.0 + (boost - 1.0) * share if share > 0 else 1.0)
    return w


def first_drift_cu(mae_file, threshold, baseline_max_cu, mae_col="mae_tuned"):
    """Read the per-checkup MAE file and return the FIRST later-life cu_index
    whose MAE exceeds `threshold`. This makes the adaptation windows start
    exactly where the model has drifted, by the model's own criterion —
    rather than a hardcoded start. Returns None if nothing crosses."""
    df = pd.read_csv(mae_file) if str(mae_file).endswith(".csv") else pd.read_parquet(mae_file)
    if mae_col not in df.columns:
        # fall back to whatever MAE column exists
        cand = [c for c in df.columns if c.startswith("mae")]
        if not cand:
            raise SystemExit(f"no MAE column in {mae_file}; columns: {list(df.columns)}")
        mae_col = cand[0]
    later = df[df["cu_index"] > baseline_max_cu].sort_values("cu_index")
    crossed = later[later[mae_col] > threshold]
    if crossed.empty:
        return None
    return int(crossed.iloc[0]["cu_index"])


def make_window_starts(drift_cu, window_size, max_cu, n_windows=None):
    """Tile non-overlapping adapt+gate window PAIRS starting at drift_cu.
    Each pair needs 2*window_size cu values (adapt then gate), so starts step
    by window_size and must leave room for a full gate window."""
    starts = []
    s = drift_cu
    while s + 2 * window_size - 1 <= max_cu:
        starts.append(s)
        s += window_size
        if n_windows and len(starts) >= n_windows:
            break
    return starts


def pick_dominant(rec, exclude_ohmic):
    """Return (dominant_cause, dominant_feature, contrib_dict_used).
    exclude_ohmic=True -> renormalize shares excluding R_ohm_1arc and
    pick the top remaining ECM mechanism (the 'aging view')."""
    contrib = rec.get("contribution_pct_full", {})
    mech_present = [f for f in ["R_ohm_1arc", "R_1arc", "n_1arc", "sigma_1arc"] if f in contrib]
    if not exclude_ohmic or OHMIC not in mech_present:
        dom = rec.get("overall_dominant", "R_ohm_1arc")
        return dom, CAUSE_TO_FEATURE.get(dom, DELTAZ_COL), contrib
    aging = [f for f in mech_present if f != OHMIC]
    dom = max(aging, key=lambda f: contrib.get(f, 0.0))
    aging_total = sum(contrib.get(f, 0.0) for f in aging) or 1.0
    aging_contrib = {f: 100.0 * contrib.get(f, 0.0) / aging_total for f in aging}
    return dom, CAUSE_TO_FEATURE.get(dom, DELTAZ_COL), aging_contrib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True)
    ap.add_argument("--rec-file", required=True,
                     help="the _shml_recommendation.json — used for the DIAGNOSIS "
                          "(which mechanism/feature to boost, and the contribution shares)")
    ap.add_argument("--params-file", default=None,
                     help="the _catboost_best_params.json from tune_earlylife_catboost.py. "
                          "If given, the frozen model's hyperparameters are read from its "
                          "'best_params' key (the tuned model). If omitted, falls back to the "
                          "rec-file's 'current_params'.")
    ap.add_argument("--baseline-max-cu", type=int, default=5)
    ap.add_argument("--error-threshold", type=float, default=3.6)
    ap.add_argument("--window-starts", default="10,14,18,22",
                     help="explicit window starts; IGNORED if --auto-start-from-mae is set")
    ap.add_argument("--auto-start-from-mae", default=None,
                     help="path to the ..._catboost_mae_per_checkup.csv/.parquet. If given, "
                          "window starts are auto-set to begin at the first later-life cu whose "
                          "MAE exceeds --error-threshold (the data-driven drift point), and tile "
                          "later life from there — instead of using --window-starts.")
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--boost-list", default="3.0",
                     help="comma-separated boost values to sweep, e.g. 1.3,1.5,2.0,3.0,5.0")
    ap.add_argument("--exclude-ohmic", action="store_true",
                     help="pick dominant mechanism from the AGING view (drop R_ohm_1arc)")
    ap.add_argument("--boosted-l2-mult", type=float, default=1.0,
                     help="multiply l2_leaf_reg by this for single_boost/shared_boost only "
                          "(counters overfitting to the boosted feature); 1.0 = no change")
    ap.add_argument("--max-iter", type=int, default=1000,
                     help="CatBoost iterations; lower for faster exploratory sweeps")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    data = pd.read_parquet(args.file)
    data = data[data["fit_succeeded"]].dropna(subset=FEATURES + ["soh_cap"]).copy()

    rec = json.load(open(args.rec_file))
    # Frozen model hyperparameters: prefer the tuned best_params.json (the actual
    # tuning output), fall back to the rec-file's current_params.
    if args.params_file:
        bp = json.load(open(args.params_file))
        params = bp.get("best_params", bp)   # accept {"best_params": {...}} or a bare {...}
        params_source = f"{os.path.basename(args.params_file)} (best_params)"
    else:
        params = rec["current_params"]
        params_source = f"{os.path.basename(args.rec_file)} (current_params)"
    # Diagnosis (which feature to boost + shares) always comes from the rec-file.
    dominant_cause, dominant_feature, contrib = pick_dominant(rec, args.exclude_ohmic)
    print(f"Frozen params        : {params}")
    print(f"  (params source)    : {params_source}")
    print(f"exclude-ohmic        : {args.exclude_ohmic}")
    print(f"Dominant cause       : {dominant_cause} -> boost feature '{dominant_feature}'")
    print(f"Contribution shares  : {contrib}")
    print(f"boosted L2 mult      : {args.boosted_l2_mult}\n")

    max_cu = int(data["cu_index"].max())
    if args.auto_start_from_mae:
        drift_cu = first_drift_cu(args.auto_start_from_mae, args.error_threshold,
                                  args.baseline_max_cu)
        if drift_cu is None:
            raise SystemExit(f"MAE never exceeds {args.error_threshold} after cu "
                             f"{args.baseline_max_cu} in {args.auto_start_from_mae}; "
                             f"nothing to adapt.")
        window_starts = make_window_starts(drift_cu, args.window_size, max_cu)
        print(f"AUTO-START: first cu with MAE > {args.error_threshold} is cu={drift_cu}")
        print(f"  -> window starts (tiled from drift point): {window_starts}\n")
        if not window_starts:
            raise SystemExit(f"drift at cu={drift_cu} leaves no room for a full adapt+gate "
                             f"pair (need {2*args.window_size} cu, max cu={max_cu}).")
    else:
        window_starts = [int(x) for x in args.window_starts.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    boosts = [float(x) for x in args.boost_list.split(",")]

    records = []
    for seed in seeds:
        frozen = train_frozen(data, args.baseline_max_cu, params, seed, max_iter=args.max_iter)
        for start in window_starts:
            adapt_cus = range(start, start + args.window_size)
            gate_cus = range(start + args.window_size, start + 2 * args.window_size)
            adapt_win = data[data["cu_index"].isin(adapt_cus)]
            gate_win = data[data["cu_index"].isin(gate_cus)]
            if len(adapt_win) < 30 or len(gate_win) < 30:
                continue
            frozen_gate = mae(frozen.predict(gate_win[FEATURES]), gate_win["soh_cap"])
            drifted = mae(frozen.predict(adapt_win[FEATURES]), adapt_win["soh_cap"]) > args.error_threshold

            # control trained once per (seed, window) -- boost-independent
            if drifted:
                m_ctrl = train(adapt_win, params, seed, feature_weights=None, init_model=frozen,
                                max_iter=args.max_iter)
                control_mae = mae(m_ctrl.predict(gate_win[FEATURES]), gate_win["soh_cap"])
                control_mae = control_mae if control_mae < frozen_gate else frozen_gate
            else:
                control_mae = frozen_gate

            for boost in boosts:
                w_single = weights_single(dominant_feature, boost)
                w_shared = weights_shared(contrib, boost)
                row = {"seed": seed, "window_start": start, "boost": boost,
                       "drifted": drifted, "frozen_gate_mae": frozen_gate,
                       "control_gate_mae": control_mae}
                for name, fw in [("single_boost", w_single), ("shared_boost", w_shared)]:
                    if not drifted:
                        row[f"{name}_gate_mae"] = frozen_gate
                        row[f"{name}_deployed"] = False
                        continue
                    m = train(adapt_win, params, seed, feature_weights=fw, init_model=frozen,
                              max_iter=args.max_iter, l2_mult=args.boosted_l2_mult)
                    cand = mae(m.predict(gate_win[FEATURES]), gate_win["soh_cap"])
                    deploy = cand < frozen_gate
                    row[f"{name}_gate_mae"] = cand if deploy else frozen_gate
                    row[f"{name}_deployed"] = deploy
                records.append(row)
        n_windows = len([s for s in window_starts
                         if len(data[data["cu_index"].isin(range(s, s + args.window_size))]) >= 30])
        drifted_here = sum(1 for r in records if r["seed"] == seed and r["boost"] == boosts[0]
                           and r["drifted"])
        print(f"[seed {seed+1}/{len(seeds)}] frozen model trained + adapted over "
              f"{n_windows} windows ({drifted_here} drifted) × {len(boosts)} boost levels")

    print(f"\nAll seeds complete — {len(records)} (seed × window × boost) evaluations.")

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No valid windows — check --window-starts / --window-size.")

    print("\n" + "=" * 78)
    print(f"SWEEP RESULTS — gate MAE by boost value ({len(seeds)} seeds x {len(window_starts)} windows)")
    print("=" * 78)
    print(f"{'boost':>7} {'control':>10} {'single_boost':>14} {'shared_boost':>14} "
          f"{'single-ctrl':>12} {'shared-ctrl':>12}")
    summary_rows = []
    for boost in boosts:
        sub = df[df["boost"] == boost]
        c = sub["control_gate_mae"].mean()
        s1 = sub["single_boost_gate_mae"].mean()
        s2 = sub["shared_boost_gate_mae"].mean()
        d1, d2 = s1 - c, s2 - c
        print(f"{boost:>7.2f} {c:>10.4f} {s1:>14.4f} {s2:>14.4f} {d1:>+12.4f} {d2:>+12.4f}")
        summary_rows.append({"boost": boost, "control_mae": c, "single_boost_mae": s1,
                              "shared_boost_mae": s2, "single_minus_control": d1,
                              "shared_minus_control": d2})

    print("\n" + "=" * 78)
    print("PAIRED STATS PER BOOST VALUE (paired t-test, control vs boosted)")
    print("=" * 78)
    for boost in boosts:
        sub = df[(df["boost"] == boost)].dropna(
            subset=["control_gate_mae", "single_boost_gate_mae", "shared_boost_gate_mae"])
        for name in ["single_boost", "shared_boost"]:
            diffs = (sub["control_gate_mae"] - sub[f"{name}_gate_mae"]).to_numpy()
            m, s, n = diffs.mean(), diffs.std(ddof=1), len(diffs)
            se = s / np.sqrt(n) if n > 1 else float("nan")
            t_stat, p_val = stats.ttest_rel(sub["control_gate_mae"], sub[f"{name}_gate_mae"])
            tag2se = ("within noise (2SE)" if abs(m) < 2 * se else
                       ("improves" if m > 0 else "worsens"))
            sig = "significant (p<0.05)" if p_val < 0.05 else "not significant"
            print(f"  boost={boost:>5.2f}  {name:>13}: mean diff {m:+.4f} (n={n}) "
                  f"-> {tag2se:>20} | t-test p={p_val:.4f} ({sig})")

    print("\nDeploy rates (drifted windows passing the gate), by boost:")
    for boost in boosts:
        sub = df[(df["boost"] == boost) & (df["drifted"])]
        for name in ["single_boost", "shared_boost"]:
            dep = sub[f"{name}_deployed"]
            if len(dep):
                print(f"  boost={boost:>5.2f}  {name:>13}: {dep.mean()*100:.0f}% ({dep.sum()}/{len(dep)})")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.file))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "shml_reweight_sweep_per_window.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "shml_reweight_sweep_summary.csv"), index=False)
    print(f"\nSaved per-window sweep results and summary to {out_dir}")


if __name__ == "__main__":
    main()