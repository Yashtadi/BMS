"""
shml_efficiency_test.py — four-policy efficiency comparison for SHML adaptation.

The reweighting experiment (shml_reweight_loop_v2.py) answered a narrow accuracy question and found
diagnosis-conditioned boosting does not beat generic adaptation. This script
reframes the comparison around DEPLOYMENT EFFICIENCY, which is where a positive
result may still exist:

    Frozen          — never adapts (the do-nothing floor)
    Full retrain    — on drift, retrain from scratch on ALL data seen so far
    Generic adapt   — on drift, warm-start on the RECENT window only
    Diagnosis adapt — generic adapt + boost the diagnosed mechanism's feature

For each policy it logs, per (seed, window):
    * gate-window MAE            (accuracy on the next unseen window)
    * whether it adapted         (an "adaptation event")
    * rows of new data used      (training-set size for that adaptation)
    * wall-clock training time   (a compute proxy)

and then reports, per policy, the aggregate the efficiency claim turns on:

    mean gate MAE (± 95% CI)  vs  #adaptation events  vs  data used  vs  compute

THE CLAIM TO TEST
    "Diagnosis-guided adaptation reaches deployment accuracy COMPARABLE to full
     retraining while retraining LESS OFTEN / on LESS DATA / for LESS COMPUTE."

    This script does NOT assume that claim — it measures whether it holds. If
    diagnosis-guided adaptation retrains just as often at the same accuracy, the
    efficiency angle collapses and the honest result stays negative. Read the
    output, don't pre-suppose it.

HONEST SCOPE
    * One dataset, one frozen-model family — same two structural limits as the
      rest of the project. This tests whether the efficiency framing is worth
      pursuing; it is not by itself a conference result.
    * Compute time is wall-clock and machine-dependent; treat it as a relative
      proxy (compare policies within one run), not an absolute cost.
    * "Full retrain from all data so far" is the honest expensive baseline; if
      your deployment reality is different (e.g. a fixed retrain budget), adjust
      MODE_FULL accordingly.

INPUTS  (mirror shml_reweight_loop_v2.py so the same artifacts work)
    --file           orthogonalized feature table (parquet)
    --rec-file       _shml_recommendation.json  (diagnosis: which feature to boost)
    --params-file    _catboost_best_params.json (frozen hyperparameters)
    --auto-start-from-mae  _catboost_mae_per_checkup.csv/.parquet (drift point)
    --error-threshold, --window-size, --baseline-max-cu, --seeds, --boost,
    --exclude-ohmic, --max-iter, --output-dir

    NOTE: --boost here is a SINGLE value (the diagnosis-guided policy uses one
    boost strength), not a sweep — the sweep already lives in the v2 loop.

OUTPUTS  (to --output-dir)
    * shml_efficiency_per_window.csv  — every (seed, window, policy) record.
    * shml_efficiency_summary.csv     — per-policy aggregates (the paper table).
    * console — the per-policy table and a plain-language verdict.

DEPENDENCIES
    pandas, numpy, scipy, catboost, scikit-learn, pyarrow
"""

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit


# --- filter CatBoost's harmless warm-start warning (same as v2 loop) --------
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


# --- feature definitions (identical to shml_reweight_loop_v2.py) ------------
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


def _fit(train_df, params, seed, feature_weights=None, init_model=None, max_iter=250):
    """Train a CatBoost model and return (model, wall_seconds, n_rows)."""
    kw = {}
    if feature_weights is not None:
        kw["feature_weights"] = feature_weights
    model = CatBoostRegressor(
        iterations=max_iter, loss_function="RMSE", eval_metric="MAE",
        monotone_constraints=build_monotone(FEATURES), bootstrap_type="Bernoulli",
        random_seed=seed, verbose=False, **params, **kw)
    fit_kw = {"init_model": init_model} if init_model is not None else {}
    t0 = time.perf_counter()
    with quiet_shrinkage_warning():
        model.fit(Pool(train_df[FEATURES], label=train_df["soh_cap"]),
                  verbose=False, **fit_kw)
    return model, time.perf_counter() - t0, len(train_df)


def train_frozen(data, baseline_max_cu, params, seed, max_iter=250):
    early = data[data["cu_index"] <= baseline_max_cu]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, _ = next(gss.split(early, groups=early["cell_id"]))
    model, _, _ = _fit(early.iloc[tr_idx], params, seed, max_iter=max_iter)
    return model


def weights_single(dominant_feature, boost):
    return [boost if f == dominant_feature else 1.0 for f in FEATURES]


def weights_shared(contrib_pct, boost, exclude_ohmic=False):
    """
    Contribution-weighted boost: every feature is boosted in proportion to its
    diagnosed contribution share, matching shml_reweight_loop_v2.py.
        w_f = 1 + (boost - 1) * share_f
    A feature with 100% share gets the full boost; smaller contributors get
    proportionally less. When exclude_ohmic is set, the ohmic feature is left
    unboosted (weight 1.0) so only genuine aging mechanisms are emphasised.
    """
    w = []
    for f in FEATURES:
        share = 0.0
        for cause, feat in CAUSE_TO_FEATURE.items():
            if feat != f or cause not in contrib_pct:
                continue
            if exclude_ohmic and cause == OHMIC:
                continue
            share = max(share, contrib_pct[cause] / 100.0)
        w.append(1.0 + (boost - 1.0) * share if share > 0 else 1.0)
    return w


def first_drift_cu(mae_file, threshold, baseline_max_cu, mae_col="mae_tuned"):
    df = pd.read_csv(mae_file) if str(mae_file).endswith(".csv") else pd.read_parquet(mae_file)
    if mae_col not in df.columns:
        cand = [c for c in df.columns if c.startswith("mae")]
        if not cand:
            raise SystemExit(f"no MAE column in {mae_file}")
        mae_col = cand[0]
    later = df[df["cu_index"] > baseline_max_cu].sort_values("cu_index")
    crossed = later[later[mae_col] > threshold]
    return None if crossed.empty else int(crossed.iloc[0]["cu_index"])


def make_window_starts(drift_cu, window_size, max_cu):
    starts, s = [], drift_cu
    while s + 2 * window_size - 1 <= max_cu:
        starts.append(s)
        s += window_size
    return starts


def pick_dominant(rec, exclude_ohmic):
    contrib = rec.get("contribution_pct_full", {})
    mech = [f for f in ["R_ohm_1arc", "R_1arc", "n_1arc", "sigma_1arc"] if f in contrib]
    if not exclude_ohmic or OHMIC not in mech:
        dom = rec.get("overall_dominant", "R_ohm_1arc")
        return dom, CAUSE_TO_FEATURE.get(dom, DELTAZ_COL)
    aging = [f for f in mech if f != OHMIC]
    dom = max(aging, key=lambda f: contrib.get(f, 0.0))
    return dom, CAUSE_TO_FEATURE.get(dom, DELTAZ_COL)


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(se * stats.t.ppf(0.975, n - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--rec-file", required=True)
    ap.add_argument("--params-file", default=None)
    ap.add_argument("--baseline-max-cu", type=int, default=5)
    ap.add_argument("--error-threshold", type=float, default=3.6)
    ap.add_argument("--window-starts", default="10,14,18,22")
    ap.add_argument("--auto-start-from-mae", default=None)
    ap.add_argument("--window-size", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--boost", type=float, default=1.5,
                    help="single boost strength for the diagnosis-guided policy")
    ap.add_argument("--exclude-ohmic", action="store_true")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    data = pd.read_parquet(args.file)
    data = data[data["fit_succeeded"]].dropna(subset=FEATURES + ["soh_cap"]).copy()

    rec = json.load(open(args.rec_file))
    if args.params_file:
        bp = json.load(open(args.params_file))
        params = bp.get("best_params", bp)
    else:
        params = rec["current_params"]
    dominant_cause, dominant_feature = pick_dominant(rec, args.exclude_ohmic)
    print(f"Frozen params       : {params}")
    print(f"exclude-ohmic       : {args.exclude_ohmic}")
    print(f"Diagnosis boost     : {dominant_cause} -> '{dominant_feature}' x {args.boost}\n")

    max_cu = int(data["cu_index"].max())
    if args.auto_start_from_mae:
        drift_cu = first_drift_cu(args.auto_start_from_mae, args.error_threshold,
                                  args.baseline_max_cu)
        if drift_cu is None:
            raise SystemExit(f"MAE never exceeds {args.error_threshold}; nothing to adapt.")
        window_starts = make_window_starts(drift_cu, args.window_size, max_cu)
        print(f"AUTO-START: drift at cu={drift_cu} -> window starts {window_starts}\n")
    else:
        window_starts = [int(x) for x in args.window_starts.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    # policies: frozen is the floor; the rest adapt only on drift.
    # diagnosis_single boosts the one dominant feature; diagnosis_shared boosts
    # all features by their contribution share.
    POLICIES = ["frozen", "full_retrain", "generic_adapt",
                "diagnosis_single", "diagnosis_shared"]

    records = []
    for seed in seeds:
        frozen = train_frozen(data, args.baseline_max_cu, params, seed, max_iter=args.max_iter)
        w_single = weights_single(dominant_feature, args.boost)
        w_shared = weights_shared(rec.get("contribution_pct_full", {}),
                                  args.boost, exclude_ohmic=args.exclude_ohmic)

        for start in window_starts:
            adapt_cus = range(start, start + args.window_size)
            gate_cus = range(start + args.window_size, start + 2 * args.window_size)
            adapt_win = data[data["cu_index"].isin(adapt_cus)]
            gate_win = data[data["cu_index"].isin(gate_cus)]
            if len(adapt_win) < 30 or len(gate_win) < 30:
                continue

            frozen_gate = mae(frozen.predict(gate_win[FEATURES]), gate_win["soh_cap"])
            drifted = mae(frozen.predict(adapt_win[FEATURES]),
                          adapt_win["soh_cap"]) > args.error_threshold

            # data seen so far = everything up to and including the adapt window
            seen_so_far = data[data["cu_index"] < start + args.window_size]

            for policy in POLICIES:
                rec_row = {"seed": seed, "window_start": start, "policy": policy,
                           "drifted": drifted}
                if policy == "frozen":
                    # never adapts
                    rec_row.update(gate_mae=frozen_gate, adapted=False,
                                   rows_used=0, train_seconds=0.0)
                    records.append(rec_row)
                    continue

                if not drifted:
                    # no drift -> no adaptation event for any adaptive policy
                    rec_row.update(gate_mae=frozen_gate, adapted=False,
                                   rows_used=0, train_seconds=0.0)
                    records.append(rec_row)
                    continue

                # drifted -> this policy performs an adaptation event
                if policy == "full_retrain":
                    m, secs, nrows = _fit(seen_so_far, params, seed,
                                          init_model=None, max_iter=args.max_iter)
                elif policy == "generic_adapt":
                    m, secs, nrows = _fit(adapt_win, params, seed,
                                          init_model=frozen, max_iter=args.max_iter)
                elif policy == "diagnosis_single":
                    m, secs, nrows = _fit(adapt_win, params, seed,
                                          feature_weights=w_single,
                                          init_model=frozen, max_iter=args.max_iter)
                else:  # diagnosis_shared
                    m, secs, nrows = _fit(adapt_win, params, seed,
                                          feature_weights=w_shared,
                                          init_model=frozen, max_iter=args.max_iter)

                cand = mae(m.predict(gate_win[FEATURES]), gate_win["soh_cap"])
                # gate: keep the adapted model only if it beats frozen
                deploy = cand < frozen_gate
                rec_row.update(gate_mae=cand if deploy else frozen_gate,
                               adapted=True, rows_used=nrows, train_seconds=secs)
                records.append(rec_row)

        print(f"[seed {seed + 1}/{len(seeds)}] done over {len(window_starts)} windows")

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No valid windows — check --window-size / starts.")

    # ---- per-policy aggregates: the efficiency comparison ------------------
    print("\n" + "=" * 92)
    print(f"FIVE-POLICY EFFICIENCY COMPARISON  ({len(seeds)} seeds x "
          f"{df['window_start'].nunique()} windows)")
    print("=" * 92)
    print(f"{'policy':>18}  {'gate MAE':>10}  {'95% CI':>8}  "
          f"{'adapt events':>12}  {'rows/adapt':>11}  {'sec/adapt':>10}  {'total sec':>10}")

    summary = []
    for policy in POLICIES:
        sub = df[df["policy"] == policy]
        m, h = mean_ci95(sub["gate_mae"])
        events = int(sub["adapted"].sum())
        adapted = sub[sub["adapted"]]
        rows_per = adapted["rows_used"].mean() if len(adapted) else 0.0
        sec_per = adapted["train_seconds"].mean() if len(adapted) else 0.0
        total_sec = sub["train_seconds"].sum()
        print(f"{policy:>18}  {m:>10.3f}  {h:>8.3f}  {events:>12d}  "
              f"{rows_per:>11.0f}  {sec_per:>10.3f}  {total_sec:>10.2f}")
        summary.append({"policy": policy, "gate_mae": m, "gate_mae_ci95": h,
                        "adapt_events": events, "rows_per_adapt": rows_per,
                        "sec_per_adapt": sec_per, "total_train_sec": total_sec})

    sdf = pd.DataFrame(summary).set_index("policy")

    # ---- the efficiency verdict -------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    full = sdf.loc["full_retrain"]
    gen = sdf.loc["generic_adapt"]

    # Efficiency axis: generic warm-start adaptation (the cheapest adaptive
    # policy) vs full retrain. "As good or better" is what the claim needs.
    acc_gap = gen["gate_mae"] - full["gate_mae"]              # negative = adapt better
    ci_sum = gen["gate_mae_ci95"] + full["gate_mae_ci95"]
    as_good = acc_gap <= ci_sum
    strictly_better = acc_gap < -ci_sum
    cheaper_data = gen["rows_per_adapt"] < full["rows_per_adapt"]
    cheaper_time = gen["total_train_sec"] < full["total_train_sec"]

    if strictly_better:
        acc_label = "BETTER than full retrain"
    elif as_good:
        acc_label = "comparable to full retrain (CIs overlap)"
    else:
        acc_label = "WORSE than full retrain"

    print("  Warm-start adaptation vs FULL retrain — the efficiency axis:")
    print(f"    accuracy gap = {acc_gap:+.3f} MAE  ({acc_label})")
    print(f"    data per adapt: {gen['rows_per_adapt']:.0f} vs {full['rows_per_adapt']:.0f} rows  "
          f"({'less' if cheaper_data else 'not less'})")
    print(f"    total compute:  {gen['total_train_sec']:.2f}s vs {full['total_train_sec']:.2f}s  "
          f"({'less' if cheaper_time else 'not less'})")
    print()

    # Does either diagnosis policy add value over generic adaptation?
    print("  Diagnosis-guided vs GENERIC adaptation — does the diagnosis add value?")
    any_diag_helps = False
    for dpol, label in [("diagnosis_single", "single boost"),
                        ("diagnosis_shared", "shared boost")]:
        d = sdf.loc[dpol]
        gap = d["gate_mae"] - gen["gate_mae"]                 # negative = diag better
        gci = d["gate_mae_ci95"] + gen["gate_mae_ci95"]
        helps = gap < -gci
        any_diag_helps = any_diag_helps or helps
        verdict = "significantly better" if helps else "no meaningful difference"
        print(f"    {label:>13}: {gap:+.4f} MAE vs generic  ({verdict})")
    print()

    # Two independent claims, reported separately and honestly.
    if as_good and (cheaper_data or cheaper_time):
        print("  => EFFICIENCY CLAIM PLAUSIBLE: warm-start adaptation reaches")
        print(f"     accuracy {acc_label.split(' (')[0]} while using less data/compute than")
        print("     full retraining. This is the positive result worth a paper.")
    else:
        print("  => EFFICIENCY CLAIM NOT SUPPORTED here: adaptation does not reach")
        print("     full-retrain accuracy at a real cost saving in this run.")
    print()
    if any_diag_helps:
        print("  => A DIAGNOSIS policy adds value over generic adaptation. Investigate further.")
    else:
        print("  => NEITHER diagnosis policy (single or shared) adds value over generic")
        print("     adaptation — the efficiency win belongs to warm-start adaptation")
        print("     itself, not to the diagnosis. Report this honestly.")
    print("\n  (One dataset, one frozen-model family — this probes whether the")
    print("   efficiency framing is worth pursuing, not a standalone conference result.)")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.file))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "shml_efficiency_per_window.csv"), index=False)
    sdf.reset_index().to_csv(os.path.join(out_dir, "shml_efficiency_summary.csv"), index=False)
    print(f"\nSaved per-window records and per-policy summary to {out_dir}")


if __name__ == "__main__":
    main()