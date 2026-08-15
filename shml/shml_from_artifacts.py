"""
shml_from_artifacts.py — build the SHML recommendation from EXISTING artifacts
only. Trains nothing. Reads:

  --pred-file     per-row predictions parquet (has abserr_tuned, cu_index, keys)
                  e.g. ..._catboost_predictions.parquet
  --ecm-file      ECM mechanism values per spectrum
                  e.g. ecm_features_v2.parquet
  --params-file   the frozen model's CatBoost hyperparameters
                  e.g. ..._catboost_best_params.json

and produces:

  1. MAE DRIFT across later check-ups (read straight from the predictions).
  2. ECM CONTRIBUTION per check-up: how far each mechanism has deviated from
     its early-life baseline, weighted by how strongly that deviation tracks
     the model's error (deviation x |Spearman(deviation, error)|).
  3. A HEURISTIC hyperparameter RECOMMENDATION: given the dominant ECM
     contribution across the drifting check-ups, nudge the frozen model's
     current hyperparameters using fixed, physically-motivated rules.

WHAT THIS IS / ISN'T:
  This trains nothing and validates nothing. The recommended hyperparameters
  are a principled, drift-informed SUGGESTION -- the direction an adaptive
  (self-healing) scheme should try -- not a fix with a measured later-life
  MAE. Producing a validated later-life MAE would require training a model,
  which this script deliberately does not do. (This project's own earlier
  results also show that params chosen from early-life data alone do not
  reliably improve later life -- another reason the recommendation is framed
  as a future-work direction, not a guaranteed improvement.)

Usage:
    python shml_from_artifacts.py \
        --pred-file  results/stage2_features_filtered_orthogonalized_catboost_predictions.parquet \
        --ecm-file   data/interim/ecm_features_v2.parquet \
        --params-file results/stage2_features_filtered_orthogonalized_catboost_best_params.json \
        --abserr-col abserr_tuned --error-threshold 3.6 --baseline-max-cu 5 \
        --output-dir results
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAW_MECH = ["R_ohm_1arc", "R_1arc", "n_1arc", "sigma_1arc"]
DELTAZ = "deltaz_integrated_abs"
MEANING = {"R_ohm_1arc": "ohmic/resistive", "R_1arc": "interfacial",
           "sigma_1arc": "diffusion", "n_1arc": "spectral",
           "deltaz_integrated_abs": "impedance-change (deltaZ)"}
KEYS = ["cell_id", "cu_index", "soc_nom", "is_rt"]

# Heuristic rules: dominant drifting mechanism -> hyperparameter nudges.
# Fixed in advance, each with a physical rationale. No later-life error is used
# to choose these -- they are drift-informed suggestions, not validated fixes.
HEURISTIC_RULES = {
    "R_ohm_1arc": {
        "adjust": {"l2_leaf_reg": ("mul", 2.0), "depth": ("add", -1)},
        "rationale": "ohmic/resistive drift is smooth and monotonic; a stiffer, "
                     "shallower model (higher l2_leaf_reg, lower depth) extrapolates "
                     "it more safely into later life"},
    "R_1arc": {
        "adjust": {"min_data_in_leaf": ("mul", 3.0)},
        "rationale": "interfacial drift is high-variance; coarser leaves "
                     "(higher min_data_in_leaf) resist overfitting its noise"},
    "n_1arc": {
        "adjust": {"learning_rate": ("mul", 0.5)},
        "rationale": "spectral-shape shifts are subtle; a slower learning rate "
                     "(with more rounds) tracks them without overshooting"},
    "sigma_1arc": {
        "adjust": {"subsample": ("mul", 1.15)},
        "rationale": "diffusion drift is rare in the population; broader row "
                     "sampling (higher subsample) keeps its signal represented"},
}


def mae(x):
    return float(np.mean(np.abs(x)))


def apply_heuristic(params, dom):
    p = dict(params)
    rule = HEURISTIC_RULES.get(dom)
    if not rule:
        return p, "no rule for dominant mechanism; keep current params"
    for k, (op, v) in rule["adjust"].items():
        cur = p.get(k)
        if cur is None:
            continue
        if k == "depth":
            p[k] = max(2, int(cur) + int(v))
        elif k == "min_data_in_leaf":
            p[k] = int(cur * v)
        elif op == "mul":
            p[k] = min(1.0, float(cur) * v) if k == "subsample" else float(cur) * v
        elif op == "add":
            p[k] = float(cur) + v
    return p, rule["rationale"]


def scaled_adjust(base_value, param, op, full_v, share):
    """Apply a rule adjustment SCALED by the mechanism's contribution share.
    share in [0,1]: at share=1 the full rule fires; at share=0 nothing changes.
    Multiplicative rules interpolate toward the factor; additive toward the delta."""
    if base_value is None:
        return None
    if param == "depth":
        # additive integer: round(share * full_v)
        return max(2, int(base_value) + int(round(full_v * share)))
    if param == "min_data_in_leaf":
        factor = 1.0 + (full_v - 1.0) * share   # interpolate factor toward full_v
        return int(base_value * factor)
    if op == "mul":
        factor = 1.0 + (full_v - 1.0) * share
        val = float(base_value) * factor
        return min(1.0, val) if param == "subsample" else val
    if op == "add":
        val = float(base_value) + full_v * share
        return val
    return base_value


def recommend_combined(params, agg, mech_present, deltaz_present):
    """Build ONE param set from contribution-weighted adjustments.
    Aging mechanisms (all ECM except ohmic, weighted by share within the aging
    view) each nudge their knob; ohmic (R_ohm_1arc) is handled on its OWN track.
    Returns (final_params, provenance) where provenance[param] = list of
    (source_mechanism, old, new) so each change is attributable to aging vs ohmic."""
    OHMIC = "R_ohm_1arc"
    aging_mechs = [f for f in mech_present if f != OHMIC]  # R_1arc, n_1arc, sigma_1arc
    aging_total = sum(agg[f] for f in aging_mechs) or 1.0

    final = dict(params)
    provenance = {}

    # --- aging-mechanism track: each mechanism scaled by its share of aging contribution ---
    for mech in aging_mechs:
        share = agg[mech] / aging_total
        rule = HEURISTIC_RULES.get(mech)
        if not rule:
            continue
        for k, (op, v) in rule["adjust"].items():
            old = final.get(k)
            new = scaled_adjust(old, k, op, v, share)
            if new is None or new == old:
                continue
            final[k] = new
            provenance.setdefault(k, []).append(
                (f"aging:{mech} ({MEANING[mech]}, share={share:.2f})", old, new))

    # --- ohmic track: R_ohm handled separately, scaled by its OWN share of the FULL total ---
    if OHMIC in mech_present:
        full_total = sum(agg.values()) or 1.0
        ohmic_share = agg[OHMIC] / full_total
        rule = HEURISTIC_RULES.get(OHMIC)
        if rule:
            for k, (op, v) in rule["adjust"].items():
                old = final.get(k)
                new = scaled_adjust(old, k, op, v, ohmic_share)
                if new is None or new == old:
                    continue
                final[k] = new
                provenance.setdefault(k, []).append(
                    (f"ohmic:R_ohm_1arc (separate track, share={ohmic_share:.2f})", old, new))

    return final, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--ecm-file", required=True)
    ap.add_argument("--params-file", required=True)
    ap.add_argument("--abserr-col", default="abserr_tuned")
    ap.add_argument("--error-threshold", type=float, default=3.6)
    ap.add_argument("--baseline-max-cu", type=int, default=5)
    ap.add_argument("--exclude-ohmic", action="store_true",
                    help="also produce an 'aging-mechanism' view with ohmic resistance "
                         "(R_ohm_1arc) factored out, and base the recommendation on the aging "
                         "mechanisms. Rationale: ohmic/series-resistance growth often reflects "
                         "contact/electrolyte/cabling effects rather than electrode-level aging "
                         "(SEI growth, active-material loss), so excluding it surfaces the "
                         "degradation modes that describe HOW the cell ages. The full view "
                         "(with R_ohm) is always shown too -- R_ohm is set aside deliberately "
                         "and transparently, not hidden.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    preds = pd.read_parquet(args.pred_file)
    ecm = pd.read_parquet(args.ecm_file)
    params_json = json.load(open(args.params_file))
    params = params_json.get("best_params", params_json)

    if args.abserr_col not in preds.columns:
        avail = [c for c in preds.columns if c.startswith("abserr")]
        raise SystemExit(f"'{args.abserr_col}' not in pred-file. Available: {avail}")

    mech_present = [c for c in RAW_MECH if c in ecm.columns]
    if not mech_present:
        raise SystemExit(f"none of {RAW_MECH} found in --ecm-file. Columns: {list(ecm.columns)[:20]}")
    # deltaZ is the model's dominant engineered input -- include its drift in the
    # contribution breakdown alongside the raw ECM mechanisms (reported, but it
    # has no hyperparameter rule of its own since it's the monotone-constrained
    # feature, not a tunable knob).
    deltaz_present = DELTAZ in ecm.columns
    all_factors = mech_present + ([DELTAZ] if deltaz_present else [])
    key_join = [k for k in KEYS if k in preds.columns and k in ecm.columns]
    if "cell_id" not in key_join or "cu_index" not in key_join:
        raise SystemExit(f"need cell_id + cu_index in both files to join; shared keys: {key_join}")

    df = preds.merge(ecm[key_join + all_factors].drop_duplicates(key_join),
                     on=key_join, how="left")
    print(f"Joined predictions + ECM on {key_join}: {len(df)} rows")
    print(f"Drift factors: {all_factors}"
          + ("" if deltaz_present else "  (deltaZ not found in ecm-file, skipped)") + "\n")

    # ---------- 1. MAE drift ----------
    base = df[df["cu_index"] <= args.baseline_max_cu]
    bmean = base[all_factors].mean()
    bstd = base[all_factors].std().replace(0, np.nan)

    print("=" * 70)
    print(f"1. MAE DRIFT across later check-ups (threshold = {args.error_threshold})")
    print("=" * 70)
    print(f"{'cu':>4} {'MAE':>8} {'over?':>6}   dominant drift contribution")
    per_cu = []
    for cu in range(args.baseline_max_cu + 1, int(df["cu_index"].max()) + 1):
        g = df[df["cu_index"] == cu]
        if len(g) < 20:
            continue
        m = mae(g[args.abserr_col])
        upto = df[(df["cu_index"] > args.baseline_max_cu) & (df["cu_index"] <= cu)]
        contrib = {}
        for f in all_factors:
            dev = (g[f] - bmean[f]).abs() / bstd[f]
            du = (upto[f] - bmean[f]).abs() / bstd[f]
            rho, _ = spearmanr(du, upto[args.abserr_col], nan_policy="omit")
            rho = 0.0 if np.isnan(rho) else rho
            contrib[f] = float(np.nanmean(dev) * abs(rho))
        dom = max(contrib, key=contrib.get)
        over = m > args.error_threshold
        per_cu.append({"cu": int(cu), "mae": m, "over": bool(over),
                       "dominant": dom, "contrib": contrib})
        print(f"{cu:>4} {m:>8.3f} {'YES' if over else 'no':>6}   "
              f"{dom} ({MEANING[dom]}) [{contrib[dom]:.3f}]")

    # ---------- 2. contribution summary ----------
    over_cus = [r for r in per_cu if r["over"]]
    agg = {f: 0.0 for f in all_factors}
    for r in (over_cus or per_cu):
        for f in all_factors:
            agg[f] += r["contrib"][f]
    total = sum(agg.values()) or 1.0
    # overall dominant across ALL factors (for reporting)
    overall_dom = max(agg, key=agg.get)
    # dominant ECM MECHANISM only (for the hyperparameter rule -- deltaZ has no knob rule)
    ecm_agg = {f: agg[f] for f in mech_present}
    ecm_dom = max(ecm_agg, key=ecm_agg.get)

    print("\n" + "=" * 70)
    print("2. DRIFT CONTRIBUTION to MAE (summed over drifting check-ups)")
    print("=" * 70)
    print("  [FULL VIEW — all factors]")
    for f in sorted(all_factors, key=lambda x: -agg[x]):
        star = "  <- deltaZ (model's main feature)" if f == DELTAZ else ""
        print(f"    {f:22s} ({MEANING[f]:26s}): {100*agg[f]/total:5.1f}%{star}")
    print(f"\n  Overall dominant driver     : {overall_dom} ({MEANING[overall_dom]})")
    print(f"  Dominant ECM mechanism      : {ecm_dom} ({MEANING[ecm_dom]})")

    OHMIC = "R_ohm_1arc"
    aging_view = None
    aging_dom = ecm_dom  # default: recommendation uses full-view ECM dominant
    if args.exclude_ohmic and OHMIC in mech_present:
        # aging-mechanism view: renormalize contributions with ohmic factored out.
        # deltaZ is imaginary-part-only (reactive/diffusive), so it stays in the
        # aging view; only the real-part ohmic resistance is set aside.
        aging_factors = [f for f in all_factors if f != OHMIC]
        aging_total = sum(agg[f] for f in aging_factors) or 1.0
        aging_view = {f: 100 * agg[f] / aging_total for f in aging_factors}
        # dominant AGING MECHANISM for the rule = top ECM mechanism excluding ohmic
        aging_ecm = [f for f in mech_present if f != OHMIC]
        aging_dom = max(aging_ecm, key=lambda f: agg[f])
        print("\n  [AGING-MECHANISM VIEW — ohmic resistance (R_ohm_1arc) factored out]")
        print("  Rationale: ohmic/series-resistance growth often reflects contact, electrolyte,")
        print("             or cabling effects rather than electrode-level aging (SEI growth,")
        print("             active-material loss). Factoring it out surfaces the degradation")
        print("             modes that describe HOW the cell ages. R_ohm is set aside")
        print("             deliberately and transparently; the full view above still stands.")
        for f in sorted(aging_factors, key=lambda x: -aging_view[x]):
            star = "  <- deltaZ (imaginary-part, reactive/diffusive)" if f == DELTAZ else ""
            print(f"    {f:22s} ({MEANING[f]:26s}): {aging_view[f]:5.1f}%{star}")
        print(f"\n  Dominant AGING mechanism    : {aging_dom} ({MEANING[aging_dom]})  "
              f"<- used for the recommendation")

    # ---------- 3. Combined contribution-weighted recommendation ----------
    # Individual aging-mechanism contributions each nudge their knob (scaled by
    # share within the aging view), and R_ohm is handled on its own separate
    # track. All merge into ONE param set, with every change attributed to its
    # source (aging vs ohmic).
    rec_params, provenance = recommend_combined(params, agg, mech_present, deltaz_present)
    print("\n" + "=" * 70)
    print("3. SHML HYPERPARAMETER RECOMMENDATION (contribution-weighted, drift-informed)")
    print("=" * 70)
    print(f"  Current (frozen) params : {json.dumps(params)}")
    print("\n  Individual aging-mechanism contributions driving the adjustment")
    print("  (deltaZ is the aging SIGNAL; the ECM params below diagnose its cause):")
    OHMIC = "R_ohm_1arc"
    aging_mechs = [f for f in mech_present if f != OHMIC]
    aging_tot = sum(agg[f] for f in aging_mechs) or 1.0
    for f in sorted(aging_mechs, key=lambda x: -agg[x]):
        rule = HEURISTIC_RULES.get(f, {})
        knobs = ", ".join(rule.get("adjust", {}).keys()) or "-"
        print(f"    {f:12s} ({MEANING[f]:26s}): share {100*agg[f]/aging_tot:5.1f}%  -> knob: {knobs}")
    print(f"\n  Ohmic (separate track): R_ohm_1arc share "
          f"{100*agg[OHMIC]/total:.1f}% of full -> knob: l2_leaf_reg, depth")

    print("\n  Recommended params (one merged set):")
    print(f"    {json.dumps(rec_params)}")
    print("\n  Change provenance (which source moved each knob):")
    if not provenance:
        print("    (no changes — contributions too small to move any knob)")
    for k, sources in provenance.items():
        for src, old, new in sources:
            ov = f"{old:.4g}" if isinstance(old, float) else str(old)
            nv = f"{new:.4g}" if isinstance(new, float) else str(new)
            print(f"    {k:16s}: {ov} -> {nv}   [{src}]")
    print("\n  NOTE: this is a drift-informed SUGGESTION for a future adaptive scheme,")
    print("        not a validated fix. No model was trained or evaluated here.")

    # ---------- save ----------
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(args.pred_file).replace(".parquet", "").replace("_predictions", "")

    rec = {"error_threshold": args.error_threshold, "abserr_col": args.abserr_col,
           "recommendation_method": "contribution_weighted_combined",
           "overall_dominant": overall_dom, "overall_dominant_meaning": MEANING[overall_dom],
           "contribution_pct_full": {f: 100*agg[f]/total for f in all_factors},
           "aging_view_excludes_ohmic": bool(args.exclude_ohmic),
           "aging_contribution_pct": aging_view,
           "current_params": params, "recommended_params": rec_params,
           "change_provenance": {k: [{"source": s, "old": o, "new": n} for s, o, n in v]
                                 for k, v in provenance.items()},
           "note": "contribution-weighted heuristic suggestion; no training/validation performed. "
                   "Aging mechanisms (R_1arc/n_1arc/sigma_1arc) each nudge their knob scaled by "
                   "contribution share; R_ohm_1arc (real-part ohmic) is handled on a separate "
                   "track. deltaZ (imaginary-part) is the aging signal, diagnosed via the ECM "
                   "params, not itself a tunable knob."}
    rec_path = os.path.join(out_dir, f"{base_name}_shml_recommendation.json")
    json.dump(rec, open(rec_path, "w"), indent=2)

    per_cu_df = pd.DataFrame([{"cu": r["cu"], "mae": r["mae"], "over": r["over"],
                               "dominant": r["dominant"],
                               **{f"contrib_{k}": v for k, v in r["contrib"].items()}}
                              for r in per_cu])
    csv_path = os.path.join(out_dir, f"{base_name}_shml_per_checkup.csv")
    per_cu_df.to_csv(csv_path, index=False)

    print(f"\nSaved recommendation: {rec_path}")
    print(f"Saved per-check-up drift + contribution: {csv_path}")


if __name__ == "__main__":
    main()