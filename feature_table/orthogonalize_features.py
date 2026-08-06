"""
orthogonalize_features.py — VIF check + orthogonalize ECM parameters
against deltaZ, WITHOUT fitting any f_physics candidate curves.

This replaces fit_f_physics.py for pipelines that don't use f_physics at
all (e.g. fit_monotonic_gbm.py) — fit_f_physics.py's VIF/orthogonalization
step (Part 2 of physics_residual_model.md) is still needed (the GBM's ECM
features are still collinear with deltaZ and need the same residualization
treatment), but its Part 3 (comparing linear/exponential/power curve forms)
is dead weight if nothing downstream ever fits f_physics.

FIX vs. fit_f_physics.py: orthogonalizes against deltaz_integrated_abs, not
deltaz_mean_all. fit_f_physics.py orthogonalized against deltaz_mean_all
because that was the original default target (physics_residual_model.md
Part 2's recommendation), but fit_monotonic_gbm.py's actual primary deltaZ
feature is deltaz_integrated_abs -- a DIFFERENT column. Orthogonalizing
against the wrong column would leave R_ohm_orth/R_1arc_orth/sigma_1arc_orth
still partly collinear with the deltaZ feature the GBM actually uses. This
script orthogonalizes against whichever column --deltaz-col names, matching
fit_monotonic_gbm.py's DELTAZ_COL by default.

Usage:
    python orthogonalize_features.py --file data/interim/stage2_features_filtered.parquet
"""

import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ECM_PARAM_COLS = ["R_ohm", "R_1arc", "sigma_1arc"]


def run_vif_check(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """VIF > 5 is commonly treated as moderate collinearity, > 10 as severe."""
    clean = df[columns].dropna()
    X = clean.to_numpy()
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": columns, "VIF": vifs}).sort_values("VIF", ascending=False)


def orthogonalize(df: pd.DataFrame, target_col: str, against_col: str) -> pd.Series:
    """
    Regress target_col on against_col (simple linear regression) and return
    the RESIDUAL -- the part of target_col that against_col doesn't already
    explain. Rows with NaN in either column keep NaN in the output.
    """
    mask = df[target_col].notna() & df[against_col].notna()
    x = df.loc[mask, against_col].to_numpy()
    y = df.loc[mask, target_col].to_numpy()

    if mask.sum() < 3:
        return pd.Series(np.nan, index=df.index)

    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)

    out = pd.Series(np.nan, index=df.index)
    out.loc[mask] = residual
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/interim/stage2_features_filtered.parquet")
    parser.add_argument("--deltaz-col", default="deltaz_integrated_abs",
                         help="orthogonalize ECM params against THIS column -- should match whichever "
                              "deltaZ feature your downstream model (e.g. fit_monotonic_gbm.py's "
                              "DELTAZ_COL) actually treats as primary")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df = df[df["fit_succeeded"]].copy()

    print("=" * 60)
    print("STEP 1: VIF check")
    print("=" * 60)
    vif_cols = [args.deltaz_col] + ECM_PARAM_COLS
    print(run_vif_check(df, vif_cols).to_string(index=False))
    print()

    print("=" * 60)
    print(f"STEP 2: Orthogonalize ECM params against {args.deltaz_col}")
    print("=" * 60)
    for col in ECM_PARAM_COLS:
        df[f"{col}_orth"] = orthogonalize(df, col, args.deltaz_col)
    print(f"Added: {[c + '_orth' for c in ECM_PARAM_COLS]}")

    print()
    print("=== VIF after orthogonalization (should be ~1 for each _orth column vs. deltaZ) ===")
    orth_cols = [f"{c}_orth" for c in ECM_PARAM_COLS]
    print(run_vif_check(df, [args.deltaz_col] + orth_cols).to_string(index=False))

    out_path = args.file.replace(".parquet", "_orthogonalized.parquet")
    df.to_parquet(out_path, index=False)
    print()
    print(f"Saved: {out_path}  {df.shape}")


if __name__ == "__main__":
    main()