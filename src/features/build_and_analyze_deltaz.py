"""
build_and_analyze_deltaz.py

End-to-end script: load eis_labeled.parquet, build the deltaZ tables via
deltaz.py, and run the diagnostic checks that should be reviewed before
using the tables for modeling (Stage 2 README Part 3 onward).

Usage:
    python build_and_analyze_deltaz.py [path/to/eis_labeled.parquet]

Requires deltaz.py to be in the same directory (or on PYTHONPATH).
"""

import sys
import pandas as pd

from deltaz import build_deltaz_table


def diagnose_baseline_validity(df: pd.DataFrame) -> None:
    """
    Check 1: does every (cell_id, soc_nom, is_rt) group have a usable
    cu_index == 0 baseline, and how many of those cu_index == 0 points are
    actually flagged invalid (which triggers the fallback-to-next-checkup
    logic in build_baseline_table)?
    """
    groups = df.groupby(["cell_id", "soc_nom", "is_rt"])
    min_cu = groups["cu_index"].min()

    print("=== Baseline validity check ===")
    print("min cu_index across (cell, soc, pass) groups:")
    print(min_cu.value_counts())

    cu0 = df[df["cu_index"] == 0]
    n_invalid = (cu0["valid"] == 0).sum()
    print(f"\ncu_index==0 rows: {len(cu0):,} | invalid: {n_invalid:,} "
          f"({n_invalid / len(cu0):.1%})")
    print("(these trigger the fallback to the next valid check-up as baseline)\n")


def diagnose_zero_valid_spectra(deltaz_features: pd.DataFrame) -> pd.DataFrame:
    """
    Check 2: find spectra with zero valid frequency points at all — these
    will have NaN in every deltaZ summary column and need an explicit
    drop/impute decision before modeling.
    """
    zero_valid = deltaz_features[deltaz_features["n_valid_points"] == 0]

    print("=== Zero-valid-point spectra check ===")
    print(f"Total: {len(zero_valid)} spectra with 0 valid frequency points "
          f"out of {len(deltaz_features):,}")
    if len(zero_valid):
        print("\nBreakdown by (soc_nom, is_rt):")
        print(zero_valid.groupby(["soc_nom", "is_rt"]).size())
        print(f"\nUnique cells affected: {zero_valid['cell_id'].nunique()}")
    print()

    return zero_valid


def diagnose_soh_correlation(
    df: pd.DataFrame, deltaz_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Check 3: quick raw-correlation sanity check of each candidate deltaZ
    summary statistic against soh_cap. NOT a substitute for the actual
    curve-fit comparison in Stage 2 README Part 3 — just a fast check that
    the signal is present before investing in the full fit.
    """
    soh = (
        df.groupby(["cell_id", "cu_index", "soc_nom", "is_rt"])["soh_cap"]
        .first()
        .reset_index()
    )
    merged = deltaz_features.merge(
        soh, on=["cell_id", "cu_index", "soc_nom", "is_rt"], how="left"
    )

    clean = merged[merged["n_valid_points"] > 0]

    print("=== soh_cap correlation check (raw, not curve-fit) ===")
    for col in ["deltaz_mean_all", "deltaz_mean_lowfreq", "deltaz_integrated_abs"]:
        r = clean[col].corr(clean["soh_cap"])
        n = clean[col].notna().sum()
        print(f"  {col:24s} r = {r:+.3f}  (n={n:,})")
    print()

    return merged


def main(in_path: str) -> None:
    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows, {df['cell_id'].nunique()} cells\n")

    diagnose_baseline_validity(df)

    deltaz_long, deltaz_features = build_deltaz_table(df)
    print(f"Built deltaz_long: {deltaz_long.shape}")
    print(f"Built deltaz_features: {deltaz_features.shape}\n")

    print("n_valid_points distribution:")
    print(deltaz_features["n_valid_points"].describe())
    print()

    diagnose_zero_valid_spectra(deltaz_features)
    merged = diagnose_soh_correlation(df, deltaz_features)

    from pathlib import Path

    out_dir = Path(in_path).resolve().parent
    deltaz_long.to_parquet(out_dir / "deltaz_long.parquet")
    deltaz_features.to_parquet(out_dir / "deltaz_features.parquet")
    print(f"Saved to: {out_dir}")
    print("Saved: data/interim/deltaz_long.parquet, "
          "data/interim/deltaz_features.parquet")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/interim/eis_labeled.parquet"
    main(path)