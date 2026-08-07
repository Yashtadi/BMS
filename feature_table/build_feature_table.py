"""
build_feature_table.py — merge ecm_features + deltaz_features + soh_cap
(+ optionally deltaz_wide, for per-frequency comparison) into one table
ready for Stage 2 (physics_residual_model.md Part 2 onward).

Usage:
    python build_feature_table.py \
        --ecm data/interim/ecm_features_v2.parquet \
        --deltaz data/interim/deltaz_features.parquet \
        --labeled data/interim/eis_labeled.parquet \
        --deltaz-wide data/interim/deltaz_wide.parquet \
        --out data/interim/stage2_features.parquet
"""

import argparse
import pandas as pd

SPECTRUM_KEYS = ["cell_id", "cu_index", "soc_nom", "is_rt"]


def build_feature_table(
    ecm: pd.DataFrame,
    deltaz: pd.DataFrame,
    labeled: pd.DataFrame,
    deltaz_wide: pd.DataFrame = None,
) -> pd.DataFrame:
    # soh_cap lives once per spectrum in the long labeled table (repeated
    # across all 28 frequency rows) — collapse it back to one row per spectrum.
    soh = (
        labeled.groupby(SPECTRUM_KEYS)[["soh_cap", "aging_type", "temp_degC"]]
        .first()
        .reset_index()
    )

    merged = ecm.merge(deltaz, on=SPECTRUM_KEYS, how="inner", suffixes=("", "_deltaz"))
    merged = merged.merge(soh, on=SPECTRUM_KEYS, how="left", validate="one_to_one")

    if deltaz_wide is not None:
        merged = merged.merge(deltaz_wide, on=SPECTRUM_KEYS, how="left", validate="one_to_one")
        freq_cols = [c for c in deltaz_wide.columns if c.startswith("deltaz_f_")]
        print(f"Merged {len(freq_cols)} per-frequency deltaZ columns")

    n_before = len(merged)
    print(f"Merged table: {n_before} rows")
    print(f"  fit_succeeded: {merged['fit_succeeded'].sum()}")
    print(f"  soh_cap missing: {merged['soh_cap'].isna().sum()}")
    print(f"  deltaz_mean_all missing: {merged['deltaz_mean_all'].isna().sum()}")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecm", default="data/interim/ecm_features_v2.parquet")
    parser.add_argument("--deltaz", default="data/interim/deltaz_features.parquet")
    parser.add_argument("--labeled", default="data/interim/eis_labeled.parquet")
    parser.add_argument("--deltaz-wide", default=None,
                         help="optional: path to deltaz_wide.parquet (from build_deltaz_wide.py) "
                              "to add per-frequency deltaZ columns for individual-frequency comparison")
    parser.add_argument("--out", default="data/interim/stage2_features.parquet")
    args = parser.parse_args()

    ecm = pd.read_parquet(args.ecm)
    deltaz = pd.read_parquet(args.deltaz)
    labeled = pd.read_parquet(args.labeled)
    deltaz_wide = pd.read_parquet(args.deltaz_wide) if args.deltaz_wide else None

    merged = build_feature_table(ecm, deltaz, labeled, deltaz_wide)
    merged.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}  {merged.shape}")


if __name__ == "__main__":
    main()