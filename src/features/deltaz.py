"""
deltaz.py — Stage 1 -> Stage 2 feature bridge: build the deltaZ table.

deltaZ(f) = Z_baseline(f) - Z_now(f), where Z_baseline is each cell's own
first VALID check-up spectrum, per (cell_id, soc_nom, is_rt) — this is the
convention locked in the Stage 0 doc  and restated in the Stage 2
README (Part 0), chosen so it lines up with how the dataset's own
z_ref_init is defined.

Assumes Z_imag_mOhm's sign convention has already been locked/corrected
upstream during Stage 1 ECM fitting (Stage 2 README Part 7) — this module
does not attempt to fix sign issues itself.

Produces two outputs:
  1. deltaz_long     — one row per (cell_id, cu_index, soc_nom, is_rt, freq_Hz):
                        the full-resolution deltaZ signal. Needed for anything
                        downstream that wants the per-frequency curve rather
                        than a summary (e.g. future architectures, Nyquist-
                        style diagnostics).
  2. deltaz_features — one row per spectrum: the three candidate summary
                        statistics from Stage 2 README Part 3, so they can be
                        compared against soh_cap before locking which one
                        actually feeds f_physics.
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "cell_id", "cu_index", "soc_nom", "is_rt",
    "freq_Hz", "Z_imag_mOhm", "valid",
]

# Aging-sensitive band; matches the region z_ref_init is anchored to (Stage 0 §6.2).
LOW_FREQ_BAND_HZ = (1.0, 5.0)


def _check_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")


def build_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (cell_id, soc_nom, is_rt, freq_Hz), select the EARLIEST valid
    check-up as the baseline reading.

    We don't assume cu_index == 0 is always usable — if the first check-up
    had an invalid point at a given frequency, we fall back to the next
    check-up where that exact (cell, soc, pass, freq) combination is valid.
    This keeps the baseline physically real rather than trusting cu_index 0
    blindly.

    Returns one row per (cell_id, soc_nom, is_rt, freq_Hz).
    """
    _check_columns(df)

    valid_df = df[df["valid"] == 1].copy()
    if valid_df.empty:
        raise ValueError("No valid rows found — cannot build any baseline.")

    valid_df = valid_df.sort_values(
        ["cell_id", "soc_nom", "is_rt", "freq_Hz", "cu_index"]
    )

    baseline = (
        valid_df
        .groupby(["cell_id", "soc_nom", "is_rt", "freq_Hz"], as_index=False)
        .first()[["cell_id", "soc_nom", "is_rt", "freq_Hz", "cu_index", "Z_imag_mOhm"]]
        .rename(columns={
            "cu_index": "baseline_cu_index",
            "Z_imag_mOhm": "Z_baseline_mOhm",
        })
    )

    return baseline


def compute_deltaz_long(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the baseline onto every spectrum and compute
    deltaZ(f) = Z_baseline(f) - Z_now(f) at each frequency.

    Rows where the CURRENT point is invalid keep deltaZ as NaN rather than
    silently computing a number from a flagged-bad reading — downstream
    aggregation must handle these NaNs explicitly, not have them dropped
    silently upstream.
    """
    _check_columns(df)

    merged = df.merge(
        baseline,
        on=["cell_id", "soc_nom", "is_rt", "freq_Hz"],
        how="left",
        validate="many_to_one",  # guard: exactly one baseline row per (cell, soc, pass, freq)
    )

    n_unmatched = merged["Z_baseline_mOhm"].isna().sum()
    if n_unmatched:
        raise ValueError(
            f"{n_unmatched} rows have no baseline match — check that every "
            "(cell_id, soc_nom, is_rt, freq_Hz) combination has at least one "
            "valid spectrum somewhere in its history."
        )

    merged["deltaZ_imag_mOhm"] = np.where(
        merged["valid"] == 1,
        merged["Z_baseline_mOhm"] - merged["Z_imag_mOhm"],
        np.nan,
    )

    # The baseline spectrum's own deltaZ is 0 by construction — useful as a
    # sanity check row, not a real "aged" data point. Flagged, not dropped,
    # so downstream code can decide whether to exclude it.
    merged["is_baseline_row"] = merged["cu_index"] == merged["baseline_cu_index"]

    return merged


def compute_deltaz_features(
    deltaz_long: pd.DataFrame,
    low_freq_band: tuple = LOW_FREQ_BAND_HZ,
) -> pd.DataFrame:
    """
    Collapse the per-frequency deltaZ table into one row per spectrum, with
    the three candidate summary statistics from Stage 2 README Part 3:

      - deltaz_mean_all       : mean deltaZ across all valid frequency points
      - deltaz_mean_lowfreq   : mean deltaZ restricted to the aging-sensitive
                                low-frequency band (default 1-5 Hz)
      - deltaz_integrated_abs : sum of |deltaZ| across all valid frequency
                                points (unweighted; swap for a frequency-
                                weighted integral later if diagnostics call
                                for it)

    Also reports n_valid_points / n_total_points per spectrum, so spectra
    with too few valid points can be filtered or flagged before modeling —
    this is deliberately left as a visible column rather than silently
    dropped rows, consistent with the "surface don't hide" pattern used
    throughout Stage 0.
    """
    lo, hi = low_freq_band
    grp_cols = ["cell_id", "cu_index", "soc_nom", "is_rt"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        valid_g = g[g["valid"] == 1]
        low_band = valid_g[(valid_g["freq_Hz"] >= lo) & (valid_g["freq_Hz"] <= hi)]

        return pd.Series({
            "deltaz_mean_all": valid_g["deltaZ_imag_mOhm"].mean(),
            "deltaz_mean_lowfreq": (
                low_band["deltaZ_imag_mOhm"].mean() if len(low_band) else np.nan
            ),
            "deltaz_integrated_abs": valid_g["deltaZ_imag_mOhm"].abs().sum(),
            "n_valid_points": len(valid_g),
            "n_total_points": len(g),
            "baseline_cu_index": g["baseline_cu_index"].iloc[0],
        })

    features = (
        deltaz_long
        .groupby(grp_cols)
        .apply(_agg)
        .reset_index()
    )

    return features


def build_deltaz_table(df: pd.DataFrame, low_freq_band: tuple = LOW_FREQ_BAND_HZ):
    """
    End-to-end entry point: eis_labeled (+ ECM columns from Stage 1) ->
    (deltaz_long, deltaz_features).

    Usage:
        eis = pd.read_parquet("data/interim/eis_with_ecm.parquet")
        deltaz_long, deltaz_features = build_deltaz_table(eis)
        deltaz_long.to_parquet("data/interim/deltaz_long.parquet")
        deltaz_features.to_parquet("data/interim/deltaz_features.parquet")

    `deltaz_features` is what Stage 2 README Part 3 needs directly: fit each
    of the three candidate f_physics forms (exponential, power law, linear)
    against soh_cap using each of the three summary columns in turn, and
    compare R^2 to decide which (summary statistic, functional form) pair
    becomes f_physics.
    """
    baseline = build_baseline_table(df)
    deltaz_long = compute_deltaz_long(df, baseline)
    deltaz_features = compute_deltaz_features(deltaz_long, low_freq_band)
    return deltaz_long, deltaz_features


if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/interim/eis_with_ecm.parquet"
    eis = pd.read_parquet(in_path)

    deltaz_long, deltaz_features = build_deltaz_table(eis)

    deltaz_long.to_parquet("data/interim/deltaz_long.parquet")
    deltaz_features.to_parquet("data/interim/deltaz_features.parquet")

    print(f"deltaz_long:     {len(deltaz_long):,} rows")
    print(f"deltaz_features: {len(deltaz_features):,} rows (spectra)")
    print(
        "n_valid_points distribution:\n",
        deltaz_features["n_valid_points"].describe(),
    )