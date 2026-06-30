"""
Final merge step: combine the original EIS data with the completed
per-sweep ECM fit checkpoint (fits_partial.csv) to produce the single
output CSV containing all ECM params and deltaZ" per row.
"""
import numpy as np
import pandas as pd

from ecm_pipeline import IN_PATH, OUT_PATH, FITS_PATH, GROUP_KEYS


def main():
    print("Loading original data...")
    df = pd.read_csv(IN_PATH)
    original_cols = list(df.columns)
    df["_sweep_id"] = df.groupby(GROUP_KEYS).ngroup()

    print("Loading completed fits...")
    fit_df = pd.read_csv(FITS_PATH)
    print(f"  {len(fit_df):,} sweeps fit")

    print("Merging...")
    merged = df.merge(fit_df, on="_sweep_id", how="left")
    merged = merged.drop(columns=["_sweep_id"])

    print("Computing Z_healthy and deltaZ\" for every row...")
    w = 2 * np.pi * merged["freq_Hz"].to_numpy()
    R0 = merged["R0_mOhm"].to_numpy()
    R1 = merged["R1_mOhm"].to_numpy()
    Q1 = merged["Q1"].to_numpy()
    n1 = merged["n1"].to_numpy()
    R2 = merged["R2_mOhm"].to_numpy()
    Q2 = merged["Q2"].to_numpy()
    n2 = merged["n2"].to_numpy()

    Z1 = 1.0 / (1.0 / R1 + Q1 * (1j * w) ** n1)
    Z2 = 1.0 / (1.0 / R2 + Q2 * (1j * w) ** n2)
    Z_pred = R0 + Z1 + Z2  # standard convention Z' + jZ''

    # dataset's Z_imag_mOhm uses the opposite sign convention -> flip
    merged["Z_real_healthy_mOhm"] = Z_pred.real
    merged["Z_imag_healthy_mOhm"] = -Z_pred.imag
    merged["deltaZ_imag_mOhm"] = merged["Z_imag_mOhm"] - merged["Z_imag_healthy_mOhm"]

    param_cols = ["R0_mOhm", "R1_mOhm", "Q1", "n1", "R2_mOhm", "Q2", "n2",
                  "fit_rmse_mOhm", "fit_r2"]
    derived_cols = ["Z_real_healthy_mOhm", "Z_imag_healthy_mOhm", "deltaZ_imag_mOhm"]
    merged = merged[original_cols + param_cols + derived_cols]

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"  {len(merged):,} rows x {len(merged.columns)} columns")
    print(f"  median fit_rmse_mOhm: {fit_df['fit_rmse_mOhm'].median():.4f}")
    print(f"  median fit_r2: {fit_df['fit_r2'].median():.4f}")
    print(f"  fraction sweeps with r2 < 0.8: {(fit_df['fit_r2'] < 0.8).mean():.4f}")
    print(f"  fraction sweeps with r2 < 0.5: {(fit_df['fit_r2'] < 0.5).mean():.4f}")


if __name__ == "__main__":
    main()