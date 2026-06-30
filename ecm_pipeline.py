"""
ECM (Equivalent Circuit Model) fitting pipeline for EIS data.

Circuit topology:  R0 + (R1 || CPE1) + (R2 || CPE2)
  - R0:        ohmic / high-frequency series resistance
  - R1 || CPE1: first (high-frequency) suppressed semicircle - charge transfer + dl
  - R2 || CPE2: second (low-frequency) suppressed semicircle - SEI / diffusion-related

NOTE on sign convention: this dataset's Z_imag_mOhm is stored with the
OPPOSITE sign from the standard Z(w) = Z' + j*Z'' convention used in the
circuit math below (verified empirically: fitting with the standard
convention gives R^2 ~ 0.73 with a mirror-image residual pattern; flipping
the sign of the model's imaginary part to match the data's convention
gives R^2 ~ 0.98 on the same sweep). So throughout this script:
    Z_imag_mOhm (data)  ==  -Im( Z_model )

Each unique EIS sweep = one (source_file, SoC_pct, temp_degC, cycles,
SoH_pct) combination -- NOT one source_file. Each source_file in this
dataset contains many distinct sweeps (different SoC/temp/cycle/SoH
points all logged to the same file), so grouping by source_file alone
mixes unrelated Nyquist curves together and gives a bad fit.

For each sweep, the 7 ECM parameters [R0, R1, Q1, n1, R2, Q2, n2] are fit
by nonlinear least squares against the complex impedance
(Z_real_mOhm, Z_imag_mOhm) across all frequency points in that sweep.

Outputs (single CSV, no k-fold / cross-validation of any kind):
  - all original columns
  - fitted ECM params for that row's sweep: R0_mOhm, R1_mOhm, Q1, n1,
    R2_mOhm, Q2, n2
  - fit_rmse_mOhm, fit_r2 (quality-of-fit diagnostics, NOT model validation)
  - Z_real_healthy_mOhm, Z_imag_healthy_mOhm -> impedance predicted by the
    fitted ECM (the "healthy" / model-expected impedance)
  - deltaZ_imag_mOhm -> Z_imag_mOhm (measured) - Z_imag_healthy_mOhm
    (the residual / drift signal feeding the PINN + drift detector)
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings("ignore")

IN_PATH = "eis_cleaned.csv"  
OUT_PATH = "eis_ecm_deltaZ.csv"     
FITS_PATH = "/home/claude/ecm_pipeline/fits_partial.csv"
DONE_FLAG = "/home/claude/ecm_pipeline/fitting_done.flag"

GROUP_KEYS = ["source_file", "SoC_pct", "temp_degC", "cycles", "SoH_pct"]

# A handful of initial-guess combinations to try per sweep (multi-start),
# since this circuit can have local minima. Cheap insurance for 37k fits.
INIT_GRID = [
    (15, 1e-3, 0.8, 30, 1e-2, 0.6),
    (8, 2.5e-4, 0.95, 57, 9e-3, 0.55),
]


# ----------------------------------------------------------------------
# ECM model: two nested suppressed RC arcs
# ----------------------------------------------------------------------
def z_model(params, freq_hz):
    """Complex impedance of R0 + (R1||CPE1) + (R2||CPE2), in mOhm.
    Standard convention Z = Z' + j*Z'' (NOT yet sign-flipped)."""
    R0, R1, Q1, n1, R2, Q2, n2 = params
    w = 2 * np.pi * freq_hz
    Z1 = 1.0 / (1.0 / R1 + Q1 * (1j * w) ** n1)
    Z2 = 1.0 / (1.0 / R2 + Q2 * (1j * w) ** n2)
    return R0 + Z1 + Z2


def residuals(params, freq_hz, z_real, z_imag_data):
    """Stacked real+imag residuals. z_imag_data uses the dataset's sign
    convention, so we compare against -Im(z_model)."""
    z_pred = z_model(params, freq_hz)
    return np.concatenate([
        z_pred.real - z_real,
        (-z_pred.imag) - z_imag_data
    ])


def fit_ecm(freq_hz, z_real, z_imag_data):
    """Fit one EIS sweep with multi-start least squares.
    Returns (best_params, rmse, r2)."""
    R0_guess = np.min(z_real)
    lb = [0.0, 1e-6, 1e-8, 0.3, 1e-6, 1e-8, 0.1]
    ub = [100.0, 1000.0, 10.0, 1.0, 1000.0, 100.0, 1.0]

    best_cost = np.inf
    best_params = np.array([np.nan] * 7)
    best_resid = None

    for (R1g, Q1g, n1g, R2g, Q2g, n2g) in INIT_GRID:
        x0 = [R0_guess, R1g, Q1g, n1g, R2g, Q2g, n2g]
        # clip x0 into bounds just in case
        x0 = np.clip(x0, lb, ub)
        try:
            result = least_squares(
                residuals, x0, args=(freq_hz, z_real, z_imag_data),
                bounds=(lb, ub), method="trf", max_nfev=2000
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
                best_resid = result.fun
        except Exception:
            continue

    if best_resid is not None:
        rmse = np.sqrt(np.mean(best_resid ** 2))
        z_stack = np.concatenate([z_real, z_imag_data])
        ss_res = np.sum(best_resid ** 2)
        ss_tot = np.sum((z_stack - np.mean(z_stack)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        rmse, r2 = np.nan, np.nan

    return best_params, rmse, r2


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def main():
    print("Loading data...")
    df = pd.read_csv(IN_PATH)
    print(f"  {len(df):,} rows, {df['source_file'].nunique()} source files")

    original_cols = list(df.columns)
    df["_sweep_id"] = df.groupby(GROUP_KEYS).ngroup()
    n_sweeps = df["_sweep_id"].nunique()
    print(f"  {n_sweeps:,} individual EIS sweeps "
          f"(grouped by source_file + SoC + temp + cycles + SoH)")

    fit_records = []

    print("Fitting ECM per sweep...")
    grouped = df.groupby("_sweep_id")
    for i, (sweep_id, sub) in enumerate(grouped):
        freq = sub["freq_Hz"].to_numpy()
        zr = sub["Z_real_mOhm"].to_numpy()
        zi = sub["Z_imag_mOhm"].to_numpy()

        params, rmse, r2 = fit_ecm(freq, zr, zi)
        R0, R1, Q1, n1, R2, Q2, n2 = params

        fit_records.append({
            "_sweep_id": sweep_id,
            "R0_mOhm": R0,
            "R1_mOhm": R1,
            "Q1": Q1,
            "n1": n1,
            "R2_mOhm": R2,
            "Q2": Q2,
            "n2": n2,
            "fit_rmse_mOhm": rmse,
            "fit_r2": r2,
        })

        if (i + 1) % 2000 == 0 or (i + 1) == n_sweeps:
            print(f"  fitted {i+1:,}/{n_sweeps:,} sweeps")

    fit_df = pd.DataFrame(fit_records)

    print("Computing z\"_healthy and deltaZ\" for every row...")
    merged = df.merge(fit_df, on="_sweep_id", how="left")
    merged = merged.drop(columns=["_sweep_id"])

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
    Z_pred = R0 + Z1 + Z2   # standard convention Z' + jZ''

    # dataset's Z_imag_mOhm uses the opposite sign convention (see module
    # docstring) -> healthy z" must be sign-flipped to match
    merged["Z_real_healthy_mOhm"] = Z_pred.real
    merged["Z_imag_healthy_mOhm"] = -Z_pred.imag
    merged["deltaZ_imag_mOhm"] = merged["Z_imag_mOhm"] - merged["Z_imag_healthy_mOhm"]

    # column order: original cols, then ECM params, then z"_healthy / deltaZ"
    param_cols = ["R0_mOhm", "R1_mOhm", "Q1", "n1", "R2_mOhm", "Q2", "n2",
                  "fit_rmse_mOhm", "fit_r2"]
    derived_cols = ["Z_real_healthy_mOhm", "Z_imag_healthy_mOhm", "deltaZ_imag_mOhm"]
    merged = merged[original_cols + param_cols + derived_cols]

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(f"  {len(merged):,} rows x {len(merged.columns)} columns")
    print(f"  median fit_rmse_mOhm: {fit_df['fit_rmse_mOhm'].median():.4f}")
    print(f"  median fit_r2: {fit_df['fit_r2'].median():.4f}")


if __name__ == "__main__":
    main()