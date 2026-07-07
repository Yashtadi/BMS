"""
Stage 0 capacity labels: extract capacity-based SOH from the cell_eocv2 files
and align it onto the EIS spectra by nearest time.

soh_cap is the model's real target: it is measured by an independent full
charge/discharge, so predicting it from impedance-derived features is
non-circular. We use the DISCHARGE capacity of the standardized CHECK-UP test
(cyc_condition == 2, at room temperature), not the per-cycle operational
capacity (cyc_condition == 1). See notebooks/README.md for the exploration.
"""
import glob
import os
import pandas as pd


def load_capacity_labels(eoc_dir) -> pd.DataFrame:
    """One discharge-capacity SOH per (cell, check-up) from the cell_eocv2 files.

    Filters:
      cyc_condition == 2  -> standardized check-up capacity test (~25 C RT),
                             not the per-cycle operational capacity (== 1)
      cyc_charged   == 0  -> discharge (usable capacity = the standard SOH)

    Returns: cell_key, timestamp_s, soh_cap, cap_aged_est_Ah, t_start_degC.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(str(eoc_dir), "cell_eocv2_*.csv"))):
        d = pd.read_csv(f, sep=";")
        if len(d) == 0:                       # empty P000 placeholder
            continue
        cap = d[(d["cyc_condition"] == 2) & (d["cyc_charged"] == 0)].copy()
        if len(cap) == 0:
            continue
        cap["cell_key"] = os.path.basename(f).replace("cell_eocv2_", "").replace(".csv", "")
        rows.append(cap[["cell_key", "timestamp_s", "soh_cap",
                         "cap_aged_est_Ah", "t_start_degC"]])
    return pd.concat(rows, ignore_index=True)


def attach_capacity_labels(enriched: pd.DataFrame, caps: pd.DataFrame,
                           tolerance_days: float = 3.0) -> pd.DataFrame:
    """Attach soh_cap to each EIS spectrum by nearest-in-time capacity test (per cell).

    Uses merge_asof(direction="nearest"): each spectrum gets the capacity test
    closest in time for the same physical cell. `tolerance_days` keeps the match
    inside the same check-up (never a neighbour); unmatched rows get NaN.
    """
    left = enriched.copy()
    left["cell_key"] = left["cell_id"].str.replace("cell_eisv2_", "", regex=False)

    right = caps.copy()
    right["cap_timestamp_s"] = right["timestamp_s"]     # keep to measure match distance
    right = right[["cell_key", "timestamp_s", "cap_timestamp_s",
                   "soh_cap", "cap_aged_est_Ah"]]

    left = left.sort_values("timestamp_s")
    right = right.sort_values("timestamp_s")

    merged = pd.merge_asof(
        left, right, on="timestamp_s", by="cell_key",
        direction="nearest", tolerance=int(tolerance_days * 86400),
    )

    # merge_asof leaves rows in global-time order (all cells interleaved). Re-sort into
    # the natural raw-file order so the table is human-readable: cell -> check-up ->
    # RT-then-OT pass -> SoC ascending -> frequency descending (10 kHz down to 0.05 Hz).
    merged = merged.sort_values(
        ["cell_id", "cu_index", "is_rt", "soc_nom", "freq_Hz"],
        ascending=[True, True, False, True, False],
    ).reset_index(drop=True)
    return merged
