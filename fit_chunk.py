"""
Resumable chunked ECM fitting driver.

Run this repeatedly (e.g. each call does up to MAX_SECONDS of work).
It picks up where it left off using fits_partial.csv as a checkpoint.
"""
import time
import sys
import pandas as pd
import numpy as np

from ecm_pipeline import fit_ecm, GROUP_KEYS, IN_PATH, FITS_PATH

MAX_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0


def main():
    t0 = time.time()

    print("Loading data...")
    df = pd.read_csv(IN_PATH)
    df["_sweep_id"] = df.groupby(GROUP_KEYS).ngroup()
    n_sweeps = df["_sweep_id"].nunique()

    # Load existing progress if any
    try:
        done_df = pd.read_csv(FITS_PATH)
        done_ids = set(done_df["_sweep_id"].tolist())
        records = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids):,}/{n_sweeps:,} sweeps already fit")
    except FileNotFoundError:
        done_ids = set()
        records = []
        print(f"Starting fresh: 0/{n_sweeps:,} sweeps fit")

    grouped = df.groupby("_sweep_id")
    n_done_this_call = 0

    for sweep_id, sub in grouped:
        if sweep_id in done_ids:
            continue
        if time.time() - t0 > MAX_SECONDS:
            print(f"Time budget reached, stopping early this call.")
            break

        freq = sub["freq_Hz"].to_numpy()
        zr = sub["Z_real_mOhm"].to_numpy()
        zi = sub["Z_imag_mOhm"].to_numpy()

        params, rmse, r2 = fit_ecm(freq, zr, zi)
        R0, R1, Q1, n1, R2, Q2, n2 = params

        records.append({
            "_sweep_id": sweep_id,
            "R0_mOhm": R0, "R1_mOhm": R1, "Q1": Q1, "n1": n1,
            "R2_mOhm": R2, "Q2": Q2, "n2": n2,
            "fit_rmse_mOhm": rmse, "fit_r2": r2,
        })
        done_ids.add(sweep_id)
        n_done_this_call += 1

    pd.DataFrame(records).to_csv(FITS_PATH, index=False)
    elapsed = time.time() - t0
    print(f"This call: fit {n_done_this_call:,} sweeps in {elapsed:.1f}s")
    print(f"Total progress: {len(done_ids):,}/{n_sweeps:,} "
          f"({100*len(done_ids)/n_sweeps:.1f}%)")

    if len(done_ids) >= n_sweeps:
        with open("/home/claude/ecm_pipeline/fitting_done.flag", "w") as f:
            f.write("done\n")
        print("ALL SWEEPS FIT. Ready for final merge step.")
    else:
        remaining = n_sweeps - len(done_ids)
        rate = n_done_this_call / elapsed if elapsed > 0 else 1
        eta = remaining / rate if rate > 0 else float("inf")
        print(f"Remaining: {remaining:,} sweeps, ETA ~{eta/60:.1f} more min "
              f"at this rate -> run this script again.")


if __name__ == "__main__":
    main()