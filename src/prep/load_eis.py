"""
Stage 0 loader: turn one raw KIT EIS cell file into a tidy, spectrum-level table.

Units: impedance in milliohms (mOhm), frequency in Hz, temperature in degC,
SoC in percent. One output row = one frequency point of one clean spectrum.
"""
import re
from pathlib import Path
import pandas as pd

# 48h = verified boundary between within-check-up (<=~33h) and between-check-up (>=~125h) gaps.
CU_GAP_THRESHOLD_HOURS = 48.0

# filename like: cell_eisv2_P001_1_S01_C10.csv
_FNAME_RE = re.compile(r"cell_eisv2_(P\d+)_(\d+)_(S\d+)_(C\d+)\.csv")


def parse_cell_id(path) -> dict:
    """Parse pack / replicate / board / channel out of a raw filename."""
    name = Path(path).name
    m = _FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected filename format: {name}")
    pack, replicate, board, channel = m.groups()
    return {
        "cell_id": Path(path).stem,
        "pack": pack,
        "replicate": int(replicate),
        "board": board,
        "channel": channel,
    }


def assign_cu_index(timestamps: pd.Series,
                    gap_threshold_hours: float = CU_GAP_THRESHOLD_HOURS) -> pd.Series:
    """Number the check-ups (0,1,2,...) by sessionizing on time gaps.

    A gap larger than `gap_threshold_hours` starts a new check-up.
    Returns an int Series aligned to the input order. Empty in -> empty out.
    """
    ts = timestamps.reset_index(drop=True)
    if len(ts) == 0:
        return pd.Series([], dtype=int)
    order = ts.sort_values().index
    sorted_ts = ts.loc[order]
    is_new = (sorted_ts.diff() > gap_threshold_hours * 3600).fillna(False)
    return pd.Series(index=order, data=is_new.cumsum().values).sort_index()


def process_cell_file(path) -> pd.DataFrame:
    """Clean ONE raw cell CSV into a tidy spectrum-level table.

    1. read (semicolon-separated)
    2. drop the seq_nr==0 priming ping (keeps the 28 real points)
    3. reconstruct the check-up index from timestamps
    4. keep the LATEST measurement when a (cu, soc, pass) is duplicated
    5. select + rename to the output schema

    Returns an empty DataFrame for the empty P000 placeholder files.
    """
    ids = parse_cell_id(path)
    df = pd.read_csv(path, sep=";")
    if len(df) == 0:                       # empty P000 reference slot
        return pd.DataFrame()

    # 2. drop priming row
    df = df[df["seq_nr"] != 0].copy()

    # 3. check-up index
    ts_unique = pd.Series(sorted(df["timestamp_s"].unique()))
    cu = assign_cu_index(ts_unique)
    ts_to_cu = pd.Series(cu.values, index=ts_unique.values)
    df["cu_index"] = df["timestamp_s"].map(ts_to_cu)

    # 4. dedup -> keep latest timestamp per (cu, soc, pass)
    latest = df.groupby(["cu_index", "soc_nom", "is_rt"])["timestamp_s"].transform("max")
    df = df[df["timestamp_s"] == latest].copy()

    # 5. output schema
    out = pd.DataFrame({
        "cell_id": ids["cell_id"],
        "pack": ids["pack"],
        "replicate": ids["replicate"],
        "board": ids["board"],
        "channel": ids["channel"],
        "cu_index": df["cu_index"].astype(int),
        "timestamp_s": df["timestamp_s"],           # needed to align with capacity data by time
        "soc_nom": df["soc_nom"],
        "is_rt": df["is_rt"],
        "freq_Hz": df["freq_Hz"],
        "Z_real_mOhm": df["z_re_comp_mOhm"],
        "Z_imag_mOhm": df["z_im_comp_mOhm"],
        "temp_degC": df["t_avg_degC"],
        "soh_imp": df["soh_imp"],
        "z_ref_init_mOhm": df["z_ref_init_mOhm"],   # per-condition healthy baseline (ready-made ΔZ)
        "z_ref_now_mOhm": df["z_ref_now_mOhm"],      # current reference impedance
        "valid": df["valid"],
    })
    return out.reset_index(drop=True)


def load_all_cells(raw_dir, save_path=None):
    """Process every raw cell file in `raw_dir` into one combined tidy table.

    Skips empty placeholder files (the 12 P000 reference slots).
    If `save_path` is given, writes the combined table to Parquet (uncapped,
    typed, compressed — no Excel row limit).

    Returns (combined_dataframe, list_of_skipped_filenames).
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("cell_eisv2_*.csv"))
    frames, skipped = [], []
    for f in files:
        out = process_cell_file(f)
        if out.empty:
            skipped.append(f.name)
            continue
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(save_path, index=False)

    return combined, skipped
