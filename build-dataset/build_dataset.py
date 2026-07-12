"""
Rebuild the Stage 0 working dataset from the raw KIT files.

Run from the repo root, with the `bms` environment active:

    conda activate bms
    python build_dataset.py

This regenerates three Parquet files under data/interim/:
    eis_spectra.parquet   - cleaned EIS spectra (228 cells)
    eis_enriched.parquet  - spectra + recipe metadata
    eis_labeled.parquet   - + capacity-SOH target   <-- the working dataset

The raw data is NOT in Git. Download it first (DOI 10.35097/1969) into data/raw/
and extract cell_eisv2.zip and cell_eocv2.zip -- see CONTRIBUTING.md, Part 2.3.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file is in build-dataset/)
sys.path.insert(0, str(ROOT))

from src.prep.load_eis import load_all_cells
from src.prep.metadata import build_enriched_table
from src.prep.capacity import load_capacity_labels, attach_capacity_labels
from src.prep.labels import clip_soh_cap

DS = ROOT / "data" / "raw" / "10.35097-1969" / "10.35097-1969" / "data" / "dataset"
INTERIM = ROOT / "data" / "interim"
RECIPE_XLSX = DS / "Cycling Experiment Cell Overview (2024-09-26 15-31).xlsx"

# Expected numbers for a correct rebuild (see CONTRIBUTING.md, Part 2.5).
EXPECTED_CELLS = 228
EXPECTED_ROWS = 1_102_304


def _check_raw_data() -> None:
    """Fail early with a clear message if the raw data isn't where we expect it."""
    problems = []
    if not (DS / "cell_eisv2").is_dir():
        problems.append(f"  missing EIS folder:      {DS / 'cell_eisv2'}")
    if not (DS / "cell_eocv2").is_dir():
        problems.append(f"  missing capacity folder: {DS / 'cell_eocv2'}")
    if not RECIPE_XLSX.is_file():
        problems.append(f"  missing metadata xlsx:   {RECIPE_XLSX}")
    if problems:
        print("ERROR: raw data not found. Did you download and extract it?")
        print("\n".join(problems))
        print("\nSee CONTRIBUTING.md, Part 2.3 for the download & extraction steps.")
        sys.exit(1)


def main() -> None:
    _check_raw_data()
    INTERIM.mkdir(parents=True, exist_ok=True)

    print("[1/3] Cleaning EIS spectra from raw cell files ...")
    load_all_cells(DS / "cell_eisv2", save_path=INTERIM / "eis_spectra.parquet")

    print("[2/3] Joining recipe metadata ...")
    enriched = build_enriched_table(
        INTERIM / "eis_spectra.parquet",
        RECIPE_XLSX,
        save_path=INTERIM / "eis_enriched.parquet",
    )

    print("[3/3] Extracting capacity-SOH and aligning by time ...")
    caps = load_capacity_labels(DS / "cell_eocv2")
    labeled = attach_capacity_labels(enriched, caps)
    labeled = clip_soh_cap(labeled)  # soh_cap can't be physically negative; see src/prep/labels.py
    labeled.to_parquet(INTERIM / "eis_labeled.parquet", index=False)

    # --- verification -----------------------------------------------------
    n_cells = labeled["cell_id"].nunique()
    n_rows = len(labeled)
    pts = sorted(int(p) for p in
                 labeled.groupby(["cell_id", "cu_index", "soc_nom", "is_rt"]).size().unique())
    label_cov = labeled["soh_cap"].notna().mean() * 100

    print("\nBuilt data/interim/eis_labeled.parquet")
    print(f"  cells:               {n_cells}   (expected {EXPECTED_CELLS})")
    print(f"  rows:                {n_rows:,}   (expected {EXPECTED_ROWS:,})")
    print(f"  points per spectrum: {pts}   (expected [28])")
    print(f"  soh_cap coverage:    {label_cov:.1f}%   (expected 100.0%)")

    ok = (n_cells == EXPECTED_CELLS and n_rows == EXPECTED_ROWS
          and pts == [28] and abs(label_cov - 100.0) < 1e-6)
    if ok:
        print("\nOK - your dataset matches the reference. You're ready to build.")
    else:
        print("\nWARNING - numbers differ from the reference. Check the raw data "
              "layout and re-run before building on top (see CONTRIBUTING.md, Part 2.5).")
        sys.exit(1)


if __name__ == "__main__":
    main()
