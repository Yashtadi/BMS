"""
Stage 0 metadata: parse the KIT recipe (parameter) table and join it onto the
spectra table so every spectrum knows its aging story (why it degrades).

The recipe sheet is messy: a title row on top, newline-laden headers, and
unrelated sub-tables below the 76 recipes. See notebooks/README.md for the
exploration that motivated this parser.
"""
from pathlib import Path
import pandas as pd

# The four temperature-controlled baths, decoded from the sheet's legend.
TEMP_CATEGORY_TO_DEGC = {"A": 0, "B": 10, "C": 25, "D": 40}

# Real recipes carry one of these aging types; anything else is a junk/legend row.
_AGING_TYPES = ["Calendar", "Cyclic", "Profile"]


def load_recipe_table(xlsx_path, sheet=0) -> pd.DataFrame:
    """Parse the 76-recipe parameter table, keyed by param_id (the P-code number).

    Steps:
      - take row 1 as the header (row 0 is a title banner)
      - normalise the newline-laden column names
      - keep only rows whose aging_type is a real regime, which uniquely selects
        the 76 recipe rows and drops the sub-tables that share the id column
      - decode the A/B/C/D temperature category to real degrees Celsius

    Raises ValueError if param_id is not unique (which would duplicate rows on join).
    """
    r = pd.read_excel(xlsx_path, sheet_name=sheet, header=1)
    r.columns = [str(c).replace("\n", " ").strip() for c in r.columns]
    r = r.rename(columns={
        "Param. ID": "param_id", "Age Type": "aging_type", "Temp.": "temp_cat",
        "SoC idle": "soc_idle_cat", "SoC Limits": "soc_limits_cat",
        "C- Rates": "c_rate_cat", "Profile": "profile",
    })
    cols = ["param_id", "aging_type", "temp_cat", "soc_idle_cat",
            "soc_limits_cat", "c_rate_cat", "profile"]
    r = r[cols].copy()

    r = r[r["aging_type"].isin(_AGING_TYPES)].copy()
    r["param_id"] = r["param_id"].astype(int)
    r["temp_setpoint_degC"] = r["temp_cat"].map(TEMP_CATEGORY_TO_DEGC)

    if not r["param_id"].is_unique:
        raise ValueError("recipe param_id not unique -- join would duplicate rows")
    return r.reset_index(drop=True)


def join_recipe_metadata(spectra: pd.DataFrame, recipe: pd.DataFrame) -> pd.DataFrame:
    """Attach recipe columns onto the spectra table by pack -> param_id.

    Guards against the silent one-to-many join bug: asserts the row count is
    unchanged and that every spectrum found a matching recipe.
    """
    out = spectra.copy()
    out["param_id"] = out["pack"].str[1:].astype(int)   # 'P001' -> 1
    merged = out.merge(recipe, on="param_id", how="left")

    if len(merged) != len(out):
        raise ValueError(f"join changed row count: {len(out)} -> {len(merged)}")
    n_missing = int(merged["aging_type"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} spectra rows had no recipe match")
    return merged


def build_enriched_table(spectra_parquet, recipe_xlsx, save_path=None) -> pd.DataFrame:
    """Load the spectra Parquet, join recipe metadata, optionally save to Parquet."""
    spectra = pd.read_parquet(spectra_parquet)
    recipe = load_recipe_table(recipe_xlsx)
    enriched = join_recipe_metadata(spectra, recipe)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(save_path, index=False)
    return enriched
