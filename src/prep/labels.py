"""
Stage 0 label cleaning: soh_cap (capacity-based SOH) is physically bounded at 0% --
a negative reading means the cell has ~0% capacity left plus measurement noise, not
a genuinely negative capacity. Confirmed rare (0.09% of spectra, see
notebooks/README.md) so we clip rather than drop, to avoid losing real end-of-life
data points that matter for RUL modeling.

The original value is kept in soh_cap_raw so nothing is silently hidden -- consistent
with the "flag, don't silently filter" pattern used throughout src/prep and src/ecm.
"""
import pandas as pd


def clip_soh_cap(df: pd.DataFrame, min_value: float = 0.0) -> pd.DataFrame:
    """Clip soh_cap at `min_value`, preserving the original in soh_cap_raw.

    Idempotent: if soh_cap_raw already exists (e.g. this was already run once),
    re-clips from the preserved raw column rather than double-clipping.
    """
    out = df.copy()
    raw = out["soh_cap_raw"] if "soh_cap_raw" in out.columns else out["soh_cap"]
    out["soh_cap_raw"] = raw
    out["soh_cap"] = raw.clip(lower=min_value)

    n_clipped = int((raw < min_value).sum())
    print(f"  soh_cap: clipped {n_clipped} spectra below {min_value} "
          f"({n_clipped / len(out) * 100:.2f}%); originals kept in soh_cap_raw")
    return out
