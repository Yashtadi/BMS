"""
filter_stage2_features.py — drop rows with a KK-validation failure or an
implausible one-arc fit (near-zero R_ohm_1arc, absurdly large R_1arc, or a
pinned n_1arc CPE exponent).

Before saving anything, reports WHERE the dropped rows concentrate (by life
stage and by SoC) — if drops cluster heavily at late check-ups, filtering
risks systematically removing the most-aged, most-informative spectra
rather than just noise, and a downweighting approach should be considered
instead of a hard filter.

Usage:
    python filter_stage2_features.py --file data/interim/stage2_features.parquet
"""

import argparse
import pandas as pd

R_OHM_MIN_PLAUSIBLE = 1.0     # mOhm — see fit_ecm.py
R_1ARC_MAX_PLAUSIBLE = 150.0  # mOhm — see fit_ecm.py


def compute_r1arc_implausible(df: pd.DataFrame) -> pd.Series:
    """
    Same checks as fit_1arc_implausible in the newer fit_ecm.py — computed
    manually here for files predating that column.
    """
    return (
        (df["R_ohm_1arc"] < R_OHM_MIN_PLAUSIBLE)
        | (df["R_1arc"] > R_1ARC_MAX_PLAUSIBLE)
        | (df["n_1arc"] >= 0.999)
        | (df["n_1arc"] <= 0.001)
    )


def report_drop_concentration(df: pd.DataFrame, problem_mask: pd.Series) -> None:
    check = df.copy()
    check["life_stage"] = pd.cut(
        check["cu_index"], bins=[-1, 5, 15, 30], labels=["early", "mid", "late"]
    )

    print("=== Where the dropped rows concentrate ===")
    print("By life stage (drop rate):")
    print(check.groupby("life_stage").apply(lambda g: problem_mask.loc[g.index].mean()))
    print()
    print("By SoC (drop rate):")
    print(check.groupby("soc_nom").apply(lambda g: problem_mask.loc[g.index].mean()))
    print()


def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    r1arc_implausible = compute_r1arc_implausible(df)
    kk_failed = df["kk_valid"] == False  # noqa: E712 — explicit comparison for clarity
    problem_mask = r1arc_implausible | kk_failed

    print(f"Total rows: {len(df)}")
    print(f"R_1arc implausible: {r1arc_implausible.sum()}")
    print(f"kk_valid == False:  {kk_failed.sum()}")
    print(f"Either (to be dropped): {problem_mask.sum()} ({problem_mask.mean():.2%})")
    print()

    report_drop_concentration(df, problem_mask)

    df_clean = df[~problem_mask].copy()
    print(f"Rows remaining after filter: {len(df_clean)} ({len(df_clean) / len(df):.1%} of original)")

    return df_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/interim/stage2_features.parquet")
    parser.add_argument("--out", default=None,
                         help="defaults to <file>_filtered.parquet next to --file")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    df_clean = filter_features(df)

    out_path = args.out or args.file.replace(".parquet", "_filtered.parquet")
    df_clean.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()