# filter_stage2_features.py

Drops spectra with a Kramers–Kronig (KK) validation failure or an implausible
one-arc ECM fit, and — before dropping anything — reports **where** the dropped
rows concentrate, so you can tell whether the filter is removing noise or
systematically removing the most-aged spectra.

## What it does

Takes the merged Stage-2 feature table (from `build_feature_table.py`) and
removes rows that are unreliable for downstream modelling, on two grounds:

1. **KK-validation failure** — rows where `kk_valid == False`. Kramers–Kronig
   consistency is a physical validity check on an impedance spectrum; a failure
   means the spectrum is not trustworthy.
2. **Implausible one-arc fit** — rows whose fitted ECM parameters fall outside
   physically sensible bounds:
   - `R_ohm_1arc < 1.0` mOhm (near-zero series resistance — unphysical)
   - `R_1arc > 150.0` mOhm (absurdly large charge-transfer resistance)
   - `n_1arc >= 0.999` or `n_1arc <= 0.001` (a pinned CPE exponent — the fit hit
     its bound rather than converging)

A row is dropped if **either** condition holds.

## The drop-concentration report (why it exists)

Before writing the filtered file, the script prints the drop rate broken down by
**life stage** (early / mid / late, by `cu_index`) and by **SoC** (`soc_nom`).

This matters because a hard filter is only safe if the dropped rows are noise
scattered across the dataset. If drops cluster heavily at **late** check-ups,
the filter would be systematically removing the most-aged, most-informative
spectra — exactly the data later-life analysis depends on. In that case a
**downweighting** approach (keep the rows but trust them less) is preferable to
a hard drop. The report lets you make that call from evidence rather than
assuming the filter is harmless.

## Thresholds

Defined at the top of the file, matching `fit_ecm.py`:

| constant | value | meaning |
|---|---|---|
| `R_OHM_MIN_PLAUSIBLE` | 1.0 mOhm | minimum plausible series resistance |
| `R_1ARC_MAX_PLAUSIBLE` | 150.0 mOhm | maximum plausible charge-transfer resistance |

The `n_1arc` bounds (0.001 / 0.999) are hard-coded in
`compute_r1arc_implausible`.

## Inputs

- `--file` — the merged Stage-2 feature table
  (default `data/interim/stage2_features.parquet`). Must contain
  `kk_valid`, `R_ohm_1arc`, `R_1arc`, `n_1arc`, `cu_index`, `soc_nom`.

## Outputs

- `--out` — filtered table. Defaults to `<file>_filtered.parquet` next to the
  input.

## Usage

```bash
python filter_stage2_features.py --file data/interim/stage2_features.parquet
```

With an explicit output path:

```bash
python filter_stage2_features.py \
    --file data/interim/stage2_features.parquet \
    --out  data/interim/stage2_features_filtered.parquet
```

## Console output

The script prints:
- total rows, and counts for each drop reason (implausible fit, KK failure, either)
- the drop-concentration report (by life stage, by SoC)
- rows remaining after filtering, as a fraction of the original

## Where it sits in the pipeline

```
build_feature_table.py  ->  filter_stage2_features.py  ->  orthogonalize_features.py
   (merge sources)            (drop bad spectra)            (decorrelate ECM vs deltaZ)
```