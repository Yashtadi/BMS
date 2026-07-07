# ECM Fitting (Stage 1)

This document explains the **Equivalent Circuit Model (ECM) fitting pipeline** — the stage that turns
each raw impedance spectrum in `eis_labeled.parquet` / `eis_spectra.parquet` into physically
interpretable circuit parameters (R_ohm, R_sf, R_ct, CPE terms, Warburg σ). This is Stage 1 in the
project's pipeline, sitting between Stage 0 (`data_exploration.md`) and the deltaZ / Stage 2
physics-residual model (`deltaz.md`, `physics_residual_model.md`).

Code: `fit_ecm.py`. Companion exploration tools: `explore_ecm.py`,
`compare_specific_spectra.py`.

---

## Part 0 — Data source

**Input:** either `data/interim/eis_labeled.parquet` or `data/interim/eis_spectra.parquet` — both work,
since fitting only touches `cell_id, cu_index, soc_nom, is_rt, freq_Hz, Z_real_mOhm, Z_imag_mOhm,
valid`. Neither `soh_cap` nor recipe metadata is used here. If you fit against `eis_spectra.parquet`
(no `soh_cap` column), you'll need to merge `soh_cap` back in from `eis_labeled.parquet` afterward,
joining on the four spectrum-key columns, before building the Stage 2 feature table.

---

## Part 1 — What's being fit

Every spectrum gets **two** circuit fits:

```
L + R_ohm + (R_sf ∥ CPE_sf) + (R_ct ∥ CPE_dl) + Warburg      (primary, 2-arc — this is what's kept)
L + R_ohm + (R_mid ∥ CPE_mid) + Warburg                       (reference, 1-arc — comparison only)
```

The 1-arc fit is **never used as a fallback**. Every spectrum's output always has the same fixed set of
columns (`R_sf`, `R_ct`, `Q_sf`, `n_sf`, `Q_ct`, `n_ct`, …) — the 1-arc fit exists solely to compute an
AIC comparison and the quality flags below.

**Why two arcs:** confirmed on real spectra (see the deltaZ/physics-residual model discussion) — the
two-arc model beats the one-arc model on AIC for the large majority of spectra (~89% in early
testing), and this matches the planned Stage 2 feature schema (`Q1/n1`, `Q2/n2` — two CPEs).

**The known limitation:** the two arcs are often too close together in characteristic frequency
(commonly ~7–10x apart) to cleanly separate — well below the ~100–1000x separation that would produce
two visibly distinct semicircles on a Nyquist plot. This means the *individual* R_sf/R_ct split is
frequently unreliable, even when the two-arc model as a whole is clearly justified. The quality flags
below exist specifically to track this, per spectrum.

---

## Part 2 — Quality flags, and what `reliable_split` currently means

| Column | What it checks |
|---|---|
| `peak_freq_ratio` | How far apart the two arcs' characteristic frequencies are. **Computed and stored, but does NOT currently gate `reliable_split`** (see decision below). |
| `cpe_bound_hit` | `True` if either CPE exponent (`n_sf`, `n_ct`) hit the literal 0/1 optimizer bound, OR fell below `CPE_N_PLAUSIBLE_MIN = 0.3` (a near-zero exponent produces numerically absurd `peak_freq_ratio` values that look like great separation but are actually a degenerate fit). |
| `aic_prefers_2arc` | `True` if the 2-arc model beats the 1-arc model on AIC for **this specific spectrum** — not decided once globally. |
| `reliable_split` | `aic_prefers_2arc AND NOT cpe_bound_hit`. |
| `R_mid_total` | `R_sf + R_ct` — the combined resistance, well-constrained even when the individual split isn't (same idea as weighing two overlapping objects together vs. separately). |

> **Decision (current):** `peak_freq_ratio` was deliberately removed from the `reliable_split` gate —
> arc-separation distance is not treated as a reliability requirement in this version. Only CPE
> plausibility and the AIC comparison determine `reliable_split`. Reversible — revisit if downstream
> results suggest the split needs the stricter bar back.

> **Known follow-up, not yet implemented:** a small number of spectra (found during testing, across
> multiple cells) converge to a physically impossible near-zero `R_ohm` (a real 18650 cell never has
> ~0 mΩ ohmic resistance) despite the fit reporting success with no error. A plausibility guard for
> this (`r_ohm_implausible`, gating `reliable_split`) is planned but **not included in the current
> version** of `fit_ecm.py`. Until it's added, spot-check `R_ohm` for implausibly small values before
> trusting `reliable_split == True` at face value.

### What to actually do when `reliable_split` is `False`
Use `R_mid_total` as the resistance feature rather than the individual `R_sf`/`R_ct` split. Don't make
claims like "the surface film specifically grew faster than charge-transfer" for these rows — that
individual story isn't supported. `R_ohm` and `R_mid_total` are the two genuinely solid numbers out of
every fit, regardless of `reliable_split`.

---

## Part 3 — Output columns (`ecm_features.parquet`)

One row per spectrum (`cell_id, cu_index, soc_nom, is_rt`):

| Column | Meaning |
|---|---|
| `n_valid_points` | How many of the 28 frequency points were usable (valid flag + non-null) |
| `fit_succeeded` | `False` if fewer than 10 valid points, or the fit itself raised an error |
| `L`, `R_ohm` | Inductance, ohmic resistance |
| `R_sf`, `Q_sf`, `n_sf` | Surface-film arc: resistance, CPE magnitude, CPE exponent |
| `R_ct`, `Q_ct`, `n_ct` | Charge-transfer arc: same three |
| `sigma` | Warburg (diffusion) coefficient |
| `R_mid_total` | `R_sf + R_ct` — see Part 2 |
| `rss_2arc`, `aic_2arc`, `rss_1arc`, `aic_1arc` | Fit-quality numbers behind `aic_prefers_2arc` |
| `peak_freq_ratio`, `cpe_bound_hit`, `aic_prefers_2arc`, `reliable_split` | Quality flags, Part 2 |
| `error` | Reason a fit failed, if `fit_succeeded == False` |

---

## Part 4 — How to run this

**Prerequisites:** `bms` conda environment active, with `impedance` installed (already in
`environment.yml`).

### Quick sample test first (always do this before a full run)
```powershell
python fit_ecm.py --file data\interim\eis_labeled.parquet --out data\interim\ecm_features_sample.parquet --sample 100 --n-jobs 6
```
Check the printed `Succeeded: X/100 | Reliable R_sf/R_ct split: X/X` line looks sane before continuing.

### Full run
```powershell
python fit_ecm.py --file data\interim\eis_spectra.parquet --out data\interim\ecm_features.parquet --n-jobs 6
```
Set `--n-jobs` to roughly your CPU core count (check with `python -c "import os; print(os.cpu_count())"`),
leaving one or two free. Expect **45–90 minutes** for all ~39,368 spectra on a typical multi-core
laptop; this scales from the ~0.5 s/spectrum single-core rate measured during testing.

### Interrupting and resuming
Progress is checkpointed to the `--out` file every `checkpoint_every` spectra (default 2000). If you
Ctrl+C, close the terminal, or your machine sleeps mid-run, **just rerun the exact same command** — it
detects what's already in the output file and only fits what's left:
```
Resuming: found 12000 already-fitted spectra in data\interim\ecm_features.parquet
Fitting 27368 spectra (12000 already done, n_jobs=6)...
```

**To force a full restart from zero** (e.g. after updating the script itself), delete the existing
output first — otherwise it will resume using whatever's already there, even if it came from an older
version of the fitting logic:
```powershell
del data\interim\ecm_features.parquet
python fit_ecm.py --file data\interim\eis_spectra.parquet --out data\interim\ecm_features.parquet --n-jobs 6
```

### Exploring the output afterward
```powershell
python explore_ecm.py --file data\interim\ecm_features.parquet
```
Prints summary stats and saves a plot of `R_mid_total` vs. check-up index for a handful of cells.

For comparing specific spectra side-by-side (e.g. same cell across check-ups, or same condition across
cells), use `compare_specific_spectra.py` in a Jupyter cell — edit the `spectra_to_compare` list at the
top to whichever `(cell_id, cu_index, soc_nom, is_rt)` combinations you want.

---

## Part 5 — Findings from real data runs

- **Reliable-split rate:** roughly 30–50% of spectra get `reliable_split == True`, depending on the
  exact criteria version — meaning for around half of all spectra (or more, before the
  `peak_freq_ratio` requirement was removed), the individual R_sf/R_ct split should not be trusted at
  face value.
- **`aic_prefers_2arc` is true for ~89% of spectra** — the two-arc topology is justified for the large
  majority of the dataset, even where the specific split it finds isn't reliable.
- **Manufacturing consistency check:** comparing 5 different fresh cells (check-up 0, same SoC/pass)
  across different aging recipes showed a tight spread (`R_ohm` 13.7–14.8 mΩ, `R_mid_total` 3.8–4.3
  mΩ) — confirming cell-to-cell variation at the start of life is small, so later divergence between
  cells reflects the aging recipe, not manufacturing noise.
- **RT vs OT resistance:** consistently higher resistance during the OT (native operating temperature)
  pass than the RT (~25°C) pass for the same SoC — physically expected, and a useful sanity check that
  the fits are tracking something real.
- **Degenerate R_ohm fits:** found across multiple cells, not isolated to one — see the Part 2 "known
  follow-up" note.

---

## Part 6 — Decisions log

| # | Decision | Why | Reversible? |
|---|---|---|---|
| 1 | Two-arc topology, no per-spectrum fallback to one-arc | AIC prefers it for ~89% of spectra; matches planned Stage 2 feature schema | yes |
| 2 | `reliable_split` = `aic_prefers_2arc AND NOT cpe_bound_hit` (peak_freq_ratio removed) | Arc-separation distance not treated as a reliability requirement | yes — was previously stricter, may revisit |
| 3 | `R_mid_total` provided as a fallback feature | Trustworthy even when the individual split isn't | yes |
| 4 | Checkpointing every 2000 spectra, resumable by rerunning the same command | Full run takes 45–90+ minutes; must survive interruption | yes |
| — | `r_ohm_implausible` guard | **Planned, not yet implemented** — known degenerate-fit mode found during testing | — |

---

