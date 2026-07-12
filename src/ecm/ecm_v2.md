# ECM Fitting v2 — R_1arc and Kramers-Kronig Validation

This document covers **`fit_ecm_v2_with_kk.py`** — an updated version of the Stage 1 ECM fitting
pipeline documented in `ecm.md`. It's kept as a separate file (not a replacement of `fit_ecm.py`) so
both versions remain available side by side. Read `ecm.md` first for the base two-arc/one-arc fitting
approach and the original quality flags — this document only covers what's new.

---

## Part 0 — What's different from `fit_ecm.py`

| | `fit_ecm.py` (original) | `fit_ecm_v2_with_kk.py` |
|---|---|---|
| 1-arc fit output | Only `rss_1arc`, `aic_1arc` (comparison only) | Full parameters saved: `R_1arc`, `L_1arc`, `R_ohm_1arc`, `Q_1arc`, `n_1arc`, `sigma_1arc` |
| Primary resistance feature | `R_mid_total` (2-arc `R_sf + R_ct`) | **`R_1arc`** (see Part 1 — why this changed) |
| Raw-data validation | None | Kramers-Kronig linear validity test, before any circuit fitting |
| `r_ohm_implausible` guard | Included | Included (unchanged) |
| Total output columns | 25 | 37 |

---

## Part 1 — Why `R_1arc` replaced `R_mid_total` as the primary resistance feature

This was a real reversal from the original plan, based on evidence found during testing, not a
pre-existing design decision:

- **The 2-arc fit is demonstrably unstable.** The same spectrum was found to flip which model AIC
  preferred (2-arc vs. 1-arc) based on a trivial change in the optimizer's starting guess — a sign the
  two-arc split isn't reliably identifiable, not just occasionally noisy.
- **`reliable_split` is reliable in only a narrow slice of the data.** Broken down by condition: as low
  as **5% reliable at 50% SoC / RT pass** — the single most common measurement condition — versus
  ~66% at the SoC extremes (10%/90%). Across the full dataset, only ~30–50% of spectra pass, depending
  on exact criteria version.
- **Even where `reliable_split == True`, `R_mid_total` still doesn't reliably match the simpler
  1-arc estimate.** A direct comparison (44–50 spectra, reliable vs. unreliable groups) found the same
  ~11.6% average disagreement between `R_1arc` and `R_mid_total` in *both* groups — meaning
  `reliable_split` tracks whether the CPE values look sane, not whether the total resistance estimate
  is stable across topology choices. These are different questions.
- **`R_1arc` is structurally immune to the 2-arc fit's instability**, because it comes from a
  completely separate, simpler (6-parameter) optimization that never touches the fragile 9-parameter
  fit at all — not a formula or weighted combination of `R_sf`/`R_ct`, an independent fit to the same
  raw data.
- **Typical agreement is still good** (~4–6% apart) for well-behaved spectra — the divergence mainly
  concentrates at high-resistance, late-life conditions, where it can reach 45–90%.

> **Decision:** use `R_1arc` as the default resistance feature for Stage 2 (`g_NN`). Keep the 2-arc fit
> and `reliable_split`/`aic_prefers_2arc` available as a secondary/diagnostic signal — e.g., which
> spectra might be worth revisiting later with DRT — not as the primary feature source.

---

## Part 2 — Kramers-Kronig (KK) validation

**What it checks:** whether the *raw measured spectrum* is internally consistent with what EIS
assumes — linearity, causality, stability, and time-invariance during the sweep (e.g. the cell didn't
drift partway through a slow low-frequency measurement). This runs **before** and **independent of**
any circuit fitting — a spectrum can fail KK even if some circuit fits it with low error, and can pass
KK even if no simple circuit fits it cleanly. It answers "is the measurement trustworthy," not "does
this particular circuit explain it well."

**Method:** the linear KK test (Schönleber et al. 2014), via `impedance.py`'s `linKK` — fits a generic
"measurement model" (a flexible series of RC elements, not the physical ECM circuit) to the spectrum
and checks the residuals.

**New columns:**

| Column | Meaning |
|---|---|
| `kk_M` | Number of RC elements the measurement model needed |
| `kk_mu` | Schönleber under/over-fitting statistic |
| `kk_max_residual` | Largest relative residual across all frequencies (real + imaginary) |
| `kk_valid` | `True` if `kk_max_residual < 5%` |
| `kk_error` | Separate error field if KK validation itself failed — kept apart from the circuit-fit `error` field so a KK failure can't leave a stale error message on an otherwise-successful fit |

**KK validation flags, it does not reject.** Consistent with every other quality check in this
pipeline (`cpe_bound_hit`, `r_ohm_implausible`), `kk_valid == False` spectra still get fit normally —
downstream stages decide whether to exclude them, not this script.

**Real result from testing:** across a 150-spectrum sample, **92% passed** (median residual ~3%,
threshold 5%). No meaningful performance cost — timing stayed at ~0.48s/spectrum, same as before KK
was added, since fitting the linear measurement model is cheap relative to the nonlinear circuit
optimization.

---

## Part 3 — A library bug found and fixed during testing

**Symptom:** on first implementation, KK validation failed on every single spectrum tested — not a
data problem, a bug inside `impedance.py`'s own `linKK` function.

**Root cause:** `linKK`'s internal progress-printing (triggered whenever the adaptive element search
hits a multiple of 10) calls a routine that `eval()`s a circuit string, using
`impedance.validation.circuit_elements` as the evaluation namespace — but that dictionary is missing a
`numpy` reference the evaluated code needs, raising `NameError: name 'np' is not defined`. Since which
spectra trigger a multiple-of-10 element count is unpredictable, this could have silently broken KK
validation for an unpredictable subset of the full ~39k-spectrum run if left unpatched.

**Fix, applied before any `linKK` call:**
```python
from impedance.validation import circuit_elements as _kk_circuit_elements
_kk_circuit_elements["np"] = np
```

**Verified fixed:** confirmed working across a 150-spectrum batch, including spectra that specifically
hit `M == 10` (the exact original trigger point) — zero crashes, zero entries in `kk_error`.

---

## Part 4 — How to run this

Same pattern as `fit_ecm.py` (see `ecm.md` Part 4 for the full walkthrough) — sample test first, then
full run, checkpointed/resumable:

```powershell
python fit_ecm_v2_with_kk.py --file data\interim\eis_spectra.parquet --out data\interim\ecm_features_v2.parquet --sample 100 --n-jobs 6
```
Check the printed line looks sane:
```
Succeeded: X/100  |  Reliable R_sf/R_ct split: X/X  |  KK-valid: X/100
```

Then the full run:
```powershell
python fit_ecm_v2_with_kk.py --file data\interim\eis_spectra.parquet --out data\interim\ecm_features_v2.parquet --n-jobs 6
```

> **Note the different output filename** (`ecm_features_v2.parquet`) — kept separate from the original
> `ecm_features.parquet` so both versions' outputs coexist, matching the decision to keep both scripts
> as separate files rather than replacing the original.

---

## Part 5 — Decisions log

| # | Decision | Why | Reversible? |
|---|---|---|---|
| 1 | `R_1arc` (not `R_mid_total`) is the primary resistance feature | Structurally immune to the 2-arc fit's instability; `reliable_split` doesn't predict `R_mid_total` stability anyway | yes |
| 2 | Keep both `fit_ecm.py` and `fit_ecm_v2_with_kk.py` as separate files | Preserve the original version rather than overwrite it | yes — can consolidate later once v2 is fully trusted |
| 3 | KK validation flags, doesn't reject | Consistent with the project's existing quality-flag philosophy | yes |
| 4 | Patch `impedance.validation.circuit_elements` to add missing `np` | Library bug, not fixable by changing our own logic | N/A — required for `linKK` to work at all in this environment |

---

## Part 6 — What's next

1. Run the full dataset with `fit_ecm_v2_with_kk.py` and confirm succeeded/reliable/KK-valid counts.
2. Update the deltaZ / Stage 2 feature table to draw `R_1arc` (not `R_mid_total`) as the resistance
   input to `g_NN`.
3. Decide whether `kk_valid == False` spectra should be excluded, downweighted, or just tracked when
   building the Stage 2 feature table — not yet decided.
4. Consider consolidating back into a single `fit_ecm.py` once `fit_ecm_v2_with_kk.py` is fully
   validated against the complete dataset.
