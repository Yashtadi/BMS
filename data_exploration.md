# Data Exploration & Preparation (Stage 0)

This document is the complete, plain-language record of everything we did to the data **before**
building the ECM (equivalent-circuit model). It is written for someone with no prior context: it
explains the dataset, every step we took *in the order we took it*, every decision and the evidence
that led to it, and finishes with a detailed description of the final cleaned dataset.

If you are picking this project up (or picking it back up after a break), read this top to bottom and
you will understand exactly what the data is and why the pipeline looks the way it does.

---

## Part 0 — What the dataset is

We use the **KIT comprehensive battery aging dataset** (RADAR4KIT, DOI `10.35097/1969`; published in
*Scientific Data*, 2024).

Researchers took **228 identical lithium-ion cells** (LG INR18650HG2 — NMC cathode, graphite+SiO
anode, 3.0 Ah nominal) and deliberately aged them over ~**600 days**. Each cell was assigned a
**recipe** (a "lifestyle"):

- **Calendar aging** — the cell just sits at a fixed charge level and temperature (storage).
- **Cyclic aging** — the cell is charged/discharged over and over (heavy use).
- **Profile aging** — the cell follows a real driving-cycle power profile (WLTP, i.e. EV use).

There are **76 recipes** (16 calendar + 48 cyclic + 12 profile), each run on **3 replicate cells**
(76 × 3 = 228). Recipes also vary temperature (0 / 10 / 25 / 40 °C), state-of-charge windows, and
charge/discharge rates.

### The check-up (CU)
Every ~3 weeks each cell was paused for a standardized **check-up** — a health exam. During a check-up
the cell is measured with several techniques, and each technique is logged into its **own file
collection**:

| File collection | Measurement | We use it for |
|---|---|---|
| `cell_eisv2` | **EIS** — impedance spectra (the cell's electrical "fingerprint") | ECM features |
| `cell_eocv2` | **Capacity test** — full charge/discharge, coulomb counting | the **health label** (`soh_cap`) |
| `cell_plsv2` | Pulse tests | not used (yet) |

A single check-up measures EIS **twice**:
- **RT pass** — cell brought to room temperature (~25 °C), swept at SoC = 10/30/50/70/90 % (stepping *up*).
- **OT pass** — cell returned to its operating temperature (0/10/25/40 °C), swept at the same 5 SoC (stepping *down*).

So one check-up = **5 SoC × 2 passes = 10 EIS spectra**, plus a capacity test.

### Filenames = cell identity
Raw EIS files are named like `cell_eisv2_P001_1_S01_C10.csv`:
`P001` = recipe, `_1` = which of the 3 replicates, `S01` = circuit board, `C10` = channel. That whole
combination is **one physical cell tracked for its whole life** — and it is the grouping key we must
never split across train/test sets (or the model would cheat by seeing the same cell in both).

> **Note:** raw files are **semicolon-separated** (`sep=";"`), not comma-separated.

---

## Part 1 — Why we threw away the first "cleaned" CSV

We started with a pre-cleaned `eis_cleaned.csv` (made by a teammate). Profiling it revealed serious
problems:
- It had **exactly 1,048,575 rows** — the **Excel row limit**. It had been opened/saved in Excel and
  **truncated**, silently dropping data.
- It contained only **216 of the 228 cells** and **8 of the 22 useful columns** — the quality flag,
  the reference-impedance columns, the real cycle counters, and the aging metadata had all been
  stripped out.

**Decision:** abandon the cleaned CSV and rebuild everything from the **raw KIT files**, which we
placed at `data/raw/10.35097-1969/…`. All raw/interim data is gitignored (never committed).

---

## Part 2 — Understanding one raw EIS file (notebook `01_raw_schema_explore.ipynb`)

We worked on a single cell (`cell_eisv2_P001_1_S01_C10.csv`) until we fully understood the format,
then scaled up. Steps in order:

### 2.1 The raw schema (22 columns)
The columns that matter: `timestamp_s` (Unix time), `soc_nom` (10/30/50/70/90), `is_rt` (RT vs OT
pass), `valid` (instrument quality flag), `seq_nr` (frequency index in a sweep), `freq_Hz`,
`z_re_comp_mOhm` / `z_im_comp_mOhm` (temperature/hardware-**compensated** real & imaginary
impedance — the "good" numbers), and `soh_imp` (an impedance-based SOH the instrument stores).

### 2.2 Each spectrum is 29 rows, but only 28 are real
Grouping rows into spectra gave **29 rows each**, not 28. The first row of every sweep (`seq_nr == 0`)
is a throwaway "priming" ping with `NaN` compensated impedance.
> **Decision #1 — drop `seq_nr == 0`**, leaving a clean 28 frequency points per spectrum.

### 2.3 The missing time axis, and how we rebuilt it
The biggest problem: **nothing tells you the order of check-ups** (no check-up index). But we realized
the *timestamps* encode it. Sorting spectra by time and looking at the **gap to the previous one**
revealed two clearly separated worlds:
- **Within a check-up:** gaps of ~1–2.5 hours (between SoC steps).
- **Between check-ups:** gaps of ~125–980 hours (~5 days to ~41 days).

Because nothing falls in the empty middle, we can label each big gap as "a new check-up." This
technique is called **sessionization**: sort by time, mark any gap over a threshold as a new session,
then take a running count. That count *is* the **check-up index (`cu_index`)** — the time axis we
were missing.
> **Decision #2 — reconstruct `cu_index` by sessionizing on `timestamp_s`.**

### 2.4 Choosing the gap threshold: 24 h was wrong, 48 h is right
A first threshold of **24 hours** mostly worked but produced anomalies. Investigating each one taught
us something real:
- **One check-up got split in two** because that cell's RT→OT temperature transition took **31.75 h**
  (reaching an extreme temperature like 0 °C needs long soak time) — longer than 24 h.
- **Fix:** every genuine between-check-up gap is ≥ 125 h, so **48 hours** sits safely in the empty
  zone between 31.75 h and 125 h. Re-running with 48 h merged the split check-up and left everything
  else intact.
> **Decision #3 — use a 48-hour gap threshold for `cu_index`.**

### 2.5 Real edge cases we chose to keep (not bugs)
- **First check-up sometimes has an extra spectrum** — an initial "as-received" measurement.
- **One check-up had its entire OT pass measured twice**, and *both* copies were `valid`. A genuine
  duplicate (protocol re-run), not a failed-and-retried measurement.
- **Last check-up may be partial** — the experiment ended mid-check-up.

To handle duplicates uniformly:
> **Decision #4 — when a `(cell, cu, SoC, pass)` has more than one valid spectrum, keep the LATEST**
> (most recent timestamp).

### 2.6 Decoding `cyc_charged`
This column tracks perfectly with `is_rt`: `1` during the RT pass (SoC stepping up = charging), `0`
during the OT pass (stepping down = discharging). It is **not** a cycle counter — just a
charge-direction flag. Dropped as redundant.

### 2.7 Verifying the threshold generalizes
A threshold tuned on one cell proves nothing. We re-ran across many cells (different recipes,
temperatures) and measured, per cell, the **largest within-check-up gap** vs the **smallest
between-check-up gap**. Across all: within-gaps stayed **12.9–32.5 h**, between-gaps **125.3–128.9 h**.
48 h sits cleanly between them for *every* cell. ✅

### 2.8 Counting the real cells (and the empty files)
Scanning all files: **240 files = 228 real cells + 12 empty files**. The 12 empty ones are exactly
`cell_eisv2_P000_0_S20_C00.csv` … `C11.csv` — a reserved reference slot (`P000`, board `S20`) that was
never populated. This explains the old CSV's "216": it dropped these 12 empties **and** lost 12 more
real cells to Excel truncation.
> **Decision #5 — process the 228 non-empty cells; skip the 12 empty `P000` files.**

---

## Part 3 — Building the combined table (notebook `02_raw_schema_explore.ipynb`)

We lifted the verified logic into reusable modules under `src/prep/` (see [Part 8](#part-8--the-code)).

### 3.1 The loader
`load_eis.process_cell_file(path)` cleans **one** file (read → drop priming → reconstruct `cu_index` →
dedup keep-latest → select columns). `load_eis.load_all_cells(dir)` runs it over all 240 files, skips
the 12 empties, and concatenates the 228 cells into one table.
> **We save to Parquet, not CSV/Excel.** Parquet has **no row limit**, keeps data types, and is
> compressed — so the ~1.1-million-row table is never truncated (the exact bug that ruined the old
> file). Verified: every spectrum came out to exactly **28 points**.

### 3.2 Validating against physics (before trusting anything)
- **Nyquist plot** of one spectrum showed a textbook EIS shape (high-frequency branch → charge-transfer
  semicircle → diffusion tail) — confirming the impedance columns and grouping are correct.
  *(This dataset's `Z_imag` sign is flipped vs. the classic −Z″ convention; we will lock the sign when
  we set up ECM fitting.)*
- **SOH-over-life** trended downward, confirming `cu_index` is a real time axis. But for gentle recipes
  the impedance-based `soh_imp` barely moved (100 → ~97 %), and a **synchronized dip at check-up 27**
  across all cells flagged a systematic measurement artifact. Both facts told us impedance-SOH is a
  weak, noisy label — motivating the capacity target in Part 5.
- Looking at the **most-aged cells**, capacity/impedance clearly falls all the way toward 0, and the
  3 replicates of a recipe age together — strong evidence the data and pipeline are trustworthy.

---

## Part 4 — Joining the recipe metadata (why each cell ages)

The KIT parameter spreadsheet (`Cycling Experiment Cell Overview…xlsx`) maps each recipe (`P`-code) to
its aging story. It is messy — a title banner on top, newline-filled headers, and unrelated
sub-tables *below* the 76 recipes. We read it with `header=1`, normalised the column names, and **kept
only rows whose `aging_type` is Calendar/Cyclic/Profile** — which uniquely selects the 76 real
recipes. Temperature categories were decoded **A/B/C/D → 0/10/25/40 °C**.

> **Bug caught here — the silent one-to-many join.** A first attempt filtered by `param_id` number
> range and let in 11 junk rows carrying **duplicate** ids. On `merge`, pandas produced a row for
> *every* match, silently **duplicating** every spectrum from packs P002–P012 — with no error. We now
> **assert `param_id` is unique before joining, and that the row count is unchanged after.** This is
> the essential guard against a classic data-corruption pitfall.

Result: 76 recipes (16 Calendar / 48 Cyclic / 12 Profile), joined onto the spectra by `pack → param_id`,
**0 unmatched rows**. Saved as `data/interim/eis_enriched.parquet`.

---

## Part 5 — The health target: capacity-SOH (`soh_cap`)

### 5.1 Why not the impedance-SOH we already had?
The model's target must be **non-circular**. `soh_imp` is *derived from impedance*, and our ECM
features will *also* be derived from impedance — so predicting `soh_imp` from ECM features would be
predicting a thing from itself (impressive-looking, scientifically hollow). The independent,
physically meaningful target is **capacity-based SOH (`soh_cap`)** — measured by actually counting the
Amp-hours in/out during a full charge/discharge. It has nothing to do with impedance.

Practical payoff: a capacity test is **slow and disruptive** (hours, out of service); an EIS
measurement is **fast** (minutes). Proving *EIS → capacity-SOH* means estimating true health quickly.
And **RUL** is defined against capacity (end-of-life ≈ 80 % capacity), so capacity is required.
> **Decision #6 — target = capacity-based SOH (`soh_cap`); keep `soh_imp` only as a sanity check.**

### 5.2 Extracting `soh_cap` correctly (two filters)
The capacity data lives in `cell_eocv2`. Understanding it took two careful filters:
- **`cyc_condition == 2`** selects the standardized **check-up capacity test** (done at ~25 °C RT),
  *not* `cyc_condition == 1` which is the per-cycle **operational** capacity logged during aging.
  Missing this inflated a cyclic cell from ~29 to thousands of measurements — caught by
  sanity-checking measurements-per-cell.
- **`cyc_charged == 0`** selects the **discharge** capacity (usable capacity = the standard SOH
  definition), vs. `== 1` (charge capacity, slightly higher due to coulombic inefficiency).
> **Decision #7 — `soh_cap` = discharge capacity of the check-up test (`cyc_condition==2 & cyc_charged==0`).**

This gives **3,999 discharge-SOH measurements** across 228 cells (~1 per check-up), all at ~25 °C.
Capacity-SOH shows a **strong** aging signal (100 % → 0 %), far richer than impedance-SOH.

### 5.3 Aligning capacity onto the spectra (nearest-time join)
The capacity test and the EIS sweeps happen in the *same check-up* (within ~2 days) but are logged in
different files. We pair them with **`merge_asof(direction="nearest")`**: for each EIS spectrum, find
the capacity test **closest in time for that same cell**. A 3-day tolerance keeps every match inside
its own check-up.

Result: **100 % of spectra labeled**, median match gap **0.48 days**, max **2.22 days**. Saved as the
final Stage 0 output, `data/interim/eis_labeled.parquet`.

---

## Part 6 — Two subtleties we clarified (so you don't get confused later)

### 6.1 `soh_imp = 100 %` but `soh_cap ≈ 98 %` at the first check-up — not a contradiction
They use different reference frames:
- `soh_imp` is measured **against the cell's own first check-up**, so it is *forced* to 100 % there by
  definition (it is `z_ref_now / z_ref_init`, and at the first CU those are equal).
- `soh_cap` is measured **against a fixed beginning-of-life reference capacity** (~3.03 Ah for that
  cell), so it can start below 100 % — the cell genuinely holds ~98 % of rated capacity (normal
  manufacturing spread + a little formation loss).

So the first check-up spectrum is still our **baseline** for ΔZ — just call it "baseline," not
"pristine." The model predicts **absolute `soh_cap`** using ΔZ (change-since-start) **plus** absolute
features (R_ohm, R_ct…), so a ~98 % starting label is simply the honest truth, not a problem.

### 6.2 What `z_ref_init` is, and how we'll use it
`z_ref_init_mOhm` is a **single scalar** — the impedance magnitude `|Z|` at a fixed low reference
frequency (~1.5 Hz, the aging-sensitive region) at the cell's first check-up. It's what `soh_imp` is
built from. It is **not** used to build our ΔZ feature (ΔZ needs the full 28-point healthy spectrum,
which we already have). `z_ref_init` will serve only as a **cross-check** (our baseline spectrum should
agree with it at the low-frequency point) and as an optional normalization constant.

> **ΔZ design (to build in Stage 1):** for each `(cell, SoC, pass)`, take that cell's **first check-up
> spectrum** as `Z_baseline(f)`, then `ΔZ(f) = Z_baseline(f) − Z_now(f)` across all 28 frequencies.

---

## Part 7 — The final cleaned dataset: `data/interim/eis_labeled.parquet`

The single, model-ready Stage 0 output. **One row = one frequency point of one clean spectrum.**

- **Rows:** 1,102,304
- **Columns:** 29
- **Cells:** 228 (Calendar 48 / Cyclic 144 / Profile 36)
- **Spectra:** 39,368 (each exactly 28 frequency points)
- **Check-ups:** `cu_index` ranges 0–28 (varies per cell by how long it lived)
- **Label coverage:** `soh_cap` present on 100 % of rows (0 nulls)

### Column dictionary

| Column | Meaning |
|---|---|
| **Identity** | |
| `cell_id` | Full cell name, e.g. `cell_eisv2_P001_1_S01_C10` — the grouping key for splits |
| `pack` | Recipe code, `P001`…`P076` |
| `replicate` | Which of the 3 replicate cells (1/2/3) |
| `board`, `channel` | Physical circuit board / channel (test-rig location) |
| `param_id` | Recipe number (int, `P001`→1) — join key to metadata |
| `cell_key` | `P001_1_S01_C10` — shared key linking EIS ↔ capacity files |
| **Time axis** | |
| `cu_index` | Reconstructed check-up index (0,1,2,…) — the ageing order |
| `timestamp_s` | Unix timestamp of the spectrum |
| `cap_timestamp_s` | Timestamp of the matched capacity test (for QA of the alignment) |
| **Conditions** | |
| `soc_nom` | State of charge (10/30/50/70/90 %) |
| `is_rt` | 1 = room-temperature pass, 0 = operating-temperature pass |
| `temp_degC` | Actual measured temperature during the sweep |
| **Impedance (the raw signal for ECM)** | |
| `freq_Hz` | Test frequency (28 values per spectrum, 0.05 Hz – 10 kHz) |
| `Z_real_mOhm` | Real part Z′ (compensated) |
| `Z_imag_mOhm` | Imaginary part Z″ (compensated; sign convention to be locked at ECM time) |
| **Impedance-based health (secondary/sanity)** | |
| `soh_imp` | Impedance-SOH; 100 % at each cell's first CU by definition |
| `z_ref_init_mOhm` | Scalar reference \|Z\| at the first check-up (baseline) |
| `z_ref_now_mOhm` | Scalar reference \|Z\| at the current check-up |
| **Recipe metadata (why it ages)** | |
| `aging_type` | Calendar / Cyclic / Profile |
| `temp_setpoint_degC` | Recipe temperature setpoint (0/10/25/40 °C) |
| `temp_cat` | Raw temperature category (A/B/C/D) |
| `soc_idle_cat`, `soc_limits_cat`, `c_rate_cat`, `profile` | Recipe SoC/C-rate/profile category codes |
| **Quality** | |
| `valid` | Instrument quality flag (1 good / 0 bad) — per frequency point |
| **THE TARGET** | |
| `soh_cap` | **Capacity-based SOH (%)** — the label the model will predict |
| `cap_aged_est_Ah` | Estimated aged discharge capacity (Ah) behind `soh_cap` |

### Value ranges to be aware of
- `soh_cap`: **−35 % to 98 %.** Negative values are dead cells (capacity fell below the reference); tops
  out ~98 % (cells start slightly below rated — see §6.1).
- `soh_imp`: **−66 % to 112.5 %.** >100 % and <0 % are valid by design (own-reference definition).

---

## Part 8 — The code

All Stage 0 logic is reusable, under `src/prep/`:

| Module | Key functions | Does |
|---|---|---|
| `load_eis.py` | `parse_cell_id`, `assign_cu_index`, `process_cell_file`, `load_all_cells` | Clean one/all EIS files → spectra table |
| `metadata.py` | `load_recipe_table`, `join_recipe_metadata`, `build_enriched_table` | Parse recipe xlsx, safe join |
| `capacity.py` | `load_capacity_labels`, `attach_capacity_labels` | Extract `soh_cap`, nearest-time align |

The whole pipeline can be rebuilt from these three modules (see the last cell of notebook 02).

---

## Part 9 — Decisions log (consolidated)

| # | Decision | Why | Reversible? |
|---|---|---|---|
| 1 | Drop `seq_nr == 0` priming row | NaN impedance; leaves clean 28 points | yes |
| 2 | Reconstruct `cu_index` from timestamp gaps | No time axis exists in the raw files | yes |
| 3 | 48-hour gap threshold | Sits in the empty zone between within-CU (≤33 h) and between-CU (≥125 h) gaps | yes (configurable) |
| 4 | Keep-latest on duplicate spectra | Handles re-run passes uniformly | yes |
| 5 | Process 228 cells, skip 12 empty `P000` | Those files have no data | yes |
| 6 | Target = capacity-SOH, not impedance-SOH | Non-circular, physical, needed for RUL | yes (revisit at Stage 2) |
| 7 | `soh_cap` = check-up discharge capacity (`cyc_condition==2 & cyc_charged==0`) | Standardized, comparable, usable-capacity definition | yes |
| — | ΔZ baseline = each cell's first check-up spectrum, per `(cell, SoC, pass)` | Matches dataset's own convention; normalizes per-cell offset | yes |

---

## Part 10 — Known issues / deferred to Stage 1

- **Label cleaning:** `soh_cap` goes negative / bounces at end-of-life (dead cells), and there's the
  synchronized **CU-27 artifact**. Decide a clip/drop policy when building the ECM feature table.
- **`Z_imag` sign convention:** confirm and lock before ECM fitting (needed by the fitting library).
- **SoC / C-rate category decode:** `soc_*_cat`, `c_rate_cat` are still raw A/B/C/D codes; decode from
  the spreadsheet legends if/when the model needs their real values.

---

## What's next: Stage 1 — ECM fitting

With a clean, labeled, time-indexed dataset in hand, Stage 1 turns each impedance spectrum into
**physical parameters** (R_ohm, R_ct, CPE, ΔZ, …) by fitting an equivalent circuit — the *inputs* the
model will use to predict `soh_cap`. Planned order: lock the sign convention → Kramers–Kronig
validation → fit `L + R_ohm + (R_ct ∥ CPE) + Warburg` to one spectrum → scale up → assemble the
feature table joined to `soh_cap`.
