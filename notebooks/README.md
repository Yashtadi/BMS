# Notebooks — Exploratory Analysis

This folder holds Jupyter notebooks used to *understand* the data before writing any production
pipeline code. Nothing here is final — it's a workbench. Once a piece of logic is verified here, it
graduates into a proper reusable module under `src/`.

This README is written for a **complete beginner**. It walks through exactly what we did, in the
order we did it, and explains every column and term the first time it shows up.

---

## Background: what this data is

We use the **KIT comprehensive battery aging dataset** (RADAR4KIT, DOI `10.35097/1969`).

Researchers took **228 identical lithium-ion cells** (LG INR18650HG2 — same type used in power tools
and some EVs) and deliberately aged them over ~**600 days**. Each cell got a different "lifestyle"
(a *recipe*): some just sat on a shelf (calendar aging), most were charged/discharged over and over
(cyclic aging), a few followed a real driving profile (profile aging). The goal was to watch *how*
and *why* batteries wear out.

Every ~3 weeks each cell was paused for a **check-up (CU)** — a standardized health exam. During a
check-up the cell is measured with **EIS** (Electrochemical Impedance Spectroscopy): a technique that
sends tiny AC signals through the cell at many frequencies and measures how much the cell "resists"
each one. The result — an **impedance spectrum** — is a fingerprint of the cell's internal chemistry
that changes in tell-tale ways as the cell ages.

Each check-up measures the cell **twice**:
- **RT pass** — cell brought to room temperature (~25 °C), measured at 5 states of charge:
  10, 30, 50, 70, 90 % (stepping *up*).
- **OT pass** — cell returned to its real operating temperature (0/10/25/40 °C), measured again at the
  same 5 SoC points (stepping *down*).

So one complete check-up = **5 SoC × 2 passes = 10 impedance spectra**.

### The raw files
Raw EIS data lives in `data/raw/.../cell_eisv2/` as **240 CSV files, one per physical cell**. The
filename is the cell's fingerprint, e.g. `cell_eisv2_P001_1_S01_C10.csv`:
`P001` = recipe, `_1` = which of 3 replicate cells, `S01` = circuit board, `C10` = channel on that
board. That whole combination = **one physical cell tracked for its entire life**, and it is the
grouping key we must never split across train/test sets.

> **Note on the separators:** these raw CSVs use a **semicolon (`;`)** between columns, not a comma.
> That's why we load them with `pd.read_csv(path, sep=";")`.

---

## Notebook: `01_raw_schema_explore.ipynb`

**Goal:** understand the raw file format and solve the biggest problem the project README flagged —
**there is no ready-made "time axis"** telling us the order of check-ups. Without knowing which
measurement came first, second, third… we can't talk about aging *over time*, and RUL is impossible.
We work this out on one example file, `cell_eisv2_P001_1_S01_C10.csv`, before scaling to all cells.

### The columns we care about (raw schema)
The raw file has **22 columns** (the earlier "cleaned" CSV had only 8 and threw most of these away).
The ones that matter:

| Column | Plain-English meaning |
|---|---|
| `timestamp_s` | The exact moment of the measurement, as a **Unix timestamp** — the number of seconds since 1 Jan 1970. Just an integer, but `pd.to_datetime(x, unit="s")` turns it into a real date. This is the key that lets us order measurements in time. |
| `soc_nom` | **Nominal state of charge** — how full the cell was (10/30/50/70/90 %). |
| `is_rt` | 1 during the **room-temperature** pass, 0 during the **operating-temperature** pass. |
| `cyc_charged` | Whether SoC is being stepped *up* (1, charging) or *down* (0, discharging). **Not** a cycle counter. |
| `valid` | The instrument's own quality flag: 1 = good measurement, 0 = bad. |
| `seq_nr` | The index of a frequency point within one sweep (0 through 28). |
| `freq_Hz` | The AC frequency used for that row. |
| `z_re_comp_mOhm` / `z_im_comp_mOhm` | The **compensated** real and imaginary impedance (in milliohms) — the "good", temperature/hardware-corrected numbers we'll actually model. |
| `soh_imp` | An impedance-based State of Health estimate the instrument records. |

---

### Step 1 — How many spectra are in this file, and how big is each?

We grouped rows by `(timestamp_s, soc_nom, is_rt)` — the idea being that one unique combination of
"moment + charge level + which pass" should equal one single impedance sweep — and counted the rows in
each group:

```python
spectra = df.groupby(["timestamp_s", "soc_nom", "is_rt"]).size()
```

**Output:** 291 spectra, each of size **29**. We also printed `df["cyc_charged"].unique()` → `[1 0]`.

**What this told us:** the grouping works (we get clean, equal-sized spectra), *but* each spectrum is
**29 rows, not the 28 we expected.**

### Step 2 — Why 29 rows instead of 28?

Looking back at a raw row dump, the very first row of every sweep has `seq_nr == 0`, a stray high
frequency, and **`NaN`** (empty) values in the compensated impedance columns. It's a throwaway
"priming" reading the instrument takes before the real 28-point sweep (`seq_nr` 1–28).

> **Decision #1:** drop `seq_nr == 0` (or equivalently filter on `valid`) so every spectrum is a clean
> **28 real frequency points**.

### Step 3 — Turning timestamps into real dates and measuring the gaps

`timestamp_s` on its own is an unreadable big integer. We converted the unique spectrum timestamps to
real dates and, crucially, measured the **time gap between each spectrum and the one before it**:

```python
timestamps = sorted(df["timestamp_s"].unique())
dates = pd.to_datetime(timestamps, unit="s")
gaps_hours = pd.Series(timestamps).diff() / 3600
big_gaps = gaps_hours[gaps_hours > 24*5]     # gaps longer than 5 days
```

**Output:** dates ran from Oct 2022 onward (consistent with a ~600-day experiment). The gaps split
into two very separate worlds:
- **Small gaps (~1–2.5 hours):** the time between consecutive SoC steps *inside* one check-up.
- **Big gaps (~125 to ~980 hours):** the ~3-week waits *between* check-ups. There were **28** of them.

**Why this matters:** because nothing falls in the empty middle ground, we can use a gap size to
tell "same check-up" from "new check-up." That's the trick that reconstructs the missing time axis.

### Step 4 — Reconstructing the check-up index ("sessionization")

The technique is called **sessionization** (the same idea used to split website clicks into visits):
sort everything by time, mark any gap bigger than a threshold as "the start of a new session," then
run a **cumulative sum** of those marks to get an increasing ID. Here that ID *is* the check-up number.
We wrote `assign_cu_index(timestamps, gap_threshold_hours)` to do exactly this, first with a
**24-hour** threshold, and counted spectra per check-up:

**Output (24 h):** most check-ups had exactly **10** spectra (great — that's 5 SoC × 2 passes). But a
few were off: CU 0 had 11, CU 3 had 15, CU 11 and CU 12 had 5 each, and the final CU had 5. We
investigated every anomaly instead of ignoring them.

### Step 5 — Investigating the anomalies (this is where the real learning happened)

We printed the exact timestamps, dates, and gaps around each suspicious check-up. To do that on the
*full* dataframe we first had to attach the check-up number to every row. A subtle bug appeared here:
`cu_sorted` had one entry per **spectrum** (291) while `df` had one row per **frequency point** (8439),
so a direct mask failed. The fix was to map by timestamp value, not by position:

```python
ts_to_cu = pd.Series(cu_sorted.values, index=sorted_ts.values)
df["cu"] = df["timestamp_s"].map(ts_to_cu)
```

What each anomaly turned out to be:

- **CU 11 / CU 12 (a real check-up wrongly split in two).** The gap between the RT cluster and the OT
  cluster was **31.75 hours** — longer than our 24 h cutoff, so the threshold sliced one check-up into
  two halves of 5. The likely cause: swinging the cell to an extreme operating temperature (0 °C or
  40 °C) needs extra soak time. **This is a threshold problem, not real data structure.**

- **CU 3 (15 spectra).** All 15 happened within ~42 hours, so it isn't two merged check-ups. Grouping
  by `(soc_nom, is_rt, cyc_charged)` showed only the normal 10 combinations — but the **entire OT pass
  was measured twice**, and checking the `valid` flag showed *both* copies were good (min=max=1). So
  it's a **genuine duplicate**, not a failed-and-retried measurement.

- **CU 0 (11 spectra).** One extra measurement at the very start of the cell's life — most likely an
  initial "as-received" characterization before the regular schedule began.

- **Last CU (5 spectra).** Preceded by a real ~41-day gap, so it's a real check-up — the experiment
  just ended after only one pass.

### Step 6 — Fixing the threshold: 24 h → 48 h

Every genuine between-check-up gap we saw was **≥ 125 hours**, and the false-split gap was **31.75
hours**. So any threshold between ~32 h and ~125 h fixes the split without risking merging two real
check-ups. We chose **48 hours** and re-ran:

**Output (48 h):** CU 11 and CU 12 merged back into one clean check-up of 10; the total number of
check-ups dropped by exactly 1; and CU 0 (11), CU 3 (15), and the final CU (5) were untouched — as
expected, since those are real edge cases, not threshold artifacts.

### Step 7 — Decoding `cyc_charged`

While grouping we noticed `cyc_charged` tracks perfectly with `is_rt`: charged = 1 during the RT pass
(SoC steps up), 0 during the OT pass (SoC steps down). So it is **not** a cycle counter — it's a
charging-direction flag, matching the protocol's "RT steps up, OT steps down."

---

## Decisions locked in from this notebook

1. **Time axis:** reconstruct the check-up index by sessionizing on `timestamp_s` with a **48-hour
   gap threshold**. *(Provisional — still to be re-validated on more cells.)*
2. **Priming row:** drop `seq_nr == 0`; keep the 28 real frequency points per spectrum.
3. **Duplicate spectra:** when the same `(cell, CU, SoC, pass)` has more than one *valid* spectrum,
   **keep the latest** (most recent timestamp).
4. **Quality filter:** trust the instrument's `valid` flag as the first-line filter (a Kramers–Kronig
   check can be layered on later).

All four are reversible choices, recorded here so a future iteration can revisit them.

### Step 8 — Verifying the 48-hour threshold across many cells

A threshold tuned on one cell proves nothing on its own, so we re-ran the reconstruction on a spread
of cells across different recipes and check-up counts (P001, P010, P030, P076). For each we measured
the two numbers that matter:
- **`max_within_h`** — the largest gap that occurs *inside* a single check-up.
- **`min_between_h`** — the smallest gap that occurs *between* two check-ups.

**Output:** across every real cell, `max_within_h` stayed between **12.9 and 32.5 hours**, while
`min_between_h` stayed between **125.3 and 128.9 hours**. There is a wide empty canyon between ~32.5 h
and ~125 h, and **48 h sits safely inside it for every cell.** The check-up cadence is also strikingly
regular (the smallest between-check-up gap is always ~5.2 days). Conclusion: the 48-hour threshold
generalizes — safe to promote into `src/prep/`.

> **A robustness fix we made here:** the original `assign_cu_index` set the first element with
> `is_new_cu.iloc[0] = False`, which **crashes on an empty Series** ("iloc cannot enlarge its target
> object"). We replaced it with `(...).fillna(False)`, which handles the first element (whose gap is
> always `NaN`) cleanly and never crashes on empty input.

### Step 9 — Counting the real cells (and finding the empty ones)

The crash above was actually a *clue*: the very first file we sampled, a `P000` cell, was **empty**.
So we scanned all 240 files and counted how many have any data:

**Output:** **228 non-empty (real) cells + 12 empty files**, and the 12 empty files are *exactly*
`cell_eisv2_P000_0_S20_C00.csv` through `C11.csv` — a single reserved reference slot (`P000`, board
`S20`) that was never populated.

This reconciles a mystery from the project README: the old "cleaned" CSV had only **216** cells
because it dropped these 12 empty `P000` files **and** lost another 12 real cells to Excel's
1,048,575-row truncation. Reading the raw files recovers the full **228** — matching the dataset's
official cell count exactly.

> **Decision #5:** the loader processes the **228 non-empty cells** and skips the 12 empty `P000`
> files.

## Verified — ready to build

Everything needed for the Stage 0 loader is now confirmed on real data: 228 cells, the 48-hour CU
reconstruction, the priming-row drop, the keep-latest-valid rule, and the `valid` quality filter. The
next step is to lift this notebook logic into a reusable module under `src/prep/`.

---

## Notebook: `02_raw_schema_explore.ipynb` — assembling the full Stage 0 dataset

This notebook takes the verified logic and builds the complete, model-ready Stage 0 table. The
reusable code lives in three modules under `src/prep/`:
- **`load_eis.py`** — clean one EIS cell file (`process_cell_file`) and all of them (`load_all_cells`).
- **`metadata.py`** — parse the recipe/parameter spreadsheet and join it on.
- **`capacity.py`** — extract the capacity-SOH target and align it to the spectra by time.

### Step 1 — Build the combined spectra table
`load_all_cells` runs the single-file cleaner over all 240 files, skips the 12 empty `P000`
placeholders, and concatenates the 228 real cells into one table saved as
`data/interim/eis_spectra.parquet`. **Parquet, not CSV/Excel** — Parquet has no row limit, so the
~1.1 M-row table is never truncated (the exact bug that ruined the earlier "cleaned" file). Every
spectrum came out to exactly 28 frequency points.

### Step 2 — Validate against physics (before trusting anything)
We plotted **SOH vs check-up index** and a **Nyquist plot** of one spectrum:
- The Nyquist plot showed a textbook EIS shape (high-frequency branch, charge-transfer semicircle,
  diffusion tail) — confirming the impedance columns and spectrum grouping are correct. *(Note: this
  dataset's `Z_imag` sign is flipped vs. the classic −Z″ convention; to be locked at ECM time.)*
- SOH-over-life trends downward, confirming `cu_index` is a real time axis. But for gentle recipes the
  impedance-based `soh_imp` barely moves (~100 → 97 %), and a **synchronized dip at check-up 27**
  across all cells flags a systematic measurement artifact. Both reinforce that impedance-SOH is a
  weak, noisy label — motivating the capacity-based target below.

### Step 3 — Join recipe metadata (why each cell ages)
The KIT parameter spreadsheet (`Cycling Experiment Cell Overview…xlsx`) maps each recipe (`P`-code) to
its aging story. It's messy — a title banner, newline-filled headers, and unrelated sub-tables below
the 76 recipes. We read it with `header=1`, normalised the column names, and **kept only rows whose
`aging_type` is Calendar/Cyclic/Profile** (which uniquely selects the 76 real recipes). The temperature
category was decoded **A/B/C/D → 0/10/25/40 °C**.

> **Bug caught here — the silent one-to-many join.** A first attempt filtered by `param_id` range and
> let in 11 junk rows with duplicate ids. On `merge`, pandas duplicated every spectrum from packs
> P002–P012 *without any error*. We now **assert `param_id` is unique before joining and that the row
> count is unchanged after** — the essential guard against this classic pitfall. Result: 76 recipes
> (16 Calendar / 48 Cyclic / 12 Profile), 228 cells, 0 unmatched. Saved as
> `data/interim/eis_enriched.parquet`.

### Step 4 — The capacity-SOH target (the real label)
The model's target must be **non-circular**: `soh_imp` is derived from impedance, and so are our ECM
features, so predicting one from the other would be predicting a thing from itself. The independent,
physically-meaningful target is **capacity-based SOH (`soh_cap`)**, measured by a full charge/discharge
in a *separate* file collection, **`cell_eocv2`**.

Understanding that data took two careful filters:
- **`cyc_condition == 2`** selects the standardized **check-up capacity test** (done at ~25 °C RT),
  *not* `cyc_condition == 1` which is the per-cycle **operational** capacity. Missing this filter
  inflated a cyclic cell from ~29 to thousands of measurements — caught by sanity-checking
  measurements-per-cell.
- **`cyc_charged == 0`** selects the **discharge** capacity (usable capacity = the standard SOH),
  vs. `cyc_charged == 1` (charge capacity, slightly higher due to coulombic inefficiency).

`load_capacity_labels` applies both, giving one discharge-SOH per (cell, check-up): 3,999 measurements
across 228 cells, ~1 per check-up, all at ~25 °C. Capacity-SOH shows a **strong** aging signal (100 %
→ 0 %), far richer than impedance-SOH.

### Step 5 — Align capacity onto the spectra by nearest time
`attach_capacity_labels` uses **`merge_asof(direction="nearest")`** to pair each EIS spectrum with the
capacity test **closest in time for the same cell** — because the capacity test and EIS sweeps happen
in the same check-up (within ~2 days). A 3-day tolerance keeps every match inside its own check-up.
Result: **100 % of spectra labeled**, median time-gap **0.48 days**, max **2.22 days**. Saved as the
final Stage 0 output, `data/interim/eis_labeled.parquet`.

### Health-target decision (locked in)
**Target = discharge-capacity check-up SOH (`soh_cap`).** Impedance-`soh_imp` is kept only as a
secondary sanity check. Rationale: non-circular, physically meaningful (usable capacity), and the basis
for RUL (end-of-life is defined at a capacity threshold, e.g. 80 %). *Reversible — revisit at Stage 2
if needed.*

### Still open (small, deferred to Stage 1)
- **Label cleaning:** `soh_cap` goes negative / bounces at end-of-life (dead cells) and there's the
  CU-27 artifact — decide clip/drop policy when building the ECM feature table.
- **Sign convention:** confirm and lock `Z_imag` sign before ECM fitting.

## Stage 0: complete

`data/interim/eis_labeled.parquet` — 228 cells, ~39 k spectra, uncapped, time-indexed, metadata-joined,
and labeled with capacity-SOH. Every later stage builds on this file. Next: **Stage 1 — ECM fitting.**
