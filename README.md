# EIS-Driven Battery SOH/RUL Estimation with Self-Healing Drift Adaptation

A research framework for estimating battery **State of Health (SOH)** and **Remaining Useful Life (RUL)** from **Electrochemical Impedance Spectroscopy (EIS)** measurements. Raw impedance spectra are first reduced to a small set of physically interpretable **Equivalent Circuit Model (ECM)** parameters. Those parameters — together with operating conditions — feed a machine-learning model that learns how battery health evolves and predicts SOH/RUL with a confidence estimate. A **drift-detection and self-healing (SHML) loop** keeps the deployed model accurate as cells age and the impedance signature shifts over time.

> **Status:** **Stage 0 (data preparation) is complete** — the raw KIT dataset has been rebuilt into a clean, uncapped, time-indexed, metadata-joined, capacity-labeled table (`data/interim/eis_labeled.parquet`). **Stage 1 (ECM feature extraction) is next.** Downstream components, model choices, and formulas are deliberately left open where a decision has not yet been made (see [Open Decisions](#11-open-decisions)). The framework is designed to be revisited and improved at every stage.
>
> 📄 For the full, step-by-step account of the data work — every decision and the evidence behind it — see [`data_exploration.md`](data_exploration.md).

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Why EIS + ECM + ML](#2-why-eis--ecm--ml)
3. [Dataset](#3-dataset)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Stage 0 — Data Preparation](#5-stage-0--data-preparation)
6. [Stage 1 — Feature Extraction (EIS → ECM)](#6-stage-1--feature-extraction-eis--ecm)
7. [Stage 2 — Health Model (ML)](#7-stage-2--health-model-ml)
8. [Stage 3 — Drift Detection & Self-Healing Loop](#8-stage-3--drift-detection--self-healing-loop)
9. [Drift Score](#9-drift-score)
10. [Evaluation & Validation](#10-evaluation--validation)
11. [Open Decisions](#11-open-decisions)
12. [Roadmap](#12-roadmap)
13. [References](#13-references)

---

## 1. Motivation

Battery health is not directly measurable in the field. Capacity tests are slow and disruptive, and simple voltage/current proxies miss the underlying electrochemical changes that drive ageing. EIS probes a cell across a wide frequency range and separates distinct physical processes — ohmic conduction, charge transfer, double-layer behaviour, and diffusion — by the timescale on which they respond. These processes carry a rich, early signature of degradation.

The goal of this project is a health-estimation system that is:

- **Physically grounded** — features map to real electrochemical quantities, not opaque statistics.
- **Uncertainty-aware** — every SOH/RUL estimate carries a confidence value, because a confidently wrong estimate in a safety-critical system is dangerous.
- **Lifelong** — the model does not silently degrade as cells age out of the training distribution; it detects drift and adapts.

## 2. Why EIS + ECM + ML

A purely data-driven model trained on raw spectra can be accurate but brittle and hard to interpret. A purely physics-based ECM is interpretable but cannot, by itself, capture the full nonlinear way that parameters evolve with SOH, State of Charge (SOC), and temperature. This framework is a **gray-box** combination:

- The **ECM** compresses each spectrum into a handful of meaningful parameters (resistances, capacitive behaviour, diffusion). This is dimensionality reduction with physical meaning.
- The **ML model** learns the empirical *evolution law* — how those parameters and their deviations relate to SOH/RUL across operating conditions.

This split keeps the inputs interpretable and the predictor flexible, which generally generalises better and is easier to trust than either approach alone.

## 3. Dataset

The data comes from the **KIT comprehensive battery aging dataset** (RADAR4KIT, DOI `10.35097/1969`; published in *Scientific Data*, 2024 — *"Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell"*).

**228 commercial LG INR18650HG2 cells** — NMC cathode, graphite + SiO anode, 3.0 Ah nominal — were aged for ~600 days under **76 parameter sets** (16 calendar, 48 cyclic, 12 driving-profile/WLTP aging), 3 replicate cells per set, across four temperatures (0 / 10 / 25 / 40 °C), three SoC windows, and varying charge/discharge rates. Roughly every three weeks each cell was paused for a standardized **check-up (CU)**: brought to room temperature, given a full capacity test, then stepped through SoC = 10/30/50/70/90 % with an EIS sweep + current-pulse at each stop (the **RT pass**); then returned to its operating temperature and the EIS + pulse sequence repeated at the same five SoC points (the **OT pass**). This is why every cell has a tight ~25 °C measurement cluster (RT) plus a cluster at its own native temperature (OT).

### The three "lifestyles" (aging types)
Every cell falls into one of three regimes, and this is the single most important thing to know about any given cell, because it explains why it's degrading:
**1) Calendar aging (16 of the 76 groups)** — the cell just sits there, held at a constant voltage (constant SoC), at one of four temperatures. No cycling at all between check-ups. This simulates a battery sitting in storage or barely used. The stress factors here are purely temperature and how charged the cell is kept.
**2) Cyclic aging (48 of the 76 groups, the largest chunk)** — the cell is continuously charged and discharged back and forth between two SoC limits (e.g., 0–100%, or 10–90%), at a chosen charge rate and discharge rate, at one of four temperatures. This simulates a battery in constant heavy use.
**3) Profile aging (12 of the 76 groups)** — instead of a simple back-and-forth, the cell is discharged following a real driving-cycle power profile (WLTP — the standard European driving test cycle used for EV range testing), then recharged. This simulates an EV battery in actual use.
Every group also gets one of four temperatures: 0°C, 10°C, 25°C, 40°C — these are physically four separate temperature-controlled oil baths ("pools") the cells sit in.

### What happens during a check-up (CU)
Roughly every three weeks, a cell's regular charge/discharge routine is paused, and it goes through a standardized health check, always in the same order: first its temperature is brought to room temperature (RT, ~25°C) regardless of what temperature it normally lives at; then its capacity is measured with one full 0→100→0% charge-discharge; then, staying at RT, it's stepped up through SoC = 10%, 30%, 50%, 70%, 90%, and at each stop an EIS measurement (impedance spectrum) plus a current-pulse test is taken. Once that's done, the cell is brought back to its real operating temperature (OT — 0, 10, 25, or 40°C, whichever it normally lives at), and the same EIS + pulse test sequence is repeated at the same five SoC points, but now sweeping back down 90→70→50→30→10%. Only after both passes are done does the cell resume its normal aging routine. This is why every cell, no matter its real living temperature, has a cluster of measurements at a tight ~25°C (the RT pass) and a separate cluster at its own native temperature (the OT pass) — exactly the pattern that showed up in your data as the cycles/is_rt flag.


| `source_file` | — | Cell fingerprint: `P###` (recipe 1–72) `_#` (replicate 1–3) `S##` (board 1–19) `C##` (channel 0–11). One combination = one physical cell — **the grouping key for all splits**. |
| `freq_Hz` | Hz | Test frequency for this row. |
| `Z_real_mOhm` | mΩ | Real part Z′, temperature/hardware-**compensated** (official `z_re_comp_mOhm`). |
| `Z_imag_mOhm` | mΩ | Imaginary part Z″, compensated (`z_im_comp_mOhm`). |
| `SoC_pct` | % | One of the five checkpoints (10/30/50/70/90). |
| `temp_degC` | °C | Actual measured temperature during the spectrum (with real-hardware noise/self-heating). |
| `SoH_pct` | % | **Impedance-based SOH, reset to 100 % at the first valid CU for each (SoC, temperature) condition** (0 % ⇔ impedance tripled). |
| `cycles` | — | **`is_rt` flag**: 1 = RT pass, 0 = OT pass. Not a cycle count. |

**Verified against the file (profiling confirms your understanding):**

### Stage 0 output — the working dataset

The original teammate-cleaned `eis_cleaned.csv` was **discarded**: it had been truncated at the Excel
1,048,575-row cap (only 216 of 228 cells) and stripped of most useful columns. We rebuilt everything
from the raw KIT files into **`data/interim/eis_labeled.parquet`** — the single working dataset for all
downstream stages. **One row = one frequency point of one clean spectrum.**

- **228 cells** (Calendar 48 / Cyclic 144 / Profile 36) · **39,368 spectra** · **1,102,304 rows** · 28 frequency points per spectrum.
- **Reconstructed time axis** (`cu_index`, 0–28) — the check-up order, absent from the raw files, rebuilt from timestamps.
- **Recipe metadata joined** (aging type, temperature setpoint, SoC/C-rate categories) per cell.
- **Capacity-based SOH (`soh_cap`) attached to 100 % of rows** as the model target.

Key columns (full dictionary in [`data_exploration.md`](data_exploration.md)):

| Column | Unit | Meaning |
|---|---|---|
| `cell_id` (+ `pack`,`replicate`,`board`,`channel`,`param_id`) | — | Cell fingerprint `P###_#_S##_C##`. One cell = **the grouping key for all splits**. |
| `cu_index` | — | Reconstructed check-up index (the ageing/time axis). |
| `freq_Hz` | Hz | Test frequency (28 per spectrum, 0.05 Hz – 10 kHz). |
| `Z_real_mOhm` / `Z_imag_mOhm` | mΩ | Compensated real Z′ / imaginary Z″ (sign convention to be locked at ECM time). |
| `soc_nom` | % | One of the five checkpoints (10/30/50/70/90). |
| `is_rt` | — | 1 = RT pass (~25 °C), 0 = OT pass (native temperature). |
| `temp_degC` | °C | Actual measured temperature during the spectrum. |
| `aging_type`, `temp_setpoint_degC` | — | Recipe metadata: why the cell ages. |
| `soh_imp`, `z_ref_init_mOhm`, `z_ref_now_mOhm` | % / mΩ | Impedance-based SOH and its reference impedances (secondary/sanity). |
| **`soh_cap`** | % | **Capacity-based SOH — the prediction target.** |

**Dataset caveats — all resolved in Stage 0** (details in [`data_exploration.md`](data_exploration.md)):

- ~~Excel truncation~~ → rebuilt from raw to Parquet (uncapped); all 228 cells recovered.
- ~~No time/age axis~~ → `cu_index` reconstructed by sessionizing timestamps (48-hour gap threshold).
- ~~Missing recipe metadata~~ → joined from the KIT parameter spreadsheet by `P`-code.
- ~~Health-target undecided~~ → capacity-SOH chosen and attached (see [§7](#7-stage-2--health-model-ml)).

## 4. Pipeline Overview

<img width="2064" height="2304" alt="pipeline_diagram" src="https://github.com/user-attachments/assets/e763ecfe-834e-4049-a29c-1dfb012d43f4" />
A rendered version of this pipeline is in [`pipeline_diagram.png`](pipeline_diagram.png).

## 5. Stage 0 — Data Preparation

> ✅ **COMPLETE.** Output: `data/interim/eis_labeled.parquet`. Reusable code in [`src/prep/`](src/prep/); full narrative in [`data_exploration.md`](data_exploration.md).

1. ✅ **Regenerate an uncapped dataset.** Rebuilt from the raw KIT files to Parquet — all **228 cells / 76 recipes** recovered (the old CSV had only 216). Empty `P000` placeholder files (12) skipped.
2. ✅ **Reconstruct the time/age axis.** `cu_index` rebuilt by sessionizing spectrum timestamps with a **48-hour gap threshold** (verified across recipes/temperatures).
3. ✅ **Join recipe metadata.** Aging type, temperature setpoint (A/B/C/D → 0/10/25/40 °C), SoC and C-rate categories joined per `P`-code — with an explicit guard against a one-to-many join duplication bug.
4. ✅ **Choose the health target.** **Capacity-based SOH (`soh_cap`)** selected (non-circular), extracted from the check-up discharge capacity test (`cell_eocv2`, `cyc_condition==2 & cyc_charged==0`) and aligned to each spectrum by nearest time (100 % matched).
5. ⬜ **Clean labels and define grouping.** Grouping key defined (`cell_id`). Remaining: `soh_cap` end-of-life noise / negatives and the CU-27 artifact — deferred into Stage 1 feature-table construction.

## 6. Stage 1 — Feature Extraction (EIS → ECM)

Each spectrum (one cell, one SoC, one pass, one check-up) is processed as follows.

**6.1 Validate first (Kramers–Kronig).** EIS assumes the system is linear, causal, stable, and time-invariant during the sweep. Low-frequency points take a long time to acquire, during which a cell can drift, invalidating the spectrum. A Kramers–Kronig consistency check is run before fitting; spectra with large systematic residuals are flagged or rejected so that bad data never reaches the model.

**6.2 Fit an Equivalent Circuit Model.** A Randles-type circuit is fit to each validated spectrum. Because the chemistry is now known (NMC / graphite-SiO 18650), a concrete starting topology is recommended (see [Open Decision 1](#11-open-decisions)):

```
L  +  R_ohm  +  ( R_ct ∥ CPE )  +  Warburg                         # baseline
L  +  R_ohm  +  ( R_sf ∥ CPE_sf ) + ( R_ct ∥ CPE_dl ) + Warburg    # if a surface-film/SEI arc resolves
```

The series inductance `L` accounts for the high-frequency inductive tail visible in the data (Z″ < 0 at kHz). Use DRT to decide how many R∥CPE arcs the spectra actually justify before committing.

**6.3 Extracted features.** Computed per spectrum and used as model inputs:

| Feature | Symbol | Physical meaning |
|---|---|---|
| Ohmic resistance | `R_ohm` | High-frequency real-axis intercept — electrolyte, contacts, current collectors. Grows with ageing. |
| Charge-transfer resistance | `R_ct` | Mid-frequency semicircle diameter — electrode-reaction kinetics. Grows as the interface degrades. |
| CPE exponent | `n` (and magnitude `Q`) | Depression of the semicircle — surface inhomogeneity / roughness of the double layer. |
| Health deviation | `ΔZ = Z_healthy − Z_measured` | Per-frequency (and aggregated) deviation of the present spectrum from a fresh-cell baseline. A direct, physics-anchored health signal. |
| Warburg coefficient | `σ` (optional) | Low-frequency diffusion behaviour (~45° tail) — solid-state/electrolyte transport. |

Operating conditions (`SoC_pct`, `temp_degC`) are carried alongside, because impedance depends strongly on both.

**On ΔZ:** define the healthy reference as the **first valid check-up for the same (cell, SoC, pass)** — this matches the dataset's own per-condition SoH reset, so `ΔZ` and `SoH_pct` share a consistent baseline. Prefer the **RT pass** (all cells at ~25 °C) when comparing features *across* cells, since it removes the operating-temperature confound; use OT-pass features when modelling native-temperature behaviour explicitly.

## 7. Stage 2 — Health Model (ML)

The model maps `[ECM features + SoC + temp + (aging metadata)] → SOH` (and RUL vs. an end-of-life threshold) **with a confidence estimate**.

> **Health target — DECIDED (Stage 0): capacity-based SOH (`soh_cap`).** `soh_imp` is *derived from impedance*, and so are the ECM features, so predicting it from them is partly circular — deceptively accurate but hollow. We instead train against the **measured capacity fade** from each check-up's full charge/discharge (`cell_eocv2`), which is independent of impedance and the basis for RUL. `soh_cap` is already extracted and attached to every spectrum in `eis_labeled.parquet`; `soh_imp` is retained only as a secondary sanity check.

Build order:

1. **No-leakage split first** (leave-one-cell-out; additionally hold out by **aging regime** and **temperature** to test robustness across conditions).
2. **Baseline model.** Start simple and interpretable (tree ensemble — XGBoost/LightGBM/RF — on the tabular feature table). Model family is an [open decision](#11-open-decisions).
3. **Uncertainty.** Add prediction intervals / variance (quantile, ensemble, or a UQ-native model). Confidence is a first-class requirement.
4. **Attribution.** Use feature importance / SHAP to confirm the model leans on physically meaningful features (`R_ct`, `ΔZ`, `R_ohm`), not artifacts.

Outputs must respect physics: SOH near-monotonic non-increasing over life; resistances grow/hold; uncertainty widens out-of-distribution.

## 8. Stage 3 — Drift Detection & Self-Healing Loop

A model trained today will eventually face cells whose impedance signature has drifted beyond what it has seen. The system detects this and repairs itself rather than silently degrading.

**8.1 Source the "aged" EIS.** This dataset already spans ~600 days of repeated check-ups, so the *"cell aged a few months later"* spectra are **real, not synthetic** — later-CU spectra of the same cells (or held-out late-life recipes) can serve directly as the drift test set. Synthetic generation becomes *optional*, for stress-testing beyond the observed range (see [Open Decision 3](#11-open-decisions)).

**8.2 Re-extract parameters.** New spectra pass through the same Stage 1 pipeline to produce a new ECM feature set (new `R_ohm`, `R_ct`, `ΔZ`, CPE exponent, etc.).

**8.3 Compute a drift score** quantifying the shift of ECM parameters (and/or model residuals) relative to the reference distribution (see [Drift Score](#9-drift-score)).

**8.4 Compare to threshold and act:**

- **Drift score < threshold** → distribution stable; keep the current model.
- **Drift score ≥ threshold** → enter the **Self-Healing (SHML) loop**:
  1. **Update ECM parameters** for the new operating regime.
  2. **Update the evolution laws** the model relies on.
  3. **Fine-tune or fully retrain** in the cloud — fine-tune for mild drift, full retrain for severe drift, cut-off governed by the drift-score magnitude.
  4. **Deploy a lightweight model back to the edge** device, replacing the previous version.

This mirrors the **Self-Healing Machine Learning** pattern: autonomously detect degradation, diagnose its severity, choose a remediation proportional to the damage, and adapt — heavy training in the cloud, a compact model on-device.

## 9. Drift Score

The drift score reduces "how far have the parameters moved?" to one comparable number. The exact formula is an [open decision](#11-open-decisions); leading candidates that fit this framework:

- **Population Stability Index (PSI)** — bins a feature's distribution now vs. baseline and sums the relative-entropy contribution. Common reading: `< 0.1` no meaningful shift, `0.1–0.25` moderate, `> 0.25` significant.
- **KL divergence** — relative entropy of the new parameter distribution against baseline (PSI is essentially a symmetrised, binned KL).
- **Mahalanobis distance** — distance of the new ECM parameter vector from the healthy/baseline cloud, covariance-aware; a clean single multivariate score with a statistical threshold.
- **Normalised parameter drift** — a weighted sum of per-parameter relative changes, e.g. `Σ wᵢ · |θᵢ − θᵢ⁰| / |θᵢ⁰|`; easy to map onto fine-tune vs. retrain bands.
- **Sequential detectors (Page–Hinkley / ADWIN)** — run on the stream of model residuals to raise a discrete alarm; useful for continuous online monitoring.

A practical design uses a **two-band threshold**: a lower band that triggers fine-tuning and an upper band that triggers full retraining.

## 10. Evaluation & Validation

- **No-leakage splits.** Group by physical cell (leave-one-cell-out); additionally hold out by aging regime and temperature. Never random-split rows of a sweep/cell.
- **Metrics.** Report SOH/RUL error (MAE/RMSE) *with* prediction-interval coverage and calibration; mean ± std across folds/seeds, not a single split.
- **Plausibility checks.** Verify monotonic SOH, non-decreasing resistance, and that uncertainty grows out-of-distribution before trusting any number.
- **Drift-loop validation.** Confirm the drift score rises on later-life / out-of-regime data and that fine-tune/retrain recovers accuracy on the shifted distribution.

## 11. Open Decisions

These choices are **not yet final** and are expected to evolve. Tracked here so the design stays honest about what is settled and what is not.

> **Resolved in Stage 0:** *Health target* → capacity-based SOH (`soh_cap`). *Cell chemistry* → NMC / graphite-SiO (known). *Aged-EIS source* → the dataset's own later check-ups (real, not synthetic).

1. **Which ECM circuit?** Chemistry is now **known** (NMC / graphite-SiO, LG INR18650HG2). Recommended starting point: `L + R_ohm + (R_ct ∥ CPE) + Warburg`, with an optional second `R∥CPE` for the surface-film/SEI arc if DRT shows it resolves. Still to confirm: number of arcs, Warburg type (semi-infinite vs. finite-length), and whether the SEI branch is identifiable at all SoC/temperature points.
2. **Which ML model for SOH?** XGBoost / Random Forest / LightGBM / UQ-native alternatives. Weigh accuracy, calibrated confidence, and edge-deployability.
3. **How is the "aged" EIS obtained?** The dataset's own later check-ups provide **real** aged spectra, so the primary path is to use held-out late-life data rather than synthesise. Still open: whether (and how) to *additionally* synthesise out-of-range spectra — parameter perturbation along ageing trends, physics-based impedance simulation, or measured-spectrum augmentation — to stress-test the drift loop.
4. **What formula for the drift score?** PSI / KL / Mahalanobis / normalised drift / sequential detector, plus the threshold band(s) separating keep / fine-tune / retrain.

## 12. Roadmap

The framework is explicitly **iterative**. Expected progression:

0. ✅ **Data prep (DONE)** — uncapped dataset rebuilt, time axis (`cu_index`) reconstructed, recipe metadata joined, capacity-SOH target chosen and attached → `data/interim/eis_labeled.parquet`. See [`data_exploration.md`](data_exploration.md).
1. Per-cell grouping (done) and label cleaning (end-of-life `soh_cap` noise, CU-27 artifact — pending).
2. K–K validation + ECM fitting; baseline feature table.
3. First health model with uncertainty; no-leakage evaluation.
4. Drift score implementation and threshold calibration on real later-life data.
5. Self-healing loop (fine-tune/retrain trigger + edge export).
6. Iterate: revisit circuit topology, features, model family, drift score, and thresholds as results come in.

Every stage above is a candidate for improvement; the architecture is meant to be refined repeatedly rather than frozen after a first pass.

## 13. References

- KIT comprehensive battery aging dataset (*Scientific Data*, 2024) — <https://www.nature.com/articles/s41597-024-03831-x>
- Dataset record (RADAR4KIT) — <https://radar.kit.edu/radar/en/dataset/kww7jv8ajuvchcah>
- Equivalent-circuit modelling of commercial Li-ion cells — <https://pmc.ncbi.nlm.nih.gov/articles/PMC7671193/>
- Guide to equivalent-circuit fitting for impedance & battery state estimation — <https://www.sciencedirect.com/science/article/pii/S2352152X2303788X>
- Concept-drift detection methods overview — <https://ai-infrastructure.org/8-concept-drift-detection-methods/>
- KL-divergence drift detection — <https://link.springer.com/article/10.1007/s42488-024-00119-y>
- Self-Healing Machine Learning framework — <https://proceedings.neurips.cc/paper_files/paper/2024/file/4a86ec12e94ef1fe306362e7bdcd5894-Paper-Conference.pdf>
- ML-based digital twin for EV battery modelling (edge–cloud retraining) — <https://arxiv.org/pdf/2206.08080>