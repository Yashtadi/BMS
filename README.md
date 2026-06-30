# EIS-Driven Battery SOH/RUL Estimation with Self-Healing Drift Adaptation

A research framework for estimating battery **State of Health (SOH)** and **Remaining Useful Life (RUL)** from **Electrochemical Impedance Spectroscopy (EIS)** measurements. Raw impedance spectra are first reduced to a small set of physically interpretable **Equivalent Circuit Model (ECM)** parameters. Those parameters — together with operating conditions — feed a machine-learning model that learns how battery health evolves and predicts SOH/RUL with a confidence estimate. A **drift-detection and self-healing (SHML) loop** keeps the deployed model accurate as cells age and the impedance signature shifts over time.

> **Status:** Planning / research phase. This document describes the intended design. Components, model choices, and formulas are deliberately left open where a decision has not yet been made (see [Open Decisions](#open-decisions)). The framework is designed to be revisited and improved at every stage.

---


## Table of Contents

1. [Motivation](#1-motivation)
2. [Why EIS + ECM + ML](#2-why-eis--ecm--ml)
3. [Dataset](#3-dataset)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Stage 1 — Feature Extraction (EIS → ECM)](#5-stage-1--feature-extraction-eis--ecm)
6. [Stage 2 — Health Model (ML)](#6-stage-2--health-model-ml)
7. [Stage 3 — Drift Detection & Self-Healing Loop](#7-stage-3--drift-detection--self-healing-loop)
8. [Drift Score](#8-drift-score)
9. [Evaluation & Validation](#9-evaluation--validation)
10. [Proposed Repository Structure](#10-proposed-repository-structure)
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

The working dataset is `eis_cleaned.csv` — a long-format table where each row is one impedance point at one frequency for one measurement condition.

| Column | Meaning |
|---|---|
| `source_file` | Identifier of the originating measurement file / cell campaign |
| `freq_Hz` | Excitation frequency (Hz) |
| `Z_real_mOhm` | Real part of impedance, Z′ (mΩ) |
| `Z_imag_mOhm` | Imaginary part of impedance, Z″ (mΩ) |
| `SoC_pct` | State of charge at measurement (%) |
| `temp_degC` | Cell temperature (°C) |
| `SoH_pct` | State of health label (%) |
| `cycles` | Cycle index |

**Observed characteristics (from profiling the file):**

- **~216** distinct measurement sources (`source_file` groups).
- **28** frequencies per sweep, spanning **0.05 Hz – 10 kHz** (mHz diffusion tail through kHz ohmic/inductive region).
- **SoC** sampled at five levels: **10, 30, 50, 70, 90 %**.
- **Temperature** spans roughly **0 °C – 41 °C** (continuous, not gridded).
- **SoH** ranges from about **7 % to 112 %**. Values above 100 % and the wide spread indicate measurement noise and/or augmentation; the label is effectively continuous and will need cleaning/clipping.
- The `cycles` column currently contains only a narrow range of values and is **not** a reliable cycle counter; SoH is the primary health label.

> **Notes / caveats to resolve during data prep:** the file is at the ~1,048,575-row spreadsheet limit, so it may be truncated at the source — verify completeness. SoH > 100 % must be handled (clip or treat as noise). Per-cell grouping must be defined precisely from `source_file` (pack / cell / slot / channel encoding in the name) so that train/test splits never leak across the same physical cell.

## 4. Pipeline Overview

<img width="2064" height="2304" alt="pipeline_diagram" src="https://github.com/user-attachments/assets/e763ecfe-834e-4049-a29c-1dfb012d43f4" />

A rendered version of this pipeline is in [`pipeline_diagram.png`](pipeline_diagram.png).

## 5. Stage 1 — Feature Extraction (EIS → ECM)

Each spectrum (one cell, one SoC, one temperature, one age state) is processed as follows.

**5.1 Validate first (Kramers–Kronig).** EIS assumes the system is linear, causal, stable, and time-invariant during the sweep. Low-frequency points take a long time to acquire, during which a cell can drift, invalidating the spectrum. A Kramers–Kronig consistency check is run before fitting; spectra with large systematic residuals are flagged or rejected so that bad data never reaches the model.

**5.2 Fit an Equivalent Circuit Model.** A Randles-type circuit is fit to each validated spectrum to extract interpretable parameters. The exact circuit topology depends on the cell chemistry and is an [open decision](#open-decisions); the canonical starting point is:

```
R_ohm  +  ( R_ct ∥ CPE )  +  Warburg
```

**5.3 Extracted features.** The following are computed per spectrum and become the model inputs:

| Feature | Symbol | Physical meaning |
|---|---|---|
| Ohmic resistance | `R_ohm` | High-frequency real-axis intercept — electrolyte, contacts, current collectors. Grows with ageing. |
| Charge-transfer resistance | `R_ct` | Mid-frequency semicircle diameter — kinetics of the electrode reaction. Grows as the interface degrades. |
| CPE exponent | `n` (and magnitude `Q`) | Depression of the semicircle — surface inhomogeneity / roughness of the double layer. |
| Health deviation | `ΔZ = Z_healthy − Z_measured` | Per-frequency (and aggregated) deviation of the present spectrum from a fresh-cell baseline. A direct, physics-anchored health signal that teaches the model what "moving away from healthy" looks like. |
| Warburg coefficient | `σ` (optional) | Low-frequency diffusion behaviour (~45° tail) — solid-state/electrolyte transport. |

Operating conditions (`SoC_pct`, `temp_degC`) are carried alongside these features, because impedance depends strongly on both and they must be conditioned on, never ignored.

**On ΔZ:** a healthy-cell reference spectrum is needed per (SoC, temperature) operating point so that `ΔZ` compares like with like. How the healthy baseline is defined (first measurement per cell, a fitted nominal model, or a population reference) is a design choice to settle during implementation.

## 6. Stage 2 — Health Model (ML)

The ML model consumes the ECM feature table plus operating conditions and learns:

- **The evolution law** — how the features (and `ΔZ`) move as SOH, SOC, and temperature change.
- **A predictor** — mapping the current feature vector to **SOH** (and, by extension, **RUL** against an end-of-life threshold).
- **A confidence estimate** — a prediction interval or variance accompanying each point estimate.

The specific algorithm is an [open decision](#open-decisions). Tree ensembles (XGBoost / LightGBM / Random Forest) are strong, fast baselines on tabular impedance features and pair naturally with feature-importance attribution; models with native uncertainty (quantile/ensemble methods, Gaussian processes) are attractive because confidence is a first-class requirement here.

**Physical plausibility constraints** the model output should respect: SOH should be (near-)monotonically non-increasing over a cell's life apart from measurement noise; resistance features should grow or hold, not spontaneously drop; and uncertainty should widen for inputs that fall outside the training distribution (new temperature, new age regime).

## 7. Stage 3 — Drift Detection & Self-Healing Loop

A model trained today will eventually face cells whose impedance signature has drifted beyond what it has seen. The system detects this and repairs itself rather than silently degrading.

**7.1 Generate new EIS.** For research purposes, a *new* EIS dataset is synthesised representing a cell after additional months of use. The synthesis method is an [open decision](#open-decisions) — candidates include perturbing ECM parameters along known ageing trends, physics-based impedance simulation, or augmentation of measured spectra.

**7.2 Re-extract parameters.** The new spectra are passed through the same Stage 1 pipeline to produce a new ECM feature set (new `R_ohm`, `R_ct`, `ΔZ`, CPE exponent, etc.).

**7.3 Compute a drift score.** The shift of the ECM parameters (and/or model residuals) relative to the reference distribution is quantified as a single **drift score** (see [Drift Score](#8-drift-score)).

**7.4 Compare to threshold and act:**

- **Drift score < threshold** → distribution is stable; keep the current model in service.
- **Drift score ≥ threshold** → enter the **Self-Healing (SHML) loop**:
  1. **Update ECM parameters** for the new operating regime.
  2. **Update the evolution laws** the model relies on.
  3. **Fine-tune or fully retrain** in the cloud — fine-tune for mild drift, full retrain for severe drift, with the cut-off governed by the drift score magnitude.
  4. **Deploy a lightweight model back to the edge** device, replacing the previous version.

This mirrors the **Self-Healing Machine Learning** pattern: autonomously detect degradation, diagnose its severity, choose a remediation proportional to the damage, and adapt — with the heavy training in the cloud and a compact model running on-device.

## 8. Drift Score

The drift score reduces "how far have the parameters moved?" to one comparable number. The exact formula is an [open decision](#open-decisions); the leading candidates, all of which fit this framework, are:

- **Population Stability Index (PSI)** — bins a feature's distribution now vs. baseline and sums the relative-entropy contribution. Common reading: `< 0.1` no meaningful shift, `0.1–0.25` moderate, `> 0.25` significant. Simple and interpretable per feature.
- **KL divergence** — relative entropy of the new parameter distribution against the baseline (PSI is essentially a symmetrised, binned KL).
- **Mahalanobis distance** — distance of the new ECM parameter vector from the healthy/baseline parameter cloud, accounting for feature covariance. Naturally produces a single multivariate score with a statistical threshold.
- **Normalised parameter drift** — a weighted sum of per-parameter relative changes, e.g. `Σ wᵢ · |θᵢ − θᵢ⁰| / |θᵢ⁰|`, easy to interpret and to map onto fine-tune vs. retrain bands.
- **Sequential detectors (Page–Hinkley / ADWIN)** — run on the stream of model residuals to raise a discrete drift alarm rather than a static comparison; useful for continuous online monitoring.

A practical design is to combine a **two-band threshold** on the chosen score: a lower band that triggers fine-tuning and an upper band that triggers full retraining.

## 9. Evaluation & Validation

- **No-leakage splits.** Group by physical cell (leave-one-cell-out) and, where possible, leave-one-condition-out. Never use random row splits — adjacent rows from the same sweep/cell leak health information.
- **Metrics.** Report SOH/RUL error (MAE/RMSE) *with* prediction-interval coverage and calibration, plus mean ± std across folds/seeds, not a single split.
- **Plausibility checks.** Verify monotonic SOH, non-decreasing resistance, and that uncertainty grows out-of-distribution before trusting any number.
- **Drift-loop validation.** Confirm the drift score rises on the synthesised aged data and that fine-tune/retrain recovers accuracy on the shifted distribution.

## 10. Proposed Repository Structure

```
.
├── data/
│   ├── raw/                  # original EIS exports
│   └── eis_cleaned.csv       # working dataset
├── src/
│   ├── validation/           # Kramers–Kronig checks
│   ├── ecm/                  # circuit fitting & feature extraction
│   ├── features/             # ΔZ baselines, feature assembly
│   ├── model/                # training, inference, uncertainty
│   ├── drift/                # drift score, thresholds, monitors
│   └── healing/              # fine-tune / retrain / edge export
├── notebooks/                # exploration & analysis
├── configs/                  # experiment configs (seeds, params)
├── pipeline_diagram.png
├── implementation.md         # build guide for contributors / coding assistant
└── README.md
```

## 11. Open Decisions

These choices are **not yet made** and are expected to evolve. They are tracked here so the design stays honest about what is settled and what is not.

1. **Which ECM circuit?** The exact equivalent circuit depends on the cell chemistry, which in turn depends on the chemistry of the cells in this dataset. The chemistry must be identified (or assumed and stated) before the circuit topology — Randles, Randles + second R∥CPE for SEI, with/without inductance, finite vs. semi-infinite Warburg — can be fixed.
2. **Which ML model for SOH?** Candidate predictors include XGBoost, Random Forest, LightGBM, and uncertainty-native alternatives. Selection should weigh accuracy, calibrated confidence, and edge-deployability.
3. **How is the new EIS synthesised?** The method for generating post-ageing spectra (parameter perturbation along ageing trends, physics-based simulation, or measured-spectrum augmentation) is undecided and affects how convincingly the drift loop can be validated.
4. **What formula for the drift score?** PSI, KL divergence, Mahalanobis distance, normalised parameter drift, or a sequential detector — and the threshold band(s) that separate "keep", "fine-tune", and "retrain" — are open.

## 12. Roadmap

The framework is explicitly **iterative**. Expected progression:

1. Data preparation and per-cell grouping; SoH cleaning.
2. K–K validation + ECM fitting; baseline feature table.
3. First health model with uncertainty; no-leakage evaluation.
4. Drift score implementation and threshold calibration on synthesised aged data.
5. Self-healing loop (fine-tune/retrain trigger + edge export).
6. Iterate: revisit circuit topology, features, model family, drift score, and thresholds as results come in.

Every stage above is a candidate for improvement; the architecture is meant to be refined repeatedly rather than frozen after a first pass.

## 13. References

- EIS interpretation, ECM fitting, and EIS-based ML — domain notes compiled for this project.
- Equivalent-circuit modelling of commercial Li-ion cells: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7671193/>
- Guide to equivalent-circuit fitting for impedance and battery state estimation: <https://www.sciencedirect.com/science/article/pii/S2352152X2303788X>
- Concept-drift detection methods overview: <https://ai-infrastructure.org/8-concept-drift-detection-methods/>
- KL-divergence drift detection: <https://link.springer.com/article/10.1007/s42488-024-00119-y>
- Self-Healing Machine Learning framework: <https://proceedings.neurips.cc/paper_files/paper/2024/file/4a86ec12e94ef1fe306362e7bdcd5894-Paper-Conference.pdf>
- ML-based digital twin for EV battery modelling (edge–cloud retraining): <https://arxiv.org/pdf/2206.08080>
