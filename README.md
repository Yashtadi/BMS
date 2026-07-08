# Physics-Residual SoH Model (Stage 2)

This document is the complete, plain-language record of the **SoH prediction model** — the stage that
comes after ECM fitting. It assumes Stage 0 (`eis_labeled.parquet`) and Stage 1 (ECM feature fitting)
are complete, and explains the model architecture, why it's shaped this way, how it plugs into the
self-healing drift-adaptation loop, and what decisions are still open.

If you are picking this project up cold, read [Part 0](#part-0--recap-where-we-are) first, then the
architecture in [Part 1](#part-1--why-a-physics-residual-model), then the decisions log in
[Part 8](#part-8--decisions-log-consolidated) for the fast version of everything.

---

## Part 0 — Recap: where we are

Stage 0 produced `data/interim/eis_labeled.parquet`: 228 cells, ~39k spectra, each spectrum labeled
with `soh_cap` (the non-circular, capacity-based health target — see Stage 0 doc §5 for why).

Stage 1 (ECM fitting, in progress per the Stage 0 doc's "What's next") will add, per spectrum:
- `R_ohm`, `R_ct`, `Q1`/`n1` (CPE1), `Q2`/`n2` (CPE2), Warburg `σ` — the fitted circuit parameters
- `deltaZ_imag` — the aging-drift signal, computed as `Z_baseline(f) − Z_now(f)` at each frequency,
  where `Z_baseline` is **each cell's own first check-up spectrum**, per `(cell, SoC, pass)` — this
  convention was locked in the Stage 0 doc (§6.2) specifically so it lines up with how the dataset's
  own `z_ref_init` is defined, even though we don't use `z_ref_init` directly as an input.

This document (Stage 2) describes what happens **after** that feature table exists: how it's turned
into a `soh_cap` prediction.

---

## Part 1 — Why a physics-residual model

### 1.1 The problem it solves

`deltaZ_imag` and the ECM parameters (`R_ct`, CPE terms, etc.) are **not independent** — they're
derived from the same underlying impedance spectrum. A plain model (NN, GBM, etc.) fed all of them at
once has no reason to treat `deltaZ` as special; it will happily let `R_ct` or the CPE exponent soak up
the same degradation signal, and which feature "gets credit" becomes arbitrary and unstable across
retrains.

We want `deltaZ` to be the **primary, physically-anchored driver** of the SoH prediction — both because
it's the cleanest aging signal in the feature set, and because it's the piece that plugs directly into
the self-healing drift loop (Part 5).

### 1.2 The architecture

```
soh_cap_pred = f_physics(deltaZ)  +  g_NN(ECM params, SoC, temp_degC)
```

- **`f_physics(deltaZ)`** — a small, parametric function (a handful of learnable parameters, not a
  full network) that maps the aging-drift signal directly to a SoH estimate. This *is* the physics
  term — a stand-in for the empirical relationship between impedance growth and capacity fade.
- **`g_NN(...)`** — a small correction network that captures everything `deltaZ` alone can't explain:
  ECM parameters, `soc_nom`, `temp_degC`. Regularized (small weight penalty, or a capped output range)
  so it corrects rather than dominates.

**The key rule: `deltaZ` never enters `g_NN`.** This is enforced architecturally, not by hoping
regularization sorts it out — it's the only way to guarantee the correction network can't reconstruct
`deltaZ`'s contribution from the correlated ECM parameters and quietly override the physics term.

---

## Part 2 — Handling the deltaZ / ECM-parameter collinearity

Because `R_ct` (and to a lesser extent the CPE terms) are fit from the same spectrum `deltaZ` is
computed against, they will be correlated — likely substantially, since a growing charge-transfer
resistance is often the main driver of the imaginary-impedance shift `deltaZ` captures.

**Before building the feature table for `g_NN`:**
1. Run a VIF (variance inflation factor) check across `deltaZ`, `R_ohm`, `R_ct`, `Q1`, `n1`, `Q2`,
   `n2`, `σ`. This is worth doing regardless of final model choice, and gives a defensible table for
   the methods section.
2. If VIF between `deltaZ` and `R_ct` is high (likely), **orthogonalize**: regress `R_ct` on `deltaZ`
   and use the residual as the actual `R_ct` input to `g_NN`. This guarantees `g_NN` only sees the part
   of `R_ct` that `deltaZ` doesn't already explain — i.e., `deltaZ`'s contribution is structurally
   protected from being "stolen" by a correlated feature.
3. Repeat the same residualization for the other ECM parameters that show high VIF against `deltaZ`.

> **Decision needed before Stage 2 build:** confirm the orthogonalization target — orthogonalize each
> ECM parameter against `deltaZ` individually, or against a summary drift statistic (e.g. mean
> `deltaZ` across frequency bands)? Recommend starting with the latter (simpler, one regression) and
> only going per-frequency-band if diagnostics show it's insufficient.

---

## Part 3 — `f_physics`: functional form

`deltaZ_imag` is a per-frequency signal (28 points per spectrum). `f_physics` needs a summary
statistic or small set of statistics as its actual input — decide this before fitting:

**Candidate inputs (pick one to start, compare later):**
- Mean `deltaZ` across all 28 frequencies
- Mean `deltaZ` restricted to the aging-sensitive low-frequency band (~1–5 Hz — the same region
  `z_ref_init` is anchored to, per Stage 0 §6.2, which makes this choice easy to cross-check)
- Integrated drift (area under `|deltaZ(f)|` across the spectrum)

**Candidate functional forms:**
- Exponential decay: `soh = a - b * exp(c * deltaZ_summary)`
- Power law: `soh = a - b * deltaZ_summary^c`
- Simple linear (as a baseline to beat): `soh = a - b * deltaZ_summary`

> **Decision needed:** fit all three candidate forms against `soh_cap` using only `deltaZ` (ignore
> ECM params entirely) as a quick standalone check — whichever gives the best R² becomes `f_physics`.
> This can be done with `scipy.optimize.curve_fit`, no NN needed for this step.

---

## Part 4 — `g_NN`: correction network

**Inputs:** orthogonalized ECM parameters (Part 2) + `soc_nom` + `temp_degC`. Never `deltaZ`.

**Design constraints:**
- Small — this is a correction term, not the main model. 1–2 hidden layers, modest width.
- Regularized output — either an explicit output-magnitude penalty in the loss, or a `tanh`-scaled
  output capped to a small range, so it can't swing the prediction as far as `f_physics` can.
- Grouped train/test split — same `cell_id` grouping rule as everywhere else in this project (Stage 0
  Part 0), so no physical cell appears in both train and test.

**Loss:**
```
L = MSE(soh_cap, f_physics(deltaZ) + g_NN(...))  +  λ * ||g_NN_output||²
```
`λ` controls how much correction the network is allowed to apply — start high (favor `f_physics`
doing most of the work) and relax only if residual error analysis shows `g_NN` is under-fitting a real
pattern (e.g. temperature-dependent structure `f_physics` can't capture alone).

**Training order (staged, not joint, at least initially):**
1. Fit `f_physics` alone against `soh_cap` (Part 3).
2. Freeze `f_physics`, train `g_NN` on the *residual* (`soh_cap − f_physics(deltaZ)`).
3. Optionally unfreeze both for a final joint fine-tune pass, with `λ` still active to prevent drift
   away from the staged solution.

This staged order also mirrors the mild/severe distinction in Part 5 — the two components are
designed to be updated independently, so training them independently from the start avoids painting
yourself into a jointly-entangled model that can't cleanly split later.

---

## Part 5 — Connection to the self-healing drift loop

This is the reason the model is split into two pieces in the first place — it maps directly onto the
mild-vs-severe branch of the drift-adaptation pipeline:

| Drift severity | What updates | Why it's cheap/expensive |
|---|---|---|
| **Mild** | Refit `f_physics` only | A handful of parameters, fast, can run without a full training job |
| **Severe** | Retrain `g_NN` (cloud) | Full network retrain, needed when the ECM-parameter relationships have genuinely shifted, not just `deltaZ`'s scale |

**Drift detection should be defined separately for each component** — don't use one combined drift
score for both. A reasonable starting design:
- **Physics-side drift signal:** how much has the `deltaZ → soh_cap` relationship shifted on newly
  synthesized/incoming EIS versus what `f_physics` currently predicts.
- **Correction-side drift signal:** how much has the residual (`soh_cap − f_physics(deltaZ)`) pattern
  shifted versus what `g_NN` currently predicts.

> **Open design question:** exact thresholds for "mild" vs "severe" aren't set yet — this needs real
> drift data (synthesized aged cells) to calibrate against, not a value picked in the abstract.

---

## Part 6 — Evaluation plan

- **Split:** group by `cell_id` (never split a physical cell across train/test — same rule as Stage 0).
- **Primary metric:** RMSE / MAE of `soh_cap_pred` vs `soh_cap`, on held-out cells.
- **Diagnostic, not primary:** compare against `soh_imp` (impedance-based) as a sanity check only —
  per Stage 0 Part 5, `soh_imp` is not the target and shouldn't be optimized against.
- **Ablation:** report `f_physics`-only performance vs full `f_physics + g_NN`, to demonstrate the
  correction network is adding real value and not just fitting noise.
- **Per-recipe breakdown:** check performance separately for Calendar / Cyclic / Profile aging types
  (Stage 0's `aging_type` column) — a model that only works well on one aging type is a real finding,
  not just a footnote.

---

## Part 7 — Known issues inherited from Stage 0 / Stage 1 (must resolve first)

These are blocking issues carried forward from the Stage 0 doc's "Known issues" section — none of the
above should be built on real data until these are settled:

- **`soh_cap` label cleaning:** negative/bouncing values near end-of-life, and the synchronized CU-27
  artifact (Stage 0 Part 2.2, Part 10). Needs a clip/drop policy before this becomes training data —
  an artifact affecting every cell at once will otherwise get learned as if it were a real physical
  signal.
- **`Z_imag` sign convention:** must be locked during ECM fitting (Stage 1) before `deltaZ` means
  anything consistent.
- **ECM fit quality:** per earlier project history, a meaningful fraction of sweeps have `R2_mOhm`
  saturating at its upper bound — worth checking whether these should be excluded or flagged before
  computing `deltaZ` and the orthogonalized ECM features from them.

---

## Part 8 — Decisions log (consolidated)

| # | Decision | Why | Reversible? |
|---|---|---|---|
| 1 | Architecture = `f_physics(deltaZ) + g_NN(ECM params, SoC, temp)` | Forces deltaZ to be the primary, protected driver; maps onto mild/severe drift split | yes, in principle — but changing this restructures the whole drift loop, so treat as a soft commitment |
| 2 | `deltaZ` never enters `g_NN` | Prevents the correction network from reconstructing/overriding the physics term via correlated ECM params | yes (but defeats the point if reversed) |
| 3 | Orthogonalize ECM params against deltaZ before feeding `g_NN` | deltaZ and R_ct/CPE are collinear (same source spectrum) | yes |
| 4 | Staged training: fit `f_physics` → freeze → train `g_NN` on residual → optional joint fine-tune | Keeps the two components independently updatable, matching the drift loop | yes |
| 5 | Separate drift signals for physics-side vs correction-side | One combined score can't tell you which component actually needs updating | yes |
| — | `f_physics` input summary statistic (mean deltaZ vs low-freq band vs integrated) | **Not yet decided** — needs the standalone curve-fit comparison in Part 3 | — |
| — | Mild/severe drift thresholds | **Not yet decided** — needs calibration data | — |

---

## Part 9 — What's next

1. Resolve the Part 7 blocking issues (label cleaning, sign convention, ECM fit quality) — these sit
   upstream and will silently corrupt everything below if skipped.
2. Run the VIF check (Part 2) once the ECM feature table exists.
3. Fit and compare the three `f_physics` candidate forms (Part 3) as a standalone check, before any
   NN work starts.
4. Build `g_NN` on residuals (Part 4), staged training.
5. Run the ablation + per-recipe evaluation (Part 6).
6. Only after the model is validated: design the actual drift-score calibration (Part 5's open
   question) using synthesized aged-cell data.
