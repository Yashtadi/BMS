"""
fit_ecm.py — Stage 1: Equivalent Circuit Model fitting.

Before any circuit fitting, each spectrum is checked with a linear
Kramers-Kronig (KK) validity test (Schönleber et al. 2014, via
impedance.py's linKK). This is a data-quality check on the RAW measured
spectrum, independent of any circuit topology — it fits a generic
"measurement model" (a series of RC elements) and checks the residuals.
Large residuals mean the spectrum likely violates the linearity, causality,
stability, or time-invariance assumptions EIS requires (e.g. the cell drifted
during a slow low-frequency sweep). This FLAGS spectra (kk_valid), it does
NOT reject/drop them — consistent with the project's general "flag, don't
silently filter" approach; downstream stages decide whether to exclude
kk_valid == False rows.

Fits, per spectrum:
    L + R_ohm + (R_sf || CPE_sf) + (R_ct || CPE_dl) + Warburg      (primary, 2-arc)
    L + R_ohm + (R_mid || CPE_mid) + Warburg                        (reference, 1-arc)

Both fits are run on every spectrum. The 2-arc parameters are the primary
feature-table values (R_sf, R_ct, R_mid_total = R_sf + R_ct). The 1-arc
fit's own resistance (R_1arc) is ALSO saved — it is a genuinely separate,
simpler optimization (not derived from the 2-arc fit's R_sf/R_ct), and is
generally the more stable choice when a single, always-trustworthy
resistance feature is needed (see project notes: the 2-arc split can be
numerically unstable — sensitive to initial guess, prone to CPE exponents
collapsing to implausible values — and R_1arc avoids exposure to that
instability entirely, unlike R_mid_total, which is still computed from
the same fragile 9-parameter optimization and can inherit its problems).

Because the two arcs can overlap (peak frequencies too close together to fully
resolve — see project notes), every fit ships with quality-diagnostic columns so
downstream stages (deltaZ orthogonalization, g_NN training) can decide how much to
trust the individual R_sf / R_ct split on a given spectrum, rather than treating
every fit as equally reliable:

    - peak_freq_ratio      : how well-separated the two arcs' characteristic
                              frequencies are (computed for reference; does NOT
                              gate reliable_split — see project decision log)
    - cpe_bound_hit         : True if either CPE exponent saturated at 0/1, or
                              fell below a physically plausible minimum
                              (a sign the optimizer couldn't find a genuine value)
    - aic_prefers_2arc      : True if the 2-arc model beats the 1-arc model on AIC
                              for THIS specific spectrum (not decided once globally)
    - R_mid_total           : R_sf + R_ct — combined total from the 2-arc fit
    - R_1arc                : resistance from the SEPARATE 1-arc fit — the more
                              stable choice as a default resistance feature
    - reliable_split        : composite flag for whether the 2-arc SPLIT specifically
                              (not R_mid_total, not R_1arc) is trustworthy
    - kk_valid              : whether the raw spectrum passes the Kramers-Kronig
                              linearity check (independent of circuit topology)

Sign convention: this dataset stores Z_imag_mOhm with the OPPOSITE sign from the
standard EIS convention (see data_exploration.md). This module flips it on load.
"""

import contextlib
import io
import warnings
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit
from impedance.validation import linKK, circuit_elements as _kk_circuit_elements
from scipy.optimize import OptimizeWarning

# Workaround for a bug in this version of impedance.py: linKK's internal
# progress-printing path (triggered whenever the adaptive RC-element search
# hits a multiple of 10) calls eval_linKK(), which eval()s a circuit string
# using impedance.validation.circuit_elements as the globals dict — but that
# dict is missing 'np', so it raises NameError: name 'np' is not defined.
# This can hit ANY spectrum depending on how many RC elements its measurement
# model needs, so it must be patched before calling linKK, not worked around
# per-spectrum. Confirmed necessary and sufficient by testing directly against
# impedance.validation.eval_linKK's eval() call.
_kk_circuit_elements["np"] = np

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SPECTRUM_KEYS = ["cell_id", "cu_index", "soc_nom", "is_rt"]

CIRCUIT_2ARC = "L0-R0-p(R1,CPE1)-p(R2,CPE2)-W1"
CIRCUIT_1ARC = "L0-R0-p(R1,CPE1)-W1"

MIN_VALID_POINTS = 10  # below this, a 9-parameter 2-arc fit is not meaningfully identifiable
CPE_BOUND_EPS = 1e-3   # how close to 0 or 1 counts as "hit the exact bound"
CPE_N_PLAUSIBLE_MIN = 0.3  # below this, n is physically implausible for a battery electrode
                            # even if it hasn't hit the literal 0/1 optimizer bound — a near-zero
                            # n produces numerically absurd peak-frequency ratios (10^20+) that
                            # look like excellent arc separation but are actually a degenerate fit
PEAK_RATIO_RELIABLE = 10.0  # below this ratio, treat the R_sf/R_ct split as unreliable
PEAK_RATIO_MAX_SANE = 1e4   # above this, treat the ratio itself as a numerical artifact, not real separation
R_OHM_MIN_PLAUSIBLE = 1.0   # mOhm — a real 18650 cell never has near-zero ohmic resistance;
                            # a fit landing below this converged to a degenerate solution

KK_CUTOFF = 0.85       # mu cutoff for the linKK measurement-model fit (Schönleber et al. default)
KK_MAX_M = 20          # cap on RC elements tried — keeps runtime bounded across ~39k spectra
KK_RESIDUAL_THRESHOLD = 0.05  # 5% — linKK residuals are already normalized by |Z|; above this,
                               # flag the raw spectrum as likely violating KK assumptions


@dataclass
class FitResult:
    cell_id: str
    cu_index: int
    soc_nom: int
    is_rt: int
    n_valid_points: int
    fit_succeeded: bool
    # 2-arc parameters (the feature-table values)
    L: float = np.nan
    R_ohm: float = np.nan
    R_sf: float = np.nan
    Q_sf: float = np.nan
    n_sf: float = np.nan
    R_ct: float = np.nan
    Q_ct: float = np.nan
    n_ct: float = np.nan
    sigma: float = np.nan
    R_mid_total: float = np.nan
    rss_2arc: float = np.nan
    aic_2arc: float = np.nan
    # 1-arc reference fit: full parameters now saved, not just RSS/AIC —
    # this is a genuinely separate, simpler optimization (not derived from
    # the 2-arc fit), and R_1arc is the intended primary resistance feature
    L_1arc: float = np.nan
    R_ohm_1arc: float = np.nan
    R_1arc: float = np.nan
    Q_1arc: float = np.nan
    n_1arc: float = np.nan
    sigma_1arc: float = np.nan
    rss_1arc: float = np.nan
    aic_1arc: float = np.nan
    # quality flags
    peak_freq_ratio: float = np.nan
    cpe_bound_hit: bool = False
    r_ohm_implausible: bool = False
    aic_prefers_2arc: bool = False
    reliable_split: bool = False
    # Kramers-Kronig validation (on the raw spectrum, independent of circuit topology)
    kk_M: int = -1
    kk_mu: float = np.nan
    kk_max_residual: float = np.nan
    kk_valid: bool = False
    kk_error: Optional[str] = None
    error: Optional[str] = None


def get_spectrum_arrays(spec_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (freq, Z) for one spectrum, using only valid points, in the
    standard EIS sign convention (this dataset stores the opposite sign —
    see data_exploration.md — so we flip it here).

    Filters on `valid == 1` AND non-null impedance values — the instrument's
    own `valid` flag does not fully guarantee the value is actually present;
    some rows are flagged valid==1 with Z_real_mOhm/Z_imag_mOhm still NaN.
    """
    valid = spec_df[
        (spec_df["valid"] == 1)
        & spec_df["Z_real_mOhm"].notna()
        & spec_df["Z_imag_mOhm"].notna()
    ].sort_values("freq_Hz")
    freq = valid["freq_Hz"].to_numpy()
    Z = valid["Z_real_mOhm"].to_numpy() + 1j * (-valid["Z_imag_mOhm"].to_numpy())
    return freq, Z


def kk_validate(
    freq: np.ndarray, Z: np.ndarray, c: float = KK_CUTOFF, max_M: int = KK_MAX_M
) -> tuple[int, float, float]:
    """
    Linear Kramers-Kronig validity test (Schönleber et al. 2014), via
    impedance.py's linKK. Fits a series of RC elements (a generic
    "measurement model" — NOT the physical ECM circuit) to the spectrum
    and checks the residuals. This is a check on the RAW measured data —
    it runs independent of, and before, any equivalent-circuit fitting,
    and answers a different question: "is this spectrum internally
    consistent with what EIS assumes (linear, causal, stable,
    time-invariant during the sweep)?" — not "does a given circuit fit it
    well." A spectrum can fail KK even if some circuit fits it with low
    RSS, and can pass KK even if no simple circuit fits it cleanly.

    linKK's residuals are already normalized by |Z| at each frequency
    (fractional/relative residuals), so KK_RESIDUAL_THRESHOLD is a plain
    fraction (0.05 = 5%), not raw mOhm.

    Returns
    -------
    M   : number of RC elements the measurement model needed (higher M for
          a fixed cutoff can itself indicate a harder-to-fit spectrum)
    mu  : Schönleber under/over-fitting measure
    max_residual : largest |relative residual| across all frequencies,
          real and imaginary combined
    """
    # linKK prints progress lines (every 10th M) directly to stdout — this
    # would flood output across ~39k spectra run in parallel; suppress it.
    with contextlib.redirect_stdout(io.StringIO()):
        M, mu, Z_fit, res_real, res_imag = linKK(
            freq, Z, c=c, max_M=max_M, fit_type="complex", add_cap=True
        )
    max_residual = float(np.max(np.abs(np.concatenate([res_real, res_imag]))))
    return M, mu, max_residual


def adaptive_initial_guess_2arc(freq: np.ndarray, Z: np.ndarray) -> list:
    """
    Build a data-driven initial guess rather than a fixed constant, since
    resistance scale varies a lot between a fresh cell and a near-end-of-life
    one — a fixed guess that works for one can fail to converge for the other.
    """
    r_high_f = Z.real[np.argmax(freq)]       # proxy for R_ohm
    r_low_f = Z.real[np.argmin(freq)]        # proxy for R_ohm + R_sf + R_ct (+ some Warburg)
    total_mid = max(r_low_f - r_high_f, 1e-4)

    return [
        1e-7,                 # L
        max(r_high_f, 1e-4),  # R_ohm
        total_mid * 0.4,      # R_sf
        1e-3, 0.8,            # Q_sf, n_sf
        total_mid * 0.6,      # R_ct
        1e-3, 0.8,            # Q_ct, n_ct
        1.0,                  # Warburg sigma
    ]


def adaptive_initial_guess_1arc(freq: np.ndarray, Z: np.ndarray) -> list:
    r_high_f = Z.real[np.argmax(freq)]
    r_low_f = Z.real[np.argmin(freq)]
    total_mid = max(r_low_f - r_high_f, 1e-4)
    return [1e-7, max(r_high_f, 1e-4), total_mid, 1e-3, 0.8, 1.0]


def _aic(Z_meas: np.ndarray, Z_fit: np.ndarray, n_params: int) -> tuple[float, float]:
    resid = np.concatenate([Z_meas.real - Z_fit.real, Z_meas.imag - Z_fit.imag])
    rss = float(np.sum(resid**2))
    n = len(resid)
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + 2 * n_params, rss


def _peak_freq(R: float, Q: float, n: float) -> float:
    """Characteristic frequency of an R-CPE pair: f = 1 / (2*pi*(R*Q)^(1/n))."""
    if R <= 0 or Q <= 0 or n <= 0:
        return np.nan
    tau = (R * Q) ** (1.0 / n)
    if tau <= 0:
        return np.nan
    return 1.0 / (2 * np.pi * tau)


def fit_one_spectrum(key: tuple, spec_df: pd.DataFrame) -> FitResult:
    """
    Fit both the 2-arc (primary) and 1-arc (reference) circuits to one
    spectrum, and compute the quality-diagnostic flags described in the
    module docstring. Never raises — fit failures are captured in the
    result's `error` field so one bad spectrum can't crash a full run.
    """
    cell_id, cu_index, soc_nom, is_rt = key
    freq, Z = get_spectrum_arrays(spec_df)
    n_valid = len(freq)

    result = FitResult(
        cell_id=cell_id, cu_index=cu_index, soc_nom=soc_nom, is_rt=is_rt,
        n_valid_points=n_valid, fit_succeeded=False,
    )

    if n_valid < MIN_VALID_POINTS:
        result.error = f"too few valid points ({n_valid} < {MIN_VALID_POINTS})"
        return result

    # Kramers-Kronig validation runs first, independent of circuit fitting.
    # A KK failure is recorded as a flag, NOT treated as a reason to skip
    # the circuit fits below — consistent with the "flag, don't silently
    # filter" approach used throughout this module.
    try:
        kk_M, kk_mu, kk_max_res = kk_validate(freq, Z)
        result.kk_M = kk_M
        result.kk_mu = kk_mu
        result.kk_max_residual = kk_max_res
        result.kk_valid = bool(kk_max_res < KK_RESIDUAL_THRESHOLD)
    except Exception as exc:  # noqa: BLE001 — KK failure must not block circuit fitting
        result.kk_valid = False
        result.kk_error = f"kk validation raised: {exc}"

    try:
        guess2 = adaptive_initial_guess_2arc(freq, Z)
        c2 = CustomCircuit(circuit=CIRCUIT_2ARC, initial_guess=guess2)
        c2.fit(freq, Z)
        Z2 = c2.predict(freq)
        names2, vals2 = c2.get_param_names()[0], c2.parameters_
        p2 = dict(zip(names2, vals2))
        aic2, rss2 = _aic(Z, Z2, n_params=9)

        guess1 = adaptive_initial_guess_1arc(freq, Z)
        c1 = CustomCircuit(circuit=CIRCUIT_1ARC, initial_guess=guess1)
        c1.fit(freq, Z)
        Z1 = c1.predict(freq)
        names1, vals1 = c1.get_param_names()[0], c1.parameters_
        p1 = dict(zip(names1, vals1))
        aic1, rss1 = _aic(Z, Z1, n_params=6)

    except Exception as exc:  # noqa: BLE001 — deliberately broad: one bad spectrum must not kill a batch run
        result.error = f"fit raised: {exc}"
        return result

    result.fit_succeeded = True
    result.L = p2["L0"]
    result.R_ohm = p2["R0"]
    result.R_sf = p2["R1"]
    result.Q_sf = p2["CPE1_0"]
    result.n_sf = p2["CPE1_1"]
    result.R_ct = p2["R2"]
    result.Q_ct = p2["CPE2_0"]
    result.n_ct = p2["CPE2_1"]
    result.sigma = p2["W1"]
    result.R_mid_total = p2["R1"] + p2["R2"]
    result.rss_2arc = rss2
    result.aic_2arc = aic2
    result.L_1arc = p1["L0"]
    result.R_ohm_1arc = p1["R0"]
    result.R_1arc = p1["R1"]
    result.Q_1arc = p1["CPE1_0"]
    result.n_1arc = p1["CPE1_1"]
    result.sigma_1arc = p1["W1"]
    result.rss_1arc = rss1
    result.aic_1arc = aic1

    f1 = _peak_freq(p2["R1"], p2["CPE1_0"], p2["CPE1_1"])
    f2 = _peak_freq(p2["R2"], p2["CPE2_0"], p2["CPE2_1"])
    if np.isfinite(f1) and np.isfinite(f2) and min(f1, f2) > 0:
        result.peak_freq_ratio = max(f1, f2) / min(f1, f2)
    else:
        result.peak_freq_ratio = np.nan

    result.cpe_bound_hit = bool(
        p2["CPE1_1"] <= CPE_BOUND_EPS or p2["CPE1_1"] >= 1 - CPE_BOUND_EPS
        or p2["CPE2_1"] <= CPE_BOUND_EPS or p2["CPE2_1"] >= 1 - CPE_BOUND_EPS
        or p2["CPE1_1"] < CPE_N_PLAUSIBLE_MIN or p2["CPE2_1"] < CPE_N_PLAUSIBLE_MIN
    )
    result.r_ohm_implausible = bool(result.R_ohm < R_OHM_MIN_PLAUSIBLE)

    result.aic_prefers_2arc = bool(aic2 < aic1)
    # NOTE: peak_freq_ratio is still computed and stored above for reference/diagnostics,
    # but it no longer gates reliable_split — per project decision, arc-separation distance
    # is not treated as a reliability requirement. Only fit-quality checks (CPE plausibility,
    # R_ohm plausibility, AIC preference) determine reliable_split now.
    result.reliable_split = bool(
        result.aic_prefers_2arc
        and not result.cpe_bound_hit
        and not result.r_ohm_implausible
    )

    return result


def _fit_worker(args: tuple) -> dict:
    key, spec_df = args
    return asdict(fit_one_spectrum(key, spec_df))


def fit_all_spectra(
    df: pd.DataFrame,
    n_jobs: int = 1,
    sample: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 2000,
) -> pd.DataFrame:
    """
    Fit every spectrum in `df` (grouped by SPECTRUM_KEYS) and return a
    one-row-per-spectrum DataFrame of ECM parameters + quality flags.

    Parameters
    ----------
    n_jobs : use >1 to parallelize across spectra (each fit is independent).
    sample : if set, only fit this many spectra — for a quick local test
             before committing to a full run across all ~39k spectra.
    checkpoint_path : if set, progress is saved to this parquet path every
             `checkpoint_every` spectra. If this file already exists when
             the run starts, already-fitted spectra are SKIPPED and the run
             resumes from where it left off — safe to Ctrl+C and rerun the
             exact same command without losing prior work.
    """
    groups = list(df.groupby(SPECTRUM_KEYS))
    if sample is not None:
        groups = groups[:sample]

    already_done: set = set()
    existing_rows: list = []
    if checkpoint_path is not None:
        try:
            prev = pd.read_parquet(checkpoint_path)
            existing_rows = prev.to_dict("records")
            already_done = set(
                zip(prev["cell_id"], prev["cu_index"], prev["soc_nom"], prev["is_rt"])
            )
            print(f"Resuming: found {len(already_done)} already-fitted spectra in {checkpoint_path}")
        except (FileNotFoundError, OSError):
            pass

    remaining = [(key, spec_df) for key, spec_df in groups if key not in already_done]
    print(f"Fitting {len(remaining)} spectra ({len(already_done)} already done, n_jobs={n_jobs})...")

    if not remaining:
        return pd.DataFrame(existing_rows)

    all_rows = list(existing_rows)

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(_fit_worker, remaining), start=1):
                all_rows.append(row)
                if checkpoint_path is not None and i % checkpoint_every == 0:
                    pd.DataFrame(all_rows).to_parquet(checkpoint_path, index=False)
                    print(f"  checkpoint saved: {len(all_rows)} spectra done")
    else:
        for i, g in enumerate(remaining, start=1):
            all_rows.append(_fit_worker(g))
            if checkpoint_path is not None and i % checkpoint_every == 0:
                pd.DataFrame(all_rows).to_parquet(checkpoint_path, index=False)
                print(f"  checkpoint saved: {len(all_rows)} spectra done")

    out = pd.DataFrame(all_rows)

    n_ok = out["fit_succeeded"].sum()
    n_reliable = out["reliable_split"].sum()
    n_kk_valid = out["kk_valid"].sum()
    print(f"Succeeded: {n_ok}/{len(out)}  |  Reliable R_sf/R_ct split: {n_reliable}/{n_ok if n_ok else 1}"
          f"  |  KK-valid: {n_kk_valid}/{len(out)}")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/interim/eis_labeled.parquet")
    parser.add_argument("--out", default="data/interim/ecm_features.parquet")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--sample", type=int, default=None, help="fit only N spectra, for a quick test run")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                         help="save progress every N spectra; rerun the same command to resume after Ctrl+C")
    args = parser.parse_args()

    df = pd.read_parquet(args.file)
    ecm = fit_all_spectra(
        df, n_jobs=args.n_jobs, sample=args.sample,
        checkpoint_path=args.out, checkpoint_every=args.checkpoint_every,
    )
    ecm.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}  ({ecm.shape})")