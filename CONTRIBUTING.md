# Contributing & Collaboration Guide

This guide explains how two (or more) people, working from different machines, can build the **ECM
pipeline** for this project together — reliably, reproducibly, and without sacrificing quality.

It complements the two shorter docs already in the repo:
- [`collaboration.md`](collaboration.md) — the basic Git command cheat-sheet.
- [`data_exploration.md`](data_exploration.md) — what the data is and how Stage 0 was built (**read this first**).

If you are a new collaborator, read this document top-to-bottom **once** before writing any code, then
follow [Part 2](#part-2--one-time-setup) exactly.

---

## Part 0 — How remote collaboration actually works here

We never share a computer or a hard drive. Everyone works on their **own laptop at home**, and the
**single source of truth is the GitHub repository** (`github.com/Yashtadi/BMS`). The model is simple:

```
   Your laptop                     GitHub (the hub)                 Teammate's laptop
 ┌───────────────┐   git push    ┌──────────────────┐   git pull   ┌────────────────┐
 │ code + your   │ ────────────▶ │  code only        │ ───────────▶ │ code + her     │
 │ local data    │ ◀──────────── │ (NO data files)   │ ◀─────────── │ local data     │
 └───────────────┘   git pull    └──────────────────┘   git push   └────────────────┘
```

Two rules make this work:

1. **Only code and documents live in Git.** The dataset does **not** — it is far too large and is
   deliberately git-ignored. Each person keeps their **own local copy** of the data.
2. **The data is reproducible from code.** Because the raw KIT files are public and our Stage 0
   pipeline is deterministic, anyone can **regenerate the identical working dataset** by downloading
   the raw files once and running the pipeline. We never email Parquet files around.

So collaboration = *sharing code through GitHub*, and *each person regenerating the data locally from
the same public source using the same code.*

---

## Part 1 — Prerequisites (install once)

| Tool | Why | Notes |
|---|---|---|
| **Git** | Version control | [git-scm.com](https://git-scm.com) |
| **Miniconda** | Python environment manager | [Miniconda](https://www.anaconda.com/download/success). During install, **tick "Add to PATH"**. |
| **VS Code** | Editor + notebooks | Install the **Python** and **Jupyter** extensions. |
| **A GitHub account** | Access to the repo | Ask the repo owner to add you as a **collaborator** (Settings → Collaborators). |

---

## Part 2 — One-time setup

Do these steps in order. Budget ~30–45 minutes (most of it is the dataset download).

### 2.1 Clone the repository and create your branch

```bash
git clone https://github.com/Yashtadi/BMS.git
cd BMS
git checkout -b <your-name>-ecm        # e.g. priya-ecm — your own working branch
git push -u origin <your-name>-ecm
```

> **Never commit to `main`.** Always work on your own branch (see [Part 4](#part-4--daily-git-workflow)).

### 2.2 Create the Python environment

We use a conda environment named **`bms`** (Python 3.11). Create it from the spec file:

```bash
conda env create -f environment.yml
conda activate bms
```

> If `environment.yml` does not exist yet, create it at the repo root with this content, then run the
> command above:
>
> ```yaml
> name: bms
> channels:
>   - conda-forge
> dependencies:
>   - python=3.11
>   - numpy
>   - pandas
>   - scipy
>   - matplotlib
>   - jupyter
>   - ipykernel
>   - pyarrow          # Parquet read/write
>   - openpyxl         # read the KIT metadata .xlsx
>   - pip
>   - pip:
>       - impedance    # EIS / equivalent-circuit fitting (Stage 1)
> ```

Register the environment as a notebook kernel:

```bash
python -m ipykernel install --user --name bms
```

### 2.3 Download the raw dataset (once)

The working dataset is **not** in Git. Download the raw KIT dataset yourself:

1. Go to the **RADAR4KIT** record for **DOI `10.35097/1969`** (the *"Comprehensive battery aging
   dataset"*, LG INR18650HG2).
2. Download the archive (~333 MB, a `.tar`) and unpack it.
3. Copy the unpacked folder into the repo under **`data/raw/`** so the path looks like:

   ```
   data/raw/10.35097-1969/10.35097-1969/data/dataset/
       ├── cell_eisv2.zip
       ├── cell_eocv2.zip
       ├── Cycling Experiment Cell Overview (2024-09-26 15-31).xlsx
       └── … (other files)
   ```

4. **Extract the two zips we use**, in place, so their CSVs sit in sibling folders:

   ```powershell
   # from the dataset folder above
   Expand-Archive cell_eisv2.zip -DestinationPath cell_eisv2
   Expand-Archive cell_eocv2.zip -DestinationPath cell_eocv2
   ```

You should now have `…/dataset/cell_eisv2/` (240 EIS CSVs) and `…/dataset/cell_eocv2/` (240 capacity
CSVs).

> Everything under `data/` is git-ignored, so none of this will ever be committed. That is intended.

### 2.4 Regenerate the working dataset

Rebuild the Stage 0 output (`data/interim/eis_labeled.parquet`) from the raw files using the shared
pipeline. Save this as **`build_dataset.py`** at the repo root and run `python build_dataset.py`:

```python
"""Rebuild the Stage 0 working dataset from the raw KIT files."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.prep.load_eis import load_all_cells
from src.prep.metadata import build_enriched_table
from src.prep.capacity import load_capacity_labels, attach_capacity_labels

DS = ROOT / "data/raw/10.35097-1969/10.35097-1969/data/dataset"
INTERIM = ROOT / "data/interim"

load_all_cells(DS / "cell_eisv2", save_path=INTERIM / "eis_spectra.parquet")
enriched = build_enriched_table(
    INTERIM / "eis_spectra.parquet",
    DS / "Cycling Experiment Cell Overview (2024-09-26 15-31).xlsx",
    save_path=INTERIM / "eis_enriched.parquet",
)
caps = load_capacity_labels(DS / "cell_eocv2")
labeled = attach_capacity_labels(enriched, caps)
labeled.to_parquet(INTERIM / "eis_labeled.parquet", index=False)
print("Built eis_labeled.parquet:", labeled.shape,
      "| cells:", labeled["cell_id"].nunique())
```

### 2.5 Verify your setup

Your rebuild is correct if you see **exactly** these numbers:

| Check | Expected |
|---|---|
| Cells | **228** |
| Spectra | **39,368** |
| Rows | **1,102,304** |
| Points per spectrum | all **28** |
| `soh_cap` non-null | **100 %** |

If your numbers differ, **stop and ask** before building on top — a mismatch means the raw data or the
environment is off, and we must be identical here.

---

## Part 3 — The golden rules for data

1. **Never `git add` anything under `data/`.** It is git-ignored; keep it that way. If `git status`
   ever shows a `.csv`, `.parquet`, or a `data/` path, do **not** commit — tell the team.
2. **Never email or Slack Parquet/CSV files to each other.** If someone's data looks wrong, they
   re-run `build_dataset.py`. Sharing files by hand is how two people silently end up training on
   different data.
3. **The pipeline is the contract.** If you change how the data is built, change it **in
   `src/prep/`**, open a PR, and everyone re-runs `build_dataset.py` after merging. The code defines
   the data — not a file on someone's disk.
4. **Set seeds and record them.** Any randomness (splits, model init) uses a fixed seed committed in
   `configs/`, so results are reproducible across machines.

---

## Part 4 — Daily Git workflow

We work in parallel on separate branches and integrate through pull requests. Every working session:

```bash
# 1. Start from the latest shared code
git checkout main
git pull origin main

# 2. Move to your branch and bring in the latest main
git checkout <your-name>-ecm
git merge main

# 3. …do your work, committing in small logical steps…
git add <specific files>          # add files by name, not "git add ."
git commit -m "ecm: add Kramers–Kronig residual check"

# 4. Push and open a Pull Request on GitHub
git push origin <your-name>-ecm
```

**Rules (also in [`collaboration.md`](collaboration.md)):**
- 🚫 Never push to `main`. 🚫 Never force-push. 🚫 Never work on someone else's branch.
- ✅ Pull `main` **before** you start and **before** you open a PR (fewer conflicts).
- ✅ Keep commits **small and focused**, with clear messages (`area: what changed`).
- ✅ **Open a PR and get one review** before merging. **Do not merge your own PR** without a review.

---

## Part 5 — Dividing the ECM work without collisions

The fastest way to a merge nightmare is two people editing the same file. We avoid it by **agreeing on
interfaces first, then owning separate modules.**

### 5.1 Agree the data contract *before* coding
Stage 1 turns each spectrum into features. Before splitting work, agree **on paper** (in a GitHub
issue) on:
- **Input:** one spectrum = the 28 rows of `eis_labeled.parquet` sharing a
  `(cell_id, cu_index, soc_nom, is_rt)` key, with `freq_Hz, Z_real_mOhm, Z_imag_mOhm`.
- **Output:** the **feature-table schema** — one row per spectrum, columns
  `[cell_id, cu_index, soc_nom, is_rt, R_ohm, R_ct, Q, n, sigma, …ΔZ features…, soh_cap]`.

Once the *shape of the hand-off* is fixed, each person can build their side independently.

### 5.2 Suggested module ownership
These are **separate folders**, so two people rarely touch the same file:

| Module | Responsibility | Natural owner |
|---|---|---|
| `src/validation/` | Kramers–Kronig validation (flag/reject bad spectra) | Person A |
| `src/ecm/` | Circuit fitting → `R_ohm, R_ct, CPE, Warburg` | Person A |
| `src/features/` | ΔZ baselines + assemble the final feature table | Person B |
| `notebooks/` | **One notebook per person** (see [Part 6](#part-6--notebooks-vs-modules)) | each |

Reusable logic **always** lives in `src/…`; notebooks only *call* it. That keeps merges to Python
files (easy) instead of notebooks (hard).

### 5.3 Talk in issues
Use **GitHub Issues** to track tasks and decisions ("Which Warburg element?", "ΔZ baseline = first
valid CU?"). Decisions get recorded there and in the relevant `.md` — not lost in chat.

---

## Part 6 — Notebooks vs modules (and notebook hygiene)

Notebooks are wonderful for exploration and **terrible for Git** — they are large JSON files with
embedded outputs that produce ugly, unmergeable conflicts.

**Our policy:**
- **Explore** in a notebook, but the moment logic works, **move it into a `src/…` module** with a
  proper function, docstring, and units. Notebooks should mostly be a few `import` + call cells.
- **One notebook per person.** Never have two people editing the same `.ipynb` on different branches.
- **Restart & Run All before committing** a notebook, and keep them small. (Optional but recommended:
  clear heavy outputs so diffs stay readable.)
- If you and your teammate must both prototype the same idea, do it in **separate notebooks**
  (`03_ecm_priya.ipynb`, `03_ecm_ana.ipynb`) and reconcile into a module.

---

## Part 7 — Code quality standards

These are the things that keep this project trustworthy. They are **non-negotiable** because a
confidently-wrong battery-health model is worse than no model.

1. **No leakage, ever.** Train/test splits group by **physical cell** (`cell_id`). Never random-split
   rows of a spectrum or a cell. This is the single most important rule.
2. **State units and assumptions.** Every function declares units in its docstring (mΩ, Hz, °C, %) and
   what it assumes. Impedance is in **mΩ**; confirm the `Z_imag` sign convention before fitting.
3. **Physics first.** Sanity-check every feature against battery physics: resistances are non-negative
   and grow/hold with age; SOH is (near-)monotonic non-increasing; fits with implausible parameters
   are rejected, not kept.
4. **Guard your joins and merges.** After any `merge`, assert the row count is what you expect and that
   nothing silently duplicated (we were bitten by this in Stage 0 — see `data_exploration.md`).
5. **Fail loudly.** Prefer explicit checks and clear exceptions over silently returning wrong numbers.
6. **Keep it configurable.** Circuit topology, feature choices, thresholds, and seeds live in
   `configs/`, not hard-coded — so we can iterate and compare.
7. **Match the surrounding style.** Same naming, docstring format, and structure as the existing
   `src/prep/` modules.

---

## Part 8 — Code review & "Definition of Done"

Every PR is reviewed by the other person before merging. A piece of work is **done** when:

- [ ] Logic lives in a `src/…` module (not only in a notebook), with a docstring stating **units +
      assumptions**.
- [ ] It runs end-to-end on a fresh `build_dataset.py` output.
- [ ] **No `data/` files** are staged; `git status` is clean of data.
- [ ] Splits group by `cell_id` (no leakage) wherever modelling/evaluation is involved.
- [ ] Any `merge`/join has a row-count / uniqueness guard.
- [ ] A quick **physics sanity check** was run and noted in the PR description (e.g. "R_ct grows with
      age; fits with R<0 rejected").
- [ ] The relevant doc is updated (`data_exploration.md`, `README.md`, or a new stage doc) if behaviour
      or the data contract changed.
- [ ] The PR is **small and focused**, with a clear description of *what* and *why*.

**Reviewer's job:** actually pull the branch, run it, and check the physics — not just skim the diff.

---

## Part 9 — Communication & cadence

- **GitHub Issues** for tasks, questions, and decisions (searchable, permanent).
- **Pull Request descriptions** explain *what changed and why*, and how it was sanity-checked.
- A short **weekly sync** (call) to agree the next slice of work and unblock each other.
- When a design decision is made, **record it** in the relevant `.md` (with the reasoning and what
  would make us revisit it) — chat messages get lost.

---

## Part 10 — Troubleshooting / FAQ

| Symptom | Likely cause & fix |
|---|---|
| `conda: command not found` | Miniconda not on PATH. Reinstall with "Add to PATH", **reopen** the terminal, or run `conda init powershell`. |
| `ModuleNotFoundError: No module named 'src'` | Notebook can't see the repo root. Add `import sys; sys.path.insert(0, "..")` before importing, and run from `notebooks/`. |
| `ImportError: … pyarrow …` on `read_parquet` | Wrong environment. `conda activate bms` (and select the **bms** kernel in the notebook). |
| Edited a `.py` but the notebook ignores it | Kernel cached the old import. **Restart the kernel**, or add `%load_ext autoreload` + `%autoreload 2` at the top. |
| Verify numbers don't match Part 2.5 | Raw data placed/extracted wrong, or a stale `data/interim`. Recheck the `data/raw/.../dataset/` layout and re-run `build_dataset.py`. |
| A big file (data) shows up in `git status` | **Do not commit.** Confirm it's under `data/` and that `.gitignore` covers it; tell the team. |
| Notebook merge conflict | Expected — notebooks don't merge. Keep logic in modules; if it happens, one person re-does their cells rather than hand-merging JSON. |

---

### TL;DR for a new collaborator
1. Get added to the repo → clone → make your own branch.
2. `conda env create -f environment.yml` → `conda activate bms`.
3. Download the raw KIT data (DOI `10.35097/1969`) into `data/raw/`, extract the two zips.
4. Run `python build_dataset.py` and confirm **228 cells / 1,102,304 rows**.
5. Read [`data_exploration.md`](data_exploration.md), pick a module from [Part 5](#part-5--dividing-the-ecm-work-without-collisions), and open small PRs with reviews.

Welcome aboard — build carefully, keep it physical, and never split a cell across train and test. 🔋
