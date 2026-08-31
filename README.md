# Saliency-Based Explainability of Antigen Specificity in mvTCR

Feature-level attribution for the [mvTCR](https://github.com/SchubertLab/mvTCR)
joint latent space. Given a T cell embedded from paired transcriptome and
receptor sequence, which genes and which CDR3 positions place it in its
antigen's region of that space?

Course project, M.Sc. Computer Science; Deep Learning in the Clinic.

---

## Main result

Integrated Gradients needs a **baseline**: a reference input each cell is
compared against. The most common default is the zero vector, and that
default is not neutral.

Attribution scales with `(x - baseline)`, so with a baseline of zero this factor
reduces to raw expression, and the highest-expressed genes in any cell are
mitochondrial and ribosomal transcripts, which mark cell quality rather than
biological state.

|                                              | zero baseline        | mean baseline |
| -------------------------------------------- | -------------------- | ------------- |
| attribution on housekeeping genes             | 14–23% (≈30× their share of the panel) | 2–4% (≈5×) |
| housekeeping genes in a class's top 12        | 5–9                  | 0–3           |
| documented V-segment biases recovered         | **0**                | **3**         |

With a mean-expression baseline the pipeline recovers three documented V gene
biases: *TRBV19* for influenza GILGFVFTL (top-ranked gene of any kind),
*TRAV12-2* for melanoma ELAGIGILTV, and *TRAV5* for EBV GLCTLVAML. The encoder
only ever sees the CDR3 junction and never V-segment identity, so this is a
control the method could have failed. None of the three appears under the zero
baseline.

---

## Files

```
attribution_core.py        wrapper, prototypes, IG, normalisation, aggregation
test_attribution_core.py   33 unit tests against a mock model — no GPU, seconds
01_run_analysis.py         one baseline, end to end
02_compare_baselines.py    zero vs mean comparison — no GPU
run_ig.sh                  SLURM array: task 0 = zero, task 1 = mean
environment.yml            pinned environment
```

This is analysis code only. The data, the pretrained checkpoints and the mvTCR
package itself are not redistributed here.

---

## Where these files go

The scripts use relative paths (`../mvTCR/`, `../saved_models/`, `../results/`),
so they are meant to be dropped into a checkout of the upstream reproducibility
repository:

```
mvTCR_reproducibility/
├── mvTCR/                 # pip --target install of mvtcr==0.1.3
├── saved_models/          # Zenodo checkpoints (record v4)
├── results/               # written by 01_run_analysis.py
├── figures/               # written by 01_run_analysis.py
└── interpretability/      # <- clone this repo here
```

You will need:

| | source |
| --- | --- |
| 10x CD8⁺ dextramer data (`v7_avidity.h5ad`) | [10x Genomics datasets](https://www.10xgenomics.com/datasets) |
| pretrained mvTCR checkpoints | Zenodo `10.5281/zenodo.10634209` — **record v4** |
| the `mvTCR` package | [SchubertLab/mvTCR](https://github.com/SchubertLab/mvTCR), version 0.1.3 |

---

## Setup

The pins are not optional. Each of the points below cost time to discover.

**Pin `mvtcr==0.1.3`.** `main` has since been refactored (renamed module,
Python 3.10 / PyTorch 2.x) and is incompatible with both the published
checkpoints and the evaluation notebooks.

**Install into the repo, not site-packages:**
`pip install mvtcr==0.1.3 --target ./mvTCR`, so the package sits beside `data/`.
A normal install lets a site-packages copy shadow the local one, producing
import errors that look like data problems.

**`python-igraph` must be 0.9.11** for `leidenalg` 0.8.4.

**Use Zenodo record v4**, not v1 — v1 has donors missing and a folder layout the
notebooks do not expect.

---

## Running

```bash
# 1. unit tests — login node, no GPU, seconds
python test_attribution_core.py                 # expect 33/33 passed

# 2. both baselines — SLURM array, ~2.3 h and ~3.6 h in parallel on an A100
export MVTCR_ENV=/path/to/your/conda/env        # or just the env name
sbatch run_ig.sh

# 3. comparison — login node, no GPU
python 02_compare_baselines.py 2>&1 | tee comparison.txt
```

`run_ig.sh` runs the unit tests first and aborts if they fail, so a broken build
costs seconds rather than hours. Each split's results are written as it
finishes, so a late failure does not lose completed work.

Single baseline, or a subset of splits:

```bash
python 01_run_analysis.py --baseline mean --splits 0 1
```

**Note on outputs:** `01_run_analysis.py` writes raw per-cell attributions to
`../results/interpretability/{baseline}/attr_split*.npz`. These are ~2.5 GB per
split, ~25 GB for both baselines. `.gitignore` excludes them; keeping them lets
you re-aggregate without recomputing.

---

## Method

**Joint embedding.** `z = ½(μ_RNA + μ_TCR)`, deterministic means. The reference
implementation computes this under `torch.no_grad()`, so the encoder path is
reimplemented as a differentiable module. Equivalence to `get_latent()` is
asserted numerically at the start of every run.

**Target.** Scoring a cell by its distance to its own class prototype returns
features common to all T cells. We instead score how much *closer* it is to its
own class than to the others; the quadratic terms cancel, leaving a linear
projection onto the axis separating that antigen from the rest.

**Prototypes.** Clonotype-balanced — cells are averaged within each clone, then
clone means are averaged, so every clone counts once. In three of eight classes
a single clone contributes 31–55% of the cells, so a plain average would
describe one expanded clone rather than the antigen. Computed from training
cells only.

**Discrete inputs.** CDR3s are integer tokens and cannot be interpolated, so
attribution happens at the amino-acid embedding layer, with interpolated
embeddings injected in place of the lookup. Padding embeds to exactly zero, so
padding positions receive exactly zero attribution.

**Normalisation.** Each split is a separately trained VAE with an arbitrary
latent scale. Every cell of a class is divided by that class's median
`|F(x) − F(x')|` — one positive constant per class per split. Dividing by each
cell's own value instead is unstable: cells near a class boundary barely move,
so that denominator approaches zero.

**Verification.** IG guarantees attributions sum to `F(x) − F(x')`. This
residual is computed for every cell; the median relative error is 0.4%.

---

## Scope

Integrated Gradients identifies the features the encoder *uses* to place a cell
in its latent space. It does not identify the molecular determinants of TCR–pMHC
binding. All claims concern the learned representation of the mvTCR model.

---

## Acknowledgements

Built on [mvTCR](https://github.com/SchubertLab/mvTCR) (Drost et al.) and the
`mvTCR_reproducibility` evaluation code. Data from 10x Genomics; checkpoints
from the authors' Zenodo archive. Compute provided by the LUIS HPC cluster,
Leibniz University Hannover.
