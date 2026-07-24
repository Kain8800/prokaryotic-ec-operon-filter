*English · [Русский](README.ru.md)*

# EC-number prediction for prokaryotic enzymes with operon-context verification

A binary operon filter as post-processing for EC-number predictions from two
deep-learning models: **ProteInfer** (Sanderson et al., 2023) and
**CLEAN** (Yu et al., 2023). Master's thesis.

## Overview

Deep-learning models predict enzyme EC numbers from the amino-acid sequence, but
they make confident errors, especially on low-homology proteins. This work
proposes a **verification filter based on operon context**. The operon context
of a protein is the function of its operon neighbors: their EC numbers and the
KEGG metabolic pathways they participate in. A predicted EC number is accepted if
at least one of two conditions holds: it participates in a KEGG pathway shared
with an operon neighbor, or it matches a neighbor's EC number. For proteins
without operon context (no neighbors with a known function) the filter keeps the
prediction as is.

The filter relies on the functional homogeneity of operons: in the vast majority
of operons, neighboring genes participate in shared KEGG metabolic pathways.

Evaluation is stratified by homology to the training set (HIGH > 70%,
MEDIUM 40–70%, LOW < 40%, NONE: no match). The stratum boundaries reflect how
much a prediction can rely on the similarity of the evaluation protein to the
training data: above 40% identity to the nearest training protein the EC number
(first three digits) is already determined by similarity (Tian & Skolnick, 2003),
while below 40% similarity stops being a reliable basis; in the 40–70% range
(MEDIUM) the first three digits are reliable, and above 70% (HIGH) the full
four-digit function is preserved (Rost, 2002).

The filter does not depend on which model produced the prediction and is applied
identically to both; the confidence threshold τ is chosen by the criterion of
maximum macro-F1 **after** the filter is applied.

## Key result

On an evaluation set of 51,890 proteins (30,977 operons, 2,095 organisms) the
operon filter on top of ProteInfer at τ = 0.90 (threshold choice justified in
"Confidence-threshold selection") raises macro-F1 from 0.807 to **0.887** and
accuracy from 0.967 to 0.982, while coverage drops from 0.879 to 0.716.

Throughout, the main model is ProteInfer: in the region of comparable coverage it
outperforms CLEAN, and it is on ProteInfer that the filter gives a stable gain.
The CLEAN analysis and the model comparison are moved to "A limitation of the
threshold-selection criterion".

The gain is non-negative across all homology strata, and it is larger the lower
the test protein's homology to the training data:

| Stratum | F1 no filter | F1 + filter | ΔF1 | Coverage |
|---|---|---|---|---|
| HIGH | 0.8595 | 0.9273 | +0.0679 | 0.956 → 0.757 |
| MEDIUM | 0.7853 | 0.8443 | +0.0590 | 0.833 → 0.705 |
| LOW | 0.7369 | **0.8608** | **+0.1239** | 0.375 → 0.276 |
| ALL | 0.8073 | 0.8870 | +0.0797 | 0.879 → 0.716 |

Among predictions that pass the confidence threshold τ = 0.90, the error rate is
3.3%, and the operon filter rejects 56.9% of these errors. Detection
effectiveness grows with the error rate: from 60.3% on the HIGH stratum to
**88.0% on LOW**. The filter is most useful where the model is most prone to
error, so it carries a signal independent of model confidence.

Detailed metrics by EC-hierarchy level and enzyme class are in the thesis
(`KARMANOV_VR_01061.pdf`) and in the plots under `docs/`.

## Confidence-threshold selection

Both models return a numeric confidence score together with the EC number. The
threshold τ cuts off unconfident predictions, raising accuracy at the cost of
coverage. `7_EVALUATION.py` runs a sweep over a grid of twelve values and picks
the threshold by the criterion of maximum macro-F1 after the operon filter; the
criterion is the same for both models.

```python
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
```

The upper bound 0.995 is set by CLEAN's ceiling: the model's maximum score in
`query_maxsep.csv` is 0.9989, so a threshold of 0.999 yields an empty prediction
set.

Sweep result for ProteInfer:

| τ | F1 no filter | F1 + filter | Coverage no filter | Coverage + filter |
|---|---|---|---|---|
| 0.00 | 0.6366 | 0.8432 | 0.944 | 0.757 |
| 0.30 | 0.7982 | 0.8747 | 0.892 | 0.725 |
| 0.50 | 0.7980 | 0.8791 | 0.888 | 0.723 |
| 0.60 | 0.7994 | 0.8793 | 0.886 | 0.721 |
| 0.70 | 0.8018 | 0.8807 | 0.885 | 0.720 |
| 0.80 | 0.8057 | 0.8836 | 0.882 | 0.719 |
| **0.90** | 0.8073 | **0.8870** | 0.879 | **0.716** |
| 0.95 | 0.8099 | 0.8854 | 0.876 | 0.714 |
| 0.97 | 0.8116 | 0.8866 | 0.874 | 0.713 |
| 0.98 | 0.8125 | 0.8865 | 0.872 | 0.711 |
| 0.99 | 0.8132 | 0.8862 | 0.869 | 0.709 |
| 0.995 | 0.8170 | 0.8843 | 0.866 | 0.707 |

**τ = 0.90** is selected. This is an interior maximum of the grid: at every
stricter threshold the post-filter F1 is lower.

The no-filter optimum and the with-filter optimum do not coincide, so comparing at
a single operating point understates the no-filter result. Comparing each
configuration at its own optimum (0.8170 without the filter at τ = 0.995, 0.8870
with it at τ = 0.90) gives a gain of **+0.0700**, versus +0.0797 at a single
operating point. The value remains an upper bound: the no-filter F1 has not
plateaued and keeps rising at the edge of the grid.

## Why the filter is not equivalent to a stricter threshold

A natural objection: the operon filter rejects predictions just like raising the
threshold, so its effect reduces to a stricter τ. The data refute this.

**ProteInfer's confidence distribution is degenerate.** The median top-1
confidence is 1.0000, so more than half of the predictions are pinned to the
sigmoid ceiling. As a result, the threshold barely controls coverage: at τ = 0.90,
93.05% of predictions pass; at τ = 0.999, 91.06%. Raising the threshold by nine
hundredths cuts off 2 pp. In terms of the evaluation set, the no-filter model's
coverage has a floor of ≈ 0.86: the threshold cannot push it lower.

**The regions do not overlap.** The operon filter operates at coverage 0.716,
15 pp below the floor reachable by the threshold. At no τ does the no-filter model
enter this region. The filter rejects on a feature independent of model confidence
(the numeric score that τ acts on), and therefore works where confidence no longer
distinguishes anything.

This is consistent with the error analysis: 88.0% of the model's errors on the LOW
stratum are rejected by the filter, even though all of them passed the threshold
τ = 0.90.

## A limitation of the threshold-selection criterion

The criterion "maximum macro-F1 after the filter" **does not constrain coverage**,
so on a wide enough grid it degenerates: the optimum drifts to strict thresholds
where F1 rises by rejecting predictions. This shows up clearly on CLEAN.

| Model | τ | F1 + filter | Coverage + filter |
|---|---|---|---|
| ProteInfer | 0.90 | 0.8870 | **0.716** |
| CLEAN | 0.995 | **0.9019** | 0.261 |

Formally CLEAN wins on F1 (+0.0149), but at the cost of dropping three
quarters of the proteins. When sweeping only up to 0.95 this effect did not
appear: the grid's upper bound acted as an implicit coverage constraint.

For this reason the script does not declare a winner by F1 alone and prints both
values. The correct comparison is by Pareto front (`docs/sec21_pareto.png`). In
the region where the models are comparable in coverage, ProteInfer dominates CLEAN
everywhere:

| Coverage | CLEAN + filter | ProteInfer + filter |
|---|---|---|
| ~0.727 | 0.8650 | **0.8870** |
| ~0.713 | 0.8704 | **0.8870** |
| ~0.704 | 0.8719 | **0.8870** |

CLEAN beats ProteInfer on F1 only at coverage ≤ 0.487, which ProteInfer does not
reach. At the limiting threshold τ = 0.995 the models' no-filter F1 are almost
equal (0.8170 vs 0.8172), but ProteInfer keeps coverage 0.866 against CLEAN's
0.340: the same quality at 2.5× the reach.

## Data sources

The data itself is not stored in the repository (excluded via `.gitignore`); the
scripts expect it in the working directory. Data retrieved on 2026-06-16.

- Conserved operons from ODB (OperonDB): <https://operondb.jp/conserved>.
  Processing: `preprocessing/1OPERON_RS.py` (input `conserved_operon.download.txt`).
- Assembly summaries and GFF annotations from NCBI RefSeq:
  - <https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt>
  - <https://ftp.ncbi.nlm.nih.gov/genomes/refseq/archaea/assembly_summary.txt>

  Processing: `preprocessing/2PARS_REFSEQ_PROK.py`, `preprocessing/3GFF_PARSER.py`.
- Sequences and EC numbers from UniProtKB via the REST API:
  <https://rest.uniprot.org/uniprotkb/stream> (fields `accession, organism_id,
  gene_names, xref_refseq, ec, sequence`). Scripts
  `preprocessing/4UNIPROT_DOWNLOAD.py`, `preprocessing/5UNIPROT_FILTER.py`.
- EC ↔ metabolic-pathway mapping from KEGG:
  <https://rest.kegg.jp/link/ec/pathway>. The script
  `preprocessing/KEGG_PATHWAY_EC_DOWNLOAD.py` builds `pathid_ec.tsv`; this file is
  small and is stored in the repository (`preprocessing/pathid_ec.tsv`). KEGG
  access is for academic use only, ≤ 3 requests/sec.

## Pipeline

The scripts are numbered by stage and run in order. The canonical exchange format
between steps is Apache Parquet; external dumps are TSV.

| Step | Script | Purpose | Output |
|----|--------|-----------|-------|
| 0 | `preprocessing/0PATHID_PREPROCESS.py` | Clean `pathid_ec.tsv`: remove three global KEGG maps (01100, 01110, 01120) that create spurious functional connectivity. Idempotent (backup `pathid_ec.raw.tsv`). | `pathid_ec.tsv` |
| 1 | `preprocessing/1OPERON_RS.py` | Parse ODB operons, explode genes into long format, drop operons with < 2 genes. | `operon_genes_RS.parquet`, `taxids.txt` |
| 2 | `preprocessing/2PARS_REFSEQ_PROK.py` | Select the best genome assembly per organism from `assembly_summary`. | `selected_assemblies_with_gff.parquet` |
| 3 | `preprocessing/3GFF_PARSER.py` | Download and parse GFF, map `locus_tag` → RefSeq `protein_id`. | `refseq_locus_to_protein.parquet` |
| 4 | `preprocessing/4UNIPROT_DOWNLOAD.py` | Batch download from UniProtKB (checkpoints, retry). | `uniprot_with_refseq.tsv` |
| 5 | `preprocessing/5UNIPROT_FILTER.py` | Filter UniProt: only RefSeq of the form `WP_`, EC normalization, deduplication. | `uniprot_with_refseq_clean.parquet` |
| 6 | `preprocessing/6MERGE_FINAL.py` | Merge operons + RefSeq mapping + UniProt; drop operons with < 2 proteins. | `merged_operon_uniprot_refseq_filtered.parquet` |
| 7 | `7_EVALUATION.py` | Build the evaluation set, stratify by homology (MMseqs2), evaluate ProteInfer and CLEAN, apply the operon filter, select the threshold, produce plots and metrics. | tables and plots in `evaluation2/results1/` |

The shared code of step 7 (building the evaluation set, stratification, the
operon filter and metrics) lives in `eval_common.py`. A separate utility
`preprocessing/KEGG_PATHWAY_EC_DOWNLOAD.py` fetches all "pathway ↔ EC" pairs with a
single request to KEGG REST `link/ec/pathway`, normalizes the map-variant prefix to
`rn` (the five-digit number is the same for map/ko/ec/rn) and stores them as
`pathid_ec.tsv` with columns `Path` and `EC_Numbers`; run it on a machine with
internet access.

`7_EVALUATION.py` additionally uses the model predictions, which
**are included in the repository**: ProteInfer (`proteinfer_predictions2.tsv`,
22 MB) and CLEAN (`query_maxsep.csv`, 3 MB). The models' training sets (the
ProteInfer TFRecord and CLEAN's `split100.csv`) are needed for strict leakage
exclusion by sequence match; they are not in the repository because of their size,
but they are publicly available from the model authors and go into the repository
root:

```bash
# ProteInfer: SwissProt (clustered) training set.
# Public bucket of the github.com/google-research/proteinfer project.
wget -O research_proteins_proteinfer_datasets_swissprot_clustered_train.tfrecord \
  https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/datasets/swissprot/clustered/train.tfrecord
# (if the link requires authorization, the same object is available via gsutil:
#  gsutil cp gs://brain-genomics-public/research/proteins/proteinfer/datasets/swissprot/clustered/train.tfrecord .)

# CLEAN: split100 training split (~85 MB), repo github.com/tttianhao/CLEAN.
wget -O split100.csv \
  https://raw.githubusercontent.com/tttianhao/CLEAN/main/app/data/split100.csv
```

The `-O` flag sets the exact filename that `eval_common.py` expects.

## Repository structure

```
.
├── eval_common.py                       shared code of step 7
├── 7_EVALUATION.py                      filter evaluation, threshold selection, plots
├── preprocessing/
│   ├── 0PATHID_PREPROCESS.py
│   ├── 1OPERON_RS.py
│   ├── 2PARS_REFSEQ_PROK.py
│   ├── 3GFF_PARSER.py
│   ├── 4UNIPROT_DOWNLOAD.py
│   ├── 5UNIPROT_FILTER.py
│   ├── 6MERGE_FINAL.py
│   ├── KEGG_PATHWAY_EC_DOWNLOAD.py
│   └── pathid_ec.tsv
├── docs/                                run plots (for the README)
│   ├── sec21_baselines.png
│   ├── sec21_pareto.png
│   ├── sec22_binary_delta_f1.png
│   ├── sec22_f1_by_ec_level.png
│   ├── sec22_ec_class_heatmap.png
│   └── sec24_high_conf_errors.png
├── KARMANOV_VR_01061.pdf
├── requirements.txt
├── SETUP_ENV.md
├── SETUP_ENV.ru.md
├── .gitignore
├── LICENSE
├── README.md
└── README.ru.md
```

The `docs/` directory holds the committed run plots that the README links to. On
each run the script writes its own plots to `evaluation2/results1/`, so a new run's
results do not overwrite the reference ones.

## Environment

Requires **Python 3.10** (`numpy 1.22.3` / `pandas 1.4.2` have no wheels for
3.11+):

```bash
python3.10 -m venv clean_env
source clean_env/bin/activate
pip install -r requirements.txt
```

The external tool is **MMseqs2 v18.8cc5c** (homology stratification in
`7_EVALUATION.py`, run with `-s 7.5`, e-value 10). Pick the build for your CPU
instruction set: on nodes without AVX2 the AVX2 binary crashes with `rc=-4`
(SIGILL), so a static SSE4.1 build is needed. Full install instructions and an
environment check are in `SETUP_ENV.md`.

## Running

```bash
export PROJECT_BASE_DIR=/path/to/repo    # defaults to the current directory
export MMSEQS_BIN=/path/to/mmseqs        # defaults to mmseqs from PATH

python preprocessing/0PATHID_PREPROCESS.py    # cleans the pathid_ec.tsv shipped in the repo
python preprocessing/1OPERON_RS.py
python preprocessing/2PARS_REFSEQ_PROK.py
python preprocessing/3GFF_PARSER.py           # needs internet
python preprocessing/4UNIPROT_DOWNLOAD.py     # needs internet
python preprocessing/5UNIPROT_FILTER.py
python preprocessing/6MERGE_FINAL.py
python 7_EVALUATION.py                        # needs MMseqs2
```

`pathid_ec.tsv` is already in the repository, so step 0 works right after cloning.
To rebuild it from KEGG, run `preprocessing/KEGG_PATHWAY_EC_DOWNLOAD.py` (needs
internet) before step 0.

The scripts `3GFF_PARSER.py`, `4UNIPROT_DOWNLOAD.py` and
`KEGG_PATHWAY_EC_DOWNLOAD.py` reach out to the network, so run them on a machine
with internet access rather than on an isolated compute node.

The MMseqs2 stratification is cached: if `evaluation2/mmseqs_work/result.tsv`
exists and is non-empty, the search is not rerun. When the input data changes, the
cache must be deleted manually.

## Work done after the defense

The thesis captured the state at the time of the defense. When preparing the
repository the pipeline was rerun, and all results reproduced exactly. Beyond the
defended work, the following was added:

- The threshold grid was extended from eight values to twelve (0.97, 0.98, 0.99,
  0.995 added; the upper bound is set by CLEAN's score ceiling of 0.9989). In the
  thesis the sweep stopped at 0.95, so τ = 0.90 was a maximum only within the
  scanned range. The extension showed this is a true interior maximum: at every
  stricter threshold the post-filter F1 is lower. All ProteInfer conclusions held
  unchanged.
- ProteInfer's confidence distribution was analyzed. The median top-1 prediction
  equals 1.0000, and the no-filter model's coverage cannot be pushed below ≈ 0.86
  by the threshold. It follows that the operon filter, operating at coverage
  0.716, rejects on a feature unreachable by the confidence threshold, i.e. it
  does not reduce to a stricter τ.
- A limitation of the threshold-selection criterion was found (see the section
  above): maximum macro-F1 without a coverage constraint degenerates on a wide
  grid. In the thesis the effect did not appear because the upper bound 0.95
  acted as an implicit coverage constraint. The script was updated to print
  coverage next to F1 and not to declare a winner by a single metric.
- The script output was cross-checked against the thesis text. The values in the
  "F1 no filter" column of Table 4 at τ = 0.00 and τ = 0.95 (0.6383 and 0.8052)
  diverged from the script output (0.6366 and 0.8099). These are transcription
  typos in the table: Figure 4 is built by the same script from the same sweep
  and shows the correct values. The other columns of Table 4, as well as
  Tables 5–8, matched fully.

## Thesis

`KARMANOV_VR_01061.pdf` is the full text of the thesis (in Russian,
GOST-formatted), including the methodology, per-stratum metric tables and the
bibliography.

## License

MIT, see [LICENSE](LICENSE).
