# Au_gene_markers


Marker discovery pipeline for **_Aerococcus urinaeequi_ (Au)** using **orthogroup (OG) presence/absence** to study associations with **somatic cell count (SCC)**.  
This repo contains scripts to run:

- **Random Forest (RF) biomarker pipeline** (supervised labels + unsupervised RF-proximity clustering)
- **Elastic Net logistic regression** with **repeated cross-validation** (stable marker selection)
- Optional **TOI (Target of Interest)** reporting (e.g., focus on *Low SCC*)

---

## 1) Project structure

```

Au_gene_markers/
├── README.md
├── biomARkers_multiclass.py
├── rfbiomarker_multiclass.py
├── elastic_net_logistic_repeatedcv.py
├── elastic_net_multiclass_repeatedcv_with_toi.py
└── elastic_net_scc.py

```

> Notes:
> - `biomARkers_multiclass.py` is the main RF entry script.
> - `rfbiomarker_multiclass.py` contains RF logic + clustering + reporting.
> - `elastic_net_logistic_repeatedcv.py` is for **binary** SCC (e.g., Low vs High).
> - `elastic_net_multiclass_repeatedcv_with_toi.py` is for **3-class SCC** and outputs TOI-focused marker summaries.
> - `elastic_net_scc.py` contains helper utilities for SCC-oriented runs.

---

## 2) Input files

### 2.1 Biomarkers input matrix (required)
A tab-delimited file with:

- first column: `sample`
- one column per orthogroup: `OG000xxxx` values in `{0,1}`
- target column: `target_column` (SCC labels)

Example header:

```

sample    OG0000001  OG0000002  ...  target_column

````

### 2.2 SCC labels (used in this repo)
We used **three SCC categories**:

- `level_2` = Low SCC  
- `level_3` = Middle SCC  *(25 < SCC < 200)*  
- `level_1` = High SCC

> Keep your labels consistent across scripts (or map them before running).

---

## 3) Conda environment (recommended)

These scripts use common scientific Python packages. A clean environment is recommended on HPC.

```bash
conda create -n biomarkers python=3.11 -y
conda activate biomarkers
conda install -c conda-forge numpy pandas scipy scikit-learn -y
````

---

## 4) Random Forest biomARkers (multiclass)

This reproduces the RF workflow while allowing **3 SCC labels**.

### Command example

```bash
python biomARkers_multiclass.py \
  --input biomarkers_three_classes.tsv \
  --outdir biomarker_results_SCC3 \
  --targets_col target_column \
  --id_col sample \
  --toi level_2 \
  --min 1 \
  --max 70 \
  --test_size 0.2 \
  --max_biomk 4 \
  --pval 0.01 \
  --lift 1.5
```

### Key outputs (typical)

* `RF_clusters*.tsv` — RF-proximity based unsupervised clusters (e.g., RFhigh vs RFlow)
* `log.txt` / run log — includes top RF features and run parameters
* additional summary tables used for downstream plots (heatmaps, cluster summaries)

### Important concept

Even if RF is trained with **3 supervised SCC labels**, the **RF-proximity clustering** step is **unsupervised** and can yield **2 major genomic clusters** if the data structure supports it.

---

## 5) Elastic Net (binary) — repeated CV

Used for **Low vs High** (recommended when sample size is limited).

```bash
python elastic_net_logistic_repeatedcv.py \
  --input biomarkers_low_vs_high.tsv \
  --outdir enet_low_vs_high \
  --targets_col target_column \
  --id_col sample \
  --toi level_2 \
  --test_size 0.2 \
  --outer_splits 5 \
  --outer_repeats 20 \
  --inner_cv 5
```

### Outputs (used in figures/tables)

* `outerCV_fold_metrics.tsv` — performance per fold
* `outerCV_predictions_all.tsv` — predictions per isolate per split
* `outerCV_predictions_summary.tsv` — aggregated prediction summary
* `elastic_net_logistic_marker_stability.tsv` — marker stability (selection rate across fits)
* `*_summary.txt` — run summary, settings, warnings

**Top stable markers** = OGs repeatedly selected (non-zero coefficient) across many CV fits.

---

## 6) Elastic Net (multiclass) + TOI

Used when you want **Low/Middle/High** simultaneously, and still summarize markers for a **TOI**.

```bash
python elastic_net_multiclass_repeatedcv_with_toi.py \
  --input biomarkers_three_classes.tsv \
  --outdir enet_SCC3 \
  --targets_col target_column \
  --id_col sample \
  --toi level_2 \
  --test_size 0.2 \
  --outer_splits 5 \
  --outer_repeats 20 \
  --inner_cv 5
```

Interpretation (TOI = Low SCC):

* **Positive coefficient** → enriched in Low SCC
* **Negative coefficient** → depleted in Low SCC
* **Selection rate** → stability across repeated CV fits

---

## 7) Parameter glossary

* `--targets_col`: target label column (e.g., `target_column`)
* `--id_col`: sample ID column (e.g., `sample`)
* `--toi`: *Target of interest* label used for focused reporting (e.g., `level_2` for Low SCC)
* `--test_size 0.2`: 20% held out for testing each split (80% training)
* `--outer_splits`: number of outer CV folds
* `--outer_repeats`: how many times to repeat the outer CV
* `--inner_cv`: folds used inside training for model selection

---

