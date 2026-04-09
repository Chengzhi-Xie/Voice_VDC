# Disorder Classification Final README

## Overview
This folder contains the final multiclass disorder-classification package for the VOICED disease-only setting. It focuses on three pathological classes:

- `hyperkinetic`
- `hypokinetic`
- `laryngitis`

The workflow supports:

- supervised multiclass classification with 6 models
- unsupervised clustering with 3 methods
- subgroup analysis by `age`, `gender`, `RSI`, and `VHI`

The notebook is built around a pre-extracted acoustic feature CSV and reports record-level results.

## Recommended Anaconda Environment
Open a terminal in this folder, then create or activate a Conda environment such as `phm5015_voiced`.

Example setup:

```powershell
conda create -n phm5015_voiced python=3.10 -y
conda activate phm5015_voiced
pip install catboost scikit-learn pandas numpy matplotlib seaborn scipy jupyter
```

If you already have the environment, only activation is needed:

```powershell
conda activate phm5015_voiced
```

## Folder Layout
- `./voice_disorder_classification.ipynb`
  Main notebook for multiclass supervised learning, clustering, visualization, and subgroup analysis.
- `./results/`
  - `part1_supervised_summary.csv`: 5-fold out-of-fold supervised metrics
  - `part2_clustering_summary.csv`: clustering evaluation metrics
  - `subgroup_predictions.csv`: CatBoost OOF predictions with subgroup metadata
- `./figures/`
  - `fig1_disease_embedding.png`: disease feature space visualization
  - `fig2_tsne_cluster_alignment.png`: true-label vs cluster-alignment view
  - `fig3_confusion_matrices.png`: confusion matrices for supervised models
  - `fig_subgroup_age.png`
  - `fig_subgroup_gender.png`
  - `fig_subgroup_rsi.png`
  - `fig_subgroup_vhi.png`
- `./RUN_FLOW.md`
  Step-by-step execution notes.
- `./RESULTS_SUMMARY.md`
  Detailed metric summary and conclusions.
- `./README.md`
  This final consolidated entry point.

## Input Data Requirement
The notebook expects one pre-extracted acoustic feature CSV as input.

Current assumptions:
- the input can be either record-level or window-level
- if window-level input is provided, the notebook aggregates it to record-level by mean pooling
- the feature backbone is the same handcrafted acoustic representation used in the earlier VOICED work, centered on approximately 219 acoustic dimensions

Recommended relative-path workflow:
- keep the notebook in `./`
- keep the final feature CSV either in `./` or upload it directly inside the notebook session

Examples:
- local Jupyter run: place `binary_window_features_sr16000_w1p0_h0p25.csv` in `./`
- Colab run: upload the CSV interactively when prompted

## How To Run
Two practical options are supported.

### Option 1: Run in Jupyter locally
Start Jupyter from this folder:

```powershell
jupyter notebook
```

Then open:

```text
./voice_disorder_classification.ipynb
```

### Option 2: Run in Google Colab
Upload `./voice_disorder_classification.ipynb` to Colab and execute cells in order.

## Recommended Execution Flow
1. Install dependencies inside the notebook if needed.
2. Import all required libraries.
3. Load or upload the feature CSV.
4. Confirm feature dimensions and filter to disease-only records.
5. Run Part 1: supervised multiclass classification with 5-fold OOF.
6. Run Part 2: unsupervised clustering with `k=3`.
7. Generate all figures into `./figures/`.
8. Run subgroup analysis and export subgroup outputs into `./results/`.

Generated outputs are saved to relative paths only:
- `./results/part1_supervised_summary.csv`
- `./results/part2_clustering_summary.csv`
- `./results/subgroup_predictions.csv`
- `./figures/*.png`

## Core Results
### Supervised Multiclass Classification
Best model by balanced accuracy and macro F1:
- `CatBoost`
- OOF ACC: `0.5298`
- Balanced ACC: `0.5172`
- Macro F1: `0.5176`
- Macro OvR AUC: `0.6970`

Other important supervised observations:
- Highest OOF ACC: `ExtraTrees = 0.5364`
- Highest Macro OvR AUC: `RandomForest = 0.7121`

### Unsupervised Clustering
Best clustering method overall:
- `GMM (k=3)`
- ARI: `0.0422`
- NMI: `0.0865`
- Purity: `0.5166`
- Aligned ACC: `0.4503`
- Aligned Balanced ACC: `0.4734`

Interpretation:
- clustering quality is weak and should be treated as exploratory only
- the supervised setting is the main result line for this project

### Subgroup Highlights
Best-performing subgroup findings from CatBoost OOF predictions:
- age `< 38`: ACC `0.605`, Balanced ACC `0.593`
- high RSI (`>= 13`): ACC `0.567`, Balanced ACC `0.587`
- gender split was broadly similar between female and male groups
- severe VHI (`61+`) showed only moderate gains

## Most Important Files To Read
- `./README.md`
- `./RUN_FLOW.md`
- `./RESULTS_SUMMARY.md`
- `./results/part1_supervised_summary.csv`
- `./results/part2_clustering_summary.csv`
- `./results/subgroup_predictions.csv`
- `./figures/fig3_confusion_matrices.png`
- `./figures/fig2_tsne_cluster_alignment.png`

## Final Recommendation
If you need one final supervised multiclass result from this folder, report:

- `CatBoost` as the primary supervised model
- `GMM (k=3)` as the exploratory unsupervised baseline

The main takeaway is that supervised acoustic classification is usable but still difficult, while unsupervised clustering does not reliably recover the clinical disorder labels.
