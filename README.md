# Voice_VDC
A leakage-aware acoustic–clinical fusion pipeline for robust voice disorder classification on the VOICED dataset, combining handcrafted speech features, clinical metadata, subgroup analysis, and SHAP-based interpretability.

## Overview

This folder is the final delivery summary for the VOICED voice-disorder project. It consolidates the four main project components described in the final report:

- `./binary_classification/`
- `./gaussian_augmentation/`
- `./wav2vec_baseline/`
- `./disorder_classification/`

Together, these four components cover:

- strict binary classification under a leakage-aware evaluation protocol
- Gaussian-balanced data construction and cross-project result comparison
- the baseline handcrafted / wav2vec feature-learning workspace
- three-class disorder classification and subgroup analysis

The recommended way to read this project is:

1. `./binary_classification/` for the strict final binary benchmark
2. `./gaussian_augmentation/` for the final cross-project comparison package
3. `./wav2vec_baseline/` for the early baseline workspace and strongest window-level binary model
4. `./disorder_classification/` for multiclass disease-only experiments

## Shared Environment

All Python-based components were designed around a Conda environment such as `phm5015_voiced`.

Typical setup:

```powershell
conda create -n phm5015_voiced python=3.10 -y
conda activate phm5015_voiced
```

Common packages used across the project:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `joblib`

Additional packages used by specific components:

- `catboost`
- `librosa`
- `wfdb`
- `torch`
- `torchaudio`
- `transformers`
- `tqdm`
- `seaborn`
- `jupyter`

## Top-Level Structure

- `./binary_classification/`
  Final strict binary classification package with supervised and unsupervised models.
- `./gaussian_augmentation/`
  Final report-style comparison package across the main project branches.
- `./wav2vec_baseline/`
  Baseline VOICED workspace with handcrafted features, optional wav2vec features, and early supervised/unsupervised experiments.
- `./disorder_classification/`
  Multiclass disease-only notebook package for `hyperkinetic`, `hypokinetic`, and `laryngitis`.

## Component 1: `./binary_classification/`

### Purpose

This is the strict final binary-classification workspace built on the Gaussian-balanced VOICED setting. It keeps the fixed test split, performs fold-wise Gaussian augmentation, and avoids the optimistic leakage issues present in earlier versions.

### Main Files

- `./binary_classification/src/01_audit_and_build_final_dataset.py`
- `./binary_classification/src/02_strict_supervised_benchmark.py`
- `./binary_classification/src/03_strict_unsupervised_benchmark.py`
- `./binary_classification/results/`
- `./binary_classification/artifacts/`
- `./binary_classification/LEAKAGE_AUDIT.md`

### How To Run

From `./binary_classification/`:

```powershell
conda activate phm5015_voiced
python .\src\01_audit_and_build_final_dataset.py --seed 42 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20
python .\src\02_strict_supervised_benchmark.py --folds 5 --seed 42 --n_bags 5 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20 --output_name strict_supervised_benchmark
python .\src\03_strict_unsupervised_benchmark.py --folds 5 --seed 42 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20 --output_name strict_unsupervised_benchmark
```

### Core Results

Best supervised model:

- `Blend_CatBoost_KNN_RF`
- Test AUROC: `0.8000`
- Test ACC: `0.6750`
- Test F1: `0.7111`
- Test MCC: `0.3615`

Best single supervised backbone:

- `HybridRandomForest`
- Test AUROC: `0.7900`
- Test ACC: `0.6250`

Best unsupervised models:

- AUROC: `IsolationForestHybridPCA` -> `0.7825 / 0.5500`
- ACC: `Blend_Gaussian_IsoClinical` -> `0.7275 / 0.6250`

## Component 2: `./gaussian_augmentation/`

### Purpose

This is the final reporting package for the Gaussian-augmentation branch. It does not primarily retrain models; instead, it collects, compares, and visualizes the strongest outputs from the main project branches.

### Main Files

- `./gaussian_augmentation/src/01_collect_final_project_results.py`
- `./gaussian_augmentation/selected_results/figures/`
- `./gaussian_augmentation/selected_results/tables/`
- `./gaussian_augmentation/PROJECT_SUMMARY_FINAL.md`

### How To Run

From `./gaussian_augmentation/`:

```powershell
conda activate phm5015_voiced
python .\src\01_collect_final_project_results.py
```

Note:

- this script expects the original source workspaces to be available at relative locations such as `..\final\`, `..\data_balance_final\`, and `..\Wav2vec_o\Wav2vec\`
- if you only need the packaged figures and tables, rerunning is not necessary

### Core Results

Best supervised result across the compared branches:

- `data_balance_final/enhanced -> HybridKNN`
- Test AUROC: `0.8125`
- Test ACC: `0.7000`

Best strict base supervised result:

- `Blend_CatBoost_KNN_RF`
- Test AUROC: `0.8000`
- Test ACC: `0.6750`

Best unsupervised result by AUROC:

- `IsolationForestHybridPCA`
- Test AUROC: `0.7825`
- Test ACC: `0.5500`

Best unsupervised result by ACC:

- `Blend_Gaussian_IsoClinical`
- Test AUROC: `0.7275`
- Test ACC: `0.6250`

## Component 3: `./wav2vec_baseline/`

### Purpose

This workspace contains the early baseline experiments, including handcrafted features, optional wav2vec feature extraction, global supervised baselines, unsupervised baselines, and the strongest validated window-level binary model.

Important:

- despite the folder name, the strongest validated model in this workspace is still the window-level handcrafted binary classifier
- the wav2vec branch is a baseline comparison, not the final selected model

### Main Files

- `./wav2vec_baseline/src/07_build_metadata_strong.py`
- `./wav2vec_baseline/src/08_extract_voiced_features_strong.py`
- `./wav2vec_baseline/src/09_train_supervised_strong.py`
- `./wav2vec_baseline/src/10_train_unsupervised_strong.py`
- `./wav2vec_baseline/src/11_train_windowed_supervised_strong.py`
- `./wav2vec_baseline/results/`
- `./wav2vec_baseline/artifacts/`

### How To Run

From `./wav2vec_baseline/`:

```powershell
conda activate phm5015_voiced
python .\src\07_build_metadata_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0
python .\src\08_extract_voiced_features_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0
python .\src\11_train_windowed_supervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --eval_mode holdout --window_sec 1.0 --hop_sec 0.25 --output_name final_holdout
python .\src\11_train_windowed_supervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --eval_mode cv5 --window_sec 1.0 --hop_sec 0.25 --output_name final_cv5
```

Optional wav2vec feature extraction:

```powershell
python .\src\08_extract_voiced_features_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --skip_handcrafted --use_wav2vec --local_files_only --device cpu
```

### Core Results

Best model in this workspace:

- `src/11_train_windowed_supervised_strong.py`
- 5-fold CV: AUROC `0.6772`, ACC `0.7500`, F1 `0.8395`, MCC `0.2999`
- Holdout: AUROC `0.6833`, ACC `0.7619`, F1 `0.8485`, MCC `0.3443`

Other key baselines:

- Global handcrafted binary holdout: AUROC `0.6528`, ACC `0.5952`
- Global fusion binary CV: AUROC `0.6675`, ACC `0.6202`
- Multiclass global CV: Accuracy `0.4423`, Macro OvR AUC `0.7150`
- Unsupervised reference: IsolationForest AUC approximately `0.5515`

## Component 4: `./disorder_classification/`

### Purpose

This folder contains the final disease-only multiclass classification package for three pathological classes:

- `hyperkinetic`
- `hypokinetic`
- `laryngitis`

It supports supervised multiclass classification, unsupervised clustering, and subgroup analysis.

### Main Files

- `./disorder_classification/voice_disorder_classification.ipynb`
- `./disorder_classification/results/part1_supervised_summary.csv`
- `./disorder_classification/results/part2_clustering_summary.csv`
- `./disorder_classification/results/subgroup_predictions.csv`
- `./disorder_classification/figures/`

### How To Run

Two practical options are supported:

1. Local Jupyter:

```powershell
conda activate phm5015_voiced
jupyter notebook
```

Then open:

```text
./voice_disorder_classification.ipynb
```

2. Google Colab:

- upload `./voice_disorder_classification.ipynb`
- upload the pre-extracted acoustic feature CSV when prompted

Expected outputs:

- `./results/part1_supervised_summary.csv`
- `./results/part2_clustering_summary.csv`
- `./results/subgroup_predictions.csv`
- `./figures/*.png`

### Core Results

Best supervised multiclass model:

- `CatBoost`
- OOF ACC: `0.5298`
- Balanced ACC: `0.5172`
- Macro F1: `0.5176`
- Macro OvR AUC: `0.6970`

Best unsupervised clustering model:

- `GMM (k=3)`
- ARI: `0.0422`
- NMI: `0.0865`
- Purity: `0.5166`
- Aligned ACC: `0.4503`
- Aligned Balanced ACC: `0.4734`

Subgroup highlights:

- age `< 38`: ACC `0.605`, Balanced ACC `0.593`
- high RSI (`>= 13`): ACC `0.567`, Balanced ACC `0.587`
