# Binary Classification Final README

## Overview
This folder is the final strict binary-classification package built on top of the Gaussian-balanced VOICED setting. It keeps the fixed test split, uses fold-wise Gaussian augmentation and subgroup fitting, and provides both supervised and unsupervised binary baselines under a leakage-aware evaluation protocol.

## Recommended Anaconda Environment
Minimum Python packages required here:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `joblib`
- `catboost`

## Folder Layout
- `src/`
  - `01_audit_and_build_final_dataset.py`: builds the strict balanced dataset and writes the leakage audit
  - `02_strict_supervised_benchmark.py`: strict supervised benchmark
  - `03_strict_unsupervised_benchmark.py`: strict unsupervised benchmark
  - `strict_balance_utils.py`: Gaussian augmentation, subgroup fitting, and feature-view definitions
- `artifacts/`
  - strict train/test tables and leakage-audit JSON files
- `results/`
  - `strict_supervised_benchmark/`
  - `strict_unsupervised_benchmark/`
  - `key_results_summary.csv`
- `LEAKAGE_AUDIT.md`: audit notes
- `RESULTS_SUMMARY.md`: detailed metrics summary
- `RUN_FLOW.md`: step-by-step run order
- `run_all.ps1`: convenience wrapper

## Important Path Requirement
This folder is a delivery copy of the original `data_balance_final` workspace. The code expects shared source artifacts at the relative parent location `..\artifacts\`. If you want to rerun the pipeline from this delivery copy, you should either:
- restore the shared artifacts under `..\artifacts\`, or
- update `BASE_ARTIFACTS_DIR` in `.\src\strict_balance_utils.py` to the correct relative source location.

## How To Run
Recommended sequence:

1. Build the strict final dataset and leakage audit
```powershell
python .\src\01_audit_and_build_final_dataset.py --seed 42 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20
```

2. Run the supervised benchmark
```powershell
python .\src\02_strict_supervised_benchmark.py --folds 5 --seed 42 --n_bags 5 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20 --output_name strict_supervised_benchmark
```

3. Run the unsupervised benchmark
```powershell
python .\src\03_strict_unsupervised_benchmark.py --folds 5 --seed 42 --acoustic_noise_scale 0.30 --clinical_noise_scale 0.20 --output_name strict_unsupervised_benchmark
```

## Core Results
Supervised:
- Best overall model: `Blend_CatBoost_KNN_RF`
- Test AUROC: `0.8000`
- Test ACC: `0.6750`
- Test F1: `0.7111`
- Test MCC: `0.3615`

Best single supervised backbone:
- `HybridRandomForest`
- Test AUROC: `0.7900`
- Test ACC: `0.6250`

Unsupervised:
- Best AUROC: `IsolationForestHybridPCA` -> `0.7825 / 0.5500`
- Best ACC: `Blend_Gaussian_IsoClinical` -> `0.7275 / 0.6250`

## Most Important Files To Read
- `README.md`
- `LEAKAGE_AUDIT.md`
- `RESULTS_SUMMARY.md`
- `results\strict_supervised_benchmark\strict_supervised_model_summary.csv`
- `results\strict_unsupervised_benchmark\strict_unsupervised_model_summary.csv`
- `results\key_results_summary.csv`
