# Gaussian Augmentation Final README

## Overview
This folder is the final report package for the Gaussian-augmentation track. It does not retrain models by itself; instead, it collects and compares the most important outputs from three validated source lines:
- `final`
- `data_balance_final`
- `Wav2vec_o/Wav2vec`

The goal of this folder is to provide one compact delivery package for reporting, with curated tables, copied key artifacts, and unified comparison figures.

## Recommended Anaconda Environment
Minimum Python packages required here:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

## Folder Layout
- `src/`
  - `01_collect_final_project_results.py`: copies selected source artifacts and generates summary figures
- `selected_results/`
  - `figures/`: final presentation-ready comparison plots
  - `tables/`: compact summary tables
  - `final/`: selected artifacts copied from the nested-CV baseline
  - `data_balance_final/`: selected artifacts copied from the strict Gaussian-balanced workspaces
  - `wav2vec/`: selected artifacts copied from the frozen wav2vec baseline
  - `project_summary_snapshot.json`: short machine-readable summary of the main metrics
  - `copied_artifact_manifest.csv`: traceability log of copied files
- `PROJECT_SUMMARY_FINAL.md`: final narrative summary
- `README.md`: this consolidated entry point

## Important Path Requirement
If you want to rerun the packaging script instead of only reading the packaged results, restore the original source workspaces at relative locations such as:
- `..\final\`
- `..\data_balance_final\`
- `..\Wav2vec_o\Wav2vec\`

The packaging script reads those source folders and regenerates the curated comparison package in `.\selected_results\`.

## How To Run
Run the packaging script from this folder after the source workspaces already exist:

```powershell
python .\src\01_collect_final_project_results.py
```

This will:
- copy the selected source artifacts
- build unified AUROC/ACC comparison figures
- generate a project snapshot JSON
- refresh the curated summary package under `selected_results/`

## Core Results
Best supervised result across the compared sources:
- `data_balance_final/enhanced -> HybridKNN`
- Test AUROC: `0.8125`
- Test ACC: `0.7000`

Best base strict supervised result:
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

Wav2vec baseline takeaway:
- The frozen wav2vec + classical ML line remained a comparison baseline rather than the final selected route.

## Most Important Files To Read
- `README.md`
- `PROJECT_SUMMARY_FINAL.md`
- `selected_results\figures\overall_supervised_selected_models.png`
- `selected_results\figures\overall_unsupervised_selected_models.png`
- `selected_results\tables\overall_supervised_selected_models.csv`
- `selected_results\tables\overall_unsupervised_selected_models.csv`
- `selected_results\project_summary_snapshot.json`
