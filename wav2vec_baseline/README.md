# Wav2Vec Baseline Final README

## Overview
This folder is the baseline VOICED workspace used for the early supervised and unsupervised experiments. It includes handcrafted feature extraction, optional wav2vec2 feature extraction, global supervised baselines, unsupervised baselines, and the strongest validated window-level binary model.

Important note: despite the folder name, the strongest validated result in this workspace is still the window-level handcrafted binary model. The optional wav2vec branch is included as a feature-extraction baseline, not as the final best model.

## Recommended Anaconda Environment
Minimum Python packages required here:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `joblib`
- `librosa`
- `wfdb`
- `torch`
- `torchaudio`
- `transformers`
- `tqdm`

## Folder Layout
- `src/`
  - `07_build_metadata_strong.py`: metadata construction
  - `08_extract_voiced_features_strong.py`: handcrafted features and optional wav2vec features
  - `09_train_supervised_strong.py`: global supervised baselines
  - `10_train_unsupervised_strong.py`: unsupervised reference experiments
  - `11_train_windowed_supervised_strong.py`: strongest validated binary model in this workspace
  - `common_voiced_strong.py`: shared utilities and project path helpers
- `artifacts/`: intermediate files already generated in this delivery copy
- `results/`
  - `supervised/`
  - `unsupervised/`
- `RUN_FLOW.md`: recommended execution order
- `RESULTS_SUMMARY.md`: validated metric summary
- `run_all.ps1`: convenience runner

## Important Path Requirement
This workspace expects the raw VOICED dataset at a relative location such as:

```text
..\voice-icar-federico-ii-database-1.0.0\
```

If your dataset is stored somewhere else, pass a relative `--data_dir` and, when needed, a relative `--project_root`.

## How To Run
Recommended execution order:

1. Build metadata
```powershell
python .\src\07_build_metadata_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0
```

2. Extract handcrafted features
```powershell
python .\src\08_extract_voiced_features_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0
```

3. Optional: extract local wav2vec2-large features
```powershell
python .\src\08_extract_voiced_features_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --skip_handcrafted --use_wav2vec --local_files_only --device cpu
```

4. Run the strongest binary model
```powershell
python .\src\11_train_windowed_supervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --eval_mode holdout --window_sec 1.0 --hop_sec 0.25 --output_name final_holdout
python .\src\11_train_windowed_supervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --eval_mode cv5 --window_sec 1.0 --hop_sec 0.25 --output_name final_cv5
```

5. Optional: run the global multiclass and unsupervised references
```powershell
python .\src\09_train_supervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --task multiclass --eval_mode cv5 --output_name multiclass_cv5
python .\src\10_train_unsupervised_strong.py --project_root . --data_dir ..\voice-icar-federico-ii-database-1.0.0 --feature_set tabular --output_name final_unsupervised
```

## Core Results
Best supervised model in this workspace:
- `src/11_train_windowed_supervised_strong.py`
- 5-fold CV: AUROC `0.6772`, ACC `0.7500`, F1 `0.8395`, MCC `0.2999`
- Holdout: AUROC `0.6833`, ACC `0.7619`, F1 `0.8485`, MCC `0.3443`

Other validated baselines:
- Global handcrafted binary holdout: AUROC `0.6528`, ACC `0.5952`
- Global fusion binary CV: AUROC `0.6675`, ACC `0.6202`
- Global multiclass CV: Accuracy `0.4423`, Macro OVR AUC `0.7150`
- Unsupervised reference: IsolationForest AUC approximately `0.5515`

## Most Important Files To Read
- `README.md`
- `RESULTS_SUMMARY.md`
- `RUN_FLOW.md`
- `results\supervised\`
- `results\unsupervised\`
- `src\11_train_windowed_supervised_strong.py`
