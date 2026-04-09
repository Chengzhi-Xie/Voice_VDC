from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

from common_voiced_strong import (
    binary_metrics,
    build_metadata_frame,
    build_parser,
    ensure_project_paths,
    json_dump,
    normalize_waveform,
    read_signal,
    resample_waveform,
    set_seed,
    tune_binary_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = build_parser("Train a stronger window-level supervised binary VOICED model.")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--eval_mode", type=str, default="holdout", choices=["holdout", "cv5"])
    parser.add_argument("--window_sec", type=float, default=1.5)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--output_name", type=str, default="windowed_binary")
    return parser.parse_args()


def resolve_metadata(project_root: Path, provided: str, data_dir: str) -> pd.DataFrame:
    if provided:
        return pd.read_csv(provided)
    default_csv = project_root / "data" / "processed" / "voiced_metadata_strong.csv"
    if default_csv.exists():
        return pd.read_csv(default_csv)
    return build_metadata_frame(data_dir)


def segment_waveform(y: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> List[np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if win <= 0 or len(y) <= win:
        return [y]
    starts = list(range(0, max(1, len(y) - win + 1), max(1, hop)))
    if starts[-1] + win < len(y):
        starts.append(len(y) - win)
    views = [y[start : start + win] for start in starts]
    return views


def extract_window_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    y = normalize_waveform(y)
    feat: Dict[str, float] = {}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=64)
    mel_db = librosa.power_to_db(mel + 1e-10)
    for idx in range(mel_db.shape[0]):
        feat[f"mel_{idx + 1}_mean"] = float(np.mean(mel_db[idx]))
        feat[f"mel_{idx + 1}_std"] = float(np.std(mel_db[idx]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    for idx in range(mfcc.shape[0]):
        feat[f"mfcc_{idx + 1}_mean"] = float(np.mean(mfcc[idx]))
        feat[f"mfcc_{idx + 1}_std"] = float(np.std(mfcc[idx]))
        feat[f"mfcc_delta_{idx + 1}_mean"] = float(np.mean(delta[idx]))
        feat[f"mfcc_delta_{idx + 1}_std"] = float(np.std(delta[idx]))

    feat["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)))
    feat["rms_std"] = float(np.std(librosa.feature.rms(y=y)))
    feat["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feat["zcr_std"] = float(np.std(librosa.feature.zero_crossing_rate(y)))
    feat["centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feat["bandwidth_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feat["rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feat["flatness_mean"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        voiced = ~np.isnan(f0)
        feat["f0_mean"] = float(np.nanmean(f0))
        feat["f0_std"] = float(np.nanstd(f0))
        feat["voiced_fraction"] = float(np.mean(voiced))
    except Exception:
        feat["f0_mean"] = np.nan
        feat["f0_std"] = np.nan
        feat["voiced_fraction"] = np.nan
    return feat


def build_window_table(
    metadata: pd.DataFrame,
    record_ids: np.ndarray,
    target_sr: int,
    window_sec: float,
    hop_sec: float,
) -> pd.DataFrame:
    meta_subset = metadata.set_index("record_id")
    rows: List[Dict[str, float]] = []
    for record_id in record_ids:
        row = meta_subset.loc[record_id]
        y, sr = read_signal(row["wfdb_record_path"])
        y = resample_waveform(y, sr, target_sr)
        windows = segment_waveform(y, target_sr, window_sec, hop_sec)
        for window_idx, window in enumerate(windows):
            feat = extract_window_features(window, target_sr)
            feat["record_id"] = record_id
            feat["window_index"] = window_idx
            feat["label"] = int(row["label_binary_pathology"])
            feat["age"] = row["age"]
            feat["vhi"] = row["vhi"]
            feat["rsi"] = row["rsi"]
            feat["gender_male"] = 1.0 if str(row["gender"]).lower() == "m" else 0.0
            rows.append(feat)
    return pd.DataFrame(rows)


def train_window_models(X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, object]:
    models = {
        "xgb": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
        ),
        "catboost": CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_state=42,
            verbose=False,
        ),
    }
    trained: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def aggregate_record_probs(frame: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    tmp = frame[["record_id", "label"]].copy()
    tmp["prob"] = prob
    grouped = tmp.groupby("record_id", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))
    return grouped


def pick_best_model(
    val_frame: pd.DataFrame,
    y_val: np.ndarray,
    models: Dict[str, object],
) -> Tuple[str, float]:
    best_name = ""
    best_score = -np.inf
    for name, model in models.items():
        prob = model.predict_proba(val_frame)[:, 1]
        score = float(roc_auc_safe(y_val, prob))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def split_holdout(y: np.ndarray, test_size: float, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y))
    y_train_val = y[train_val_idx]
    inner_val_size = val_size / max(1e-8, (1.0 - test_size))
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_size, random_state=seed + 1)
    train_rel, val_rel = next(sss_val.split(np.zeros(len(train_val_idx)), y_train_val))
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return train_idx, val_idx, test_idx


def evaluate_split(
    metadata: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    target_sr: int,
    window_sec: float,
    hop_sec: float,
) -> Dict[str, object]:
    train_table = build_window_table(metadata, train_ids, target_sr, window_sec, hop_sec)
    val_table = build_window_table(metadata, val_ids, target_sr, window_sec, hop_sec)
    test_table = build_window_table(metadata, test_ids, target_sr, window_sec, hop_sec)

    feature_cols = [c for c in train_table.columns if c not in {"record_id", "window_index", "label"}]
    X_train = train_table[feature_cols]
    y_train = train_table["label"].to_numpy(dtype=int)
    X_val = val_table[feature_cols]
    y_val = val_table["label"].to_numpy(dtype=int)
    X_test = test_table[feature_cols]

    models = train_window_models(X_train, y_train)
    best_name, _ = pick_best_model(X_val, y_val, models)
    best_model = models[best_name]

    val_record = aggregate_record_probs(val_table, best_model.predict_proba(X_val)[:, 1])
    threshold, _ = tune_binary_threshold(val_record["label"].to_numpy(dtype=int), val_record["prob"].to_numpy(dtype=float), metric="mcc")

    test_record = aggregate_record_probs(test_table, best_model.predict_proba(X_test)[:, 1])
    y_true = test_record["label"].to_numpy(dtype=int)
    y_score = test_record["prob"].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    metrics = binary_metrics(y_true, y_pred, y_score)
    return {
        "model": best_name,
        "threshold": threshold,
        "metrics": metrics,
        "test_predictions": test_record,
    }


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    paths = ensure_project_paths(args.project_root)
    metadata = resolve_metadata(paths["root"], args.metadata_csv, args.data_dir).sort_values("record_id").reset_index(drop=True)
    y = metadata["label_binary_pathology"].to_numpy(dtype=int)
    record_ids = metadata["record_id"].to_numpy()

    if args.eval_mode == "holdout":
        train_idx, val_idx, test_idx = split_holdout(y, args.test_size, args.val_size, args.seed)
        result = evaluate_split(
            metadata,
            record_ids[train_idx],
            record_ids[val_idx],
            record_ids[test_idx],
            args.target_sr,
            args.window_sec,
            args.hop_sec,
        )
        summary = {
            "eval_mode": "holdout",
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "model": result["model"],
            "threshold": result["threshold"],
            "metrics": result["metrics"],
        }
        print("Windowed holdout metrics:")
        print(pd.Series(result["metrics"]).to_string())
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_rows: List[Dict[str, float]] = []
        oof = []
        for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(record_ids, y), start=1):
            inner_y = y[train_val_idx]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold_id)
            train_rel, val_rel = next(sss.split(np.zeros(len(train_val_idx)), inner_y))
            train_idx = train_val_idx[train_rel]
            val_idx = train_val_idx[val_rel]
            result = evaluate_split(
                metadata,
                record_ids[train_idx],
                record_ids[val_idx],
                record_ids[test_idx],
                args.target_sr,
                args.window_sec,
                args.hop_sec,
            )
            row = {"fold": fold_id, "model": result["model"], **result["metrics"]}
            fold_rows.append(row)
            test_pred: pd.DataFrame = result["test_predictions"]  # type: ignore[assignment]
            test_pred = test_pred.assign(fold=fold_id)
            oof.append(test_pred)

        oof_df = pd.concat(oof, ignore_index=True)
        aggregate = binary_metrics(
            oof_df["label"].to_numpy(dtype=int),
            (oof_df["prob"].to_numpy(dtype=float) >= 0.5).astype(int),
            oof_df["prob"].to_numpy(dtype=float),
        )
        summary = {
            "eval_mode": "cv5",
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "fold_metrics": fold_rows,
            "aggregate_metrics": aggregate,
        }
        print("Windowed CV metrics:")
        print(pd.Series(aggregate).to_string())

    out_dir = paths["supervised_dir"] / f"windowed_binary_{args.eval_mode}_{args.output_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dump(summary, out_dir / "metrics.json")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(parse_args())
