from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import wfdb
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT_DEFAULT = Path(r"D:\NUS\Second sem\5015\group_report\Project")
DATA_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "voice-icar-federico-ii-database-1.0.0"

MULTICLASS_ORDER = ["healthy", "hyperkinetic", "hypokinetic", "laryngitis"]
CAT_META_COLUMNS = [
    "gender",
    "occupation_status",
    "smoker",
    "alcohol_consumption",
    "eating_habits",
    "carbonated_beverages",
    "tomatoes",
    "coffee",
    "chocolate",
    "soft_cheese",
    "citrus_fruits",
]
NUM_META_COLUMNS = [
    "age",
    "vhi",
    "rsi",
    "cigarettes_per_day",
    "alcohol_glasses_per_day",
    "water_liters_per_day",
    "carbonated_glasses_per_day",
    "coffee_cups_per_day",
    "chocolate_grams_per_day",
    "soft_cheese_grams_per_day",
    "citrus_fruits_per_day",
]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def ensure_project_paths(project_root: str | Path) -> Dict[str, Path]:
    root = Path(project_root)
    processed_dir = root / "data" / "processed"
    outputs_dir = root / "outputs" / "voiced_strong"
    supervised_dir = outputs_dir / "supervised"
    unsupervised_dir = outputs_dir / "unsupervised"
    for directory in [processed_dir, outputs_dir, supervised_dir, unsupervised_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "processed_dir": processed_dir,
        "outputs_dir": outputs_dir,
        "supervised_dir": supervised_dir,
        "unsupervised_dir": unsupervised_dir,
    }


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> int:
    value = 2166136261
    for ch in text.encode("utf-8"):
        value ^= ch
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def normalize_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text in {"", "na", "n/a", "nu", "none", "unknown", "nan"}:
        return None
    return text


def safe_float(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "na", "n/a", "nu", "none", "unknown", "nan"}:
        return np.nan
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9.\-]+", "", text)
    if text in {"", ".", "-", "-.", ".-"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_info_file(info_path: Path) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for raw_line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def map_diagnosis_to_group(diagnosis: object) -> str:
    text = normalize_text(diagnosis) or ""
    if text == "healthy":
        return "healthy"
    if "reflux laryngitis" in text or text == "laryngitis":
        return "laryngitis"
    if "hyperkinetic dysphonia" in text:
        return "hyperkinetic"
    if "hypokinetic dysphonia" in text:
        return "hypokinetic"
    raise ValueError(f"Unmapped diagnosis: {diagnosis!r}")


def load_record_ids(data_dir: Path) -> List[str]:
    records_file = data_dir / "RECORDS"
    if records_file.exists():
        return [line.strip() for line in records_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    return sorted(path.stem for path in data_dir.glob("voice*.hea"))


def build_metadata_frame(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    record_ids = load_record_ids(data_dir)
    rows: List[Dict[str, object]] = []
    for record_id in record_ids:
        info_path = data_dir / f"{record_id}-info.txt"
        fields = parse_info_file(info_path)
        diagnosis_raw = fields.get("Diagnosis", "")
        class_group = map_diagnosis_to_group(diagnosis_raw)
        row = {
            "record_id": record_id,
            "diagnosis_raw": diagnosis_raw,
            "class_group": class_group,
            "label_multiclass": MULTICLASS_ORDER.index(class_group),
            "label_binary_pathology": 0 if class_group == "healthy" else 1,
            "age": safe_float(fields.get("Age")),
            "gender": normalize_text(fields.get("Gender")),
            "occupation_status": normalize_text(fields.get("Occupation status")),
            "vhi": safe_float(fields.get("Voice Handicap Index (VHI) Score", fields.get("Voice Handicap Index (VHI) score"))),
            "rsi": safe_float(fields.get("Reflux Symptom Index (RSI) Score", fields.get("Reflux Symptom Index (RSI) score"))),
            "smoker": normalize_text(fields.get("Smoker")),
            "cigarettes_per_day": safe_float(fields.get("Number of cigarettes smoked per day")),
            "alcohol_consumption": normalize_text(fields.get("Alcohol consumption")),
            "alcohol_glasses_per_day": safe_float(fields.get("Number of glasses containing alcoholic beverage drinked in a day")),
            "water_liters_per_day": safe_float(fields.get("Amount of water's litres drink every day")),
            "eating_habits": normalize_text(fields.get("Eating habits")),
            "carbonated_beverages": normalize_text(fields.get("Carbonated beverages")),
            "carbonated_glasses_per_day": safe_float(fields.get("Amount of glasses drinked in a day")),
            "tomatoes": normalize_text(fields.get("Tomatoes")),
            "coffee": normalize_text(fields.get("Coffee")),
            "coffee_cups_per_day": safe_float(fields.get("Number of cups of coffee drinked in a day")),
            "chocolate": normalize_text(fields.get("Chocolate")),
            "chocolate_grams_per_day": safe_float(fields.get("Gramme of chocolate eaten in  a day")),
            "soft_cheese": normalize_text(fields.get("Soft cheese")),
            "soft_cheese_grams_per_day": safe_float(fields.get("Gramme of soft cheese eaten in a day")),
            "citrus_fruits": normalize_text(fields.get("Citrus fruits")),
            "citrus_fruits_per_day": safe_float(fields.get("Number of citrus fruits eaten in a day")),
            "info_path": str(info_path.resolve()),
            "wfdb_record_path": str((data_dir / record_id).resolve()),
            "txt_signal_path": str((data_dir / f"{record_id}.txt").resolve()),
        }
        rows.append(row)

    metadata = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    return metadata


def read_signal(record_base_path: str | Path) -> Tuple[np.ndarray, int]:
    record = wfdb.rdrecord(str(record_base_path))
    signal = record.p_signal
    waveform = signal[:, 0] if signal.ndim == 2 else signal
    return np.asarray(waveform, dtype=np.float32), int(record.fs)


def normalize_waveform(y: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    peak = np.max(np.abs(y)) + 1e-8
    return y / peak


def resample_waveform(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    tensor = torch.tensor(np.asarray(y, dtype=np.float32))
    resampled = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=target_sr)
    return resampled.numpy().astype(np.float32)


def safe_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    for kwargs in [
        {"fmin": 50.0, "n_bands": 4},
        {"fmin": 50.0, "n_bands": 3},
        {"fmin": 40.0, "n_bands": 3},
        {"fmin": 30.0, "n_bands": 2},
    ]:
        try:
            return librosa.feature.spectral_contrast(y=y, sr=sr, **kwargs)
        except Exception:
            continue
    return np.full((1, max(1, librosa.feature.rms(y=y).shape[1])), np.nan, dtype=float)


def estimate_periods_from_f0(f0: np.ndarray, sr: int) -> np.ndarray:
    valid = np.asarray(f0, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) < 3:
        return np.array([], dtype=float)
    return sr / np.clip(valid, 1e-6, None)


def jitter_features_from_f0(f0: np.ndarray, sr: int) -> Dict[str, float]:
    periods = estimate_periods_from_f0(f0, sr)
    if len(periods) < 3:
        return {"jitter_local": np.nan, "jitter_rap": np.nan, "jitter_ppq5": np.nan}
    diff1 = np.abs(np.diff(periods))
    local = np.mean(diff1) / (np.mean(periods) + 1e-8)
    rap = []
    for idx in range(1, len(periods) - 1):
        rap.append(abs(periods[idx] - np.mean(periods[idx - 1 : idx + 2])))
    ppq5 = []
    for idx in range(2, len(periods) - 2):
        ppq5.append(abs(periods[idx] - np.mean(periods[idx - 2 : idx + 3])))
    return {
        "jitter_local": float(local),
        "jitter_rap": float(np.mean(rap) / (np.mean(periods) + 1e-8)) if rap else np.nan,
        "jitter_ppq5": float(np.mean(ppq5) / (np.mean(periods) + 1e-8)) if ppq5 else np.nan,
    }


def shimmer_features(y: np.ndarray, sr: int, f0: np.ndarray) -> Dict[str, float]:
    periods = estimate_periods_from_f0(f0, sr)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    if len(periods) < 4 or len(rms) < 6:
        return {"shimmer_local": np.nan, "shimmer_apq3": np.nan, "shimmer_apq5": np.nan}
    amp = np.asarray(rms, dtype=float)
    diff1 = np.abs(np.diff(amp))
    local = np.mean(diff1) / (np.mean(amp) + 1e-8)
    apq3 = []
    for idx in range(1, len(amp) - 1):
        apq3.append(abs(amp[idx] - np.mean(amp[idx - 1 : idx + 2])))
    apq5 = []
    for idx in range(2, len(amp) - 2):
        apq5.append(abs(amp[idx] - np.mean(amp[idx - 2 : idx + 3])))
    return {
        "shimmer_local": float(local),
        "shimmer_apq3": float(np.mean(apq3) / (np.mean(amp) + 1e-8)) if apq3 else np.nan,
        "shimmer_apq5": float(np.mean(apq5) / (np.mean(amp) + 1e-8)) if apq5 else np.nan,
    }


def hnr_proxy(y: np.ndarray, sr: int) -> float:
    try:
        ac = librosa.autocorrelate(y, max_size=min(len(y), int(sr / 50)))
        if len(ac) < 3:
            return np.nan
        ac0 = ac[0] + 1e-8
        peak = np.max(ac[1:]) if len(ac) > 1 else 0.0
        ratio = peak / max(ac0 - peak, 1e-8)
        return float(10 * np.log10(max(ratio, 1e-8)))
    except Exception:
        return np.nan


def cpp_proxy(y: np.ndarray, sr: int) -> Dict[str, float]:
    try:
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=128)) + 1e-8
        log_spec = np.log(stft)
        cepstrum = np.abs(np.fft.rfft(log_spec, axis=0))
        quefrency = np.arange(cepstrum.shape[0]) / sr
        mask = (quefrency >= 1 / 500) & (quefrency <= 1 / 50)
        if not np.any(mask):
            return {"cpp_proxy_mean": np.nan, "cpp_proxy_std": np.nan, "cpps_proxy": np.nan}
        region = cepstrum[mask, :]
        peak = region.max(axis=0)
        baseline = np.median(region, axis=0)
        cpp = peak - baseline
        smooth = np.convolve(cpp, np.ones(5) / 5, mode="same") if len(cpp) >= 5 else cpp
        return {
            "cpp_proxy_mean": float(np.mean(cpp)),
            "cpp_proxy_std": float(np.std(cpp)),
            "cpps_proxy": float(np.mean(smooth)),
        }
    except Exception:
        return {"cpp_proxy_mean": np.nan, "cpp_proxy_std": np.nan, "cpps_proxy": np.nan}


def spectral_slope_proxy(y: np.ndarray, sr: int) -> float:
    try:
        spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-8
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        log_freqs = np.log(freqs[1:] + 1e-8)
        slopes = []
        for frame in spec[1:].T:
            coeff = np.polyfit(log_freqs, np.log(frame), deg=1)
            slopes.append(coeff[0])
        return float(np.mean(slopes))
    except Exception:
        return np.nan


def extract_handcrafted_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    y = normalize_waveform(y)
    features: Dict[str, float] = {
        "duration_sec": len(y) / sr,
        "signal_mean": float(np.mean(y)),
        "signal_std": float(np.std(y)),
        "signal_abs_mean": float(np.mean(np.abs(y))),
        "signal_energy": float(np.mean(y**2)),
    }

    feature_blocks = {
        "rms": librosa.feature.rms(y=y)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(y=y)[0],
    }
    contrast = safe_spectral_contrast(y=y, sr=sr)
    feature_blocks["contrast_mean_over_bands"] = np.nanmean(contrast, axis=0)
    for name, arr in feature_blocks.items():
        features[f"{name}_mean"] = float(np.nanmean(arr))
        features[f"{name}_std"] = float(np.nanstd(arr))

    harmonic, percussive = librosa.effects.hpss(y)
    features["harmonic_energy"] = float(np.mean(harmonic**2))
    features["percussive_energy"] = float(np.mean(percussive**2))
    features["harmonic_percussive_ratio"] = float(features["harmonic_energy"] / (features["percussive_energy"] + 1e-8))
    features["spectral_slope_proxy"] = spectral_slope_proxy(y, sr)

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        voiced_mask = ~np.isnan(f0)
        features["voiced_fraction"] = float(np.mean(voiced_mask))
        features["f0_mean"] = float(np.nanmean(f0))
        features["f0_std"] = float(np.nanstd(f0))
        features["f0_min"] = float(np.nanmin(f0))
        features["f0_max"] = float(np.nanmax(f0))
        features["pitch_range_ratio"] = float((np.nanmax(f0) - np.nanmin(f0)) / (np.nanmean(f0) + 1e-8))
    except Exception:
        f0 = np.array([np.nan])
        features["voiced_fraction"] = np.nan
        features["f0_mean"] = np.nan
        features["f0_std"] = np.nan
        features["f0_min"] = np.nan
        features["f0_max"] = np.nan
        features["pitch_range_ratio"] = np.nan

    features.update(jitter_features_from_f0(f0, sr))
    features.update(shimmer_features(y, sr, f0))
    features["hnr_proxy"] = hnr_proxy(y, sr)
    features.update(cpp_proxy(y, sr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    for prefix, block in [("mfcc", mfcc), ("mfcc_delta", mfcc_d1), ("mfcc_delta2", mfcc_d2)]:
        for idx in range(block.shape[0]):
            features[f"{prefix}_{idx + 1}_mean"] = float(np.mean(block[idx]))
            features[f"{prefix}_{idx + 1}_std"] = float(np.std(block[idx]))

    mel20 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel20_db = librosa.power_to_db(mel20 + 1e-10)
    for idx in range(mel20_db.shape[0]):
        features[f"logmel20_{idx + 1}_mean"] = float(np.mean(mel20_db[idx]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for idx in range(chroma.shape[0]):
        features[f"chroma_{idx + 1}_mean"] = float(np.mean(chroma[idx]))
        features[f"chroma_{idx + 1}_std"] = float(np.std(chroma[idx]))

    mel64 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel64_db = librosa.power_to_db(mel64 + 1e-10)
    features["logmel64_mean"] = float(np.mean(mel64_db))
    features["logmel64_std"] = float(np.std(mel64_db))
    features["logmel64_min"] = float(np.min(mel64_db))
    features["logmel64_max"] = float(np.max(mel64_db))
    return features


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        auc = roc_auc_score(y_true_arr, y_score_arr)
    except Exception:
        auc = np.nan
    return {
        "auc": auc,
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    try:
        macro_auc = roc_auc_score(y_true_arr, prob, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = np.nan
    return {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
        "macro_ovr_auc": macro_auc,
    }


def tune_binary_threshold(y_true: Sequence[int], y_score: Sequence[float], metric: str = "mcc") -> Tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (y_score_arr >= threshold).astype(int)
        if metric == "f1":
            value = f1_score(y_true_arr, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true_arr, y_pred)
        else:
            value = matthews_corrcoef(y_true_arr, y_pred)
        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_value


def cluster_purity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(y_pred, dtype=int)
    total = len(true_arr)
    correct = 0
    for cluster in np.unique(pred_arr):
        mask = pred_arr == cluster
        labels, counts = np.unique(true_arr[mask], return_counts=True)
        correct += int(counts.max())
    return float(correct / total) if total else np.nan


def clustering_metrics(X: np.ndarray, y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    metrics = {
        "ari": adjusted_rand_score(y_true, y_pred),
        "ami": adjusted_mutual_info_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
        "purity": cluster_purity(y_true, y_pred),
    }
    unique_clusters = np.unique(y_pred)
    if len(unique_clusters) > 1 and len(unique_clusters) < len(X):
        metrics["silhouette"] = silhouette_score(X, y_pred)
        metrics["davies_bouldin"] = davies_bouldin_score(X, y_pred)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, y_pred)
    else:
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan
    return metrics


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


def json_dump(data: Dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
