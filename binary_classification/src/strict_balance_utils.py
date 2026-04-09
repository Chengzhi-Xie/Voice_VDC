from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PARENT_BALANCE_DIR = PACKAGE_ROOT.parent
PROJECT_ROOT = PARENT_BALANCE_DIR.parent
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
RESULTS_DIR = PACKAGE_ROOT / "results"
BASE_ARTIFACTS_DIR = PARENT_BALANCE_DIR / "artifacts"

CORE_COLS = ["age", "vhi", "rsi", "gender_male"]
META_OTHER_COLS = [
    "cigarettes_per_day",
    "alcohol_glasses_per_day",
    "water_liters_per_day",
    "carbonated_glasses_per_day",
    "coffee_cups_per_day",
    "chocolate_grams_per_day",
    "soft_cheese_grams_per_day",
    "citrus_fruits_per_day",
    "duration_sec",
]
ENGINEERED_BASE_COLS = [
    "symptom_sum",
    "symptom_diff",
    "symptom_ratio",
    "symptom_product",
    "age_x_vhi",
    "age_x_rsi",
    "age_x_symptom",
    "male_x_vhi",
    "male_x_rsi",
    "male_x_symptom",
    "age_sq",
    "vhi_sq",
    "rsi_sq",
    "vhi_rsi_gap_abs",
    "hydration_gap",
    "irritant_load",
    "diet_reflux_load",
    "smoke_alcohol_interaction",
    "age_lt35",
    "age_35_50",
    "age_ge50",
    "vhi_ge20",
    "rsi_ge13",
    "clinical_subgroup",
]
EXCLUDED_COLS = {"record_id", "class_group", "label_binary_pathology", "source"}


def ensure_dirs() -> Dict[str, Path]:
    for directory in [ARTIFACTS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "package_root": PACKAGE_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "results_dir": RESULTS_DIR,
        "base_artifacts_dir": BASE_ARTIFACTS_DIR,
    }


def load_base_real_splits() -> Dict[str, pd.DataFrame]:
    return {
        "healthy_train_real": pd.read_csv(BASE_ARTIFACTS_DIR / "healthy_train_real_37.csv"),
        "disease_train_real": pd.read_csv(BASE_ARTIFACTS_DIR / "disease_train_real_131.csv"),
        "fixed_test": pd.read_csv(BASE_ARTIFACTS_DIR / "balanced_test.csv"),
    }


def split_raw_feature_groups(frame: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    feature_cols = [column for column in frame.columns if column not in EXCLUDED_COLS]
    core_cols = [column for column in CORE_COLS if column in feature_cols]
    meta_other_cols = [column for column in META_OTHER_COLS if column in feature_cols]
    acoustic_cols = [column for column in feature_cols if column not in set(core_cols + meta_other_cols)]
    return core_cols, meta_other_cols, acoustic_cols


def fit_subgroup_models(
    healthy_train_real: pd.DataFrame,
    disease_train_real: pd.DataFrame,
    core_cols: List[str],
    seed: int,
    n_healthy_subgroups: int = 3,
    n_clinical_subgroups: int = 4,
) -> Dict[str, object]:
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    healthy_core = healthy_train_real[core_cols].copy()
    all_real = pd.concat([healthy_train_real, disease_train_real], ignore_index=True)

    healthy_prep = prep.fit(healthy_core)
    healthy_X = healthy_prep.transform(healthy_core)
    n_h = max(1, min(n_healthy_subgroups, len(healthy_train_real)))
    healthy_kmeans = KMeans(n_clusters=n_h, random_state=seed, n_init=30)
    healthy_cluster = healthy_kmeans.fit_predict(healthy_X)

    clinical_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    clinical_X = clinical_prep.fit_transform(all_real[core_cols])
    n_c = max(2, min(n_clinical_subgroups, len(all_real)))
    clinical_kmeans = KMeans(n_clusters=n_c, random_state=seed, n_init=30)
    clinical_kmeans.fit(clinical_X)

    return {
        "healthy_prep": healthy_prep,
        "healthy_kmeans": healthy_kmeans,
        "healthy_cluster": healthy_cluster,
        "clinical_prep": clinical_prep,
        "clinical_kmeans": clinical_kmeans,
    }


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    scale: float,
    lower: float,
    upper: float,
) -> float:
    scale = max(float(scale), 1e-6)
    for _ in range(128):
        value = float(rng.normal(mean, scale))
        if lower <= value <= upper:
            return value
    return float(np.clip(mean, lower, upper))


def generate_subgroup_gaussian_healthy(
    healthy_train_real: pd.DataFrame,
    disease_train_real: pd.DataFrame,
    seed: int = 42,
    n_synthetic_healthy: int | None = None,
    acoustic_noise_scale: float = 0.30,
    clinical_noise_scale: float = 0.20,
    anchor_mix: float = 0.70,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    core_cols, meta_other_cols, acoustic_cols = split_raw_feature_groups(healthy_train_real)
    models = fit_subgroup_models(healthy_train_real, disease_train_real, core_cols, seed=seed)

    healthy_with_group = healthy_train_real.copy()
    healthy_with_group["healthy_subgroup"] = models["healthy_cluster"]
    subgroup_share = healthy_with_group["healthy_subgroup"].value_counts(normalize=True).sort_index()

    if n_synthetic_healthy is None:
        n_synthetic_healthy = max(0, len(disease_train_real) - len(healthy_train_real))

    all_feature_cols = core_cols + meta_other_cols + acoustic_cols
    real_numeric = healthy_train_real[all_feature_cols].copy()
    real_numeric = real_numeric.fillna(real_numeric.median(numeric_only=True))
    global_mean = real_numeric.mean(axis=0)
    global_std = real_numeric.std(axis=0, ddof=0).replace(0.0, 1e-6)
    global_lower = global_mean - 6.0 * global_std
    global_upper = global_mean + 6.0 * global_std

    original_keys = {tuple(np.round(row, 8)) for row in real_numeric.to_numpy(dtype=float)}
    used = set()
    synth_rows: List[Dict[str, float]] = []

    while len(synth_rows) < n_synthetic_healthy:
        subgroup = int(rng.choice(subgroup_share.index.to_numpy(), p=subgroup_share.to_numpy()))
        subgroup_df = healthy_with_group.loc[healthy_with_group["healthy_subgroup"] == subgroup].reset_index(drop=True)
        anchor = subgroup_df.iloc[int(rng.integers(0, len(subgroup_df)))]

        subgroup_stats = subgroup_df[all_feature_cols].copy().fillna(real_numeric.median(numeric_only=True))
        subgroup_mean = subgroup_stats.mean(axis=0)
        subgroup_std = subgroup_stats.std(axis=0, ddof=0).replace(0.0, np.nan)
        subgroup_min = subgroup_stats.min(axis=0)
        subgroup_max = subgroup_stats.max(axis=0)

        row: Dict[str, float] = {}
        if "gender_male" in core_cols:
            row["gender_male"] = float(anchor["gender_male"])

        for col in [c for c in ["age", "vhi", "rsi"] if c in core_cols]:
            mean = anchor_mix * float(anchor[col]) + (1.0 - anchor_mix) * float(subgroup_mean[col])
            scale = float(np.nan_to_num(subgroup_std[col], nan=global_std[col]) * clinical_noise_scale + global_std[col] * 0.05)
            lower = float(max(subgroup_min[col] - 0.15 * global_std[col], global_lower[col]))
            upper = float(min(subgroup_max[col] + 0.15 * global_std[col], global_upper[col]))
            row[col] = _sample_truncated_normal(rng, mean, scale, lower, upper)

        for col in meta_other_cols:
            if col == "duration_sec":
                mean = anchor_mix * float(anchor[col]) + (1.0 - anchor_mix) * float(subgroup_mean[col])
                scale = float(np.nan_to_num(subgroup_std[col], nan=global_std[col]) * 0.15 + global_std[col] * 0.03)
                lower = float(max(subgroup_min[col] - 0.1 * global_std[col], global_lower[col]))
                upper = float(min(subgroup_max[col] + 0.1 * global_std[col], global_upper[col]))
                row[col] = _sample_truncated_normal(rng, mean, scale, lower, upper)
                continue
            choices = subgroup_df[col].dropna().to_numpy(dtype=float)
            if len(choices) == 0:
                choices = healthy_train_real[col].dropna().to_numpy(dtype=float)
            sampled = float(rng.choice(choices)) if len(choices) else float(global_mean[col])
            row[col] = max(0.0, sampled)

        for col in acoustic_cols:
            anchor_val = float(anchor[col])
            local_std = float(np.nan_to_num(subgroup_std[col], nan=global_std[col]))
            scale = max(local_std * acoustic_noise_scale, float(global_std[col]) * 0.04, 1e-6)
            lower = float(max(subgroup_min[col] - 0.25 * global_std[col], global_lower[col]))
            upper = float(min(subgroup_max[col] + 0.25 * global_std[col], global_upper[col]))
            row[col] = _sample_truncated_normal(rng, anchor_val, scale, lower, upper)

        key = tuple(np.round([row[col] for col in all_feature_cols], 8))
        if key in used or key in original_keys:
            continue
        used.add(key)
        row["record_id"] = f"strict_synth_healthy_{len(synth_rows):03d}"
        row["class_group"] = "healthy"
        row["label_binary_pathology"] = 0
        row["source"] = "synthetic_healthy_subgroup_strict"
        synth_rows.append(row)

    synthetic = pd.DataFrame(synth_rows)
    return synthetic, {
        "healthy_subgroup_share": subgroup_share.to_dict(),
        "healthy_n_subgroups": int(len(subgroup_share)),
        "clinical_n_subgroups": int(models["clinical_kmeans"].n_clusters),
        "clinical_prep": models["clinical_prep"],
        "clinical_kmeans": models["clinical_kmeans"],
    }


def add_engineered_features(frame: pd.DataFrame, clinical_prep: Pipeline, clinical_kmeans: KMeans) -> pd.DataFrame:
    out = frame.copy()
    for col in CORE_COLS + META_OTHER_COLS:
        if col not in out.columns:
            out[col] = np.nan

    age = out["age"].fillna(out["age"].median())
    vhi = out["vhi"].fillna(0.0)
    rsi = out["rsi"].fillna(0.0)
    male = out["gender_male"].fillna(0.0)
    water = out["water_liters_per_day"].fillna(0.0)
    coffee = out["coffee_cups_per_day"].fillna(0.0)
    carbonated = out["carbonated_glasses_per_day"].fillna(0.0)
    alcohol = out["alcohol_glasses_per_day"].fillna(0.0)
    cigarettes = out["cigarettes_per_day"].fillna(0.0)
    citrus = out["citrus_fruits_per_day"].fillna(0.0)
    chocolate = out["chocolate_grams_per_day"].fillna(0.0)
    cheese = out["soft_cheese_grams_per_day"].fillna(0.0)

    out["symptom_sum"] = vhi + rsi
    out["symptom_diff"] = vhi - rsi
    out["symptom_ratio"] = vhi / (rsi + 1.0)
    out["symptom_product"] = vhi * rsi
    out["age_x_vhi"] = age * vhi
    out["age_x_rsi"] = age * rsi
    out["age_x_symptom"] = age * out["symptom_sum"]
    out["male_x_vhi"] = male * vhi
    out["male_x_rsi"] = male * rsi
    out["male_x_symptom"] = male * out["symptom_sum"]
    out["age_sq"] = age ** 2
    out["vhi_sq"] = vhi ** 2
    out["rsi_sq"] = rsi ** 2
    out["vhi_rsi_gap_abs"] = (vhi - rsi).abs()
    out["hydration_gap"] = water - 0.35 * coffee - 0.20 * carbonated - 0.15 * alcohol
    out["irritant_load"] = cigarettes + alcohol + coffee + carbonated + citrus + chocolate / 25.0 + cheese / 25.0
    out["diet_reflux_load"] = coffee + citrus + chocolate / 25.0 + cheese / 25.0
    out["smoke_alcohol_interaction"] = cigarettes * alcohol

    out["age_lt35"] = (age < 35).astype(float)
    out["age_35_50"] = ((age >= 35) & (age < 50)).astype(float)
    out["age_ge50"] = (age >= 50).astype(float)
    out["vhi_ge20"] = (vhi >= 20).astype(float)
    out["rsi_ge13"] = (rsi >= 13).astype(float)

    core_input = out[[col for col in CORE_COLS if col in out.columns]]
    subgroup = clinical_kmeans.predict(clinical_prep.transform(core_input))
    out["clinical_subgroup"] = subgroup.astype(int)
    for idx in range(int(clinical_kmeans.n_clusters)):
        out[f"clinical_subgroup_{idx}"] = (subgroup == idx).astype(float)
    return out


def build_augmented_train_eval(
    healthy_train_real: pd.DataFrame,
    disease_train_real: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    seed: int,
    n_synthetic_healthy: int | None = None,
    acoustic_noise_scale: float = 0.30,
    clinical_noise_scale: float = 0.20,
) -> Dict[str, object]:
    healthy_train_real = healthy_train_real.copy()
    disease_train_real = disease_train_real.copy()
    eval_df = None if eval_df is None else eval_df.copy()

    healthy_train_real["source"] = healthy_train_real.get("source", "real_healthy_train")
    disease_train_real["source"] = disease_train_real.get("source", "real_disease_train")

    synthetic_healthy, meta = generate_subgroup_gaussian_healthy(
        healthy_train_real=healthy_train_real,
        disease_train_real=disease_train_real,
        seed=seed,
        n_synthetic_healthy=n_synthetic_healthy,
        acoustic_noise_scale=acoustic_noise_scale,
        clinical_noise_scale=clinical_noise_scale,
    )

    clinical_prep = meta["clinical_prep"]
    clinical_kmeans = meta["clinical_kmeans"]

    healthy_aug = add_engineered_features(healthy_train_real, clinical_prep, clinical_kmeans)
    disease_aug = add_engineered_features(disease_train_real, clinical_prep, clinical_kmeans)
    synthetic_aug = add_engineered_features(synthetic_healthy, clinical_prep, clinical_kmeans)
    eval_aug = None if eval_df is None else add_engineered_features(eval_df, clinical_prep, clinical_kmeans)

    train_aug = pd.concat([healthy_aug, synthetic_aug, disease_aug], ignore_index=True)
    train_aug = train_aug.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return {
        "train_df": train_aug,
        "eval_df": eval_aug,
        "synthetic_healthy": synthetic_aug,
        "meta": {
            "seed": seed,
            "n_train": int(len(train_aug)),
            "n_real_healthy": int(len(healthy_train_real)),
            "n_real_disease": int(len(disease_train_real)),
            "n_synthetic_healthy": int(len(synthetic_aug)),
            "acoustic_noise_scale": acoustic_noise_scale,
            "clinical_noise_scale": clinical_noise_scale,
            **{k: v for k, v in meta.items() if k not in {"clinical_prep", "clinical_kmeans"}},
        },
    }


def get_feature_views(frame: pd.DataFrame) -> Dict[str, List[str]]:
    feature_cols = [column for column in frame.columns if column not in EXCLUDED_COLS]
    subgroup_cols = sorted([column for column in frame.columns if column.startswith("clinical_subgroup_")])
    clinical_cols = []
    for column in CORE_COLS + META_OTHER_COLS + ENGINEERED_BASE_COLS + subgroup_cols:
        if column in frame.columns and column not in clinical_cols:
            clinical_cols.append(column)
    all_cols = [column for column in feature_cols if column not in {"clinical_subgroup"}] + (["clinical_subgroup"] if "clinical_subgroup" in frame.columns else [])
    acoustic_cols = [column for column in all_cols if column not in set(clinical_cols)]
    hybrid_cols = clinical_cols + acoustic_cols
    return {
        "clinical": clinical_cols,
        "acoustic": acoustic_cols,
        "hybrid": hybrid_cols,
        "all": all_cols,
    }


def build_final_full_artifacts(
    seed: int = 42,
    acoustic_noise_scale: float = 0.30,
    clinical_noise_scale: float = 0.20,
) -> Dict[str, object]:
    base = load_base_real_splits()
    built = build_augmented_train_eval(
        healthy_train_real=base["healthy_train_real"],
        disease_train_real=base["disease_train_real"],
        eval_df=base["fixed_test"],
        seed=seed,
        n_synthetic_healthy=len(base["disease_train_real"]) - len(base["healthy_train_real"]),
        acoustic_noise_scale=acoustic_noise_scale,
        clinical_noise_scale=clinical_noise_scale,
    )
    return {
        "healthy_train_real": add_engineered_features(
            base["healthy_train_real"],
            fit_subgroup_models(base["healthy_train_real"], base["disease_train_real"], [c for c in CORE_COLS if c in base["healthy_train_real"].columns], seed)["clinical_prep"],
            fit_subgroup_models(base["healthy_train_real"], base["disease_train_real"], [c for c in CORE_COLS if c in base["healthy_train_real"].columns], seed)["clinical_kmeans"],
        ),
        "disease_train_real": add_engineered_features(
            base["disease_train_real"],
            fit_subgroup_models(base["healthy_train_real"], base["disease_train_real"], [c for c in CORE_COLS if c in base["healthy_train_real"].columns], seed)["clinical_prep"],
            fit_subgroup_models(base["healthy_train_real"], base["disease_train_real"], [c for c in CORE_COLS if c in base["healthy_train_real"].columns], seed)["clinical_kmeans"],
        ),
        "synthetic_healthy": built["synthetic_healthy"],
        "train_df": built["train_df"],
        "test_df": built["eval_df"],
        "meta": built["meta"],
    }
