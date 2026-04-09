from __future__ import annotations

import argparse
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common_voiced_strong import binary_metrics, json_dump, set_seed
from strict_balance_utils import ARTIFACTS_DIR, RESULTS_DIR, ensure_dirs, build_augmented_train_eval, get_feature_views, load_base_real_splits


BASE_BLEND_MODELS = ["ClinicalCatBoost", "HybridKNN", "HybridRandomForest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict leakage-aware supervised benchmark with fold-wise Gaussian augmentation.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bags", type=int, default=5)
    parser.add_argument("--acoustic_noise_scale", type=float, default=0.30)
    parser.add_argument("--clinical_noise_scale", type=float, default=0.20)
    parser.add_argument("--output_name", type=str, default="strict_supervised_benchmark")
    return parser.parse_args()


def _split_select_params(params: Dict[str, object]) -> Tuple[Dict[str, object], int | None]:
    clean = dict(params)
    select_k = clean.pop("select_k", None)
    return clean, None if select_k is None else int(select_k)


def _maybe_selector(select_k: int | None) -> List[Tuple[str, object]]:
    if select_k is None:
        return []
    return [("select", SelectKBest(score_func=mutual_info_classif, k=select_k))]


def build_model_specs(seed: int) -> Dict[str, Dict[str, object]]:
    return {
        "ClinicalCatBoost": {
            "view": "clinical",
            "grid": [
                {"iterations": 300, "depth": 4, "learning_rate": 0.05},
                {"iterations": 500, "depth": 5, "learning_rate": 0.03},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_state=s, verbose=False, **p)),
            ]),
        },
        "ClinicalRandomForest": {
            "view": "clinical",
            "grid": [
                {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=s, n_jobs=-1, **p)),
            ]),
        },
        "ClinicalBayesianRidge": {
            "view": "clinical",
            "grid": [
                {"alpha_1": 1e-6, "alpha_2": 1e-6},
                {"alpha_1": 1e-5, "alpha_2": 1e-5},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(**p)),
            ]),
        },
        "HybridKNN": {
            "view": "hybrid",
            "grid": [
                {"n_neighbors": 7, "weights": "distance", "select_k": 60},
                {"n_neighbors": 9, "weights": "distance", "select_k": 100},
                {"n_neighbors": 11, "weights": "distance", "select_k": 140},
            ],
            "builder": lambda p, s: (lambda clean, k: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                *_maybe_selector(k),
                ("model", KNeighborsClassifier(**clean)),
            ]))(*_split_select_params(p)),
        },
        "HybridSVM": {
            "view": "hybrid",
            "grid": [
                {"C": 1.5, "kernel": "rbf", "gamma": "scale", "select_k": 60},
                {"C": 2.0, "kernel": "rbf", "gamma": "scale", "select_k": 100},
                {"C": 3.0, "kernel": "rbf", "gamma": "scale", "select_k": 140},
            ],
            "builder": lambda p, s: (lambda clean, k: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                *_maybe_selector(k),
                ("model", SVC(probability=True, random_state=s, **clean)),
            ]))(*_split_select_params(p)),
        },
        "HybridRandomForest": {
            "view": "hybrid",
            "grid": [
                {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 2},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=s, n_jobs=-1, **p)),
            ]),
        },
        "HybridExtraTrees": {
            "view": "hybrid",
            "grid": [
                {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
                {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 2},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(random_state=s, n_jobs=-1, **p)),
            ]),
        },
        "HybridCatBoost": {
            "view": "hybrid",
            "grid": [
                {"iterations": 350, "depth": 5, "learning_rate": 0.04},
                {"iterations": 500, "depth": 6, "learning_rate": 0.03},
            ],
            "builder": lambda p, s: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_state=s, verbose=False, **p)),
            ]),
        },
    }


def predict_score(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    model = estimator.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    if np.allclose(scores.min(), scores.max()):
        threshold = float(scores.mean())
        pred = (scores >= threshold).astype(int)
        return threshold, float(accuracy_score(y_true, pred))
    grid = np.linspace(float(scores.min()), float(scores.max()), 401)
    best_threshold = float(grid[0])
    best_acc = -np.inf
    best_mcc = -np.inf
    for threshold in grid:
        pred = (scores >= threshold).astype(int)
        acc = float(accuracy_score(y_true, pred))
        mcc = float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else -1.0
        if acc > best_acc or (np.isclose(acc, best_acc) and mcc > best_mcc):
            best_acc = acc
            best_mcc = mcc
            best_threshold = float(threshold)
    return best_threshold, best_acc


def average_bagged_scores(
    spec: Dict[str, object],
    params: Dict[str, object],
    healthy_train_real: pd.DataFrame,
    disease_train_real: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed: int,
    n_bags: int,
    acoustic_noise_scale: float,
    clinical_noise_scale: float,
) -> np.ndarray:
    bag_scores: List[np.ndarray] = []
    for bag in range(n_bags):
        bag_seed = seed + 1009 * bag
        built = build_augmented_train_eval(
            healthy_train_real=healthy_train_real,
            disease_train_real=disease_train_real,
            eval_df=eval_df,
            seed=bag_seed,
            acoustic_noise_scale=acoustic_noise_scale,
            clinical_noise_scale=clinical_noise_scale,
        )
        train_aug = built["train_df"]
        eval_aug = built["eval_df"]
        view_cols = get_feature_views(train_aug)[str(spec["view"])]
        estimator = spec["builder"](params, bag_seed)  # type: ignore[index]
        estimator.fit(train_aug[view_cols], train_aug["label_binary_pathology"].to_numpy(dtype=int))
        bag_scores.append(predict_score(estimator, eval_aug[view_cols]))
    return np.mean(np.vstack(bag_scores), axis=0)


def evaluate_param_set(
    model_name: str,
    spec: Dict[str, object],
    params: Dict[str, object],
    real_train_df: pd.DataFrame,
    folds: int,
    seed: int,
    n_bags: int,
    acoustic_noise_scale: float,
    clinical_noise_scale: float,
) -> Dict[str, object]:
    y = real_train_df["label_binary_pathology"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(real_train_df), dtype=float)
    fold_rows: List[Dict[str, object]] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(real_train_df, y), start=1):
        train_real = real_train_df.iloc[train_idx].reset_index(drop=True)
        val_real = real_train_df.iloc[val_idx].reset_index(drop=True)
        healthy_fold = train_real.loc[train_real["label_binary_pathology"] == 0].reset_index(drop=True)
        disease_fold = train_real.loc[train_real["label_binary_pathology"] == 1].reset_index(drop=True)
        val_scores = average_bagged_scores(
            spec=spec,
            params=params,
            healthy_train_real=healthy_fold,
            disease_train_real=disease_fold,
            eval_df=val_real,
            seed=seed + 37 * fold,
            n_bags=n_bags,
            acoustic_noise_scale=acoustic_noise_scale,
            clinical_noise_scale=clinical_noise_scale,
        )
        oof[val_idx] = val_scores
        fold_rows.append({
            "fold": fold,
            "auc": safe_auc(val_real["label_binary_pathology"].to_numpy(dtype=int), val_scores),
        })

    threshold, acc = tune_threshold(y, oof)
    pred = (oof >= threshold).astype(int)
    metrics = binary_metrics(y, pred, oof)
    return {
        "model": model_name,
        "params": params,
        "cv_auc": metrics["auc"],
        "cv_accuracy": acc,
        "cv_balanced_accuracy": metrics["balanced_accuracy"],
        "cv_f1": metrics["f1"],
        "cv_mcc": metrics["mcc"],
        "threshold": threshold,
        "oof_scores": oof,
        "fold_metrics": fold_rows,
    }


def choose_best(results: List[Dict[str, object]]) -> Dict[str, object]:
    return sorted(results, key=lambda item: (-float(item["cv_auc"]), -float(item["cv_accuracy"]), -float(item["cv_mcc"])))[0]


def grid_search_blend_weights(oof_map: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) / 0.05)) + 1):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            weights = np.array([w1, w2, w3], dtype=float)
            if np.isclose(weights.sum(), 0.0):
                continue
            blend = sum(weights[idx] * oof_map[name] for idx, name in enumerate(BASE_BLEND_MODELS))
            threshold, acc = tune_threshold(y_true, blend)
            pred = (blend >= threshold).astype(int)
            metrics = binary_metrics(y_true, pred, blend)
            candidate = {
                "weights": {name: float(weights[idx]) for idx, name in enumerate(BASE_BLEND_MODELS)},
                "cv_auc": metrics["auc"],
                "cv_accuracy": acc,
                "cv_balanced_accuracy": metrics["balanced_accuracy"],
                "cv_f1": metrics["f1"],
                "cv_mcc": metrics["mcc"],
                "threshold": threshold,
                "oof_scores": blend,
            }
            if best is None or (
                float(candidate["cv_auc"]) > float(best["cv_auc"])
                or (
                    np.isclose(float(candidate["cv_auc"]), float(best["cv_auc"]))
                    and float(candidate["cv_accuracy"]) > float(best["cv_accuracy"])
                )
            ):
                best = candidate
    assert best is not None
    return best


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dirs()
    base = load_base_real_splits()
    real_train_df = pd.concat([base["healthy_train_real"], base["disease_train_real"]], ignore_index=True)
    real_train_df = real_train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    fixed_test = base["fixed_test"].copy().reset_index(drop=True)

    specs = build_model_specs(args.seed)
    out_dir = RESULTS_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {}
    oof_pred_rows: List[pd.DataFrame] = []
    test_pred_rows: List[pd.DataFrame] = []
    best_map: Dict[str, Dict[str, object]] = {}
    base_oof_map: Dict[str, np.ndarray] = {}
    base_test_map: Dict[str, np.ndarray] = {}

    for model_name, spec in specs.items():
        print(f"[{model_name}] leakage-aware CV search")
        param_results = [
            evaluate_param_set(
                model_name=model_name,
                spec=spec,
                params=params,
                real_train_df=real_train_df,
                folds=args.folds,
                seed=args.seed,
                n_bags=args.n_bags,
                acoustic_noise_scale=args.acoustic_noise_scale,
                clinical_noise_scale=args.clinical_noise_scale,
            )
            for params in spec["grid"]  # type: ignore[index]
        ]
        best = choose_best(param_results)
        best_map[model_name] = best

        healthy_full = base["healthy_train_real"].copy().reset_index(drop=True)
        disease_full = base["disease_train_real"].copy().reset_index(drop=True)
        test_scores = average_bagged_scores(
            spec=spec,
            params=best["params"],
            healthy_train_real=healthy_full,
            disease_train_real=disease_full,
            eval_df=fixed_test,
            seed=args.seed + 999,
            n_bags=args.n_bags,
            acoustic_noise_scale=args.acoustic_noise_scale,
            clinical_noise_scale=args.clinical_noise_scale,
        )
        threshold = float(best["threshold"])
        test_pred = (test_scores >= threshold).astype(int)
        y_train_real = real_train_df["label_binary_pathology"].to_numpy(dtype=int)
        y_test = fixed_test["label_binary_pathology"].to_numpy(dtype=int)
        train_oof = np.asarray(best["oof_scores"], dtype=float)
        train_pred = (train_oof >= threshold).astype(int)
        train_metrics = binary_metrics(y_train_real, train_pred, train_oof)
        test_metrics = binary_metrics(y_test, test_pred, test_scores)

        summary_rows.append({
            "model": model_name,
            "selected_params": str(best["params"]),
            "cv_train_auc": train_metrics["auc"],
            "cv_train_accuracy": train_metrics["accuracy"],
            "cv_train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "cv_train_f1": train_metrics["f1"],
            "cv_train_mcc": train_metrics["mcc"],
            "test_auc": test_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_mcc": test_metrics["mcc"],
            "selected_threshold": threshold,
        })
        details[model_name] = {
            "best_cv": {key: value for key, value in best.items() if key != "oof_scores"},
            "train_oof_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        oof_pred_rows.append(pd.DataFrame({
            "record_id": real_train_df["record_id"],
            "label": y_train_real,
            "score": train_oof,
            "pred": train_pred,
            "model": model_name,
        }))
        test_pred_rows.append(pd.DataFrame({
            "record_id": fixed_test["record_id"],
            "label": y_test,
            "score": test_scores,
            "pred": test_pred,
            "model": model_name,
        }))
        if model_name in BASE_BLEND_MODELS:
            base_oof_map[model_name] = train_oof
            base_test_map[model_name] = test_scores
        print(f"  cv_train_auc={train_metrics['auc']:.4f} test_auc={test_metrics['auc']:.4f} test_acc={test_metrics['accuracy']:.4f}")

    if all(name in base_oof_map for name in BASE_BLEND_MODELS):
        print("[Blend_CatBoost_KNN_RF] OOF weight search")
        y_train_real = real_train_df["label_binary_pathology"].to_numpy(dtype=int)
        y_test = fixed_test["label_binary_pathology"].to_numpy(dtype=int)
        blend_best = grid_search_blend_weights(base_oof_map, y_train_real)
        blend_oof = np.asarray(blend_best["oof_scores"], dtype=float)
        blend_test = sum(blend_best["weights"][name] * base_test_map[name] for name in BASE_BLEND_MODELS)
        threshold = float(blend_best["threshold"])
        train_pred = (blend_oof >= threshold).astype(int)
        test_pred = (blend_test >= threshold).astype(int)
        train_metrics = binary_metrics(y_train_real, train_pred, blend_oof)
        test_metrics = binary_metrics(y_test, test_pred, blend_test)
        summary_rows.append({
            "model": "Blend_CatBoost_KNN_RF",
            "selected_params": str(blend_best["weights"]),
            "cv_train_auc": train_metrics["auc"],
            "cv_train_accuracy": train_metrics["accuracy"],
            "cv_train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "cv_train_f1": train_metrics["f1"],
            "cv_train_mcc": train_metrics["mcc"],
            "test_auc": test_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_mcc": test_metrics["mcc"],
            "selected_threshold": threshold,
        })
        details["Blend_CatBoost_KNN_RF"] = {
            "best_cv": {key: value for key, value in blend_best.items() if key != "oof_scores"},
            "train_oof_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        oof_pred_rows.append(pd.DataFrame({
            "record_id": real_train_df["record_id"],
            "label": y_train_real,
            "score": blend_oof,
            "pred": train_pred,
            "model": "Blend_CatBoost_KNN_RF",
        }))
        test_pred_rows.append(pd.DataFrame({
            "record_id": fixed_test["record_id"],
            "label": y_test,
            "score": blend_test,
            "pred": test_pred,
            "model": "Blend_CatBoost_KNN_RF",
        }))
        print(f"  cv_train_auc={train_metrics['auc']:.4f} test_auc={test_metrics['auc']:.4f} test_acc={test_metrics['accuracy']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["test_auc", "test_accuracy", "test_mcc"], ascending=False).reset_index(drop=True)
    summary_df.to_csv(out_dir / "strict_supervised_model_summary.csv", index=False)
    pd.concat(oof_pred_rows, ignore_index=True).to_csv(out_dir / "strict_supervised_train_oof_predictions.csv", index=False)
    pd.concat(test_pred_rows, ignore_index=True).to_csv(out_dir / "strict_supervised_test_predictions.csv", index=False)

    report = {
        "best_by_test_auc": summary_df.sort_values(["test_auc", "test_accuracy"], ascending=False).iloc[0].to_dict(),
        "best_by_test_accuracy": summary_df.sort_values(["test_accuracy", "test_auc"], ascending=False).iloc[0].to_dict(),
        "details": details,
    }
    json_dump(report, out_dir / "strict_supervised_summary.json")

    print("\nStrict supervised summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved: {out_dir / 'strict_supervised_summary.json'}")


if __name__ == "__main__":
    main(parse_args())
