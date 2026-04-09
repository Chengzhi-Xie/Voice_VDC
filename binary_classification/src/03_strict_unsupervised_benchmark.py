from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from common_voiced_strong import binary_metrics, json_dump, set_seed
from strict_balance_utils import RESULTS_DIR, ensure_dirs, build_augmented_train_eval, get_feature_views, load_base_real_splits


BASE_BLEND_MODELS = ["SubgroupGaussianClinical", "IsolationForestClinical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict leakage-aware unsupervised benchmark with fold-wise Gaussian augmentation.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acoustic_noise_scale", type=float, default=0.30)
    parser.add_argument("--clinical_noise_scale", type=float, default=0.20)
    parser.add_argument("--output_name", type=str, default="strict_unsupervised_benchmark")
    return parser.parse_args()


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


def build_model_specs(seed: int) -> Dict[str, List[Dict[str, object]]]:
    return {
        "SubgroupGaussianClinical": [
            {"n_clusters": 2},
            {"n_clusters": 3},
        ],
        "SubgroupGaussianHybridPCA": [
            {"n_clusters": 2, "n_components": 8},
            {"n_clusters": 3, "n_components": 12},
        ],
        "IsolationForestClinical": [
            {"n_estimators": 400, "contamination": 0.45, "random_state": seed},
            {"n_estimators": 600, "contamination": 0.50, "random_state": seed},
        ],
        "IsolationForestHybridPCA": [
            {"n_estimators": 500, "contamination": 0.45, "random_state": seed, "n_components": 8},
            {"n_estimators": 700, "contamination": 0.50, "random_state": seed, "n_components": 12},
        ],
        "OneClassSVMClinical": [
            {"nu": 0.30, "gamma": "scale"},
            {"nu": 0.40, "gamma": "scale"},
        ],
        "LOFClinical": [
            {"n_neighbors": 10, "novelty": True, "contamination": 0.45},
            {"n_neighbors": 20, "novelty": True, "contamination": 0.50},
        ],
    }


def _fit_scaled(df_fit: pd.DataFrame, df_train: pd.DataFrame, df_eval: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_fit = prep.fit_transform(df_fit[columns])
    X_train = prep.transform(df_train[columns])
    X_eval = prep.transform(df_eval[columns])
    return X_fit, X_train, X_eval


def build_clinical_representation(train_aug: pd.DataFrame, eval_aug: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = get_feature_views(train_aug)
    clinical_cols = views["clinical"]
    healthy = train_aug.loc[train_aug["label_binary_pathology"] == 0].reset_index(drop=True)
    return _fit_scaled(healthy, train_aug, eval_aug, clinical_cols)


def build_hybrid_pca_representation(train_aug: pd.DataFrame, eval_aug: pd.DataFrame, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = get_feature_views(train_aug)
    clinical_cols = views["clinical"]
    acoustic_cols = views["acoustic"]
    healthy = train_aug.loc[train_aug["label_binary_pathology"] == 0].reset_index(drop=True)

    Zc_fit, Zc_train, Zc_eval = _fit_scaled(healthy, train_aug, eval_aug, clinical_cols)
    if not acoustic_cols:
        return Zc_fit, Zc_train, Zc_eval

    Xa_fit, Xa_train, Xa_eval = _fit_scaled(healthy, train_aug, eval_aug, acoustic_cols)
    max_comp = max(1, min(n_components, Xa_fit.shape[0] - 1, Xa_fit.shape[1]))
    pca = PCA(n_components=max_comp, random_state=42)
    Pa_fit = pca.fit_transform(Xa_fit)
    Pa_train = pca.transform(Xa_train)
    Pa_eval = pca.transform(Xa_eval)
    return np.hstack([Zc_fit, Pa_fit]), np.hstack([Zc_train, Pa_train]), np.hstack([Zc_eval, Pa_eval])


def subgroup_gaussian_scores(X_fit_healthy: np.ndarray, X_eval: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_clusters = max(1, min(int(n_clusters), len(X_fit_healthy)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=30)
    assign = kmeans.fit_predict(X_fit_healthy)
    clusters: List[Tuple[np.ndarray, np.ndarray]] = []
    for cluster_id in range(n_clusters):
        cluster_data = X_fit_healthy[assign == cluster_id]
        if len(cluster_data) == 0:
            continue
        mu = cluster_data.mean(axis=0)
        var = cluster_data.var(axis=0) + 1e-3
        clusters.append((mu, var))
    all_scores: List[np.ndarray] = []
    for mu, var in clusters:
        nll = 0.5 * np.sum(np.log(var)) + 0.5 * np.sum(((X_eval - mu) ** 2) / var, axis=1)
        all_scores.append(nll)
    return np.min(np.vstack(all_scores), axis=0) if all_scores else np.zeros(len(X_eval), dtype=float)


def fit_score_model(model_name: str, params: Dict[str, object], train_aug: pd.DataFrame, eval_aug: pd.DataFrame, seed: int) -> np.ndarray:
    healthy_mask = train_aug["label_binary_pathology"].to_numpy(dtype=int) == 0

    if model_name == "SubgroupGaussianClinical":
        X_fit, _, X_eval = build_clinical_representation(train_aug, eval_aug)
        return subgroup_gaussian_scores(X_fit, X_eval, int(params["n_clusters"]), seed)

    if model_name == "SubgroupGaussianHybridPCA":
        X_fit, _, X_eval = build_hybrid_pca_representation(train_aug, eval_aug, int(params["n_components"]))
        return subgroup_gaussian_scores(X_fit, X_eval, int(params["n_clusters"]), seed)

    if model_name == "IsolationForestClinical":
        X_fit, _, X_eval = build_clinical_representation(train_aug, eval_aug)
        clean = {k: v for k, v in params.items() if k != "n_components"}
        model = IsolationForest(**clean)
        model.fit(X_fit)
        return -model.decision_function(X_eval)

    if model_name == "IsolationForestHybridPCA":
        X_fit, _, X_eval = build_hybrid_pca_representation(train_aug, eval_aug, int(params["n_components"]))
        clean = {k: v for k, v in params.items() if k != "n_components"}
        model = IsolationForest(**clean)
        model.fit(X_fit)
        return -model.decision_function(X_eval)

    if model_name == "OneClassSVMClinical":
        X_fit, _, X_eval = build_clinical_representation(train_aug, eval_aug)
        model = OneClassSVM(**params)
        model.fit(X_fit)
        return -model.decision_function(X_eval)

    X_fit, _, X_eval = build_clinical_representation(train_aug, eval_aug)
    model = LocalOutlierFactor(**params)
    model.fit(X_fit)
    return -model.decision_function(X_eval)


def evaluate_param_set(
    model_name: str,
    params: Dict[str, object],
    real_train_df: pd.DataFrame,
    folds: int,
    seed: int,
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
        built = build_augmented_train_eval(
            healthy_train_real=healthy_fold,
            disease_train_real=disease_fold,
            eval_df=val_real,
            seed=seed + 53 * fold,
            acoustic_noise_scale=acoustic_noise_scale,
            clinical_noise_scale=clinical_noise_scale,
        )
        val_scores = fit_score_model(model_name, params, built["train_df"], built["eval_df"], seed + 97 * fold)
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


def empirical_cdf_scores(train_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(train_scores, dtype=float))
    return np.searchsorted(ordered, np.asarray(query_scores, dtype=float), side="right") / float(len(ordered))


def search_blend(oof_map: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    left = BASE_BLEND_MODELS[0]
    right = BASE_BLEND_MODELS[1]
    left_rank = pd.Series(oof_map[left]).rank(method="average", pct=True).to_numpy(dtype=float)
    right_rank = pd.Series(oof_map[right]).rank(method="average", pct=True).to_numpy(dtype=float)
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * left_rank + (1.0 - w) * right_rank
        threshold, acc = tune_threshold(y_true, blend)
        pred = (blend >= threshold).astype(int)
        metrics = binary_metrics(y_true, pred, blend)
        candidate = {
            "weights": {left: float(w), right: float(1.0 - w)},
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
    oof_rows: List[pd.DataFrame] = []
    test_rows: List[pd.DataFrame] = []
    base_oof_map: Dict[str, np.ndarray] = {}
    base_test_map: Dict[str, np.ndarray] = {}

    for model_name, grid in specs.items():
        print(f"[{model_name}] leakage-aware CV search")
        results = [
            evaluate_param_set(
                model_name=model_name,
                params=params,
                real_train_df=real_train_df,
                folds=args.folds,
                seed=args.seed,
                acoustic_noise_scale=args.acoustic_noise_scale,
                clinical_noise_scale=args.clinical_noise_scale,
            )
            for params in grid
        ]
        best = choose_best(results)

        built = build_augmented_train_eval(
            healthy_train_real=base["healthy_train_real"],
            disease_train_real=base["disease_train_real"],
            eval_df=fixed_test,
            seed=args.seed + 999,
            acoustic_noise_scale=args.acoustic_noise_scale,
            clinical_noise_scale=args.clinical_noise_scale,
        )
        test_scores = fit_score_model(model_name, best["params"], built["train_df"], built["eval_df"], args.seed + 999)

        y_train_real = real_train_df["label_binary_pathology"].to_numpy(dtype=int)
        y_test = fixed_test["label_binary_pathology"].to_numpy(dtype=int)
        threshold = float(best["threshold"])
        train_oof = np.asarray(best["oof_scores"], dtype=float)
        train_pred = (train_oof >= threshold).astype(int)
        test_pred = (test_scores >= threshold).astype(int)
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
        oof_rows.append(pd.DataFrame({
            "record_id": real_train_df["record_id"],
            "label": y_train_real,
            "score": train_oof,
            "pred": train_pred,
            "model": model_name,
        }))
        test_rows.append(pd.DataFrame({
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
        print("[Blend_Gaussian_IsoClinical] OOF blend search")
        y_train_real = real_train_df["label_binary_pathology"].to_numpy(dtype=int)
        y_test = fixed_test["label_binary_pathology"].to_numpy(dtype=int)
        blend_best = search_blend(base_oof_map, y_train_real)

        left = BASE_BLEND_MODELS[0]
        right = BASE_BLEND_MODELS[1]
        blend_oof = np.asarray(blend_best["oof_scores"], dtype=float)
        left_test = empirical_cdf_scores(base_oof_map[left], base_test_map[left])
        right_test = empirical_cdf_scores(base_oof_map[right], base_test_map[right])
        blend_test = blend_best["weights"][left] * left_test + blend_best["weights"][right] * right_test
        threshold = float(blend_best["threshold"])
        train_pred = (blend_oof >= threshold).astype(int)
        test_pred = (blend_test >= threshold).astype(int)
        train_metrics = binary_metrics(y_train_real, train_pred, blend_oof)
        test_metrics = binary_metrics(y_test, test_pred, blend_test)

        summary_rows.append({
            "model": "Blend_Gaussian_IsoClinical",
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
        details["Blend_Gaussian_IsoClinical"] = {
            "best_cv": {key: value for key, value in blend_best.items() if key != "oof_scores"},
            "train_oof_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        oof_rows.append(pd.DataFrame({
            "record_id": real_train_df["record_id"],
            "label": y_train_real,
            "score": blend_oof,
            "pred": train_pred,
            "model": "Blend_Gaussian_IsoClinical",
        }))
        test_rows.append(pd.DataFrame({
            "record_id": fixed_test["record_id"],
            "label": y_test,
            "score": blend_test,
            "pred": test_pred,
            "model": "Blend_Gaussian_IsoClinical",
        }))
        print(f"  cv_train_auc={train_metrics['auc']:.4f} test_auc={test_metrics['auc']:.4f} test_acc={test_metrics['accuracy']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["test_auc", "test_accuracy", "test_mcc"], ascending=False).reset_index(drop=True)
    summary_df.to_csv(out_dir / "strict_unsupervised_model_summary.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(out_dir / "strict_unsupervised_train_oof_predictions.csv", index=False)
    pd.concat(test_rows, ignore_index=True).to_csv(out_dir / "strict_unsupervised_test_predictions.csv", index=False)
    report = {
        "best_by_test_auc": summary_df.sort_values(["test_auc", "test_accuracy"], ascending=False).iloc[0].to_dict(),
        "best_by_test_accuracy": summary_df.sort_values(["test_accuracy", "test_auc"], ascending=False).iloc[0].to_dict(),
        "details": details,
    }
    json_dump(report, out_dir / "strict_unsupervised_summary.json")

    print("\nStrict unsupervised summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved: {out_dir / 'strict_unsupervised_summary.json'}")


if __name__ == "__main__":
    main(parse_args())
