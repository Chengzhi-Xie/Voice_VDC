from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common_voiced_strong import (
    CAT_META_COLUMNS,
    MULTICLASS_ORDER,
    NUM_META_COLUMNS,
    binary_metrics,
    build_parser,
    ensure_project_paths,
    json_dump,
    multiclass_metrics,
    set_seed,
    tune_binary_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = build_parser("Train stronger supervised VOICED tasks with clean features and late fusion.")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--handcrafted_csv", type=str, default="")
    parser.add_argument("--wav2vec_csv", type=str, default="")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--eval_mode", type=str, default="holdout", choices=["holdout", "cv5"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--fusion_metric", type=str, default="mcc", choices=["mcc", "f1", "balanced_accuracy"])
    parser.add_argument("--output_name", type=str, default="default")
    return parser.parse_args()


def resolve_input_path(project_root: Path, provided: str, fallback_name: str) -> Path:
    if provided:
        return Path(provided)
    return project_root / "data" / "processed" / fallback_name


def load_feature_tables(project_root: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    metadata = pd.read_csv(resolve_input_path(project_root, args.metadata_csv, "voiced_metadata_strong.csv"))
    handcrafted = pd.read_csv(resolve_input_path(project_root, args.handcrafted_csv, "voiced_handcrafted_features_strong.csv"))
    wav2vec = None
    if args.wav2vec_csv:
        wav2vec = pd.read_csv(args.wav2vec_csv)
    else:
        candidates = sorted((project_root / "data" / "processed").glob("voiced_wav2vec2*features_strong.csv"))
        if candidates:
            wav2vec = pd.read_csv(candidates[0])
    return metadata, handcrafted, wav2vec


def build_design_matrices(
    metadata: pd.DataFrame,
    handcrafted: pd.DataFrame,
    wav2vec: Optional[pd.DataFrame],
    task: str,
) -> Dict[str, object]:
    df = metadata.merge(handcrafted, on="record_id", how="inner")
    if wav2vec is not None:
        df = df.merge(wav2vec, on="record_id", how="left")

    y = df["label_binary_pathology"].to_numpy(dtype=int) if task == "binary" else df["label_multiclass"].to_numpy(dtype=int)
    target_name = "label_binary_pathology" if task == "binary" else "label_multiclass"

    tabular_cat = [col for col in CAT_META_COLUMNS if col in df.columns]
    tabular_num = [col for col in NUM_META_COLUMNS if col in df.columns]
    handcrafted_cols = [col for col in handcrafted.columns if col not in {"record_id", "sample_rate"}]
    tabular_cols = tabular_num + tabular_cat + handcrafted_cols
    X_tabular = df[tabular_cols].copy()
    for col in tabular_cat:
        X_tabular[col] = X_tabular[col].fillna("missing").astype(str)

    X_w2v = None
    w2v_cols: List[str] = []
    if wav2vec is not None:
        w2v_cols = [col for col in wav2vec.columns if col not in {"record_id", "n_views", "target_sample_rate"}]
        X_w2v = df[w2v_cols].copy()

    return {
        "full_df": df,
        "target_name": target_name,
        "y": y,
        "X_tabular": X_tabular,
        "tabular_cat_cols": tabular_cat,
        "X_w2v": X_w2v,
        "w2v_cols": w2v_cols,
    }


def fit_tabular_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
    task: str,
    seed: int,
) -> CatBoostClassifier:
    if task == "binary":
        model = CatBoostClassifier(
            iterations=900,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_state=seed,
            verbose=False,
        )
    else:
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            auto_class_weights="Balanced",
            random_state=seed,
            verbose=False,
        )
    model.fit(X_train, y_train, cat_features=cat_cols)
    return model


def fit_w2v_model(X_train: pd.DataFrame, y_train: np.ndarray, task: str, seed: int) -> Pipeline:
    svc = SVC(
        C=3.0,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        decision_function_shape="ovr",
        random_state=seed,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.97, svd_solver="full")),
            ("svc", svc),
        ]
    )
    model.fit(X_train, y_train)
    return model


def search_binary_fusion(
    y_val: np.ndarray,
    tab_prob: np.ndarray,
    w2v_prob: Optional[np.ndarray],
    metric: str,
) -> Tuple[float, float]:
    if w2v_prob is None:
        return 1.0, tune_binary_threshold(y_val, tab_prob, metric=metric)[0]

    best_weight = 1.0
    best_threshold = 0.5
    best_value = -np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        fused = weight * tab_prob + (1.0 - weight) * w2v_prob
        threshold, value = tune_binary_threshold(y_val, fused, metric=metric)
        if value > best_value:
            best_value = value
            best_weight = float(weight)
            best_threshold = float(threshold)
    return best_weight, best_threshold


def search_multiclass_fusion(
    y_val: np.ndarray,
    tab_prob: np.ndarray,
    w2v_prob: Optional[np.ndarray],
) -> float:
    if w2v_prob is None:
        return 1.0
    best_weight = 1.0
    best_value = -np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        fused = weight * tab_prob + (1.0 - weight) * w2v_prob
        pred = fused.argmax(axis=1)
        value = f1_score(y_val, pred, average="macro", zero_division=0)
        if value > best_value:
            best_value = value
            best_weight = float(weight)
    return best_weight


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


def evaluate_one_split(
    data: Dict[str, object],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    task: str,
    seed: int,
    fusion_metric: str,
) -> Dict[str, object]:
    X_tabular: pd.DataFrame = data["X_tabular"]  # type: ignore[assignment]
    X_w2v: Optional[pd.DataFrame] = data["X_w2v"]  # type: ignore[assignment]
    y: np.ndarray = data["y"]  # type: ignore[assignment]
    cat_cols: List[str] = data["tabular_cat_cols"]  # type: ignore[assignment]

    tab_model = fit_tabular_model(X_tabular.iloc[train_idx], y[train_idx], cat_cols, task, seed)
    tab_val_prob = tab_model.predict_proba(X_tabular.iloc[val_idx])
    tab_test_prob = tab_model.predict_proba(X_tabular.iloc[test_idx])

    w2v_model = None
    w2v_val_prob = None
    w2v_test_prob = None
    if X_w2v is not None:
        w2v_model = fit_w2v_model(X_w2v.iloc[train_idx], y[train_idx], task, seed)
        w2v_val_prob = w2v_model.predict_proba(X_w2v.iloc[val_idx])
        w2v_test_prob = w2v_model.predict_proba(X_w2v.iloc[test_idx])

    if task == "binary":
        tab_val_prob_1 = tab_val_prob[:, 1]
        tab_test_prob_1 = tab_test_prob[:, 1]
        w2v_val_prob_1 = None if w2v_val_prob is None else w2v_val_prob[:, 1]
        w2v_test_prob_1 = None if w2v_test_prob is None else w2v_test_prob[:, 1]
        fusion_weight, threshold = search_binary_fusion(y[val_idx], tab_val_prob_1, w2v_val_prob_1, fusion_metric)
        fused_test_prob = tab_test_prob_1 if w2v_test_prob_1 is None else fusion_weight * tab_test_prob_1 + (1.0 - fusion_weight) * w2v_test_prob_1
        fused_test_pred = (fused_test_prob >= threshold).astype(int)
        metrics = binary_metrics(y[test_idx], fused_test_pred, fused_test_prob)
        result = {
            "fusion_weight": fusion_weight,
            "threshold": threshold,
            "metrics": metrics,
            "y_true": y[test_idx],
            "y_pred": fused_test_pred,
            "y_score": fused_test_prob,
        }
    else:
        fusion_weight = search_multiclass_fusion(y[val_idx], tab_val_prob, w2v_val_prob)
        fused_prob = tab_test_prob if w2v_test_prob is None else fusion_weight * tab_test_prob + (1.0 - fusion_weight) * w2v_test_prob
        fused_pred = fused_prob.argmax(axis=1)
        metrics = multiclass_metrics(y[test_idx], fused_pred, fused_prob)
        result = {
            "fusion_weight": fusion_weight,
            "metrics": metrics,
            "y_true": y[test_idx],
            "y_pred": fused_pred,
            "y_prob": fused_prob,
        }
    return result


def run_holdout(
    data: Dict[str, object],
    task: str,
    test_size: float,
    val_size: float,
    seed: int,
    fusion_metric: str,
) -> Dict[str, object]:
    y: np.ndarray = data["y"]  # type: ignore[assignment]
    train_idx, val_idx, test_idx = split_holdout(y, test_size, val_size, seed)
    return {
        "splits": {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
        "evaluation": evaluate_one_split(data, train_idx, val_idx, test_idx, task, seed, fusion_metric),
    }


def run_cv5(
    data: Dict[str, object],
    task: str,
    seed: int,
    fusion_metric: str,
) -> Dict[str, object]:
    y: np.ndarray = data["y"]  # type: ignore[assignment]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results: List[Dict[str, object]] = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float) if task == "binary" else None
    oof_prob = np.zeros((len(y), len(np.unique(y))), dtype=float) if task == "multiclass" else None

    for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        inner_y = y[train_val_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold_id)
        train_rel, val_rel = next(sss.split(np.zeros(len(train_val_idx)), inner_y))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
        result = evaluate_one_split(data, train_idx, val_idx, test_idx, task, seed + fold_id, fusion_metric)
        metrics = {"fold": fold_id, **result["metrics"]}  # type: ignore[index]
        fold_results.append(metrics)
        oof_pred[test_idx] = result["y_pred"]  # type: ignore[index]
        if task == "binary":
            oof_score[test_idx] = result["y_score"]  # type: ignore[index]
        else:
            oof_prob[test_idx] = result["y_prob"]  # type: ignore[index]

    aggregate = binary_metrics(y, oof_pred, oof_score) if task == "binary" else multiclass_metrics(y, oof_pred, oof_prob)
    return {"fold_metrics": fold_results, "aggregate_metrics": aggregate}


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    paths = ensure_project_paths(args.project_root)
    metadata, handcrafted, wav2vec = load_feature_tables(paths["root"], args)
    data = build_design_matrices(metadata, handcrafted, wav2vec, args.task)

    if args.eval_mode == "holdout":
        summary = run_holdout(data, args.task, args.test_size, args.val_size, args.seed, args.fusion_metric)
    else:
        summary = run_cv5(data, args.task, args.seed, args.fusion_metric)

    tag = f"{args.task}_{args.eval_mode}_{args.output_name}"
    out_dir = paths["supervised_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dump(summary, out_dir / "metrics.json")

    full_df: pd.DataFrame = data["full_df"]  # type: ignore[assignment]
    full_df.to_csv(out_dir / "merged_feature_index.csv", index=False)

    if args.task == "binary" and args.eval_mode == "holdout":
        print("Holdout binary metrics:")
        print(pd.Series(summary["evaluation"]["metrics"]).to_string())  # type: ignore[index]
    elif args.task == "multiclass" and args.eval_mode == "holdout":
        print("Holdout multiclass metrics:")
        print(pd.Series(summary["evaluation"]["metrics"]).to_string())  # type: ignore[index]
    else:
        print("CV summary:")
        print(pd.Series(summary["aggregate_metrics"]).to_string())  # type: ignore[index]
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(parse_args())
