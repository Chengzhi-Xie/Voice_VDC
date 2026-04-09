from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_voiced_strong import (
    CAT_META_COLUMNS,
    NUM_META_COLUMNS,
    binary_metrics,
    build_parser,
    clustering_metrics,
    ensure_project_paths,
    json_dump,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = build_parser("Run unsupervised VOICED tasks: clustering and anomaly scoring.")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--handcrafted_csv", type=str, default="")
    parser.add_argument("--wav2vec_csv", type=str, default="")
    parser.add_argument("--feature_set", type=str, default="fusion", choices=["tabular", "fusion"])
    parser.add_argument("--output_name", type=str, default="default")
    return parser.parse_args()


def resolve_input_path(project_root: Path, provided: str, fallback_name: str) -> Path:
    if provided:
        return Path(provided)
    return project_root / "data" / "processed" / fallback_name


def load_tables(project_root: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
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


def prepare_matrix(
    metadata: pd.DataFrame,
    handcrafted: pd.DataFrame,
    wav2vec: Optional[pd.DataFrame],
    feature_set: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    df = metadata.merge(handcrafted, on="record_id", how="inner")
    if feature_set == "fusion" and wav2vec is not None:
        df = df.merge(wav2vec, on="record_id", how="left")

    meta_cols = [col for col in NUM_META_COLUMNS + CAT_META_COLUMNS if col in df.columns]
    hand_cols = [col for col in handcrafted.columns if col not in {"record_id", "sample_rate"}]
    feature_cols = meta_cols + hand_cols
    if feature_set == "fusion" and wav2vec is not None:
        feature_cols += [col for col in wav2vec.columns if col not in {"record_id", "n_views", "target_sample_rate"}]

    feature_df = df[["record_id", "label_binary_pathology", "label_multiclass", "class_group"] + feature_cols].copy()
    cat_cols = [col for col in CAT_META_COLUMNS if col in feature_df.columns]
    feature_matrix_df = pd.get_dummies(feature_df.drop(columns=["record_id", "label_binary_pathology", "label_multiclass", "class_group"]), columns=cat_cols, dummy_na=True)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
        ]
    )
    X = pipeline.fit_transform(feature_matrix_df)
    return X, feature_df


def align_clusters_to_labels(y_true: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    mapped = np.zeros_like(clusters)
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        labels, counts = np.unique(y_true[mask], return_counts=True)
        mapped[mask] = labels[np.argmax(counts)]
    return mapped


def run_clustering_suite(X: np.ndarray, y_true: np.ndarray, n_clusters: int) -> List[Dict[str, float]]:
    algorithms = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
        "gmm": GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    }
    rows: List[Dict[str, float]] = []
    for name, model in algorithms.items():
        if name == "gmm":
            clusters = model.fit_predict(X)
        else:
            clusters = model.fit_predict(X)
        metrics = clustering_metrics(X, y_true, clusters)
        row = {"algorithm": name, **metrics}
        if n_clusters == 2:
            aligned = align_clusters_to_labels(y_true, clusters)
            binary = binary_metrics(y_true, aligned, aligned.astype(float))
            row.update(
                {
                    "accuracy": binary["accuracy"],
                    "precision": binary["precision"],
                    "recall": binary["recall"],
                    "f1": binary["f1"],
                    "mcc": binary["mcc"],
                }
            )
        rows.append(row)
    return rows


def run_anomaly_task(X: np.ndarray, y_binary: np.ndarray) -> Dict[str, float]:
    model = IsolationForest(
        n_estimators=500,
        contamination=0.25,
        random_state=42,
    )
    model.fit(X)
    anomaly_score = -model.decision_function(X)
    threshold = np.quantile(anomaly_score, 0.75)
    y_pred = (anomaly_score >= threshold).astype(int)
    metrics = binary_metrics(y_binary, y_pred, anomaly_score)
    return metrics


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    paths = ensure_project_paths(args.project_root)
    metadata, handcrafted, wav2vec = load_tables(paths["root"], args)
    X, feature_df = prepare_matrix(metadata, handcrafted, wav2vec, args.feature_set)

    cluster_binary = run_clustering_suite(X, feature_df["label_binary_pathology"].to_numpy(dtype=int), n_clusters=2)
    cluster_multiclass = run_clustering_suite(X, feature_df["label_multiclass"].to_numpy(dtype=int), n_clusters=4)
    anomaly = run_anomaly_task(X, feature_df["label_binary_pathology"].to_numpy(dtype=int))

    summary = {
        "feature_set": args.feature_set,
        "cluster_binary": cluster_binary,
        "cluster_multiclass": cluster_multiclass,
        "anomaly_binary": anomaly,
    }

    out_dir = paths["unsupervised_dir"] / f"{args.feature_set}_{args.output_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dump(summary, out_dir / "metrics.json")
    feature_df.to_csv(out_dir / "feature_index.csv", index=False)

    print("Binary clustering:")
    print(pd.DataFrame(cluster_binary).sort_values(["ari", "nmi"], ascending=False).to_string(index=False))
    print("\nFour-class clustering:")
    print(pd.DataFrame(cluster_multiclass).sort_values(["ari", "nmi"], ascending=False).to_string(index=False))
    print("\nAnomaly task:")
    print(pd.Series(anomaly).to_string())
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(parse_args())
