from __future__ import annotations

from common_voiced_strong import build_metadata_frame, build_parser, ensure_project_paths


def main(project_root: str, data_dir: str) -> None:
    paths = ensure_project_paths(project_root)
    metadata = build_metadata_frame(data_dir)
    out_path = paths["processed_dir"] / "voiced_metadata_strong.csv"
    metadata.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Class distribution:")
    print(metadata["class_group"].value_counts().to_string())
    print("Binary distribution:")
    print(metadata["label_binary_pathology"].map({0: "healthy", 1: "pathological"}).value_counts().to_string())


if __name__ == "__main__":
    parser = build_parser("Build a clean VOICED metadata table for supervised and unsupervised tasks.")
    args = parser.parse_args()
    main(args.project_root, args.data_dir)
