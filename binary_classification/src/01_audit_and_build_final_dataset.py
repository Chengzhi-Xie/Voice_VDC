from __future__ import annotations

import argparse

import pandas as pd

from common_voiced_strong import json_dump, set_seed
from strict_balance_utils import ARTIFACTS_DIR, ensure_dirs, build_augmented_train_eval, load_base_real_splits


LEAKAGE_AUDIT = {
    "fixed_test_hard_leakage": "not_detected",
    "cv_augmentation_leakage_risk": "present_in_previous_versions",
    "cv_subgroup_fit_leakage_risk": "present_in_previous_versions",
    "test_set_selection_bias_risk": "present_if_best_model_is_chosen_after_viewing_fixed_test_results",
    "main_findings": [
        "Previous data_balance and data_balance_o pipelines generated synthetic healthy samples before CV using all real training subjects, so each validation fold indirectly influenced the augmentation distribution.",
        "Previous data_balance_o also fit subgroup and clinical KMeans once on all real training subjects before CV, so validation folds indirectly influenced subgroup assignments and engineered subgroup features.",
        "Those issues do not leak fixed test samples into training, but they make CV metrics optimistic and can bias model selection.",
        "The new data_balance_final pipeline removes that risk by fitting Gaussian augmentation and subgroup models inside each training fold only, while keeping the same fixed test untouched.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict final artifacts and record leakage audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acoustic_noise_scale", type=float, default=0.30)
    parser.add_argument("--clinical_noise_scale", type=float, default=0.20)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dirs()
    base = load_base_real_splits()
    built = build_augmented_train_eval(
        healthy_train_real=base["healthy_train_real"],
        disease_train_real=base["disease_train_real"],
        eval_df=base["fixed_test"],
        seed=args.seed,
        acoustic_noise_scale=args.acoustic_noise_scale,
        clinical_noise_scale=args.clinical_noise_scale,
    )

    train_df = built["train_df"]
    test_df = built["eval_df"]
    synth_df = built["synthetic_healthy"]
    meta = built["meta"]

    base["healthy_train_real"].to_csv(ARTIFACTS_DIR / "strict_healthy_train_real_37.csv", index=False)
    base["disease_train_real"].to_csv(ARTIFACTS_DIR / "strict_disease_train_real_131.csv", index=False)
    synth_df.to_csv(ARTIFACTS_DIR / "strict_synthetic_healthy.csv", index=False)
    train_df.to_csv(ARTIFACTS_DIR / "strict_augmented_train.csv", index=False)
    test_df.to_csv(ARTIFACTS_DIR / "strict_fixed_test_features.csv", index=False)

    summary = {
        "seed": args.seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_real_healthy": int(len(base["healthy_train_real"])),
        "n_real_disease": int(len(base["disease_train_real"])),
        "n_synthetic_healthy": int(len(synth_df)),
        "train_label_counts": train_df["label_binary_pathology"].value_counts().sort_index().to_dict(),
        "test_label_counts": test_df["label_binary_pathology"].value_counts().sort_index().to_dict(),
        "n_features": int(len([c for c in train_df.columns if c not in {"record_id", "class_group", "label_binary_pathology", "source"}])),
        **meta,
    }
    json_dump(summary, ARTIFACTS_DIR / "strict_dataset_summary.json")
    json_dump(LEAKAGE_AUDIT, ARTIFACTS_DIR / "leakage_audit.json")

    print(pd.Series(summary).to_string())
    print(f"Saved: {ARTIFACTS_DIR / 'strict_dataset_summary.json'}")
    print(f"Saved: {ARTIFACTS_DIR / 'leakage_audit.json'}")


if __name__ == "__main__":
    main(parse_args())
