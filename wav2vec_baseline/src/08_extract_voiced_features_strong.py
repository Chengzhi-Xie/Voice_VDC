from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from common_voiced_strong import (
    build_metadata_frame,
    build_parser,
    ensure_project_paths,
    extract_handcrafted_features,
    normalize_waveform,
    read_signal,
    resample_waveform,
    set_seed,
)


@dataclass
class Wav2VecConfig:
    model_name: str
    batch_size: int
    target_sr: int
    local_files_only: bool
    window_sec: float
    hop_sec: float
    device: str


def segment_waveform(y: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> List[np.ndarray]:
    if window_sec <= 0:
        return [y.astype(np.float32)]
    window = int(window_sec * sr)
    hop = int(max(1, hop_sec * sr))
    if window <= 0 or len(y) <= window:
        return [y.astype(np.float32)]
    segments: List[np.ndarray] = []
    for start in range(0, max(1, len(y) - window + 1), hop):
        end = start + window
        if end > len(y):
            break
        segments.append(y[start:end].astype(np.float32))
    if not segments:
        segments.append(y.astype(np.float32))
    return segments


def pooled_embedding(hidden: torch.Tensor) -> torch.Tensor:
    mean_pool = hidden.mean(dim=1)
    std_pool = hidden.std(dim=1)
    max_pool = hidden.amax(dim=1)
    return torch.cat([mean_pool, std_pool, max_pool], dim=1)


class Wav2VecExtractor:
    def __init__(self, config: Wav2VecConfig) -> None:
        use_cuda = config.device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.model = Wav2Vec2Model.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_views(self, views: Sequence[np.ndarray]) -> np.ndarray:
        pooled_batches: List[np.ndarray] = []
        for start in range(0, len(views), self.config.batch_size):
            batch_views = views[start : start + self.config.batch_size]
            batch = self.feature_extractor(
                batch_views,
                sampling_rate=self.config.target_sr,
                padding=True,
                return_tensors="pt",
            )
            input_values = batch["input_values"].to(self.device)
            hidden = self.model(input_values=input_values).last_hidden_state
            pooled = pooled_embedding(hidden)
            pooled_batches.append(pooled.float().cpu().numpy())
        return np.concatenate(pooled_batches, axis=0)


def extract_handcrafted_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="handcrafted_strong"):
        y, sr = read_signal(row["wfdb_record_path"])
        features = extract_handcrafted_features(y, sr)
        features["record_id"] = row["record_id"]
        features["sample_rate"] = sr
        rows.append(features)
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def extract_wav2vec_table(metadata: pd.DataFrame, config: Wav2VecConfig) -> pd.DataFrame:
    extractor = Wav2VecExtractor(config)
    rows: List[Dict[str, float]] = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="wav2vec_strong"):
        y, sr = read_signal(row["wfdb_record_path"])
        y = normalize_waveform(resample_waveform(y, sr, config.target_sr))
        segments = segment_waveform(y, config.target_sr, config.window_sec, config.hop_sec)
        segment_embeddings = extractor.encode_views(segments)
        emb_mean = segment_embeddings.mean(axis=0)
        emb_std = segment_embeddings.std(axis=0) if len(segment_embeddings) > 1 else np.zeros_like(emb_mean)

        feature_row: Dict[str, float] = {
            "record_id": row["record_id"],
            "n_views": int(len(segments)),
            "target_sample_rate": int(config.target_sr),
        }
        for idx, value in enumerate(emb_mean):
            feature_row[f"w2v_viewmean_{idx}"] = float(value)
        for idx, value in enumerate(emb_std):
            feature_row[f"w2v_viewstd_{idx}"] = float(value)
        rows.append(feature_row)
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = build_parser("Extract clean handcrafted and optional wav2vec2 features from VOICED.")
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--skip_handcrafted", action="store_true")
    parser.add_argument("--use_wav2vec", action="store_true")
    parser.add_argument("--wav2vec_model", type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument("--wav2vec_batch_size", type=int, default=4)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--window_sec", type=float, default=0.0)
    parser.add_argument("--hop_sec", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    paths = ensure_project_paths(args.project_root)
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv)
    else:
        metadata = build_metadata_frame(args.data_dir)
    metadata = metadata.sort_values("record_id").reset_index(drop=True)
    metadata_out = paths["processed_dir"] / "voiced_metadata_strong.csv"
    metadata.to_csv(metadata_out, index=False)
    print(f"Saved metadata: {metadata_out}")

    if not args.skip_handcrafted:
        handcrafted = extract_handcrafted_table(metadata)
        handcrafted_out = paths["processed_dir"] / "voiced_handcrafted_features_strong.csv"
        handcrafted.to_csv(handcrafted_out, index=False)
        print(f"Saved handcrafted features: {handcrafted_out} {handcrafted.shape}")

    if args.use_wav2vec:
        config = Wav2VecConfig(
            model_name=args.wav2vec_model,
            batch_size=args.wav2vec_batch_size,
            target_sr=args.target_sr,
            local_files_only=args.local_files_only,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            device=args.device,
        )
        wav2vec = extract_wav2vec_table(metadata, config)
        model_tag = args.wav2vec_model.split("/")[-1].replace("-", "_")
        suffix = "wholeclip" if args.window_sec <= 0 else f"win{str(args.window_sec).replace('.', 'p')}"
        wav2vec_out = paths["processed_dir"] / f"voiced_{model_tag}_{suffix}_features_strong.csv"
        wav2vec.to_csv(wav2vec_out, index=False)
        print(f"Saved wav2vec features: {wav2vec_out} {wav2vec.shape}")


if __name__ == "__main__":
    main(parse_args())
