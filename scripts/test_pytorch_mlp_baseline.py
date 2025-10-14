#!/usr/bin/env python3
"""
Quick smoke test for the PyTorch MLP baseline on a small data subset.

Usage:
    python scripts/test_pytorch_mlp_baseline.py \
        --data_path Dataset_and_train_sequence/antimicrobial_training_data_expanded.csv \
        --sample_size 2000 \
        --seed 42

The script reuses the preprocessing and PyTorch MLP from
`ablation_study/run_generic_baselines.py`, but limits training to a
manageable subset and a handful of epochs so we can confirm the model
trains end-to-end without the full baseline runtime.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for PyTorch MLP baseline.")
    parser.add_argument(
        "--data_path",
        required=True,
        type=Path,
        help="Path to CSV/JSON dataset (same format as generic baselines).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Max number of rows to sample for the quick training run.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=2000,
        help="Cap TF-IDF vocabulary size per text column to keep tensors small.",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF (lower for tiny samples).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and train/val split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Avoid TensorFlow import issues when we bring in the baseline helpers.
    os.environ.setdefault("BASELINES_DISABLE_TF", "1")

    # Ensure repository root is on sys.path so we can import ablation_study.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Lazy import after env var is set.
    from ablation_study.run_generic_baselines import (  # pylint: disable=import-outside-toplevel
        PyTorchMLPClassifier,
        build_preprocessor,
        detect_columns,
        load_df,
    )

    df = load_df(str(args.data_path))
    print(f"[mlp_smoke] Loaded {len(df)} records from {args.data_path}")

    num_cols, cat_cols, text_cols, label_col = detect_columns(
        df, ["label", "antimicrobial_activity", "y", "target", "class"]
    )
    sample_n = min(args.sample_size, len(df))
    df_sample = df.sample(n=sample_n, random_state=args.seed)
    y = df_sample[label_col].astype(int).to_numpy()
    X = df_sample.drop(columns=[label_col])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(
        f"[mlp_smoke] Sample size: {sample_n} "
        f"(train={len(y_train)}, val={len(y_val)})"
    )

    preprocessor = build_preprocessor(num_cols, cat_cols, text_cols)
    # Limit TF-IDF dimensionality for the smoke test to keep tensors small.
    tfidf_params = {}
    max_features = max(1, args.max_features)
    min_df = max(1, args.min_df)
    for name, _, _ in preprocessor.transformers:
        if name.startswith("tfidf_"):
            tfidf_params[f"{name}__max_features"] = max_features
            tfidf_params[f"{name}__min_df"] = min_df
            tfidf_params[f"{name}__dtype"] = np.float32
    if tfidf_params:
        preprocessor.set_params(**tfidf_params)

    print("[mlp_smoke] Fitting preprocessing pipeline...")
    preprocessor.fit(X_train, y_train)

    def to_dense_float32(matrix):
        if hasattr(matrix, "toarray"):
            return matrix.toarray().astype(np.float32)
        return np.asarray(matrix, dtype=np.float32)

    X_train_proc = to_dense_float32(preprocessor.transform(X_train))
    X_val_proc = to_dense_float32(preprocessor.transform(X_val))
    print(f"[mlp_smoke] Feature dimension after preprocessing: {X_train_proc.shape[1]}")

    clf = PyTorchMLPClassifier(
        input_dim=X_train_proc.shape[1],
        epochs=5,
        batch_size=64,
        lr=0.001,
        hidden_dims=[256, 128],
        seed=args.seed,
    )

    print("[mlp_smoke] Training PyTorch MLP on subset...")
    clf.fit(X_train_proc, y_train)

    print("[mlp_smoke] Evaluating on validation split...")
    scores = clf.predict_proba(X_val_proc)[:, 1]
    preds = (scores >= 0.5).astype(int)

    print("[mlp_smoke] Classification report:")
    print(classification_report(y_val, preds, digits=4))
    try:
        auc = roc_auc_score(y_val, scores)
    except ValueError:
        auc = float("nan")
    print(f"[mlp_smoke] ROC-AUC: {auc:.4f}")
    print("[mlp_smoke] Done.")


if __name__ == "__main__":
    main()
