"""Command-line helper for fast TabPFN inference using the KV cache."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier


def _load_split(path: Path, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a CSV split and return feature matrix, target array and feature names."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in {path}. Available columns: {list(df.columns)}"
        )

    y = df.pop(target).to_numpy()
    feature_names = df.columns.tolist()
    return df.to_numpy(), y, feature_names


def _load_features(path: Path, feature_names: list[str], target: str | None) -> np.ndarray:
    df = pd.read_csv(path)

    if target and target in df.columns:
        df = df.drop(columns=target)

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(
            "Test split is missing feature columns required by the training split: "
            + ", ".join(sorted(missing))
        )

    # Reorder columns to match the training data order.
    return df[feature_names].to_numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit TabPFN with the KV cache enabled so that repeated inference runs "
            "are significantly faster."
        )
    )
    parser.add_argument("train", type=Path, help="Path to the training CSV file")
    parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column in the training CSV",
    )
    parser.add_argument(
        "--test",
        type=Path,
        help="Optional CSV file with the features to score. If omitted the training data is reused.",
    )
    parser.add_argument(
        "--predict-proba",
        action="store_true",
        help="Emit class probabilities instead of hard labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the predictions as CSV.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="Number of ensemble members to sample (default: 8).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference (default: auto).",
    )
    parser.add_argument(
        "--n-preprocessing-jobs",
        type=int,
        default=1,
        help="How many CPU jobs to use for preprocessing (default: 1).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat the inference call for benchmarking (default: 1).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    X_train, y_train, feature_names = _load_split(args.train, args.target)

    if args.test:
        X_test = _load_features(args.test, feature_names, args.target)
    else:
        X_test = X_train

    clf = TabPFNClassifier(
        fit_mode="fit_with_cache",
        n_estimators=args.n_estimators,
        device=args.device,
        n_preprocessing_jobs=args.n_preprocessing_jobs,
    )

    print("Fitting TabPFN with KV cache enabled...")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    print(f"Fit completed in {t_fit:.2f}s")

    predict_fn = clf.predict_proba if args.predict_proba else clf.predict

    preds = None
    for idx in range(args.repeat):
        t1 = time.perf_counter()
        preds = predict_fn(X_test)
        t_pred = time.perf_counter() - t1
        print(f"Inference run {idx + 1}/{args.repeat}: {t_pred:.4f}s")

    if args.output and preds is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.predict_proba:
            pred_df = pd.DataFrame(preds, columns=[f"class_{i}" for i in range(preds.shape[1])])
        else:
            pred_df = pd.DataFrame({"prediction": preds})
        pred_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
