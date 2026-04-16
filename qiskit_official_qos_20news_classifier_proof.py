#!/usr/bin/env python3
"""Bounded classifier proof on top of the official 20NG Qiskit bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
OFFICIAL_REAL_DATASETS = ROOT / "official_qos" / "real_datasets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OFFICIAL_REAL_DATASETS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REAL_DATASETS))

from qiskit_official_qos_20news_bridge import load_20news_binary, load_curve_point  # noqa: E402
from qiskit_official_qos_splice_bridge import choose_feature_subset  # noqa: E402
from qiskit_official_qos_splice_classifier_proof import (  # noqa: E402
    fit_and_score,
    largest_power_of_two_at_most,
    make_quantum_features,
    stratified_binary_split,
    transform_feature_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded classifier proof on top of the official 20NG Qiskit bridge"
    )
    parser.add_argument("--category-a", type=str, default="talk.politics.mideast")
    parser.add_argument("--category-b", type=str, default="sci.crypt")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--bridge-dim", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--general-degree", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantum-feature-view",
        choices=["raw", "abs", "sq", "raw_abs", "raw_sq", "all"],
        default="abs",
    )
    parser.add_argument(
        "--official-accuracy-json",
        type=Path,
        default=OFFICIAL_REAL_DATASETS / "20newsgroups_size_vs_accuracy.json",
    )
    parser.add_argument(
        "--official-variance-json",
        type=Path,
        default=OFFICIAL_REAL_DATASETS / "20newsgroups_size_vs_variance.json",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bridge_dim & (args.bridge_dim - 1):
        raise ValueError("bridge_dim must be a power of two")
    if args.num_samples % args.general_degree != 0:
        raise ValueError("num_samples must be divisible by general_degree")

    rng = np.random.default_rng(args.seed)

    categories = [args.category_a, args.category_b]
    raw_documents, y = load_20news_binary(categories)
    y = np.asarray(y)

    train_idx, test_idx = stratified_binary_split(y, args.n_train, args.n_test, rng)
    train_docs = [raw_documents[int(i)] for i in train_idx]
    test_docs = [raw_documents[int(i)] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    vectorizer = TfidfVectorizer(min_df=args.min_df, stop_words="english")
    x_train_f = vectorizer.fit_transform(train_docs)
    x_test_f = vectorizer.transform(test_docs)

    available_dim = int(x_train_f.shape[1])
    effective_bridge_dim = min(args.bridge_dim, largest_power_of_two_at_most(available_dim))
    if effective_bridge_dim < 2:
        raise ValueError(
            f"train-only filtered feature space is too small for a useful bridge: available_dim={available_dim}"
        )

    chosen_cols, mean_diff, selection_mode = choose_feature_subset(
        x_train_f, y_train, effective_bridge_dim
    )
    feature_names = vectorizer.get_feature_names_out()
    selected_feature_names = feature_names[chosen_cols]

    x_train_sel = np.asarray(x_train_f[:, chosen_cols].toarray(), dtype=np.float64)
    x_test_sel = np.asarray(x_test_f[:, chosen_cols].toarray(), dtype=np.float64)

    q_train, q_train_meta = make_quantum_features(
        x_train_sel,
        num_samples=args.num_samples,
        degree=args.general_degree,
        seed=args.seed + 1000,
    )
    q_test, q_test_meta = make_quantum_features(
        x_test_sel,
        num_samples=args.num_samples,
        degree=args.general_degree,
        seed=args.seed + 2000,
    )
    q_train_view = transform_feature_view(q_train, args.quantum_feature_view)
    q_test_view = transform_feature_view(q_test, args.quantum_feature_view)

    raw_metrics = fit_and_score(x_train_sel, y_train, x_test_sel, y_test)
    quantum_metrics = fit_and_score(q_train_view, y_train, q_test_view, y_test)

    payload = {
        "dataset": "20 Newsgroups binary pair",
        "label_names": categories,
        "category_pair": categories,
        "min_df": args.min_df,
        "requested_bridge_dim": args.bridge_dim,
        "effective_bridge_dim": effective_bridge_dim,
        "num_samples": args.num_samples,
        "general_degree": args.general_degree,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "quantum_feature_view": args.quantum_feature_view,
        "class_balance_full": {
            str(categories[0]): int(np.sum(y == 0)),
            str(categories[1]): int(np.sum(y == 1)),
        },
        "class_balance_train": {
            str(categories[0]): int(np.sum(y_train == 0)),
            str(categories[1]): int(np.sum(y_train == 1)),
        },
        "class_balance_test": {
            str(categories[0]): int(np.sum(y_test == 0)),
            str(categories[1]): int(np.sum(y_test == 1)),
        },
        "train_vectorized_shape": [int(x_train_f.shape[0]), int(x_train_f.shape[1])],
        "test_vectorized_shape": [int(x_test_f.shape[0]), int(x_test_f.shape[1])],
        "selection_mode": selection_mode,
        "selected_positive_count": int(np.sum(mean_diff > 0)),
        "selected_negative_count": int(np.sum(mean_diff < 0)),
        "selected_feature_indices": chosen_cols.tolist(),
        "selected_feature_names": selected_feature_names.tolist(),
        "mean_diff_vector": mean_diff.tolist(),
        "mean_diff_norm": float(np.linalg.norm(mean_diff)),
        "train_shape_selected": [int(x_train_sel.shape[0]), int(x_train_sel.shape[1])],
        "test_shape_selected": [int(x_test_sel.shape[0]), int(x_test_sel.shape[1])],
        "quantum_feature_train_shape": [int(q_train_view.shape[0]), int(q_train_view.shape[1])],
        "quantum_feature_test_shape": [int(q_test_view.shape[0]), int(q_test_view.shape[1])],
        "official_curve_reference_at_min_df_accuracy": load_curve_point(
            args.official_accuracy_json, args.min_df, "accuracy", "accuracy_mean"
        ),
        "official_curve_reference_at_min_df_variance": load_curve_point(
            args.official_variance_json,
            args.min_df,
            "variance_recovery",
            "variance_recovery",
        ),
        "raw_baseline": raw_metrics,
        "quantum_feature_classifier": quantum_metrics,
        "quantum_feature_train_meta": q_train_meta,
        "quantum_feature_test_meta": q_test_meta,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
