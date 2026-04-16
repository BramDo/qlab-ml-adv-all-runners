#!/usr/bin/env python3
"""Bounded classifier proof on top of the official PBMC68k Qiskit bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OFFICIAL_REAL_DATASETS = ROOT / "official_qos" / "real_datasets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OFFICIAL_REAL_DATASETS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REAL_DATASETS))

import pbmc68k_utils  # noqa: E402
from qiskit_official_qos_pbmc_bridge import (  # noqa: E402
    compute_pbmc_space_metrics,
    load_official_curve_point,
)
from qiskit_official_qos_splice_bridge import choose_feature_subset  # noqa: E402
from qiskit_official_qos_splice_classifier_proof import (  # noqa: E402
    filter_by_train_frequency,
    fit_and_score,
    largest_power_of_two_at_most,
    make_quantum_features,
    stratified_binary_split,
    transform_feature_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded classifier proof on top of the official PBMC68k Qiskit bridge"
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--bridge-dim", type=int, default=32)
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
        "--official-json",
        type=Path,
        default=OFFICIAL_REAL_DATASETS / "pbmc68k_size_vs_accuracy.json",
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

    x_full, y, label_names = pbmc68k_utils.load_pbmc68k_data(
        min_samples=1,
        normalize=True,
        binary=True,
    )
    x_full_official, kept_official = pbmc68k_utils.filter_genes_by_frequency(x_full, args.min_samples)

    train_idx, test_idx = stratified_binary_split(y, args.n_train, args.n_test, rng)
    x_train_full = x_full[train_idx]
    x_test_full = x_full[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_train_f, x_test_f, kept_train = filter_by_train_frequency(
        x_train_full, x_test_full, args.min_samples
    )
    available_dim = int(x_train_f.shape[1])
    effective_bridge_dim = min(args.bridge_dim, largest_power_of_two_at_most(available_dim))
    if effective_bridge_dim < 2:
        raise ValueError(
            f"train-only filtered feature space is too small for a useful bridge: available_dim={available_dim}"
        )

    chosen_cols, mean_diff, selection_mode = choose_feature_subset(
        x_train_f, y_train, effective_bridge_dim
    )
    selected_global_cols_train = kept_train[chosen_cols]

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
        "dataset": "PBMC68k binary top2 classes",
        "label_names": [str(x) for x in label_names],
        "min_samples": args.min_samples,
        "requested_bridge_dim": args.bridge_dim,
        "effective_bridge_dim": effective_bridge_dim,
        "num_samples": args.num_samples,
        "general_degree": args.general_degree,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "quantum_feature_view": args.quantum_feature_view,
        "full_shape": [int(x_full.shape[0]), int(x_full.shape[1])],
        "official_filtered_shape": [int(x_full_official.shape[0]), int(x_full_official.shape[1])],
        "train_filtered_shape": [int(x_train_f.shape[0]), int(x_train_f.shape[1])],
        "test_filtered_shape": [int(x_test_f.shape[0]), int(x_test_f.shape[1])],
        "class_balance_full": {
            str(label_names[0]): int(np.sum(y == 0)),
            str(label_names[1]): int(np.sum(y == 1)),
        },
        "class_balance_train": {
            str(label_names[0]): int(np.sum(y_train == 0)),
            str(label_names[1]): int(np.sum(y_train == 1)),
        },
        "class_balance_test": {
            str(label_names[0]): int(np.sum(y_test == 0)),
            str(label_names[1]): int(np.sum(y_test == 1)),
        },
        "paper_space_metrics_pair_specific": compute_pbmc_space_metrics(x_full_official),
        "official_curve_reference_at_min_samples": load_official_curve_point(
            args.official_json, args.min_samples
        ),
        "selection_mode": selection_mode,
        "selected_positive_count": int(np.sum(mean_diff > 0)),
        "selected_negative_count": int(np.sum(mean_diff < 0)),
        "selected_train_filtered_gene_indices": chosen_cols.tolist(),
        "selected_global_gene_indices_train_filter": selected_global_cols_train.tolist(),
        "selected_global_gene_indices_official_overlap": np.intersect1d(
            kept_official, selected_global_cols_train
        ).astype(int).tolist(),
        "mean_diff_vector": mean_diff.tolist(),
        "train_shape_selected": [int(x_train_sel.shape[0]), int(x_train_sel.shape[1])],
        "test_shape_selected": [int(x_test_sel.shape[0]), int(x_test_sel.shape[1])],
        "quantum_feature_train_shape": [int(q_train_view.shape[0]), int(q_train_view.shape[1])],
        "quantum_feature_test_shape": [int(q_test_view.shape[0]), int(q_test_view.shape[1])],
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
