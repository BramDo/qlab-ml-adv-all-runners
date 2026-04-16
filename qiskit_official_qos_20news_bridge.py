#!/usr/bin/env python3
"""Small real-data bridge from official 20 Newsgroups to the Qiskit QOS kernel port."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
OFFICIAL_REAL_DATASETS = ROOT / "official_qos" / "real_datasets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OFFICIAL_REAL_DATASETS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REAL_DATASETS))

import qiskit_official_qos_sampling_port as qport  # noqa: E402
from qiskit_official_qos_splice_bridge import choose_feature_subset  # noqa: E402


def require_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("bridge_dim must be a positive power of two")


def load_20news_binary(categories: list[str]) -> tuple[list[str], np.ndarray]:
    kwargs = {
        "categories": categories,
        "remove": ("headers", "footers", "quotes"),
        "return_X_y": True,
    }
    train_docs, y_train = fetch_20newsgroups(subset="train", **kwargs)
    test_docs, y_test = fetch_20newsgroups(subset="test", **kwargs)
    raw_documents = list(train_docs) + list(test_docs)
    y = np.concatenate([y_train, y_test]).astype(np.int32)
    return raw_documents, y


def compute_20news_space_metrics(x) -> dict[str, float | int]:
    shape = x.get_shape()
    num_samples = int(shape[0])
    feature_dim = int(shape[1])
    row_sparsity = int(x.getnnz(axis=1).max())
    col_sparsity = int(x.getnnz(axis=0).max())
    sparsity = max(row_sparsity, col_sparsity)
    return {
        "num_samples": num_samples,
        "feature_dim": feature_dim,
        "nnz": int(x.getnnz()),
        "row_sparsity_max": row_sparsity,
        "col_sparsity_max": col_sparsity,
        "sparsity": sparsity,
        "space_streaming": feature_dim,
        "space_sparse": int(x.getnnz()),
        "space_quantum": float(
            2 * np.ceil(np.log2(num_samples + 2 * feature_dim))
            + np.ceil(np.log2(sparsity + 1))
            + 4
        ),
    }


def load_curve_point(
    official_json: Path,
    min_df: int,
    source_metric_key: str,
    output_metric_key: str,
) -> dict[str, float | int] | None:
    if not official_json.exists():
        return None
    payload = json.loads(official_json.read_text(encoding="utf-8"))
    raw = payload.get("raw_data_by_min_df", {})
    point = raw.get(str(min_df))
    if point is None:
        return None

    return {
        "min_df": int(min_df),
        "n_pairs": int(payload.get("n_pairs", 0)),
        "streaming_space_mean": float(np.mean(point["streaming"]["space"])),
        "sparse_space_mean": float(np.mean(point["sparse"]["space"])),
        "quantum_space_mean": float(np.mean(point["quantum"]["space"])),
        output_metric_key: float(np.mean(point["streaming"][source_metric_key])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge official 20 Newsgroups data into the small Qiskit QOS kernel port"
    )
    parser.add_argument("--category-a", type=str, default="talk.politics.mideast")
    parser.add_argument("--category-b", type=str, default="sci.crypt")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--bridge-dim", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--general-degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=4096)
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
    require_power_of_two(args.bridge_dim)
    if args.num_samples % args.general_degree != 0:
        raise ValueError("num_samples must be divisible by general_degree")

    categories = [args.category_a, args.category_b]
    raw_documents, y = load_20news_binary(categories)

    vectorizer = TfidfVectorizer(min_df=args.min_df, stop_words="english")
    x_filtered = vectorizer.fit_transform(raw_documents)
    if x_filtered.shape[1] == 0:
        raise ValueError("20NG vectorization removed all features")
    if args.bridge_dim > x_filtered.shape[1]:
        raise ValueError(
            f"bridge_dim={args.bridge_dim} exceeds available filtered features={x_filtered.shape[1]}"
        )

    chosen_cols, mean_diff, selection_mode = choose_feature_subset(
        x_filtered,
        np.asarray(y),
        args.bridge_dim,
    )
    feature_names = vectorizer.get_feature_names_out()
    selected_feature_names = feature_names[chosen_cols]

    sign_vector = np.where(mean_diff >= 0.0, 1.0, -1.0).astype(np.float64)
    truth_table = (mean_diff >= 0.0).astype(np.int32)
    rng = np.random.default_rng(args.seed)

    sampled_bool_idx, sampled_bool_vals = qport.sample_from_truth_table(
        truth_table, args.num_samples, rng
    )
    sampled_flat_idx, sampled_flat_vals = qport.sample_from_vector(
        sign_vector, args.num_samples, rng
    )
    sampled_gen_idx, sampled_gen_vals = qport.sample_from_vector(
        mean_diff.astype(np.float64), args.num_samples, rng
    )

    payload = {
        "dataset": "20 Newsgroups binary pair",
        "label_names": categories,
        "category_pair": categories,
        "min_df": args.min_df,
        "full_shape": [int(len(raw_documents)), int(x_filtered.shape[1])],
        "class_balance": {
            str(categories[0]): int(np.sum(np.asarray(y) == 0)),
            str(categories[1]): int(np.sum(np.asarray(y) == 1)),
        },
        "paper_space_metrics_pair_specific": compute_20news_space_metrics(x_filtered),
        "official_curve_reference_at_min_df_accuracy": load_curve_point(
            args.official_accuracy_json, args.min_df, "accuracy", "accuracy_mean"
        ),
        "official_curve_reference_at_min_df_variance": load_curve_point(
            args.official_variance_json,
            args.min_df,
            "variance_recovery",
            "variance_recovery",
        ),
        "bridge_dim": args.bridge_dim,
        "selection_mode": selection_mode,
        "selected_positive_count": int(np.sum(mean_diff > 0)),
        "selected_negative_count": int(np.sum(mean_diff < 0)),
        "selected_feature_indices": chosen_cols.tolist(),
        "selected_feature_names": selected_feature_names.tolist(),
        "mean_diff_vector": mean_diff.tolist(),
        "mean_diff_norm": float(np.linalg.norm(mean_diff)),
        "flat_sign_vector": sign_vector.tolist(),
        "boolean_truth_table": truth_table.tolist(),
        "boolean_kernel": qport.run_boolean_case_from_truth_table(
            truth_table, sampled_bool_idx, sampled_bool_vals
        ),
        "flat_kernel": qport.run_flat_case_from_vector(
            sign_vector, sampled_flat_idx, sampled_flat_vals, args.shots
        ),
        "general_state_kernel": qport.run_general_state_case_from_vector(
            mean_diff.astype(np.float64),
            sampled_gen_idx,
            sampled_gen_vals,
            args.seed + 7,
            args.general_degree,
        ),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
