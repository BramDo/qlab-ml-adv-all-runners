#!/usr/bin/env python3
"""Small real-data bridge from official Splice to the Qiskit QOS kernel port."""

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

import qiskit_official_qos_sampling_port as qport  # noqa: E402
import splice_utils  # noqa: E402


def largest_power_of_two_at_most(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def choose_feature_subset(
    x: np.ndarray,
    y: np.ndarray,
    bridge_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    class0 = x[y == 0]
    class1 = x[y == 1]
    diff = np.asarray(class1.mean(axis=0) - class0.mean(axis=0)).ravel()
    if bridge_dim == 1:
        chosen = np.array([int(np.argmax(np.abs(diff)))], dtype=np.int64)
        return chosen, diff[chosen], "top_abs"

    if bridge_dim % 2 != 0:
        raise ValueError("bridge_dim must be even for balanced signed selection")

    pos_idx = np.where(diff > 0)[0]
    neg_idx = np.where(diff < 0)[0]
    half = bridge_dim // 2

    if len(pos_idx) >= half and len(neg_idx) >= half:
        top_pos = pos_idx[np.argsort(diff[pos_idx])[::-1][:half]]
        top_neg = neg_idx[np.argsort(np.abs(diff[neg_idx]))[::-1][:half]]
        chosen = np.sort(np.concatenate([top_pos, top_neg]))
        return chosen, diff[chosen], "balanced_signed_top_abs"

    order = np.argsort(np.abs(diff))[::-1]
    chosen = np.sort(order[:bridge_dim])
    return chosen, diff[chosen], "top_abs_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge official Splice data into the small Qiskit QOS kernel port"
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--bridge-dim", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--general-degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_full, y, label_names = splice_utils.load_splice_data(binary=True, min_samples=1)
    x_filtered, kept = splice_utils.filter_features_by_frequency(x_full, args.min_samples)

    available_dim = x_filtered.shape[1]
    if available_dim <= 0:
        raise ValueError("Splice filtering removed all features")
    if args.bridge_dim > available_dim:
        raise ValueError(
            f"bridge_dim={args.bridge_dim} exceeds available filtered features={available_dim}"
        )
    if args.bridge_dim & (args.bridge_dim - 1):
        raise ValueError("bridge_dim must be a power of two")
    if args.num_samples % args.general_degree != 0:
        raise ValueError("num_samples must be divisible by general_degree")

    chosen_cols, mean_diff, selection_mode = choose_feature_subset(x_filtered, y, args.bridge_dim)
    selected_global_cols = kept[chosen_cols]

    sign_vector = np.where(mean_diff >= 0.0, 1.0, -1.0).astype(np.float64)
    truth_table = (mean_diff >= 0.0).astype(np.int32)

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
        "dataset": "Splice binary EI_vs_IE",
        "label_names": [str(x) for x in label_names],
        "min_samples": args.min_samples,
        "full_shape": [int(x_full.shape[0]), int(x_full.shape[1])],
        "filtered_shape": [int(x_filtered.shape[0]), int(x_filtered.shape[1])],
        "class_balance": {
            str(label_names[0]): int(np.sum(y == 0)),
            str(label_names[1]): int(np.sum(y == 1)),
        },
        "bridge_dim": args.bridge_dim,
        "selection_mode": selection_mode,
        "selected_positive_count": int(np.sum(mean_diff > 0)),
        "selected_negative_count": int(np.sum(mean_diff < 0)),
        "selected_filtered_feature_indices": chosen_cols.tolist(),
        "selected_global_feature_indices": selected_global_cols.tolist(),
        "mean_diff_vector": mean_diff.tolist(),
        "mean_diff_norm": float(np.linalg.norm(mean_diff)),
        "flat_sign_vector": sign_vector.tolist(),
        "boolean_truth_table": truth_table.tolist(),
        "boolean_kernel": qport.run_boolean_case_from_truth_table(
            truth_table,
            sampled_bool_idx,
            sampled_bool_vals,
        ),
        "flat_kernel": qport.run_flat_case_from_vector(
            sign_vector,
            sampled_flat_idx,
            sampled_flat_vals,
            args.shots,
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
