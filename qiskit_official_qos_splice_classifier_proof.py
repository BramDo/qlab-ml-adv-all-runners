#!/usr/bin/env python3
"""Small classifier proof on top of the validated official Splice Qiskit bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent
OFFICIAL_REAL_DATASETS = ROOT / "official_qos" / "real_datasets"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OFFICIAL_REAL_DATASETS) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REAL_DATASETS))

import qiskit_official_qos_sampling_port as qport  # noqa: E402
from qiskit_official_qos_splice_bridge import choose_feature_subset  # noqa: E402
import splice_utils  # noqa: E402


def largest_power_of_two_at_most(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def filter_by_train_frequency(x_train, x_test, min_samples: int):
    feature_counts = np.asarray((x_train != 0).sum(axis=0)).ravel()
    kept = np.where(feature_counts >= min_samples)[0]
    return x_train[:, kept], x_test[:, kept], kept


def stratified_binary_split(y: np.ndarray, n_train: int, n_test: int, rng: np.random.Generator):
    if n_train % 2 or n_test % 2:
        raise ValueError("n_train and n_test must be even for balanced binary split")
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    half_train = n_train // 2
    half_test = n_test // 2
    if len(idx0) < half_train + half_test or len(idx1) < half_train + half_test:
        raise ValueError("not enough samples per class for requested split")
    train_idx = np.sort(np.concatenate([idx0[:half_train], idx1[:half_train]]))
    test_idx = np.sort(
        np.concatenate(
            [
                idx0[half_train : half_train + half_test],
                idx1[half_train : half_train + half_test],
            ]
        )
    )
    return train_idx, test_idx


def make_quantum_features(
    x_dense: np.ndarray,
    *,
    num_samples: int,
    degree: int,
    seed: int,
    include_jax_parity: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    features = []
    raw_l2_errors = []
    max_abs_errors = []
    zero_rows = 0

    for row_idx, row in enumerate(x_dense):
        if np.linalg.norm(row) <= 1e-12:
            zero_rows += 1
        rng = np.random.default_rng(seed + row_idx)
        sampled_idx, sampled_vals = qport.sample_from_vector(row.astype(np.float64), num_samples, rng)
        details = qport.general_state_sketch_from_vector_samples(
            row.astype(np.float64),
            sampled_idx,
            sampled_vals,
            seed + 10_000 + row_idx,
            degree,
            include_jax=include_jax_parity,
        )
        q_state = np.asarray(details["qiskit_state"], dtype=np.float64)
        features.append(q_state)
        if include_jax_parity and details["jax_state"] is not None:
            j_state = np.asarray(details["jax_state"], dtype=np.float64)
            raw_l2_errors.append(float(np.linalg.norm(q_state - j_state)))
            max_abs_errors.append(float(np.max(np.abs(q_state - j_state))))

    feature_matrix = np.asarray(features, dtype=np.float64)
    meta = {
        "num_rows": int(len(x_dense)),
        "zero_rows": int(zero_rows),
        "include_jax_parity": bool(include_jax_parity),
    }
    if raw_l2_errors:
        meta.update(
            {
                "mean_raw_l2_err_vs_jax": float(np.mean(raw_l2_errors)),
                "max_raw_l2_err_vs_jax": float(np.max(raw_l2_errors)),
                "mean_max_abs_err_vs_jax": float(np.mean(max_abs_errors)),
                "max_max_abs_err_vs_jax": float(np.max(max_abs_errors)),
            }
        )
    return feature_matrix, meta


def transform_feature_view(features: np.ndarray, view: str) -> np.ndarray:
    if view == "raw":
        return features
    if view == "abs":
        return np.abs(features)
    if view == "sq":
        return features**2
    if view == "raw_abs":
        return np.concatenate([features, np.abs(features)], axis=1)
    if view == "raw_sq":
        return np.concatenate([features, features**2], axis=1)
    if view == "all":
        return np.concatenate([features, np.abs(features), features**2], axis=1)
    raise ValueError(f"unsupported feature view: {view}")


def cosine_prototype_accuracy(train_x, train_y, test_x, test_y) -> float:
    train_n = normalize(train_x, norm="l2")
    test_n = normalize(test_x, norm="l2")
    proto0 = train_n[train_y == 0].mean(axis=0, keepdims=True)
    proto1 = train_n[train_y == 1].mean(axis=0, keepdims=True)
    prototypes = normalize(np.vstack([proto0, proto1]), norm="l2")
    pred = np.argmax(test_n @ prototypes.T, axis=1)
    return float(accuracy_score(test_y, pred))


def corr_prototype_accuracy(train_x, train_y, test_x, test_y) -> float:
    train_c = train_x - train_x.mean(axis=1, keepdims=True)
    test_c = test_x - test_x.mean(axis=1, keepdims=True)
    train_n = normalize(train_c, norm="l2")
    test_n = normalize(test_c, norm="l2")
    proto0 = train_n[train_y == 0].mean(axis=0, keepdims=True)
    proto1 = train_n[train_y == 1].mean(axis=0, keepdims=True)
    prototypes = normalize(np.vstack([proto0, proto1]), norm="l2")
    pred = np.argmax(test_n @ prototypes.T, axis=1)
    return float(accuracy_score(test_y, pred))


def fit_and_score(train_x, train_y, test_x, test_y):
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    test_z = scaler.transform(test_x)

    ridge = RidgeClassifier(random_state=42)
    ridge.fit(train_z, train_y)
    ridge_acc = float(accuracy_score(test_y, ridge.predict(test_z)))

    logistic = LogisticRegression(random_state=42, max_iter=5000)
    logistic.fit(train_z, train_y)
    log_acc = float(accuracy_score(test_y, logistic.predict(test_z)))

    linearsvc = LinearSVC(random_state=42, max_iter=10_000)
    linearsvc.fit(train_z, train_y)
    linearsvc_acc = float(accuracy_score(test_y, linearsvc.predict(test_z)))

    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(train_z, train_y)
    knn3_acc = float(accuracy_score(test_y, knn3.predict(test_z)))

    centroid = NearestCentroid()
    centroid.fit(train_z, train_y)
    centroid_acc = float(accuracy_score(test_y, centroid.predict(test_z)))

    cosine_proto_acc = cosine_prototype_accuracy(train_x, train_y, test_x, test_y)
    corr_proto_acc = corr_prototype_accuracy(train_x, train_y, test_x, test_y)

    return {
        "ridge_accuracy": ridge_acc,
        "logistic_accuracy": log_acc,
        "linearsvc_accuracy": linearsvc_acc,
        "knn3_accuracy": knn3_acc,
        "centroid_accuracy": centroid_acc,
        "cosine_proto_accuracy": cosine_proto_acc,
        "corr_proto_accuracy": corr_proto_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small classifier proof on top of the official Splice Qiskit bridge"
    )
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--bridge-dim", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--general-degree", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantum-feature-view",
        choices=["raw", "abs", "sq", "raw_abs", "raw_sq", "all"],
        default="raw",
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

    x_full, y, label_names = splice_utils.load_splice_data(binary=True, min_samples=1)
    train_idx, test_idx = stratified_binary_split(y, args.n_train, args.n_test, rng)

    x_train_full = x_full[train_idx]
    x_test_full = x_full[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_train_f, x_test_f, kept = filter_by_train_frequency(x_train_full, x_test_full, args.min_samples)
    available_dim = int(x_train_f.shape[1])
    effective_bridge_dim = min(args.bridge_dim, largest_power_of_two_at_most(available_dim))
    if effective_bridge_dim < 2:
        raise ValueError(
            f"train-only filtered feature space is too small for a useful bridge: available_dim={available_dim}"
        )

    chosen_cols, mean_diff, selection_mode = choose_feature_subset(x_train_f, y_train, effective_bridge_dim)
    selected_global_cols = kept[chosen_cols]

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
        "dataset": "Splice binary EI_vs_IE",
        "label_names": [str(x) for x in label_names],
        "min_samples": args.min_samples,
        "requested_bridge_dim": args.bridge_dim,
        "effective_bridge_dim": effective_bridge_dim,
        "num_samples": args.num_samples,
        "general_degree": args.general_degree,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "quantum_feature_view": args.quantum_feature_view,
        "selection_mode": selection_mode,
        "selected_positive_count": int(np.sum(mean_diff > 0)),
        "selected_negative_count": int(np.sum(mean_diff < 0)),
        "selected_filtered_feature_indices": chosen_cols.tolist(),
        "selected_global_feature_indices": selected_global_cols.tolist(),
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
