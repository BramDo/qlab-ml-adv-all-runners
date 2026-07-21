#!/usr/bin/env python3
"""Five-split real-data gate for the official flat QOS sampling kernel.

The local gate is deliberately provider-free.  It evaluates the exact ideal
distribution produced by the same ``q_state_sketch_flat`` interference circuit
used by ``qiskit_official_qos_flat_fireopal_pilot.py`` on two real-data routes:

* GSE132080 cells with the repository's fixed residualised semi-synthetic task;
* PBMC68k cells with their natural T-reg versus memory-T labels.

Small-width IBM-hardware eligibility is granted per dataset only when a frozen
ideal QOS width beats the complete tested classical frontier on at least four
of five larger splits and has positive mean balanced-accuracy delta.  Fire Opal
is reserved for a later q40/q60 route.  This module never authenticates,
validates, or submits to any provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import qiskit_official_qos_sampling_port as qos_port
import qiskit_qos_gse132080_semisynth_utils as semisynth
import qiskit_qos_gse132080_thirdorder_screen as thirdorder
import qiskit_qos_pbmc68k_pairwise_screen as pairwise
import qiskit_qos_pbmc68k_utils as pbmc


SCHEMA_VERSION = "1.0"
KIND = "official_flat_qos_realdata_five_split_gate"
DATASET_GSE = "gse132080-semisynth"
DATASET_PBMC = "pbmc68k"
SUPPORTED_DATASETS = (DATASET_GSE, DATASET_PBMC)
DEFAULT_SPLIT_SEEDS = (11, 17, 23, 29, 37)
DEFAULT_CLASSICAL_DIMS = (16, 64, 256, 1024, 4096, 16384, 65536)
DEFAULT_DIMENSIONS = (16, 64, 256, 1024)
DEFAULT_DIMENSION = 16
DEFAULT_SAMPLES = 64
DEFAULT_SAMPLES_PER_DIMENSION = 4
DEFAULT_SKETCH_SEED = 6_604_096
DEFAULT_OUTPUT = Path(
    "realdata_flat_qos_gate/official_flat_qos_q4_q6_q8_q10_five_split_gate.json"
)
GATE_MIN_WINS = 4


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    x: Any
    y: np.ndarray
    row_ids: np.ndarray
    source: dict[str, Any]
    task: dict[str, Any]
    build_interactions: Callable[..., tuple[np.ndarray, dict[str, Any]]]
    interaction_kwargs: dict[str, Any]
    train_size: int
    test_size: int
    ambient_feature_dim: int


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_datasets(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SUPPORTED_DATASETS
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(parsed) - set(SUPPORTED_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported datasets: {', '.join(unknown)}")
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one dataset")
    return parsed


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}|".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_sparse(value: Any) -> str:
    matrix = sparse.csr_matrix(value)
    digest = hashlib.sha256()
    digest.update(f"csr|{matrix.shape}|{matrix.dtype.str}|".encode("utf-8"))
    for array in (matrix.indptr, matrix.indices, matrix.data):
        contiguous = np.ascontiguousarray(array)
        digest.update(f"{contiguous.dtype.str}|{contiguous.shape}|".encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if amount < 1024.0 or unit == "PB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _model_bytes(model: Any, fallback: int = 0) -> int:
    total = 0
    for name in (
        "coef_",
        "intercept_",
        "support_",
        "support_vectors_",
        "dual_coef_",
        "classes_",
        "class_count_",
        "class_log_prior_",
        "feature_count_",
        "feature_log_prob_",
        "_fit_X",
        "_y",
    ):
        value = getattr(model, name, None)
        if value is not None:
            total += int(np.asarray(value).nbytes)
    for estimator in getattr(model, "estimators_", []):
        tree = estimator.tree_
        for name in (
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "value",
        ):
            total += int(np.asarray(getattr(tree, name)).nbytes)
    return max(int(total), int(fallback))


def _candidate_result(
    *,
    name: str,
    family: str,
    model: Any,
    predictions: np.ndarray,
    y_test: np.ndarray,
    feature_dim: int,
    fallback_bytes: int,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "name": name,
        "family": family,
        "feature_dim": int(feature_dim),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "model_bytes": _model_bytes(model, fallback_bytes),
        "predictions": [int(value) for value in predictions],
    }


def _linear_candidates(
    train: Any,
    test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    prefix: str,
    feature_dim: int,
    seed: int,
    include_nb: bool,
    include_nonlinear: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_linear = (int(feature_dim) + 1) * 8

    ridge = Ridge(alpha=1.0)
    ridge.fit(train, y_train)
    train_scores = np.asarray(ridge.predict(train), dtype=np.float64)
    threshold = 0.5 * (
        float(np.mean(train_scores[y_train > 0]))
        + float(np.mean(train_scores[y_train < 0]))
    )
    ridge_predictions = np.where(np.asarray(ridge.predict(test)) >= threshold, 1, -1)
    rows.append(
        _candidate_result(
            name=f"{prefix}_ridge_d{feature_dim}",
            family=f"{prefix}_ridge",
            model=ridge,
            predictions=ridge_predictions,
            y_test=y_test,
            feature_dim=feature_dim,
            fallback_bytes=fallback_linear,
        )
    )

    svc = LinearSVC(random_state=seed, max_iter=30_000, class_weight="balanced")
    svc.fit(train, y_train)
    rows.append(
        _candidate_result(
            name=f"{prefix}_linearsvc_d{feature_dim}",
            family=f"{prefix}_linearsvc",
            model=svc,
            predictions=svc.predict(test),
            y_test=y_test,
            feature_dim=feature_dim,
            fallback_bytes=fallback_linear,
        )
    )

    logreg = LogisticRegression(
        max_iter=10_000,
        random_state=seed,
        solver="liblinear",
        class_weight="balanced",
    )
    logreg.fit(train, y_train)
    rows.append(
        _candidate_result(
            name=f"{prefix}_logreg_d{feature_dim}",
            family=f"{prefix}_logreg",
            model=logreg,
            predictions=logreg.predict(test),
            y_test=y_test,
            feature_dim=feature_dim,
            fallback_bytes=fallback_linear,
        )
    )

    if include_nb:
        for family, model in (
            ("multinb", MultinomialNB()),
            ("complementnb", ComplementNB()),
        ):
            model.fit(train, y_train)
            rows.append(
                _candidate_result(
                    name=f"{prefix}_{family}_d{feature_dim}",
                    family=f"{prefix}_{family}",
                    model=model,
                    predictions=model.predict(test),
                    y_test=y_test,
                    feature_dim=feature_dim,
                    fallback_bytes=(2 * int(feature_dim) + 2) * 8,
                )
            )

    if include_nonlinear:
        dense_train = np.asarray(train, dtype=np.float64)
        dense_test = np.asarray(test, dtype=np.float64)
        rbf = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
        rbf.fit(dense_train, y_train)
        rows.append(
            _candidate_result(
                name=f"{prefix}_rbfsvc_d{feature_dim}",
                family=f"{prefix}_rbfsvc",
                model=rbf,
                predictions=rbf.predict(dense_test),
                y_test=y_test,
                feature_dim=feature_dim,
                fallback_bytes=0,
            )
        )

        knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine")
        knn.fit(dense_train, y_train)
        rows.append(
            _candidate_result(
                name=f"{prefix}_cosine_knn7_d{feature_dim}",
                family=f"{prefix}_cosine_knn",
                model=knn,
                predictions=knn.predict(dense_test),
                y_test=y_test,
                feature_dim=feature_dim,
                fallback_bytes=int(dense_train.nbytes + y_train.nbytes),
            )
        )

        trees = ExtraTreesClassifier(
            n_estimators=192,
            random_state=seed,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=1,
        )
        trees.fit(dense_train, y_train)
        rows.append(
            _candidate_result(
                name=f"{prefix}_extratrees_d{feature_dim}",
                family=f"{prefix}_extratrees",
                model=trees,
                predictions=trees.predict(dense_test),
                y_test=y_test,
                feature_dim=feature_dim,
                fallback_bytes=0,
            )
        )
    return rows


def _pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["model_bytes"]), row["name"]))
    frontier: list[dict[str, Any]] = []
    best = -math.inf
    for row in ordered:
        score = float(row["balanced_accuracy"])
        if score > best + 1e-15:
            frontier.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "predictions"
                }
            )
            best = score
    return frontier


def _classical_frontier(
    spec: DatasetSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    feature_dims: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    x_train = spec.x[train_idx]
    x_test = spec.x[test_idx]
    y_train = spec.y[train_idx]
    y_test = spec.y[test_idx]
    rows: list[dict[str, Any]] = []

    rows.extend(
        _linear_candidates(
            x_train,
            x_test,
            y_train,
            y_test,
            prefix="raw_gene",
            feature_dim=int(spec.x.shape[1]),
            seed=seed,
            include_nb=False,
            include_nonlinear=False,
        )
    )

    budget_summaries: list[dict[str, Any]] = []
    for feature_dim in feature_dims:
        encoded_train, train_stats = spec.build_interactions(
            x_train, feature_dim=int(feature_dim), **spec.interaction_kwargs
        )
        encoded_test, test_stats = spec.build_interactions(
            x_test, feature_dim=int(feature_dim), **spec.interaction_kwargs
        )
        minimum = min(float(np.min(encoded_train)), float(np.min(encoded_test)))
        candidates = _linear_candidates(
            encoded_train,
            encoded_test,
            y_train,
            y_test,
            prefix=("thirdhash" if spec.name == DATASET_GSE else "pairhash"),
            feature_dim=int(feature_dim),
            seed=seed,
            include_nb=minimum >= 0.0,
            include_nonlinear=int(feature_dim) <= 1024,
        )
        rows.extend(candidates)
        best_budget = max(
            candidates,
            key=lambda row: (
                float(row["balanced_accuracy"]),
                float(row["accuracy"]),
                -int(row["model_bytes"]),
            ),
        )
        budget_summaries.append(
            {
                "feature_dim": int(feature_dim),
                "train_stats": train_stats,
                "test_stats": test_stats,
                "best_name": best_budget["name"],
                "best_balanced_accuracy": best_budget["balanced_accuracy"],
                "best_accuracy": best_budget["accuracy"],
                "best_model_bytes": best_budget["model_bytes"],
            }
        )

    best = max(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["accuracy"]),
            -int(row["model_bytes"]),
        ),
    )
    return {
        "candidate_count": len(rows),
        "best_name": best["name"],
        "best_family": best["family"],
        "best_feature_dim": best["feature_dim"],
        "best_accuracy": best["accuracy"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "best_model_bytes": best["model_bytes"],
        "best_model_bytes_human": _human_bytes(int(best["model_bytes"])),
        "best_predictions": best["predictions"],
        "pareto_frontier": _pareto_frontier(rows),
        "budget_summaries": budget_summaries,
        "candidates": [
            {key: value for key, value in row.items() if key != "predictions"}
            for row in rows
        ],
        "selection_note": (
            "The maximum held-out score across the complete tested frontier is used "
            "as a deliberately conservative no-hardware gate, not as an unbiased "
            "publication estimate of a tuned classical model."
        ),
    }


def training_median_flat_vectors(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train_matrix, dtype=np.float64)
    test = np.asarray(test_matrix, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test interaction matrices must have matching widths")
    thresholds = np.median(train, axis=0)
    train_flat = np.where(train >= thresholds[None, :], 1.0, -1.0)
    test_flat = np.where(test >= thresholds[None, :], 1.0, -1.0)
    return train_flat, test_flat, thresholds


def _row_sketch_seed(base_seed: int, dataset: str, row_id: int) -> int:
    payload = f"{int(base_seed)}|{dataset}|{int(row_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def exact_flat_qos_features(
    flat_vectors: np.ndarray,
    row_ids: np.ndarray,
    *,
    dataset: str,
    num_samples: int,
    sketch_seed: int,
    verify_rows: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectors = np.asarray(flat_vectors, dtype=np.float64)
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[0] != row_ids.shape[0]:
        raise ValueError("flat vectors and row ids have incompatible shapes")
    dim = int(vectors.shape[1])
    qos_port.require_power_of_two(dim)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not np.all(np.isin(vectors, (-1.0, 1.0))):
        raise ValueError("flat vectors must contain only -1 and +1")

    features = np.zeros((vectors.shape[0], dim), dtype=np.float64)
    max_jax_probability_error = 0.0
    sample_hashes: list[str] = []
    for position, (vector, row_id) in enumerate(zip(vectors, row_ids, strict=True)):
        rng = np.random.default_rng(_row_sketch_seed(sketch_seed, dataset, int(row_id)))
        sampled_indices, sampled_values = qos_port.sample_from_vector(
            vector, int(num_samples), rng
        )
        probabilities = qos_port.flat_interference_probabilities_from_samples(
            sampled_indices, sampled_values, dim
        )
        features[position] = probabilities
        sample_hashes.append(
            _sha256_array(np.column_stack([sampled_indices, sampled_values]))
        )
        if position < int(verify_rows):
            jax_state = qos_port.flat_interference_state_from_jax(
                sampled_indices, sampled_values, dim
            )
            error = float(
                np.max(np.abs(probabilities - np.abs(jax_state) ** 2))
            )
            max_jax_probability_error = max(max_jax_probability_error, error)

    normalization_error = float(np.max(np.abs(np.sum(features, axis=1) - 1.0)))
    if normalization_error > 1e-10:
        raise GateError("flat QOS feature normalization failed")
    return features, {
        "rows": int(features.shape[0]),
        "dimension": dim,
        "num_samples_per_row": int(num_samples),
        "normalization_max_abs_error": normalization_error,
        "minimum_probability": float(np.min(features)),
        "maximum_probability": float(np.max(features)),
        "max_abs_probability_error_vs_official_jax": max_jax_probability_error,
        "probability_matrix_sha256": _sha256_array(features),
        "aggregate_samples_sha256": hashlib.sha256(
            "".join(sample_hashes).encode("ascii")
        ).hexdigest(),
    }


def hellinger_fidelity_kernel(
    first: np.ndarray,
    second: np.ndarray | None = None,
) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = first_array if second is None else np.asarray(second, dtype=np.float64)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise ValueError("probability features must be matrices")
    if first_array.shape[1] != second_array.shape[1]:
        raise ValueError("probability feature widths must match")
    if np.any(first_array < 0.0) or np.any(second_array < 0.0):
        raise ValueError("probabilities must be non-negative")
    return np.square(np.sqrt(first_array) @ np.sqrt(second_array).T)


def _quantum_kernel_result(
    spec: DatasetSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    dimension: int,
    num_samples: int,
    sketch_seed: int,
    split_seed: int,
) -> dict[str, Any]:
    x_train = spec.x[train_idx]
    x_test = spec.x[test_idx]
    y_train = spec.y[train_idx]
    y_test = spec.y[test_idx]
    encoded_train, train_stats = spec.build_interactions(
        x_train, feature_dim=dimension, **spec.interaction_kwargs
    )
    encoded_test, test_stats = spec.build_interactions(
        x_test, feature_dim=dimension, **spec.interaction_kwargs
    )
    train_flat, test_flat, thresholds = training_median_flat_vectors(
        encoded_train, encoded_test
    )
    train_probabilities, train_verification = exact_flat_qos_features(
        train_flat,
        spec.row_ids[train_idx],
        dataset=spec.name,
        num_samples=num_samples,
        sketch_seed=sketch_seed,
        verify_rows=2,
    )
    test_probabilities, test_verification = exact_flat_qos_features(
        test_flat,
        spec.row_ids[test_idx],
        dataset=spec.name,
        num_samples=num_samples,
        sketch_seed=sketch_seed,
        verify_rows=2,
    )
    train_kernel = hellinger_fidelity_kernel(train_probabilities)
    test_kernel = hellinger_fidelity_kernel(test_probabilities, train_probabilities)
    symmetry_error = float(np.max(np.abs(train_kernel - train_kernel.T)))
    diagonal_error = float(np.max(np.abs(np.diag(train_kernel) - 1.0)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(train_kernel)))
    if symmetry_error > 1e-10 or diagonal_error > 1e-10 or minimum_eigenvalue < -1e-9:
        raise GateError("Hellinger fidelity kernel failed PSD/normalization checks")

    model = SVC(
        C=1.0,
        kernel="precomputed",
        class_weight="balanced",
        random_state=split_seed,
    )
    model.fit(train_kernel, y_train)
    predictions = np.asarray(model.predict(test_kernel), dtype=np.int64)
    return {
        "dimension": int(dimension),
        "qubits": int(math.log2(dimension)),
        "samples_per_cell": int(num_samples),
        "model": "svc_precomputed_hellinger_fidelity_c1",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "predictions": [int(value) for value in predictions],
        "test_correct": [bool(value) for value in predictions == y_test],
        "support_vectors": int(np.sum(model.n_support_)),
        "estimated_classical_sidecar_bytes": int(
            train_probabilities.nbytes
            + test_probabilities.nbytes
            + train_kernel.nbytes
            + test_kernel.nbytes
            + thresholds.nbytes
            + _model_bytes(model)
        ),
        "flat_vector_train_sha256": _sha256_array(train_flat),
        "flat_vector_test_sha256": _sha256_array(test_flat),
        "threshold_sha256": _sha256_array(thresholds),
        "train_interaction_stats": train_stats,
        "test_interaction_stats": test_stats,
        "train_feature_verification": train_verification,
        "test_feature_verification": test_verification,
        "kernel_verification": {
            "symmetry_max_abs_error": symmetry_error,
            "diagonal_max_abs_error": diagonal_error,
            "minimum_eigenvalue": minimum_eigenvalue,
            "passed": True,
        },
    }


def _cluster_bootstrap_ci(
    split_deltas: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    values = np.asarray(split_deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(int(replicates), len(values)))
    means = np.mean(values[choices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def evaluate_gate_summary(
    split_rows: Sequence[dict[str, Any]],
    *,
    min_wins: int = GATE_MIN_WINS,
    bootstrap_seed: int = 91_771,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    deltas = [float(row["balanced_accuracy_delta"]) for row in split_rows]
    wins = sum(delta > 0.0 for delta in deltas)
    ties = sum(abs(delta) <= 1e-15 for delta in deltas)
    ci = _cluster_bootstrap_ci(
        deltas, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    mean_delta = float(np.mean(deltas))
    passed = wins >= int(min_wins) and mean_delta > 0.0
    return {
        "split_count": len(split_rows),
        "strict_wins": int(wins),
        "ties": int(ties),
        "losses": int(len(split_rows) - wins - ties),
        "minimum_required_wins": int(min_wins),
        "mean_balanced_accuracy_delta": mean_delta,
        "median_balanced_accuracy_delta": float(np.median(deltas)),
        "cluster_bootstrap_95pct_ci_mean_delta": [ci[0], ci[1]],
        "bootstrap_replicates": int(bootstrap_replicates),
        "passed": bool(passed),
        "rule": (
            "strict quantum wins on at least four of five splits and positive "
            "mean balanced-accuracy delta against the maximum tested classical frontier"
        ),
        "ci_note": (
            "The split-cluster bootstrap interval is diagnostic and is not an "
            "additional eligibility condition for this five-split screening gate."
        ),
    }


def _load_gse(args: argparse.Namespace) -> DatasetSpec:
    x_pair, guide_y, source_meta, pair_meta = semisynth.load_hard_polr1d_pair(
        cache_dir=args.gse_cache_dir,
        positive_guide=args.positive_guide,
        negative_guide=args.negative_guide,
    )
    labels, task_meta = semisynth.build_residualized_semisynth_labels(
        x_pair,
        guide_y,
        teacher_dim=args.teacher_dim,
        shortcut_dim=args.shortcut_dim,
        hash_seed=args.gse_hash_seed,
        shortcut_hash_seed=args.shortcut_hash_seed,
        value_mode="log-product",
        max_active_genes=args.gse_max_active_genes,
        hash_repeats=2,
        signed_hash=True,
        activation_scale=2.0,
        seed=args.gse_task_seed,
        ridge_alpha=1.0,
    )
    return DatasetSpec(
        name=DATASET_GSE,
        x=x_pair,
        y=np.asarray(labels, dtype=np.int64),
        row_ids=np.arange(x_pair.shape[0], dtype=np.int64),
        source={**source_meta, **pair_meta},
        task=task_meta,
        build_interactions=thirdorder.build_thirdorder_hashed_matrix,
        interaction_kwargs={
            "hash_seed": int(args.gse_hash_seed),
            "value_mode": "log-product",
            "max_active_genes": int(args.gse_max_active_genes),
            "hash_repeats": 2,
            "signed_hash": True,
            "activation_scale": 2.0,
        },
        train_size=int(args.gse_train_size),
        test_size=int(args.gse_test_size),
        ambient_feature_dim=int(comb(int(x_pair.shape[1]), 3)),
    )


def _load_pbmc(args: argparse.Namespace) -> DatasetSpec:
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=args.pbmc_cache_dir)
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    return DatasetSpec(
        name=DATASET_PBMC,
        x=x_pair,
        y=np.asarray(y_pair, dtype=np.int64),
        row_ids=np.arange(x_pair.shape[0], dtype=np.int64),
        source={**source_meta, **pair_meta},
        task={
            "task_type": "natural_binary_cell_type_labels",
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
        },
        build_interactions=pairwise.build_pairwise_hashed_matrix,
        interaction_kwargs={
            "hash_seed": int(args.pbmc_hash_seed),
            "value_mode": "log-product",
            "max_active_genes": int(args.pbmc_max_active_genes),
        },
        train_size=int(args.pbmc_train_size),
        test_size=int(args.pbmc_test_size),
        ambient_feature_dim=int(comb(int(x_pair.shape[1]), 2)),
    )


def _evaluate_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for split_number, seed in enumerate(args.split_seeds, start=1):
        split_started = time.perf_counter()
        train_idx, test_idx = thirdorder.balanced_binary_split(
            spec.y,
            seed=int(seed),
            train_fraction=0.67,
            max_train_samples=int(spec.train_size),
            max_test_samples=int(spec.test_size),
        )
        classical = _classical_frontier(
            spec,
            train_idx,
            test_idx,
            feature_dims=args.classical_dims,
            seed=int(seed),
        )
        quantum_candidates: list[dict[str, Any]] = []
        for dimension in args.dimensions:
            quantum = _quantum_kernel_result(
                spec,
                train_idx,
                test_idx,
                dimension=int(dimension),
                num_samples=int(args.samples_per_dimension * dimension),
                sketch_seed=int(args.sketch_seed),
                split_seed=int(seed),
            )
            delta = float(
                quantum["balanced_accuracy"]
                - classical["best_balanced_accuracy"]
            )
            quantum_candidates.append(
                {
                    **quantum,
                    "balanced_accuracy_delta": delta,
                    "strict_quantum_win": bool(delta > 0.0),
                }
            )
        row = {
            "split_number": int(split_number),
            "split_seed": int(seed),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_indices_sha256": _sha256_array(train_idx),
            "test_indices_sha256": _sha256_array(test_idx),
            "train_class_balance": {
                "positive": int(np.sum(spec.y[train_idx] > 0)),
                "negative": int(np.sum(spec.y[train_idx] < 0)),
            },
            "test_class_balance": {
                "positive": int(np.sum(spec.y[test_idx] > 0)),
                "negative": int(np.sum(spec.y[test_idx] < 0)),
            },
            "quantum_candidates": quantum_candidates,
            "classical_frontier": classical,
            "elapsed_seconds": float(time.perf_counter() - split_started),
        }
        split_rows.append(row)
        quantum_text = " ".join(
            f"q={candidate['qubits']}:"
            f"{candidate['balanced_accuracy']:.4f}"
            f"({candidate['balanced_accuracy_delta']:+.4f})"
            for candidate in quantum_candidates
        )
        print(
            f"[{spec.name}] split {split_number}/5 seed={seed}: "
            f"classical={classical['best_balanced_accuracy']:.4f} "
            f"via {classical['best_name']} | {quantum_text}",
            flush=True,
        )

    width_gates: list[dict[str, Any]] = []
    for dimension in args.dimensions:
        dimension_rows = []
        for split_row in split_rows:
            candidate = next(
                item
                for item in split_row["quantum_candidates"]
                if int(item["dimension"]) == int(dimension)
            )
            dimension_rows.append(
                {"balanced_accuracy_delta": candidate["balanced_accuracy_delta"]}
            )
        width_gate = evaluate_gate_summary(
            dimension_rows,
            min_wins=GATE_MIN_WINS,
            bootstrap_seed=int(args.bootstrap_seed + dimension),
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
        width_gates.append(
            {
                "dimension": int(dimension),
                "qubits": int(math.log2(dimension)),
                "samples_per_cell": int(args.samples_per_dimension * dimension),
                **width_gate,
            }
        )
    eligible_widths = [
        int(row["qubits"]) for row in width_gates if bool(row["passed"])
    ]
    gate = {
        "passed": bool(eligible_widths),
        "eligible_qubit_widths": eligible_widths,
        "width_gates": width_gates,
        "selection_rule": (
            "Every qubit width is judged independently on the same five frozen "
            "splits; no post-hoc best-width score is used."
        ),
    }
    return {
        "dataset": spec.name,
        "source": spec.source,
        "task": spec.task,
        "dataset_validation": {
            "rows": int(spec.x.shape[0]),
            "features": int(spec.x.shape[1]),
            "nnz": int(spec.x.nnz),
            "sparse_matrix_sha256": _sha256_sparse(spec.x),
            "labels_sha256": _sha256_array(spec.y),
            "positive_labels": int(np.sum(spec.y > 0)),
            "negative_labels": int(np.sum(spec.y < 0)),
        },
        "ambient_feature_dim": int(spec.ambient_feature_dim),
        "ambient_dense_weight_bytes": int(spec.ambient_feature_dim * 8),
        "ambient_dense_weight_human": _human_bytes(spec.ambient_feature_dim * 8),
        "splits": split_rows,
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_parse_datasets, default=SUPPORTED_DATASETS)
    parser.add_argument("--split-seeds", type=_parse_ints, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--classical-dims", type=_parse_ints, default=DEFAULT_CLASSICAL_DIMS)
    parser.add_argument("--dimensions", type=_parse_ints, default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--samples-per-dimension",
        type=int,
        default=DEFAULT_SAMPLES_PER_DIMENSION,
        help="M/dim ratio for every flat-QOS width",
    )
    parser.add_argument("--sketch-seed", type=int, default=DEFAULT_SKETCH_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=91_771)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--gse-cache-dir", default="data_cache/gse132080")
    parser.add_argument("--positive-guide", default="POLR1D_+_28196016.23-P1_08")
    parser.add_argument("--negative-guide", default="POLR1D_+_28196016.23-P1_00")
    parser.add_argument("--teacher-dim", type=int, default=65_536)
    parser.add_argument("--shortcut-dim", type=int, default=4_096)
    parser.add_argument("--shortcut-hash-seed", type=int)
    parser.add_argument("--gse-task-seed", type=int, default=7)
    parser.add_argument("--gse-hash-seed", type=int, default=7)
    parser.add_argument("--gse-max-active-genes", type=int, default=48)
    parser.add_argument("--gse-train-size", type=int, default=160)
    parser.add_argument("--gse-test-size", type=int, default=160)
    parser.add_argument("--pbmc-cache-dir", default="data_cache/pbmc68k")
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--pbmc-hash-seed", type=int, default=7)
    parser.add_argument("--pbmc-max-active-genes", type=int, default=48)
    parser.add_argument("--pbmc-train-size", type=int, default=256)
    parser.add_argument("--pbmc-test-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.split_seeds) != 5:
        raise GateError("the frozen gate requires exactly five split seeds")
    for dimension in args.dimensions:
        qos_port.require_power_of_two(int(dimension))
    if args.samples_per_dimension <= 0:
        raise GateError("samples-per-dimension must be positive")
    if args.output.exists() and not args.force:
        raise GateError(f"output already exists: {args.output}")

    loaders = {DATASET_GSE: _load_gse, DATASET_PBMC: _load_pbmc}
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"Loading and evaluating {dataset}", flush=True)
        results.append(_evaluate_dataset(loaders[dataset](args), args))

    eligible = [
        {
            "dataset": result["dataset"],
            "qubit_widths": result["gate"]["eligible_qubit_widths"],
        }
        for result in results
        if result["gate"]["passed"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "local_gate_complete",
        "protocol": {
            "official_kernel": "official_qos.qos_sampling.q_state_sketch_flat",
            "source_mapping": (
                "real interaction hash vector thresholded to +/-1 using training-only "
                "per-coordinate medians"
            ),
            "phase_equation": (
                "phi[j] = pi * dim / M * sum_m 1[i_m=j] * (1-v_m)/2"
            ),
            "interference_state": "H^n D(phi) H^n |0...0>",
            "kernel": "(sum_z sqrt(p_x[z] p_y[z]))^2",
            "dimensions": [int(value) for value in args.dimensions],
            "qubit_widths": [int(math.log2(value)) for value in args.dimensions],
            "samples_per_dimension": int(args.samples_per_dimension),
            "samples_per_cell_by_dimension": {
                str(int(value)): int(args.samples_per_dimension * value)
                for value in args.dimensions
            },
            "sketch_seed": int(args.sketch_seed),
            "split_seeds": [int(value) for value in args.split_seeds],
            "classical_feature_dims": [int(value) for value in args.classical_dims],
            "gate_minimum_strict_wins": GATE_MIN_WINS,
        },
        "equation_registry": [
            {
                "id": "eq:flat-phase",
                "code": "qiskit_official_qos_sampling_port.flat_phase_from_samples",
                "status": "verified against official JAX",
            },
            {
                "id": "eq:interference-probability",
                "code": "qiskit_official_qos_sampling_port.flat_interference_probabilities_from_samples",
                "status": "normalization and official-JAX parity checked",
            },
            {
                "id": "eq:hellinger-kernel",
                "code": "qiskit_official_qos_realdata_gate.hellinger_fidelity_kernel",
                "status": "symmetry, unit diagonal, and PSD checked per split",
            },
        ],
        "datasets": results,
        "hardware_gate": {
            "eligible_candidates": eligible,
            "passed": bool(eligible),
            "provider_calls": [],
            "execution_attempted": False,
            "automatic_submission": False,
            "small_width_provider_route": "direct IBM Runtime after a separate frozen plan",
            "fire_opal_policy": "deferred and reserved for a later q40/q60 shallow route",
            "decision": (
                "eligible for a separate direct-IBM small-width pilot plan"
                if eligible
                else "blocked locally; do not transmit circuits or submit any hardware"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "Passing this gate would identify a real-data candidate for a separately "
            "validated Fire Opal hardware experiment. It would not itself establish "
            "computational or exponential quantum advantage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Saved gate artifact: {args.output}", flush=True)
    print(f"Direct-IBM small-width eligible candidates: {eligible or 'none'}", flush=True)
    print("Fire Opal policy: deferred for q40/q60", flush=True)
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
