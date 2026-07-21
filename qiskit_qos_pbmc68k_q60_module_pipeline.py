#!/usr/bin/env python3
"""Local 60-qubit PBMC68k coexpression-module research pipeline.

This module owns the data freeze, leakage-safe representation, five-split MPS
screen, and matched hardware analysis.  It deliberately contains no Fire Opal
import or provider call.  Hardware payload validation and submission live in
separate runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score

import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_pbmc_coherent_stream_hardness_screen as coherent
import qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis as q40_analysis
import qiskit_qos_pbmc_width_scaled_entangler_screen as width_scaled


SCHEMA_VERSION = "1.0"
KIND = "pbmc68k_q60_coexpression_modules_b4_study"
LOCAL_SCREEN_KIND = "pbmc68k_q60_coexpression_modules_b4_five_split_mps_screen"
HARDWARE_ANALYSIS_KIND = "pbmc68k_q60_coexpression_modules_b4_hardware_analysis"
ARCHITECTURE = "coexpression_modules_b4_width_scaled_entangler"
QUBITS = 60
BLOCK_COUNT = 4
SCALE_LAW = "sqrt_q"
PAIR_MULTIPLIER = math.sqrt(QUBITS)
SEED = 11
MODULE_SEED = 6110
ALLOCATION_SEED = 6111
CV_SEED = 6011
POSITIVE_LABEL = "CD4+/CD25 T Reg"
NEGATIVE_LABEL = "CD4+/CD45RO+ Memory"
MODULE_POOL_SAMPLES = 512
SELECTED_GENES = 1200
MODULE_COUNT = 60
DEVELOPMENT_SPLITS = 5
TRAIN_SAMPLES = 256
TEST_SAMPLES = 256
SENTINEL_TRAIN_SAMPLES = 32
SENTINEL_TEST_SAMPLES = 32
DETECTION_MIN = 0.01
DETECTION_MAX = 0.95
MPS_BOND_DIMENSIONS = (64, 128, 256, 512)
MPS_THRESHOLD = 1e-10
CONVERGENCE_TOLERANCE = 1e-3
EXPECTED_LOGICAL_DEPTH = 20
EXPECTED_LOGICAL_TWO_QUBIT_GATES = 134
EXPECTED_OBSERVABLES = 627
DEFAULT_SOURCE_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_seed11_train_only_tuning.json"
)
DEFAULT_SPECIFICATION = Path(
    "q60_module_b4/pbmc68k_q60_module_b4_seed11_study.json"
)
DEFAULT_LOCAL_SCREEN = Path(
    "q60_module_b4/pbmc68k_q60_module_b4_seed11_five_split_screen.json"
)


class ModulePipelineError(RuntimeError):
    """Raised when a frozen study invariant is violated."""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(header + array.tobytes())


def _sha256_sparse(matrix: sparse.spmatrix) -> str:
    value = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value.indptr).tobytes())
    digest.update(np.ascontiguousarray(value.indices).tobytes())
    digest.update(np.ascontiguousarray(value.data).tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ModulePipelineError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModulePipelineError(
            f"{label} could not be read ({type(exc).__name__})"
        ) from None
    if not isinstance(value, dict):
        raise ModulePipelineError(f"{label} must be a JSON object")
    return value


def runtime_environment() -> dict[str, Any]:
    import sklearn

    try:
        import qiskit

        qiskit_version: str | None = qiskit.__version__
    except ImportError:
        qiskit_version = None
    try:
        import qiskit_aer

        aer_version: str | None = qiskit_aer.__version__
    except ImportError:
        aer_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit_version,
        "qiskit_aer": aer_version,
        "float_precision": "float64",
    }


def library_log1p(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Normalize every cell to 10,000 counts and apply log1p."""

    value = matrix.tocsr().astype(np.float64, copy=True)
    totals = np.asarray(value.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(
        10_000.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0.0,
    )
    value = sparse.diags(scale) @ value
    value.data = np.log1p(value.data)
    return value.tocsr()


def allocate_study_indices(
    labels: np.ndarray,
    *,
    sentinel_train: Sequence[int],
    sentinel_test: Sequence[int],
    module_pool_samples: int = MODULE_POOL_SAMPLES,
    development_splits: int = DEVELOPMENT_SPLITS,
    train_samples: int = TRAIN_SAMPLES,
    test_samples: int = TEST_SAMPLES,
    module_seed: int = MODULE_SEED,
    allocation_seed: int = ALLOCATION_SEED,
) -> dict[str, Any]:
    """Freeze mutually disjoint module, development, final, and sentinel rows."""

    y = np.asarray(labels, dtype=np.int64)
    if set(np.unique(y).tolist()) != {-1, 1}:
        raise ModulePipelineError("Study allocation requires binary -1/+1 labels")
    sentinel_train_values = np.asarray(sentinel_train, dtype=np.int64)
    sentinel_test_values = np.asarray(sentinel_test, dtype=np.int64)
    sentinel = np.concatenate((sentinel_train_values, sentinel_test_values))
    if (
        len(np.unique(sentinel)) != len(sentinel)
        or len(sentinel_train_values) != SENTINEL_TRAIN_SAMPLES
        or len(sentinel_test_values) != SENTINEL_TEST_SAMPLES
        or np.min(sentinel) < 0
        or np.max(sentinel) >= len(y)
    ):
        raise ModulePipelineError("Frozen sentinel indices are invalid")

    available = np.setdiff1d(np.arange(len(y), dtype=np.int64), sentinel)
    module_rng = np.random.default_rng(int(module_seed))
    module_order = module_rng.permutation(available)
    if len(module_order) < int(module_pool_samples):
        raise ModulePipelineError("Not enough rows for the module-learning pool")
    module_pool = np.asarray(module_order[: int(module_pool_samples)], dtype=np.int64)
    remaining = np.setdiff1d(available, module_pool)

    per_class_train = int(train_samples) // 2
    per_class_test = int(test_samples) // 2
    if train_samples % 2 or test_samples % 2:
        raise ModulePipelineError("Balanced train and test sizes must be even")
    per_class_split = per_class_train + per_class_test
    required_per_class = (int(development_splits) + 1) * per_class_split
    allocation_rng = np.random.default_rng(int(allocation_seed))
    by_class: dict[int, np.ndarray] = {}
    for label in (-1, 1):
        candidates = remaining[y[remaining] == label]
        if len(candidates) < required_per_class:
            raise ModulePipelineError(
                f"Not enough label {label} rows for disjoint development/final splits"
            )
        by_class[label] = allocation_rng.permutation(candidates)

    def take_split(offset: int) -> dict[str, list[int]]:
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        start = offset * per_class_split
        for label in (-1, 1):
            rows = by_class[label][start : start + per_class_split]
            train_parts.append(rows[:per_class_train])
            test_parts.append(rows[per_class_train:])
        train = allocation_rng.permutation(np.concatenate(train_parts))
        test = allocation_rng.permutation(np.concatenate(test_parts))
        return {
            "train_indices": [int(value) for value in train],
            "test_indices": [int(value) for value in test],
        }

    development = [
        {"split_id": f"dev_{index + 1}", **take_split(index)}
        for index in range(int(development_splits))
    ]
    final = {"split_id": "final_blind", **take_split(int(development_splits))}
    sections: list[Sequence[int]] = [
        sentinel_train_values,
        sentinel_test_values,
        module_pool,
        *[
            row[key]
            for row in development
            for key in ("train_indices", "test_indices")
        ],
        final["train_indices"],
        final["test_indices"],
    ]
    flattened = np.concatenate([np.asarray(value, dtype=np.int64) for value in sections])
    if len(np.unique(flattened)) != len(flattened):
        raise ModulePipelineError("Study allocation contains cross-section leakage")
    return {
        "allocation_protocol": "predeclared disjoint allocation before feature evaluation",
        "module_pool_seed": int(module_seed),
        "allocation_seed": int(allocation_seed),
        "sentinel": {
            "split_id": "seed11_sentinel",
            "train_indices": [int(value) for value in sentinel_train_values],
            "test_indices": [int(value) for value in sentinel_test_values],
        },
        "module_learning_pool": {
            "indices": [int(value) for value in module_pool],
            "selection_used_labels": False,
            "samples": int(len(module_pool)),
            "class_counts_for_audit_only": {
                "negative": int(np.sum(y[module_pool] < 0)),
                "positive": int(np.sum(y[module_pool] > 0)),
            },
        },
        "development": development,
        "final": final,
        "all_sections_pairwise_disjoint": True,
        "allocated_unique_rows": int(len(flattened)),
    }


def _gene_names(cache_dir: Path, expected_genes: int) -> list[str]:
    archive = cache_dir / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    lines = pbmc._read_member_text(archive, "genes.tsv")
    names = pbmc._parse_features(lines)["gene_name"].astype(str).tolist()
    if len(names) != int(expected_genes):
        raise ModulePipelineError("Gene-name count differs from matrix width")
    return names


def learn_coexpression_modules(
    x_pool: sparse.spmatrix,
    *,
    gene_names: Sequence[str] | None = None,
    selected_genes: int = SELECTED_GENES,
    module_count: int = MODULE_COUNT,
    detection_min: float = DETECTION_MIN,
    detection_max: float = DETECTION_MAX,
    random_state: int = MODULE_SEED,
    n_init: int = 20,
) -> dict[str, Any]:
    """Learn stable label-free modules from centered normalized gene profiles."""

    raw = x_pool.tocsr()
    if raw.shape[0] < 2 or raw.shape[1] < int(selected_genes):
        raise ModulePipelineError("Module-learning matrix is too small")
    detection = np.asarray(raw.getnnz(axis=0), dtype=np.float64).ravel() / raw.shape[0]
    eligible = np.flatnonzero(
        (detection >= float(detection_min)) & (detection <= float(detection_max))
    )
    if len(eligible) < int(selected_genes):
        raise ModulePipelineError("Detection filter retained fewer than 1,200 genes")
    logged = library_log1p(raw)
    eligible_values = logged[:, eligible]
    means = np.asarray(eligible_values.mean(axis=0), dtype=np.float64).ravel()
    squared_means = np.asarray(
        eligible_values.multiply(eligible_values).mean(axis=0), dtype=np.float64
    ).ravel()
    variances = np.maximum(0.0, squared_means - means**2)
    order = np.lexsort((eligible, -variances))
    selected = np.asarray(eligible[order[: int(selected_genes)]], dtype=np.int64)

    profiles = logged[:, selected].transpose().toarray().astype(np.float64, copy=False)
    profiles -= profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = np.divide(
        profiles, norms, out=np.zeros_like(profiles), where=norms > 0.0
    )
    model = KMeans(
        n_clusters=int(module_count),
        random_state=int(random_state),
        n_init=int(n_init),
    )
    raw_labels = np.asarray(model.fit_predict(profiles), dtype=np.int64)
    raw_modules = [selected[raw_labels == label] for label in range(int(module_count))]
    if any(len(values) == 0 for values in raw_modules):
        raise ModulePipelineError("KMeans produced an empty coexpression module")
    raw_modules.sort(key=lambda values: int(np.min(values)))

    names = list(gene_names) if gene_names is not None else None
    modules: list[dict[str, Any]] = []
    for module_index, values in enumerate(raw_modules):
        ordered = np.asarray(sorted(int(value) for value in values), dtype=np.int64)
        row: dict[str, Any] = {
            "module_index": int(module_index),
            "gene_indices": ordered.tolist(),
            "gene_count": int(len(ordered)),
        }
        if names is not None:
            row["gene_names"] = [str(names[index]) for index in ordered]
        modules.append(row)
    membership = np.full(raw.shape[1], -1, dtype=np.int64)
    for row in modules:
        membership[np.asarray(row["gene_indices"], dtype=np.int64)] = int(
            row["module_index"]
        )
    return {
        "selection": {
            "detection_frequency_minimum": float(detection_min),
            "detection_frequency_maximum": float(detection_max),
            "eligible_genes": int(len(eligible)),
            "selected_most_variable_genes": int(len(selected)),
            "variance_definition": "population variance of library-normalized log1p",
            "selected_gene_indices": selected.tolist(),
            "selected_gene_indices_sha256": _sha256_array(selected),
        },
        "clustering": {
            "algorithm": "sklearn.cluster.KMeans",
            "profile": "centered L2-normalized gene expression across module-pool cells",
            "random_state": int(random_state),
            "n_init": int(n_init),
            "module_count": int(module_count),
            "inertia": float(model.inertia_),
            "module_sizes": [int(row["gene_count"]) for row in modules],
            "membership_sha256": _sha256_array(membership),
        },
        "modules": modules,
    }


def module_statistics(
    matrix: sparse.spmatrix, modules: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Return N x 4 x 60 mean/detection/RMS/upper-quartile features."""

    raw = matrix.tocsr()
    logged = library_log1p(raw)
    features = np.empty((raw.shape[0], 4, len(modules)), dtype=np.float64)
    for module_index, module in enumerate(modules):
        genes = np.asarray(module["gene_indices"], dtype=np.int64)
        if not len(genes):
            raise ModulePipelineError("Module contains no genes")
        values = logged[:, genes].toarray()
        detected = raw[:, genes].getnnz(axis=1) / float(len(genes))
        upper_count = max(1, int(math.ceil(len(genes) / 4.0)))
        upper = np.partition(values, values.shape[1] - upper_count, axis=1)[
            :, -upper_count:
        ]
        features[:, 0, module_index] = np.mean(values, axis=1)
        features[:, 1, module_index] = np.asarray(detected).ravel()
        features[:, 2, module_index] = np.sqrt(np.mean(values**2, axis=1))
        features[:, 3, module_index] = np.mean(upper, axis=1)
    if not np.all(np.isfinite(features)) or np.min(features) < 0.0:
        raise ModulePipelineError("Module statistics are non-finite or negative")
    return features


def fit_robust_block_scaler(training_statistics: np.ndarray) -> dict[str, Any]:
    values = np.asarray(training_statistics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (BLOCK_COUNT, QUBITS):
        raise ModulePipelineError("Scaler expects N x 4 x 60 module statistics")
    median = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    iqr = q75 - q25
    safe_iqr = np.where(iqr > 0.0, iqr, 1.0)
    return {
        "median": median,
        "iqr": safe_iqr,
        "zero_iqr_count": int(np.sum(iqr <= 0.0)),
        "fit_samples": int(len(values)),
        "fit_scope": "training rows only",
    }


def transform_robust_blocks(
    statistics: np.ndarray, scaler: Mapping[str, Any]
) -> np.ndarray:
    values = np.asarray(statistics, dtype=np.float64)
    median = np.asarray(scaler["median"], dtype=np.float64)
    iqr = np.asarray(scaler["iqr"], dtype=np.float64)
    transformed = np.tanh(((values - median) / iqr) / 3.0)
    norms = np.linalg.norm(transformed, axis=2, keepdims=True)
    transformed = np.divide(
        transformed,
        norms,
        out=np.zeros_like(transformed),
        where=norms > 0.0,
    )
    if not np.all(np.isfinite(transformed)) or np.max(np.abs(transformed)) > 1.0 + 1e-12:
        raise ModulePipelineError("Scaled blocks violate finite unit bounds")
    return transformed


def _serializable_scaler(scaler: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "median": np.asarray(scaler["median"], dtype=np.float64).tolist(),
        "iqr": np.asarray(scaler["iqr"], dtype=np.float64).tolist(),
        "zero_iqr_count": int(scaler["zero_iqr_count"]),
        "fit_samples": int(scaler["fit_samples"]),
        "fit_scope": str(scaler["fit_scope"]),
        "transform": "tanh(((x - median) / IQR) / 3), then per-block L2",
    }


def load_prepared_specification(path: Path) -> dict[str, Any]:
    value = _load_json(path, label="Prepared q60 module study")
    config = value.get("config", {})
    if (
        value.get("kind") != KIND
        or value.get("completed") is not True
        or value.get("execution_attempted") is not False
        or int(config.get("qubits", -1)) != QUBITS
        or int(config.get("block_count", -1)) != BLOCK_COUNT
        or int(config.get("seed", -1)) != SEED
        or int(config.get("module_count", -1)) != MODULE_COUNT
        or value.get("splits", {}).get("all_sections_pairwise_disjoint") is not True
    ):
        raise ModulePipelineError("Prepared study does not match the frozen q60 design")
    return value


def _load_pair(cache_dir: Path) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
    )
    return x_pair, y_pair, {**source_meta, **pair_meta}


def prepare_study(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise ModulePipelineError(f"Refusing to overwrite existing artifact: {args.output}")
    started = time.perf_counter()
    source = _load_json(args.source_report, label="Seed-11 source report")
    source_config = source.get("config", {})
    split = source.get("split", {})
    if (
        int(source_config.get("seed", -1)) != SEED
        or len(split.get("train_indices", [])) != SENTINEL_TRAIN_SAMPLES
        or len(split.get("test_indices", [])) != SENTINEL_TEST_SAMPLES
    ):
        raise ModulePipelineError("Source report lacks the expected seed-11 32/32 split")
    x_pair, y_pair, source_meta = _load_pair(args.cache_dir)
    allocations = allocate_study_indices(
        y_pair,
        sentinel_train=split["train_indices"],
        sentinel_test=split["test_indices"],
    )
    module_indices = np.asarray(
        allocations["module_learning_pool"]["indices"], dtype=np.int64
    )
    names = _gene_names(args.cache_dir, x_pair.shape[1])
    representation = learn_coexpression_modules(
        x_pair[module_indices], gene_names=names
    )
    selected = np.asarray(
        representation["selection"]["selected_gene_indices"], dtype=np.int64
    )
    module_pool_features = module_statistics(
        x_pair[module_indices], representation["modules"]
    )
    module_sizes = representation["clustering"]["module_sizes"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "completed": True,
        "captured_at_utc": _utc_now(),
        "environment": runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "provider_calls": [],
        "config": {
            "dataset": "PBMC68k",
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "seed": SEED,
            "module_seed": MODULE_SEED,
            "allocation_seed": ALLOCATION_SEED,
            "architecture": ARCHITECTURE,
            "qubits": QUBITS,
            "grid_shape": [6, 10],
            "block_count": BLOCK_COUNT,
            "block_statistics": [
                "mean_library_normalized_log1p",
                "detection_fraction",
                "rms_library_normalized_log1p",
                "upper_quartile_mean_library_normalized_log1p",
            ],
            "scale_law": SCALE_LAW,
            "pair_multiplier": PAIR_MULTIPLIER,
            "module_pool_samples": MODULE_POOL_SAMPLES,
            "selected_genes": SELECTED_GENES,
            "module_count": MODULE_COUNT,
            "kmeans_n_init": 20,
            "development_splits": DEVELOPMENT_SPLITS,
            "train_samples_per_split": TRAIN_SAMPLES,
            "test_samples_per_split": TEST_SAMPLES,
            "mps_bond_dimensions": list(MPS_BOND_DIMENSIONS),
            "mps_convergence_tolerance": CONVERGENCE_TOLERANCE,
        },
        "dataset": {
            **source_meta,
            "pair_sparse_matrix_sha256": _sha256_sparse(x_pair),
            "pair_labels_sha256": _sha256_array(y_pair),
            "cache_artifacts": {
                str(path.name): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in sorted(args.cache_dir.iterdir())
                if path.is_file()
            },
        },
        "source_artifact": {
            "path": str(args.source_report.resolve()),
            "sha256": _sha256_file(args.source_report),
        },
        "splits": allocations,
        "representation": {
            **representation,
            "module_pool_feature_shape": list(module_pool_features.shape),
            "module_pool_features_sha256": _sha256_array(module_pool_features),
            "selected_gene_log1p_sha256": _sha256_sparse(
                library_log1p(x_pair[module_indices])[:, selected]
            ),
            "module_size_minimum": int(min(module_sizes)),
            "module_size_maximum": int(max(module_sizes)),
        },
        "leakage_audit": {
            "module_learning_used_labels": False,
            "module_learning_pool_disjoint_from_sentinel": True,
            "module_learning_pool_disjoint_from_development": True,
            "module_learning_pool_disjoint_from_final": True,
            "development_splits_mutually_disjoint": True,
            "final_split_reserved_before_any_model_result": True,
            "robust_scalers_not_fitted_yet": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This artifact freezes a label-free module representation and disjoint "
            "data allocation only. It contains no ideal or hardware performance result."
        ),
    }
    _atomic_write_json(args.output, payload)
    return payload


def _split_record(specification: Mapping[str, Any], split_id: str) -> dict[str, Any]:
    splits = specification["splits"]
    candidates = [splits["sentinel"], *splits["development"], splits["final"]]
    row = next((value for value in candidates if value.get("split_id") == split_id), None)
    if row is None:
        raise ModulePipelineError(f"Unknown frozen split: {split_id}")
    return dict(row)


def rebuild_split(
    specification: Mapping[str, Any], *, split_id: str, cache_dir: Path
) -> dict[str, Any]:
    x_pair, y_pair, source_meta = _load_pair(cache_dir)
    if (
        _sha256_sparse(x_pair) != specification["dataset"]["pair_sparse_matrix_sha256"]
        or _sha256_array(y_pair) != specification["dataset"]["pair_labels_sha256"]
    ):
        raise ModulePipelineError("Cached PBMC68k pair differs from the frozen study")
    split = _split_record(specification, split_id)
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    modules = specification["representation"]["modules"]
    train_statistics = module_statistics(x_pair[train_indices], modules)
    test_statistics = module_statistics(x_pair[test_indices], modules)
    scaler = fit_robust_block_scaler(train_statistics)
    train_blocks = transform_robust_blocks(train_statistics, scaler)
    test_blocks = transform_robust_blocks(test_statistics, scaler)
    selected = np.asarray(
        specification["representation"]["selection"]["selected_gene_indices"],
        dtype=np.int64,
    )
    train_classical = library_log1p(x_pair[train_indices])[:, selected]
    test_classical = library_log1p(x_pair[test_indices])[:, selected]
    return {
        "split_id": split_id,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "y_train": np.asarray(y_pair[train_indices], dtype=np.int64),
        "y_test": np.asarray(y_pair[test_indices], dtype=np.int64),
        "train_blocks": train_blocks,
        "test_blocks": test_blocks,
        "train_classical": train_classical,
        "test_classical": test_classical,
        "scaler": scaler,
        "metadata": {
            "source": source_meta,
            "train_statistics_sha256": _sha256_array(train_statistics),
            "test_statistics_sha256": _sha256_array(test_statistics),
            "train_blocks_sha256": _sha256_array(train_blocks),
            "test_blocks_sha256": _sha256_array(test_blocks),
            "train_classical_sha256": _sha256_sparse(train_classical),
            "test_classical_sha256": _sha256_sparse(test_classical),
            "scaler": _serializable_scaler(scaler),
        },
    }


def build_unmeasured_circuits(blocks: np.ndarray) -> list[Any]:
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (BLOCK_COUNT, QUBITS):
        raise ModulePipelineError("Circuit builder expects N x 4 x 60 blocks")
    circuits = [
        coherent.coherent_stream_circuit(row, pair_multiplier=PAIR_MULTIPLIER)
        for row in values
    ]
    metrics = coherent.shallow.q40_validate.circuit_metrics(circuits[0])
    if (
        int(metrics["depth"]) != EXPECTED_LOGICAL_DEPTH
        or int(metrics["two_qubit_gates"]) != EXPECTED_LOGICAL_TWO_QUBIT_GATES
    ):
        raise ModulePipelineError("q60 circuit metrics differ from the frozen design")
    return circuits


def observable_mappings() -> list[dict[int, str]]:
    mappings = coherent.grid_aligned_mappings(QUBITS)
    if len(mappings) != EXPECTED_OBSERVABLES:
        raise ModulePipelineError("q60 grid panel must contain 627 observables")
    return mappings


def _probe_rows(blocks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    chosen: list[int] = []
    for label in (-1, 1):
        positions = np.flatnonzero(np.asarray(labels) == label)
        if len(positions) < 4:
            raise ModulePipelineError("MPS probe needs four training rows per class")
        chosen.extend(int(value) for value in positions[:4])
    return np.asarray(blocks[np.asarray(chosen, dtype=np.int64)], dtype=np.float64)


def mps_convergence_probe(blocks: np.ndarray, labels: np.ndarray, *, label: str) -> dict[str, Any]:
    result = coherent.mps_convergence_screen(
        _probe_rows(blocks, labels),
        bond_dimensions=MPS_BOND_DIMENSIONS,
        threshold=MPS_THRESHOLD,
        tolerance=CONVERGENCE_TOLERANCE,
        label=label,
        pair_multiplier=PAIR_MULTIPLIER,
    )
    chi256 = next(
        row for row in result["comparisons"] if int(row["bond_dimension"]) == 256
    )
    result["convergence_definition"] = "max_j |f_j(chi=256)-f_j(chi=512)| <= 1e-3"
    result["converged_at_chi512"] = bool(chi256["within_tolerance"])
    return result


def _family_candidates(representation: str, family: str) -> list[dict[str, Any]]:
    candidates = q40_analysis._candidate_grid([representation])
    rows = [row for row in candidates if row["family"] == family]
    for index, row in enumerate(rows):
        row["candidate_id"] = f"{family}_{index:02d}"
    return rows


def evaluate_feature_split(
    quantum_train: np.ndarray,
    quantum_test: np.ndarray,
    classical_train: sparse.spmatrix,
    classical_test: sparse.spmatrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Fit all models on training rows only, then open the fixed test rows once."""

    quantum_name = "quantum_pauli_627"
    quantum_candidates = q40_analysis._candidate_grid([quantum_name])
    quantum_selection = q40_analysis.select_model_training_only(
        {quantum_name: np.asarray(quantum_train, dtype=np.float64)},
        y_train,
        quantum_candidates,
        cv_seed=CV_SEED,
    )
    quantum_result = q40_analysis.fit_frozen_candidate(
        quantum_selection["chosen"],
        {quantum_name: np.asarray(quantum_train, dtype=np.float64)},
        {quantum_name: np.asarray(quantum_test, dtype=np.float64)},
        y_train,
        y_test,
        seed=CV_SEED,
    )
    classical_results: dict[str, Any] = {}
    for family in ("linear_svc", "rbf_svc"):
        candidates = _family_candidates("raw_gene_log1p", family)
        selection = q40_analysis.select_model_training_only(
            {"raw_gene_log1p": classical_train},
            y_train,
            candidates,
            cv_seed=CV_SEED,
        )
        fitted = q40_analysis.fit_frozen_candidate(
            selection["chosen"],
            {"raw_gene_log1p": classical_train},
            {"raw_gene_log1p": classical_test},
            y_train,
            y_test,
            seed=CV_SEED,
        )
        classical_results[family] = {"training_only_selection": selection, "fixed_test": fitted}
    quantum_accuracy = float(quantum_result["test_balanced_accuracy"])
    classical_accuracies = {
        family: float(row["fixed_test"]["test_balanced_accuracy"])
        for family, row in classical_results.items()
    }
    return {
        "quantum": {
            "training_only_selection": quantum_selection,
            "fixed_test": quantum_result,
        },
        "classical": classical_results,
        "quantum_test_balanced_accuracy": quantum_accuracy,
        "classical_test_balanced_accuracies": classical_accuracies,
        "quantum_beats_linear_and_rbf": bool(
            all(quantum_accuracy > value for value in classical_accuracies.values())
        ),
        "test_labels_used_for_fitting_or_selection": False,
    }


def run_local_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise ModulePipelineError(f"Refusing to overwrite existing artifact: {args.output}")
    specification = load_prepared_specification(args.specification)
    started = time.perf_counter()
    split_results: list[dict[str, Any]] = []
    for split_index in range(DEVELOPMENT_SPLITS):
        split_id = f"dev_{split_index + 1}"
        data = rebuild_split(specification, split_id=split_id, cache_dir=args.cache_dir)
        convergence = mps_convergence_probe(
            data["train_blocks"], data["y_train"], label=split_id
        )
        evaluation: dict[str, Any] | None = None
        if convergence["converged_at_chi512"]:
            selected = convergence.get("selected_lower_converged_bond_dimension") or 256
            all_blocks = np.concatenate(
                (data["train_blocks"], data["test_blocks"]), axis=0
            )
            features, seconds = coherent.architecture.simulate_feature_rows(
                build_unmeasured_circuits(all_blocks),
                observable_mappings(),
                bond_dimension=int(selected),
                threshold=MPS_THRESHOLD,
                progress_label=f"{split_id}-full",
            )
            evaluation = evaluate_feature_split(
                features[:TRAIN_SAMPLES],
                features[TRAIN_SAMPLES:],
                data["train_classical"],
                data["test_classical"],
                data["y_train"],
                data["y_test"],
            )
            evaluation["full_feature_simulation_seconds"] = float(seconds)
            evaluation["full_feature_matrix_sha256"] = _sha256_array(features)
        split_results.append(
            {
                "split_id": split_id,
                "train_indices": data["train_indices"].tolist(),
                "test_indices": data["test_indices"].tolist(),
                "data": data["metadata"],
                "mps_convergence": convergence,
                "evaluation": evaluation,
            }
        )
        partial = {
            "schema_version": SCHEMA_VERSION,
            "kind": LOCAL_SCREEN_KIND,
            "status": "in_progress",
            "completed_splits": len(split_results),
            "splits": split_results,
        }
        _atomic_write_json(args.output, partial)

    all_converged = all(
        row["mps_convergence"]["converged_at_chi512"] for row in split_results
    )
    passing_splits = sum(
        bool(row["evaluation"] and row["evaluation"]["quantum_beats_linear_and_rbf"])
        for row in split_results
    )
    if all_converged:
        gate_status = "performance_candidate" if passing_splits >= 4 else "performance_gate_failed"
        large_allowed = passing_splits >= 4
        claim_tier = "predictive_candidate" if large_allowed else "local_negative_result"
    else:
        gate_status = "mps_not_converged_at_chi512"
        large_allowed = True
        claim_tier = "hardware_feasibility_only"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": LOCAL_SCREEN_KIND,
        "status": gate_status,
        "completed": True,
        "captured_at_utc": _utc_now(),
        "environment": runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "provider_calls": [],
        "specification": {
            "path": str(args.specification.resolve()),
            "sha256": _sha256_file(args.specification),
        },
        "config": {
            "development_splits": DEVELOPMENT_SPLITS,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "bond_dimensions": list(MPS_BOND_DIMENSIONS),
            "mps_threshold": MPS_THRESHOLD,
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "performance_gate_required_splits": 4,
        },
        "splits": split_results,
        "aggregate_gate": {
            "all_five_mps_probes_converged": all_converged,
            "quantum_beats_both_classical_models_splits": int(passing_splits),
            "performance_gate_passed": bool(all_converged and passing_splits >= 4),
            "large_hardware_phase_allowed": bool(large_allowed),
            "large_hardware_claim_tier": claim_tier,
            "rule_if_converged": "strictly beat both linear and RBF on at least four of five splits",
            "rule_if_not_converged": "large hardware may proceed as feasibility-only",
            "rule_if_converged_and_failed": "stop before large hardware",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "MPS chi=512 is a numerical reference, not an exact ideal result. The "
            "local screen is a predeclared selection gate and uses no provider calls."
        ),
    }
    _atomic_write_json(args.output, report)
    return report


def _hardware_rows(result: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = result.get("hardware_feature_rows")
    if not isinstance(rows, list) or not rows:
        raise ModulePipelineError("Hardware result contains no feature rows")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["base_circuit_index"]))
    features = np.asarray([row["features"] for row in ordered], dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != EXPECTED_OBSERVABLES
        or not np.all(np.isfinite(features))
        or np.max(np.abs(features)) > 1.0 + 1e-9
    ):
        raise ModulePipelineError("Hardware features violate the frozen 627-feature panel")
    return features, ordered


def analyze_hardware(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise ModulePipelineError(f"Refusing to overwrite existing artifact: {args.output}")
    specification = load_prepared_specification(args.specification)
    hardware = _load_json(args.hardware_result, label="Fire Opal hardware result")
    if hardware.get("status") != "retrieved_and_structurally_validated":
        raise ModulePipelineError("Hardware result is not structurally validated")
    phase = str(hardware.get("phase"))
    split_id = "seed11_sentinel" if phase == "sentinel" else "final_blind" if phase == "large" else ""
    if not split_id:
        raise ModulePipelineError("Hardware phase must be sentinel or large")
    data = rebuild_split(specification, split_id=split_id, cache_dir=args.cache_dir)
    features, rows = _hardware_rows(hardware)
    expected_rows = SENTINEL_TRAIN_SAMPLES + SENTINEL_TEST_SAMPLES if phase == "sentinel" else TRAIN_SAMPLES + TEST_SAMPLES
    train_samples = SENTINEL_TRAIN_SAMPLES if phase == "sentinel" else TRAIN_SAMPLES
    if len(features) != expected_rows:
        raise ModulePipelineError("Hardware feature row count differs from the frozen phase")
    expected_indices = np.concatenate((data["train_indices"], data["test_indices"]))
    actual_indices = np.asarray([int(row["source_row_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise ModulePipelineError("Hardware feature order differs from the frozen split")
    evaluation = evaluate_feature_split(
        features[:train_samples],
        features[train_samples:],
        data["train_classical"],
        data["test_classical"],
        data["y_train"],
        data["y_test"],
    )
    quantum_predictions = np.asarray(
        evaluation["quantum"]["fixed_test"]["test_predictions"], dtype=np.int64
    )
    classical_family = max(
        evaluation["classical"],
        key=lambda family: evaluation["classical"][family]["fixed_test"]["test_balanced_accuracy"],
    )
    classical_predictions = np.asarray(
        evaluation["classical"][classical_family]["fixed_test"]["test_predictions"],
        dtype=np.int64,
    )
    paired = q40_analysis.paired_test_statistics(
        data["y_test"],
        quantum_predictions,
        classical_predictions,
        replicates=10_000,
        seed=6111,
    )
    candidate = bool(phase == "large" and evaluation["quantum_beats_linear_and_rbf"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": HARDWARE_ANALYSIS_KIND,
        "completed": True,
        "captured_at_utc": _utc_now(),
        "execution_attempted": False,
        "additional_quantum_seconds_used": 0,
        "provider_calls": [],
        "phase": phase,
        "specification": {
            "path": str(args.specification.resolve()),
            "sha256": _sha256_file(args.specification),
        },
        "hardware_result": {
            "path": str(args.hardware_result.resolve()),
            "sha256": _sha256_file(args.hardware_result),
            "action_ids": hardware.get("action_ids", []),
            "provider_reported_quantum_seconds": hardware.get(
                "provider_reported_quantum_seconds"
            ),
        },
        "evaluation": evaluation,
        "matched_classical_family": classical_family,
        "paired_statistics": paired,
        "claim_gate": {
            "empirical_hardware_advantage_candidate": candidate,
            "requires_large_unseen_256_cell_test": True,
            "hardware_strictly_beats_linear_and_rbf": evaluation[
                "quantum_beats_linear_and_rbf"
            ],
            "formal_or_asymptotic_quantum_advantage": False,
        },
        "claim_boundary": (
            "A positive large-split result is a task-bound empirical hardware-advantage "
            "candidate only. It is not a general or asymptotic quantum-advantage proof."
        ),
    }
    _atomic_write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze data, modules, and splits")
    prepare.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    prepare.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    prepare.add_argument("--output", type=Path, default=DEFAULT_SPECIFICATION)
    prepare.add_argument("--force", action="store_true")
    screen = subparsers.add_parser("local-screen", help="run five-split MPS selection gate")
    screen.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    screen.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    screen.add_argument("--output", type=Path, default=DEFAULT_LOCAL_SCREEN)
    screen.add_argument("--force", action="store_true")
    analysis = subparsers.add_parser("hardware-analysis", help="analyze a retrieved frozen hardware result")
    analysis.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    analysis.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    analysis.add_argument("--hardware-result", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_study(args)
        print("PBMC68k q60 coexpression-module study frozen")
        print(f"- modules: {report['config']['module_count']}")
        print(f"- disjoint allocated rows: {report['splits']['allocated_unique_rows']}")
        print("- provider calls: 0")
        print(f"- report: {args.output}")
        return 0
    if args.command == "local-screen":
        report = run_local_screen(args)
        print("PBMC68k q60 five-split local screen complete")
        print(f"- status: {report['status']}")
        print(
            "- quantum beats both classical models: "
            f"{report['aggregate_gate']['quantum_beats_both_classical_models_splits']}/5"
        )
        print("- provider calls: 0")
        print(f"- report: {args.output}")
        return 0
    report = analyze_hardware(args)
    print("PBMC68k q60 frozen hardware analysis complete")
    print(f"- phase: {report['phase']}")
    print(
        "- empirical hardware-advantage candidate: "
        f"{report['claim_gate']['empirical_hardware_advantage_candidate']}"
    )
    print("- additional quantum seconds: 0")
    print(f"- report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
