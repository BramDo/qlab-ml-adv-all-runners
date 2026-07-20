#!/usr/bin/env python3
"""Local-only analysis of the frozen q40 sqrt(q), B=4 hardware pilot.

Model selection uses four-fold stratified cross-validation on the 32 training
rows only.  The 32 test rows are evaluated once after the hardware model and a
matched classical frontier have each been frozen.  This module has no provider
authentication, submission, retrieval, or hardware-execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from scipy.stats import binomtest
import sklearn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_validate as frozen
import qiskit_qos_pbmc68k_utils as pbmc


SCHEMA_VERSION = "1.0"
KIND = "pbmc68k_q40_sqrtq_b4_seed11_hardware_train_only_analysis"
ACTION_ID = "2334162"
CV_SEED = 6011
CV_FOLDS = 4
BOOTSTRAP_SEED = 6111
DEFAULT_BOOTSTRAP_REPLICATES = 10_000

ARTIFACT_DIR = Path("fire_opal_pbmc68k_q40_sqrtq_b4")
DEFAULT_RESULT = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_hardware_result.json"
)
DEFAULT_PLAN = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_fireopal_hardware_plan.json"
)
DEFAULT_OUTPUT = ARTIFACT_DIR / (
    "pbmc68k_q40_sqrtq_b4_seed11_hardware_analysis.json"
)
PINNED_RESULT_SHA256 = (
    "50d2be52b7715f6014af43fa6f1712ab38381289ef555233ec9e01aaf067f030"
)
PINNED_PLAN_SHA256 = (
    "095c95dc049a1fac9c589de466bc598ed4601f3a4d7ff61b5a4100b7421f2f50"
)


class AnalysisError(RuntimeError):
    """Raised when a frozen-input or analysis invariant is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"Required artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Could not read {path} ({type(exc).__name__})") from None
    if not isinstance(value, dict):
        raise AnalysisError(f"Artifact root is not an object: {path}")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_frozen_hardware(
    result_path: Path, plan_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and independently verify the exact retrieved hardware feature rows."""

    result_hash = _sha256_file(result_path)
    plan_hash = _sha256_file(plan_path)
    if result_hash != PINNED_RESULT_SHA256:
        raise AnalysisError("Hardware-result SHA256 differs from the retrieved pin")
    if plan_hash != PINNED_PLAN_SHA256:
        raise AnalysisError("Hardware-plan SHA256 differs from the predeclared pin")
    result = _load_json(result_path)
    plan = _load_json(plan_path)
    distribution = result.get("distribution_validation", {})
    observable = result.get("observable_validation", {})
    pilot = plan.get("pilot", {})
    predeclared = plan.get("predeclared_readout", {})
    rows = result.get("hardware_feature_rows")
    if (
        result.get("kind") != "pbmc68k_q40_sqrtq_b4_fireopal_hardware_result"
        or result.get("status") != "retrieved_and_structurally_validated"
        or str(result.get("action_id")) != ACTION_ID
        or result.get("classifier_analysis_performed") is not False
        or distribution.get("passed") is not True
        or int(distribution.get("circuit_count", -1)) != 192
        or observable.get("passed") is not True
        or int(observable.get("observable_count_per_sample", -1)) != 405
        or not isinstance(rows, list)
        or len(rows) != 64
    ):
        raise AnalysisError("Hardware result does not match the frozen passing result")
    if (
        plan.get("kind")
        != "pbmc68k_q40_sqrtq_b4_fireopal_seed11_hardware_pilot_plan"
        or int(pilot.get("qubits", -1)) != 40
        or int(pilot.get("train_samples", -1)) != 32
        or int(pilot.get("test_samples", -1)) != 32
        or int(pilot.get("shots_per_circuit", -1)) != 128
        or str(pilot.get("backend")) != "ibm_fez"
        or str(predeclared.get("model_selection"))
        != "four-fold stratified training-only CV, seed 6011"
        or predeclared.get(
            "classical_frontier_must_be_recomputed_on_identical_32/32_split"
        )
        is not True
    ):
        raise AnalysisError("Hardware plan no longer matches the predeclared analysis")

    expected_splits = ["train"] * 32 + ["test"] * 32
    for index, (row, split) in enumerate(zip(rows, expected_splits, strict=True)):
        if (
            not isinstance(row, Mapping)
            or int(row.get("base_circuit_index", -1)) != index
            or str(row.get("split")) != split
            or int(row.get("sample_position", -1)) != index % 32
            or len(row.get("features", [])) != 405
            or float(row.get("label_for_matched_analysis", 0.0)) not in {-1.0, 1.0}
        ):
            raise AnalysisError("Hardware feature-row order or shape changed")
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray(
        [row["label_for_matched_analysis"] for row in rows], dtype=np.int64
    )
    source_indices = np.asarray(
        [row["source_row_index"] for row in rows], dtype=np.int64
    )
    if (
        not np.all(np.isfinite(features))
        or float(np.max(np.abs(features))) > 1.0 + 1e-9
        or len(np.unique(source_indices)) != 64
        or not all(np.array_equal(np.sort(labels[start : start + 32]), [-1] * 16 + [1] * 16)
                   for start in (0, 32))
    ):
        raise AnalysisError("Hardware feature values, labels, or source rows are invalid")
    return (
        features[:32],
        features[32:],
        labels[:32],
        labels[32:],
        {
            "result_path": str(result_path.resolve()),
            "result_sha256": result_hash,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": plan_hash,
            "action_id": ACTION_ID,
            "backend": "ibm_fez",
            "shots_per_circuit": 128,
            "measured_circuits": 192,
            "total_shots": 24_576,
            "train_source_indices": source_indices[:32].tolist(),
            "test_source_indices": source_indices[32:].tolist(),
            "hardware_features_sha256": _sha256_array(features),
        },
    )


def _library_log1p(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Per-cell 10k normalization and log1p; each row is transformed alone."""

    transformed = matrix.astype(np.float64, copy=True).tocsr()
    totals = np.asarray(transformed.sum(axis=1)).ravel()
    scales = np.divide(
        10_000.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0.0,
    )
    transformed = sparse.diags(scales).dot(transformed).tocsr()
    transformed.data = np.log1p(transformed.data)
    if transformed.data.size and not np.all(np.isfinite(transformed.data)):
        raise AnalysisError("Classical log-normalized matrix is non-finite")
    return transformed


def load_matched_classical(
    cache_dir: Path, expected_metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Rebuild raw-gene and exact B=4 classical inputs for the identical rows."""

    screen, source = frozen.load_frozen_specifications(
        frozen.DEFAULT_SCREEN_REPORT, frozen.DEFAULT_SOURCE_REPORT
    )
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    config = screen["configuration"]
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(config["positive_label"]),
        negative_label=str(config["negative_label"]),
    )
    if (
        frozen.coherent.flat_gate._sha256_sparse(x_pair)
        != str(screen["dataset"]["sparse_matrix_sha256"])
        or frozen.coherent.flat_gate._sha256_array(y_pair)
        != str(screen["dataset"]["labels_sha256"])
    ):
        raise AnalysisError("Cached PBMC68k pair differs from the frozen dataset")
    train_indices = np.asarray(source["split"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(source["split"]["test_indices"], dtype=np.int64)
    if (
        train_indices.tolist() != list(expected_metadata["train_source_indices"])
        or test_indices.tolist() != list(expected_metadata["test_source_indices"])
    ):
        raise AnalysisError("Classical rows differ from the hardware source rows")
    raw_selected = _library_log1p(x_pair[np.concatenate((train_indices, test_indices))])
    encoding = {
        "num_qubits": 40,
        "block_count": 4,
        "hash_seed": int(config["pbmc_hash_seed"]),
        "value_mode": "log-product",
        "max_active_genes": int(config["pbmc_max_active_genes"]),
    }
    blocks, block_stats = frozen.coherent.build_coherent_blocks(
        x_pair[np.concatenate((train_indices, test_indices))], **encoding
    )
    flattened = np.asarray(blocks.reshape(64, 160), dtype=np.float64)
    metadata = {
        "dataset": {**source_meta, **pair_meta},
        "sparse_matrix_sha256": str(screen["dataset"]["sparse_matrix_sha256"]),
        "labels_sha256": str(screen["dataset"]["labels_sha256"]),
        "raw_preprocessing": "per-cell library-size normalization to 10000 then log1p",
        "raw_train_shape": [32, int(raw_selected.shape[1])],
        "raw_test_shape": [32, int(raw_selected.shape[1])],
        "same_encoding": encoding,
        "same_encoding_statistics": block_stats,
        "same_encoding_features_sha256": _sha256_array(flattened),
    }
    representations = {
        "raw_gene_log1p": {
            "train": raw_selected[:32],
            "test": raw_selected[32:],
            "sparse": True,
        },
        "same_b4_pairhash_160": {
            "train": flattened[:32],
            "test": flattened[32:],
            "sparse": False,
        },
    }
    return representations, metadata, y_pair[train_indices], y_pair[test_indices]


def _candidate_grid(representations: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in representations:
        dense = representation != "raw_gene_log1p"
        if dense:
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                rows.append(
                    {"representation": representation, "family": "ridge", "alpha": alpha}
                )
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            rows.append(
                {"representation": representation, "family": "linear_svc", "C": c_value}
            )
            rows.append(
                {"representation": representation, "family": "logistic_l2", "C": c_value}
            )
        if dense:
            for c_value in (0.1, 1.0, 10.0):
                for gamma in ("scale", 0.01, 0.1):
                    rows.append(
                        {
                            "representation": representation,
                            "family": "rbf_svc",
                            "C": c_value,
                            "gamma": gamma,
                        }
                    )
        else:
            for c_value in (0.1, 1.0, 10.0):
                rows.append(
                    {
                        "representation": representation,
                        "family": "rbf_svc",
                        "C": c_value,
                        "gamma": "scale",
                    }
                )
    for index, row in enumerate(rows):
        row["candidate_id"] = f"c{index:03d}"
    return rows


def _build_model(candidate: Mapping[str, Any], *, seed: int):
    family = str(candidate["family"])
    sparse_input = str(candidate["representation"]) == "raw_gene_log1p"
    if family == "ridge":
        estimator = RidgeClassifier(alpha=float(candidate["alpha"]))
    elif family == "linear_svc":
        estimator = LinearSVC(
            C=float(candidate["C"]), dual="auto", max_iter=50_000, random_state=seed
        )
    elif family == "logistic_l2":
        estimator = LogisticRegression(
            C=float(candidate["C"]),
            solver="liblinear",
            max_iter=50_000,
            random_state=seed,
        )
    elif family == "rbf_svc":
        estimator = SVC(
            C=float(candidate["C"]), kernel="rbf", gamma=candidate["gamma"]
        )
    else:
        raise AnalysisError(f"Unknown model family: {family}")
    return estimator if sparse_input else make_pipeline(StandardScaler(), estimator)


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    family_rank = {"ridge": 0, "logistic_l2": 1, "linear_svc": 2, "rbf_svc": 3}
    family = str(row["family"])
    if family == "ridge":
        flexibility = 1.0 / float(row["alpha"])
    else:
        flexibility = float(row.get("C", 1.0))
        if family == "rbf_svc":
            gamma = row.get("gamma")
            flexibility *= 1.0 if gamma == "scale" else 1.0 + float(gamma)
    return (
        -float(row["cv_mean_balanced_accuracy"]),
        -float(row["cv_worst_balanced_accuracy"]),
        float(row["cv_std_balanced_accuracy"]),
        family_rank[family],
        flexibility,
        str(row["representation"]),
        str(row["candidate_id"]),
    )


def select_model_training_only(
    training_representations: Mapping[str, Any],
    y_train: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    *,
    cv_seed: int = CV_SEED,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Choose a candidate using training rows only; no test argument exists."""

    labels = np.asarray(y_train, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        representation = str(candidate["representation"])
        matrix = training_representations[representation]
        fold_scores: list[float] = []
        for train_fold, validation_fold in splits:
            model = _build_model(candidate, seed=cv_seed)
            model.fit(matrix[train_fold], labels[train_fold])
            predictions = model.predict(matrix[validation_fold])
            fold_scores.append(
                float(balanced_accuracy_score(labels[validation_fold], predictions))
            )
        rows.append(
            {
                **dict(candidate),
                "fold_balanced_accuracy": fold_scores,
                "cv_mean_balanced_accuracy": float(np.mean(fold_scores)),
                "cv_worst_balanced_accuracy": float(np.min(fold_scores)),
                "cv_std_balanced_accuracy": float(np.std(fold_scores)),
            }
        )
    chosen = min(rows, key=_selection_key)
    return {
        "selection_scope": "training rows only; test features and labels unavailable to this function",
        "cv": "StratifiedKFold",
        "cv_folds": cv_folds,
        "cv_seed": cv_seed,
        "candidate_count": len(rows),
        "tie_break": "mean desc, worst-fold desc, std asc, simpler family/hyperparameters",
        "chosen": chosen,
        "candidates": rows,
    }


def _decision_scores(model: Any, matrix: Any) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix), dtype=np.float64).ravel()
    return np.asarray(model.predict(matrix), dtype=np.float64).ravel()


def fit_frozen_candidate(
    chosen: Mapping[str, Any],
    training_representations: Mapping[str, Any],
    test_representations: Mapping[str, Any],
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = CV_SEED,
) -> dict[str, Any]:
    representation = str(chosen["representation"])
    model = _build_model(chosen, seed=seed)
    model.fit(training_representations[representation], y_train)
    train_predictions = np.asarray(
        model.predict(training_representations[representation]), dtype=np.int64
    )
    test_predictions = np.asarray(
        model.predict(test_representations[representation]), dtype=np.int64
    )
    return {
        "frozen_candidate": {
            key: value
            for key, value in chosen.items()
            if key not in {"fold_balanced_accuracy"}
        },
        "train_balanced_accuracy": float(
            balanced_accuracy_score(y_train, train_predictions)
        ),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(y_test, test_predictions)
        ),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_correct": int(np.sum(test_predictions == y_test)),
        "test_samples": int(len(y_test)),
        "test_predictions": test_predictions.tolist(),
        "test_decision_scores": _decision_scores(
            model, test_representations[representation]
        ).tolist(),
    }


def _wilson_interval(correct: int, samples: int, z: float = 1.95996398454) -> list[float]:
    proportion = correct / samples
    denominator = 1.0 + z**2 / samples
    center = (proportion + z**2 / (2.0 * samples)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + z**2 / (4.0 * samples**2)
        )
        / denominator
    )
    return [float(center - margin), float(center + margin)]


def paired_test_statistics(
    labels: np.ndarray,
    hardware_predictions: np.ndarray,
    classical_predictions: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    hardware_correct = np.asarray(hardware_predictions) == labels
    classical_correct = np.asarray(classical_predictions) == labels
    hardware_only = int(np.sum(hardware_correct & ~classical_correct))
    classical_only = int(np.sum(~hardware_correct & classical_correct))
    discordant = hardware_only + classical_only
    mcnemar_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(hardware_only, classical_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    classes = [np.flatnonzero(labels == value) for value in (-1, 1)]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in classes]
        )
        deltas[index] = float(
            np.mean(hardware_correct[sampled]) - np.mean(classical_correct[sampled])
        )
    return {
        "paired_discordance": {
            "hardware_only_correct": hardware_only,
            "classical_only_correct": classical_only,
            "both_or_neither": int(len(labels) - discordant),
            "mcnemar_exact_two_sided_p": mcnemar_p,
        },
        "stratified_paired_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "mean_hardware_minus_classical": float(np.mean(deltas)),
            "ci95_percentile": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "fraction_hardware_better": float(np.mean(deltas > 0.0)),
            "fraction_tied": float(np.mean(deltas == 0.0)),
        },
        "limitation": (
            "Resampling covers finite test-cell uncertainty only; it does not include "
            "independent hardware reruns, shot-noise reruns, calibration drift, or split variation."
        ),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    x_hw_train, x_hw_test, y_train, y_test, hardware_meta = load_frozen_hardware(
        args.result, args.plan
    )
    classical_representations, classical_meta, y_class_train, y_class_test = (
        load_matched_classical(args.cache_dir, hardware_meta)
    )
    if not np.array_equal(y_train, y_class_train) or not np.array_equal(
        y_test, y_class_test
    ):
        raise AnalysisError("Hardware and classical labels differ")

    hardware_train = {"hardware_pauli_405": x_hw_train}
    hardware_test = {"hardware_pauli_405": x_hw_test}
    hardware_selection = select_model_training_only(
        hardware_train, y_train, _candidate_grid(tuple(hardware_train))
    )
    hardware_final = fit_frozen_candidate(
        hardware_selection["chosen"],
        hardware_train,
        hardware_test,
        y_train,
        y_test,
    )

    classical_train = {
        name: value["train"] for name, value in classical_representations.items()
    }
    classical_test = {
        name: value["test"] for name, value in classical_representations.items()
    }
    classical_selection = select_model_training_only(
        classical_train, y_train, _candidate_grid(tuple(classical_train))
    )
    classical_final = fit_frozen_candidate(
        classical_selection["chosen"],
        classical_train,
        classical_test,
        y_train,
        y_test,
    )

    hardware_predictions = np.asarray(
        hardware_final["test_predictions"], dtype=np.int64
    )
    classical_predictions = np.asarray(
        classical_final["test_predictions"], dtype=np.int64
    )
    paired = paired_test_statistics(
        y_test,
        hardware_predictions,
        classical_predictions,
        replicates=int(args.bootstrap_replicates),
        seed=BOOTSTRAP_SEED,
    )
    hardware_ba = float(hardware_final["test_balanced_accuracy"])
    classical_ba = float(classical_final["test_balanced_accuracy"])
    delta = hardware_ba - classical_ba
    ci_low = float(paired["stratified_paired_bootstrap"]["ci95_percentile"][0])
    p_value = float(paired["paired_discordance"]["mcnemar_exact_two_sided_p"])
    if delta <= 0.0:
        verdict = "hardware_does_not_beat_matched_classical_frontier_on_this_split"
    elif ci_low <= 0.0 or p_value >= 0.05:
        verdict = "exploratory_hardware_lead_not_statistically_resolved"
    else:
        verdict = "paired_predictive_signal_only_not_quantum_advantage"

    hardware_final["test_accuracy_wilson_95"] = _wilson_interval(
        int(hardware_final["test_correct"]), 32
    )
    classical_final["test_accuracy_wilson_95"] = _wilson_interval(
        int(classical_final["test_correct"]), 32
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "provider_calls_made": 0,
        "execution_attempted": False,
        "additional_qpu_seconds": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_hardware_input": hardware_meta,
        "matched_classical_input": classical_meta,
        "protocol": {
            "primary_metric": "fixed-test balanced accuracy",
            "model_selection": "four-fold stratified training-only CV",
            "cv_seed": CV_SEED,
            "test_use": "once, after each route's model selection was frozen",
            "same_32_32_split": True,
            "hardware_feature_count": 405,
            "classical_frontier_recomputed": True,
        },
        "hardware_route": {
            "training_only_selection": hardware_selection,
            "fixed_test": hardware_final,
        },
        "matched_classical_frontier": {
            "representations": ["raw_gene_log1p", "same_b4_pairhash_160"],
            "training_only_selection": classical_selection,
            "fixed_test": classical_final,
        },
        "comparison": {
            "hardware_test_balanced_accuracy": hardware_ba,
            "classical_test_balanced_accuracy": classical_ba,
            "hardware_minus_classical": delta,
            "paired_statistics": paired,
            "verdict": verdict,
        },
        "claim_boundary": (
            "This is one predeclared 32/32 hardware pilot. Even a positive paired "
            "classifier result would not establish computational quantum advantage: "
            "the test set is small, the split is single, the classical frontier is "
            "finite, and independent hardware/shot/calibration repetitions are absent."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_replicates < 1000:
        raise SystemExit("--bootstrap-replicates must be at least 1000")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    try:
        report = run_analysis(args)
        _atomic_write_json(args.output, report)
    except (AnalysisError, frozen.RunnerError, ValueError, KeyError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from None
    comparison = report["comparison"]
    hardware = report["hardware_route"]["fixed_test"]
    classical = report["matched_classical_frontier"]["fixed_test"]
    print(f"Saved: {args.output}")
    print(
        f"Hardware: {hardware['test_balanced_accuracy']:.5f} "
        f"({hardware['test_correct']}/32)"
    )
    print(
        f"Classical: {classical['test_balanced_accuracy']:.5f} "
        f"({classical['test_correct']}/32)"
    )
    print(f"Delta: {comparison['hardware_minus_classical']:+.5f}")
    print(f"Verdict: {comparison['verdict']}")
    print("Additional QPU seconds: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
