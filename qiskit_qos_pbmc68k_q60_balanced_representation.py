#!/usr/bin/env python3
"""Repair the shallow q60 representation with fixed train-only quotas.

The previously frozen scales are retained.  Observable selection is changed
from an unconstrained top-k ranking to a pre-registered 8/8/8 representation:
single-qubit Z baselines, pair-scale-sensitive local X/Y observables, and
pair-scale-sensitive multiqubit observables.  Sensitivity and label ranking are
computed on training inputs only.

The representation is first checked on the old seeds' training folds.  A new,
fully disjoint confirmation split is opened only if the predeclared CV and
structural gates pass.  This module is local-only and has no provider path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_TUNING_REPORTS = freeze.DEFAULT_REPORTS
DEFAULT_PRIOR_CONFIRMATION = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_cross_seed_frozen_confirmation.json"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_representation.json"
)
DEFAULT_Z_QUOTA = 8
DEFAULT_TRANSVERSE_QUOTA = 8
DEFAULT_MULTIQUBIT_QUOTA = 8
DEFAULT_SENSITIVITY_THRESHOLD = 1e-10
DEFAULT_CV_MEAN_GATE = 0.55
DEFAULT_CV_WORST_SEED_GATE = 0.45
DEFAULT_FRESH_SEED_START = 25
DEFAULT_SEED_SCAN = 200
DEFAULT_CONFIRMATION_SAMPLES = 48

RunnerError = q60.RunnerError


def _basis_label(mapping: Mapping[int, str]) -> str:
    return "+".join(sorted(set(str(value) for value in mapping.values())))


def _rank_category(
    selection_scores: np.ndarray,
    candidates: Sequence[int],
    *,
    count: int,
    category: str,
) -> np.ndarray:
    candidates_array = np.asarray(candidates, dtype=np.int64)
    if count < 0:
        raise RunnerError(f"Negative quota for {category}")
    if len(candidates_array) < count:
        raise RunnerError(
            f"Only {len(candidates_array)} candidates available for {category}; "
            f"quota is {count}"
        )
    order = np.lexsort(
        (candidates_array, -selection_scores[candidates_array])
    )
    return candidates_array[order[:count]]


def select_balanced_train_only_features(
    query_train: np.ndarray,
    pair_zero_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    shot_intent: int,
    sensitivity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select fixed observable families using training inputs and labels only."""

    query_train = np.asarray(query_train, dtype=np.float64)
    pair_zero_train = np.asarray(pair_zero_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if query_train.shape != pair_zero_train.shape:
        raise RunnerError("Original and pair-zero training features differ in shape")
    if query_train.ndim != 2 or len(query_train) != len(y_train):
        raise RunnerError("Training feature matrix and labels are inconsistent")
    if query_train.shape[1] != len(mappings):
        raise RunnerError("Feature count and observable mappings are inconsistent")
    if sensitivity_threshold < 0.0:
        raise RunnerError("Sensitivity threshold cannot be negative")
    total_quota = int(z_quota + transverse_quota + multiqubit_quota)
    if total_quota < 1:
        raise RunnerError("At least one observable must be selected")

    _, selection_scores = q60.select_train_only_features(
        query_train,
        y_train,
        count=len(mappings),
        shot_intent=int(shot_intent),
    )
    pair_deltas = np.max(np.abs(query_train - pair_zero_train), axis=0)
    sensitive = pair_deltas > float(sensitivity_threshold)
    z_candidates: list[int] = []
    transverse_candidates: list[int] = []
    multiqubit_candidates: list[int] = []
    for index, mapping in enumerate(mappings):
        support = len(mapping)
        paulis = set(str(value) for value in mapping.values())
        if support == 1 and paulis == {"Z"}:
            z_candidates.append(index)
        elif support == 1 and paulis.issubset({"X", "Y"}) and sensitive[index]:
            transverse_candidates.append(index)
        elif support > 1 and sensitive[index]:
            multiqubit_candidates.append(index)

    selected_z = _rank_category(
        selection_scores, z_candidates, count=z_quota, category="single Z"
    )
    selected_transverse = _rank_category(
        selection_scores,
        transverse_candidates,
        count=transverse_quota,
        category="pair-sensitive local X/Y",
    )
    selected_multiqubit = _rank_category(
        selection_scores,
        multiqubit_candidates,
        count=multiqubit_quota,
        category="pair-sensitive multiqubit",
    )
    selected = np.concatenate(
        [selected_z, selected_transverse, selected_multiqubit]
    ).astype(np.int64)
    if len(set(int(value) for value in selected)) != total_quota:
        raise RunnerError("Balanced observable categories overlap")
    selected_bases = [_basis_label(mappings[int(index)]) for index in selected]
    return selected, selection_scores, {
        "selection_scope": "training_inputs_and_labels_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "sensitivity_reference": "training features at frozen pair scale versus pair_scale=0",
        "sensitivity_threshold": float(sensitivity_threshold),
        "quotas": {
            "single_z": int(z_quota),
            "pair_sensitive_local_xy": int(transverse_quota),
            "pair_sensitive_multiqubit": int(multiqubit_quota),
        },
        "candidate_counts": {
            "single_z": len(z_candidates),
            "pair_sensitive_local_xy": len(transverse_candidates),
            "pair_sensitive_multiqubit": len(multiqubit_candidates),
        },
        "selected_counts": {
            "single_z": len(selected_z),
            "pair_sensitive_local_xy": len(selected_transverse),
            "pair_sensitive_multiqubit": len(selected_multiqubit),
        },
        "selected_pair_sensitive_count": int(np.sum(sensitive[selected])),
        "selected_pair_sensitive_fraction": float(np.mean(sensitive[selected])),
        "selected_measurement_basis_counts": dict(
            sorted(Counter(selected_bases).items())
        ),
        "selected_indices_by_category": {
            "single_z": [int(value) for value in selected_z],
            "pair_sensitive_local_xy": [
                int(value) for value in selected_transverse
            ],
            "pair_sensitive_multiqubit": [
                int(value) for value in selected_multiqubit
            ],
        },
        "selected_pair_scale_max_abs_deltas": [
            float(pair_deltas[int(value)]) for value in selected
        ],
    }


def _feature_matrices(
    encoded_train: np.ndarray,
    encoded_test: np.ndarray | None,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    train_parameters = [q60.query_parameters(row) for row in encoded_train]
    test_parameters = (
        []
        if encoded_test is None
        else [q60.query_parameters(row) for row in encoded_test]
    )
    common = {
        "single_scale": float(configuration["single_scale"]),
        "phase_scale": float(configuration["phase_scale"]),
    }
    train_features = tuner.scaled_feature_matrix(
        train_parameters,
        mappings,
        pair_scale=float(configuration["pair_scale"]),
        **common,
    )
    train_pair_zero = tuner.scaled_feature_matrix(
        train_parameters, mappings, pair_scale=0.0, **common
    )
    result = {
        "train": train_features,
        "train_pair_zero": train_pair_zero,
    }
    if encoded_test is not None:
        result["test"] = tuner.scaled_feature_matrix(
            test_parameters,
            mappings,
            pair_scale=float(configuration["pair_scale"]),
            **common,
        )
        result["test_pair_zero"] = tuner.scaled_feature_matrix(
            test_parameters, mappings, pair_scale=0.0, **common
        )
    return result


def _sketch_features(
    encoded_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    pair_scale: float | None = None,
) -> np.ndarray:
    sketch_linear, sketch_pair = q60.sketch_parameters(encoded_fit, y_fit)
    return tuner.scaled_feature_matrix(
        [(sketch_linear, sketch_pair)],
        mappings,
        single_scale=float(configuration["single_scale"]),
        phase_scale=float(configuration["phase_scale"]),
        pair_scale=(
            float(configuration["pair_scale"])
            if pair_scale is None
            else float(pair_scale)
        ),
    )[0]


def _fit_selected_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    sketch_features: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = sketch_features[selected]
    head_train = q60._head_features(model, train_features[:, selected])
    head_test = q60._head_features(model, test_features[:, selected])
    return q60._ridge_scores(head_train, head_test, y_train)


def cross_validate_representation(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    cv_folds: int,
    seed: int,
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Compare balanced and legacy selectors using one training split only."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    features = _feature_matrices(
        encoded_train, None, mappings=mappings, configuration=configuration
    )
    splitter = StratifiedKFold(
        n_splits=int(cv_folds), shuffle=True, random_state=int(seed)
    )
    folds: list[dict[str, Any]] = []
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(encoded_train, y_train > 0.0)
    ):
        fit_indices = fit_indices.astype(np.int64)
        validation_indices = validation_indices.astype(np.int64)
        selected, _, audit = select_balanced_train_only_features(
            features["train"][fit_indices],
            features["train_pair_zero"][fit_indices],
            y_train[fit_indices],
            mappings=mappings,
            z_quota=z_quota,
            transverse_quota=transverse_quota,
            multiqubit_quota=multiqubit_quota,
            shot_intent=shot_intent,
            sensitivity_threshold=sensitivity_threshold,
        )
        legacy_selected, _ = q60.select_train_only_features(
            features["train"][fit_indices],
            y_train[fit_indices],
            count=int(configuration["selected_feature_count"]),
            shot_intent=int(shot_intent),
        )
        sketch = _sketch_features(
            encoded_train[fit_indices],
            y_train[fit_indices],
            mappings=mappings,
            configuration=configuration,
        )
        _, balanced_scores = _fit_selected_scores(
            features["train"][fit_indices],
            features["train"][validation_indices],
            y_train[fit_indices],
            sketch,
            selected,
        )
        _, legacy_scores = _fit_selected_scores(
            features["train"][fit_indices],
            features["train"][validation_indices],
            y_train[fit_indices],
            sketch,
            legacy_selected,
        )
        pair_deltas = np.max(
            np.abs(
                features["train"][fit_indices]
                - features["train_pair_zero"][fit_indices]
            ),
            axis=0,
        )
        folds.append(
            {
                "fold_index": int(fold_index),
                "fit_samples": len(fit_indices),
                "validation_samples": len(validation_indices),
                "balanced_accuracy": q60._balanced_accuracy(
                    y_train[validation_indices], balanced_scores
                ),
                "legacy_balanced_accuracy": q60._balanced_accuracy(
                    y_train[validation_indices], legacy_scores
                ),
                "selection_audit": audit,
                "legacy_selected_pair_sensitive_count": int(
                    np.sum(pair_deltas[legacy_selected] > sensitivity_threshold)
                ),
                "legacy_selected_indices": [
                    int(value) for value in legacy_selected
                ],
            }
        )
    balanced_scores = [float(row["balanced_accuracy"]) for row in folds]
    legacy_scores = [float(row["legacy_balanced_accuracy"]) for row in folds]
    return {
        "selection_scope": "training_split_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "seed": int(seed),
        "training_samples": len(encoded_train),
        "cv_folds": int(cv_folds),
        "balanced_cv_mean": float(np.mean(balanced_scores)),
        "balanced_cv_worst_fold": float(np.min(balanced_scores)),
        "balanced_cv_std": float(np.std(balanced_scores)),
        "legacy_cv_mean": float(np.mean(legacy_scores)),
        "legacy_cv_worst_fold": float(np.min(legacy_scores)),
        "legacy_cv_std": float(np.std(legacy_scores)),
        "balanced_minus_legacy_cv_mean": float(
            np.mean(balanced_scores) - np.mean(legacy_scores)
        ),
        "folds": folds,
    }


def _load_source_training_data(
    report_paths: Sequence[Path], *, cache_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any, np.ndarray]:
    raw_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    shared_config = freeze._shared_source_configuration(raw_reports)
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(shared_config["positive_label"]),
        negative_label=str(shared_config["negative_label"]),
    )
    runs: list[dict[str, Any]] = []
    for path, report in zip(report_paths, raw_reports, strict=True):
        tuning = report["train_only_tuning"]
        if tuning["test_inputs_seen"] is not False or tuning["test_labels_seen"] is not False:
            raise RunnerError(f"Source tuning was not train-only: {path}")
        config = report["config"]
        train_indices = np.asarray(report["split"]["train_indices"], dtype=np.int64)
        encoded_train, encoding_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_pair[train_indices],
            feature_dim=int(config["qubits"]),
            hash_seed=int(config["seed"]),
            value_mode=str(config["value_mode"]),
            max_active_genes=int(config["max_active_genes"]),
        )
        actual_hash = q60.q40_validate._array_sha256(encoded_train)
        if actual_hash != str(report["split"]["encoded_train_sha256"]):
            raise RunnerError(f"Training hash mismatch for {path}")
        runs.append(
            {
                "seed": int(config["seed"]),
                "encoded_train": encoded_train,
                "y_train": y_pair[train_indices].astype(np.float64),
                "train_indices": train_indices,
                "encoding_stats": encoding_stats,
                "report_path": str(path.resolve()),
                "report_sha256": freeze._file_sha256(path),
                "raw_report": report,
            }
        )
    return runs, raw_reports, shared_config, x_pair, y_pair


def _find_legacy_candidate(
    report: Mapping[str, Any], configuration: Mapping[str, Any]
) -> Mapping[str, Any]:
    target = freeze._configuration_key(configuration)
    for row in report["train_only_tuning"]["leaderboard"]:
        if freeze._configuration_key(row) == target:
            return row
    raise RunnerError("Frozen configuration is absent from a source leaderboard")


def _aggregate_cv(
    rows: Sequence[Mapping[str, Any]],
    *,
    cv_mean_gate: float,
    cv_worst_seed_gate: float,
    expected_pair_sensitive: int,
) -> dict[str, Any]:
    balanced = np.asarray([row["balanced_cv_mean"] for row in rows], dtype=float)
    legacy = np.asarray([row["legacy_cv_mean"] for row in rows], dtype=float)
    structural = all(
        fold["selection_audit"]["selected_pair_sensitive_count"]
        == expected_pair_sensitive
        for row in rows
        for fold in row["folds"]
    )
    mean_value = float(np.mean(balanced))
    worst_seed = float(np.min(balanced))
    passed = bool(
        structural
        and mean_value >= float(cv_mean_gate)
        and worst_seed >= float(cv_worst_seed_gate)
    )
    return {
        "equal_seed_balanced_cv_mean": mean_value,
        "worst_seed_balanced_cv_mean": worst_seed,
        "between_seed_balanced_cv_std": float(np.std(balanced)),
        "equal_seed_legacy_cv_mean": float(np.mean(legacy)),
        "balanced_minus_legacy_cv_mean": float(
            np.mean(balanced) - np.mean(legacy)
        ),
        "structural_gate": structural,
        "cv_mean_gate": float(cv_mean_gate),
        "cv_worst_seed_gate": float(cv_worst_seed_gate),
        "passes_confirmation_gate": passed,
    }


def _load_fresh_split(
    args: argparse.Namespace,
    *,
    x_pair: Any,
    y_pair: np.ndarray,
    shared_config: Mapping[str, Any],
    excluded_indices: set[int],
) -> dict[str, Any]:
    chosen: tuple[int, np.ndarray, np.ndarray] | None = None
    for seed in range(
        int(args.fresh_seed_start),
        int(args.fresh_seed_start) + int(args.seed_scan),
    ):
        train_indices, test_indices = genomics_runner.benchmark_indices(
            x_pair.shape[0],
            seed=seed,
            train_fraction=float(args.train_fraction),
            max_train_samples=int(args.max_train_samples),
            max_test_samples=int(args.max_test_samples),
            labels=y_pair,
        )
        proposed = set(int(value) for value in train_indices) | set(
            int(value) for value in test_indices
        )
        if proposed.isdisjoint(excluded_indices):
            chosen = (seed, train_indices, test_indices)
            break
    if chosen is None:
        raise RunnerError("No fully disjoint fresh split found")
    seed, train_indices, test_indices = chosen
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=int(shared_config["qubits"]),
        hash_seed=seed,
        value_mode=str(shared_config["value_mode"]),
        max_active_genes=int(shared_config["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=int(shared_config["qubits"]),
        hash_seed=seed,
        value_mode=str(shared_config["value_mode"]),
        max_active_genes=int(shared_config["max_active_genes"]),
    )
    return {
        "seed": seed,
        "encoded_train": encoded_train,
        "encoded_test": encoded_test,
        "y_train": y_pair[train_indices].astype(np.float64),
        "y_test": y_pair[test_indices].astype(np.float64),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_encoding_stats": train_stats,
        "test_encoding_stats": test_stats,
    }


def evaluate_fresh_representation(
    data: Mapping[str, Any],
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    features = _feature_matrices(
        data["encoded_train"],
        data["encoded_test"],
        mappings=mappings,
        configuration=configuration,
    )
    selected, selection_scores, audit = select_balanced_train_only_features(
        features["train"],
        features["train_pair_zero"],
        data["y_train"],
        mappings=mappings,
        z_quota=z_quota,
        transverse_quota=transverse_quota,
        multiqubit_quota=multiqubit_quota,
        shot_intent=shot_intent,
        sensitivity_threshold=sensitivity_threshold,
    )
    sketch = _sketch_features(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
    )
    balanced_train, balanced_test = _fit_selected_scores(
        features["train"],
        features["test"],
        data["y_train"],
        sketch,
        selected,
    )
    pair_zero_sketch = _sketch_features(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=0.0,
    )
    _, pair_zero_test = _fit_selected_scores(
        features["train_pair_zero"],
        features["test_pair_zero"],
        data["y_train"],
        pair_zero_sketch,
        selected,
    )
    legacy_selected, _ = q60.select_train_only_features(
        features["train"],
        data["y_train"],
        count=int(configuration["selected_feature_count"]),
        shot_intent=int(shot_intent),
    )
    _, legacy_test = _fit_selected_scores(
        features["train"],
        features["test"],
        data["y_train"],
        sketch,
        legacy_selected,
    )
    classical_train, classical_test = q60._ridge_scores(
        data["encoded_train"], data["encoded_test"], data["y_train"]
    )
    seed = int(data["seed"])
    balanced_accuracy = q60._balanced_accuracy(data["y_test"], balanced_test)
    pair_zero_accuracy = q60._balanced_accuracy(data["y_test"], pair_zero_test)
    legacy_accuracy = q60._balanced_accuracy(data["y_test"], legacy_test)
    classical_accuracy = q60._balanced_accuracy(data["y_test"], classical_test)
    return {
        "configuration_fixed_before_fresh_split": True,
        "selection_uses_training_only": True,
        "representation_audit": audit,
        "selected_observables": [
            tuner._mapping_row(int(index), mappings[int(index)], selection_scores[int(index)])
            for index in selected
        ],
        "balanced_representation": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], balanced_train
            ),
            "test_balanced_accuracy": balanced_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], balanced_test, seed=seed + 9300
            ),
        },
        "same_selected_pair_zero_ablation": {
            "post_hoc_diagnostic_only": True,
            "test_balanced_accuracy": pair_zero_accuracy,
            "delta_from_balanced_representation": (
                pair_zero_accuracy - balanced_accuracy
            ),
        },
        "legacy_top24_reference": {
            "test_balanced_accuracy": legacy_accuracy,
            "selected_indices": [int(value) for value in legacy_selected],
        },
        "classical_60bin_reference": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], classical_train
            ),
            "test_balanced_accuracy": classical_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], classical_test, seed=seed + 9301
            ),
        },
        "balanced_minus_legacy_test": balanced_accuracy - legacy_accuracy,
        "balanced_minus_classical_test": balanced_accuracy - classical_accuracy,
        "representation_problem_repaired": bool(
            audit["selected_pair_sensitive_count"]
            == transverse_quota + multiqubit_quota
        ),
        "paper_feasibility_signal": bool(balanced_accuracy >= 0.55),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_paths = [Path(path) for path in args.tuning_reports]
    prior_path = Path(args.prior_confirmation)
    prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
    configuration = freeze._configuration_only(prior_report["selection"]["chosen"])
    runs, raw_reports, shared_config, x_pair, y_pair = _load_source_training_data(
        report_paths, cache_dir=Path(args.cache_dir)
    )
    mappings = toy.pauli_feature_mappings(
        int(shared_config["qubits"]), family="local"
    )
    cv_rows: list[dict[str, Any]] = []
    for run_data in runs:
        cv = cross_validate_representation(
            run_data["encoded_train"],
            run_data["y_train"],
            mappings=mappings,
            configuration=configuration,
            cv_folds=int(shared_config["cv_folds"]),
            seed=int(run_data["seed"]),
            shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
            z_quota=int(args.z_quota),
            transverse_quota=int(args.transverse_quota),
            multiqubit_quota=int(args.multiqubit_quota),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        expected = _find_legacy_candidate(
            run_data["raw_report"], configuration
        )
        if not np.isclose(
            cv["legacy_cv_mean"],
            float(expected["cv_mean_balanced_accuracy"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RunnerError(
                f"Legacy CV reproduction failed for seed {run_data['seed']}"
            )
        cv["legacy_report_metric_reproduced"] = True
        cv["source_report"] = run_data["report_path"]
        cv["source_report_sha256"] = run_data["report_sha256"]
        cv_rows.append(cv)

    expected_sensitive = int(args.transverse_quota + args.multiqubit_quota)
    aggregate = _aggregate_cv(
        cv_rows,
        cv_mean_gate=float(args.cv_mean_gate),
        cv_worst_seed_gate=float(args.cv_worst_seed_gate),
        expected_pair_sensitive=expected_sensitive,
    )
    confirmation: dict[str, Any] = {
        "executed": False,
        "reason": "training-only CV gate did not pass",
    }
    if aggregate["passes_confirmation_gate"]:
        excluded = freeze._excluded_source_indices(raw_reports)
        prior_split = prior_report["confirmation"]["split"]
        excluded.update(int(value) for value in prior_split["train_indices"])
        excluded.update(int(value) for value in prior_split["test_indices"])
        fresh = _load_fresh_split(
            args,
            x_pair=x_pair,
            y_pair=y_pair,
            shared_config=shared_config,
            excluded_indices=excluded,
        )
        evaluation = evaluate_fresh_representation(
            fresh,
            mappings=mappings,
            configuration=configuration,
            shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
            z_quota=int(args.z_quota),
            transverse_quota=int(args.transverse_quota),
            multiqubit_quota=int(args.multiqubit_quota),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        fresh_indices = set(int(value) for value in fresh["train_indices"]) | set(
            int(value) for value in fresh["test_indices"]
        )
        confirmation = {
            "executed": True,
            "actual_seed": int(fresh["seed"]),
            "train_samples": len(fresh["encoded_train"]),
            "test_samples": len(fresh["encoded_test"]),
            "excluded_source_indices": len(excluded),
            "source_index_overlap": len(fresh_indices & excluded),
            "train_test_index_overlap": len(
                set(int(value) for value in fresh["train_indices"])
                & set(int(value) for value in fresh["test_indices"])
            ),
            "split": {
                "train_indices": [int(value) for value in fresh["train_indices"]],
                "test_indices": [int(value) for value in fresh["test_indices"]],
                "encoded_train_sha256": q60.q40_validate._array_sha256(
                    fresh["encoded_train"]
                ),
                "encoded_test_sha256": q60.q40_validate._array_sha256(
                    fresh["encoded_test"]
                ),
                "train_encoding_stats": fresh["train_encoding_stats"],
                "test_encoding_stats": fresh["test_encoding_stats"],
            },
            "evaluation": evaluation,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_balanced_representation",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "pre_registered_change": {
            "frozen_configuration": configuration,
            "only_selection_rule_changed": True,
            "z_quota": int(args.z_quota),
            "transverse_quota": int(args.transverse_quota),
            "multiqubit_quota": int(args.multiqubit_quota),
            "sensitivity_threshold": float(args.sensitivity_threshold),
            "prior_confirmation_report": str(prior_path.resolve()),
            "prior_confirmation_sha256": freeze._file_sha256(prior_path),
        },
        "training_only_cross_seed_validation": {
            "source_test_results_used": False,
            "source_seeds": [int(run["seed"]) for run in runs],
            "per_seed": cv_rows,
            "aggregate": aggregate,
        },
        "fresh_confirmation": confirmation,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "The observable quotas were fixed before the fresh split was opened. "
            "Source test results were not used in cross-validation. Any fresh "
            "result remains an exact local causal-cone model diagnostic, not "
            "hardware evidence or evidence of quantum advantage."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-reports",
        type=Path,
        nargs="+",
        default=DEFAULT_TUNING_REPORTS,
    )
    parser.add_argument(
        "--prior-confirmation", type=Path, default=DEFAULT_PRIOR_CONFIRMATION
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--z-quota", type=int, default=DEFAULT_Z_QUOTA)
    parser.add_argument(
        "--transverse-quota", type=int, default=DEFAULT_TRANSVERSE_QUOTA
    )
    parser.add_argument(
        "--multiqubit-quota", type=int, default=DEFAULT_MULTIQUBIT_QUOTA
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=DEFAULT_SENSITIVITY_THRESHOLD,
    )
    parser.add_argument("--cv-mean-gate", type=float, default=DEFAULT_CV_MEAN_GATE)
    parser.add_argument(
        "--cv-worst-seed-gate", type=float, default=DEFAULT_CV_WORST_SEED_GATE
    )
    parser.add_argument(
        "--fresh-seed-start", type=int, default=DEFAULT_FRESH_SEED_START
    )
    parser.add_argument("--seed-scan", type=int, default=DEFAULT_SEED_SCAN)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples", type=int, default=DEFAULT_CONFIRMATION_SAMPLES
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=DEFAULT_CONFIRMATION_SAMPLES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if min(args.z_quota, args.transverse_quota, args.multiqubit_quota) < 0:
        raise RunnerError("Representation quotas cannot be negative")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    aggregate = report["training_only_cross_seed_validation"]["aggregate"]
    print("PBMC68k q60 balanced representation")
    print(
        f"- training-only aggregate CV: "
        f"{aggregate['equal_seed_balanced_cv_mean']:.4f}"
    )
    print(
        f"- legacy aggregate CV: {aggregate['equal_seed_legacy_cv_mean']:.4f}"
    )
    print(f"- confirmation gate: {aggregate['passes_confirmation_gate']}")
    confirmation = report["fresh_confirmation"]
    if confirmation["executed"]:
        evaluation = confirmation["evaluation"]
        print(f"- fresh seed: {confirmation['actual_seed']}")
        print(
            "- balanced q60 held-out: "
            f"{evaluation['balanced_representation']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- legacy held-out: "
            f"{evaluation['legacy_top24_reference']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- classical held-out: "
            f"{evaluation['classical_60bin_reference']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- selected pair-sensitive: "
            f"{evaluation['representation_audit']['selected_pair_sensitive_count']}/"
            f"{sum(evaluation['representation_audit']['selected_counts'].values())}"
        )
    else:
        print(f"- fresh confirmation skipped: {confirmation['reason']}")
    print("- provider calls: 0")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
