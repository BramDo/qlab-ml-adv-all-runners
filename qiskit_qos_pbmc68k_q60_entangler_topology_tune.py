#!/usr/bin/env python3
"""Tune a shallow q60 RZZ topology with strict pair-zero ablations.

The 60-qubit register and frozen single-qubit scales remain unchanged.  Four
hardware-friendly subsets of the 59 path edges are compared using only the old
training folds: the full path, the even and odd depth-one matchings, and an
exact maximum-weight path matching selected from each fit fold.  The supervised
matching score is the absolute label-weighted adjacent product.

Every topology is evaluated against all RZZ angles set to zero while retaining
the identical selected observables.  A fully disjoint split is opened only if
the cross-validation winner strictly beats both its own zero ablation and the
strongest zero ablation across all topology-specific observable sets.  This
script has no provider or hardware execution path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_pbmc68k_q60_balanced_representation as balanced
import qiskit_qos_pbmc68k_q60_pair_scale_tune as pair_tune
import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
TOPOLOGY_NAMES = (
    "full_path",
    "even_matching",
    "odd_matching",
    "supervised_matching",
)
DEFAULT_BALANCED_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_balanced_representation.json"
)
DEFAULT_PAIR_SCALE_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_pair_scale_tuning.json"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_entangler_topology_tuning.json"
)
DEFAULT_FRESH_SEED_START = 53
DEFAULT_SEED_SCAN = 300

RunnerError = q60.RunnerError


def fixed_topology_mask(name: str, num_qubits: int) -> np.ndarray:
    """Return a deterministic mask over the path edges."""

    if num_qubits < 1:
        raise RunnerError("At least one qubit is required")
    edge_count = num_qubits - 1
    mask = np.zeros(edge_count, dtype=np.float64)
    if name == "full_path":
        mask[:] = 1.0
    elif name == "even_matching":
        mask[0::2] = 1.0
    elif name == "odd_matching":
        mask[1::2] = 1.0
    else:
        raise RunnerError(f"Unsupported fixed topology: {name}")
    return mask


def maximum_weight_path_matching(
    encoded_fit: np.ndarray, y_fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select the exact maximum-weight matching on adjacent path edges."""

    encoded_fit = np.asarray(encoded_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    if encoded_fit.ndim != 2 or len(encoded_fit) != len(y_fit):
        raise RunnerError("Fit matrix and labels are inconsistent")
    if encoded_fit.shape[1] < 1:
        raise RunnerError("At least one encoded feature is required")
    weight_l1 = float(np.sum(np.abs(y_fit)))
    if weight_l1 <= 0.0:
        raise RunnerError("Matching labels have zero L1 weight")

    products = encoded_fit[:, :-1] * encoded_fit[:, 1:]
    signed = np.sum(y_fit[:, None] * products, axis=0) / weight_l1
    weights = np.abs(signed)
    num_vertices = encoded_fit.shape[1]
    best = np.zeros(num_vertices + 1, dtype=np.float64)
    take_edge = np.zeros(num_vertices + 1, dtype=bool)
    for vertex_count in range(2, num_vertices + 1):
        edge = vertex_count - 2
        take = best[vertex_count - 2] + weights[edge]
        skip = best[vertex_count - 1]
        if take > skip + 1e-15:
            best[vertex_count] = take
            take_edge[vertex_count] = True
        else:
            best[vertex_count] = skip

    mask = np.zeros(num_vertices - 1, dtype=np.float64)
    vertex_count = num_vertices
    while vertex_count >= 2:
        if take_edge[vertex_count]:
            mask[vertex_count - 2] = 1.0
            vertex_count -= 2
        else:
            vertex_count -= 1
    return mask, signed


def _validate_matching_mask(mask: np.ndarray) -> None:
    active = np.flatnonzero(np.asarray(mask) > 0.5)
    if len(active) > 1 and np.any(np.diff(active) == 1):
        raise RunnerError("A depth-one matching contains adjacent active edges")


def _mask_metadata(
    name: str, mask: np.ndarray, signed_scores: np.ndarray | None = None
) -> dict[str, Any]:
    active = np.flatnonzero(mask > 0.5)
    if name != "full_path":
        _validate_matching_mask(mask)
    row: dict[str, Any] = {
        "topology": name,
        "active_edge_count": int(len(active)),
        "active_left_qubits": [int(value) for value in active],
        "logical_entangler_depth": 2 if name == "full_path" else 1,
        "matching_constraint_satisfied": bool(
            name == "full_path"
            or len(active) < 2
            or not np.any(np.diff(active) == 1)
        ),
    }
    if signed_scores is not None:
        row["fit_only_signed_edge_scores"] = [
            float(value) for value in signed_scores
        ]
        row["selected_abs_score_sum"] = float(
            np.sum(np.abs(signed_scores[active]))
        )
    return row


def _parameter_rows(encoded: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for sample in np.asarray(encoded, dtype=np.float64):
        linear, pair = q60.query_parameters(sample)
        if pair.shape != mask.shape:
            raise RunnerError("Topology mask does not match encoded width")
        rows.append((linear, pair * mask))
    return rows


def _feature_matrix(
    encoded: np.ndarray,
    mask: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
) -> np.ndarray:
    return tuner.scaled_feature_matrix(
        _parameter_rows(encoded, mask),
        mappings,
        single_scale=float(configuration["single_scale"]),
        phase_scale=float(configuration["phase_scale"]),
        pair_scale=float(configuration["pair_scale"]),
    )


def _sketch_feature(
    encoded_fit: np.ndarray,
    y_fit: np.ndarray,
    mask: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
) -> np.ndarray:
    linear, pair = q60.sketch_parameters(encoded_fit, y_fit)
    return tuner.scaled_feature_matrix(
        [(linear, pair * mask)],
        mappings,
        single_scale=float(configuration["single_scale"]),
        phase_scale=float(configuration["phase_scale"]),
        pair_scale=float(configuration["pair_scale"]),
    )[0]


def _topology_mask(
    name: str, encoded_fit: np.ndarray, y_fit: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "supervised_matching":
        mask, scores = maximum_weight_path_matching(encoded_fit, y_fit)
        return mask, _mask_metadata(name, mask, scores)
    mask = fixed_topology_mask(name, encoded_fit.shape[1])
    return mask, _mask_metadata(name, mask)


def cross_validate_topologies(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    topologies: Sequence[str],
    cv_folds: int,
    seed: int,
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Compare topology pipelines with fit-only masks and zero ablations."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    names = tuple(str(value) for value in topologies)
    if encoded_train.ndim != 2 or len(encoded_train) != len(y_train):
        raise RunnerError("Training matrix and labels are inconsistent")
    if not names or len(set(names)) != len(names):
        raise RunnerError("Topology candidates must be unique and non-empty")
    if any(name not in TOPOLOGY_NAMES for name in names):
        raise RunnerError("Unsupported topology candidate")

    zero_mask = np.zeros(encoded_train.shape[1] - 1, dtype=np.float64)
    feature_cache: dict[tuple[float, ...], np.ndarray] = {}

    def features(mask: np.ndarray) -> np.ndarray:
        key = tuple(float(value) for value in mask)
        if key not in feature_cache:
            feature_cache[key] = _feature_matrix(
                encoded_train,
                mask,
                mappings=mappings,
                configuration=configuration,
            )
        return feature_cache[key]

    zero_features = features(zero_mask)
    for name in names:
        if name != "supervised_matching":
            features(fixed_topology_mask(name, encoded_train.shape[1]))

    splitter = StratifiedKFold(
        n_splits=int(cv_folds), shuffle=True, random_state=int(seed)
    )
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(encoded_train, y_train > 0.0)
    ):
        fit_indices = fit_indices.astype(np.int64)
        validation_indices = validation_indices.astype(np.int64)
        zero_sketch = _sketch_feature(
            encoded_train[fit_indices],
            y_train[fit_indices],
            zero_mask,
            mappings=mappings,
            configuration=configuration,
        )
        topology_rows: list[dict[str, Any]] = []
        for name in names:
            mask, mask_audit = _topology_mask(
                name, encoded_train[fit_indices], y_train[fit_indices]
            )
            query_features = features(mask)
            selected, _, selection_audit = (
                balanced.select_balanced_train_only_features(
                    query_features[fit_indices],
                    zero_features[fit_indices],
                    y_train[fit_indices],
                    mappings=mappings,
                    z_quota=int(z_quota),
                    transverse_quota=int(transverse_quota),
                    multiqubit_quota=int(multiqubit_quota),
                    shot_intent=int(shot_intent),
                    sensitivity_threshold=float(sensitivity_threshold),
                )
            )
            topology_sketch = _sketch_feature(
                encoded_train[fit_indices],
                y_train[fit_indices],
                mask,
                mappings=mappings,
                configuration=configuration,
            )
            _, topology_scores = balanced._fit_selected_scores(
                query_features[fit_indices],
                query_features[validation_indices],
                y_train[fit_indices],
                topology_sketch,
                selected,
            )
            _, zero_scores = balanced._fit_selected_scores(
                zero_features[fit_indices],
                zero_features[validation_indices],
                y_train[fit_indices],
                zero_sketch,
                selected,
            )
            topology_accuracy = q60._balanced_accuracy(
                y_train[validation_indices], topology_scores
            )
            zero_accuracy = q60._balanced_accuracy(
                y_train[validation_indices], zero_scores
            )
            topology_rows.append(
                {
                    "topology": name,
                    "balanced_accuracy": topology_accuracy,
                    "same_observables_pair_zero_balanced_accuracy": zero_accuracy,
                    "topology_minus_pair_zero": topology_accuracy - zero_accuracy,
                    "selected_indices": [int(value) for value in selected],
                    "selection_audit": selection_audit,
                    "mask_audit": mask_audit,
                    "mask_selection_samples": int(len(fit_indices)),
                    "mask_test_inputs_seen": False,
                    "mask_test_labels_seen": False,
                }
            )
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "fit_samples": int(len(fit_indices)),
                "validation_samples": int(len(validation_indices)),
                "topologies": topology_rows,
            }
        )

    candidates: list[dict[str, Any]] = []
    for name in names:
        rows = [
            next(item for item in fold["topologies"] if item["topology"] == name)
            for fold in fold_rows
        ]
        values = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        zero_values = np.asarray(
            [row["same_observables_pair_zero_balanced_accuracy"] for row in rows],
            dtype=float,
        )
        candidates.append(
            {
                "topology": name,
                "cv_mean_balanced_accuracy": float(np.mean(values)),
                "cv_worst_balanced_accuracy": float(np.min(values)),
                "cv_std_balanced_accuracy": float(np.std(values)),
                "same_observables_pair_zero_cv_mean": float(np.mean(zero_values)),
                "topology_minus_pair_zero_cv_mean": float(
                    np.mean(values - zero_values)
                ),
                "fold_balanced_accuracies": [float(value) for value in values],
                "fold_pair_zero_balanced_accuracies": [
                    float(value) for value in zero_values
                ],
                "active_edge_count_mean": float(
                    np.mean(
                        [row["mask_audit"]["active_edge_count"] for row in rows]
                    )
                ),
                "logical_entangler_depth": int(
                    rows[0]["mask_audit"]["logical_entangler_depth"]
                ),
            }
        )
    return {
        "selection_scope": "training_split_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "supervised_mask_selected_inside_each_fit_fold": True,
        "same_selected_observables_used_for_each_pair_zero_ablation": True,
        "seed": int(seed),
        "training_samples": int(len(encoded_train)),
        "cv_folds": int(cv_folds),
        "candidates": candidates,
        "folds": fold_rows,
    }


def _aggregate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    topology_order = {name: index for index, name in enumerate(TOPOLOGY_NAMES)}
    return (
        -round(float(row["equal_seed_cv_mean"]), 12),
        -round(float(row["worst_seed_cv_mean"]), 12),
        -round(float(row["equal_seed_topology_minus_zero"]), 12),
        round(float(row["between_seed_cv_std"]), 12),
        topology_order[str(row["topology"])],
    )


def aggregate_topology_cv(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    cv_mean_gate: float,
    cv_worst_seed_gate: float,
    expected_pair_sensitive: int,
) -> dict[str, Any]:
    """Rank topology pipelines and enforce a global pair-zero gate."""

    if len(per_seed) < 2:
        raise RunnerError("Topology aggregation needs at least two seeds")
    candidate_sets = [
        {str(row["topology"]) for row in seed_row["candidates"]}
        for seed_row in per_seed
    ]
    if any(values != candidate_sets[0] for values in candidate_sets[1:]):
        raise RunnerError("Topology candidate sets differ across seeds")

    rows: list[dict[str, Any]] = []
    for name in TOPOLOGY_NAMES:
        if name not in candidate_sets[0]:
            continue
        seed_candidates = [
            next(row for row in seed_row["candidates"] if row["topology"] == name)
            for seed_row in per_seed
        ]
        means = np.asarray(
            [row["cv_mean_balanced_accuracy"] for row in seed_candidates],
            dtype=float,
        )
        zero_means = np.asarray(
            [row["same_observables_pair_zero_cv_mean"] for row in seed_candidates],
            dtype=float,
        )
        rows.append(
            {
                "topology": name,
                "equal_seed_cv_mean": float(np.mean(means)),
                "worst_seed_cv_mean": float(np.min(means)),
                "between_seed_cv_std": float(np.std(means)),
                "equal_seed_pair_zero_cv_mean": float(np.mean(zero_means)),
                "equal_seed_topology_minus_zero": float(
                    np.mean(means - zero_means)
                ),
                "logical_entangler_depth": int(
                    seed_candidates[0]["logical_entangler_depth"]
                ),
                "active_edge_count_mean": float(
                    np.mean(
                        [row["active_edge_count_mean"] for row in seed_candidates]
                    )
                ),
                "per_seed": [
                    {
                        "seed": int(seed_row["seed"]),
                        "cv_mean_balanced_accuracy": float(
                            candidate["cv_mean_balanced_accuracy"]
                        ),
                        "same_observables_pair_zero_cv_mean": float(
                            candidate["same_observables_pair_zero_cv_mean"]
                        ),
                        "topology_minus_pair_zero_cv_mean": float(
                            candidate["topology_minus_pair_zero_cv_mean"]
                        ),
                    }
                    for seed_row, candidate in zip(
                        per_seed, seed_candidates, strict=True
                    )
                ],
            }
        )
    chosen = dict(sorted(rows, key=_aggregate_rank_key)[0])
    strongest_zero = max(float(row["equal_seed_pair_zero_cv_mean"]) for row in rows)
    own_gain = float(chosen["equal_seed_topology_minus_zero"])
    global_gain = float(chosen["equal_seed_cv_mean"] - strongest_zero)
    structural = all(
        fold_topology["selection_audit"]["selected_pair_sensitive_count"]
        == int(expected_pair_sensitive)
        for seed_row in per_seed
        for fold in seed_row["folds"]
        for fold_topology in fold["topologies"]
    )
    passes = bool(
        structural
        and round(own_gain, 12) > 0.0
        and round(global_gain, 12) > 0.0
        and float(chosen["equal_seed_cv_mean"]) >= float(cv_mean_gate)
        and float(chosen["worst_seed_cv_mean"]) >= float(cv_worst_seed_gate)
    )
    return {
        "ranking_rule": (
            "maximize equal-seed topology CV mean, then worst seed, topology "
            "gain, stability, and deterministic topology order"
        ),
        "candidates": sorted(rows, key=_aggregate_rank_key),
        "chosen": chosen,
        "strongest_topology_specific_pair_zero_cv_mean": strongest_zero,
        "chosen_minus_own_pair_zero_cv_mean": own_gain,
        "chosen_minus_strongest_pair_zero_cv_mean": global_gain,
        "chosen_strictly_beats_own_pair_zero": round(own_gain, 12) > 0.0,
        "chosen_strictly_beats_strongest_pair_zero": round(global_gain, 12) > 0.0,
        "structural_gate": structural,
        "cv_mean_gate": float(cv_mean_gate),
        "cv_worst_seed_gate": float(cv_worst_seed_gate),
        "passes_fresh_confirmation_gate": passes,
    }


def evaluate_fresh_topology(
    data: Mapping[str, Any],
    *,
    topology: str,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Evaluate one fixed topology once on a fully disjoint split."""

    mask, mask_audit = _topology_mask(
        topology, data["encoded_train"], data["y_train"]
    )
    zero_mask = np.zeros_like(mask)
    topology_train = _feature_matrix(
        data["encoded_train"],
        mask,
        mappings=mappings,
        configuration=configuration,
    )
    topology_test = _feature_matrix(
        data["encoded_test"],
        mask,
        mappings=mappings,
        configuration=configuration,
    )
    zero_train = _feature_matrix(
        data["encoded_train"],
        zero_mask,
        mappings=mappings,
        configuration=configuration,
    )
    zero_test = _feature_matrix(
        data["encoded_test"],
        zero_mask,
        mappings=mappings,
        configuration=configuration,
    )
    selected, selection_scores, selection_audit = (
        balanced.select_balanced_train_only_features(
            topology_train,
            zero_train,
            data["y_train"],
            mappings=mappings,
            z_quota=int(z_quota),
            transverse_quota=int(transverse_quota),
            multiqubit_quota=int(multiqubit_quota),
            shot_intent=int(shot_intent),
            sensitivity_threshold=float(sensitivity_threshold),
        )
    )
    topology_sketch = _sketch_feature(
        data["encoded_train"],
        data["y_train"],
        mask,
        mappings=mappings,
        configuration=configuration,
    )
    zero_sketch = _sketch_feature(
        data["encoded_train"],
        data["y_train"],
        zero_mask,
        mappings=mappings,
        configuration=configuration,
    )
    topology_train_scores, topology_test_scores = balanced._fit_selected_scores(
        topology_train,
        topology_test,
        data["y_train"],
        topology_sketch,
        selected,
    )
    _, zero_test_scores = balanced._fit_selected_scores(
        zero_train,
        zero_test,
        data["y_train"],
        zero_sketch,
        selected,
    )
    classical_train, classical_test = q60._ridge_scores(
        data["encoded_train"], data["encoded_test"], data["y_train"]
    )
    topology_accuracy = q60._balanced_accuracy(
        data["y_test"], topology_test_scores
    )
    zero_accuracy = q60._balanced_accuracy(data["y_test"], zero_test_scores)
    classical_accuracy = q60._balanced_accuracy(data["y_test"], classical_test)
    seed = int(data["seed"])
    return {
        "configuration_fixed_before_fresh_split": True,
        "topology": topology,
        "supervised_mask_uses_fresh_training_only": True,
        "mask_audit": mask_audit,
        "representation_audit": selection_audit,
        "selected_observables": [
            tuner._mapping_row(
                int(index), mappings[int(index)], selection_scores[int(index)]
            )
            for index in selected
        ],
        "topology_result": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], topology_train_scores
            ),
            "test_balanced_accuracy": topology_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], topology_test_scores, seed=seed + 9500
            ),
        },
        "same_observables_pair_zero_result": {
            "test_balanced_accuracy": zero_accuracy,
            "topology_minus_zero_test": topology_accuracy - zero_accuracy,
        },
        "classical_60bin_reference": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], classical_train
            ),
            "test_balanced_accuracy": classical_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], classical_test, seed=seed + 9501
            ),
        },
        "topology_minus_classical_test": topology_accuracy - classical_accuracy,
        "paper_feasibility_signal": bool(topology_accuracy >= 0.55),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_paths = [Path(path) for path in args.tuning_reports]
    prior_path = Path(args.prior_confirmation)
    balanced_path = Path(args.balanced_report)
    pair_scale_path = Path(args.pair_scale_report)
    prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
    balanced_report = json.loads(balanced_path.read_text(encoding="utf-8"))
    pair_scale_report = json.loads(pair_scale_path.read_text(encoding="utf-8"))
    configuration = freeze._configuration_only(prior_report["selection"]["chosen"])
    runs, raw_reports, shared_config, x_pair, y_pair = (
        balanced._load_source_training_data(
            report_paths, cache_dir=Path(args.cache_dir)
        )
    )
    mappings = toy.pauli_feature_mappings(
        int(shared_config["qubits"]), family="local"
    )

    per_seed: list[dict[str, Any]] = []
    for run_data in runs:
        result = cross_validate_topologies(
            run_data["encoded_train"],
            run_data["y_train"],
            mappings=mappings,
            configuration=configuration,
            topologies=args.topologies,
            cv_folds=int(shared_config["cv_folds"]),
            seed=int(run_data["seed"]),
            shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
            z_quota=int(args.z_quota),
            transverse_quota=int(args.transverse_quota),
            multiqubit_quota=int(args.multiqubit_quota),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        previous = next(
            row
            for row in balanced_report["training_only_cross_seed_validation"][
                "per_seed"
            ]
            if int(row["seed"]) == int(run_data["seed"])
        )
        full_path = next(
            row for row in result["candidates"] if row["topology"] == "full_path"
        )
        if not np.isclose(
            full_path["cv_mean_balanced_accuracy"],
            previous["balanced_cv_mean"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RunnerError(
                f"Full-path reference reproduction failed for seed {run_data['seed']}"
            )
        result["full_path_reference_metric_reproduced"] = True
        result["source_report"] = run_data["report_path"]
        result["source_report_sha256"] = run_data["report_sha256"]
        per_seed.append(result)

    aggregate = aggregate_topology_cv(
        per_seed,
        cv_mean_gate=float(args.cv_mean_gate),
        cv_worst_seed_gate=float(args.cv_worst_seed_gate),
        expected_pair_sensitive=int(args.transverse_quota)
        + int(args.multiqubit_quota),
    )
    fresh_confirmation: dict[str, Any] = {
        "executed": False,
        "reason": (
            "the winning topology did not strictly beat both its own and the "
            "strongest topology-specific pair-zero ablation while passing gates"
        ),
    }
    if aggregate["passes_fresh_confirmation_gate"]:
        excluded = freeze._excluded_source_indices(raw_reports)
        pair_tune._add_split_indices(excluded, prior_report)
        balanced_split = balanced_report["fresh_confirmation"]["split"]
        excluded.update(int(value) for value in balanced_split["train_indices"])
        excluded.update(int(value) for value in balanced_split["test_indices"])
        prior_topology_confirmation = pair_scale_report.get("fresh_confirmation", {})
        if prior_topology_confirmation.get("executed"):
            prior_split = prior_topology_confirmation["split"]
            excluded.update(int(value) for value in prior_split["train_indices"])
            excluded.update(int(value) for value in prior_split["test_indices"])
        fresh = balanced._load_fresh_split(
            args,
            x_pair=x_pair,
            y_pair=y_pair,
            shared_config=shared_config,
            excluded_indices=excluded,
        )
        evaluation = evaluate_fresh_topology(
            fresh,
            topology=str(aggregate["chosen"]["topology"]),
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
        fresh_confirmation = {
            "executed": True,
            "actual_seed": int(fresh["seed"]),
            "train_samples": int(len(fresh["encoded_train"])),
            "test_samples": int(len(fresh["encoded_test"])),
            "excluded_source_indices": int(len(excluded)),
            "source_index_overlap": int(len(fresh_indices & excluded)),
            "train_test_index_overlap": int(
                len(
                    set(int(value) for value in fresh["train_indices"])
                    & set(int(value) for value in fresh["test_indices"])
                )
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
            },
            "evaluation": evaluation,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_balanced_entangler_topology_tuning",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "pre_registered_protocol": {
            "topologies": [str(value) for value in args.topologies],
            "pair_scale_frozen": float(configuration["pair_scale"]),
            "single_scale_frozen": float(configuration["single_scale"]),
            "phase_scale_frozen": float(configuration["phase_scale"]),
            "supervised_matching_objective": (
                "exact maximum-weight matching on absolute fit-only "
                "label-weighted adjacent products"
            ),
            "observable_quotas": {
                "single_z": int(args.z_quota),
                "pair_sensitive_local_xy": int(args.transverse_quota),
                "pair_sensitive_multiqubit": int(args.multiqubit_quota),
            },
            "same_observables_used_for_topology_and_zero_within_fold": True,
            "balanced_report": str(balanced_path.resolve()),
            "balanced_report_sha256": freeze._file_sha256(balanced_path),
            "pair_scale_report": str(pair_scale_path.resolve()),
            "pair_scale_report_sha256": freeze._file_sha256(pair_scale_path),
        },
        "training_only_topology_cv": {
            "source_test_results_used": False,
            "source_seeds": [int(run["seed"]) for run in runs],
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        "fresh_confirmation": fresh_confirmation,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Topology and any supervised masks were selected using old training "
            "folds only. A fresh split opens only for a strict entangler gain. "
            "Results are exact local model diagnostics, not hardware evidence or "
            "quantum advantage."
        ),
    }


def _parse_topologies(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("expected unique topology names")
    if any(name not in TOPOLOGY_NAMES for name in names):
        raise argparse.ArgumentTypeError(
            f"expected a subset of {','.join(TOPOLOGY_NAMES)}"
        )
    if "full_path" not in names:
        raise argparse.ArgumentTypeError("full_path is required for reproduction")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-reports",
        type=Path,
        nargs="+",
        default=balanced.DEFAULT_TUNING_REPORTS,
    )
    parser.add_argument(
        "--prior-confirmation",
        type=Path,
        default=balanced.DEFAULT_PRIOR_CONFIRMATION,
    )
    parser.add_argument(
        "--balanced-report", type=Path, default=DEFAULT_BALANCED_REPORT
    )
    parser.add_argument(
        "--pair-scale-report", type=Path, default=DEFAULT_PAIR_SCALE_REPORT
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--topologies",
        type=_parse_topologies,
        default=TOPOLOGY_NAMES,
    )
    parser.add_argument("--z-quota", type=int, default=balanced.DEFAULT_Z_QUOTA)
    parser.add_argument(
        "--transverse-quota", type=int, default=balanced.DEFAULT_TRANSVERSE_QUOTA
    )
    parser.add_argument(
        "--multiqubit-quota", type=int, default=balanced.DEFAULT_MULTIQUBIT_QUOTA
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=balanced.DEFAULT_SENSITIVITY_THRESHOLD,
    )
    parser.add_argument(
        "--cv-mean-gate", type=float, default=balanced.DEFAULT_CV_MEAN_GATE
    )
    parser.add_argument(
        "--cv-worst-seed-gate",
        type=float,
        default=balanced.DEFAULT_CV_WORST_SEED_GATE,
    )
    parser.add_argument(
        "--fresh-seed-start", type=int, default=DEFAULT_FRESH_SEED_START
    )
    parser.add_argument("--seed-scan", type=int, default=DEFAULT_SEED_SCAN)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=balanced.DEFAULT_CONFIRMATION_SAMPLES,
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=balanced.DEFAULT_CONFIRMATION_SAMPLES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
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

    aggregate = report["training_only_topology_cv"]["aggregate"]
    chosen = aggregate["chosen"]
    print("PBMC68k q60 balanced entangler-topology tuning")
    print(
        f"- chosen topology: {chosen['topology']} "
        f"(CV {chosen['equal_seed_cv_mean']:.4f})"
    )
    print(
        "- chosen minus own pair=0 CV: "
        f"{aggregate['chosen_minus_own_pair_zero_cv_mean']:+.4f}"
    )
    print(
        "- chosen minus strongest pair=0 CV: "
        f"{aggregate['chosen_minus_strongest_pair_zero_cv_mean']:+.4f}"
    )
    print(
        "- fresh confirmation gate: "
        f"{aggregate['passes_fresh_confirmation_gate']}"
    )
    confirmation = report["fresh_confirmation"]
    if confirmation["executed"]:
        evaluation = confirmation["evaluation"]
        print(f"- fresh seed: {confirmation['actual_seed']}")
        print(
            "- topology held-out: "
            f"{evaluation['topology_result']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- same-observable pair=0 held-out: "
            f"{evaluation['same_observables_pair_zero_result']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- classical held-out: "
            f"{evaluation['classical_60bin_reference']['test_balanced_accuracy']:.4f}"
        )
    else:
        print(f"- fresh confirmation skipped: {confirmation['reason']}")
    print("- provider calls: 0")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
