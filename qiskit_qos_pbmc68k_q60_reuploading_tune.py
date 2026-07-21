#!/usr/bin/env python3
"""Test one shallow non-commuting re-uploading layer on the q60 model.

The frozen H-RY-RZ-RZZ path circuit is extended by one local RX or RY data
re-uploading layer.  Every architecture is compared with an otherwise
identical circuit whose RZZ angles are zero.  Architectures and observables are
selected using the old training folds only.

After the failed topology confirmation, the fresh-split gate is deliberately
strict: the winning non-trivial re-uploading architecture must have positive
entangler gain on every source seed, beat the strongest topology-specific zero
ablation in aggregate, and pass the existing mean and worst-seed gates.  There
is no provider or hardware execution path.
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_pbmc68k_q60_balanced_representation as balanced
import qiskit_qos_pbmc68k_q60_entangler_topology_tune as topology_tune
import qiskit_qos_pbmc68k_q60_pair_scale_tune as pair_tune
import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
ARCHITECTURES: dict[str, tuple[str, float]] = {
    "none": ("none", 0.0),
    "ry_0p5": ("ry", 0.5),
    "ry_1p0": ("ry", 1.0),
    "rx_0p5": ("rx", 0.5),
    "rx_1p0": ("rx", 1.0),
}
DEFAULT_BALANCED_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_balanced_representation.json"
)
DEFAULT_PAIR_SCALE_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_pair_scale_tuning.json"
)
DEFAULT_TOPOLOGY_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_entangler_topology_tuning.json"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_reuploading_tuning.json"
)
DEFAULT_FRESH_SEED_START = 233
DEFAULT_SEED_SCAN = 400

RunnerError = q60.RunnerError


def architecture_parameters(name: str) -> tuple[str, float]:
    try:
        return ARCHITECTURES[str(name)]
    except KeyError as exc:
        raise RunnerError(f"Unsupported re-uploading architecture: {name}") from exc


def exact_local_reupload_expectation(
    linear_values: np.ndarray,
    pair_values: np.ndarray,
    mapping: Mapping[int, str],
    *,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
    post_axis: str,
    post_scale: float,
) -> float:
    """Evaluate a local Pauli in the at-most-four-qubit causal cone."""

    linear_values = np.asarray(linear_values, dtype=np.float64)
    pair_values = np.asarray(pair_values, dtype=np.float64)
    num_qubits = len(linear_values)
    if pair_values.shape != (max(num_qubits - 1, 0),):
        raise RunnerError("Pair values do not match the qubit path")
    if post_axis not in {"none", "rx", "ry"}:
        raise RunnerError("Post axis must be none, rx, or ry")
    if not np.isfinite(post_scale) or post_scale < 0.0:
        raise RunnerError("Post scale must be finite and non-negative")
    if post_axis == "none" and post_scale != 0.0:
        raise RunnerError("The none architecture requires zero post scale")

    support = {int(qubit) for qubit in mapping}
    if not support or min(support) < 0 or max(support) >= num_qubits:
        raise RunnerError("Pauli mapping is outside the circuit")
    incident = [
        left
        for left in range(num_qubits - 1)
        if left in support or left + 1 in support
    ]
    active = sorted(support | set(incident) | {left + 1 for left in incident})
    local_index = {qubit: index for index, qubit in enumerate(active)}
    circuit = QuantumCircuit(len(active))
    circuit.h(range(len(active)))
    for global_qubit in active:
        local_qubit = local_index[global_qubit]
        value = float(linear_values[global_qubit])
        circuit.ry(float(single_scale) * value, local_qubit)
        circuit.rz(float(phase_scale) * value, local_qubit)
    for left in incident:
        circuit.rzz(
            float(pair_scale) * float(pair_values[left]),
            local_index[left],
            local_index[left + 1],
        )
    if post_axis != "none" and post_scale > 0.0:
        gate = getattr(circuit, post_axis)
        for global_qubit in active:
            gate(
                float(post_scale) * float(linear_values[global_qubit]),
                local_index[global_qubit],
            )

    label = ["I"] * len(active)
    for global_qubit, pauli in mapping.items():
        local_qubit = local_index[int(global_qubit)]
        label[len(active) - 1 - local_qubit] = str(pauli)
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(SparsePauliOp("".join(label)))
    return float(np.real_if_close(value).real)


def reupload_feature_matrix(
    parameter_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    mappings: Sequence[Mapping[int, str]],
    *,
    configuration: Mapping[str, Any],
    architecture: str,
    pair_scale: float,
) -> np.ndarray:
    post_axis, post_scale = architecture_parameters(architecture)
    return np.asarray(
        [
            [
                exact_local_reupload_expectation(
                    linear,
                    pair,
                    mapping,
                    single_scale=float(configuration["single_scale"]),
                    phase_scale=float(configuration["phase_scale"]),
                    pair_scale=float(pair_scale),
                    post_axis=post_axis,
                    post_scale=post_scale,
                )
                for mapping in mappings
            ]
            for linear, pair in parameter_rows
        ],
        dtype=np.float64,
    )


def _sketch_feature(
    encoded_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    architecture: str,
    pair_scale: float,
) -> np.ndarray:
    linear, pair = q60.sketch_parameters(encoded_fit, y_fit)
    return reupload_feature_matrix(
        [(linear, pair)],
        mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=pair_scale,
    )[0]


def cross_validate_reuploading(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    architectures: Sequence[str],
    cv_folds: int,
    seed: int,
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Compare post-rotation architectures using training folds only."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    names = tuple(str(value) for value in architectures)
    if encoded_train.ndim != 2 or len(encoded_train) != len(y_train):
        raise RunnerError("Training matrix and labels are inconsistent")
    if not names or len(set(names)) != len(names):
        raise RunnerError("Architectures must be unique and non-empty")
    if "none" not in names:
        raise RunnerError("The none architecture is required as a reference")
    for name in names:
        architecture_parameters(name)

    parameter_rows = [q60.query_parameters(row) for row in encoded_train]
    pair_scale = float(configuration["pair_scale"])
    entangled_features = {
        name: reupload_feature_matrix(
            parameter_rows,
            mappings,
            configuration=configuration,
            architecture=name,
            pair_scale=pair_scale,
        )
        for name in names
    }
    zero_features = {
        name: reupload_feature_matrix(
            parameter_rows,
            mappings,
            configuration=configuration,
            architecture=name,
            pair_scale=0.0,
        )
        for name in names
    }
    splitter = StratifiedKFold(
        n_splits=int(cv_folds), shuffle=True, random_state=int(seed)
    )
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(encoded_train, y_train > 0.0)
    ):
        fit_indices = fit_indices.astype(np.int64)
        validation_indices = validation_indices.astype(np.int64)
        architecture_rows: list[dict[str, Any]] = []
        for name in names:
            selected, _, selection_audit = (
                balanced.select_balanced_train_only_features(
                    entangled_features[name][fit_indices],
                    zero_features[name][fit_indices],
                    y_train[fit_indices],
                    mappings=mappings,
                    z_quota=int(z_quota),
                    transverse_quota=int(transverse_quota),
                    multiqubit_quota=int(multiqubit_quota),
                    shot_intent=int(shot_intent),
                    sensitivity_threshold=float(sensitivity_threshold),
                )
            )
            entangled_sketch = _sketch_feature(
                encoded_train[fit_indices],
                y_train[fit_indices],
                mappings=mappings,
                configuration=configuration,
                architecture=name,
                pair_scale=pair_scale,
            )
            zero_sketch = _sketch_feature(
                encoded_train[fit_indices],
                y_train[fit_indices],
                mappings=mappings,
                configuration=configuration,
                architecture=name,
                pair_scale=0.0,
            )
            _, entangled_scores = balanced._fit_selected_scores(
                entangled_features[name][fit_indices],
                entangled_features[name][validation_indices],
                y_train[fit_indices],
                entangled_sketch,
                selected,
            )
            _, zero_scores = balanced._fit_selected_scores(
                zero_features[name][fit_indices],
                zero_features[name][validation_indices],
                y_train[fit_indices],
                zero_sketch,
                selected,
            )
            entangled_accuracy = q60._balanced_accuracy(
                y_train[validation_indices], entangled_scores
            )
            zero_accuracy = q60._balanced_accuracy(
                y_train[validation_indices], zero_scores
            )
            post_axis, post_scale = architecture_parameters(name)
            architecture_rows.append(
                {
                    "architecture": name,
                    "post_axis": post_axis,
                    "post_scale": post_scale,
                    "logical_circuit_depth": 5 if name == "none" else 6,
                    "balanced_accuracy": entangled_accuracy,
                    "same_observables_pair_zero_balanced_accuracy": zero_accuracy,
                    "entangler_gain": entangled_accuracy - zero_accuracy,
                    "selected_indices": [int(value) for value in selected],
                    "selection_audit": selection_audit,
                }
            )
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "fit_samples": int(len(fit_indices)),
                "validation_samples": int(len(validation_indices)),
                "architectures": architecture_rows,
            }
        )

    candidates: list[dict[str, Any]] = []
    for name in names:
        rows = [
            next(
                item
                for item in fold["architectures"]
                if item["architecture"] == name
            )
            for fold in fold_rows
        ]
        values = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        zero_values = np.asarray(
            [row["same_observables_pair_zero_balanced_accuracy"] for row in rows],
            dtype=float,
        )
        post_axis, post_scale = architecture_parameters(name)
        candidates.append(
            {
                "architecture": name,
                "post_axis": post_axis,
                "post_scale": post_scale,
                "logical_circuit_depth": 5 if name == "none" else 6,
                "cv_mean_balanced_accuracy": float(np.mean(values)),
                "cv_worst_balanced_accuracy": float(np.min(values)),
                "cv_std_balanced_accuracy": float(np.std(values)),
                "same_observables_pair_zero_cv_mean": float(np.mean(zero_values)),
                "entangler_gain_cv_mean": float(np.mean(values - zero_values)),
                "fold_balanced_accuracies": [float(value) for value in values],
                "fold_pair_zero_balanced_accuracies": [
                    float(value) for value in zero_values
                ],
            }
        )
    return {
        "selection_scope": "training_split_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "same_observables_used_for_entangled_and_zero_within_fold": True,
        "seed": int(seed),
        "training_samples": int(len(encoded_train)),
        "cv_folds": int(cv_folds),
        "candidates": candidates,
        "folds": fold_rows,
    }


def _aggregate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    order = {name: index for index, name in enumerate(ARCHITECTURES)}
    return (
        -round(float(row["equal_seed_cv_mean"]), 12),
        -round(float(row["worst_seed_cv_mean"]), 12),
        -round(float(row["equal_seed_entangler_gain"]), 12),
        round(float(row["between_seed_cv_std"]), 12),
        order[str(row["architecture"])],
    )


def aggregate_reuploading_cv(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    cv_mean_gate: float,
    cv_worst_seed_gate: float,
    expected_pair_sensitive: int,
) -> dict[str, Any]:
    """Rank architectures and require positive gain on every source seed."""

    if len(per_seed) < 2:
        raise RunnerError("Re-uploading aggregation needs at least two seeds")
    candidate_sets = [
        {str(row["architecture"]) for row in seed_row["candidates"]}
        for seed_row in per_seed
    ]
    if any(values != candidate_sets[0] for values in candidate_sets[1:]):
        raise RunnerError("Architecture candidate sets differ across seeds")

    rows: list[dict[str, Any]] = []
    for name in ARCHITECTURES:
        if name not in candidate_sets[0]:
            continue
        seed_candidates = [
            next(
                row
                for row in seed_row["candidates"]
                if row["architecture"] == name
            )
            for seed_row in per_seed
        ]
        means = np.asarray(
            [row["cv_mean_balanced_accuracy"] for row in seed_candidates],
            dtype=float,
        )
        zeros = np.asarray(
            [row["same_observables_pair_zero_cv_mean"] for row in seed_candidates],
            dtype=float,
        )
        gains = means - zeros
        post_axis, post_scale = architecture_parameters(name)
        rows.append(
            {
                "architecture": name,
                "post_axis": post_axis,
                "post_scale": post_scale,
                "logical_circuit_depth": 5 if name == "none" else 6,
                "equal_seed_cv_mean": float(np.mean(means)),
                "worst_seed_cv_mean": float(np.min(means)),
                "between_seed_cv_std": float(np.std(means)),
                "equal_seed_pair_zero_cv_mean": float(np.mean(zeros)),
                "equal_seed_entangler_gain": float(np.mean(gains)),
                "worst_seed_entangler_gain": float(np.min(gains)),
                "positive_entangler_gain_every_seed": bool(
                    all(round(float(value), 12) > 0.0 for value in gains)
                ),
                "per_seed": [
                    {
                        "seed": int(seed_row["seed"]),
                        "cv_mean_balanced_accuracy": float(mean),
                        "same_observables_pair_zero_cv_mean": float(zero),
                        "entangler_gain_cv_mean": float(gain),
                    }
                    for seed_row, mean, zero, gain in zip(
                        per_seed, means, zeros, gains, strict=True
                    )
                ],
            }
        )

    chosen = dict(sorted(rows, key=_aggregate_rank_key)[0])
    strongest_zero = max(float(row["equal_seed_pair_zero_cv_mean"]) for row in rows)
    global_gain = float(chosen["equal_seed_cv_mean"] - strongest_zero)
    structural = all(
        architecture_row["selection_audit"]["selected_pair_sensitive_count"]
        == int(expected_pair_sensitive)
        for seed_row in per_seed
        for fold in seed_row["folds"]
        for architecture_row in fold["architectures"]
    )
    passes = bool(
        structural
        and str(chosen["architecture"]) != "none"
        and bool(chosen["positive_entangler_gain_every_seed"])
        and round(global_gain, 12) > 0.0
        and float(chosen["equal_seed_cv_mean"]) >= float(cv_mean_gate)
        and float(chosen["worst_seed_cv_mean"]) >= float(cv_worst_seed_gate)
    )
    return {
        "ranking_rule": (
            "maximize equal-seed CV mean, then worst seed, entangler gain, "
            "stability, and deterministic architecture order"
        ),
        "candidates": sorted(rows, key=_aggregate_rank_key),
        "chosen": chosen,
        "strongest_architecture_specific_pair_zero_cv_mean": strongest_zero,
        "chosen_minus_strongest_pair_zero_cv_mean": global_gain,
        "chosen_is_nontrivial_reuploading": str(chosen["architecture"]) != "none",
        "chosen_positive_entangler_gain_every_seed": bool(
            chosen["positive_entangler_gain_every_seed"]
        ),
        "chosen_strictly_beats_strongest_pair_zero": round(global_gain, 12) > 0.0,
        "structural_gate": structural,
        "cv_mean_gate": float(cv_mean_gate),
        "cv_worst_seed_gate": float(cv_worst_seed_gate),
        "passes_fresh_confirmation_gate": passes,
    }


def evaluate_fresh_reuploading(
    data: Mapping[str, Any],
    *,
    architecture: str,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Evaluate one frozen re-uploading architecture on one fresh split."""

    train_parameters = [q60.query_parameters(row) for row in data["encoded_train"]]
    test_parameters = [q60.query_parameters(row) for row in data["encoded_test"]]
    pair_scale = float(configuration["pair_scale"])
    entangled_train = reupload_feature_matrix(
        train_parameters,
        mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=pair_scale,
    )
    entangled_test = reupload_feature_matrix(
        test_parameters,
        mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=pair_scale,
    )
    zero_train = reupload_feature_matrix(
        train_parameters,
        mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=0.0,
    )
    zero_test = reupload_feature_matrix(
        test_parameters,
        mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=0.0,
    )
    selected, selection_scores, selection_audit = (
        balanced.select_balanced_train_only_features(
            entangled_train,
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
    entangled_sketch = _sketch_feature(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=pair_scale,
    )
    zero_sketch = _sketch_feature(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
        architecture=architecture,
        pair_scale=0.0,
    )
    entangled_train_scores, entangled_test_scores = balanced._fit_selected_scores(
        entangled_train,
        entangled_test,
        data["y_train"],
        entangled_sketch,
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
    entangled_accuracy = q60._balanced_accuracy(
        data["y_test"], entangled_test_scores
    )
    zero_accuracy = q60._balanced_accuracy(data["y_test"], zero_test_scores)
    classical_accuracy = q60._balanced_accuracy(data["y_test"], classical_test)
    post_axis, post_scale = architecture_parameters(architecture)
    seed = int(data["seed"])
    return {
        "configuration_fixed_before_fresh_split": True,
        "architecture": architecture,
        "post_axis": post_axis,
        "post_scale": post_scale,
        "logical_circuit_depth": 6,
        "selection_uses_fresh_training_only": True,
        "representation_audit": selection_audit,
        "selected_observables": [
            tuner._mapping_row(
                int(index), mappings[int(index)], selection_scores[int(index)]
            )
            for index in selected
        ],
        "reuploading_result": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], entangled_train_scores
            ),
            "test_balanced_accuracy": entangled_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], entangled_test_scores, seed=seed + 9600
            ),
        },
        "same_observables_pair_zero_result": {
            "test_balanced_accuracy": zero_accuracy,
            "reuploading_minus_zero_test": entangled_accuracy - zero_accuracy,
        },
        "classical_60bin_reference": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], classical_train
            ),
            "test_balanced_accuracy": classical_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], classical_test, seed=seed + 9601
            ),
        },
        "reuploading_minus_classical_test": entangled_accuracy - classical_accuracy,
        "paper_feasibility_signal": bool(entangled_accuracy >= 0.55),
    }


def _exclude_split(excluded: set[int], confirmation: Mapping[str, Any]) -> None:
    if not confirmation.get("executed"):
        return
    split = confirmation["split"]
    excluded.update(int(value) for value in split["train_indices"])
    excluded.update(int(value) for value in split["test_indices"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_paths = [Path(path) for path in args.tuning_reports]
    prior_path = Path(args.prior_confirmation)
    balanced_path = Path(args.balanced_report)
    pair_scale_path = Path(args.pair_scale_report)
    topology_path = Path(args.topology_report)
    prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
    balanced_report = json.loads(balanced_path.read_text(encoding="utf-8"))
    pair_scale_report = json.loads(pair_scale_path.read_text(encoding="utf-8"))
    topology_report = json.loads(topology_path.read_text(encoding="utf-8"))
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
        result = cross_validate_reuploading(
            run_data["encoded_train"],
            run_data["y_train"],
            mappings=mappings,
            configuration=configuration,
            architectures=args.architectures,
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
        none_candidate = next(
            row
            for row in result["candidates"]
            if row["architecture"] == "none"
        )
        if not np.isclose(
            none_candidate["cv_mean_balanced_accuracy"],
            previous["balanced_cv_mean"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RunnerError(
                f"No-reupload reference reproduction failed for seed {run_data['seed']}"
            )
        result["no_reupload_reference_metric_reproduced"] = True
        result["source_report"] = run_data["report_path"]
        result["source_report_sha256"] = run_data["report_sha256"]
        per_seed.append(result)

    aggregate = aggregate_reuploading_cv(
        per_seed,
        cv_mean_gate=float(args.cv_mean_gate),
        cv_worst_seed_gate=float(args.cv_worst_seed_gate),
        expected_pair_sensitive=int(args.transverse_quota)
        + int(args.multiqubit_quota),
    )
    fresh_confirmation: dict[str, Any] = {
        "executed": False,
        "reason": (
            "the winning architecture was not a non-trivial re-uploading model "
            "with positive entangler gain on every seed and a strict global "
            "pair-zero win"
        ),
    }
    if aggregate["passes_fresh_confirmation_gate"]:
        excluded = freeze._excluded_source_indices(raw_reports)
        pair_tune._add_split_indices(excluded, prior_report)
        balanced_split = balanced_report["fresh_confirmation"]["split"]
        excluded.update(int(value) for value in balanced_split["train_indices"])
        excluded.update(int(value) for value in balanced_split["test_indices"])
        _exclude_split(excluded, pair_scale_report["fresh_confirmation"])
        _exclude_split(excluded, topology_report["fresh_confirmation"])
        fresh = balanced._load_fresh_split(
            args,
            x_pair=x_pair,
            y_pair=y_pair,
            shared_config=shared_config,
            excluded_indices=excluded,
        )
        evaluation = evaluate_fresh_reuploading(
            fresh,
            architecture=str(aggregate["chosen"]["architecture"]),
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
        "kind": "pbmc68k_q60_balanced_reuploading_tuning",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "pre_registered_protocol": {
            "architectures": [str(value) for value in args.architectures],
            "pair_scale_frozen": float(configuration["pair_scale"]),
            "single_scale_frozen": float(configuration["single_scale"]),
            "phase_scale_frozen": float(configuration["phase_scale"]),
            "full_path_topology_frozen": True,
            "positive_entangler_gain_required_on_every_source_seed": True,
            "observable_quotas": {
                "single_z": int(args.z_quota),
                "pair_sensitive_local_xy": int(args.transverse_quota),
                "pair_sensitive_multiqubit": int(args.multiqubit_quota),
            },
            "same_observables_used_for_entangled_and_zero_within_fold": True,
            "balanced_report": str(balanced_path.resolve()),
            "balanced_report_sha256": freeze._file_sha256(balanced_path),
            "pair_scale_report": str(pair_scale_path.resolve()),
            "pair_scale_report_sha256": freeze._file_sha256(pair_scale_path),
            "topology_report": str(topology_path.resolve()),
            "topology_report_sha256": freeze._file_sha256(topology_path),
        },
        "training_only_reuploading_cv": {
            "source_test_results_used": False,
            "source_seeds": [int(run["seed"]) for run in runs],
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        "fresh_confirmation": fresh_confirmation,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Re-uploading architecture selection used old training folds only. "
            "A fresh split opens only after positive entangler gain on every "
            "source seed. Results are exact local model diagnostics, not hardware "
            "evidence or quantum advantage."
        ),
    }


def _parse_architectures(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("expected unique architecture names")
    if "none" not in names:
        raise argparse.ArgumentTypeError("none is required as the reference")
    if any(name not in ARCHITECTURES for name in names):
        raise argparse.ArgumentTypeError(
            f"expected a subset of {','.join(ARCHITECTURES)}"
        )
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
    parser.add_argument(
        "--topology-report", type=Path, default=DEFAULT_TOPOLOGY_REPORT
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--architectures",
        type=_parse_architectures,
        default=tuple(ARCHITECTURES),
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

    aggregate = report["training_only_reuploading_cv"]["aggregate"]
    chosen = aggregate["chosen"]
    print("PBMC68k q60 balanced non-commuting re-uploading tuning")
    print(
        f"- chosen architecture: {chosen['architecture']} "
        f"(CV {chosen['equal_seed_cv_mean']:.4f})"
    )
    print(
        "- mean entangler gain: "
        f"{chosen['equal_seed_entangler_gain']:+.4f}"
    )
    print(
        "- positive entangler gain on every seed: "
        f"{chosen['positive_entangler_gain_every_seed']}"
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
            "- re-uploading held-out: "
            f"{evaluation['reuploading_result']['test_balanced_accuracy']:.4f}"
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
