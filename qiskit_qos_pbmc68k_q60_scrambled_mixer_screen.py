#!/usr/bin/env python3
"""Local q60 non-commuting architecture and classical-hardness screen.

The architecture winner is chosen with seed-11 training folds only.  The
fixed test split is evaluated once for that winner.  Every quantum feature is
estimated locally with a bounded Aer matrix-product-state simulation; small-q
statevector parity and q60 bond-dimension probes keep approximation error
explicit.  This runner contains no provider, Fire Opal, or hardware path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_rx05_fireopal_validate as rx05
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
ARCHITECTURES = ("control_rx05_d6", "grid_mixer_d12", "scrambled_mixer_d16")
DEFAULT_SOURCE_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_seed11_train_only_tuning.json"
)
DEFAULT_REUPLOAD_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_balanced_reuploading_tuning.json"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_scrambled_mixer_architecture_screen.json"
)
DEFAULT_BOND_DIMENSION = 64
DEFAULT_PROBE_BOND_DIMENSIONS = (32, 128)
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_SELECTED_FEATURES = 24
DEFAULT_SHOT_INTENT = 128
DEFAULT_CV_FOLDS = 4
DEFAULT_CV_SEED = 6011
SINGLE_SCALE = 0.75
PHASE_SCALE = 0.25
PAIR_SCALE = 0.95
MIXER_DATA_SCALE = 0.35
REUPLOAD_SCALE = 0.20
FINAL_RX_SCALE = 0.50

RunnerError = q60.RunnerError


@dataclass(frozen=True)
class SeedData:
    encoded_train: np.ndarray
    encoded_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any]


def _utc_now() -> str:
    return q60.q40_validate._utc_now()


def _sha256_file(path: Path) -> str:
    return q60.q40_validate._sha256_file(path)


def _array_sha256(value: np.ndarray) -> str:
    return q60.q40_validate._array_sha256(value)


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


def _grid_shape(num_qubits: int) -> tuple[int, int]:
    if num_qubits < 2:
        raise RunnerError("Architecture screen needs at least two qubits")
    rows = max(
        divisor
        for divisor in range(1, int(math.sqrt(num_qubits)) + 1)
        if num_qubits % divisor == 0
    )
    return rows, num_qubits // rows


def _grid_matchings(num_qubits: int) -> list[list[tuple[int, int]]]:
    rows, columns = _grid_shape(num_qubits)
    horizontal = [
        [
            (row * columns + column, row * columns + column + 1)
            for row in range(rows)
            for column in range(parity, columns - 1, 2)
        ]
        for parity in (0, 1)
    ]
    vertical = [
        [
            (row * columns + column, (row + 1) * columns + column)
            for row in range(parity, rows - 1, 2)
            for column in range(columns)
        ]
        for parity in (0, 1)
    ]
    return [horizontal[0], vertical[0], horizontal[1], vertical[1]]


def _chord_matching(num_qubits: int) -> list[tuple[int, int]]:
    if num_qubits % 2:
        raise RunnerError("Scrambled matching requires an even qubit count")
    half = num_qubits // 2
    multiplier = next(
        value for value in (7, 5, 3, 1) if math.gcd(value, half) == 1
    )
    return [
        (index, half + ((multiplier * index + 3) % half))
        for index in range(half)
    ]


def interaction_layers(
    architecture: str, num_qubits: int
) -> list[list[tuple[int, int]]]:
    if architecture == "control_rx05_d6":
        return [
            [(left, left + 1) for left in range(parity, num_qubits - 1, 2)]
            for parity in (0, 1)
        ]
    if architecture == "grid_mixer_d12":
        return _grid_matchings(num_qubits)
    if architecture == "scrambled_mixer_d16":
        return [*_grid_matchings(num_qubits), _chord_matching(num_qubits)]
    raise RunnerError(f"Unknown architecture: {architecture}")


def architecture_circuit(
    linear_values: np.ndarray, architecture: str
) -> QuantumCircuit:
    values = np.asarray(linear_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise RunnerError("Architecture input must be a finite one-dimensional vector")
    if architecture == "control_rx05_d6":
        pair_values = values[:-1] * values[1:]
        return rx05.rx05_circuit(
            values,
            pair_values,
            single_scale=SINGLE_SCALE,
            phase_scale=PHASE_SCALE,
            pair_scale=PAIR_SCALE,
            post_scale=FINAL_RX_SCALE,
        )

    circuit = QuantumCircuit(len(values))
    circuit.h(range(len(values)))
    for qubit, value in enumerate(values):
        circuit.ry(SINGLE_SCALE * float(value), qubit)
        circuit.rz(PHASE_SCALE * float(value), qubit)
    for layer_index, edges in enumerate(interaction_layers(architecture, len(values))):
        for left, right in edges:
            circuit.rzz(
                PAIR_SCALE * float(values[left]) * float(values[right]), left, right
            )
        mixer = circuit.rx if layer_index % 2 == 0 else circuit.ry
        for qubit, value in enumerate(values):
            mixer(math.pi / 4.0 + MIXER_DATA_SCALE * float(value), qubit)
        if architecture == "scrambled_mixer_d16" and layer_index in {1, 3}:
            for qubit, value in enumerate(values):
                circuit.rz(REUPLOAD_SCALE * float(value), qubit)
    for qubit, value in enumerate(values):
        circuit.rx(FINAL_RX_SCALE * float(value), qubit)
    return circuit


def _structural_cone_sizes(
    architecture: str,
    mappings: Sequence[Mapping[int, str]],
    num_qubits: int,
) -> np.ndarray:
    layers = interaction_layers(architecture, num_qubits)
    sizes: list[int] = []
    for mapping in mappings:
        support = {int(qubit) for qubit in mapping}
        if architecture == "control_rx05_d6":
            # All path RZZ gates commute.  A newly introduced neighbour only
            # carries Z, so it does not spread again through the next RZZ.
            original = set(support)
            for edges in layers:
                for left, right in edges:
                    if left in original or right in original:
                        support.update((left, right))
        else:
            for edges in reversed(layers):
                for left, right in edges:
                    if left in support or right in support:
                        support.update((left, right))
        sizes.append(len(support))
    return np.asarray(sizes, dtype=np.int64)


def structural_hardness(
    architecture: str,
    mappings: Sequence[Mapping[int, str]],
    num_qubits: int,
) -> dict[str, Any]:
    layers = interaction_layers(architecture, num_qubits)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from(edge for layer in layers for edge in layer)
    treewidth, _ = nx.approximation.treewidth_min_fill_in(graph)
    cone_sizes = _structural_cone_sizes(architecture, mappings, num_qubits)
    crossings = np.zeros(num_qubits - 1, dtype=np.int64)
    for layer in layers:
        for left, right in layer:
            low, high = sorted((left, right))
            crossings[low:high] += 1
    max_crossings = int(np.max(crossings))
    return {
        "interaction_layers": len(layers),
        "interaction_edges_total_with_repeats": int(sum(map(len, layers))),
        "interaction_graph_edges": int(graph.number_of_edges()),
        "interaction_graph_connected": bool(nx.is_connected(graph)),
        "interaction_graph_treewidth_min_fill_upper_bound": int(treewidth),
        "causal_cone": {
            "minimum": int(np.min(cone_sizes)),
            "median": float(np.median(cone_sizes)),
            "maximum": int(np.max(cone_sizes)),
            "mean": float(np.mean(cone_sizes)),
        },
        "row_major_mps_cross_cut_rzz_count": {
            "maximum": max_crossings,
            "median": float(np.median(crossings)),
            "rzz_operator_schmidt_rank_upper_bound_at_max_cut": int(
                2**min(max_crossings, 60)
            ),
        },
        "claim_boundary": (
            "Interaction-graph treewidth and row-major cut counts are structural "
            "proxies, not proofs of asymptotic classical hardness."
        ),
    }


def _mapping_observable(
    mapping: Mapping[int, str]
) -> tuple[SparsePauliOp, list[int]]:
    qubits = sorted(int(qubit) for qubit in mapping)
    label = "".join(str(mapping[qubit]) for qubit in reversed(qubits))
    return SparsePauliOp(label), qubits


def _with_expectation_saves(
    circuit: QuantumCircuit, mappings: Sequence[Mapping[int, str]]
) -> QuantumCircuit:
    payload = circuit.copy()
    for index, mapping in enumerate(mappings):
        operator, qubits = _mapping_observable(mapping)
        payload.save_expectation_value(operator, qubits, label=f"e{index}")
    return payload


def _mps_simulator(bond_dimension: int, threshold: float) -> AerSimulator:
    return AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=int(bond_dimension),
        matrix_product_state_truncation_threshold=float(threshold),
        max_parallel_experiments=1,
    )


def simulate_feature_rows(
    circuits: Sequence[QuantumCircuit],
    mappings: Sequence[Mapping[int, str]],
    *,
    bond_dimension: int,
    threshold: float,
    progress_label: str | None = None,
) -> tuple[np.ndarray, float]:
    simulator = _mps_simulator(bond_dimension, threshold)
    features = np.empty((len(circuits), len(mappings)), dtype=np.float64)
    started = time.perf_counter()
    for row_index, circuit in enumerate(circuits):
        result = simulator.run(_with_expectation_saves(circuit, mappings)).result()
        if not result.success:
            raise RunnerError("Aer MPS simulation failed")
        data = result.data(0)
        features[row_index] = [float(np.real(data[f"e{index}"])) for index in range(len(mappings))]
        if progress_label and (
            row_index == 0 or (row_index + 1) % 8 == 0 or row_index + 1 == len(circuits)
        ):
            print(
                f"[{progress_label}] chi={bond_dimension} "
                f"{row_index + 1}/{len(circuits)}",
                flush=True,
            )
    if not np.all(np.isfinite(features)) or np.max(np.abs(features)) > 1.0 + 1e-8:
        raise RunnerError("MPS feature matrix violates finite expectation bounds")
    return features, time.perf_counter() - started


def statevector_features(
    circuit: QuantumCircuit, mappings: Sequence[Mapping[int, str]]
) -> np.ndarray:
    state = Statevector.from_instruction(circuit)
    values: list[float] = []
    for mapping in mappings:
        label = ["I"] * circuit.num_qubits
        for qubit, pauli in mapping.items():
            label[circuit.num_qubits - 1 - int(qubit)] = str(pauli)
        values.append(float(np.real(state.expectation_value(SparsePauliOp("".join(label))))))
    return np.asarray(values, dtype=np.float64)


def small_q_parity(architecture: str, *, num_qubits: int = 12) -> dict[str, Any]:
    values = np.linspace(-0.81, 0.73, num_qubits, dtype=np.float64)
    mappings = toy.pauli_feature_mappings(num_qubits, family="local")
    circuit = architecture_circuit(values, architecture)
    exact = statevector_features(circuit, mappings)
    mps, elapsed = simulate_feature_rows(
        [circuit], mappings, bond_dimension=64, threshold=1e-14
    )
    error = np.abs(exact - mps[0])
    return {
        "qubits": num_qubits,
        "observables": len(mappings),
        "statevector_norm": float(np.linalg.norm(Statevector.from_instruction(circuit).data)),
        "max_abs_mps64_minus_statevector": float(np.max(error)),
        "mean_abs_mps64_minus_statevector": float(np.mean(error)),
        "passed": bool(np.max(error) <= 1e-9),
        "mps_seconds": elapsed,
    }


def load_seed_data(args: argparse.Namespace) -> SeedData:
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    reupload_report = json.loads(args.reupload_report.read_text(encoding="utf-8"))
    configuration = rx05._configuration_from_reports(source_report, reupload_report)
    if int(configuration["seed"]) != 11 or int(configuration["qubits"]) != 60:
        raise RunnerError("Architecture screen is pinned to seed 11 and 60 qubits")
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(configuration["positive_label"]),
        negative_label=str(configuration["negative_label"]),
    )
    train_indices = np.asarray(source_report["split"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(source_report["split"]["test_indices"], dtype=np.int64)
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=60,
        hash_seed=11,
        value_mode=str(configuration["value_mode"]),
        max_active_genes=int(configuration["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=60,
        hash_seed=11,
        value_mode=str(configuration["value_mode"]),
        max_active_genes=int(configuration["max_active_genes"]),
    )
    if _array_sha256(encoded_train) != str(source_report["split"]["encoded_train_sha256"]):
        raise RunnerError("Training encoding hash does not reproduce")
    if _array_sha256(encoded_test) != str(source_report["split"]["encoded_test_sha256"]):
        raise RunnerError("Test encoding hash does not reproduce")
    return SeedData(
        encoded_train=np.asarray(encoded_train, dtype=np.float64),
        encoded_test=np.asarray(encoded_test, dtype=np.float64),
        y_train=np.asarray(y_pair[train_indices], dtype=np.float64),
        y_test=np.asarray(y_pair[test_indices], dtype=np.float64),
        train_indices=train_indices,
        test_indices=test_indices,
        metadata={
            "configuration": configuration,
            "source": {**source_meta, **pair_meta},
            "train_encoding_stats": train_stats,
            "test_encoding_stats": test_stats,
            "encoded_train_sha256": _array_sha256(encoded_train),
            "encoded_test_sha256": _array_sha256(encoded_test),
            "source_report_sha256": _sha256_file(args.source_report),
            "reupload_report_sha256": _sha256_file(args.reupload_report),
        },
    )


def _balanced_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.where(np.asarray(scores) >= 0.0, 1.0, -1.0)
    positive = labels > 0.0
    negative = labels < 0.0
    return {
        "balanced_accuracy": q60._balanced_accuracy(labels, scores),
        "accuracy": float(np.mean(predictions == labels)),
        "positive_recall": float(np.mean(predictions[positive] > 0.0)),
        "negative_recall": float(np.mean(predictions[negative] < 0.0)),
        "correct": int(np.sum(predictions == labels)),
        "samples": int(len(labels)),
    }


def classical_cv_reference(
    data: SeedData, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold_index, (fit, validation) in enumerate(folds):
        fit_scores, validation_scores = q60._ridge_scores(
            data.encoded_train[fit], data.encoded_train[validation], data.y_train[fit]
        )
        rows.append(
            {
                "fold": fold_index,
                "fit": _balanced_metrics(data.y_train[fit], fit_scores),
                "validation": _balanced_metrics(
                    data.y_train[validation], validation_scores
                ),
            }
        )
    values = [float(row["validation"]["balanced_accuracy"]) for row in rows]
    train_scores, test_scores = q60._ridge_scores(
        data.encoded_train, data.encoded_test, data.y_train
    )
    return {
        "model": "standardized ridge on the same 60 hashed bins",
        "folds": rows,
        "cv_mean_balanced_accuracy": float(np.mean(values)),
        "cv_worst_balanced_accuracy": float(np.min(values)),
        "full_train": _balanced_metrics(data.y_train, train_scores),
        "fixed_test": _balanced_metrics(data.y_test, test_scores),
    }


def _parameter_vectors(data: SeedData) -> list[np.ndarray]:
    sketch, _ = q60.sketch_parameters(data.encoded_train, data.y_train)
    return [sketch, *data.encoded_train, *data.encoded_test]


def _fold_sketch_vectors(
    data: SeedData, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for fit, _ in folds:
        sketch, _ = q60.sketch_parameters(data.encoded_train[fit], data.y_train[fit])
        output.append(sketch)
    return output


def candidate_cv(
    base_features: np.ndarray,
    fold_sketch_features: np.ndarray,
    data: SeedData,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    selected_feature_count: int,
    shot_intent: int,
) -> dict[str, Any]:
    query_train = base_features[1 : 1 + len(data.encoded_train)]
    rows: list[dict[str, Any]] = []
    for fold_index, (fit, validation) in enumerate(folds):
        selected, selection_scores = q60.select_train_only_features(
            query_train[fit],
            data.y_train[fit],
            count=selected_feature_count,
            shot_intent=shot_intent,
        )
        model = fold_sketch_features[fold_index, selected]
        head_fit = q60._head_features(model, query_train[fit][:, selected])
        head_validation = q60._head_features(
            model, query_train[validation][:, selected]
        )
        fit_scores, validation_scores = q60._ridge_scores(
            head_fit, head_validation, data.y_train[fit]
        )
        rows.append(
            {
                "fold": fold_index,
                "selected_candidate_indices": [int(value) for value in selected],
                "minimum_selected_score": float(np.min(selection_scores[selected])),
                "fit": _balanced_metrics(data.y_train[fit], fit_scores),
                "validation": _balanced_metrics(
                    data.y_train[validation], validation_scores
                ),
            }
        )
    values = [float(row["validation"]["balanced_accuracy"]) for row in rows]
    return {
        "folds": rows,
        "cv_mean_balanced_accuracy": float(np.mean(values)),
        "cv_std_balanced_accuracy": float(np.std(values)),
        "cv_worst_balanced_accuracy": float(np.min(values)),
        "selection_uses_training_fold_only": True,
    }


def final_winner_evaluation(
    features: np.ndarray,
    data: SeedData,
    *,
    selected_feature_count: int,
    shot_intent: int,
) -> dict[str, Any]:
    train_stop = 1 + len(data.encoded_train)
    query_train = features[1:train_stop]
    query_test = features[train_stop:]
    selected, selection_scores = q60.select_train_only_features(
        query_train,
        data.y_train,
        count=selected_feature_count,
        shot_intent=shot_intent,
    )
    model = features[0, selected]
    head_train = q60._head_features(model, query_train[:, selected])
    head_test = q60._head_features(model, query_test[:, selected])
    train_scores, test_scores = q60._ridge_scores(
        head_train, head_test, data.y_train
    )
    return {
        "selected_candidate_indices": [int(value) for value in selected],
        "minimum_selected_score": float(np.min(selection_scores[selected])),
        "selection_uses_training_only": True,
        "train": _balanced_metrics(data.y_train, train_scores),
        "fixed_test": _balanced_metrics(data.y_test, test_scores),
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.bond_dimension < 2 or any(value < 2 for value in args.probe_bond_dimensions):
        raise RunnerError("MPS bond dimensions must be at least two")
    if args.convergence_tolerance <= 0.0 or args.mps_threshold <= 0.0:
        raise RunnerError("MPS tolerances must be positive")
    started = time.perf_counter()
    data = load_seed_data(args)
    mappings = toy.pauli_feature_mappings(60, family="local")
    splitter = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed
    )
    folds = list(splitter.split(data.encoded_train, data.y_train))
    classical = classical_cv_reference(data, folds)
    vectors = _parameter_vectors(data)
    fold_vectors = _fold_sketch_vectors(data, folds)
    probe_indices = [0, 1, 8, 16, 32, 33, 48, 64]
    candidates: list[dict[str, Any]] = []
    feature_cache: dict[str, np.ndarray] = {}

    for architecture in args.architectures:
        print(f"[{architecture}] building circuits", flush=True)
        circuits = [architecture_circuit(vector, architecture) for vector in vectors]
        fold_circuits = [
            architecture_circuit(vector, architecture) for vector in fold_vectors
        ]
        representative_metrics = q60.q40_validate.circuit_metrics(circuits[0])
        small_q = small_q_parity(architecture)
        features, full_seconds = simulate_feature_rows(
            [*circuits, *fold_circuits],
            mappings,
            bond_dimension=args.bond_dimension,
            threshold=args.mps_threshold,
            progress_label=architecture,
        )
        base_features = features[: len(circuits)]
        fold_features = features[len(circuits) :]
        feature_cache[architecture] = base_features
        probe_rows: dict[int, dict[str, Any]] = {}
        for bond_dimension in args.probe_bond_dimensions:
            probe, probe_seconds = simulate_feature_rows(
                [circuits[index] for index in probe_indices],
                mappings,
                bond_dimension=bond_dimension,
                threshold=args.mps_threshold,
                progress_label=f"{architecture}-probe",
            )
            difference = np.abs(probe - base_features[probe_indices])
            probe_rows[int(bond_dimension)] = {
                "seconds": probe_seconds,
                "max_abs_difference_from_primary": float(np.max(difference)),
                "mean_abs_difference_from_primary": float(np.mean(difference)),
                "median_abs_difference_from_primary": float(np.median(difference)),
            }
        high_probe = max(args.probe_bond_dimensions)
        converged = bool(
            probe_rows[int(high_probe)]["max_abs_difference_from_primary"]
            <= args.convergence_tolerance
        )
        cv = candidate_cv(
            base_features,
            fold_features,
            data,
            folds,
            selected_feature_count=args.selected_features,
            shot_intent=args.shot_intent,
        )
        candidates.append(
            {
                "architecture": architecture,
                "representative_circuit_metrics": representative_metrics,
                "structural_hardness": structural_hardness(
                    architecture, mappings, 60
                ),
                "small_q_statevector_mps_parity": small_q,
                "mps": {
                    "primary_bond_dimension": args.bond_dimension,
                    "truncation_threshold": args.mps_threshold,
                    "primary_full_feature_seconds": full_seconds,
                    "probe_base_circuit_indices": probe_indices,
                    "probe_comparisons": {
                        str(key): value for key, value in probe_rows.items()
                    },
                    "convergence_tolerance": args.convergence_tolerance,
                    "primary_converged_against_high_probe": converged,
                    "accuracy_is_bounded_mps_estimate": True,
                },
                "training_cross_validation": cv,
            }
        )

    rankable = [
        row
        for row in candidates
        if row["mps"]["primary_converged_against_high_probe"] is True
        and row["small_q_statevector_mps_parity"]["passed"] is True
    ]
    winner = max(
        rankable,
        key=lambda row: (
            float(row["training_cross_validation"]["cv_mean_balanced_accuracy"]),
            float(row["training_cross_validation"]["cv_worst_balanced_accuracy"]),
            -ARCHITECTURES.index(str(row["architecture"])),
        ),
        default=None,
    )
    final_evaluation: dict[str, Any] | None = None
    gates: dict[str, Any]
    if winner is None:
        gates = {
            "mps_rankable_candidate_exists": False,
            "ready_for_provider_validation": False,
            "reason": "No candidate passed the bounded-MPS convergence gate.",
        }
    else:
        winner_name = str(winner["architecture"])
        final_evaluation = final_winner_evaluation(
            feature_cache[winner_name],
            data,
            selected_feature_count=args.selected_features,
            shot_intent=args.shot_intent,
        )
        quantum_cv_mean = float(
            winner["training_cross_validation"]["cv_mean_balanced_accuracy"]
        )
        quantum_cv_worst = float(
            winner["training_cross_validation"]["cv_worst_balanced_accuracy"]
        )
        classical_cv_mean = float(classical["cv_mean_balanced_accuracy"])
        classical_cv_worst = float(classical["cv_worst_balanced_accuracy"])
        cone_median = float(
            winner["structural_hardness"]["causal_cone"]["median"]
        )
        gates = {
            "mps_rankable_candidate_exists": True,
            "training_cv_mean_beats_classical": quantum_cv_mean > classical_cv_mean,
            "training_cv_worst_not_below_classical": quantum_cv_worst >= classical_cv_worst,
            "median_causal_cone_at_least_16": cone_median >= 16.0,
            "cross_seed_confirmation_available": False,
        }
        gates["ready_for_provider_validation"] = bool(
            gates["training_cv_mean_beats_classical"]
            and gates["training_cv_worst_not_below_classical"]
            and gates["median_causal_cone_at_least_16"]
            and gates["cross_seed_confirmation_available"]
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_noncommuting_architecture_local_screen",
        "status": "complete_local_only",
        "captured_at_utc": _utc_now(),
        "environment": {
            **q60.q40_validate.runtime_environment(),
            "platform": platform.platform(),
        },
        "provider_calls": [],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "config": {
            "seed": 11,
            "qubits": 60,
            "architectures": list(args.architectures),
            "selected_features": args.selected_features,
            "shot_intent": args.shot_intent,
            "cv_folds": args.cv_folds,
            "cv_seed": args.cv_seed,
            "mps_primary_bond_dimension": args.bond_dimension,
            "mps_probe_bond_dimensions": list(args.probe_bond_dimensions),
            "mps_truncation_threshold": args.mps_threshold,
            "mps_convergence_tolerance": args.convergence_tolerance,
        },
        "source": data.metadata,
        "classical_same_split_reference": classical,
        "candidates": candidates,
        "selection": {
            "uses_training_cv_only": True,
            "test_metrics_used_for_architecture_selection": False,
            "winner": None if winner is None else winner["architecture"],
            "fixed_test_evaluated_for_winner_only": final_evaluation is not None,
        },
        "winner_final_evaluation": final_evaluation,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is a bounded local MPS screen, not proof of classical hardness. "
            "A candidate cannot advance to provider validation until accuracy, "
            "structural, MPS-convergence, and cross-seed gates all pass."
        ),
    }
    _atomic_write_json(args.output, report)
    return report


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return parsed


def _parse_architectures(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [item for item in parsed if item not in ARCHITECTURES]
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown architectures: {unknown}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--reupload-report", type=Path, default=DEFAULT_REUPLOAD_REPORT)
    parser.add_argument("--architectures", type=_parse_architectures, default=ARCHITECTURES)
    parser.add_argument("--bond-dimension", type=int, default=DEFAULT_BOND_DIMENSION)
    parser.add_argument(
        "--probe-bond-dimensions",
        type=_parse_int_tuple,
        default=DEFAULT_PROBE_BOND_DIMENSIONS,
    )
    parser.add_argument("--mps-threshold", type=float, default=DEFAULT_MPS_THRESHOLD)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument("--selected-features", type=int, default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--shot-intent", type=int, default=DEFAULT_SHOT_INTENT)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_screen(args)
    print("PBMC68k q60 non-commuting architecture screen complete", flush=True)
    print(f"- winner: {report['selection']['winner']}", flush=True)
    print(
        "- ready for provider validation: "
        f"{report['gates']['ready_for_provider_validation']}",
        flush=True,
    )
    print(f"- provider calls: {len(report['provider_calls'])}", flush=True)
    print(f"- output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
