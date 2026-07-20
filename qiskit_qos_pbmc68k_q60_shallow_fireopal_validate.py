#!/usr/bin/env python3
"""Prepare and validate a shallow 60q PBMC68k Fire Opal batch.

This route deliberately has no hardware-execution path. It schedules the
commuting RZZ chain as even/odd matchings, measures each logical circuit in
global X/Y/Z bases, and calls at most ``fireopal.validate`` after local gates.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q40_fireopal_validate as q40_validate
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SEEDS = (11,)
DEFAULT_QUBITS = 60
DEFAULT_TRAIN_SAMPLES = 32
DEFAULT_TEST_SAMPLES = 32
DEFAULT_ACTIVE_GENES = 256
DEFAULT_SELECTED_FEATURES = 16
DEFAULT_READOUT_SHOTS = 1024
DEFAULT_SEED_TRANSPILER = 1729
DEFAULT_MAX_PAYLOAD_DEPTH = 24
MEASUREMENT_BASES = ("X", "Y", "Z")
FIRE_OPAL_MAX_BATCH = 300
SINGLE_SCALE = 1.35
PHASE_SCALE = 0.75
PAIR_SCALE = 0.95

RunnerError = q40_validate.RunnerError


def _normalized_mapping(mapping: Mapping[int, str]) -> list[dict[str, Any]]:
    return [
        {"qubit": int(qubit), "pauli": str(pauli)}
        for qubit, pauli in sorted(mapping.items())
    ]


def append_even_odd_rzz(
    circuit: QuantumCircuit, pair_values: np.ndarray, *, pair_scale: float
) -> None:
    """Append commuting path interactions in two matching layers."""
    for parity in (0, 1):
        for left in range(parity, len(pair_values), 2):
            circuit.rzz(pair_scale * float(pair_values[left]), left, left + 1)


def shallow_circuit(
    linear_values: np.ndarray,
    pair_values: np.ndarray,
    *,
    single_scale: float = SINGLE_SCALE,
    phase_scale: float = PHASE_SCALE,
    pair_scale: float = PAIR_SCALE,
) -> QuantumCircuit:
    linear_values = np.asarray(linear_values, dtype=np.float64)
    pair_values = np.asarray(pair_values, dtype=np.float64)
    if pair_values.shape != (max(len(linear_values) - 1, 0),):
        raise RunnerError("Pair values do not match the qubit path")
    circuit = QuantumCircuit(len(linear_values))
    circuit.h(range(len(linear_values)))
    for qubit, value in enumerate(linear_values):
        circuit.ry(single_scale * float(value), qubit)
        circuit.rz(phase_scale * float(value), qubit)
    append_even_odd_rzz(circuit, pair_values, pair_scale=pair_scale)
    return circuit


def query_parameters(encoded_sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = np.asarray(encoded_sample, dtype=np.float64)
    return linear, linear[:-1] * linear[1:]


def sketch_parameters(
    encoded_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if encoded_train.ndim != 2 or len(encoded_train) != len(y_train):
        raise RunnerError("Training matrix and labels are inconsistent")
    weight_l1 = float(np.sum(np.abs(y_train)))
    if weight_l1 <= 0.0:
        raise RunnerError("Weighted training sketch is empty")
    linear = np.sum(y_train[:, None] * encoded_train, axis=0) / weight_l1
    products = encoded_train[:, :-1] * encoded_train[:, 1:]
    pair = np.sum(y_train[:, None] * products, axis=0) / weight_l1
    return linear, pair


def measurement_circuit_for_basis(
    base_circuit: QuantumCircuit, basis: str
) -> QuantumCircuit:
    circuit = base_circuit.copy()
    if basis == "X":
        circuit.h(range(circuit.num_qubits))
    elif basis == "Y":
        circuit.sdg(range(circuit.num_qubits))
        circuit.h(range(circuit.num_qubits))
    elif basis != "Z":
        raise RunnerError(f"Unsupported global measurement basis: {basis}")
    circuit.measure_all()
    return circuit


def measurement_basis_for_mapping(mapping: Mapping[int, str]) -> str:
    bases = {str(value) for value in mapping.values()}
    if len(bases) != 1 or next(iter(bases)) not in MEASUREMENT_BASES:
        raise RunnerError("Mapping cannot be recovered from one global basis")
    return next(iter(bases))


def exact_local_expectation(
    linear_values: np.ndarray,
    pair_values: np.ndarray,
    mapping: Mapping[int, str],
    *,
    single_scale: float = SINGLE_SCALE,
    phase_scale: float = PHASE_SCALE,
    pair_scale: float = PAIR_SCALE,
) -> float:
    """Evaluate a local Pauli using its at-most-four-qubit causal cone."""
    linear_values = np.asarray(linear_values, dtype=np.float64)
    pair_values = np.asarray(pair_values, dtype=np.float64)
    num_qubits = len(linear_values)
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
        circuit.ry(single_scale * value, local_qubit)
        circuit.rz(phase_scale * value, local_qubit)
    for left in incident:
        circuit.rzz(
            pair_scale * float(pair_values[left]),
            local_index[left],
            local_index[left + 1],
        )
    label = ["I"] * len(active)
    for global_qubit, pauli in mapping.items():
        local_qubit = local_index[int(global_qubit)]
        label[len(active) - 1 - local_qubit] = str(pauli)
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(SparsePauliOp("".join(label)))
    return float(np.real_if_close(value).real)


def exact_local_feature_matrix(
    parameter_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    mappings: Sequence[Mapping[int, str]],
) -> np.ndarray:
    return np.asarray(
        [
            [exact_local_expectation(linear, pair, mapping) for mapping in mappings]
            for linear, pair in parameter_rows
        ],
        dtype=np.float64,
    )


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.where(np.asarray(scores) >= 0.0, 1.0, -1.0)
    positive = labels > 0.0
    negative = labels < 0.0
    if not np.any(positive) or not np.any(negative):
        raise RunnerError("Balanced accuracy needs both classes")
    return float(
        0.5
        * (
            np.mean(predictions[positive] > 0.0)
            + np.mean(predictions[negative] < 0.0)
        )
    )


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _ridge_scores(
    train: np.ndarray, test: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_z, test_z = _standardize_train_test(train, test)
    gram = train_z.T @ train_z + 1e-2 * np.eye(train_z.shape[1])
    weights = np.linalg.solve(gram, train_z.T @ y_train)
    return train_z @ weights, test_z @ weights


def _head_features(model: np.ndarray, queries: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    model_norm = float(np.linalg.norm(model))
    for query in np.asarray(queries, dtype=np.float64):
        denominator = model_norm * float(np.linalg.norm(query))
        cosine = 0.0 if denominator <= 1e-12 else float(model @ query / denominator)
        rows.append(np.concatenate([query, model * query, [cosine]]))
    return np.asarray(rows, dtype=np.float64)


def select_train_only_features(
    query_train: np.ndarray,
    y_train: np.ndarray,
    *,
    count: int,
    shot_intent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank ideal observables with training labels only and a shot-noise floor."""
    query_train = np.asarray(query_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if count < 1 or count > query_train.shape[1]:
        raise RunnerError("Selected-feature count is outside the candidate set")
    positive = query_train[y_train > 0.0]
    negative = query_train[y_train < 0.0]
    if not len(positive) or not len(negative):
        raise RunnerError("Feature selection needs both training classes")
    difference = np.abs(np.mean(positive, axis=0) - np.mean(negative, axis=0))
    shot_floor = 1.0 / np.sqrt(float(shot_intent))
    uncertainty = np.sqrt(
        np.var(positive, axis=0) / len(positive)
        + np.var(negative, axis=0) / len(negative)
        + 2.0 * shot_floor**2
    )
    scores = difference / np.maximum(uncertainty, 1e-12)
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:count].astype(np.int64), scores


def verify_even_odd_equivalence() -> dict[str, Any]:
    values = np.linspace(-0.73, 0.81, 6, dtype=np.float64)
    sequential = toy.query_circuit(
        values,
        single_scale=SINGLE_SCALE,
        phase_scale=PHASE_SCALE,
        pair_scale=PAIR_SCALE,
    )
    linear, pair = query_parameters(values)
    shallow = shallow_circuit(linear, pair)
    state_a = Statevector.from_instruction(sequential)
    state_b = Statevector.from_instruction(shallow)
    fidelity = float(abs(np.vdot(state_a.data, state_b.data)) ** 2)
    return {
        "probe_qubits": 6,
        "statevector_fidelity": fidelity,
        "passed": bool(fidelity >= 1.0 - 1e-12),
        "reason": "All nearest-neighbour RZZ gates commute; only their schedule changed.",
    }


def build_seed_circuits(
    *,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    seed: int,
    hash_seed: int,
    selected_feature_count: int,
    shot_intent: int,
    circuit_offset: int = 0,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    encoded_test = np.asarray(encoded_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if encoded_train.ndim != 2 or encoded_test.ndim != 2:
        raise RunnerError("Encoded PBMC matrices must be two-dimensional")
    if encoded_train.shape[1] != encoded_test.shape[1]:
        raise RunnerError("Train/test encoded dimensions differ")
    if len(encoded_train) != len(y_train) or len(encoded_test) != len(y_test):
        raise RunnerError("Encoded matrices and labels differ in length")

    sketch_linear, sketch_pair = sketch_parameters(encoded_train, y_train)
    base_rows: list[dict[str, Any]] = [
        {
            "circuit": shallow_circuit(sketch_linear, sketch_pair),
            "parameters": (sketch_linear, sketch_pair),
            "role": "weighted_training_sketch",
            "split": "train",
            "sample_position": None,
            "source_row_index": None,
            "label": None,
        }
    ]
    for split, encoded, labels, indices in (
        ("train", encoded_train, y_train, train_indices),
        ("test", encoded_test, y_test, test_indices),
    ):
        for position, (sample, label, source_index) in enumerate(
            zip(encoded, labels, indices, strict=True)
        ):
            linear, pair = query_parameters(sample)
            base_rows.append(
                {
                    "circuit": shallow_circuit(linear, pair),
                    "parameters": (linear, pair),
                    "role": "query",
                    "split": split,
                    "sample_position": int(position),
                    "source_row_index": int(source_index),
                    "label": float(label),
                }
            )

    mappings = toy.pauli_feature_mappings(encoded_train.shape[1], family="local")
    parameter_rows = [row["parameters"] for row in base_rows]
    features = exact_local_feature_matrix(parameter_rows, mappings)
    train_start = 1
    train_stop = train_start + len(encoded_train)
    test_stop = train_stop + len(encoded_test)
    selected, selection_scores = select_train_only_features(
        features[train_start:train_stop],
        y_train,
        count=selected_feature_count,
        shot_intent=shot_intent,
    )

    model = features[0, selected]
    query_train = features[train_start:train_stop, selected]
    query_test = features[train_stop:test_stop, selected]
    head_train = _head_features(model, query_train)
    head_test = _head_features(model, query_test)
    train_scores, test_scores = _ridge_scores(head_train, head_test, y_train)
    classical_train_scores, classical_test_scores = _ridge_scores(
        encoded_train, encoded_test, y_train
    )

    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    for base_index, row in enumerate(base_rows):
        logical_metrics = q40_validate.circuit_metrics(row["circuit"])
        for basis in MEASUREMENT_BASES:
            measured = measurement_circuit_for_basis(row["circuit"], basis)
            qasm, qasm_metadata = q40_validate.export_numeric_qasm2(
                measured, seed_transpiler=seed_transpiler
            )
            qasms.append(qasm)
            manifest.append(
                {
                    "circuit_index": circuit_offset + len(qasms) - 1,
                    "seed": int(seed),
                    "hash_seed": int(hash_seed),
                    "base_circuit_index": int(base_index),
                    "measurement_basis": basis,
                    "logical_metrics_before_measurement": logical_metrics,
                    "role": row["role"],
                    "split": row["split"],
                    "sample_position": row["sample_position"],
                    "source_row_index": row["source_row_index"],
                    "label": row["label"],
                    **qasm_metadata,
                }
            )

    selected_rows = [
        {
            "candidate_index": int(index),
            "selection_score": float(selection_scores[index]),
            "measurement_basis": measurement_basis_for_mapping(mappings[index]),
            "pauli_mapping": _normalized_mapping(mappings[index]),
        }
        for index in selected
    ]
    return qasms, manifest, {
        "seed": int(seed),
        "hash_seed": int(hash_seed),
        "logical_base_circuit_count": len(base_rows),
        "measurement_bases": list(MEASUREMENT_BASES),
        "measured_circuit_count": len(qasms),
        "candidate_observable_count": len(mappings),
        "selected_observable_count": len(selected),
        "selected_observables": selected_rows,
        "selection_protocol": {
            "uses_training_labels_only": True,
            "uses_test_labels": False,
            "ranking": (
                "absolute class-mean difference divided by training variance "
                "and a shot-noise floor"
            ),
            "future_shot_intent": int(shot_intent),
        },
        "ideal_causal_cone_screen": {
            "method": "exact statevector on at-most-four-qubit causal cones",
            "head": (
                "standardized ridge on selected query, sketch-query interaction, "
                "and cosine"
            ),
            "train_balanced_accuracy": _balanced_accuracy(y_train, train_scores),
            "test_balanced_accuracy": _balanced_accuracy(y_test, test_scores),
            "classical_60bin_train_balanced_accuracy": _balanced_accuracy(
                y_train, classical_train_scores
            ),
            "classical_60bin_test_balanced_accuracy": _balanced_accuracy(
                y_test, classical_test_scores
            ),
            "claim_boundary": "Noiseless local diagnostic only; not a hardware prediction.",
        },
        "encoded_train_sha256": q40_validate._array_sha256(encoded_train),
        "encoded_test_sha256": q40_validate._array_sha256(encoded_test),
    }


def prepare_batch(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    all_qasms: list[str] = []
    all_manifest: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_idx, test_idx = genomics_runner.benchmark_indices(
            x_pair.shape[0],
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            labels=y_pair,
        )
        y_train = y_pair[train_idx].astype(np.float64)
        y_test = y_pair[test_idx].astype(np.float64)
        encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_pair[train_idx],
            feature_dim=args.qubits,
            hash_seed=seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_pair[test_idx],
            feature_dim=args.qubits,
            hash_seed=seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        qasms, manifest, seed_meta = build_seed_circuits(
            encoded_train=encoded_train,
            encoded_test=encoded_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=train_idx,
            test_indices=test_idx,
            seed=seed,
            hash_seed=seed,
            selected_feature_count=args.selected_features,
            shot_intent=args.readout_shots,
            circuit_offset=len(all_qasms),
            seed_transpiler=args.seed_transpiler,
        )
        all_qasms.extend(qasms)
        all_manifest.extend(manifest)
        seed_rows.append(
            {
                **seed_meta,
                "train_indices": [int(value) for value in train_idx],
                "test_indices": [int(value) for value in test_idx],
                "train_class_balance": {
                    "positive": int(np.sum(y_train > 0.0)),
                    "negative": int(np.sum(y_train < 0.0)),
                },
                "test_class_balance": {
                    "positive": int(np.sum(y_test > 0.0)),
                    "negative": int(np.sum(y_test < 0.0)),
                },
                "train_encoding_stats": train_stats,
                "test_encoding_stats": test_stats,
            }
        )

    if not all_qasms or len(all_qasms) > FIRE_OPAL_MAX_BATCH:
        raise RunnerError(
            f"Fire Opal batch size {len(all_qasms)} is outside 1..{FIRE_OPAL_MAX_BATCH}"
        )
    equivalence = verify_even_odd_equivalence()
    payload_depths = [int(row["metrics"]["depth"]) for row in all_manifest]
    logical_depths = [
        int(row["logical_metrics_before_measurement"]["depth"])
        for row in all_manifest
    ]
    qasm_hashes = [row["qasm_sha256"] for row in all_manifest]
    aggregate_sha256 = q40_validate._sha256_bytes(
        json.dumps(qasm_hashes, separators=(",", ":")).encode("utf-8")
    )
    local_validation = {
        "circuit_count": len(all_qasms),
        "batch_limit": FIRE_OPAL_MAX_BATCH,
        "within_batch_limit": True,
        "aggregate_sha256": aggregate_sha256,
        "total_qasm_bytes": sum(int(row["qasm_bytes"]) for row in all_manifest),
        "max_logical_depth_before_measurement": max(logical_depths),
        "max_payload_depth": max(payload_depths),
        "payload_depth_gate": int(args.max_payload_depth),
        "payload_depth_gate_passed": max(payload_depths) <= args.max_payload_depth,
        "even_odd_unitary_equivalence": equivalence,
        "all_target_qubits": all(
            row["metrics"]["num_qubits"] == args.qubits for row in all_manifest
        ),
        "all_one_classical_register": all(
            row["classical_register_count"] == 1 for row in all_manifest
        ),
        "all_virtual_qubits_only": all(
            row["virtual_qubits_only"] for row in all_manifest
        ),
        "all_parameters_numeric": all(
            row["all_parameters_numeric"] for row in all_manifest
        ),
        "all_round_trips_validated": all(
            row["round_trip_validated"] for row in all_manifest
        ),
        "all_expected_measurement_bases": (
            {row["measurement_basis"] for row in all_manifest}
            == set(MEASUREMENT_BASES)
        ),
    }
    local_validation["passed"] = bool(
        local_validation["within_batch_limit"]
        and local_validation["payload_depth_gate_passed"]
        and equivalence["passed"]
        and local_validation["all_target_qubits"]
        and local_validation["all_one_classical_register"]
        and local_validation["all_virtual_qubits_only"]
        and local_validation["all_parameters_numeric"]
        and local_validation["all_round_trips_validated"]
        and local_validation["all_expected_measurement_bases"]
    )
    config = {
        "dataset": "PBMC68k",
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "qubits": args.qubits,
        "seeds": list(args.seeds),
        "hash_seed_policy": "equal_to_split_seed",
        "train_fraction": args.train_fraction,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "max_active_genes": args.max_active_genes,
        "value_mode": args.value_mode,
        "entangler_schedule": "even_then_odd_nearest_neighbour_rzz",
        "measurement_bases": list(MEASUREMENT_BASES),
        "candidate_observables": (
            "all single X/Y/Z and nearest-neighbour XX/ZZ"
        ),
        "selected_features": args.selected_features,
        "future_readout_shots": args.readout_shots,
        "backend": args.backend,
    }
    return all_qasms, {
        "config": config,
        "source": {
            **source_meta,
            **pair_meta,
            "cache_artifacts": q40_validate._dataset_artifacts(args.cache_dir),
        },
        "seeds": seed_rows,
        "manifest": all_manifest,
        "local_validation": local_validation,
    }


def write_qasm_bundle(
    path: Path, qasms: Sequence[str], prepared: Mapping[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_fireopal_numeric_qasm2_batch",
        "config": prepared["config"],
        "source": prepared["source"],
        "seeds": prepared["seeds"],
        "circuits": [
            {**row, "qasm": qasm}
            for row, qasm in zip(prepared["manifest"], qasms, strict=True)
        ],
    }
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    reloaded_circuits = reloaded.get("circuits", [])
    if len(reloaded_circuits) != len(qasms):
        raise RunnerError("QASM bundle round trip changed circuit count")
    hashes = [
        q40_validate._sha256_bytes(str(row["qasm"]).encode("utf-8"))
        for row in reloaded_circuits
    ]
    if hashes != [row["qasm_sha256"] for row in prepared["manifest"]]:
        raise RunnerError("QASM bundle round trip changed circuit content")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": q40_validate._sha256_file(path),
        "circuits": len(qasms),
        "gzip_json_round_trip_passed": True,
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative integers"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--qubits", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--seeds", type=_parse_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=DEFAULT_TEST_SAMPLES
    )
    parser.add_argument(
        "--max-active-genes", type=int, default=DEFAULT_ACTIVE_GENES
    )
    parser.add_argument(
        "--value-mode", choices=("binary", "log-product"), default="log-product"
    )
    parser.add_argument(
        "--selected-features", type=int, default=DEFAULT_SELECTED_FEATURES
    )
    parser.add_argument("--readout-shots", type=int, default=DEFAULT_READOUT_SHOTS)
    parser.add_argument(
        "--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER
    )
    parser.add_argument(
        "--max-payload-depth", type=int, default=DEFAULT_MAX_PAYLOAD_DEPTH
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--qiskit-account", default="default-ibm-cloud")
    parser.add_argument("--qctrl-notebook", type=Path)
    parser.add_argument("--instance")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q60_shallow/"
            "pbmc68k_q60_seed11_shallow_fireopal_qasm2.json.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q60_shallow/"
            "pbmc68k_q60_seed11_shallow_fireopal_validate.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.qubits < 2
        or args.selected_features < 1
        or args.readout_shots < 1
        or args.max_payload_depth < 1
    ):
        raise RunnerError(
            "Qubits, selected features, shots, and depth gate must be positive"
        )
    for path in (args.bundle, args.output):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")

    started = time.perf_counter()
    qasms, prepared = prepare_batch(args)
    bundle = write_qasm_bundle(args.bundle, qasms, prepared)
    provider: dict[str, Any] = {
        "requested": bool(args.validate),
        "passed": None,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "api_calls": [],
    }
    if args.validate and prepared["local_validation"]["passed"]:
        try:
            provider = {
                "requested": True,
                **q40_validate.validate_fireopal_batch(
                    qasms,
                    backend=args.backend,
                    qiskit_account=args.qiskit_account,
                    qctrl_notebook=args.qctrl_notebook,
                    instance=args.instance,
                ),
            }
        except RunnerError as exc:
            provider = {
                "requested": True,
                "passed": False,
                "execution_attempted": False,
                "quantum_seconds_used": 0,
                "api_calls": [],
                "errors": [str(exc)],
                "warnings": [],
            }
    elif args.validate:
        provider = {
            "requested": True,
            "passed": False,
            "execution_attempted": False,
            "quantum_seconds_used": 0,
            "api_calls": [],
            "errors": ["Provider validation skipped because local gates failed"],
            "warnings": [],
        }

    passed = bool(prepared["local_validation"]["passed"])
    if args.validate:
        passed = passed and provider.get("passed") is True
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_fireopal_validate_only",
        "status": "pass" if passed else "fail",
        "captured_at_utc": q40_validate._utc_now(),
        "environment": q40_validate.runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "allowed_provider_calls": [
            "fireopal.show_supported_devices",
            "fireopal.validate",
        ],
        "config": prepared["config"],
        "source": prepared["source"],
        "seeds": prepared["seeds"],
        "local_validation": prepared["local_validation"],
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "A passing provider validation establishes input compatibility only; "
            "the ideal causal-cone screen is not a hardware result or evidence "
            "of advantage."
        ),
    }
    q40_validate._atomic_write_json(args.output, report)
    print("PBMC68k 60q shallow Fire Opal validation route")
    print(f"- circuits: {len(qasms)}")
    print(
        f"- max payload depth: "
        f"{prepared['local_validation']['max_payload_depth']}"
    )
    print(
        f"- local validation passed: "
        f"{prepared['local_validation']['passed']}"
    )
    print(f"- provider validation requested: {args.validate}")
    if args.validate:
        print(f"- provider validation passed: {provider.get('passed')}")
        print(f"- validation action: {provider.get('validation_action_id')}")
        print(f"- warnings: {len(provider.get('warnings', []))}")
    print("- hardware execution attempted: False")
    print(f"- bundle: {args.bundle}")
    print(f"- report: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
