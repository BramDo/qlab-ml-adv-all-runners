#!/usr/bin/env python3
"""Five-split real-data gate for a scalable shallow projected QOS kernel.

The official flat-QOS interference distribution needs ``2**q`` explicit
amplitudes or probabilities and therefore cannot be taken literally to q40 or
q60.  This provider-free route keeps the repository's sparse interaction
hashing and frozen shallow circuit family, but reads it out through a
bounded local Pauli-shadow panel.  All observables are recoverable
from three global X/Y/Z measurement circuits per sample.

The projected kernel is an RBF kernel on training-standardised expectation
vectors.  Observable count, RBF scale, and SVC regularisation are selected by
training-only cross-validation.  The held-out gate is the same conservative
four-of-five rule used by the exact small-width flat-QOS screen.  This module
contains no provider authentication, validation, QASM export, or execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import qiskit_official_qos_realdata_gate as flat_gate
import qiskit_qos_gse132080_thirdorder_screen as thirdorder
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as shallow
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
KIND = "shallow_projected_qos_realdata_five_split_gate"
DEFAULT_ARCHITECTURE = "grid_mixer_d12"
SUPPORTED_ARCHITECTURES = ("grid_mixer_d12", "grid_mixer_d8", "path_rzz_d4")
DEFAULT_OBSERVABLE_PANEL = "local_5q_minus_2"
SUPPORTED_OBSERVABLE_PANELS = (
    DEFAULT_OBSERVABLE_PANEL,
    "multiscale_support4",
)
OBSERVABLE_PANEL_DESCRIPTIONS = {
    DEFAULT_OBSERVABLE_PANEL: (
        "homogeneous X/Y/Z singles and nearest-neighbour XX/ZZ expectations "
        "(5q-2 observables)"
    ),
    "multiscale_support4": (
        "homogeneous X/Y/Z singles plus multiscale pair, triplet, and quartet "
        "expectations with support at most four"
    ),
}
DEFAULT_WIDTHS = (40, 60)
DEFAULT_VALIDATION_WIDTHS = (4, 6, 8, 10)
DEFAULT_FEATURE_COUNTS = (16, 32, 64, 128)
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
DEFAULT_GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)
DEFAULT_BOND_DIMENSION = 64
DEFAULT_BOND_DIMENSION_CANDIDATES = (32, 64, 128, 256)
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_SHOT_INTENT = 128
DEFAULT_OUTPUT = Path(
    "realdata_projected_qos_gate/"
    "q40_q60_projected_kernel_five_split_gate.json"
)
GLOBAL_MEASUREMENT_BASES = ("X", "Y", "Z")
FIRE_OPAL_MAX_BATCH = 300


class ProjectedKernelError(RuntimeError):
    pass


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or not all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return parsed


def _parse_datasets(value: str) -> tuple[str, ...]:
    return flat_gate._parse_datasets(value)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.shape}|".encode("utf-8"))
    digest.update(array.tobytes())
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


def _mapping_key(mapping: Mapping[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((int(qubit), str(pauli)) for qubit, pauli in mapping.items()))


def homogeneous_multiscale_mappings(num_qubits: int) -> list[dict[int, str]]:
    """Return a width-generic X/Y/Z shadow panel with bounded support.

    Every mapping is homogeneous in its Pauli basis, so the entire panel can
    be reconstructed from only three global measurement settings per circuit.
    The largest support is four, independent of ``num_qubits``.
    """

    if num_qubits < 4:
        raise ValueError("the multiscale panel requires at least four qubits")
    mappings: list[dict[int, str]] = []
    seen: set[tuple[tuple[int, str], ...]] = set()

    def add(mapping: Mapping[int, str]) -> None:
        normalized = {int(qubit): str(pauli) for qubit, pauli in mapping.items()}
        key = _mapping_key(normalized)
        if key not in seen:
            seen.add(key)
            mappings.append(normalized)

    for qubit in range(num_qubits):
        for pauli in GLOBAL_MEASUREMENT_BASES:
            add({qubit: pauli})

    distances = sorted(
        {
            distance
            for distance in (1, 2, 4, 8, num_qubits // 4, num_qubits // 2)
            if 0 < distance < num_qubits
        }
    )
    for distance in distances:
        step = max(1, distance // 2)
        for left in range(0, num_qubits - distance, step):
            for pauli in GLOBAL_MEASUREMENT_BASES:
                add({left: pauli, left + distance: pauli})

    for distance in (1, 2, 4):
        if 2 * distance < num_qubits:
            for start in range(0, num_qubits - 2 * distance, 2 * distance):
                support = tuple(start + offset * distance for offset in range(3))
                for pauli in GLOBAL_MEASUREMENT_BASES:
                    add({qubit: pauli for qubit in support})
        if 3 * distance >= num_qubits:
            continue
        for start in range(0, num_qubits - 3 * distance, 2 * distance):
            support = tuple(start + offset * distance for offset in range(4))
            for pauli in GLOBAL_MEASUREMENT_BASES:
                add({qubit: pauli for qubit in support})
    return mappings


def hardware_local_mappings(num_qubits: int) -> list[dict[int, str]]:
    """Return the established 5q-2 local panel used by the q60 route.

    It contains X/Y/Z singles and nearest-neighbour XX/ZZ pairs.  The panel is
    substantially cheaper for ideal MPS screening than the exploratory
    multiscale panel, while requiring the same three global bases on hardware.
    """

    if num_qubits < 4:
        raise ValueError("the local panel requires at least four qubits")
    return toy.pauli_feature_mappings(int(num_qubits), family="local")


def observable_mappings(
    num_qubits: int, panel_name: str
) -> list[dict[int, str]]:
    if panel_name == DEFAULT_OBSERVABLE_PANEL:
        return hardware_local_mappings(num_qubits)
    if panel_name == "multiscale_support4":
        return homogeneous_multiscale_mappings(num_qubits)
    raise ValueError(f"unsupported observable panel: {panel_name}")


def mapping_panel_summary(
    mappings: Sequence[Mapping[int, str]], *, num_qubits: int
) -> dict[str, Any]:
    basis_counts = {basis: 0 for basis in GLOBAL_MEASUREMENT_BASES}
    support_counts: dict[str, int] = {}
    for mapping in mappings:
        bases = set(mapping.values())
        if len(bases) != 1 or next(iter(bases)) not in GLOBAL_MEASUREMENT_BASES:
            raise ProjectedKernelError("observable panel is not globally measurable")
        basis_counts[next(iter(bases))] += 1
        key = str(len(mapping))
        support_counts[key] = support_counts.get(key, 0) + 1
    return {
        "num_qubits": int(num_qubits),
        "observable_count": len(mappings),
        "support_size_counts": support_counts,
        "measurement_basis_counts": basis_counts,
        "global_measurement_bases": list(GLOBAL_MEASUREMENT_BASES),
        "measurement_circuits_per_sample": len(GLOBAL_MEASUREMENT_BASES),
        "largest_observable_support": max(len(mapping) for mapping in mappings),
        "all_homogeneous_xyz": True,
    }


def feature_map_circuit(vector: np.ndarray, architecture_name: str):
    values = np.asarray(vector, dtype=np.float64)
    if architecture_name == "grid_mixer_d12":
        return architecture.architecture_circuit(values, architecture_name)
    if architecture_name == "grid_mixer_d8":
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Architecture input must be a finite one-dimensional vector"
            )
        circuit = QuantumCircuit(len(values))
        circuit.h(range(len(values)))
        for qubit, value in enumerate(values):
            circuit.ry(architecture.SINGLE_SCALE * float(value), qubit)
            circuit.rz(architecture.PHASE_SCALE * float(value), qubit)
        # Frozen middle-depth route: one horizontal and one vertical matching.
        # This preserves genuinely 2D spreading while cutting two entangling/
        # mixer layers from grid_mixer_d12.
        for layer_index, edges in enumerate(architecture._grid_matchings(len(values))[:2]):
            for left, right in edges:
                circuit.rzz(
                    architecture.PAIR_SCALE
                    * float(values[left])
                    * float(values[right]),
                    left,
                    right,
                )
            mixer = circuit.rx if layer_index == 0 else circuit.ry
            for qubit, value in enumerate(values):
                mixer(
                    math.pi / 4.0
                    + architecture.MIXER_DATA_SCALE * float(value),
                    qubit,
                )
        for qubit, value in enumerate(values):
            circuit.rx(architecture.FINAL_RX_SCALE * float(value), qubit)
        return circuit
    if architecture_name == "path_rzz_d4":
        pair_values = values[:-1] * values[1:]
        return shallow.shallow_circuit(values, pair_values)
    raise ValueError(f"unsupported architecture: {architecture_name}")


def _statevector_feature_row(
    vector: np.ndarray,
    mappings: Sequence[Mapping[int, str]],
    *,
    architecture_name: str,
) -> np.ndarray:
    circuit = feature_map_circuit(vector, architecture_name)
    state = Statevector.from_instruction(circuit)
    values: list[float] = []
    for mapping in mappings:
        label = ["I"] * circuit.num_qubits
        for qubit, pauli in mapping.items():
            label[circuit.num_qubits - 1 - int(qubit)] = str(pauli)
        value = state.expectation_value(SparsePauliOp("".join(label)))
        values.append(float(np.real_if_close(value).real))
    return np.asarray(values, dtype=np.float64)


def projected_feature_matrix(
    encoded_rows: np.ndarray,
    mappings: Sequence[Mapping[int, str]],
    *,
    bond_dimension: int,
    mps_threshold: float,
    architecture_name: str = DEFAULT_ARCHITECTURE,
    progress_label: str | None = None,
) -> tuple[np.ndarray, float, str]:
    """Evaluate the projected quantum features without a ``2**q`` object."""

    rows = np.asarray(encoded_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 4 or not np.all(np.isfinite(rows)):
        raise ValueError("encoded rows must be a finite matrix with width >= 4")
    started = time.perf_counter()
    features = np.empty((len(rows), len(mappings)), dtype=np.float64)
    if rows.shape[1] <= 10:
        for index, vector in enumerate(rows):
            features[index] = _statevector_feature_row(
                vector, mappings, architecture_name=architecture_name
            )
            if progress_label and (
                index == 0 or (index + 1) % 32 == 0 or index + 1 == len(rows)
            ):
                print(f"[{progress_label}] statevector {index + 1}/{len(rows)}", flush=True)
        method = "exact_statevector"
    else:
        simulator = architecture._mps_simulator(bond_dimension, mps_threshold)
        for index, vector in enumerate(rows):
            circuit = feature_map_circuit(vector, architecture_name)
            payload = architecture._with_expectation_saves(circuit, mappings)
            result = simulator.run(payload).result()
            if not result.success:
                raise ProjectedKernelError("Aer MPS simulation failed")
            data = result.data(0)
            features[index] = [
                float(np.real(data[f"e{mapping_index}"]))
                for mapping_index in range(len(mappings))
            ]
            if progress_label and (
                index == 0 or (index + 1) % 16 == 0 or index + 1 == len(rows)
            ):
                print(
                    f"[{progress_label}] MPS chi={bond_dimension} "
                    f"{index + 1}/{len(rows)}",
                    flush=True,
                )
        method = "aer_matrix_product_state"
    if not np.all(np.isfinite(features)) or np.max(np.abs(features)) > 1.0 + 1e-8:
        raise ProjectedKernelError("projected features violate expectation bounds")
    return features, float(time.perf_counter() - started), method


def _standardize_selected(
    train: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.sum(np.square(first), axis=1)[:, None]
    second_norm = np.sum(np.square(second), axis=1)[None, :]
    return np.maximum(first_norm + second_norm - 2.0 * first @ second.T, 0.0)


def median_gamma(train: np.ndarray) -> float:
    distances = _squared_distances(train, train)
    positive = distances[np.triu_indices(len(train), k=1)]
    positive = positive[positive > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    return 1.0 / max(median, 1e-12)


def projected_rbf_kernel(
    first: np.ndarray, second: np.ndarray, *, gamma: float
) -> np.ndarray:
    if gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and positive")
    return np.exp(-float(gamma) * _squared_distances(first, second))


def _rank_observables(
    features: np.ndarray, labels: np.ndarray, *, shot_intent: int
) -> tuple[np.ndarray, np.ndarray]:
    count = features.shape[1]
    return shallow.select_train_only_features(
        features, labels, count=count, shot_intent=shot_intent
    )


def fit_projected_kernel(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_counts: Sequence[int],
    c_values: Sequence[float],
    gamma_multipliers: Sequence[float],
    cv_splits: int,
    cv_seed: int,
    shot_intent: int,
) -> dict[str, Any]:
    """Tune every model choice inside training folds, then open the test once."""

    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    counts = sorted({min(int(value), train.shape[1]) for value in feature_counts})
    if not counts or counts[0] < 1:
        raise ValueError("feature counts must be positive")
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    folds = list(splitter.split(train, y_train))
    candidates: list[dict[str, Any]] = []
    for feature_count in counts:
        for c_value in c_values:
            for gamma_multiplier in gamma_multipliers:
                fold_scores: list[float] = []
                for fit_indices, validation_indices in folds:
                    ranking, _ = _rank_observables(
                        train[fit_indices], y_train[fit_indices], shot_intent=shot_intent
                    )
                    selected = ranking[:feature_count]
                    fit, validation, _, _ = _standardize_selected(
                        train[fit_indices][:, selected],
                        train[validation_indices][:, selected],
                    )
                    gamma = median_gamma(fit) * float(gamma_multiplier)
                    kernel_fit = projected_rbf_kernel(fit, fit, gamma=gamma)
                    kernel_validation = projected_rbf_kernel(
                        validation, fit, gamma=gamma
                    )
                    model = SVC(
                        C=float(c_value),
                        kernel="precomputed",
                        class_weight="balanced",
                        random_state=cv_seed,
                    )
                    model.fit(kernel_fit, y_train[fit_indices])
                    predictions = model.predict(kernel_validation)
                    fold_scores.append(
                        float(
                            balanced_accuracy_score(
                                y_train[validation_indices], predictions
                            )
                        )
                    )
                candidates.append(
                    {
                        "feature_count": int(feature_count),
                        "c": float(c_value),
                        "gamma_multiplier": float(gamma_multiplier),
                        "fold_balanced_accuracies": fold_scores,
                        "mean_balanced_accuracy": float(np.mean(fold_scores)),
                        "worst_balanced_accuracy": float(np.min(fold_scores)),
                    }
                )
    winner = max(
        candidates,
        key=lambda row: (
            float(row["mean_balanced_accuracy"]),
            float(row["worst_balanced_accuracy"]),
            -int(row["feature_count"]),
            -float(row["c"]),
            -float(row["gamma_multiplier"]),
        ),
    )
    ranking, ranking_scores = _rank_observables(
        train, y_train, shot_intent=shot_intent
    )
    selected = ranking[: int(winner["feature_count"])]
    train_z, test_z, mean, scale = _standardize_selected(
        train[:, selected], test[:, selected]
    )
    gamma = median_gamma(train_z) * float(winner["gamma_multiplier"])
    train_kernel = projected_rbf_kernel(train_z, train_z, gamma=gamma)
    test_kernel = projected_rbf_kernel(test_z, train_z, gamma=gamma)
    symmetry_error = float(np.max(np.abs(train_kernel - train_kernel.T)))
    diagonal_error = float(np.max(np.abs(np.diag(train_kernel) - 1.0)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(train_kernel)))
    if symmetry_error > 1e-10 or diagonal_error > 1e-10 or minimum_eigenvalue < -1e-8:
        raise ProjectedKernelError("projected RBF kernel failed PSD checks")
    model = SVC(
        C=float(winner["c"]),
        kernel="precomputed",
        class_weight="balanced",
        random_state=cv_seed,
    )
    model.fit(train_kernel, y_train)
    predictions = np.asarray(model.predict(test_kernel), dtype=np.int64)
    return {
        "model": "training_only_tuned_projected_rbf_svc",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "predictions": [int(value) for value in predictions],
        "test_correct": [bool(value) for value in predictions == y_test],
        "support_vectors": int(np.sum(model.n_support_)),
        "selected_observable_indices": [int(value) for value in selected],
        "selected_ranking_scores": [float(ranking_scores[value]) for value in selected],
        "selected_feature_count": int(len(selected)),
        "c": float(winner["c"]),
        "gamma_multiplier": float(winner["gamma_multiplier"]),
        "gamma": float(gamma),
        "training_feature_mean_sha256": _array_sha256(mean),
        "training_feature_scale_sha256": _array_sha256(scale),
        "inner_cv": {
            "splits": int(cv_splits),
            "seed": int(cv_seed),
            "winner": winner,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "all_selection_training_only": True,
        },
        "kernel_verification": {
            "symmetry_max_abs_error": symmetry_error,
            "diagonal_max_abs_error": diagonal_error,
            "minimum_eigenvalue": minimum_eigenvalue,
            "passed": True,
        },
    }


def small_width_validation(
    widths: Sequence[int],
    *,
    bond_dimension: int,
    architecture_name: str,
    observable_panel_name: str = DEFAULT_OBSERVABLE_PANEL,
) -> list[dict[str, Any]]:
    """Check exact statevector/MPS parity and kernel validity at q4--q10."""

    rows: list[dict[str, Any]] = []
    for width in widths:
        mappings = observable_mappings(int(width), observable_panel_name)
        vectors = np.asarray(
            [
                np.sin(np.linspace(-0.8, 0.7, int(width)) + offset)
                for offset in (0.0, 0.2, 0.5, 0.9)
            ],
            dtype=np.float64,
        )
        exact, exact_seconds, exact_method = projected_feature_matrix(
            vectors,
            mappings,
            bond_dimension=bond_dimension,
            mps_threshold=1e-14,
            architecture_name=architecture_name,
        )
        mps, mps_seconds = architecture.simulate_feature_rows(
            [feature_map_circuit(vector, architecture_name) for vector in vectors],
            mappings,
            bond_dimension=bond_dimension,
            threshold=1e-14,
        )
        error = np.abs(exact - mps)
        exact_z, _, _, _ = _standardize_selected(exact, exact)
        gamma = median_gamma(exact_z)
        kernel = projected_rbf_kernel(exact_z, exact_z, gamma=gamma)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(kernel)))
        circuit = feature_map_circuit(vectors[0], architecture_name)
        maximum_error = float(np.max(error))
        passed = bool(
            maximum_error <= 1e-9
            and np.max(np.abs(np.diag(kernel) - 1.0)) <= 1e-12
            and minimum_eigenvalue >= -1e-10
        )
        rows.append(
            {
                "qubits": int(width),
                "observables": len(mappings),
                "exact_method": exact_method,
                "statevector_seconds": exact_seconds,
                "mps_seconds": mps_seconds,
                "max_abs_mps_minus_statevector": maximum_error,
                "mean_abs_mps_minus_statevector": float(np.mean(error)),
                "kernel_minimum_eigenvalue": minimum_eigenvalue,
                "kernel_diagonal_max_abs_error": float(
                    np.max(np.abs(np.diag(kernel) - 1.0))
                ),
                "representative_circuit_metrics": shallow.q40_validate.circuit_metrics(
                    circuit
                ),
                "passed": passed,
            }
        )
    return rows


def select_mps_bond_dimension(
    encoded_rows: np.ndarray,
    mappings: Sequence[Mapping[int, str]],
    *,
    bond_dimensions: Sequence[int],
    threshold: float,
    tolerance: float,
    label: str,
    architecture_name: str,
) -> tuple[int, dict[str, Any]]:
    """Choose the smallest candidate agreeing with the largest-chi reference."""

    if not len(encoded_rows):
        raise ValueError("cannot probe an empty feature matrix")
    candidates = sorted({int(value) for value in bond_dimensions})
    if len(candidates) < 2 or candidates[0] < 2:
        raise ValueError("bond-dimension selection needs at least two candidates")
    probe_indices = np.unique(
        np.linspace(0, len(encoded_rows) - 1, num=min(4, len(encoded_rows)), dtype=int)
    )
    circuits = [
        feature_map_circuit(encoded_rows[index], architecture_name)
        for index in probe_indices
    ]
    feature_rows: dict[int, np.ndarray] = {}
    elapsed: dict[int, float] = {}
    for bond_dimension in candidates:
        probe, seconds = architecture.simulate_feature_rows(
            circuits,
            mappings,
            bond_dimension=int(bond_dimension),
            threshold=threshold,
            progress_label=f"{label}-probe",
        )
        feature_rows[int(bond_dimension)] = probe
        elapsed[int(bond_dimension)] = float(seconds)
    reference_dimension = candidates[-1]
    reference = feature_rows[reference_dimension]
    rows: list[dict[str, Any]] = []
    selected: int | None = None
    for bond_dimension in candidates:
        difference = np.abs(feature_rows[bond_dimension] - reference)
        maximum = float(np.max(difference))
        if (
            bond_dimension != reference_dimension
            and selected is None
            and maximum <= tolerance
        ):
            selected = int(bond_dimension)
        rows.append(
            {
                "bond_dimension": int(bond_dimension),
                "seconds": elapsed[bond_dimension],
                "max_abs_difference_from_reference": maximum,
                "mean_abs_difference_from_reference": float(np.mean(difference)),
            }
        )
    if selected is None:
        raise ProjectedKernelError(
            "no lower MPS bond dimension agreed with the largest reference"
        )
    report = {
        "probe_indices": [int(value) for value in probe_indices],
        "comparisons": rows,
        "reference_bond_dimension": int(reference_dimension),
        "selected_bond_dimension": int(selected),
        "tolerance": float(tolerance),
        "selected_max_abs_difference_from_reference": next(
            row["max_abs_difference_from_reference"]
            for row in rows
            if int(row["bond_dimension"]) == int(selected)
        ),
        "passed": True,
    }
    return int(selected), report


def _split_contexts(
    spec: flat_gate.DatasetSpec, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_number, seed in enumerate(args.split_seeds, start=1):
        print(
            f"[{spec.name}] classical frontier split {split_number}/5 seed={seed}",
            flush=True,
        )
        train_indices, test_indices = thirdorder.balanced_binary_split(
            spec.y,
            seed=int(seed),
            train_fraction=0.67,
            max_train_samples=int(spec.train_size),
            max_test_samples=int(spec.test_size),
        )
        classical = flat_gate._classical_frontier(
            spec,
            train_indices,
            test_indices,
            feature_dims=args.classical_dims,
            seed=int(seed),
        )
        rows.append(
            {
                "split_number": int(split_number),
                "split_seed": int(seed),
                "train_indices": train_indices,
                "test_indices": test_indices,
                "classical_frontier": classical,
            }
        )
    return rows


def _evaluate_width(
    spec: flat_gate.DatasetSpec,
    contexts: Sequence[dict[str, Any]],
    *,
    width: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    unique_indices = np.unique(
        np.concatenate(
            [
                np.concatenate([row["train_indices"], row["test_indices"]])
                for row in contexts
            ]
        )
    )
    encoded, encoding_stats = spec.build_interactions(
        spec.x[unique_indices],
        feature_dim=int(width),
        **spec.interaction_kwargs,
    )
    encoded = np.asarray(encoded, dtype=np.float64)
    mappings = observable_mappings(int(width), str(args.observable_panel))
    selected_bond_dimension = int(args.bond_dimension)
    convergence = None
    if int(width) > 10:
        selected_bond_dimension, convergence = select_mps_bond_dimension(
            encoded,
            mappings,
            bond_dimensions=args.bond_dimension_candidates,
            threshold=float(args.mps_threshold),
            tolerance=float(args.convergence_tolerance),
            label=f"{spec.name}-q{width}",
            architecture_name=str(args.architecture),
        )
        print(
            f"[{spec.name}-q{width}] selected MPS chi={selected_bond_dimension} "
            f"against chi={convergence['reference_bond_dimension']}",
            flush=True,
        )
    features, feature_seconds, simulation_method = projected_feature_matrix(
        encoded,
        mappings,
        bond_dimension=selected_bond_dimension,
        mps_threshold=float(args.mps_threshold),
        architecture_name=str(args.architecture),
        progress_label=f"{spec.name}-q{width}",
    )
    position = {int(row_id): index for index, row_id in enumerate(unique_indices)}
    split_rows: list[dict[str, Any]] = []
    for context in contexts:
        train_positions = np.asarray(
            [position[int(value)] for value in context["train_indices"]], dtype=np.int64
        )
        test_positions = np.asarray(
            [position[int(value)] for value in context["test_indices"]], dtype=np.int64
        )
        quantum = fit_projected_kernel(
            features[train_positions],
            features[test_positions],
            spec.y[context["train_indices"]],
            spec.y[context["test_indices"]],
            feature_counts=args.feature_counts,
            c_values=args.c_values,
            gamma_multipliers=args.gamma_multipliers,
            cv_splits=int(args.cv_splits),
            cv_seed=int(args.cv_seed + context["split_seed"]),
            shot_intent=int(args.shot_intent),
        )
        classical = context["classical_frontier"]
        delta = float(
            quantum["balanced_accuracy"] - classical["best_balanced_accuracy"]
        )
        split_rows.append(
            {
                "split_number": int(context["split_number"]),
                "split_seed": int(context["split_seed"]),
                "train_size": int(len(context["train_indices"])),
                "test_size": int(len(context["test_indices"])),
                "train_indices_sha256": flat_gate._sha256_array(
                    context["train_indices"]
                ),
                "test_indices_sha256": flat_gate._sha256_array(
                    context["test_indices"]
                ),
                "classical_frontier": classical,
                "projected_quantum_kernel": quantum,
                "balanced_accuracy_delta": delta,
                "strict_quantum_win": bool(delta > 0.0),
            }
        )
        print(
            f"[{spec.name}] q={width} seed={context['split_seed']}: "
            f"classical={classical['best_balanced_accuracy']:.4f} "
            f"projected={quantum['balanced_accuracy']:.4f} ({delta:+.4f})",
            flush=True,
        )
    gate = flat_gate.evaluate_gate_summary(
        [{"balanced_accuracy_delta": row["balanced_accuracy_delta"]} for row in split_rows],
        min_wins=flat_gate.GATE_MIN_WINS,
        bootstrap_seed=int(args.bootstrap_seed + width),
        bootstrap_replicates=int(args.bootstrap_replicates),
    )
    representative = feature_map_circuit(encoded[0], str(args.architecture))
    maximum_full_split_samples = max(
        len(row["train_indices"]) + len(row["test_indices"]) for row in contexts
    )
    full_circuits = maximum_full_split_samples * len(GLOBAL_MEASUREMENT_BASES)
    pilot_circuits = (32 + 32) * len(GLOBAL_MEASUREMENT_BASES)
    return {
        "qubits": int(width),
        "architecture": str(args.architecture),
        "observable_panel_name": str(args.observable_panel),
        "simulation_method": simulation_method,
        "selected_mps_bond_dimension": int(selected_bond_dimension),
        "unique_dataset_rows_simulated": int(len(unique_indices)),
        "unique_row_ids_sha256": _array_sha256(unique_indices),
        "encoded_matrix_sha256": _array_sha256(encoded),
        "feature_matrix_sha256": _array_sha256(features),
        "encoding_stats": encoding_stats,
        "feature_seconds": feature_seconds,
        "feature_bounds": {
            "minimum": float(np.min(features)),
            "maximum": float(np.max(features)),
        },
        "observable_panel": mapping_panel_summary(mappings, num_qubits=int(width)),
        "representative_logical_circuit_metrics": shallow.q40_validate.circuit_metrics(
            representative
        ),
        "mps_convergence": convergence,
        "splits": split_rows,
        "gate": gate,
        "future_provider_resource_projection": {
            "measurement_circuits_per_sample": len(GLOBAL_MEASUREMENT_BASES),
            "full_largest_split_circuits": int(full_circuits),
            "full_largest_split_fire_opal_batches_at_300": int(
                math.ceil(full_circuits / FIRE_OPAL_MAX_BATCH)
            ),
            "balanced_32_train_32_test_pilot_circuits": int(pilot_circuits),
            "pilot_within_one_300_circuit_batch": bool(
                pilot_circuits <= FIRE_OPAL_MAX_BATCH
            ),
            "provider_calls_made": 0,
        },
    }


def _evaluate_dataset(
    spec: flat_gate.DatasetSpec, args: argparse.Namespace
) -> dict[str, Any]:
    started = time.perf_counter()
    contexts = _split_contexts(spec, args)
    widths = [
        _evaluate_width(spec, contexts, width=int(width), args=args)
        for width in args.widths
    ]
    eligible = [int(row["qubits"]) for row in widths if row["gate"]["passed"]]
    return {
        "dataset": spec.name,
        "source": spec.source,
        "task": spec.task,
        "dataset_validation": {
            "rows": int(spec.x.shape[0]),
            "features": int(spec.x.shape[1]),
            "nnz": int(spec.x.nnz),
            "sparse_matrix_sha256": flat_gate._sha256_sparse(spec.x),
            "labels_sha256": flat_gate._sha256_array(spec.y),
        },
        "widths": widths,
        "gate": {
            "passed": bool(eligible),
            "eligible_qubit_widths": eligible,
            "selection_rule": (
                "q40 and q60 are judged independently on the same five frozen "
                "splits; no post-hoc best-width score is used"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_parse_datasets, default=flat_gate.SUPPORTED_DATASETS)
    parser.add_argument("--split-seeds", type=_parse_ints, default=flat_gate.DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--widths", type=_parse_ints, default=DEFAULT_WIDTHS)
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
    )
    parser.add_argument(
        "--observable-panel",
        choices=SUPPORTED_OBSERVABLE_PANELS,
        default=DEFAULT_OBSERVABLE_PANEL,
    )
    parser.add_argument(
        "--validation-widths", type=_parse_ints, default=DEFAULT_VALIDATION_WIDTHS
    )
    parser.add_argument(
        "--classical-dims", type=_parse_ints, default=flat_gate.DEFAULT_CLASSICAL_DIMS
    )
    parser.add_argument("--feature-counts", type=_parse_ints, default=DEFAULT_FEATURE_COUNTS)
    parser.add_argument("--c-values", type=_parse_floats, default=DEFAULT_C_VALUES)
    parser.add_argument(
        "--gamma-multipliers", type=_parse_floats, default=DEFAULT_GAMMA_MULTIPLIERS
    )
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--cv-seed", type=int, default=82_019)
    parser.add_argument("--shot-intent", type=int, default=DEFAULT_SHOT_INTENT)
    parser.add_argument("--bond-dimension", type=int, default=DEFAULT_BOND_DIMENSION)
    parser.add_argument(
        "--bond-dimension-candidates",
        type=_parse_ints,
        default=DEFAULT_BOND_DIMENSION_CANDIDATES,
    )
    parser.add_argument("--mps-threshold", type=float, default=DEFAULT_MPS_THRESHOLD)
    parser.add_argument(
        "--convergence-tolerance", type=float, default=DEFAULT_CONVERGENCE_TOLERANCE
    )
    parser.add_argument("--bootstrap-seed", type=int, default=121_771)
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
        raise ProjectedKernelError("the frozen gate requires exactly five split seeds")
    if any(width < 4 for width in (*args.widths, *args.validation_widths)):
        raise ProjectedKernelError("all widths must be at least four")
    if any(value <= 0 for value in (*args.c_values, *args.gamma_multipliers)):
        raise ProjectedKernelError("kernel hyperparameters must be positive")
    if args.cv_splits < 2 or args.shot_intent < 1:
        raise ProjectedKernelError("CV splits and shot intent are outside range")
    if args.output.exists() and not args.force:
        raise ProjectedKernelError(f"output already exists: {args.output}")

    started = time.perf_counter()
    print("Running exact q4-q10 validation", flush=True)
    validation = small_width_validation(
        args.validation_widths,
        bond_dimension=int(args.bond_dimension),
        architecture_name=str(args.architecture),
        observable_panel_name=str(args.observable_panel),
    )
    if not all(row["passed"] for row in validation):
        raise ProjectedKernelError("small-width exact validation failed")

    loaders = {
        flat_gate.DATASET_GSE: flat_gate._load_gse,
        flat_gate.DATASET_PBMC: flat_gate._load_pbmc,
    }
    datasets: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"Loading and evaluating {dataset}", flush=True)
        datasets.append(_evaluate_dataset(loaders[dataset](args), args))

    eligible = [
        {
            "dataset": row["dataset"],
            "qubit_widths": row["gate"]["eligible_qubit_widths"],
        }
        for row in datasets
        if row["gate"]["passed"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "local_gate_complete",
        "protocol": {
            "input_representation": (
                "stream-hashed sparse real-data pair/triplet interactions; no dense "
                "ambient feature map and no 2**q state/probability vector"
            ),
            "quantum_feature_map": str(args.architecture),
            "observable_panel": str(args.observable_panel),
            "projected_observables": OBSERVABLE_PANEL_DESCRIPTIONS[
                str(args.observable_panel)
            ],
            "kernel": (
                "K(x,y)=exp(-gamma ||standardize(P_S(x))-"
                "standardize(P_S(y))||_2^2)"
            ),
            "selection": (
                "observable count, C, and gamma multiplier selected inside "
                "training-only stratified CV"
            ),
            "widths": [int(value) for value in args.widths],
            "split_seeds": [int(value) for value in args.split_seeds],
            "classical_feature_dims": [int(value) for value in args.classical_dims],
            "future_shot_intent": int(args.shot_intent),
            "global_measurement_bases": list(GLOBAL_MEASUREMENT_BASES),
        },
        "equation_registry": [
            {
                "id": "eq:sparse-feature-map",
                "code": (
                    "qiskit_qos_realdata_projected_kernel_gate."
                    "feature_map_circuit"
                ),
                "status": "small-width statevector/MPS parity checked",
            },
            {
                "id": "eq:projected-pauli-vector",
                "code": (
                    "qiskit_qos_realdata_projected_kernel_gate."
                    "projected_feature_matrix"
                ),
                "status": "expectation bounds and q4-q10 exact parity checked",
            },
            {
                "id": "eq:projected-rbf-kernel",
                "code": (
                    "qiskit_qos_realdata_projected_kernel_gate."
                    "projected_rbf_kernel"
                ),
                "status": "symmetry, unit diagonal, and PSD checked per split",
            },
        ],
        "small_width_validation": validation,
        "datasets": datasets,
        "hardware_gate": {
            "eligible_candidates": eligible,
            "passed": bool(eligible),
            "provider_calls": [],
            "execution_attempted": False,
            "automatic_submission": False,
            "direct_ibm_adapter": (
                "requires backend ISA transpilation and a separately frozen plan"
            ),
            "fire_opal_adapter": (
                "requires backend-free numeric QASM, manifest hashes, validation, "
                "quota availability, and separate execution authorization"
            ),
            "decision": (
                "eligible for a separate provider pilot plan"
                if eligible
                else "blocked locally; do not transmit circuits or submit hardware"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "This is an ideal local projected-quantum-kernel screen. Passing would "
            "select a hardware-feasibility candidate, not prove computational, "
            "exponential, or practical quantum advantage. The bounded-depth circuit "
            "and MPS diagnostics remain part of the classical-simulability caveat."
        ),
    }
    _atomic_write_json(args.output, payload)
    print(f"Saved gate artifact: {args.output}", flush=True)
    print(f"Eligible provider candidates: {eligible or 'none'}", flush=True)
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
