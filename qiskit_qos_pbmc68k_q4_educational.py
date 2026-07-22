#!/usr/bin/env python3
"""Small, leakage-aware PBMC68k QML example on a four-qubit simulator.

This is an educational feature-map demo, not a reproduction of the complete
Quantum Oracle Sketching protocol and not evidence of quantum advantage.

Pipeline
--------
1. Load the public PBMC68k count matrix and select two CD4 T-cell labels.
2. Freeze a balanced 16/16 train/test split (configurable).
3. Select four variable genes from the training counts without using labels.
4. Fit all scaling on training cells only and encode one value per qubit.
5. Simulate one shallow four-qubit circuit per cell with Qiskit Aer.
6. Read four Z and four neighbouring ZZ expectation values from the shots.
7. Train a small classical logistic-regression head on those eight features.

The example deliberately keeps every step visible so it can be explained in a
classroom or StackExchange answer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import qiskit_qos_pbmc68k_utils as pbmc


QUBITS = 4
SEED = 11
POSITIVE_LABEL = "CD4+/CD25 T Reg"
NEGATIVE_LABEL = "CD4+/CD45RO+ Memory"
NEIGHBOUR_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
CLAIM_BOUNDARY = (
    "Educational four-qubit simulator demo only: it does not reproduce full "
    "QOS and it makes no quantum-advantage claim."
)


def library_log1p(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Normalize each cell to 10,000 counts and apply log(1+x)."""

    value = matrix.tocsr().astype(np.float64, copy=True)
    totals = np.asarray(value.sum(axis=1), dtype=np.float64).ravel()
    factors = np.divide(
        10_000.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0.0,
    )
    value = sparse.diags(factors) @ value
    value.data = np.log1p(value.data)
    return value.tocsr()


def balanced_split_indices(
    labels: np.ndarray,
    *,
    train_size: int,
    test_size: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, balanced, disjoint indices for labels -1/+1."""

    y = np.asarray(labels, dtype=np.int64)
    if set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError("Expected binary labels encoded as -1 and +1")
    if train_size <= 0 or test_size <= 0 or train_size % 2 or test_size % 2:
        raise ValueError("train_size and test_size must be positive even numbers")

    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (-1, 1):
        candidates = rng.permutation(np.flatnonzero(y == label))
        needed = train_size // 2 + test_size // 2
        if len(candidates) < needed:
            raise ValueError(f"Not enough examples for class {label}")
        train_parts.append(candidates[: train_size // 2])
        test_parts.append(candidates[train_size // 2 : needed])

    train = rng.permutation(np.concatenate(train_parts)).astype(np.int64)
    test = rng.permutation(np.concatenate(test_parts)).astype(np.int64)
    if np.intersect1d(train, test).size:
        raise RuntimeError("Train/test split unexpectedly overlaps")
    return train, test


def select_label_free_genes(
    training_counts: sparse.spmatrix,
    *,
    n_genes: int = QUBITS,
    detection_min: float = 0.05,
    detection_max: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose variable genes from training counts without inspecting labels."""

    raw = training_counts.tocsr()
    if raw.shape[0] < 2 or raw.shape[1] < n_genes:
        raise ValueError("Training matrix is too small for gene selection")

    detection = np.asarray(raw.getnnz(axis=0), dtype=np.float64).ravel() / raw.shape[0]
    logged = library_log1p(raw)
    means = np.asarray(logged.mean(axis=0), dtype=np.float64).ravel()
    second_moment = np.asarray(
        logged.multiply(logged).mean(axis=0), dtype=np.float64
    ).ravel()
    variances = np.maximum(0.0, second_moment - means**2)

    eligible = np.flatnonzero(
        (detection >= float(detection_min)) & (detection <= float(detection_max))
    )
    if len(eligible) < n_genes:
        eligible = np.flatnonzero((detection > 0.0) & (variances > 0.0))
    if len(eligible) < n_genes:
        raise ValueError("Fewer than four non-constant genes are available")

    order = np.lexsort((eligible, -variances[eligible]))
    selected = np.asarray(eligible[order[:n_genes]], dtype=np.int64)
    return selected, variances[selected], detection[selected]


def scale_gene_inputs(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit a z-score transform on training rows and map values to [-pi, pi]."""

    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != QUBITS:
        raise ValueError("Expected train/test matrices with four columns")

    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    safe_std = np.where(std > 0.0, std, 1.0)
    train_z = (train - mean) / safe_std
    test_z = (test - mean) / safe_std
    train_angles = np.pi * (np.clip(train_z, -3.0, 3.0) / 3.0)
    test_angles = np.pi * (np.clip(test_z, -3.0, 3.0) / 3.0)
    return train_angles, test_angles, {"mean": mean, "std": safe_std}


def prepare_gene_angles(
    train_counts: sparse.spmatrix,
    test_counts: sparse.spmatrix,
    gene_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Create classical four-gene values and four quantum rotation angles."""

    genes = np.asarray(gene_indices, dtype=np.int64)
    if genes.shape != (QUBITS,):
        raise ValueError("Exactly four gene indices are required")
    train_values = library_log1p(train_counts)[:, genes].toarray()
    test_values = library_log1p(test_counts)[:, genes].toarray()
    train_angles, test_angles, scaler = scale_gene_inputs(train_values, test_values)
    train_classical = (train_values - scaler["mean"]) / scaler["std"]
    test_classical = (test_values - scaler["mean"]) / scaler["std"]
    return train_angles, test_angles, train_classical, test_classical, scaler


def build_feature_circuit(angles: Sequence[float], *, measure: bool = True) -> QuantumCircuit:
    """Encode four values, add a shallow entangling ring, and optionally measure."""

    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (QUBITS,) or not np.all(np.isfinite(values)):
        raise ValueError("angles must contain four finite values")

    circuit = QuantumCircuit(QUBITS, QUBITS if measure else 0)
    for qubit, angle in enumerate(values):
        circuit.ry(float(angle), qubit)
    for control, target in NEIGHBOUR_EDGES:
        circuit.cx(control, target)
    if measure:
        circuit.measure(range(QUBITS), range(QUBITS))
    return circuit


def _z_values(bitstring: str) -> np.ndarray:
    cleaned = bitstring.replace(" ", "").zfill(QUBITS)
    if len(cleaned) != QUBITS or set(cleaned) - {"0", "1"}:
        raise ValueError(f"Unexpected four-qubit bitstring: {bitstring!r}")
    # Qiskit displays classical bits as c3 c2 c1 c0.
    return np.asarray(
        [1.0 if cleaned[QUBITS - 1 - qubit] == "0" else -1.0 for qubit in range(QUBITS)]
    )


def features_from_counts(counts: Mapping[str, int]) -> np.ndarray:
    """Return [<Z0>...<Z3>, <Z0Z1>...<Z3Z0>] from one counts object."""

    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("Counts must contain at least one shot")
    single = np.zeros(QUBITS, dtype=np.float64)
    pairs = np.zeros(len(NEIGHBOUR_EDGES), dtype=np.float64)
    for bitstring, count in counts.items():
        weight = int(count) / total
        z = _z_values(str(bitstring))
        single += weight * z
        pairs += weight * np.asarray([z[a] * z[b] for a, b in NEIGHBOUR_EDGES])
    features = np.concatenate((single, pairs))
    if not np.all(np.isfinite(features)) or np.max(np.abs(features)) > 1.0 + 1e-12:
        raise RuntimeError("Simulator features are outside the physical expectation range")
    return features


def simulate_quantum_features(
    angle_rows: np.ndarray,
    *,
    shots: int,
    seed: int = SEED,
) -> tuple[np.ndarray, QuantumCircuit]:
    """Run all four-qubit feature circuits in one local Aer job."""

    rows = np.asarray(angle_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != QUBITS or shots <= 0:
        raise ValueError("Expected an N x 4 angle matrix and a positive shot count")

    circuits = [build_feature_circuit(row) for row in rows]
    backend = AerSimulator()
    compiled = transpile(
        circuits,
        backend,
        optimization_level=1,
        seed_transpiler=int(seed),
    )
    result = backend.run(
        compiled,
        shots=int(shots),
        seed_simulator=int(seed),
    ).result()
    count_rows: list[Mapping[str, int]]
    if len(circuits) == 1:
        count_rows = [result.get_counts()]
    else:
        count_rows = list(result.get_counts())
    features = np.vstack([features_from_counts(counts) for counts in count_rows])
    return features, circuits[0]


def classifier_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[float, np.ndarray]:
    """Fit a deliberately small classical head and return balanced accuracy."""

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", random_state=int(seed), C=1.0),
    )
    model.fit(np.asarray(train_x), np.asarray(train_y))
    predictions = np.asarray(model.predict(np.asarray(test_x)), dtype=np.int64)
    return float(balanced_accuracy_score(test_y, predictions)), predictions


def _gene_names(cache_dir: Path, expected: int) -> list[str]:
    archive = cache_dir / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
    lines = pbmc._read_member_text(archive, "genes.tsv")
    names = pbmc._parse_features(lines)["gene_name"].astype(str).tolist()
    if len(names) != expected:
        raise RuntimeError("Gene-name count does not match the PBMC matrix")
    return names


def run_tutorial(
    *,
    cache_dir: Path,
    train_size: int = 16,
    test_size: int = 16,
    shots: int = 512,
    seed: int = SEED,
) -> tuple[dict[str, Any], QuantumCircuit]:
    """Execute the complete educational PBMC68k simulator pipeline."""

    started = time.perf_counter()
    matrix, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    pair_x, pair_y, pair_meta = pbmc.select_binary_pair(
        matrix,
        labels,
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
    )
    train_indices, test_indices = balanced_split_indices(
        pair_y,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    selected, variances, detection = select_label_free_genes(pair_x[train_indices])
    names = _gene_names(cache_dir, pair_x.shape[1])
    (
        train_angles,
        test_angles,
        train_classical,
        test_classical,
        scaler,
    ) = prepare_gene_angles(pair_x[train_indices], pair_x[test_indices], selected)

    all_angles = np.vstack((train_angles, test_angles))
    all_quantum, example_circuit = simulate_quantum_features(
        all_angles,
        shots=shots,
        seed=seed,
    )
    quantum_train = all_quantum[:train_size]
    quantum_test = all_quantum[train_size:]
    y_train = pair_y[train_indices]
    y_test = pair_y[test_indices]
    quantum_accuracy, quantum_predictions = classifier_accuracy(
        quantum_train,
        y_train,
        quantum_test,
        y_test,
        seed=seed,
    )
    classical_accuracy, classical_predictions = classifier_accuracy(
        train_classical,
        y_train,
        test_classical,
        y_test,
        seed=seed,
    )

    payload: dict[str, Any] = {
        "kind": "pbmc68k_q4_educational_simulator",
        "completed": True,
        "config": {
            "qubits": QUBITS,
            "seed": int(seed),
            "shots_per_cell": int(shots),
            "train_cells": int(train_size),
            "test_cells": int(test_size),
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "quantum_features": [
                "Z0",
                "Z1",
                "Z2",
                "Z3",
                "Z0Z1",
                "Z1Z2",
                "Z2Z3",
                "Z3Z0",
            ],
        },
        "dataset": {**source_meta, **pair_meta},
        "selected_genes": [
            {
                "index": int(index),
                "name": str(names[index]),
                "training_variance": float(variance),
                "training_detection_fraction": float(fraction),
            }
            for index, variance, fraction in zip(selected, variances, detection)
        ],
        "leakage_checks": {
            "gene_selection_used_labels": False,
            "gene_selection_scope": "training cells only",
            "scaler_fit_scope": "training cells only",
            "train_test_disjoint": bool(
                np.intersect1d(train_indices, test_indices).size == 0
            ),
        },
        "results": {
            "quantum_feature_balanced_accuracy": quantum_accuracy,
            "classical_same_four_genes_balanced_accuracy": classical_accuracy,
            "quantum_correct": int(np.sum(quantum_predictions == y_test)),
            "classical_correct": int(np.sum(classical_predictions == y_test)),
            "test_cells": int(test_size),
        },
        "scaler": {
            "training_mean": scaler["mean"].tolist(),
            "training_std": scaler["std"].tolist(),
        },
        "circuit": {
            "depth": int(example_circuit.depth()),
            "two_qubit_gates": int(
                sum(
                    value
                    for name, value in example_circuit.count_ops().items()
                    if name in {"cx", "cz", "ecr"}
                )
            ),
            "circuits_simulated": int(train_size + test_size),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return payload, example_circuit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, circuit = run_tutorial(
        cache_dir=args.cache_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        shots=args.shots,
        seed=args.seed,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("PBMC68k four-qubit educational simulator")
    print("Selected genes:", ", ".join(row["name"] for row in payload["selected_genes"]))
    print(circuit.draw(output="text"))
    results = payload["results"]
    print(
        "Quantum features:",
        f"{results['quantum_feature_balanced_accuracy']:.3f}",
        f"({results['quantum_correct']}/{results['test_cells']})",
    )
    print(
        "Classical same four genes:",
        f"{results['classical_same_four_genes_balanced_accuracy']:.3f}",
        f"({results['classical_correct']}/{results['test_cells']})",
    )
    print(f"Elapsed: {payload['elapsed_seconds']:.2f} seconds")
    print(CLAIM_BOUNDARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
