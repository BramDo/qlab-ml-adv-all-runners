#!/usr/bin/env python3
"""Create concrete data, circuit, statevector, and matrix artifacts for the q4 demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator, Statevector

import qiskit_qos_pbmc68k_q4_educational as tutorial
import qiskit_qos_pbmc68k_utils as pbmc


FEATURE_NAMES = ("Z0", "Z1", "Z2", "Z3", "Z0Z1", "Z1Z2", "Z2Z3", "Z3Z0")


def _basis_label(index: int) -> str:
    """Use Qiskit's displayed little-endian basis order |q3 q2 q1 q0>."""

    return f"|{index:04b}>"


def _exact_z_features(probabilities: np.ndarray) -> np.ndarray:
    single = np.zeros(tutorial.QUBITS, dtype=np.float64)
    pairs = np.zeros(len(tutorial.NEIGHBOUR_EDGES), dtype=np.float64)
    for basis_index, probability in enumerate(probabilities):
        z = np.asarray(
            [1.0 if ((basis_index >> qubit) & 1) == 0 else -1.0 for qubit in range(4)]
        )
        single += float(probability) * z
        pairs += float(probability) * np.asarray(
            [z[a] * z[b] for a, b in tutorial.NEIGHBOUR_EDGES]
        )
    return np.concatenate((single, pairs))


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    labels = [_basis_label(index) for index in range(matrix.shape[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["output\\input", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12f}" for value in row]])


def _save_circuit(circuit, path: Path) -> None:
    figure = circuit.draw(output="mpl", fold=-1, idle_wires=True)
    figure.suptitle("PBMC68k q4 feature map for one real cell", fontsize=14, y=0.99)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_unitary_heatmap(unitary: np.ndarray, path: Path) -> None:
    magnitude = np.abs(unitary)
    figure, axis = plt.subplots(figsize=(9.3, 8.0))
    image = axis.imshow(magnitude, cmap="magma", vmin=0.0, vmax=1.0)
    labels = [_basis_label(index) for index in range(16)]
    axis.set_xticks(range(16), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(16), labels, fontsize=7)
    axis.set_xlabel("input basis state")
    axis.set_ylabel("output basis state")
    axis.set_title(r"Magnitude of the cell-specific $16\times16$ unitary, $|U_{ij}|$")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("magnitude")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_artifacts(
    *,
    cache_dir: Path,
    output_dir: Path,
    shots: int = 512,
    seed: int = tutorial.SEED,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    pair_x, pair_y, pair_meta = pbmc.select_binary_pair(
        matrix,
        labels,
        positive_label=tutorial.POSITIVE_LABEL,
        negative_label=tutorial.NEGATIVE_LABEL,
    )
    train_indices, test_indices = tutorial.balanced_split_indices(
        pair_y,
        train_size=16,
        test_size=16,
        seed=seed,
    )
    selected, variances, detection = tutorial.select_label_free_genes(
        pair_x[train_indices]
    )
    gene_names = tutorial._gene_names(cache_dir, pair_x.shape[1])
    selected_names = [gene_names[index] for index in selected]

    train_counts = pair_x[train_indices]
    test_counts = pair_x[test_indices]
    train_logged = tutorial.library_log1p(train_counts)[:, selected].toarray()
    test_logged = tutorial.library_log1p(test_counts)[:, selected].toarray()
    train_raw = train_counts[:, selected].toarray().astype(np.int64)
    test_raw = test_counts[:, selected].toarray().astype(np.int64)
    train_angles, test_angles, train_z, test_z, scaler = tutorial.prepare_gene_angles(
        train_counts,
        test_counts,
        selected,
    )

    all_angles = np.vstack((train_angles, test_angles))
    shot_features, _ = tutorial.simulate_quantum_features(
        all_angles,
        shots=shots,
        seed=seed,
    )

    first_angles = train_angles[0]
    circuit = tutorial.build_feature_circuit(first_angles, measure=False)
    measured_circuit = tutorial.build_feature_circuit(first_angles, measure=True)
    unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    probabilities = np.abs(state) ** 2
    exact_features = _exact_z_features(probabilities)

    preview_path = output_dir / "pbmc68k_q4_data_preview.csv"
    with preview_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = ["split", "split_row", "pair_row", "label", "library_umi"]
        for gene in selected_names:
            columns.extend((f"{gene}_umi", f"{gene}_log1p", f"{gene}_z", f"{gene}_theta"))
        writer.writerow(columns)
        rows = [
            ("train", train_indices, train_counts, train_raw, train_logged, train_z, train_angles),
            ("test", test_indices, test_counts, test_raw, test_logged, test_z, test_angles),
        ]
        for split, indices, counts, raw, logged, z_values, angles in rows:
            for row_index in range(4):
                label = (
                    tutorial.POSITIVE_LABEL
                    if pair_y[indices[row_index]] == 1
                    else tutorial.NEGATIVE_LABEL
                )
                values: list[Any] = [
                    split,
                    row_index,
                    int(indices[row_index]),
                    label,
                    int(counts[row_index].sum()),
                ]
                for gene_index in range(4):
                    values.extend(
                        (
                            int(raw[row_index, gene_index]),
                            f"{logged[row_index, gene_index]:.9f}",
                            f"{z_values[row_index, gene_index]:.9f}",
                            f"{angles[row_index, gene_index]:.9f}",
                        )
                    )
                writer.writerow(values)

    state_path = output_dir / "pbmc68k_q4_first_cell_statevector.csv"
    with state_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("basis_q3q2q1q0", "amplitude_real", "amplitude_imag", "probability"))
        for index, amplitude in enumerate(state):
            writer.writerow(
                (
                    _basis_label(index),
                    f"{amplitude.real:.12f}",
                    f"{amplitude.imag:.12f}",
                    f"{probabilities[index]:.12f}",
                )
            )

    real_path = output_dir / "pbmc68k_q4_first_cell_unitary_real.csv"
    imag_path = output_dir / "pbmc68k_q4_first_cell_unitary_imag.csv"
    _write_matrix_csv(real_path, unitary.real)
    _write_matrix_csv(imag_path, unitary.imag)

    circuit_path = output_dir / "pbmc68k_q4_first_cell_circuit.png"
    matrix_path = output_dir / "pbmc68k_q4_first_cell_unitary_magnitude.png"
    text_path = output_dir / "pbmc68k_q4_first_cell_circuit.txt"
    _save_circuit(measured_circuit, circuit_path)
    _save_unitary_heatmap(unitary, matrix_path)
    text_path.write_text(str(measured_circuit.draw(output="text")) + "\n", encoding="utf-8")

    first_index = int(train_indices[0])
    first_label = (
        tutorial.POSITIVE_LABEL if pair_y[first_index] == 1 else tutorial.NEGATIVE_LABEL
    )
    payload: dict[str, Any] = {
        "dataset": {
            "name": source_meta["dataset_name"],
            "cells": source_meta["rows_annotated"],
            "genes": source_meta["genes"],
            "binary_pair_cells": pair_meta["rows"],
            "matrix_url": source_meta["matrix_url"],
            "annotation_url": source_meta["annotation_url"],
        },
        "selected_genes": [
            {
                "name": name,
                "index": int(index),
                "training_variance": float(variance),
                "training_detection_fraction": float(fraction),
                "training_mean": float(scaler["mean"][position]),
                "training_std": float(scaler["std"][position]),
            }
            for position, (name, index, variance, fraction) in enumerate(
                zip(selected_names, selected, variances, detection)
            )
        ],
        "first_real_training_cell": {
            "pair_row": first_index,
            "label": first_label,
            "library_umi": int(train_counts[0].sum()),
            "raw_umi": dict(zip(selected_names, train_raw[0].tolist())),
            "normalized_log1p": dict(zip(selected_names, train_logged[0].tolist())),
            "z_score": dict(zip(selected_names, train_z[0].tolist())),
            "rotation_radians": dict(zip(selected_names, first_angles.tolist())),
            "shot_features_512": dict(zip(FEATURE_NAMES, shot_features[0].tolist())),
            "exact_features": dict(zip(FEATURE_NAMES, exact_features.tolist())),
            "state_norm": float(np.sum(probabilities)),
            "largest_output_probabilities": [
                {"basis": _basis_label(int(index)), "probability": float(probabilities[index])}
                for index in np.argsort(probabilities)[::-1][:6]
            ],
        },
        "matrix": {
            "shape": list(unitary.shape),
            "basis_order": "|q3 q2 q1 q0>",
            "max_imaginary_absolute": float(np.max(np.abs(unitary.imag))),
            "unitarity_error_frobenius": float(
                np.linalg.norm(unitary.conj().T @ unitary - np.eye(16))
            ),
        },
        "artifacts": {
            "data_preview_csv": str(preview_path),
            "statevector_csv": str(state_path),
            "unitary_real_csv": str(real_path),
            "unitary_imag_csv": str(imag_path),
            "circuit_png": str(circuit_path),
            "unitary_magnitude_png": str(matrix_path),
            "circuit_text": str(text_path),
        },
        "claim_boundary": tutorial.CLAIM_BOUNDARY,
    }
    json_path = output_dir / "pbmc68k_q4_explanation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pbmc68k_q4_explainer"),
    )
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=tutorial.SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = create_artifacts(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        shots=args.shots,
        seed=args.seed,
    )
    cell = payload["first_real_training_cell"]
    print("Data:", payload["dataset"])
    print("Cell:", cell["pair_row"], cell["label"], cell["raw_umi"])
    print("Angles:", cell["rotation_radians"])
    print("Matrix:", payload["matrix"])
    print("Artifacts:", payload["artifacts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
