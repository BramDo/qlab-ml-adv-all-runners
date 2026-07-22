from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator, Statevector

import qiskit_qos_pbmc68k_q4_educational as tutorial
from qiskit_qos_pbmc68k_q4_explain import _exact_z_features


ROOT = Path(__file__).resolve().parents[1]


def test_exact_z_features_for_computational_basis_state() -> None:
    probabilities = np.zeros(16)
    probabilities[0b1010] = 1.0

    features = _exact_z_features(probabilities)

    assert np.array_equal(features[:4], np.asarray([1.0, -1.0, 1.0, -1.0]))
    assert np.array_equal(features[4:], -np.ones(4))


def test_cell_circuit_has_a_unitary_matrix_and_normalized_state() -> None:
    angles = np.asarray([-1.2187204274, 0.3852904765, 0.2786508781, 0.0987122935])
    circuit = tutorial.build_feature_circuit(angles, measure=False)
    unitary = np.asarray(Operator(circuit).data)
    state = np.asarray(Statevector.from_instruction(circuit).data)

    assert unitary.shape == (16, 16)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(16), atol=1e-12)
    assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)
    assert np.allclose(state, unitary[:, 0])


def test_published_guide_keeps_the_three_routes_distinct() -> None:
    guide_dir = ROOT / "docs" / "beginner"
    guide = (guide_dir / "README.md").read_text(encoding="utf-8")
    explanation = json.loads(
        (guide_dir / "assets" / "pbmc68k_q4_explanation.json").read_text(
            encoding="utf-8"
        )
    )

    assert "educatief vier-qubit simulatormodel" in guide
    assert "letterlijke Qiskit-port" in guide
    assert "Nee, QOS-geïnspireerd" in guide
    assert "16 trainingscellen en 16 volledig gescheiden testcellen" in guide
    assert len(explanation["selected_genes"]) == 4

    expected_assets = {
        "qml_pipeline.png",
        "quantum_features.png",
        "umi_counting.png",
        "pbmc68k_q4_first_cell_circuit.png",
        "pbmc68k_q4_first_cell_unitary_magnitude.png",
    }
    assert expected_assets <= {path.name for path in (guide_dir / "assets").iterdir()}
    assert (guide_dir / "qml-van-umi-naar-circuit.docx").stat().st_size > 100_000
