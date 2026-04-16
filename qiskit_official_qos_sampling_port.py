#!/usr/bin/env python3
"""Small Qiskit port of the official QOS sampling kernels.

This is the first faithful bridge from the official JAX repo to Qiskit.
It intentionally targets the smallest kernels that admit direct unitary
simulation in Qiskit:

- boolean oracle sketch from ``official_qos/qos_sampling.py``
- flat state sketch from ``official_qos/qos_sampling.py``

The official real-dataset scripts do not invoke the QOS kernels directly;
they report the paper's space curves. So the correct first Qiskit port is
to establish kernel-level parity here before connecting that port to a
dataset path such as Splice.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parent
OFFICIAL_QOS_ROOT = ROOT / "official_qos"
if str(OFFICIAL_QOS_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_QOS_ROOT))

import qos_sampling  # noqa: E402
import qsvt  # noqa: E402
import utils as official_utils  # noqa: E402


def require_power_of_two(dim: int) -> int:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"dim must be a positive power of two, got {dim}")
    return int(math.log2(dim))


def random_truth_table(dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=dim, dtype=np.int32)


def random_flat_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=dim, replace=True)


def sample_uniform_indices(dim: int, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, dim, size=num_samples, dtype=np.int32)


def sample_from_truth_table(
    truth_table: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_indices = sample_uniform_indices(int(truth_table.shape[0]), num_samples, rng)
    sampled_values = truth_table[sampled_indices]
    return sampled_indices, sampled_values


def sample_from_vector(
    vector: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_indices = sample_uniform_indices(int(vector.shape[0]), num_samples, rng)
    sampled_values = vector[sampled_indices]
    return sampled_indices, sampled_values


def boolean_phase_from_samples(
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    dim: int,
) -> np.ndarray:
    phase = np.zeros(dim, dtype=np.float64)
    np.add.at(phase, sampled_indices, sampled_values.astype(np.float64))
    phase *= np.pi * dim / sampled_indices.shape[0]
    return phase


def flat_phase_from_samples(
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    dim: int,
) -> np.ndarray:
    phase = np.zeros(dim, dtype=np.float64)
    contributions = (1.0 - sampled_values.astype(np.float64)) / 2.0
    np.add.at(phase, sampled_indices, contributions)
    phase *= np.pi * dim / sampled_indices.shape[0]
    return phase


def build_boolean_oracle_circuit_from_samples(
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    dim: int,
) -> tuple[QuantumCircuit, np.ndarray]:
    num_qubits = require_power_of_two(dim)
    phase = boolean_phase_from_samples(sampled_indices, sampled_values, dim)
    diag = np.exp(1j * phase)

    qc = QuantumCircuit(num_qubits)
    qc.append(DiagonalGate(diag), range(num_qubits))
    return qc, diag


def build_flat_state_circuit_from_samples(
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    dim: int,
) -> tuple[QuantumCircuit, np.ndarray]:
    num_qubits = require_power_of_two(dim)
    phase = flat_phase_from_samples(sampled_indices, sampled_values, dim)
    diag = np.exp(1j * phase)

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.append(DiagonalGate(diag), range(num_qubits))
    return qc, diag


def top_entries(vec: np.ndarray, top_k: int = 8) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(vec))[::-1][:top_k]
    width = require_power_of_two(vec.shape[0])
    items: list[dict[str, Any]] = []
    for idx in order:
        val = vec[idx]
        items.append(
            {
                "basis": format(int(idx), f"0{width}b"),
                "abs": float(np.abs(val)),
                "real": float(np.real(val)),
                "imag": float(np.imag(val)),
            }
        )
    return items


def measure_counts(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    measured = circuit.copy()
    measured.measure_all()
    sim = AerSimulator()
    result = sim.run(measured, shots=shots).result()
    counts = result.get_counts()
    return {str(k): int(v) for k, v in counts.items()}


@functools.lru_cache(maxsize=None)
def parity_sign_matrix(dim: int) -> np.ndarray:
    signs = np.empty((dim, dim), dtype=np.float64)
    for j in range(dim):
        for u in range(dim):
            signs[j, u] = 1.0 if ((j & u).bit_count() % 2 == 0) else -1.0
    return signs


def qsp_phase_factors(angle: float) -> np.ndarray:
    return np.exp(1j * angle * np.array([1.0, -1.0], dtype=np.float64))


@functools.lru_cache(maxsize=None)
def hadamard_for_dim(dim: int) -> np.ndarray:
    return np.asarray(
        official_utils.unnormalized_hadamard_transform(require_power_of_two(dim)),
        dtype=np.float64,
    )


@functools.lru_cache(maxsize=None)
def general_state_angle_set(degree: int) -> np.ndarray:
    def func(x: np.ndarray) -> np.ndarray:
        return np.arcsin(x) / np.arcsin(1.0)

    return np.asarray(
        qsvt.get_qsvt_angles(
            func=func,
            degree=degree,
            rescale=1.0,
            cheb_domain=(-np.sin(1.0), np.sin(1.0)),
            ensure_bounded=False,
        )
    )


def imperfect_qsvt_diag_numpy(
    block_sequence: np.ndarray,
    angle_set: np.ndarray,
) -> np.ndarray:
    if block_sequence.shape[0] != angle_set.shape[0] - 1:
        raise ValueError("number of imperfect blocks must match len(angle_set)-1")

    dim = block_sequence.shape[-1]
    circ = np.zeros((dim, 2, 2), dtype=np.complex128)
    initial = np.exp(1j * (-np.pi / 2) * angle_set.shape[0]) * np.diag(
        qsp_phase_factors(float(angle_set[0]))
    )
    circ[:] = initial

    for gate_idx, angle in enumerate(angle_set[1:]):
        u_gate = block_sequence[gate_idx].transpose(2, 0, 1)
        circ = np.matmul(circ, u_gate)
        circ = circ * qsp_phase_factors(float(angle))[None, None, :]

    return circ.transpose(1, 2, 0)


def run_boolean_case(dim: int, num_samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth_table = random_truth_table(dim, rng)
    sampled_indices, sampled_values = sample_from_truth_table(truth_table, num_samples, rng)
    return run_boolean_case_from_truth_table(
        truth_table,
        sampled_indices,
        sampled_values,
    )


def run_boolean_case_from_truth_table(
    truth_table: np.ndarray,
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
) -> dict[str, Any]:
    dim = int(truth_table.shape[0])
    num_samples = int(sampled_indices.shape[0])
    jax_diag = np.asarray(
        qos_sampling.q_oracle_sketch_boolean((sampled_indices, sampled_values), dim)
    )
    qc, analytic_diag = build_boolean_oracle_circuit_from_samples(
        sampled_indices, sampled_values, dim
    )
    qiskit_diag = np.diag(Operator(qc).data)

    return {
        "dim": dim,
        "num_qubits": require_power_of_two(dim),
        "num_samples": num_samples,
        "truth_table_weight": int(np.sum(truth_table)),
        "max_abs_err_vs_jax": float(np.max(np.abs(qiskit_diag - jax_diag))),
        "max_abs_err_vs_analytic": float(np.max(np.abs(qiskit_diag - analytic_diag))),
        "top_diagonal_entries": top_entries(qiskit_diag),
    }


def run_flat_case(dim: int, num_samples: int, seed: int, shots: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vector = random_flat_vector(dim, rng)
    sampled_indices, sampled_values = sample_from_vector(vector, num_samples, rng)
    return run_flat_case_from_vector(vector, sampled_indices, sampled_values, shots)


def run_flat_case_from_vector(
    vector: np.ndarray,
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    shots: int,
) -> dict[str, Any]:
    dim = int(vector.shape[0])
    num_samples = int(sampled_indices.shape[0])
    jax_state = np.asarray(qos_sampling.q_state_sketch_flat((sampled_indices, sampled_values), dim))
    qc, analytic_diag = build_flat_state_circuit_from_samples(
        sampled_indices, sampled_values, dim
    )
    qiskit_state = Statevector.from_instruction(qc).data
    overlap = np.vdot(jax_state, qiskit_state)
    infidelity = float(1.0 - np.abs(overlap) ** 2)

    return {
        "dim": dim,
        "num_qubits": require_power_of_two(dim),
        "num_samples": num_samples,
        "shots": shots,
        "max_abs_err_vs_jax": float(np.max(np.abs(qiskit_state - jax_state))),
        "max_abs_err_vs_analytic": float(
            np.max(np.abs(qiskit_state - analytic_diag / np.sqrt(dim)))
        ),
        "state_infidelity_vs_jax": infidelity,
        "top_state_entries": top_entries(qiskit_state),
        "counts": measure_counts(qc, shots),
    }


def run_general_state_case(
    dim: int,
    num_samples: int,
    seed: int,
    degree: int,
) -> dict[str, Any]:
    if num_samples % degree != 0:
        raise ValueError("num_samples must be divisible by degree for the official sampling kernel")

    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float64)
    sampled_indices, sampled_values = sample_from_vector(vector, num_samples, rng)
    return run_general_state_case_from_vector(
        vector,
        sampled_indices,
        sampled_values,
        seed,
        degree,
    )


def run_general_state_case_from_vector(
    vector: np.ndarray,
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    seed: int,
    degree: int,
) -> dict[str, Any]:
    details = general_state_sketch_from_vector_samples(
        vector,
        sampled_indices,
        sampled_values,
        seed,
        degree,
    )

    jax_state = details["jax_state"]
    qiskit_state = details["qiskit_state"]
    pre_hadamard = details["pre_hadamard"]
    angle_set = details["angle_set"]
    norm_jax = float(np.linalg.norm(jax_state))
    norm_qiskit = float(np.linalg.norm(qiskit_state))
    raw_l2_err = float(np.linalg.norm(qiskit_state - jax_state))
    if norm_jax > 0 and norm_qiskit > 0:
        overlap = np.vdot(jax_state / norm_jax, qiskit_state / norm_qiskit)
        normalized_infidelity = float(1.0 - np.abs(overlap) ** 2)
    else:
        normalized_infidelity = 1.0

    dim = int(vector.shape[0])
    num_samples = int(sampled_indices.shape[0])

    return {
        "dim": dim,
        "num_qubits": require_power_of_two(dim),
        "num_samples": num_samples,
        "degree": degree,
        "angle_count": int(angle_set.shape[0]),
        "jax_state_norm": norm_jax,
        "qiskit_state_norm": norm_qiskit,
        "raw_l2_err_vs_jax": raw_l2_err,
        "max_abs_err_vs_jax": float(np.max(np.abs(qiskit_state - jax_state))),
        "normalized_state_infidelity_vs_jax": normalized_infidelity,
        "pre_hadamard_top_entries": top_entries(pre_hadamard.astype(np.complex128)),
        "top_state_entries": top_entries(qiskit_state.astype(np.complex128)),
    }


def general_state_sketch_from_vector_samples(
    vector: np.ndarray,
    sampled_indices: np.ndarray,
    sampled_values: np.ndarray,
    seed: int,
    degree: int,
    include_jax: bool = True,
) -> dict[str, np.ndarray]:
    dim = int(vector.shape[0])
    num_samples = int(sampled_indices.shape[0])
    if num_samples % degree != 0:
        raise ValueError("num_samples must be divisible by degree for the official sampling kernel")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        zero = np.zeros(dim, dtype=np.float64)
        return {
            "jax_state": zero.copy() if include_jax else None,
            "qiskit_state": zero.copy(),
            "pre_hadamard": zero.copy(),
            "angle_set": general_state_angle_set(degree),
            "random_signs": np.ones(dim, dtype=np.float64),
        }

    key = jrandom.PRNGKey(seed)
    if include_jax:
        jax_state = np.asarray(
            qos_sampling.q_state_sketch(
                (sampled_indices, sampled_values),
                dim,
                norm,
                key,
                degree=degree,
            )
        )
    else:
        jax_state = None

    _, subkey = jrandom.split(key)
    random_signs = np.asarray(
        jrandom.choice(subkey, jnp.array([1.0, -1.0]), shape=(dim,))
    ).astype(np.float64)

    normalized_values = sampled_values.astype(np.float64) / norm
    grouped_indices = sampled_indices.reshape(degree, -1)
    grouped_values = normalized_values.reshape(degree, -1)
    aggregated = np.zeros((degree, dim), dtype=np.float64)
    for row in range(degree):
        np.add.at(aggregated[row], grouped_indices[row], grouped_values[row])
    aggregated /= (num_samples / degree)
    aggregated *= random_signs[None, :]

    contribution = aggregated @ parity_sign_matrix(dim)
    contribution *= dim / norm / 3.0

    angle_set = general_state_angle_set(degree)

    sin = np.sin(contribution)
    cos = np.cos(contribution)
    block_sequence = np.stack(
        [
            np.stack([sin, cos], axis=0),
            np.stack([cos, -sin], axis=0),
        ],
        axis=0,
    ).transpose(2, 0, 1, 3)

    effective = imperfect_qsvt_diag_numpy(block_sequence[:-1], angle_set)
    pre_hadamard = np.real(effective[0, 0]) / np.sqrt(dim)
    qiskit_state = random_signs * ((hadamard_for_dim(dim) @ pre_hadamard) / np.sqrt(dim))

    return {
        "jax_state": jax_state,
        "qiskit_state": qiskit_state,
        "pre_hadamard": pre_hadamard,
        "angle_set": angle_set,
        "random_signs": random_signs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small Qiskit port of the official QOS sampling kernels"
    )
    parser.add_argument("--dim", type=int, default=16, help="Power-of-two support size")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of sampled oracle/vector accesses",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shots", type=int, default=4096, help="Aer shots for flat-state readout")
    parser.add_argument(
        "--general-degree",
        type=int,
        default=4,
        help="Polynomial/QSVT degree for the official general-state sampling kernel",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full result JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_power_of_two(args.dim)

    payload = {
        "paper_repo": str(OFFICIAL_QOS_ROOT),
        "seed": args.seed,
        "boolean_kernel": run_boolean_case(args.dim, args.num_samples, args.seed),
        "flat_kernel": run_flat_case(args.dim, args.num_samples, args.seed + 1, args.shots),
        "general_state_kernel": run_general_state_case(
            args.dim,
            args.num_samples,
            args.seed + 2,
            args.general_degree,
        ),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
