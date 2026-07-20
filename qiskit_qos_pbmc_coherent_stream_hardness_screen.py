#!/usr/bin/env python3
"""Provider-free coherent PBMC block-streaming hardness screen.

The original q40 projected-kernel route hashes every PBMC sample directly to
one q-dimensional vector.  This exploratory runner instead hashes to B*q
coordinates, reshapes those coordinates into B q-dimensional blocks, and
uploads the blocks coherently into the same register.  There is no measurement
or reset between blocks.

This is a structural viability screen, not a predictive benchmark and not a
quantum-advantage claim.  Small widths are checked against exact statevectors;
q20--q40 are checked for MPS bond-dimension convergence.  The module contains
no provider authentication, circuit export, validation, or execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit_official_qos_realdata_gate as flat_gate
import qiskit_qos_pbmc68k_pairwise_screen as pairwise
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_realdata_projected_kernel_gate as projected
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as shallow


SCHEMA_VERSION = "1.0"
KIND = "pbmc_coherent_block_stream_mps_hardness_screen"
DEFAULT_WIDTHS = (20, 30, 40)
DEFAULT_BLOCK_COUNTS = (1, 2, 3, 4)
DEFAULT_EXACT_WIDTHS = (4, 6, 8, 10)
DEFAULT_BOND_DIMENSIONS = (16, 32, 64, 128, 256, 512)
DEFAULT_PROBE_SAMPLES = 8
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_OUTPUT = Path(
    "coherent_stream_qos/pbmc_q20_q40_b1_b4_hardness_screen.json"
)
GLOBAL_MEASUREMENT_BASES = ("X", "Y", "Z")


class CoherentStreamError(RuntimeError):
    pass


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


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


def grid_edges(num_qubits: int) -> list[tuple[int, int]]:
    """Return every nearest-neighbour edge of the frozen row-major grid."""

    rows, columns = architecture._grid_shape(int(num_qubits))
    horizontal = [
        (row * columns + column, row * columns + column + 1)
        for row in range(rows)
        for column in range(columns - 1)
    ]
    vertical = [
        (row * columns + column, (row + 1) * columns + column)
        for row in range(rows - 1)
        for column in range(columns)
    ]
    return [*horizontal, *vertical]


def grid_aligned_mappings(num_qubits: int) -> list[dict[int, str]]:
    """X/Y/Z singles, grid edges, and plaquettes from three global bases."""

    rows, columns = architecture._grid_shape(int(num_qubits))
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
    for left, right in grid_edges(num_qubits):
        for pauli in GLOBAL_MEASUREMENT_BASES:
            add({left: pauli, right: pauli})
    for row in range(rows - 1):
        for column in range(columns - 1):
            corners = (
                row * columns + column,
                row * columns + column + 1,
                (row + 1) * columns + column,
                (row + 1) * columns + column + 1,
            )
            for pauli in GLOBAL_MEASUREMENT_BASES:
                add({qubit: pauli for qubit in corners})
    return mappings


def grid_panel_summary(num_qubits: int) -> dict[str, Any]:
    mappings = grid_aligned_mappings(int(num_qubits))
    summary = projected.mapping_panel_summary(mappings, num_qubits=int(num_qubits))
    rows, columns = architecture._grid_shape(int(num_qubits))
    return {
        **summary,
        "grid_rows": int(rows),
        "grid_columns": int(columns),
        "grid_edges": len(grid_edges(int(num_qubits))),
        "grid_plaquettes": int((rows - 1) * (columns - 1)),
    }


def block_interaction_layers(
    num_qubits: int, block_count: int
) -> list[list[tuple[int, int]]]:
    """Frozen d8 base followed by one new interaction layer per new block."""

    if block_count not in {1, 2, 3, 4}:
        raise CoherentStreamError("block_count must be one of 1, 2, 3, 4")
    grid = architecture._grid_matchings(int(num_qubits))
    additions = [grid[2], grid[3], architecture._chord_matching(int(num_qubits))]
    return [grid[0], grid[1], *additions[: block_count - 1]]


def expected_logical_depth(num_qubits: int, block_count: int) -> int:
    """Return Qiskit's expected depth, including degenerate tiny grids.

    At the target q20--q40 widths every interaction layer is nonempty and this
    reduces to 8 + 4*(B-1).  On q4--q10 an odd grid matching can be empty, so
    an added upload contributes three rather than four depth layers.
    """

    layers = block_interaction_layers(int(num_qubits), int(block_count))
    depth = 8
    for edges in layers[2:]:
        depth += 3 + int(bool(edges))
    return int(depth)


def coherent_stream_circuit(
    feature_blocks: np.ndarray, *, pair_multiplier: float = 1.0
) -> QuantumCircuit:
    """Upload B feature blocks coherently without measurement or reset.

    Equation registry (dimensionless angles):
      upload: RY(0.75*x_bq) RZ(0.25*x_bq)
      interaction: RZZ(0.95*m(q)*x_bi*x_bj)
      mixer: RX/RY(pi/4 + 0.35*x_bq)
      readout rotation: RX(0.50*mean_b(x_bq))
    """

    blocks = np.asarray(feature_blocks, dtype=np.float64)
    if blocks.ndim != 2 or blocks.shape[0] not in {1, 2, 3, 4}:
        raise CoherentStreamError("feature blocks must have shape B x q with B=1..4")
    block_count, num_qubits = blocks.shape
    if num_qubits < 4 or num_qubits % 2:
        raise CoherentStreamError("the coherent stream requires an even q >= 4")
    if not np.all(np.isfinite(blocks)) or np.max(np.abs(blocks)) > 1.0 + 1e-7:
        raise CoherentStreamError("feature blocks must be finite and bounded by one")
    if not np.isfinite(pair_multiplier) or pair_multiplier < 0.0:
        raise CoherentStreamError("pair_multiplier must be finite and non-negative")

    layers = block_interaction_layers(num_qubits, block_count)
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))

    def upload(block: np.ndarray) -> None:
        for qubit, value in enumerate(block):
            circuit.ry(architecture.SINGLE_SCALE * float(value), qubit)
            circuit.rz(architecture.PHASE_SCALE * float(value), qubit)

    def interact(block: np.ndarray, layer_index: int) -> None:
        for left, right in layers[layer_index]:
            circuit.rzz(
                architecture.PAIR_SCALE
                * float(pair_multiplier)
                * float(block[left])
                * float(block[right]),
                left,
                right,
            )
        mixer = circuit.rx if layer_index % 2 == 0 else circuit.ry
        for qubit, value in enumerate(block):
            mixer(
                math.pi / 4.0
                + architecture.MIXER_DATA_SCALE * float(value),
                qubit,
            )

    upload(blocks[0])
    interact(blocks[0], 0)
    interact(blocks[0], 1)
    for block_index in range(1, block_count):
        upload(blocks[block_index])
        interact(blocks[block_index], block_index + 1)

    final_values = np.mean(blocks, axis=0)
    for qubit, value in enumerate(final_values):
        circuit.rx(architecture.FINAL_RX_SCALE * float(value), qubit)
    return circuit


def build_coherent_blocks(
    x,
    *,
    num_qubits: int,
    block_count: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hash PBMC pair interactions to B*q buckets and normalize per block."""

    if block_count not in {1, 2, 3, 4}:
        raise CoherentStreamError("block_count must be one of 1, 2, 3, 4")
    encoded, hash_stats = pairwise.build_pairwise_hashed_matrix(
        x,
        feature_dim=int(num_qubits * block_count),
        hash_seed=int(hash_seed),
        value_mode=str(value_mode),
        max_active_genes=max_active_genes,
    )
    blocks = np.asarray(encoded, dtype=np.float64).reshape(
        len(encoded), int(block_count), int(num_qubits)
    )
    if block_count > 1:
        norms = np.linalg.norm(blocks, axis=2, keepdims=True)
        blocks = np.divide(blocks, norms, out=np.zeros_like(blocks), where=norms > 0.0)
    else:
        norms = np.linalg.norm(blocks, axis=2, keepdims=True)
    if not np.all(np.isfinite(blocks)) or np.max(np.abs(blocks)) > 1.0 + 1e-7:
        raise CoherentStreamError("block encoding violates finite unit bounds")
    block_norms = np.linalg.norm(blocks, axis=2)
    return blocks, {
        **hash_stats,
        "ambient_hash_buckets": int(num_qubits * block_count),
        "block_count": int(block_count),
        "qubits_per_block": int(num_qubits),
        "normalization": (
            "legacy full-vector L2 normalization"
            if block_count == 1
            else "independent L2 normalization of each nonempty coherent block"
        ),
        "minimum_block_norm_after_normalization": float(np.min(block_norms)),
        "maximum_block_norm_after_normalization": float(np.max(block_norms)),
        "minimum_value": float(np.min(blocks)),
        "maximum_value": float(np.max(blocks)),
    }


def _statevector_feature_rows(
    circuits: Sequence[QuantumCircuit], mappings: Sequence[Mapping[int, str]]
) -> np.ndarray:
    return np.asarray(
        [architecture.statevector_features(circuit, mappings) for circuit in circuits],
        dtype=np.float64,
    )


def exact_small_width_validation(
    widths: Sequence[int],
    block_counts: Sequence[int],
    *,
    bond_dimension: int = 64,
) -> list[dict[str, Any]]:
    """Statevector/MPS parity plus B=1 equality to the frozen d8 circuit."""

    results: list[dict[str, Any]] = []
    for width in widths:
        mappings = grid_aligned_mappings(int(width))
        for block_count in block_counts:
            feature_rows = np.asarray(
                [
                    [
                        np.sin(
                            np.linspace(-0.8, 0.7, int(width))
                            + sample_offset
                            + 0.37 * block_index
                        )
                        for block_index in range(int(block_count))
                    ]
                    for sample_offset in (0.0, 0.31, 0.79)
                ],
                dtype=np.float64,
            )
            circuits = [coherent_stream_circuit(blocks) for blocks in feature_rows]
            started = time.perf_counter()
            exact = _statevector_feature_rows(circuits, mappings)
            exact_seconds = float(time.perf_counter() - started)
            mps, mps_seconds = architecture.simulate_feature_rows(
                circuits,
                mappings,
                bond_dimension=int(bond_dimension),
                # Small-q validation must test the circuit implementation,
                # not Aer's optional singular-value truncation heuristic.
                threshold=0.0,
            )
            error = np.abs(exact - mps)
            exact_z, _, _, _ = projected._standardize_selected(exact, exact)
            gamma = projected.median_gamma(exact_z)
            kernel = projected.projected_rbf_kernel(exact_z, exact_z, gamma=gamma)
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(kernel)))
            baseline_equal: bool | None = None
            if int(block_count) == 1:
                baseline = projected.feature_map_circuit(
                    feature_rows[0, 0], "grid_mixer_d8"
                )
                baseline_equal = bool(circuits[0] == baseline)
            maximum_error = float(np.max(error))
            expected_depth = expected_logical_depth(int(width), int(block_count))
            actual_depth = int(circuits[0].depth())
            passed = bool(
                maximum_error <= 1e-9
                and np.max(np.abs(np.diag(kernel) - 1.0)) <= 1e-12
                and minimum_eigenvalue >= -1e-10
                and actual_depth == expected_depth
                and (baseline_equal is not False)
            )
            results.append(
                {
                    "qubits": int(width),
                    "block_count": int(block_count),
                    "observables": len(mappings),
                    "expected_depth": expected_depth,
                    "actual_depth": actual_depth,
                    "statevector_seconds": exact_seconds,
                    "mps_seconds": float(mps_seconds),
                    "max_abs_mps_minus_statevector": maximum_error,
                    "mean_abs_mps_minus_statevector": float(np.mean(error)),
                    "statevector_norm_max_abs_error": float(
                        max(
                            abs(np.linalg.norm(Statevector.from_instruction(circuit).data) - 1.0)
                            for circuit in circuits
                        )
                    ),
                    "kernel_minimum_eigenvalue": minimum_eigenvalue,
                    "kernel_diagonal_max_abs_error": float(
                        np.max(np.abs(np.diag(kernel) - 1.0))
                    ),
                    "b1_exactly_matches_frozen_grid_mixer_d8": baseline_equal,
                    "representative_circuit_metrics": shallow.q40_validate.circuit_metrics(
                        circuits[0]
                    ),
                    "passed": passed,
                }
            )
    return results


def mps_convergence_screen(
    feature_blocks: np.ndarray,
    *,
    bond_dimensions: Sequence[int],
    threshold: float,
    tolerance: float,
    label: str,
    pair_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Compare each MPS chi with the largest-chi reference on fixed samples."""

    rows = np.asarray(feature_blocks, dtype=np.float64)
    if rows.ndim != 3 or not len(rows):
        raise CoherentStreamError("feature_blocks must be a nonempty N x B x q array")
    candidates = sorted({int(value) for value in bond_dimensions})
    if len(candidates) < 2 or candidates[0] < 2:
        raise CoherentStreamError("at least two positive MPS bond dimensions are needed")
    mappings = grid_aligned_mappings(int(rows.shape[2]))
    circuits = [
        coherent_stream_circuit(blocks, pair_multiplier=float(pair_multiplier))
        for blocks in rows
    ]
    features: dict[int, np.ndarray] = {}
    elapsed: dict[int, float] = {}
    for bond_dimension in candidates:
        values, seconds = architecture.simulate_feature_rows(
            circuits,
            mappings,
            bond_dimension=int(bond_dimension),
            threshold=float(threshold),
            progress_label=label,
        )
        features[int(bond_dimension)] = values
        elapsed[int(bond_dimension)] = float(seconds)

    reference_dimension = int(candidates[-1])
    reference = features[reference_dimension]
    comparisons: list[dict[str, Any]] = []
    selected: int | None = None
    for bond_dimension in candidates:
        difference = np.abs(features[int(bond_dimension)] - reference)
        maximum = float(np.max(difference))
        mean = float(np.mean(difference))
        if (
            int(bond_dimension) != reference_dimension
            and selected is None
            and maximum <= float(tolerance)
        ):
            selected = int(bond_dimension)
        comparisons.append(
            {
                "bond_dimension": int(bond_dimension),
                "seconds": elapsed[int(bond_dimension)],
                "max_abs_difference_from_reference": maximum,
                "mean_abs_difference_from_reference": mean,
                "max_abs_difference_from_chi512_reference": maximum,
                "mean_abs_difference_from_chi512_reference": mean,
                "within_tolerance": bool(maximum <= float(tolerance)),
            }
        )
    chi32 = next(
        (row for row in comparisons if row["bond_dimension"] == 32), None
    )
    return {
        "probe_samples": int(len(rows)),
        "probe_feature_blocks_sha256": _array_sha256(rows),
        "reference_bond_dimension": reference_dimension,
        "reference_is_a_convergence_target_not_an_exact_result": True,
        "selected_lower_converged_bond_dimension": selected,
        "resolution": (
            "converged_below_reference"
            if selected is not None
            else "not_converged_below_reference"
        ),
        "tolerance": float(tolerance),
        "mps_truncation_threshold": float(threshold),
        "pair_multiplier": float(pair_multiplier),
        "chi32_converged": bool(chi32 and chi32["within_tolerance"]),
        "comparisons": comparisons,
        "reference_feature_matrix_sha256": _array_sha256(reference),
        "observable_panel": grid_panel_summary(int(rows.shape[2])),
        "representative_circuit_metrics": shallow.q40_validate.circuit_metrics(
            circuits[0]
        ),
    }


def evaluate_structural_gate(
    screens: Sequence[Mapping[str, Any]],
    *,
    widths: Sequence[int],
    block_counts: Sequence[int],
) -> dict[str, Any]:
    """Apply the frozen scaling gate without treating MPS cost as proof."""

    expected_widths = tuple(sorted({int(value) for value in widths}))
    by_key = {
        (int(row["qubits"]), int(row["block_count"])): row for row in screens
    }
    candidates: list[dict[str, Any]] = []
    for block_count in sorted({int(value) for value in block_counts}):
        if any((width, block_count) not in by_key for width in expected_widths):
            continue
        required_floors: list[int] = []
        unresolved_widths: list[int] = []
        chi32_converged: dict[str, bool] = {}
        for width in expected_widths:
            convergence = by_key[(width, block_count)]["mps_convergence"]
            selected = convergence["selected_lower_converged_bond_dimension"]
            if selected is None:
                required_floors.append(int(convergence["reference_bond_dimension"]))
                unresolved_widths.append(int(width))
            else:
                required_floors.append(int(selected))
            chi32_converged[str(width)] = bool(convergence["chi32_converged"])
        q40_index = expected_widths.index(max(expected_widths))
        nondecreasing = all(
            required_floors[index + 1] >= required_floors[index]
            for index in range(len(required_floors) - 1)
        )
        growth = float(required_floors[-1] / max(required_floors[0], 1))
        passed = bool(
            block_count > 1
            and not chi32_converged[str(expected_widths[q40_index])]
            and required_floors[q40_index] >= 128
            and nondecreasing
            and growth >= 4.0
        )
        candidates.append(
            {
                "block_count": int(block_count),
                "widths": list(expected_widths),
                "required_chi_floors": required_floors,
                "unresolved_at_reference_widths": unresolved_widths,
                "chi32_converged_by_width": chi32_converged,
                "nondecreasing_required_chi": bool(nondecreasing),
                "qmax_over_qmin_required_chi_growth": growth,
                "passed": passed,
            }
        )
    winners = [int(row["block_count"]) for row in candidates if row["passed"]]
    qmax = max(expected_widths)
    all_qmax_chi32 = bool(
        candidates
        and all(row["chi32_converged_by_width"][str(qmax)] for row in candidates)
    )
    if winners:
        status = "structural_promise"
    elif all_qmax_chi32:
        status = "family_falsified_at_this_depth"
    else:
        status = "no_preregistered_scaling_winner"
    return {
        "passed": bool(winners),
        "status": status,
        "eligible_block_counts": winners,
        "candidates": candidates,
        "rule": (
            "For B>1: chi32 must fail at qmax, the required chi floor at qmax "
            "must be at least 128, floors must be nondecreasing with width, and "
            "the qmax/qmin floor ratio must be at least four."
        ),
        "claim_boundary": (
            "Failure of a bounded MPS approximation is only a structural "
            "screening signal. It is not proof of classical hardness, useful "
            "quantum advantage, or predictive superiority."
        ),
    }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "widths": [int(value) for value in args.widths],
        "block_counts": [int(value) for value in args.block_counts],
        "exact_widths": [int(value) for value in args.exact_widths],
        "bond_dimensions": [int(value) for value in args.bond_dimensions],
        "probe_samples": int(args.probe_samples),
        "mps_threshold": float(args.mps_threshold),
        "convergence_tolerance": float(args.convergence_tolerance),
        "pbmc_cache_dir": str(args.pbmc_cache_dir),
        "positive_label": str(args.positive_label),
        "negative_label": str(args.negative_label),
        "pbmc_hash_seed": int(args.pbmc_hash_seed),
        "pbmc_max_active_genes": int(args.pbmc_max_active_genes),
        "value_mode": "log-product",
        "measurement_bases": list(GLOBAL_MEASUREMENT_BASES),
        "provider_calls_allowed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=_parse_ints, default=DEFAULT_WIDTHS)
    parser.add_argument(
        "--block-counts", type=_parse_ints, default=DEFAULT_BLOCK_COUNTS
    )
    parser.add_argument("--exact-widths", type=_parse_ints, default=DEFAULT_EXACT_WIDTHS)
    parser.add_argument(
        "--bond-dimensions", type=_parse_ints, default=DEFAULT_BOND_DIMENSIONS
    )
    parser.add_argument("--probe-samples", type=int, default=DEFAULT_PROBE_SAMPLES)
    parser.add_argument("--mps-threshold", type=float, default=DEFAULT_MPS_THRESHOLD)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument("--pbmc-cache-dir", default="data_cache/pbmc68k")
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--pbmc-hash-seed", type=int, default=7)
    parser.add_argument("--pbmc-max-active-genes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(width < 4 or width % 2 for width in (*args.widths, *args.exact_widths)):
        raise CoherentStreamError("all widths must be even and at least four")
    if any(block_count not in {1, 2, 3, 4} for block_count in args.block_counts):
        raise CoherentStreamError("block counts must be selected from 1,2,3,4")
    if len(set(args.widths)) != len(args.widths):
        raise CoherentStreamError("widths must not contain duplicates")
    if len(set(args.block_counts)) != len(args.block_counts):
        raise CoherentStreamError("block counts must not contain duplicates")
    if args.probe_samples < 2:
        raise CoherentStreamError("at least two fixed probe samples are required")
    if args.convergence_tolerance <= 0.0 or args.mps_threshold < 0.0:
        raise CoherentStreamError("numerical tolerances are outside range")
    if max(args.bond_dimensions) < 128:
        raise CoherentStreamError("the reference bond dimension must be at least 128")

    configuration = _configuration(args)
    report: dict[str, Any]
    if args.output.exists() and not args.force:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("configuration") != configuration:
            raise CoherentStreamError(
                "existing partial output has a different configuration; use --force"
            )
        if report.get("completed"):
            raise CoherentStreamError(f"completed output already exists: {args.output}")
        print(f"Resuming partial screen from {args.output}", flush=True)
    else:
        started = time.perf_counter()
        print("Running exact q4-q10 coherent-stream validation", flush=True)
        validation = exact_small_width_validation(
            args.exact_widths,
            args.block_counts,
            bond_dimension=max(64, min(128, max(args.bond_dimensions))),
        )
        if not all(row["passed"] for row in validation):
            raise CoherentStreamError("exact small-width validation failed")
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "configuration": configuration,
            "equation_registry": {
                "feature_hash": "PBMC pair products -> B*q deterministic hash buckets",
                "block_stream": "U(x_B)...U(x_2)U(x_1)H|0> without reset",
                "projected_features": "homogeneous X/Y/Z grid-aligned expectations",
                "mps_error": "max_j |f_j(chi)-f_j(chi_reference)|",
            },
            "exact_small_width_validation": validation,
            "screens": [],
            "completed": False,
            "provider_calls_made": 0,
            "execution_attempted": False,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "elapsed_seconds_before_dataset_load": float(time.perf_counter() - started),
        }
        _atomic_write_json(args.output, report)

    print("Loading the cached PBMC68k binary pair", flush=True)
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.pbmc_cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(args.positive_label),
        negative_label=str(args.negative_label),
    )
    probe_indices = np.unique(
        np.linspace(
            0,
            x_pair.shape[0] - 1,
            num=min(int(args.probe_samples), int(x_pair.shape[0])),
            dtype=np.int64,
        )
    )
    if len(probe_indices) != int(args.probe_samples):
        raise CoherentStreamError("could not construct the requested fixed probe set")
    report["dataset"] = {
        "source": source_meta,
        "binary_pair": pair_meta,
        "sparse_matrix_sha256": flat_gate._sha256_sparse(x_pair),
        "labels_sha256": flat_gate._sha256_array(y_pair),
        "probe_indices": [int(value) for value in probe_indices],
        "probe_indices_sha256": _array_sha256(probe_indices),
        "probe_selection": "label-free evenly spaced row indices",
        "probe_labels_not_used_for_selection_or_screen": True,
    }
    completed_keys = {
        (int(row["qubits"]), int(row["block_count"])) for row in report["screens"]
    }
    run_started = time.perf_counter()
    for width in args.widths:
        for block_count in args.block_counts:
            key = (int(width), int(block_count))
            if key in completed_keys:
                print(f"Skipping completed q={width} B={block_count}", flush=True)
                continue
            print(f"Encoding and screening q={width} B={block_count}", flush=True)
            blocks, encoding_stats = build_coherent_blocks(
                x_pair[probe_indices],
                num_qubits=int(width),
                block_count=int(block_count),
                hash_seed=int(args.pbmc_hash_seed),
                value_mode="log-product",
                max_active_genes=int(args.pbmc_max_active_genes),
            )
            convergence = mps_convergence_screen(
                blocks,
                bond_dimensions=args.bond_dimensions,
                threshold=float(args.mps_threshold),
                tolerance=float(args.convergence_tolerance),
                label=f"q{width}-B{block_count}",
            )
            report["screens"].append(
                {
                    "qubits": int(width),
                    "block_count": int(block_count),
                    "encoded_blocks_sha256": _array_sha256(blocks),
                    "encoding_stats": encoding_stats,
                    "mps_convergence": convergence,
                }
            )
            report["screens"].sort(
                key=lambda row: (int(row["qubits"]), int(row["block_count"]))
            )
            report["elapsed_seconds_current_run"] = float(
                time.perf_counter() - run_started
            )
            _atomic_write_json(args.output, report)

    report["structural_gate"] = evaluate_structural_gate(
        report["screens"], widths=args.widths, block_counts=args.block_counts
    )
    report["completed"] = True
    report["provider_calls_made"] = 0
    report["execution_attempted"] = False
    report["elapsed_seconds_current_run"] = float(time.perf_counter() - run_started)
    _atomic_write_json(args.output, report)
    print(
        f"Structural gate: {report['structural_gate']['status']} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
