#!/usr/bin/env python3
"""Local width-scaled entangler screen for the PBMC coherent B4 circuit.

The preceding B=1..4 screen showed that legacy RZZ angles are only about
0.02 radians at q40 and that chi=32 already reproduces chi=512 observables.
This runner keeps the data, B=4 circuit topology, logical depth 20, observable
panel, and label-free probes frozen.  It changes only the dimensionless RZZ
multiplier m(q).

Five pre-registered laws are tested in a cheap four-sample chi<=256 funnel.
At most two survivors are confirmed on all eight probes through chi=512.  No
classification labels enter selection.  This is a structural MPS screen, not
a predictive benchmark, hardware run, or quantum-advantage claim.
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
from qiskit.quantum_info import Statevector

import qiskit_official_qos_realdata_gate as flat_gate
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_pbmc_coherent_stream_hardness_screen as coherent
import qiskit_qos_realdata_projected_kernel_gate as projected


SCHEMA_VERSION = "1.0"
KIND = "pbmc_b4_width_scaled_entangler_mps_screen"
SCALE_LAWS = ("legacy", "sqrt_q", "q_over_4", "q_over_2", "q")
DEFAULT_WIDTHS = (20, 30, 40)
DEFAULT_EXACT_WIDTHS = (4, 6, 8, 10)
DEFAULT_PREFLIGHT_BOND_DIMENSIONS = (16, 32, 64, 128, 256)
DEFAULT_CONFIRM_BOND_DIMENSIONS = (16, 32, 64, 128, 256, 512)
DEFAULT_PROBE_SAMPLES = 8
DEFAULT_PREFLIGHT_SAMPLES = 4
DEFAULT_BLOCK_COUNT = 4
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_TOLERANCE = 1e-3
DEFAULT_MAX_CONFIRM_CANDIDATES = 2
DEFAULT_OUTPUT = Path(
    "coherent_stream_qos/pbmc_b4_width_scaled_entangler_screen.json"
)


class WidthScaleError(RuntimeError):
    pass


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_scale_laws(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one scale law")
    unknown = sorted(set(parsed) - set(SCALE_LAWS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scale laws: {', '.join(unknown)}")
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


def pair_multiplier(scale_law: str, num_qubits: int) -> float:
    """Return the dimensionless m(q) in theta_ij=0.95*m(q)*x_i*x_j."""

    q = int(num_qubits)
    if q < 1:
        raise WidthScaleError("num_qubits must be positive")
    values = {
        "legacy": 1.0,
        "sqrt_q": math.sqrt(q),
        "q_over_4": q / 4.0,
        "q_over_2": q / 2.0,
        "q": float(q),
    }
    if scale_law not in values:
        raise WidthScaleError(f"unknown scale law: {scale_law}")
    return float(values[scale_law])


def rzz_angle_statistics(
    feature_blocks: np.ndarray, *, scale_law: str
) -> dict[str, Any]:
    """Summarize every signed RZZ angle used by the B4 probe circuits."""

    blocks = np.asarray(feature_blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1] != DEFAULT_BLOCK_COUNT:
        raise WidthScaleError("angle statistics expect N x 4 x q feature blocks")
    q = int(blocks.shape[2])
    multiplier = pair_multiplier(scale_law, q)
    layers = coherent.block_interaction_layers(q, DEFAULT_BLOCK_COUNT)
    angles: list[float] = []
    for sample in blocks:
        for layer_index, edges in enumerate(layers):
            block_index = 0 if layer_index < 2 else layer_index - 1
            block = sample[block_index]
            angles.extend(
                architecture.PAIR_SCALE
                * multiplier
                * float(block[left])
                * float(block[right])
                for left, right in edges
            )
    values = np.asarray(angles, dtype=np.float64)
    absolute = np.abs(values)
    quantiles = np.quantile(absolute, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "scale_law": str(scale_law),
        "pair_multiplier": multiplier,
        "angle_count": int(len(values)),
        "nonzero_fraction": float(np.mean(absolute > 0.0)),
        "absolute_angle_quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "median", "p90", "p95", "p99", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "fraction_abs_above_pi": float(np.mean(absolute > math.pi)),
        "fraction_abs_above_two_pi": float(np.mean(absolute > 2.0 * math.pi)),
        "all_finite": bool(np.all(np.isfinite(values))),
    }


def exact_scaled_validation(
    widths: Sequence[int], scale_laws: Sequence[str]
) -> list[dict[str, Any]]:
    """Check scaled q4--q10 circuits against exact statevectors."""

    rows: list[dict[str, Any]] = []
    for width in widths:
        mappings = coherent.grid_aligned_mappings(int(width))
        feature_rows = np.asarray(
            [
                [
                    np.sin(
                        np.linspace(-0.8, 0.7, int(width))
                        + sample_offset
                        + 0.37 * block_index
                    )
                    for block_index in range(DEFAULT_BLOCK_COUNT)
                ]
                for sample_offset in (0.0, 0.31, 0.79)
            ],
            dtype=np.float64,
        )
        for scale_law in scale_laws:
            multiplier = pair_multiplier(str(scale_law), int(width))
            circuits = [
                coherent.coherent_stream_circuit(
                    blocks, pair_multiplier=multiplier
                )
                for blocks in feature_rows
            ]
            started = time.perf_counter()
            exact = np.asarray(
                [architecture.statevector_features(circuit, mappings) for circuit in circuits]
            )
            statevector_seconds = float(time.perf_counter() - started)
            mps, mps_seconds = architecture.simulate_feature_rows(
                circuits,
                mappings,
                bond_dimension=128,
                threshold=0.0,
            )
            error = np.abs(exact - mps)
            maximum_error = float(np.max(error))
            depth = int(circuits[0].depth())
            norm_error = float(
                max(
                    abs(np.linalg.norm(Statevector.from_instruction(circuit).data) - 1.0)
                    for circuit in circuits
                )
            )
            exact_z, _, _, _ = projected._standardize_selected(exact, exact)
            gamma = projected.median_gamma(exact_z)
            kernel = projected.projected_rbf_kernel(exact_z, exact_z, gamma=gamma)
            min_eigenvalue = float(np.min(np.linalg.eigvalsh(kernel)))
            passed = bool(
                maximum_error <= 1e-9
                and norm_error <= 1e-12
                and depth
                == coherent.expected_logical_depth(int(width), DEFAULT_BLOCK_COUNT)
                and np.max(np.abs(np.diag(kernel) - 1.0)) <= 1e-12
                and min_eigenvalue >= -1e-10
            )
            rows.append(
                {
                    "qubits": int(width),
                    "scale_law": str(scale_law),
                    "pair_multiplier": multiplier,
                    "actual_depth": depth,
                    "expected_depth": coherent.expected_logical_depth(
                        int(width), DEFAULT_BLOCK_COUNT
                    ),
                    "observables": len(mappings),
                    "statevector_seconds": statevector_seconds,
                    "mps_seconds": float(mps_seconds),
                    "max_abs_mps_minus_statevector": maximum_error,
                    "mean_abs_mps_minus_statevector": float(np.mean(error)),
                    "statevector_norm_max_abs_error": norm_error,
                    "kernel_minimum_eigenvalue": min_eigenvalue,
                    "kernel_diagonal_max_abs_error": float(
                        np.max(np.abs(np.diag(kernel) - 1.0))
                    ),
                    "passed": passed,
                }
            )
    return rows


def _chi_floor(convergence: Mapping[str, Any]) -> tuple[int, bool]:
    selected = convergence["selected_lower_converged_bond_dimension"]
    if selected is None:
        return int(convergence["reference_bond_dimension"]), True
    return int(selected), False


def evaluate_preflight_funnel(
    screens: Sequence[Mapping[str, Any]],
    *,
    widths: Sequence[int],
    scale_laws: Sequence[str],
    maximum_candidates: int,
) -> dict[str, Any]:
    """Choose at most two mild laws for the expensive chi512 confirmation."""

    ordered_widths = tuple(sorted({int(width) for width in widths}))
    by_key = {
        (int(row["qubits"]), str(row["scale_law"])): row for row in screens
    }
    candidates: list[dict[str, Any]] = []
    for scale_law in scale_laws:
        if any((width, str(scale_law)) not in by_key for width in ordered_widths):
            continue
        floors: list[int] = []
        unresolved: list[int] = []
        chi32: dict[str, bool] = {}
        safe_angles = True
        for width in ordered_widths:
            row = by_key[(width, str(scale_law))]
            floor, is_unresolved = _chi_floor(row["mps_convergence"])
            floors.append(floor)
            if is_unresolved:
                unresolved.append(width)
            chi32[str(width)] = bool(row["mps_convergence"]["chi32_converged"])
            safe_angles = bool(
                safe_angles
                and row["rzz_angle_statistics"]["fraction_abs_above_pi"] == 0.0
            )
        nondecreasing = all(
            floors[index + 1] >= floors[index]
            for index in range(len(floors) - 1)
        )
        growth = float(floors[-1] / max(floors[0], 1))
        qualified = bool(
            not chi32[str(ordered_widths[-1])]
            and floors[-1] >= 64
            and floors[0] <= 128
            and nondecreasing
            and growth >= 2.0
            and safe_angles
        )
        candidates.append(
            {
                "scale_law": str(scale_law),
                "widths": list(ordered_widths),
                "required_chi_floors": floors,
                "unresolved_at_reference_widths": unresolved,
                "chi32_converged_by_width": chi32,
                "nondecreasing_required_chi": bool(nondecreasing),
                "qmax_over_qmin_required_chi_growth": growth,
                "all_probe_angles_abs_le_pi": safe_angles,
                "qualified_for_confirmation": qualified,
            }
        )
    survivors = [
        str(row["scale_law"])
        for row in candidates
        if row["qualified_for_confirmation"]
    ][: int(maximum_candidates)]
    return {
        "survivors": survivors,
        "candidate_count_limit": int(maximum_candidates),
        "candidates": candidates,
        "rule": (
            "On four label-free probes through chi256: qmax chi32 must fail, "
            "the qmax chi floor must be >=64, qmin floor <=128, floors must be "
            "nondecreasing, qmax/qmin growth >=2, and all |RZZ| angles <=pi. "
            "The mildest two qualifying laws advance."
        ),
    }


def evaluate_confirmation_gate(
    screens: Sequence[Mapping[str, Any]],
    *,
    widths: Sequence[int],
    scale_laws: Sequence[str],
) -> dict[str, Any]:
    """Apply the strict eight-probe chi512 scaling gate."""

    ordered_widths = tuple(sorted({int(width) for width in widths}))
    by_key = {
        (int(row["qubits"]), str(row["scale_law"])): row for row in screens
    }
    candidates: list[dict[str, Any]] = []
    for scale_law in scale_laws:
        if any((width, str(scale_law)) not in by_key for width in ordered_widths):
            continue
        floors: list[int] = []
        unresolved: list[int] = []
        chi32: dict[str, bool] = {}
        for width in ordered_widths:
            convergence = by_key[(width, str(scale_law))]["mps_convergence"]
            floor, is_unresolved = _chi_floor(convergence)
            floors.append(floor)
            if is_unresolved:
                unresolved.append(width)
            chi32[str(width)] = bool(convergence["chi32_converged"])
        nondecreasing = all(
            floors[index + 1] >= floors[index]
            for index in range(len(floors) - 1)
        )
        growth = float(floors[-1] / max(floors[0], 1))
        passed = bool(
            not chi32[str(ordered_widths[-1])]
            and floors[-1] >= 128
            and nondecreasing
            and growth >= 4.0
        )
        candidates.append(
            {
                "scale_law": str(scale_law),
                "widths": list(ordered_widths),
                "required_chi_floors": floors,
                "unresolved_at_reference_widths": unresolved,
                "chi32_converged_by_width": chi32,
                "nondecreasing_required_chi": bool(nondecreasing),
                "qmax_over_qmin_required_chi_growth": growth,
                "passed": passed,
            }
        )
    winners = [str(row["scale_law"]) for row in candidates if row["passed"]]
    return {
        "passed": bool(winners),
        "selected_scale_law": winners[0] if winners else None,
        "eligible_scale_laws": winners,
        "candidates": candidates,
        "rule": (
            "On eight label-free probes through chi512: qmax chi32 must fail, "
            "qmax chi floor >=128, floors nondecreasing, and qmax/qmin growth >=4."
        ),
        "claim_boundary": (
            "MPS non-convergence is only a structural screening signal, not "
            "proof of classical hardness, predictive value, or quantum advantage."
        ),
    }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "widths": [int(value) for value in args.widths],
        "exact_widths": [int(value) for value in args.exact_widths],
        "scale_laws": [str(value) for value in args.scale_laws],
        "block_count": DEFAULT_BLOCK_COUNT,
        "preflight_bond_dimensions": [
            int(value) for value in args.preflight_bond_dimensions
        ],
        "confirm_bond_dimensions": [
            int(value) for value in args.confirm_bond_dimensions
        ],
        "probe_samples": int(args.probe_samples),
        "preflight_samples": int(args.preflight_samples),
        "max_confirm_candidates": int(args.max_confirm_candidates),
        "mps_threshold": float(args.mps_threshold),
        "convergence_tolerance": float(args.convergence_tolerance),
        "pbmc_cache_dir": str(args.pbmc_cache_dir),
        "positive_label": str(args.positive_label),
        "negative_label": str(args.negative_label),
        "pbmc_hash_seed": int(args.pbmc_hash_seed),
        "pbmc_max_active_genes": int(args.pbmc_max_active_genes),
        "provider_calls_allowed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=_parse_ints, default=DEFAULT_WIDTHS)
    parser.add_argument("--exact-widths", type=_parse_ints, default=DEFAULT_EXACT_WIDTHS)
    parser.add_argument("--scale-laws", type=_parse_scale_laws, default=SCALE_LAWS)
    parser.add_argument(
        "--preflight-bond-dimensions",
        type=_parse_ints,
        default=DEFAULT_PREFLIGHT_BOND_DIMENSIONS,
    )
    parser.add_argument(
        "--confirm-bond-dimensions",
        type=_parse_ints,
        default=DEFAULT_CONFIRM_BOND_DIMENSIONS,
    )
    parser.add_argument("--probe-samples", type=int, default=DEFAULT_PROBE_SAMPLES)
    parser.add_argument(
        "--preflight-samples", type=int, default=DEFAULT_PREFLIGHT_SAMPLES
    )
    parser.add_argument(
        "--max-confirm-candidates",
        type=int,
        default=DEFAULT_MAX_CONFIRM_CANDIDATES,
    )
    parser.add_argument("--mps-threshold", type=float, default=DEFAULT_MPS_THRESHOLD)
    parser.add_argument(
        "--convergence-tolerance", type=float, default=DEFAULT_TOLERANCE
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
        raise WidthScaleError("all widths must be even and at least four")
    if args.probe_samples < 2 or not 2 <= args.preflight_samples <= args.probe_samples:
        raise WidthScaleError("probe sample counts are outside range")
    if not 1 <= args.max_confirm_candidates <= 2:
        raise WidthScaleError("max_confirm_candidates must be one or two")
    if max(args.preflight_bond_dimensions) != 256:
        raise WidthScaleError("the frozen preflight reference must be chi256")
    if max(args.confirm_bond_dimensions) != 512:
        raise WidthScaleError("the frozen confirmation reference must be chi512")
    if args.convergence_tolerance <= 0.0 or args.mps_threshold < 0.0:
        raise WidthScaleError("numerical tolerances are outside range")

    configuration = _configuration(args)
    report: dict[str, Any]
    if args.output.exists() and not args.force:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("configuration") != configuration:
            raise WidthScaleError(
                "existing partial output has a different configuration; use --force"
            )
        if report.get("completed"):
            raise WidthScaleError(f"completed output already exists: {args.output}")
        print(f"Resuming partial screen from {args.output}", flush=True)
    else:
        started = time.perf_counter()
        print("Running exact q4-q10 scaled-entangler validation", flush=True)
        validation = exact_scaled_validation(args.exact_widths, args.scale_laws)
        if not all(row["passed"] for row in validation):
            failed = [
                (row["qubits"], row["scale_law"])
                for row in validation
                if not row["passed"]
            ]
            raise WidthScaleError(f"exact small-width validation failed: {failed}")
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "configuration": configuration,
            "equation_registry": {
                "rzz_angle": "theta_ij=0.95*m(q)*x_bi*x_bj",
                "scale_laws": {
                    "legacy": "m(q)=1",
                    "sqrt_q": "m(q)=sqrt(q)",
                    "q_over_4": "m(q)=q/4",
                    "q_over_2": "m(q)=q/2",
                    "q": "m(q)=q",
                },
                "mps_error": "max_j |f_j(chi)-f_j(chi_reference)|",
            },
            "exact_small_width_validation": validation,
            "preflight_screens": [],
            "confirmation_screens": [],
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

    print("Loading cached PBMC68k binary pair", flush=True)
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
            num=int(args.probe_samples),
            dtype=np.int64,
        )
    )
    preflight_positions = np.unique(
        np.linspace(
            0,
            len(probe_indices) - 1,
            num=int(args.preflight_samples),
            dtype=np.int64,
        )
    )
    if len(probe_indices) != args.probe_samples or len(preflight_positions) != args.preflight_samples:
        raise WidthScaleError("fixed probe selection produced duplicate indices")
    report["dataset"] = {
        "source": source_meta,
        "binary_pair": pair_meta,
        "sparse_matrix_sha256": flat_gate._sha256_sparse(x_pair),
        "labels_sha256": flat_gate._sha256_array(y_pair),
        "probe_indices": [int(value) for value in probe_indices],
        "probe_indices_sha256": _array_sha256(probe_indices),
        "preflight_probe_positions": [int(value) for value in preflight_positions],
        "selection": "label-free evenly spaced row indices",
        "labels_used_for_selection": False,
    }

    run_started = time.perf_counter()
    cached_blocks: dict[int, np.ndarray] = {}
    cached_stats: dict[int, dict[str, Any]] = {}
    for width in args.widths:
        blocks, encoding_stats = coherent.build_coherent_blocks(
            x_pair[probe_indices],
            num_qubits=int(width),
            block_count=DEFAULT_BLOCK_COUNT,
            hash_seed=int(args.pbmc_hash_seed),
            value_mode="log-product",
            max_active_genes=int(args.pbmc_max_active_genes),
        )
        cached_blocks[int(width)] = blocks
        cached_stats[int(width)] = encoding_stats

    ordered_widths = tuple(sorted(int(width) for width in args.widths))
    report.setdefault("preflight_pruned_scale_laws", [])

    def find_preflight(width: int, scale_law: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in report["preflight_screens"]
                if int(row["qubits"]) == int(width)
                and str(row["scale_law"]) == str(scale_law)
            ),
            None,
        )

    def prune(scale_law: str, reason: str) -> None:
        if not any(
            str(row["scale_law"]) == str(scale_law)
            for row in report["preflight_pruned_scale_laws"]
        ):
            report["preflight_pruned_scale_laws"].append(
                {"scale_law": str(scale_law), "reason": str(reason)}
            )
            _atomic_write_json(args.output, report)

    resource_checkpoint_law: str | None = None

    # Process complete width series from mild to strong.  This preserves the
    # frozen funnel while avoiding q30/q40 work for a law that q20 has already
    # made mathematically unable to meet the bounded confirmation gate.
    for scale_law in args.scale_laws:
        current_funnel = evaluate_preflight_funnel(
            report["preflight_screens"],
            widths=args.widths,
            scale_laws=args.scale_laws,
            maximum_candidates=int(args.max_confirm_candidates),
        )
        unresolved_survivor = next(
            (
                row
                for row in current_funnel["candidates"]
                if row["qualified_for_confirmation"]
                and ordered_widths[-1] in row["unresolved_at_reference_widths"]
            ),
            None,
        )
        if unresolved_survivor is not None and str(scale_law) != str(
            unresolved_survivor["scale_law"]
        ):
            resource_checkpoint_law = str(unresolved_survivor["scale_law"])
            start = list(args.scale_laws).index(resource_checkpoint_law) + 1
            for remaining in args.scale_laws[start:]:
                prune(
                    str(remaining),
                    "mildest qualifying law reached the chi256 reference at qmax; "
                    "stronger laws are outside the bounded local resource gate",
                )
            print(
                f"Resource checkpoint reached by scale={resource_checkpoint_law}",
                flush=True,
            )
            break
        if (
            str(scale_law) not in current_funnel["survivors"]
            and len(current_funnel["survivors"])
            >= int(args.max_confirm_candidates)
        ):
            prune(str(scale_law), "milder candidate limit already filled")
            print(f"Pruned scale={scale_law}: candidate limit filled", flush=True)
            continue

        qmin_row = find_preflight(ordered_widths[0], str(scale_law))
        if qmin_row is not None:
            qmin_floor, _ = _chi_floor(qmin_row["mps_convergence"])
            if qmin_floor > 128:
                prune(
                    str(scale_law),
                    "qmin chi floor exceeds 128, so fourfold growth cannot be "
                    "resolved by the chi512 confirmation",
                )
                print(
                    f"Pruned scale={scale_law}: qmin chi floor {qmin_floor} > 128",
                    flush=True,
                )
                continue

        pruned_during_widths = False
        observed_floors: list[int] = []
        for width in ordered_widths:
            existing = find_preflight(int(width), str(scale_law))
            if existing is not None:
                floor, _ = _chi_floor(existing["mps_convergence"])
                observed_floors.append(floor)
                print(f"Skipping preflight q={width} scale={scale_law}", flush=True)
                continue
            if len(observed_floors) >= 2 and observed_floors[-1] < observed_floors[-2]:
                prune(
                    str(scale_law),
                    "required chi floor decreased before qmax and cannot satisfy "
                    "the nondecreasing-width rule",
                )
                print(
                    f"Pruned scale={scale_law}: nondecreasing-width rule failed",
                    flush=True,
                )
                pruned_during_widths = True
                break
            blocks = cached_blocks[int(width)]
            key = (int(width), str(scale_law))
            multiplier = pair_multiplier(str(scale_law), int(width))
            print(f"Preflight q={width} scale={scale_law} m={multiplier:.6g}", flush=True)
            subset = blocks[preflight_positions]
            convergence = coherent.mps_convergence_screen(
                subset,
                bond_dimensions=args.preflight_bond_dimensions,
                threshold=float(args.mps_threshold),
                tolerance=float(args.convergence_tolerance),
                label=f"pre-q{width}-{scale_law}",
                pair_multiplier=multiplier,
            )
            report["preflight_screens"].append(
                {
                    "qubits": int(width),
                    "scale_law": str(scale_law),
                    "pair_multiplier": multiplier,
                    "encoded_blocks_sha256": _array_sha256(blocks),
                    "encoding_stats": cached_stats[int(width)],
                    "rzz_angle_statistics": rzz_angle_statistics(
                        blocks, scale_law=str(scale_law)
                    ),
                    "mps_convergence": convergence,
                }
            )
            report["preflight_screens"].sort(
                key=lambda row: (
                    list(args.scale_laws).index(str(row["scale_law"])),
                    int(row["qubits"]),
                )
            )
            report["elapsed_seconds_current_run"] = float(
                time.perf_counter() - run_started
            )
            _atomic_write_json(args.output, report)
            floor, _ = _chi_floor(convergence)
            observed_floors.append(floor)
        if pruned_during_widths:
            continue

    report["preflight_funnel"] = evaluate_preflight_funnel(
        report["preflight_screens"],
        widths=args.widths,
        scale_laws=args.scale_laws,
        maximum_candidates=int(args.max_confirm_candidates),
    )
    _atomic_write_json(args.output, report)
    survivors = tuple(report["preflight_funnel"]["survivors"])
    print(f"Preflight survivors: {list(survivors)}", flush=True)

    if resource_checkpoint_law is not None:
        qmax_row = next(
            row
            for row in report["preflight_screens"]
            if int(row["qubits"]) == ordered_widths[-1]
            and str(row["scale_law"]) == resource_checkpoint_law
        )
        comparison_128 = next(
            row
            for row in qmax_row["mps_convergence"]["comparisons"]
            if int(row["bond_dimension"]) == 128
        )
        comparison_256 = next(
            row
            for row in qmax_row["mps_convergence"]["comparisons"]
            if int(row["bond_dimension"]) == 256
        )
        report["resource_checkpoint"] = {
            "status": "positive_preflight_reference_unresolved",
            "scale_law": resource_checkpoint_law,
            "qmax": int(ordered_widths[-1]),
            "qmax_chi128_max_abs_difference_from_chi256": float(
                comparison_128["max_abs_difference_from_reference"]
            ),
            "qmax_chi256_four_probe_seconds": float(comparison_256["seconds"]),
            "reason": (
                "The mildest qualifying width-scaled law reaches the highest "
                "preflight reference at qmax. A local eight-probe chi512 run is "
                "not launched because it would exceed this bounded structural gate."
            ),
        }
        report["confirmation_gate"] = {
            "passed": False,
            "status": "not_run_local_chi256_reference_unresolved",
            "selected_scale_law": None,
            "structural_candidate_for_next_gate": resource_checkpoint_law,
            "claim_boundary": (
                "This is a positive resource-screen checkpoint, not confirmed "
                "MPS convergence, predictive superiority, or quantum advantage."
            ),
        }
        report["completed"] = True
        report["provider_calls_made"] = 0
        report["execution_attempted"] = False
        report["elapsed_seconds_current_run"] = float(
            time.perf_counter() - run_started
        )
        _atomic_write_json(args.output, report)
        print(
            f"Stopped at positive unresolved resource checkpoint: "
            f"{resource_checkpoint_law} -> {args.output}",
            flush=True,
        )
        return 0

    completed_confirmation = {
        (int(row["qubits"]), str(row["scale_law"]))
        for row in report["confirmation_screens"]
    }
    for scale_law in survivors:
        for width in args.widths:
            key = (int(width), str(scale_law))
            if key in completed_confirmation:
                print(f"Skipping confirmation q={width} scale={scale_law}", flush=True)
                continue
            blocks = cached_blocks[int(width)]
            multiplier = pair_multiplier(str(scale_law), int(width))
            print(f"Confirming q={width} scale={scale_law} through chi512", flush=True)
            convergence = coherent.mps_convergence_screen(
                blocks,
                bond_dimensions=args.confirm_bond_dimensions,
                threshold=float(args.mps_threshold),
                tolerance=float(args.convergence_tolerance),
                label=f"confirm-q{width}-{scale_law}",
                pair_multiplier=multiplier,
            )
            report["confirmation_screens"].append(
                {
                    "qubits": int(width),
                    "scale_law": str(scale_law),
                    "pair_multiplier": multiplier,
                    "encoded_blocks_sha256": _array_sha256(blocks),
                    "encoding_stats": cached_stats[int(width)],
                    "rzz_angle_statistics": rzz_angle_statistics(
                        blocks, scale_law=str(scale_law)
                    ),
                    "mps_convergence": convergence,
                }
            )
            report["confirmation_screens"].sort(
                key=lambda row: (
                    list(survivors).index(str(row["scale_law"])),
                    int(row["qubits"]),
                )
            )
            report["elapsed_seconds_current_run"] = float(
                time.perf_counter() - run_started
            )
            _atomic_write_json(args.output, report)

    report["confirmation_gate"] = evaluate_confirmation_gate(
        report["confirmation_screens"],
        widths=args.widths,
        scale_laws=survivors,
    )
    report["completed"] = True
    report["provider_calls_made"] = 0
    report["execution_attempted"] = False
    report["elapsed_seconds_current_run"] = float(time.perf_counter() - run_started)
    _atomic_write_json(args.output, report)
    print(
        f"Confirmation gate passed={report['confirmation_gate']['passed']} "
        f"selected={report['confirmation_gate']['selected_scale_law']} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
