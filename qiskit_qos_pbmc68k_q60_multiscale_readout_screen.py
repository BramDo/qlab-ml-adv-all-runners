#!/usr/bin/env python3
"""Local multiscale readout screen for the frozen q60 grid-mixer circuit.

All observables are homogeneous X, Y, or Z Pauli strings, so the existing
three global measurement bases can estimate them without adding hardware
circuits.  Nested panels are ranked with repeated training-only CV.  The fixed
test split is evaluated only for the CV winner.  This runner has no provider,
Fire Opal, validation, or execution path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture_screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
ARCHITECTURE = "grid_mixer_d12"
PANEL_NAMES = ("local", "multiscale_pairs", "multiscale_strings")
DEFAULT_PRIOR_SCREEN = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_scrambled_mixer_architecture_screen.json"
)
PINNED_PRIOR_SCREEN_SHA256 = (
    "d78a3b10bbc67b7d1bd00f13a5963351d274bdcf87cae3d0d28d9874df367676"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_grid_d12_multiscale_readout_screen.json"
)
DEFAULT_BOND_DIMENSION = 64
DEFAULT_PROBE_BOND_DIMENSIONS = (32, 128)
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_SELECTED_FEATURES = 24
DEFAULT_SHOT_INTENT = 128
DEFAULT_CV_SPLITS = 4
DEFAULT_CV_REPEATS = 5
DEFAULT_CV_SEED = 6121
DEFAULT_SHOT_NOISE_REPLICATES = 500
DEFAULT_SHOT_NOISE_SEED = 60128

RunnerError = architecture_screen.RunnerError


def _sha256_file(path: Path) -> str:
    return q60.q40_validate._sha256_file(path)


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


def _mapping_key(mapping: Mapping[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((int(qubit), str(pauli)) for qubit, pauli in mapping.items()))


def build_multiscale_panels(
    num_qubits: int = 60,
) -> tuple[list[dict[int, str]], dict[str, list[int]]]:
    if num_qubits != 60:
        raise RunnerError("The pre-registered multiscale panel is pinned to 60 qubits")
    mappings: list[dict[int, str]] = []
    seen: set[tuple[tuple[int, str], ...]] = set()

    def add(mapping: Mapping[int, str]) -> None:
        normalized = {int(qubit): str(pauli) for qubit, pauli in mapping.items()}
        key = _mapping_key(normalized)
        if key not in seen:
            seen.add(key)
            mappings.append(normalized)

    for mapping in toy.pauli_feature_mappings(num_qubits, family="local"):
        add(mapping)
    panels: dict[str, list[int]] = {"local": list(range(len(mappings)))}

    for distance in (2, 4, 8, 15, 30):
        step = max(1, distance // 2)
        for left in range(0, num_qubits - distance, step):
            for pauli in ("X", "Y", "Z"):
                add({left: pauli, left + distance: pauli})
    panels["multiscale_pairs"] = list(range(len(mappings)))

    rows, columns = architecture_screen._grid_shape(num_qubits)
    # Sampled 2x2 grid plaquettes: 4-qubit correlations.
    for row in range(0, rows - 1, 2):
        for column in range(0, columns - 1, 2):
            support = (
                row * columns + column,
                row * columns + column + 1,
                (row + 1) * columns + column,
                (row + 1) * columns + column + 1,
            )
            for pauli in ("X", "Y", "Z"):
                add({qubit: pauli for qubit in support})
    # Dyadic four-point strings at increasing distance.
    for distance in (2, 4, 8):
        for start in range(0, num_qubits - 3 * distance, 2 * distance):
            support = tuple(start + offset * distance for offset in range(4))
            for pauli in ("X", "Y", "Z"):
                add({qubit: pauli for qubit in support})
    # Sampled 2x4 grid rectangles: 8-qubit correlations.
    for row in range(0, rows - 1, 2):
        for column in range(0, columns - 3, 3):
            support = tuple(
                local_row * columns + local_column
                for local_row in (row, row + 1)
                for local_column in range(column, column + 4)
            )
            for pauli in ("X", "Y", "Z"):
                add({qubit: pauli for qubit in support})
    panels["multiscale_strings"] = list(range(len(mappings)))
    return mappings, panels


def panel_summary(
    mappings: Sequence[Mapping[int, str]], indices: Sequence[int]
) -> dict[str, Any]:
    selected = [mappings[index] for index in indices]
    support_counts = Counter(len(mapping) for mapping in selected)
    basis_counts = Counter(next(iter(set(mapping.values()))) for mapping in selected)
    return {
        "observable_count": len(selected),
        "support_size_counts": {
            str(key): int(value) for key, value in sorted(support_counts.items())
        },
        "measurement_basis_counts": dict(sorted(basis_counts.items())),
        "all_homogeneous_xyz": all(
            len(set(mapping.values())) == 1
            and next(iter(mapping.values())) in {"X", "Y", "Z"}
            for mapping in selected
        ),
        "compatible_with_existing_global_xyz_measurements": True,
    }


def _load_prior_screen(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RunnerError(f"Prior architecture screen is missing: {path}")
    if _sha256_file(path) != PINNED_PRIOR_SCREEN_SHA256:
        raise RunnerError("Prior architecture-screen hash differs from the pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != "pbmc68k_q60_noncommuting_architecture_local_screen"
        or payload.get("status") != "complete_local_only"
        or payload.get("execution_attempted") is not False
        or payload.get("selection", {}).get("winner") != ARCHITECTURE
    ):
        raise RunnerError("Prior screen does not freeze the expected local winner")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "winner": ARCHITECTURE,
        "provider_calls": [],
        "ready_for_provider_validation": bool(
            payload.get("gates", {}).get("ready_for_provider_validation", False)
        ),
    }


def _panel_cv(
    base_features: np.ndarray,
    fold_sketch_features: np.ndarray,
    panel_indices: Sequence[int],
    data: architecture_screen.SeedData,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    selected_feature_count: int,
    shot_intent: int,
    cv_splits: int,
) -> dict[str, Any]:
    panel_indices_array = np.asarray(panel_indices, dtype=np.int64)
    train_stop = 1 + len(data.encoded_train)
    query_train = base_features[1:train_stop][:, panel_indices_array]
    rows: list[dict[str, Any]] = []
    for fold_index, (fit, validation) in enumerate(folds):
        selected, selection_scores = q60.select_train_only_features(
            query_train[fit],
            data.y_train[fit],
            count=selected_feature_count,
            shot_intent=shot_intent,
        )
        model = fold_sketch_features[fold_index, panel_indices_array][selected]
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
                "repeat": fold_index // cv_splits,
                "selected_master_indices": [
                    int(panel_indices_array[index]) for index in selected
                ],
                "minimum_selected_score": float(np.min(selection_scores[selected])),
                "fit": architecture_screen._balanced_metrics(
                    data.y_train[fit], fit_scores
                ),
                "validation": architecture_screen._balanced_metrics(
                    data.y_train[validation], validation_scores
                ),
            }
        )
    values = np.asarray(
        [row["validation"]["balanced_accuracy"] for row in rows], dtype=np.float64
    )
    repeat_means = [
        float(np.mean(values[repeat * cv_splits : (repeat + 1) * cv_splits]))
        for repeat in range(len(values) // cv_splits)
    ]
    return {
        "folds": rows,
        "cv_mean_balanced_accuracy": float(np.mean(values)),
        "cv_std_balanced_accuracy": float(np.std(values)),
        "cv_worst_balanced_accuracy": float(np.min(values)),
        "repeat_mean_balanced_accuracy": repeat_means,
        "repeat_mean_worst": float(np.min(repeat_means)),
        "selection_uses_fit_fold_only": True,
    }


def _final_panel_evaluation(
    features: np.ndarray,
    panel_indices: Sequence[int],
    mappings: Sequence[Mapping[int, str]],
    data: architecture_screen.SeedData,
    *,
    selected_feature_count: int,
    shot_intent: int,
) -> dict[str, Any]:
    panel_indices_array = np.asarray(panel_indices, dtype=np.int64)
    panel_features = features[:, panel_indices_array]
    train_stop = 1 + len(data.encoded_train)
    query_train = panel_features[1:train_stop]
    query_test = panel_features[train_stop:]
    selected, selection_scores = q60.select_train_only_features(
        query_train,
        data.y_train,
        count=selected_feature_count,
        shot_intent=shot_intent,
    )
    master_selected = [int(panel_indices_array[index]) for index in selected]
    model = panel_features[0, selected]
    head_train = q60._head_features(model, query_train[:, selected])
    head_test = q60._head_features(model, query_test[:, selected])
    train_scores, test_scores = q60._ridge_scores(
        head_train, head_test, data.y_train
    )
    return {
        "selected_master_indices": master_selected,
        "selected_observables": [
            {
                "master_index": index,
                "mapping": [
                    {"qubit": int(qubit), "pauli": str(pauli)}
                    for qubit, pauli in sorted(mappings[index].items())
                ],
            }
            for index in master_selected
        ],
        "minimum_selected_score": float(np.min(selection_scores[selected])),
        "selection_uses_training_only": True,
        "train": architecture_screen._balanced_metrics(data.y_train, train_scores),
        "fixed_test": architecture_screen._balanced_metrics(data.y_test, test_scores),
    }


def _shot_noise_probe(
    features: np.ndarray,
    panel_indices: Sequence[int],
    data: architecture_screen.SeedData,
    *,
    selected_feature_count: int,
    shots: int,
    replicates: int,
    seed: int,
    classical_test_accuracy: float,
) -> dict[str, Any]:
    panel = np.asarray(features[:, panel_indices], dtype=np.float64)
    probabilities = np.clip(0.5 * (1.0 + panel), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    train_stop = 1 + len(data.encoded_train)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = 2.0 * rng.binomial(shots, probabilities) / shots - 1.0
        query_train = sampled[1:train_stop]
        query_test = sampled[train_stop:]
        selected, _ = q60.select_train_only_features(
            query_train,
            data.y_train,
            count=selected_feature_count,
            shot_intent=shots,
        )
        model = sampled[0, selected]
        head_train = q60._head_features(model, query_train[:, selected])
        head_test = q60._head_features(model, query_test[:, selected])
        _, test_scores = q60._ridge_scores(head_train, head_test, data.y_train)
        values[replicate] = q60._balanced_accuracy(data.y_test, test_scores)
    return {
        "method": "independent_binomial_expectation_sampling",
        "replicates": replicates,
        "seed": seed,
        "shots_per_observable": shots,
        "mean_test_balanced_accuracy": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "ci95_percentile": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "fraction_beating_classical_fixed_test": float(
            np.mean(values > classical_test_accuracy)
        ),
        "does_not_model_shared_basis_covariance_or_hardware_drift": True,
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.selected_features < 1 or args.shot_intent < 1:
        raise RunnerError("Selected features and shot intent must be positive")
    started = time.perf_counter()
    prior = _load_prior_screen(args.prior_screen)
    data = architecture_screen.load_seed_data(args)
    mappings, panels = build_multiscale_panels()
    if args.selected_features > min(len(indices) for indices in panels.values()):
        raise RunnerError("Selected-feature count exceeds the smallest panel")
    splitter = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.cv_seed,
    )
    folds = list(splitter.split(data.encoded_train, data.y_train))
    classical = architecture_screen.classical_cv_reference(data, folds)
    vectors = architecture_screen._parameter_vectors(data)
    fold_vectors = architecture_screen._fold_sketch_vectors(data, folds)
    circuits = [
        architecture_screen.architecture_circuit(vector, ARCHITECTURE)
        for vector in vectors
    ]
    fold_circuits = [
        architecture_screen.architecture_circuit(vector, ARCHITECTURE)
        for vector in fold_vectors
    ]
    features, primary_seconds = architecture_screen.simulate_feature_rows(
        [*circuits, *fold_circuits],
        mappings,
        bond_dimension=args.bond_dimension,
        threshold=args.mps_threshold,
        progress_label="grid-d12-multiscale",
    )
    base_features = features[: len(circuits)]
    fold_features = features[len(circuits) :]
    probe_indices = [0, 1, 8, 16, 32, 33, 48, 64]
    probe_comparisons: dict[str, Any] = {}
    for bond_dimension in args.probe_bond_dimensions:
        probe, seconds = architecture_screen.simulate_feature_rows(
            [circuits[index] for index in probe_indices],
            mappings,
            bond_dimension=bond_dimension,
            threshold=args.mps_threshold,
            progress_label="grid-d12-multiscale-probe",
        )
        difference = np.abs(probe - base_features[probe_indices])
        probe_comparisons[str(bond_dimension)] = {
            "seconds": seconds,
            "max_abs_difference_from_primary": float(np.max(difference)),
            "mean_abs_difference_from_primary": float(np.mean(difference)),
            "median_abs_difference_from_primary": float(np.median(difference)),
        }
    high_probe = str(max(args.probe_bond_dimensions))
    mps_converged = bool(
        probe_comparisons[high_probe]["max_abs_difference_from_primary"]
        <= args.convergence_tolerance
    )

    panel_rows: list[dict[str, Any]] = []
    for name in PANEL_NAMES:
        indices = panels[name]
        panel_rows.append(
            {
                "panel": name,
                "summary": panel_summary(mappings, indices),
                "structural_hardness": architecture_screen.structural_hardness(
                    ARCHITECTURE, [mappings[index] for index in indices], 60
                ),
                "training_cross_validation": _panel_cv(
                    base_features,
                    fold_features,
                    indices,
                    data,
                    folds,
                    selected_feature_count=args.selected_features,
                    shot_intent=args.shot_intent,
                    cv_splits=args.cv_splits,
                ),
            }
        )
    winner = max(
        panel_rows,
        key=lambda row: (
            float(row["training_cross_validation"]["cv_mean_balanced_accuracy"]),
            float(row["training_cross_validation"]["repeat_mean_worst"]),
            -int(row["summary"]["observable_count"]),
        ),
    )
    winner_name = str(winner["panel"])
    final = _final_panel_evaluation(
        base_features,
        panels[winner_name],
        mappings,
        data,
        selected_feature_count=args.selected_features,
        shot_intent=args.shot_intent,
    )
    classical_test = float(classical["fixed_test"]["balanced_accuracy"])
    shot_noise = _shot_noise_probe(
        base_features,
        panels[winner_name],
        data,
        selected_feature_count=args.selected_features,
        shots=args.shot_intent,
        replicates=args.shot_noise_replicates,
        seed=args.shot_noise_seed,
        classical_test_accuracy=classical_test,
    )
    quantum_cv = winner["training_cross_validation"]
    gates = {
        "mps_converged": mps_converged,
        "cv_mean_beats_classical": float(quantum_cv["cv_mean_balanced_accuracy"])
        > float(classical["cv_mean_balanced_accuracy"]),
        "cv_repeat_worst_not_below_classical": float(
            quantum_cv["repeat_mean_worst"]
        )
        >= float(
            min(
                np.mean(
                    [
                        fold["validation"]["balanced_accuracy"]
                        for fold in classical["folds"][
                            repeat * args.cv_splits : (repeat + 1) * args.cv_splits
                        ]
                    ]
                )
                for repeat in range(args.cv_repeats)
            )
        ),
        "fixed_test_beats_classical": float(final["fixed_test"]["balanced_accuracy"])
        > classical_test,
        "shot_noise_mean_beats_classical": float(
            shot_noise["mean_test_balanced_accuracy"]
        )
        > classical_test,
        "cross_seed_confirmation_available": False,
    }
    gates["ready_for_provider_validation"] = bool(all(gates.values()))

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_multiscale_readout_local_screen",
        "status": "complete_local_only",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "provider_calls": [],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "prior_architecture_screen": prior,
        "config": {
            "architecture": ARCHITECTURE,
            "qubits": 60,
            "seed": 11,
            "panels": list(PANEL_NAMES),
            "master_observable_count": len(mappings),
            "selected_features": args.selected_features,
            "shot_intent": args.shot_intent,
            "cv_splits": args.cv_splits,
            "cv_repeats": args.cv_repeats,
            "cv_seed": args.cv_seed,
            "bond_dimension": args.bond_dimension,
            "probe_bond_dimensions": list(args.probe_bond_dimensions),
        },
        "source": data.metadata,
        "measurement_protocol": {
            "global_bases": ["X", "Y", "Z"],
            "additional_hardware_circuits_relative_to_local_panel": 0,
            "all_observables_homogeneous_pauli_strings": True,
        },
        "mps": {
            "primary_full_feature_seconds": primary_seconds,
            "probe_indices": probe_indices,
            "probe_comparisons": probe_comparisons,
            "convergence_tolerance": args.convergence_tolerance,
            "converged": mps_converged,
        },
        "classical_same_split_reference": classical,
        "panels": panel_rows,
        "selection": {
            "uses_repeated_training_cv_only": True,
            "test_metrics_used_for_panel_selection": False,
            "winner": winner_name,
            "fixed_test_evaluated_for_winner_only": True,
        },
        "winner_final_evaluation": final,
        "winner_shot_noise_probe": shot_noise,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This local repeated-CV screen tests whether multiscale homogeneous "
            "Pauli readout improves the frozen grid circuit. It is not a provider "
            "result or proof of quantum advantage; cross-seed confirmation remains "
            "mandatory before any provider validation."
        ),
    }
    _atomic_write_json(args.output, report)
    return report


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--source-report",
        type=Path,
        default=architecture_screen.DEFAULT_SOURCE_REPORT,
    )
    parser.add_argument(
        "--reupload-report",
        type=Path,
        default=architecture_screen.DEFAULT_REUPLOAD_REPORT,
    )
    parser.add_argument("--prior-screen", type=Path, default=DEFAULT_PRIOR_SCREEN)
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
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--cv-repeats", type=int, default=DEFAULT_CV_REPEATS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument(
        "--shot-noise-replicates", type=int, default=DEFAULT_SHOT_NOISE_REPLICATES
    )
    parser.add_argument("--shot-noise-seed", type=int, default=DEFAULT_SHOT_NOISE_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_screen(args)
    print("PBMC68k q60 grid-d12 multiscale readout screen complete", flush=True)
    print(f"- panel winner: {report['selection']['winner']}", flush=True)
    print(
        "- fixed-test balanced accuracy: "
        f"{report['winner_final_evaluation']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- ready for provider validation: "
        f"{report['gates']['ready_for_provider_validation']}",
        flush=True,
    )
    print("- provider calls: 0", flush=True)
    print(f"- output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
