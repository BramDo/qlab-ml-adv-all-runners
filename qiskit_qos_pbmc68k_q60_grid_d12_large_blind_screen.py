#!/usr/bin/env python3
"""Large disjoint PBMC68k screen for the frozen q60 grid-d12 representation.

The circuit, hash encoding, and 24 multiscale observables are frozen from the
seed-11 pilot. A balanced 256+256 split is drawn only from source rows absent
from the historical q60 reports. The run is local-only and has no provider,
validation, submission, or retrieval path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_grid_d12_fireopal_validate as grid_validate
import qiskit_qos_pbmc68k_q60_multiscale_readout_screen as multiscale
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_utils as pbmc


SCHEMA_VERSION = "1.0"
ARCHITECTURE = "grid_mixer_d12"
ENCODING_HASH_SEED = 11
DEFAULT_SPLIT_SEED = 611256
DEFAULT_TRAIN_SAMPLES = 256
DEFAULT_TEST_SAMPLES = 256
DEFAULT_BOND_DIMENSION = 64
DEFAULT_PROBE_BOND_DIMENSION = 128
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_SHOTS = 128
DEFAULT_SHOT_REPLICATES = 1000
DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_grid_d12_large_blind_256x256.json"
)
ARTIFACT_DIR = Path("fire_opal_pbmc68k_q60_shallow")
HISTORICAL_REPORT_NAMES = (
    "pbmc68k_q60_seed11_train_only_tuning.json",
    "pbmc68k_q60_seed13_48x48_train_only_tuning.json",
    "pbmc68k_q60_seed17_48x48_train_only_tuning.json",
    "pbmc68k_q60_cross_seed_frozen_confirmation.json",
    "pbmc68k_q60_balanced_representation.json",
    "pbmc68k_q60_balanced_pair_scale_tuning.json",
    "pbmc68k_q60_balanced_entangler_topology_tuning.json",
    "pbmc68k_q60_balanced_reuploading_tuning.json",
    "pbmc68k_q60_scrambled_mixer_architecture_screen.json",
    "pbmc68k_q60_grid_d12_multiscale_readout_screen.json",
)

RunnerError = q60.RunnerError


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


def _index_values(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"train_indices", "test_indices"} and isinstance(item, list):
                found.update(int(index) for index in item)
            else:
                found.update(_index_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_index_values(item))
    return found


def collect_historical_exclusions(
    artifact_dir: Path = ARTIFACT_DIR,
) -> tuple[set[int], list[dict[str, Any]]]:
    excluded: set[int] = set()
    sources: list[dict[str, Any]] = []
    for name in HISTORICAL_REPORT_NAMES:
        path = artifact_dir / name
        if not path.is_file():
            raise RunnerError(f"Historical exclusion report is missing: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        indices = _index_values(report)
        excluded.update(indices)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": q60.q40_validate._sha256_file(path),
                "indices_found": len(indices),
            }
        )
    return excluded, sources


def balanced_disjoint_split(
    labels: np.ndarray,
    *,
    excluded_indices: set[int],
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise RunnerError("Large blind split requires exactly two classes")
    if train_samples % 2 or test_samples % 2:
        raise RunnerError("Train and test sizes must be even for exact balance")
    train_quota = train_samples // 2
    test_quota = test_samples // 2
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels), dtype=np.int64)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in classes:
        available = np.asarray(
            [
                int(index)
                for index in all_indices[labels == label]
                if int(index) not in excluded_indices
            ],
            dtype=np.int64,
        )
        if len(available) < train_quota + test_quota:
            raise RunnerError("Insufficient historically unseen rows for balanced split")
        shuffled = rng.permutation(available)
        train_parts.append(shuffled[:train_quota])
        test_parts.append(shuffled[train_quota : train_quota + test_quota])
    train = np.asarray(rng.permutation(np.concatenate(train_parts)), dtype=np.int64)
    test = np.asarray(rng.permutation(np.concatenate(test_parts)), dtype=np.int64)
    if (
        len(train) != train_samples
        or len(test) != test_samples
        or set(train) & set(test)
        or (set(train) | set(test)) & excluded_indices
    ):
        raise RunnerError("Large blind split violates size or disjointness")
    return train, test


def _wilson_interval(correct: int, samples: int, z: float = 1.95996398454) -> list[float]:
    proportion = correct / samples
    denominator = 1.0 + z**2 / samples
    center = (proportion + z**2 / (2.0 * samples)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / samples
            + z**2 / (4.0 * samples**2)
        )
        / denominator
    )
    return [float(center - margin), float(center + margin)]


def paired_test_statistics(
    labels: np.ndarray,
    quantum_scores: np.ndarray,
    classical_scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    quantum_correct = (np.where(quantum_scores >= 0.0, 1.0, -1.0) == labels)
    classical_correct = (np.where(classical_scores >= 0.0, 1.0, -1.0) == labels)
    quantum_only = int(np.sum(quantum_correct & ~classical_correct))
    classical_only = int(np.sum(~quantum_correct & classical_correct))
    discordant = quantum_only + classical_only
    mcnemar_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(quantum_only, classical_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    positive = np.flatnonzero(labels > 0.0)
    negative = np.flatnonzero(labels < 0.0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        deltas[replicate] = float(
            np.mean(quantum_correct[sampled]) - np.mean(classical_correct[sampled])
        )
    return {
        "paired_discordance": {
            "quantum_only_correct": quantum_only,
            "classical_only_correct": classical_only,
            "both_or_neither": int(len(labels) - discordant),
            "mcnemar_exact_two_sided_p": mcnemar_p,
        },
        "stratified_paired_bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "mean_quantum_minus_classical": float(np.mean(deltas)),
            "ci95_percentile": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "fraction_quantum_better": float(np.mean(deltas > 0.0)),
            "fraction_tied": float(np.mean(deltas == 0.0)),
        },
    }


def frozen_shot_noise_probe(
    features: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    shots: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    probabilities = np.clip(0.5 * (1.0 + features), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    train_stop = 1 + len(y_train)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = 2.0 * rng.binomial(shots, probabilities) / shots - 1.0
        head_train = q60._head_features(sampled[0], sampled[1:train_stop])
        head_test = q60._head_features(sampled[0], sampled[train_stop:])
        _, test_scores = q60._ridge_scores(head_train, head_test, y_train)
        values[replicate] = q60._balanced_accuracy(y_test, test_scores)
    return {
        "method": "independent_binomial_expectation_sampling_fixed_observables",
        "shots_per_basis_circuit": shots,
        "replicates": replicates,
        "seed": seed,
        "mean_test_balanced_accuracy": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "ci95_percentile": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "does_not_model_shared_basis_covariance_or_hardware_drift": True,
    }


def _load_large_data(
    args: argparse.Namespace,
) -> tuple[architecture.SeedData, dict[str, Any]]:
    source_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    config = source_report["config"]
    if int(config["seed"]) != ENCODING_HASH_SEED or int(config["qubits"]) != 60:
        raise RunnerError("Source report does not freeze the expected encoding")
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(config["positive_label"]),
        negative_label=str(config["negative_label"]),
    )
    excluded, exclusion_sources = collect_historical_exclusions(
        Path(args.artifact_dir)
    )
    train_indices, test_indices = balanced_disjoint_split(
        y_pair,
        excluded_indices=excluded,
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        seed=int(args.split_seed),
    )
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=60,
        hash_seed=ENCODING_HASH_SEED,
        value_mode=str(config["value_mode"]),
        max_active_genes=int(config["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=60,
        hash_seed=ENCODING_HASH_SEED,
        value_mode=str(config["value_mode"]),
        max_active_genes=int(config["max_active_genes"]),
    )
    data = architecture.SeedData(
        encoded_train=np.asarray(encoded_train, dtype=np.float64),
        encoded_test=np.asarray(encoded_test, dtype=np.float64),
        y_train=np.asarray(y_pair[train_indices], dtype=np.float64),
        y_test=np.asarray(y_pair[test_indices], dtype=np.float64),
        train_indices=train_indices,
        test_indices=test_indices,
        metadata={
            "configuration": {
                **config,
                "split_seed": int(args.split_seed),
                "encoding_hash_seed_frozen": ENCODING_HASH_SEED,
            },
            "source": {**source_meta, **pair_meta},
            "train_encoding_stats": train_stats,
            "test_encoding_stats": test_stats,
            "encoded_train_sha256": q60.q40_validate._array_sha256(encoded_train),
            "encoded_test_sha256": q60.q40_validate._array_sha256(encoded_test),
        },
    )
    split_audit = {
        "historical_reports": exclusion_sources,
        "historically_used_unique_indices": len(excluded),
        "train_indices": [int(value) for value in train_indices],
        "test_indices": [int(value) for value in test_indices],
        "train_test_overlap": 0,
        "historical_overlap": 0,
        "train_class_counts": {
            str(label): int(np.sum(data.y_train == label))
            for label in np.unique(data.y_train)
        },
        "test_class_counts": {
            str(label): int(np.sum(data.y_test == label))
            for label in np.unique(data.y_test)
        },
        "test_labels_used_for_fitting_or_selection": False,
    }
    return data, split_audit


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if min(
        args.train_samples,
        args.test_samples,
        args.bond_dimension,
        args.probe_bond_dimension,
        args.shots,
        args.shot_replicates,
        args.bootstrap_replicates,
    ) < 1:
        raise RunnerError("Sample, MPS, shot, and replicate counts must be positive")
    started = time.perf_counter()
    frozen_report, mappings, selected = grid_validate._load_frozen_screen(
        Path(args.screen_report)
    )
    selected_mappings = [mappings[index] for index in selected]
    measurement_bases = sorted(
        {q60.measurement_basis_for_mapping(mapping) for mapping in selected_mappings}
    )
    data, split_audit = _load_large_data(args)
    sketch, _ = q60.sketch_parameters(data.encoded_train, data.y_train)
    vectors = [sketch, *data.encoded_train, *data.encoded_test]
    circuits = [
        architecture.architecture_circuit(vector, ARCHITECTURE) for vector in vectors
    ]
    features, primary_seconds = architecture.simulate_feature_rows(
        circuits,
        selected_mappings,
        bond_dimension=int(args.bond_dimension),
        threshold=float(args.mps_threshold),
        progress_label="grid-d12-large-blind",
    )
    probe_indices = np.linspace(
        0, len(circuits) - 1, num=min(16, len(circuits)), dtype=np.int64
    )
    probe, probe_seconds = architecture.simulate_feature_rows(
        [circuits[int(index)] for index in probe_indices],
        selected_mappings,
        bond_dimension=int(args.probe_bond_dimension),
        threshold=float(args.mps_threshold),
        progress_label="grid-d12-large-blind-probe",
    )
    difference = np.abs(probe - features[probe_indices])
    mps_converged = bool(
        np.max(difference) <= float(args.convergence_tolerance)
    )
    train_stop = 1 + len(data.encoded_train)
    head_train = q60._head_features(features[0], features[1:train_stop])
    head_test = q60._head_features(features[0], features[train_stop:])
    quantum_train_scores, quantum_test_scores = q60._ridge_scores(
        head_train, head_test, data.y_train
    )
    classical_train_scores, classical_test_scores = q60._ridge_scores(
        data.encoded_train, data.encoded_test, data.y_train
    )
    rbf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    rbf.fit(data.encoded_train, data.y_train)
    rbf_train_scores = np.asarray(rbf.decision_function(data.encoded_train))
    rbf_test_scores = np.asarray(rbf.decision_function(data.encoded_test))
    quantum_train = architecture._balanced_metrics(
        data.y_train, quantum_train_scores
    )
    quantum_test = architecture._balanced_metrics(data.y_test, quantum_test_scores)
    classical_train = architecture._balanced_metrics(
        data.y_train, classical_train_scores
    )
    classical_test = architecture._balanced_metrics(
        data.y_test, classical_test_scores
    )
    rbf_train = architecture._balanced_metrics(data.y_train, rbf_train_scores)
    rbf_test = architecture._balanced_metrics(data.y_test, rbf_test_scores)
    paired_linear = paired_test_statistics(
        data.y_test,
        quantum_test_scores,
        classical_test_scores,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.split_seed) + 1000,
    )
    paired_rbf = paired_test_statistics(
        data.y_test,
        quantum_test_scores,
        rbf_test_scores,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.split_seed) + 2000,
    )
    shot_noise = frozen_shot_noise_probe(
        features,
        data.y_train,
        data.y_test,
        shots=int(args.shots),
        replicates=int(args.shot_replicates),
        seed=int(args.split_seed) + 3000,
    )
    for metrics in (quantum_train, quantum_test, classical_train, classical_test, rbf_train, rbf_test):
        metrics["wilson_95"] = _wilson_interval(
            int(metrics["correct"]), int(metrics["samples"])
        )
    measured_circuits = len(circuits) * len(measurement_bases)
    batch_limit = q60.FIRE_OPAL_MAX_BATCH
    full_batches, remainder = divmod(measured_circuits, batch_limit)
    batch_sizes = [batch_limit] * full_batches + ([remainder] if remainder else [])
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_large_disjoint_blind_screen",
        "status": "complete_local_only",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "provider_calls": [],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "config": {
            "dataset": "PBMC68k",
            "architecture": ARCHITECTURE,
            "panel": "multiscale_pairs",
            "qubits": 60,
            "encoding_hash_seed": ENCODING_HASH_SEED,
            "split_seed": int(args.split_seed),
            "train_samples": len(data.encoded_train),
            "test_samples": len(data.encoded_test),
            "selected_observables": len(selected),
            "measurement_bases_needed": measurement_bases,
            "bond_dimension": int(args.bond_dimension),
            "probe_bond_dimension": int(args.probe_bond_dimension),
            "shots_for_noise_projection": int(args.shots),
        },
        "frozen_source": {
            "screen_report": str(Path(args.screen_report).resolve()),
            "screen_report_sha256": q60.q40_validate._sha256_file(
                Path(args.screen_report)
            ),
            "selection_winner": frozen_report["selection"]["winner"],
            "selected_master_indices": [int(value) for value in selected],
            "configuration_frozen_before_large_split": True,
        },
        "source": data.metadata,
        "split_audit": split_audit,
        "mps": {
            "primary_seconds": primary_seconds,
            "probe_seconds": probe_seconds,
            "probe_indices": [int(value) for value in probe_indices],
            "max_abs_chi128_minus_chi64": float(np.max(difference)),
            "mean_abs_chi128_minus_chi64": float(np.mean(difference)),
            "convergence_tolerance": float(args.convergence_tolerance),
            "converged": mps_converged,
        },
        "ideal_quantum_frozen_representation": {
            "model": "ridge on 24 frozen quantum observables, sketch products, and cosine",
            "train": quantum_train,
            "fixed_test": quantum_test,
        },
        "classical_same_encoding_linear": {
            "model": "standardized ridge on the same 60 hashed bins",
            "train": classical_train,
            "fixed_test": classical_test,
        },
        "classical_same_encoding_rbf": {
            "model": "fixed C=1 RBF SVC on standardized same 60 hashed bins",
            "train": rbf_train,
            "fixed_test": rbf_test,
            "hyperparameters_selected_without_test": True,
        },
        "paired_quantum_vs_linear": paired_linear,
        "paired_quantum_vs_rbf": paired_rbf,
        "shot_noise_projection": shot_noise,
        "hardware_scope_if_separately_authorized": {
            "logical_base_circuits": len(circuits),
            "measurement_bases": measurement_bases,
            "measured_circuits": measured_circuits,
            "fire_opal_batch_limit": batch_limit,
            "required_batches": len(batch_sizes),
            "batch_sizes": batch_sizes,
            "shots_per_circuit": int(args.shots),
            "total_requested_shots": measured_circuits * int(args.shots),
            "submission_authorized": False,
        },
        "gates": {
            "historically_disjoint": True,
            "test_unused_for_fitting_or_selection": True,
            "mps_converged": mps_converged,
            "ideal_quantum_beats_linear_classical": float(
                quantum_test["balanced_accuracy"]
            )
            > float(classical_test["balanced_accuracy"]),
            "ideal_quantum_beats_rbf_classical": float(
                quantum_test["balanced_accuracy"]
            )
            > float(rbf_test["balanced_accuracy"]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This larger historically disjoint local screen estimates accuracy and "
            "uncertainty for a frozen 60-qubit representation. It is not a hardware "
            "result or a proof of quantum advantage; any later hardware execution "
            "requires separate authorization and multiple Fire Opal batches."
        ),
    }
    _atomic_write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=ARTIFACT_DIR / "pbmc68k_q60_seed11_train_only_tuning.json",
    )
    parser.add_argument(
        "--screen-report", type=Path, default=multiscale.DEFAULT_OUTPUT
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES
    )
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument(
        "--bond-dimension", type=int, default=DEFAULT_BOND_DIMENSION
    )
    parser.add_argument(
        "--probe-bond-dimension", type=int, default=DEFAULT_PROBE_BOND_DIMENSION
    )
    parser.add_argument("--mps-threshold", type=float, default=DEFAULT_MPS_THRESHOLD)
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE,
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--shot-replicates", type=int, default=DEFAULT_SHOT_REPLICATES
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_screen(args)
    print("PBMC68k q60 grid-d12 large blind screen complete", flush=True)
    print(
        "- ideal quantum fixed test: "
        f"{report['ideal_quantum_frozen_representation']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- classical ridge fixed test: "
        f"{report['classical_same_encoding_linear']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- classical RBF fixed test: "
        f"{report['classical_same_encoding_rbf']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- separate hardware batches if later authorized: "
        f"{report['hardware_scope_if_separately_authorized']['required_batches']}",
        flush=True,
    )
    print("- provider calls: 0", flush=True)
    print(f"- output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
