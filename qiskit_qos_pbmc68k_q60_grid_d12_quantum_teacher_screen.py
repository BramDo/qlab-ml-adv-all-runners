#!/usr/bin/env python3
"""Local quantum-teacher screen on real PBMC68k cell inputs.

This is a deliberately semi-synthetic proof-of-principle task.  A frozen
60-qubit grid-d12 feature map defines labels from broad four- and eight-qubit
correlations.  The task therefore gives the quantum representation a genuine
signal-alignment opportunity without presenting the result as natural-label
biology or an unconditional quantum-advantage result.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_grid_d12_large_blind_screen as large
import qiskit_qos_pbmc68k_q60_multiscale_readout_screen as multiscale
import qiskit_qos_pbmc68k_q60_scrambled_mixer_screen as architecture
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_utils as pbmc


SCHEMA_VERSION = "1.0"
ARCHITECTURE = "grid_mixer_d12"
ENCODING_HASH_SEED = 11
DEFAULT_POOL_SEED = 112060
DEFAULT_CONSTRUCTION_SAMPLES = 256
DEFAULT_CANDIDATE_SAMPLES = 1024
DEFAULT_TRAIN_SAMPLES = 256
DEFAULT_TEST_SAMPLES = 256
DEFAULT_TEACHER_OBSERVABLES = 16
DEFAULT_BOND_DIMENSION = 64
DEFAULT_PROBE_BOND_DIMENSION = 128
DEFAULT_MPS_THRESHOLD = 1e-10
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
DEFAULT_SHOTS = 128
DEFAULT_SHOT_REPLICATES = 1000
ARTIFACT_DIR = Path("fire_opal_pbmc68k_q60_shallow")
DEFAULT_CONTROL_REPORT = (
    ARTIFACT_DIR / "pbmc68k_q60_grid_d12_large_blind_256x256.json"
)
DEFAULT_OUTPUT = (
    ARTIFACT_DIR / "pbmc68k_q60_grid_d12_quantum_teacher_256x256.json"
)

RunnerError = q60.RunnerError


def broad_yz_observables() -> tuple[list[dict[int, str]], list[int]]:
    mappings, _ = multiscale.build_multiscale_panels()
    indices = [
        index
        for index, mapping in enumerate(mappings)
        if len(mapping) in {4, 8}
        and len(set(mapping.values())) == 1
        and next(iter(mapping.values())) in {"Y", "Z"}
    ]
    if len(indices) != 94:
        raise RunnerError("Frozen broad Y/Z observable panel changed")
    return mappings, indices


def select_teacher_observables(
    construction_features: np.ndarray,
    master_indices: Sequence[int],
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(construction_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(master_indices):
        raise RunnerError("Construction feature panel has the wrong shape")
    if count < 2 or count > features.shape[1]:
        raise RunnerError("Teacher-observable count is outside the panel")
    standard_deviations = np.std(features, axis=0)
    ranked = np.lexsort((np.asarray(master_indices), -standard_deviations))
    selected_columns = np.sort(ranked[:count])
    selected = features[:, selected_columns]
    center = np.mean(selected, axis=0)
    _, _, right = np.linalg.svd(selected - center, full_matrices=False)
    weights = np.asarray(right[0], dtype=np.float64)
    pivot = int(np.argmax(np.abs(weights)))
    if weights[pivot] < 0.0:
        weights *= -1.0
    weights /= np.linalg.norm(weights)
    return selected_columns, center, weights


def construct_balanced_margin_task(
    teacher_scores: np.ndarray,
    candidate_source_indices: np.ndarray,
    *,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    scores = np.asarray(teacher_scores, dtype=np.float64)
    source_indices = np.asarray(candidate_source_indices, dtype=np.int64)
    if len(scores) != len(source_indices) or not np.all(np.isfinite(scores)):
        raise RunnerError("Teacher scores and candidate indices are inconsistent")
    if train_samples % 2 or test_samples % 2:
        raise RunnerError("Teacher train and test sizes must be even")
    per_class = (train_samples + test_samples) // 2
    if 2 * per_class > len(scores):
        raise RunnerError("Teacher candidate pool is too small")
    threshold = float(np.median(scores))
    order = np.argsort(scores, kind="stable")
    low = order[:per_class]
    high = order[-per_class:]
    if np.max(scores[low]) >= np.min(scores[high]):
        raise RunnerError("Teacher margin selection is not separated")
    rng = np.random.default_rng(seed)
    low = rng.permutation(low)
    high = rng.permutation(high)
    train_per_class = train_samples // 2
    train_positions = np.concatenate(
        [low[:train_per_class], high[:train_per_class]]
    )
    test_positions = np.concatenate(
        [low[train_per_class:], high[train_per_class:]]
    )
    train_positions = np.asarray(rng.permutation(train_positions), dtype=np.int64)
    test_positions = np.asarray(rng.permutation(test_positions), dtype=np.int64)
    labels = np.where(scores >= threshold, 1.0, -1.0)
    selected_margins = np.abs(scores[np.concatenate([train_positions, test_positions])] - threshold)
    audit = {
        "teacher_threshold": threshold,
        "discarded_middle_candidates": int(len(scores) - train_samples - test_samples),
        "selected_margin_minimum": float(np.min(selected_margins)),
        "selected_margin_median": float(np.median(selected_margins)),
        "selected_margin_maximum": float(np.max(selected_margins)),
        "train_source_indices": [int(value) for value in source_indices[train_positions]],
        "test_source_indices": [int(value) for value in source_indices[test_positions]],
        "train_test_overlap": int(
            len(set(source_indices[train_positions]) & set(source_indices[test_positions]))
        ),
    }
    return train_positions, test_positions, labels, threshold, audit


def shot_noise_probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    shots: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    features = np.vstack([train_features, test_features])
    probabilities = np.clip(0.5 * (1.0 + features), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = 2.0 * rng.binomial(shots, probabilities) / shots - 1.0
        sampled_train = sampled[: len(train_features)]
        sampled_test = sampled[len(train_features) :]
        _, scores = q60._ridge_scores(sampled_train, sampled_test, y_train)
        values[replicate] = q60._balanced_accuracy(y_test, scores)
    return {
        "method": "independent_binomial_sampling_of_each_frozen_observable",
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


def _load_encoded_pools(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    config = source_report["config"]
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, _, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(config["positive_label"]),
        negative_label=str(config["negative_label"]),
    )
    excluded, exclusion_sources = large.collect_historical_exclusions(
        Path(args.artifact_dir)
    )
    control_path = Path(args.control_report)
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control_indices = large._index_values(control)
    excluded.update(control_indices)
    exclusion_sources.append(
        {
            "path": str(control_path.resolve()),
            "sha256": q60.q40_validate._sha256_file(control_path),
            "indices_found": len(control_indices),
        }
    )
    available = np.asarray(
        [index for index in range(int(x_pair.shape[0])) if index not in excluded],
        dtype=np.int64,
    )
    needed = int(args.construction_samples + args.candidate_samples)
    if len(available) < needed:
        raise RunnerError("Insufficient historically unseen PBMC rows")
    rng = np.random.default_rng(int(args.pool_seed))
    chosen = np.asarray(rng.permutation(available)[:needed], dtype=np.int64)
    construction_indices = chosen[: int(args.construction_samples)]
    candidate_indices = chosen[int(args.construction_samples) :]
    encoded, encoding_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[chosen],
        feature_dim=60,
        hash_seed=ENCODING_HASH_SEED,
        value_mode=str(config["value_mode"]),
        max_active_genes=int(config["max_active_genes"]),
    )
    encoded = np.asarray(encoded, dtype=np.float64)
    construction = encoded[: len(construction_indices)]
    candidates = encoded[len(construction_indices) :]
    metadata = {
        "configuration": config,
        "source": {**source_meta, **pair_meta},
        "encoding_stats": encoding_stats,
        "historically_used_unique_indices": len(excluded),
        "exclusion_sources": exclusion_sources,
        "construction_source_indices": [int(value) for value in construction_indices],
        "candidate_source_indices": [int(value) for value in candidate_indices],
        "construction_candidate_overlap": 0,
        "encoded_pool_sha256": q60.q40_validate._array_sha256(encoded),
    }
    return construction, candidates, candidate_indices, metadata


def _metrics_with_interval(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    metrics = architecture._balanced_metrics(labels, scores)
    metrics["wilson_95"] = large._wilson_interval(
        int(metrics["correct"]), int(metrics["samples"])
    )
    return metrics


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if min(
        args.construction_samples,
        args.candidate_samples,
        args.train_samples,
        args.test_samples,
        args.teacher_observables,
        args.bond_dimension,
        args.probe_bond_dimension,
        args.shots,
        args.shot_replicates,
    ) < 1:
        raise RunnerError("All sample, MPS, shot, and observable counts must be positive")
    started = time.perf_counter()
    construction, candidates, candidate_indices, source = _load_encoded_pools(args)
    mappings, broad_indices = broad_yz_observables()
    broad_mappings = [mappings[index] for index in broad_indices]
    construction_circuits = [
        architecture.architecture_circuit(vector, ARCHITECTURE)
        for vector in construction
    ]
    construction_features, construction_seconds = architecture.simulate_feature_rows(
        construction_circuits,
        broad_mappings,
        bond_dimension=int(args.bond_dimension),
        threshold=float(args.mps_threshold),
        progress_label="quantum-teacher-construction",
    )
    selected_columns, teacher_center, teacher_weights = select_teacher_observables(
        construction_features,
        broad_indices,
        count=int(args.teacher_observables),
    )
    selected_master_indices = [broad_indices[int(column)] for column in selected_columns]
    selected_mappings = [mappings[index] for index in selected_master_indices]
    candidate_circuits = [
        architecture.architecture_circuit(vector, ARCHITECTURE) for vector in candidates
    ]
    candidate_features, candidate_seconds = architecture.simulate_feature_rows(
        candidate_circuits,
        selected_mappings,
        bond_dimension=int(args.bond_dimension),
        threshold=float(args.mps_threshold),
        progress_label="quantum-teacher-candidates",
    )
    teacher_scores = (candidate_features - teacher_center) @ teacher_weights
    train_positions, test_positions, all_labels, threshold, task_audit = (
        construct_balanced_margin_task(
            teacher_scores,
            candidate_indices,
            train_samples=int(args.train_samples),
            test_samples=int(args.test_samples),
            seed=int(args.pool_seed) + 1,
        )
    )
    quantum_train = candidate_features[train_positions]
    quantum_test = candidate_features[test_positions]
    classical_train = candidates[train_positions]
    classical_test = candidates[test_positions]
    y_train = all_labels[train_positions]
    y_test = all_labels[test_positions]
    quantum_train_scores, quantum_test_scores = q60._ridge_scores(
        quantum_train, quantum_test, y_train
    )
    linear_train_scores, linear_test_scores = q60._ridge_scores(
        classical_train, classical_test, y_train
    )
    rbf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    rbf.fit(classical_train, y_train)
    rbf_train_scores = np.asarray(rbf.decision_function(classical_train))
    rbf_test_scores = np.asarray(rbf.decision_function(classical_test))
    quantum_train_metrics = _metrics_with_interval(y_train, quantum_train_scores)
    quantum_test_metrics = _metrics_with_interval(y_test, quantum_test_scores)
    linear_train_metrics = _metrics_with_interval(y_train, linear_train_scores)
    linear_test_metrics = _metrics_with_interval(y_test, linear_test_scores)
    rbf_train_metrics = _metrics_with_interval(y_train, rbf_train_scores)
    rbf_test_metrics = _metrics_with_interval(y_test, rbf_test_scores)
    shot_noise = shot_noise_probe(
        quantum_train,
        quantum_test,
        y_train,
        y_test,
        shots=int(args.shots),
        replicates=int(args.shot_replicates),
        seed=int(args.pool_seed) + 2,
    )
    probe_positions = np.linspace(
        0, len(candidate_circuits) - 1, num=min(16, len(candidate_circuits)), dtype=np.int64
    )
    probe, probe_seconds = architecture.simulate_feature_rows(
        [candidate_circuits[int(index)] for index in probe_positions],
        selected_mappings,
        bond_dimension=int(args.probe_bond_dimension),
        threshold=float(args.mps_threshold),
        progress_label="quantum-teacher-probe",
    )
    difference = np.abs(probe - candidate_features[probe_positions])
    mps_converged = bool(np.max(difference) <= float(args.convergence_tolerance))
    paired_linear = large.paired_test_statistics(
        y_test,
        quantum_test_scores,
        linear_test_scores,
        replicates=10000,
        seed=int(args.pool_seed) + 3,
    )
    paired_rbf = large.paired_test_statistics(
        y_test,
        quantum_test_scores,
        rbf_test_scores,
        replicates=10000,
        seed=int(args.pool_seed) + 4,
    )
    measurement_bases = sorted(
        {q60.measurement_basis_for_mapping(mapping) for mapping in selected_mappings}
    )
    logical_circuits = int(args.train_samples + args.test_samples)
    measured_circuits = logical_circuits * len(measurement_bases)
    full_batches, remainder = divmod(measured_circuits, q60.FIRE_OPAL_MAX_BATCH)
    batch_sizes = [q60.FIRE_OPAL_MAX_BATCH] * full_batches
    if remainder:
        batch_sizes.append(remainder)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_grid_d12_quantum_teacher_local_screen",
        "status": "complete_local_only",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "provider_calls": [],
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "config": {
            "dataset": "PBMC68k real cell inputs",
            "task_labels": "semi-synthetic frozen quantum teacher",
            "architecture": ARCHITECTURE,
            "qubits": 60,
            "encoding_hash_seed": ENCODING_HASH_SEED,
            "pool_seed": int(args.pool_seed),
            "construction_samples": len(construction),
            "candidate_samples": len(candidates),
            "train_samples": len(train_positions),
            "test_samples": len(test_positions),
            "teacher_observables": len(selected_master_indices),
            "shots_for_noise_projection": int(args.shots),
        },
        "source": source,
        "teacher_construction": {
            "candidate_panel_master_indices": broad_indices,
            "candidate_panel_support_sizes": [4, 8],
            "candidate_panel_bases": ["Y", "Z"],
            "selection_rule": "largest raw standard deviation on disjoint construction pool",
            "selected_master_indices": selected_master_indices,
            "selected_mappings": [
                {str(qubit): pauli for qubit, pauli in mapping.items()}
                for mapping in selected_mappings
            ],
            "selected_support_sizes": [len(mapping) for mapping in selected_mappings],
            "teacher_center": [float(value) for value in teacher_center],
            "teacher_pc1_weights": [float(value) for value in teacher_weights],
            "teacher_threshold": threshold,
            "construction_pool_disjoint_from_benchmark": True,
            "labels_are_not_natural_biological_labels": True,
        },
        "task_audit": {
            **task_audit,
            "train_class_counts": {
                "-1": int(np.sum(y_train < 0.0)),
                "+1": int(np.sum(y_train > 0.0)),
            },
            "test_class_counts": {
                "-1": int(np.sum(y_test < 0.0)),
                "+1": int(np.sum(y_test > 0.0)),
            },
            "test_labels_used_for_model_fitting": False,
        },
        "mps": {
            "construction_seconds": construction_seconds,
            "candidate_seconds": candidate_seconds,
            "probe_seconds": probe_seconds,
            "primary_bond_dimension": int(args.bond_dimension),
            "probe_bond_dimension": int(args.probe_bond_dimension),
            "max_abs_probe_difference": float(np.max(difference)),
            "mean_abs_probe_difference": float(np.mean(difference)),
            "convergence_tolerance": float(args.convergence_tolerance),
            "converged": mps_converged,
        },
        "ideal_quantum_teacher_features": {
            "model": "standardized ridge on 16 frozen broad quantum observables",
            "train": quantum_train_metrics,
            "fixed_test": quantum_test_metrics,
        },
        "classical_same_encoding_linear": {
            "model": "standardized ridge on the same 60 hashed input bins",
            "train": linear_train_metrics,
            "fixed_test": linear_test_metrics,
        },
        "classical_same_encoding_rbf": {
            "model": "fixed C=1 RBF SVC on the same standardized 60 hashed input bins",
            "train": rbf_train_metrics,
            "fixed_test": rbf_test_metrics,
            "hyperparameters_selected_without_test": True,
        },
        "paired_quantum_vs_linear": paired_linear,
        "paired_quantum_vs_rbf": paired_rbf,
        "shot_noise_projection": shot_noise,
        "hardware_scope_if_separately_authorized": {
            "logical_circuits": logical_circuits,
            "measurement_bases": measurement_bases,
            "measured_circuits": measured_circuits,
            "fire_opal_batch_limit": q60.FIRE_OPAL_MAX_BATCH,
            "required_batches": len(batch_sizes),
            "batch_sizes": batch_sizes,
            "shots_per_circuit": int(args.shots),
            "total_requested_shots": measured_circuits * int(args.shots),
            "submission_authorized": False,
        },
        "gates": {
            "real_inputs_with_transparent_semi_synthetic_labels": True,
            "construction_pool_disjoint": True,
            "balanced_historically_unseen_test": True,
            "mps_converged": mps_converged,
            "ideal_quantum_beats_linear": float(quantum_test_metrics["balanced_accuracy"])
            > float(linear_test_metrics["balanced_accuracy"]),
            "ideal_quantum_beats_rbf": float(quantum_test_metrics["balanced_accuracy"])
            > float(rbf_test_metrics["balanced_accuracy"]),
            "projected_128_shot_quantum_beats_both_classical": float(
                shot_noise["mean_test_balanced_accuracy"]
            )
            > max(
                float(linear_test_metrics["balanced_accuracy"]),
                float(rbf_test_metrics["balanced_accuracy"]),
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "This is a semi-synthetic quantum-teacher task on real PBMC68k cell inputs. "
            "It tests whether hardware can preserve a deliberately quantum-aligned broad-"
            "correlation representation. It is not natural-label biology, a faithful full "
            "QOS implementation, or proof of unconditional computational quantum advantage; "
            "the tested depth-12 circuits also remain accessible to bounded MPS checks."
        ),
    }
    large._atomic_write_json(args.output, report)
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
    parser.add_argument("--control-report", type=Path, default=DEFAULT_CONTROL_REPORT)
    parser.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED)
    parser.add_argument(
        "--construction-samples", type=int, default=DEFAULT_CONSTRUCTION_SAMPLES
    )
    parser.add_argument("--candidate-samples", type=int, default=DEFAULT_CANDIDATE_SAMPLES)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument(
        "--teacher-observables", type=int, default=DEFAULT_TEACHER_OBSERVABLES
    )
    parser.add_argument("--bond-dimension", type=int, default=DEFAULT_BOND_DIMENSION)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_screen(args)
    print("PBMC68k q60 grid-d12 quantum-teacher screen complete", flush=True)
    print(
        "- ideal quantum test: "
        f"{report['ideal_quantum_teacher_features']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- classical ridge test: "
        f"{report['classical_same_encoding_linear']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- classical RBF test: "
        f"{report['classical_same_encoding_rbf']['fixed_test']['balanced_accuracy']:.6f}",
        flush=True,
    )
    print(
        "- projected 128-shot quantum test: "
        f"{report['shot_noise_projection']['mean_test_balanced_accuracy']:.6f}",
        flush=True,
    )
    print("- provider calls: 0", flush=True)
    print(f"- output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
