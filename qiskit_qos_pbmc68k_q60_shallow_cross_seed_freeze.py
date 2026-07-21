#!/usr/bin/env python3
"""Freeze one q60 configuration across training-only runs and confirm it once.

The configuration is selected exclusively from the per-candidate training-CV
leaderboards of earlier runs.  Their final/test evaluations are deliberately
discarded before ranking.  The frozen configuration is then evaluated once on
a new PBMC68k split whose cell indices do not occur in any source run.

This module is local-only.  It has no provider or hardware execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_REPORTS = (
    Path(
        "fire_opal_pbmc68k_q60_shallow/"
        "pbmc68k_q60_seed11_train_only_tuning.json"
    ),
    Path(
        "fire_opal_pbmc68k_q60_shallow/"
        "pbmc68k_q60_seed13_48x48_train_only_tuning.json"
    ),
    Path(
        "fire_opal_pbmc68k_q60_shallow/"
        "pbmc68k_q60_seed17_48x48_train_only_tuning.json"
    ),
)
DEFAULT_CONFIRMATION_SEED_START = 19
DEFAULT_CONFIRMATION_SAMPLES = 48
DEFAULT_SEED_SCAN = 100

RunnerError = q60.RunnerError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["single_scale"]),
        float(row["phase_scale"]),
        float(row["pair_scale"]),
        int(row["selected_feature_count"]),
    )


def _validate_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    folds = [dict(fold) for fold in row.get("folds", [])]
    if len(folds) < 2:
        raise RunnerError("Every training-CV candidate needs at least two folds")
    fold_scores = np.asarray(
        [float(fold["balanced_accuracy"]) for fold in folds], dtype=np.float64
    )
    if not np.all(np.isfinite(fold_scores)):
        raise RunnerError("Training-CV leaderboard contains a non-finite score")
    expected = {
        "cv_mean_balanced_accuracy": float(np.mean(fold_scores)),
        "cv_worst_balanced_accuracy": float(np.min(fold_scores)),
        "cv_std_balanced_accuracy": float(np.std(fold_scores)),
    }
    for field, value in expected.items():
        if not np.isclose(float(row[field]), value, atol=1e-12, rtol=0.0):
            raise RunnerError(f"Stored {field} is inconsistent with fold scores")
    return {
        "single_scale": float(row["single_scale"]),
        "phase_scale": float(row["phase_scale"]),
        "pair_scale": float(row["pair_scale"]),
        "selected_feature_count": int(row["selected_feature_count"]),
        **expected,
        "fold_balanced_accuracies": [float(value) for value in fold_scores],
    }


def extract_training_only_run(
    report: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Discard all source-test results and return only training-CV evidence."""

    tuning = report.get("train_only_tuning")
    if not isinstance(tuning, Mapping):
        raise RunnerError(f"Missing train_only_tuning in {source_path}")
    if tuning.get("selection_scope") != "training_split_only":
        raise RunnerError(f"Non-training-only selection scope in {source_path}")
    if tuning.get("test_inputs_seen") is not False:
        raise RunnerError(f"Source tuning saw test inputs in {source_path}")
    if tuning.get("test_labels_seen") is not False:
        raise RunnerError(f"Source tuning saw test labels in {source_path}")
    leaderboard = [_validate_candidate(row) for row in tuning.get("leaderboard", [])]
    if not leaderboard:
        raise RunnerError(f"Empty training-CV leaderboard in {source_path}")
    if int(tuning.get("candidate_configurations", -1)) != len(leaderboard):
        raise RunnerError(f"Candidate count mismatch in {source_path}")
    keys = [_configuration_key(row) for row in leaderboard]
    if len(set(keys)) != len(keys):
        raise RunnerError(f"Duplicate configuration in {source_path}")
    config = report.get("config")
    split = report.get("split")
    if not isinstance(config, Mapping) or not isinstance(split, Mapping):
        raise RunnerError(f"Missing configuration or split metadata in {source_path}")
    return {
        "seed": int(config["seed"]),
        "training_samples": int(tuning["training_samples"]),
        "cv_folds": int(tuning["cv_folds"]),
        "encoded_train_sha256": str(split["encoded_train_sha256"]),
        "source_path": source_path,
        "source_sha256": source_sha256,
        "leaderboard": leaderboard,
    }


def _aggregate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer cross-seed mean, robustness, stability, then smaller parameters."""

    return (
        -round(float(row["aggregate_cv_mean_balanced_accuracy"]), 12),
        -round(float(row["worst_seed_cv_mean_balanced_accuracy"]), 12),
        -round(float(row["worst_fold_balanced_accuracy"]), 12),
        round(float(row["seed_cv_mean_std"]), 12),
        int(row["selected_feature_count"]),
        float(row["single_scale"]),
        float(row["phase_scale"]),
        float(row["pair_scale"]),
    )


def select_cross_seed_configuration(
    training_only_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one configuration from training-CV leaderboards only."""

    if len(training_only_runs) < 2:
        raise RunnerError("Cross-seed freezing needs at least two training-only runs")
    seeds = [int(run["seed"]) for run in training_only_runs]
    if len(set(seeds)) != len(seeds):
        raise RunnerError("Cross-seed freezing needs unique seeds")

    indexed_runs: list[dict[tuple[float, float, float, int], Mapping[str, Any]]] = []
    for run in training_only_runs:
        indexed_runs.append(
            {_configuration_key(row): row for row in run["leaderboard"]}
        )
    reference_keys = set(indexed_runs[0])
    if any(set(indexed) != reference_keys for indexed in indexed_runs[1:]):
        raise RunnerError("Training-only runs do not contain the same candidate grid")

    leaderboard: list[dict[str, Any]] = []
    for key in sorted(reference_keys):
        candidates = [indexed[key] for indexed in indexed_runs]
        seed_means = np.asarray(
            [row["cv_mean_balanced_accuracy"] for row in candidates],
            dtype=np.float64,
        )
        all_fold_scores = np.asarray(
            [
                score
                for row in candidates
                for score in row["fold_balanced_accuracies"]
            ],
            dtype=np.float64,
        )
        leaderboard.append(
            {
                "single_scale": key[0],
                "phase_scale": key[1],
                "pair_scale": key[2],
                "selected_feature_count": key[3],
                "aggregate_cv_mean_balanced_accuracy": float(np.mean(seed_means)),
                "worst_seed_cv_mean_balanced_accuracy": float(np.min(seed_means)),
                "worst_fold_balanced_accuracy": float(np.min(all_fold_scores)),
                "seed_cv_mean_std": float(np.std(seed_means)),
                "source_seed_cv": [
                    {
                        "seed": int(run["seed"]),
                        "cv_mean_balanced_accuracy": float(row["cv_mean_balanced_accuracy"]),
                        "cv_worst_balanced_accuracy": float(
                            row["cv_worst_balanced_accuracy"]
                        ),
                        "fold_balanced_accuracies": list(
                            row["fold_balanced_accuracies"]
                        ),
                    }
                    for run, row in zip(training_only_runs, candidates, strict=True)
                ],
            }
        )
    ranked = sorted(leaderboard, key=_aggregate_rank_key)
    return {
        "selection_scope": "source_training_cv_only",
        "source_test_inputs_used": False,
        "source_test_labels_used": False,
        "confirmation_inputs_seen": False,
        "confirmation_labels_seen": False,
        "source_seeds": seeds,
        "candidate_configurations": len(ranked),
        "ranking_rule": (
            "maximize equal-seed mean CV, then worst seed mean, then worst fold; "
            "minimize between-seed standard deviation, feature count, and scales"
        ),
        "chosen": dict(ranked[0]),
        "leaderboard": ranked,
    }


def _shared_source_configuration(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "dataset",
        "positive_label",
        "negative_label",
        "qubits",
        "value_mode",
        "max_active_genes",
        "shot_intent_for_feature_ranking",
        "single_scales",
        "phase_scales",
        "pair_scales",
        "selected_counts",
        "cv_folds",
    )
    reference = {field: reports[0]["config"][field] for field in fields}
    for report in reports[1:]:
        candidate = {field: report["config"][field] for field in fields}
        if candidate != reference:
            raise RunnerError("Source tuning reports do not share one experiment grid")
    return reference


def _excluded_source_indices(reports: Sequence[Mapping[str, Any]]) -> set[int]:
    excluded: set[int] = set()
    for report in reports:
        split = report.get("split", {})
        excluded.update(int(value) for value in split.get("train_indices", []))
        excluded.update(int(value) for value in split.get("test_indices", []))
    return excluded


def load_fresh_confirmation_data(
    args: argparse.Namespace,
    *,
    excluded_indices: set[int],
    shared_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Find and encode the first split disjoint from every source split."""

    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(shared_config["positive_label"]),
        negative_label=str(shared_config["negative_label"]),
    )
    chosen: tuple[int, np.ndarray, np.ndarray] | None = None
    for seed in range(
        int(args.confirmation_seed_start),
        int(args.confirmation_seed_start) + int(args.seed_scan),
    ):
        train_indices, test_indices = genomics_runner.benchmark_indices(
            x_pair.shape[0],
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            labels=y_pair,
        )
        proposed = set(int(value) for value in train_indices) | set(
            int(value) for value in test_indices
        )
        if proposed.isdisjoint(excluded_indices):
            chosen = (seed, train_indices, test_indices)
            break
    if chosen is None:
        raise RunnerError(
            "No fully disjoint confirmation split found in the requested seed scan"
        )

    seed, train_indices, test_indices = chosen
    y_train = y_pair[train_indices].astype(np.float64)
    y_test = y_pair[test_indices].astype(np.float64)
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=int(shared_config["qubits"]),
        hash_seed=seed,
        value_mode=str(shared_config["value_mode"]),
        max_active_genes=int(shared_config["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=int(shared_config["qubits"]),
        hash_seed=seed,
        value_mode=str(shared_config["value_mode"]),
        max_active_genes=int(shared_config["max_active_genes"]),
    )
    return {
        "seed": seed,
        "encoded_train": encoded_train,
        "encoded_test": encoded_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "source": {
            **source_meta,
            **pair_meta,
            "cache_artifacts": q60.q40_validate._dataset_artifacts(args.cache_dir),
        },
        "train_encoding_stats": train_stats,
        "test_encoding_stats": test_stats,
    }


def _configuration_only(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "single_scale": float(row["single_scale"]),
        "phase_scale": float(row["phase_scale"]),
        "pair_scale": float(row["pair_scale"]),
        "selected_feature_count": int(row["selected_feature_count"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_paths = [Path(path) for path in args.tuning_reports]
    if len(report_paths) < 2:
        raise RunnerError("At least two tuning reports are required")
    raw_reports: list[dict[str, Any]] = []
    training_only_runs: list[dict[str, Any]] = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        source_hash = _file_sha256(path)
        raw_reports.append(report)
        training_only_runs.append(
            extract_training_only_run(
                report,
                source_path=str(path.resolve()),
                source_sha256=source_hash,
            )
        )

    shared_config = _shared_source_configuration(raw_reports)
    selection = select_cross_seed_configuration(training_only_runs)
    frozen_configuration = _configuration_only(selection["chosen"])
    excluded_indices = _excluded_source_indices(raw_reports)
    confirmation = load_fresh_confirmation_data(
        args,
        excluded_indices=excluded_indices,
        shared_config=shared_config,
    )
    seed = int(confirmation["seed"])
    encoded_train = confirmation["encoded_train"]
    encoded_test = confirmation["encoded_test"]
    y_train = confirmation["y_train"]
    y_test = confirmation["y_test"]
    mappings = toy.pauli_feature_mappings(int(shared_config["qubits"]), family="local")
    frozen_final = tuner.evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        y_test,
        mappings=mappings,
        configuration=frozen_configuration,
        shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
        bootstrap_seed=seed + 9100,
    )
    fixed_configuration = {
        "single_scale": q60.SINGLE_SCALE,
        "phase_scale": q60.PHASE_SCALE,
        "pair_scale": q60.PAIR_SCALE,
        "selected_feature_count": q60.DEFAULT_SELECTED_FEATURES,
    }
    fixed_final = tuner.evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        y_test,
        mappings=mappings,
        configuration=fixed_configuration,
        shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
        bootstrap_seed=seed + 9101,
    )
    classical_train_scores, classical_test_scores = q60._ridge_scores(
        encoded_train, encoded_test, y_train
    )
    classical = {
        "model": "standardized ridge on the 60 hashed input bins",
        "train_balanced_accuracy": q60._balanced_accuracy(
            y_train, classical_train_scores
        ),
        "test_balanced_accuracy": q60._balanced_accuracy(
            y_test, classical_test_scores
        ),
        "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
            y_test, classical_test_scores, seed=seed + 9102
        ),
    }
    frozen_test = float(frozen_final["test_balanced_accuracy"])
    fixed_test = float(fixed_final["test_balanced_accuracy"])
    classical_test = float(classical["test_balanced_accuracy"])
    train_indices = [int(value) for value in confirmation["train_indices"]]
    test_indices = [int(value) for value in confirmation["test_indices"]]
    confirmation_indices = set(train_indices) | set(test_indices)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_cross_seed_frozen_confirmation",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "selection": {
            **selection,
            "source_runs": [
                {
                    key: value
                    for key, value in run.items()
                    if key != "leaderboard"
                }
                for run in training_only_runs
            ],
            "source_final_evaluations_ignored": True,
        },
        "confirmation": {
            "requested_seed_start": int(args.confirmation_seed_start),
            "actual_seed": seed,
            "seed_scan_limit": int(args.seed_scan),
            "configuration_fixed_before_loading_confirmation": True,
            "train_samples": len(encoded_train),
            "test_samples": len(encoded_test),
            "source_indices_excluded": len(excluded_indices),
            "source_index_overlap": len(confirmation_indices & excluded_indices),
            "train_test_index_overlap": len(set(train_indices) & set(test_indices)),
            "split": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "encoded_train_sha256": q60.q40_validate._array_sha256(
                    encoded_train
                ),
                "encoded_test_sha256": q60.q40_validate._array_sha256(encoded_test),
                "train_encoding_stats": confirmation["train_encoding_stats"],
                "test_encoding_stats": confirmation["test_encoding_stats"],
            },
            "frozen_q60": frozen_final,
            "fixed_q60_reference": fixed_final,
            "classical_60bin_reference": classical,
            "frozen_minus_fixed_test_balanced_accuracy": frozen_test - fixed_test,
            "frozen_minus_classical_test_balanced_accuracy": (
                frozen_test - classical_test
            ),
            "passes_hardware_accuracy_gate": bool(
                frozen_test >= classical_test and frozen_test >= 0.60
            ),
        },
        "source": confirmation["source"],
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "One configuration was frozen from three training-CV leaderboards; "
            "all source final evaluations were ignored. The disjoint confirmation "
            "split was evaluated once. Results are exact local causal-cone model "
            "diagnostics, not hardware evidence or evidence of quantum advantage."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-reports",
        type=Path,
        nargs="+",
        default=DEFAULT_REPORTS,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--confirmation-seed-start",
        type=int,
        default=DEFAULT_CONFIRMATION_SEED_START,
    )
    parser.add_argument("--seed-scan", type=int, default=DEFAULT_SEED_SCAN)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples", type=int, default=DEFAULT_CONFIRMATION_SAMPLES
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=DEFAULT_CONFIRMATION_SAMPLES
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q60_shallow/"
            "pbmc68k_q60_cross_seed_frozen_confirmation.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.seed_scan < 1 or args.max_train_samples < 2 or args.max_test_samples < 2:
        raise RunnerError("Seed scan and confirmation sample counts must be positive")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    chosen = report["selection"]["chosen"]
    confirmation = report["confirmation"]
    print("PBMC68k q60 cross-seed frozen confirmation")
    print(f"- source seeds: {report['selection']['source_seeds']}")
    print(
        "- frozen: "
        f"single={chosen['single_scale']}, phase={chosen['phase_scale']}, "
        f"pair={chosen['pair_scale']}, features={chosen['selected_feature_count']}"
    )
    print(
        "- aggregate training CV: "
        f"{chosen['aggregate_cv_mean_balanced_accuracy']:.4f}"
    )
    print(f"- fresh confirmation seed: {confirmation['actual_seed']}")
    print(
        "- frozen q60 held-out balanced accuracy: "
        f"{confirmation['frozen_q60']['test_balanced_accuracy']:.4f}"
    )
    print(
        "- fixed q60 held-out balanced accuracy: "
        f"{confirmation['fixed_q60_reference']['test_balanced_accuracy']:.4f}"
    )
    print(
        "- classical held-out balanced accuracy: "
        f"{confirmation['classical_60bin_reference']['test_balanced_accuracy']:.4f}"
    )
    print(f"- hardware accuracy gate: {confirmation['passes_hardware_accuracy_gate']}")
    print("- provider calls: 0")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
