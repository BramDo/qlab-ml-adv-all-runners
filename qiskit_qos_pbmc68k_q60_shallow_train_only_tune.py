#!/usr/bin/env python3
"""Tune the shallow PBMC68k q60 model locally without test-set leakage.

Hyperparameters and the observable count are chosen with stratified cross-
validation on the training split only.  The held-out test split is evaluated
once, after the winning configuration has been fixed.  This script has no
provider, Fire Opal, IBM, or hardware execution path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_SEED = 11
DEFAULT_QUBITS = 60
DEFAULT_TRAIN_SAMPLES = 32
DEFAULT_TEST_SAMPLES = 32
DEFAULT_ACTIVE_GENES = 256
DEFAULT_SHOT_INTENT = 1024
DEFAULT_CV_FOLDS = 4
DEFAULT_SINGLE_SCALES = (0.75, 1.35, 2.0)
DEFAULT_PHASE_SCALES = (0.25, 0.75, 1.25)
DEFAULT_PAIR_SCALES = (0.95, 3.0, 6.0)
DEFAULT_SELECTED_COUNTS = (8, 16, 24)

RunnerError = q60.RunnerError


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("expected finite non-negative floats")
    return values


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def scaled_feature_matrix(
    parameter_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    mappings: Sequence[Mapping[int, str]],
    *,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
) -> np.ndarray:
    return np.asarray(
        [
            [
                q60.exact_local_expectation(
                    linear,
                    pair,
                    mapping,
                    single_scale=single_scale,
                    phase_scale=phase_scale,
                    pair_scale=pair_scale,
                )
                for mapping in mappings
            ]
            for linear, pair in parameter_rows
        ],
        dtype=np.float64,
    )


def _config_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer mean/worst CV score, stability, then the smaller model."""

    return (
        -float(row["cv_mean_balanced_accuracy"]),
        -float(row["cv_worst_balanced_accuracy"]),
        float(row["cv_std_balanced_accuracy"]),
        int(row["selected_feature_count"]),
        float(row["single_scale"]),
        float(row["phase_scale"]),
        float(row["pair_scale"]),
    )


def train_only_grid_search(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    single_scales: Sequence[float],
    phase_scales: Sequence[float],
    pair_scales: Sequence[float],
    selected_counts: Sequence[int],
    cv_folds: int,
    seed: int,
    shot_intent: int,
) -> dict[str, Any]:
    """Choose a configuration using training inputs and labels only."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if encoded_train.ndim != 2 or len(encoded_train) != len(y_train):
        raise RunnerError("Training matrix and labels are inconsistent")
    if cv_folds < 2:
        raise RunnerError("Cross-validation needs at least two folds")
    class_counts = [int(np.sum(y_train < 0.0)), int(np.sum(y_train > 0.0))]
    if min(class_counts) < cv_folds:
        raise RunnerError("Each training class needs at least cv_folds samples")
    if max(selected_counts) > len(mappings):
        raise RunnerError("Selected-feature count exceeds candidate observables")

    splitter = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=int(seed)
    )
    folds = [
        (fit.astype(np.int64), validation.astype(np.int64))
        for fit, validation in splitter.split(encoded_train, y_train > 0.0)
    ]
    query_parameters = [q60.query_parameters(row) for row in encoded_train]
    leaderboard: list[dict[str, Any]] = []

    for single_scale, phase_scale, pair_scale in itertools.product(
        single_scales, phase_scales, pair_scales
    ):
        query_features = scaled_feature_matrix(
            query_parameters,
            mappings,
            single_scale=float(single_scale),
            phase_scale=float(phase_scale),
            pair_scale=float(pair_scale),
        )
        fold_cache: list[dict[str, Any]] = []
        for fold_index, (fit_indices, validation_indices) in enumerate(folds):
            sketch_linear, sketch_pair = q60.sketch_parameters(
                encoded_train[fit_indices], y_train[fit_indices]
            )
            sketch_features = scaled_feature_matrix(
                [(sketch_linear, sketch_pair)],
                mappings,
                single_scale=float(single_scale),
                phase_scale=float(phase_scale),
                pair_scale=float(pair_scale),
            )[0]
            fold_cache.append(
                {
                    "fold_index": int(fold_index),
                    "fit_indices": fit_indices,
                    "validation_indices": validation_indices,
                    "sketch_features": sketch_features,
                }
            )

        for selected_count in selected_counts:
            fold_scores: list[float] = []
            fold_rows: list[dict[str, Any]] = []
            for fold in fold_cache:
                fit_indices = fold["fit_indices"]
                validation_indices = fold["validation_indices"]
                selected, _ = q60.select_train_only_features(
                    query_features[fit_indices],
                    y_train[fit_indices],
                    count=int(selected_count),
                    shot_intent=int(shot_intent),
                )
                model = fold["sketch_features"][selected]
                head_fit = q60._head_features(
                    model, query_features[fit_indices][:, selected]
                )
                head_validation = q60._head_features(
                    model, query_features[validation_indices][:, selected]
                )
                _, validation_scores = q60._ridge_scores(
                    head_fit, head_validation, y_train[fit_indices]
                )
                balanced_accuracy = q60._balanced_accuracy(
                    y_train[validation_indices], validation_scores
                )
                fold_scores.append(balanced_accuracy)
                fold_rows.append(
                    {
                        "fold_index": int(fold["fold_index"]),
                        "fit_count": int(len(fit_indices)),
                        "validation_count": int(len(validation_indices)),
                        "fit_positive": int(np.sum(y_train[fit_indices] > 0.0)),
                        "fit_negative": int(np.sum(y_train[fit_indices] < 0.0)),
                        "validation_positive": int(
                            np.sum(y_train[validation_indices] > 0.0)
                        ),
                        "validation_negative": int(
                            np.sum(y_train[validation_indices] < 0.0)
                        ),
                        "balanced_accuracy": float(balanced_accuracy),
                    }
                )
            leaderboard.append(
                {
                    "single_scale": float(single_scale),
                    "phase_scale": float(phase_scale),
                    "pair_scale": float(pair_scale),
                    "selected_feature_count": int(selected_count),
                    "cv_mean_balanced_accuracy": float(np.mean(fold_scores)),
                    "cv_worst_balanced_accuracy": float(np.min(fold_scores)),
                    "cv_std_balanced_accuracy": float(np.std(fold_scores)),
                    "folds": fold_rows,
                }
            )

    ranked = sorted(leaderboard, key=_config_key)
    return {
        "selection_scope": "training_split_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "cv_folds": int(cv_folds),
        "training_samples": int(len(encoded_train)),
        "training_class_balance": {
            "positive": class_counts[1],
            "negative": class_counts[0],
        },
        "scale_combinations": int(
            len(single_scales) * len(phase_scales) * len(pair_scales)
        ),
        "candidate_configurations": int(len(ranked)),
        "chosen": dict(ranked[0]),
        "leaderboard": ranked,
    }


def _mapping_row(
    index: int, mapping: Mapping[int, str], selection_score: float
) -> dict[str, Any]:
    return {
        "candidate_index": int(index),
        "selection_score": float(selection_score),
        "measurement_basis": q60.measurement_basis_for_mapping(mapping),
        "pauli_mapping": q60._normalized_mapping(mapping),
    }


def _conditional_stratified_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    replicates: int = 10000,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.flatnonzero(labels > 0.0)
    negative = np.flatnonzero(labels < 0.0)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_positive = rng.choice(positive, len(positive), replace=True)
        sampled_negative = rng.choice(negative, len(negative), replace=True)
        sampled = np.concatenate([sampled_positive, sampled_negative])
        values[index] = q60._balanced_accuracy(labels[sampled], scores[sampled])
    return {
        "method": "conditional stratified bootstrap of fixed test predictions",
        "replicates": int(replicates),
        "seed": int(seed),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def evaluate_fixed_configuration(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    encoded_test: np.ndarray,
    y_test: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    shot_intent: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Fit on all training data, then evaluate the untouched test split once."""

    single_scale = float(configuration["single_scale"])
    phase_scale = float(configuration["phase_scale"])
    pair_scale = float(configuration["pair_scale"])
    selected_count = int(configuration["selected_feature_count"])
    train_parameters = [q60.query_parameters(row) for row in encoded_train]
    test_parameters = [q60.query_parameters(row) for row in encoded_test]
    train_features = scaled_feature_matrix(
        train_parameters,
        mappings,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
    )
    test_features = scaled_feature_matrix(
        test_parameters,
        mappings,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
    )
    sketch_linear, sketch_pair = q60.sketch_parameters(encoded_train, y_train)
    sketch_features = scaled_feature_matrix(
        [(sketch_linear, sketch_pair)],
        mappings,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
    )[0]
    selected, selection_scores = q60.select_train_only_features(
        train_features,
        y_train,
        count=selected_count,
        shot_intent=shot_intent,
    )
    model = sketch_features[selected]
    head_train = q60._head_features(model, train_features[:, selected])
    head_test = q60._head_features(model, test_features[:, selected])
    train_scores, test_scores = q60._ridge_scores(head_train, head_test, y_train)
    test_balanced_accuracy = q60._balanced_accuracy(y_test, test_scores)
    return {
        "configuration": {
            "single_scale": single_scale,
            "phase_scale": phase_scale,
            "pair_scale": pair_scale,
            "selected_feature_count": selected_count,
        },
        "selection_uses_training_only": True,
        "selected_observables": [
            _mapping_row(int(index), mappings[index], selection_scores[index])
            for index in selected
        ],
        "train_balanced_accuracy": q60._balanced_accuracy(y_train, train_scores),
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_score_abs_margin": {
            "minimum": float(np.min(np.abs(test_scores))),
            "median": float(np.median(np.abs(test_scores))),
        },
        "test_bootstrap_95": _conditional_stratified_bootstrap(
            y_test, test_scores, seed=bootstrap_seed
        ),
    }


def load_seed_data(args: argparse.Namespace) -> dict[str, Any]:
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    train_indices, test_indices = genomics_runner.benchmark_indices(
        x_pair.shape[0],
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        labels=y_pair,
    )
    y_train = y_pair[train_indices].astype(np.float64)
    y_test = y_pair[test_indices].astype(np.float64)
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=args.qubits,
        hash_seed=args.seed,
        value_mode=args.value_mode,
        max_active_genes=args.max_active_genes,
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=args.qubits,
        hash_seed=args.seed,
        value_mode=args.value_mode,
        max_active_genes=args.max_active_genes,
    )
    return {
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    data = load_seed_data(args)
    encoded_train = data["encoded_train"]
    encoded_test = data["encoded_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    mappings = toy.pauli_feature_mappings(args.qubits, family="local")

    tuning = train_only_grid_search(
        encoded_train,
        y_train,
        mappings=mappings,
        single_scales=args.single_scales,
        phase_scales=args.phase_scales,
        pair_scales=args.pair_scales,
        selected_counts=args.selected_counts,
        cv_folds=args.cv_folds,
        seed=args.seed,
        shot_intent=args.shot_intent,
    )
    chosen = tuning["chosen"]
    tuned_final = evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        y_test,
        mappings=mappings,
        configuration=chosen,
        shot_intent=args.shot_intent,
        bootstrap_seed=args.seed + 9000,
    )
    fixed_final = evaluate_fixed_configuration(
        encoded_train,
        y_train,
        encoded_test,
        y_test,
        mappings=mappings,
        configuration={
            "single_scale": q60.SINGLE_SCALE,
            "phase_scale": q60.PHASE_SCALE,
            "pair_scale": q60.PAIR_SCALE,
            "selected_feature_count": q60.DEFAULT_SELECTED_FEATURES,
        },
        shot_intent=args.shot_intent,
        bootstrap_seed=args.seed + 9001,
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
        "test_bootstrap_95": _conditional_stratified_bootstrap(
            y_test, classical_test_scores, seed=args.seed + 9002
        ),
    }
    tuned_test = float(tuned_final["test_balanced_accuracy"])
    fixed_test = float(fixed_final["test_balanced_accuracy"])
    classical_test = float(classical["test_balanced_accuracy"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_strict_train_only_tuning",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "config": {
            "dataset": "PBMC68k",
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "seed": args.seed,
            "qubits": args.qubits,
            "train_samples": len(encoded_train),
            "test_samples": len(encoded_test),
            "value_mode": args.value_mode,
            "max_active_genes": args.max_active_genes,
            "shot_intent_for_feature_ranking": args.shot_intent,
            "single_scales": list(args.single_scales),
            "phase_scales": list(args.phase_scales),
            "pair_scales": list(args.pair_scales),
            "selected_counts": list(args.selected_counts),
            "cv_folds": args.cv_folds,
        },
        "source": data["source"],
        "split": {
            "train_indices": [int(value) for value in data["train_indices"]],
            "test_indices": [int(value) for value in data["test_indices"]],
            "encoded_train_sha256": q60.q40_validate._array_sha256(encoded_train),
            "encoded_test_sha256": q60.q40_validate._array_sha256(encoded_test),
            "train_encoding_stats": data["train_encoding_stats"],
            "test_encoding_stats": data["test_encoding_stats"],
        },
        "candidate_observable_count": len(mappings),
        "train_only_tuning": tuning,
        "final_evaluation": {
            "test_evaluated_after_configuration_fixed": True,
            "tuned": tuned_final,
            "fixed_q60_reference": fixed_final,
            "classical_60bin_reference": classical,
            "tuned_minus_fixed_test_balanced_accuracy": tuned_test - fixed_test,
            "tuned_minus_classical_test_balanced_accuracy": (
                tuned_test - classical_test
            ),
            "passes_hardware_accuracy_gate": bool(
                tuned_test >= classical_test and tuned_test >= 0.60
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Hyperparameters were selected with training-only cross-validation. "
            "The held-out result is a 32-sample noiseless diagnostic, not a "
            "hardware prediction or evidence of quantum advantage."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qubits", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=DEFAULT_TEST_SAMPLES
    )
    parser.add_argument(
        "--max-active-genes", type=int, default=DEFAULT_ACTIVE_GENES
    )
    parser.add_argument(
        "--value-mode", choices=("binary", "log-product"), default="log-product"
    )
    parser.add_argument(
        "--single-scales", type=_parse_floats, default=DEFAULT_SINGLE_SCALES
    )
    parser.add_argument(
        "--phase-scales", type=_parse_floats, default=DEFAULT_PHASE_SCALES
    )
    parser.add_argument(
        "--pair-scales", type=_parse_floats, default=DEFAULT_PAIR_SCALES
    )
    parser.add_argument(
        "--selected-counts", type=_parse_ints, default=DEFAULT_SELECTED_COUNTS
    )
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--shot-intent", type=int, default=DEFAULT_SHOT_INTENT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q60_shallow/"
            "pbmc68k_q60_seed11_train_only_tuning.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
    if args.qubits < 2 or args.shot_intent < 1:
        raise RunnerError("Qubits and shot intent must be positive")
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
    final = report["final_evaluation"]
    chosen = report["train_only_tuning"]["chosen"]
    print("PBMC68k q60 shallow strict train-only tuning")
    print(f"- candidates: {report['train_only_tuning']['candidate_configurations']}")
    print(
        "- chosen: "
        f"single={chosen['single_scale']}, phase={chosen['phase_scale']}, "
        f"pair={chosen['pair_scale']}, features={chosen['selected_feature_count']}"
    )
    print(f"- CV balanced accuracy: {chosen['cv_mean_balanced_accuracy']:.4f}")
    print(
        f"- tuned held-out balanced accuracy: "
        f"{final['tuned']['test_balanced_accuracy']:.4f}"
    )
    print(
        f"- fixed q60 held-out balanced accuracy: "
        f"{final['fixed_q60_reference']['test_balanced_accuracy']:.4f}"
    )
    print(
        f"- classical held-out balanced accuracy: "
        f"{final['classical_60bin_reference']['test_balanced_accuracy']:.4f}"
    )
    print(f"- hardware accuracy gate: {final['passes_hardware_accuracy_gate']}")
    print("- provider calls: 0")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
