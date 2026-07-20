#!/usr/bin/env python3
"""Tune only the RZZ pair scale with fixed balanced observables.

For every old training fold, the 8/8/8 observable set is selected once at the
pre-registered reference scale 0.95.  Exactly those observables are then used
to compare pair scales 0, 0.95, 3, and 6.  This isolates the entangling layer
from feature-selection changes.

A fresh split is opened only if a non-zero scale beats the pair-zero ablation
in training-only cross-validation and passes the predeclared quality gates.
There is no provider or hardware execution path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import qiskit_qos_pbmc68k_q60_balanced_representation as balanced
import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_PAIR_SCALES = (0.0, 0.95, 3.0, 6.0)
DEFAULT_REFERENCE_PAIR_SCALE = 0.95
DEFAULT_BALANCED_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_representation.json"
)
DEFAULT_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_balanced_pair_scale_tuning.json"
)
DEFAULT_FRESH_SEED_START = 53
DEFAULT_SEED_SCAN = 300

RunnerError = q60.RunnerError


def _scaled_features(
    parameter_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    mappings: Sequence[Mapping[int, str]],
    *,
    configuration: Mapping[str, Any],
    pair_scale: float,
) -> np.ndarray:
    return tuner.scaled_feature_matrix(
        parameter_rows,
        mappings,
        single_scale=float(configuration["single_scale"]),
        phase_scale=float(configuration["phase_scale"]),
        pair_scale=float(pair_scale),
    )


def cross_validate_pair_scales(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    pair_scales: Sequence[float],
    reference_pair_scale: float,
    cv_folds: int,
    seed: int,
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Compare scales with one train-only observable set per fold."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    scales = tuple(float(value) for value in pair_scales)
    if len(set(scales)) != len(scales) or not scales:
        raise RunnerError("Pair-scale candidates must be unique and non-empty")
    if 0.0 not in scales:
        raise RunnerError("Pair-scale candidates must include the zero ablation")
    if float(reference_pair_scale) not in scales:
        raise RunnerError("Reference pair scale must be one of the candidates")
    if any(not np.isfinite(value) or value < 0.0 for value in scales):
        raise RunnerError("Pair scales must be finite and non-negative")

    parameters = [q60.query_parameters(row) for row in encoded_train]
    feature_by_scale = {
        scale: _scaled_features(
            parameters,
            mappings,
            configuration=configuration,
            pair_scale=scale,
        )
        for scale in scales
    }
    splitter = StratifiedKFold(
        n_splits=int(cv_folds), shuffle=True, random_state=int(seed)
    )
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(encoded_train, y_train > 0.0)
    ):
        fit_indices = fit_indices.astype(np.int64)
        validation_indices = validation_indices.astype(np.int64)
        selected, _, audit = balanced.select_balanced_train_only_features(
            feature_by_scale[float(reference_pair_scale)][fit_indices],
            feature_by_scale[0.0][fit_indices],
            y_train[fit_indices],
            mappings=mappings,
            z_quota=int(z_quota),
            transverse_quota=int(transverse_quota),
            multiqubit_quota=int(multiqubit_quota),
            shot_intent=int(shot_intent),
            sensitivity_threshold=float(sensitivity_threshold),
        )
        scale_scores: list[dict[str, Any]] = []
        for scale in scales:
            sketch = balanced._sketch_features(
                encoded_train[fit_indices],
                y_train[fit_indices],
                mappings=mappings,
                configuration=configuration,
                pair_scale=scale,
            )
            _, validation_scores = balanced._fit_selected_scores(
                feature_by_scale[scale][fit_indices],
                feature_by_scale[scale][validation_indices],
                y_train[fit_indices],
                sketch,
                selected,
            )
            scale_scores.append(
                {
                    "pair_scale": scale,
                    "balanced_accuracy": q60._balanced_accuracy(
                        y_train[validation_indices], validation_scores
                    ),
                }
            )
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "fit_samples": len(fit_indices),
                "validation_samples": len(validation_indices),
                "selected_indices": [int(value) for value in selected],
                "selection_audit": audit,
                "pair_scale_scores": scale_scores,
            }
        )

    candidates: list[dict[str, Any]] = []
    for scale in scales:
        values = [
            next(
                item["balanced_accuracy"]
                for item in fold["pair_scale_scores"]
                if item["pair_scale"] == scale
            )
            for fold in fold_rows
        ]
        candidates.append(
            {
                "pair_scale": scale,
                "cv_mean_balanced_accuracy": float(np.mean(values)),
                "cv_worst_balanced_accuracy": float(np.min(values)),
                "cv_std_balanced_accuracy": float(np.std(values)),
                "fold_balanced_accuracies": [float(value) for value in values],
            }
        )
    return {
        "selection_scope": "training_split_only",
        "test_inputs_seen": False,
        "test_labels_seen": False,
        "observable_selection_fixed_across_pair_scales_within_each_fold": True,
        "reference_pair_scale": float(reference_pair_scale),
        "seed": int(seed),
        "training_samples": len(encoded_train),
        "cv_folds": int(cv_folds),
        "candidates": candidates,
        "folds": fold_rows,
    }


def _aggregate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -round(float(row["equal_seed_cv_mean"]), 12),
        -round(float(row["worst_seed_cv_mean"]), 12),
        round(float(row["between_seed_cv_std"]), 12),
        float(row["pair_scale"]),
    )


def aggregate_pair_scale_cv(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    cv_mean_gate: float,
    cv_worst_seed_gate: float,
) -> dict[str, Any]:
    """Rank scales equally across seeds and apply the entanglement gate."""

    if len(per_seed) < 2:
        raise RunnerError("Pair-scale aggregation needs at least two seeds")
    candidate_sets = [
        {float(row["pair_scale"]) for row in seed_row["candidates"]}
        for seed_row in per_seed
    ]
    if any(values != candidate_sets[0] for values in candidate_sets[1:]):
        raise RunnerError("Pair-scale candidate sets differ across seeds")
    rows: list[dict[str, Any]] = []
    for scale in sorted(candidate_sets[0]):
        seed_rows = [
            next(
                row
                for row in seed_row["candidates"]
                if float(row["pair_scale"]) == scale
            )
            for seed_row in per_seed
        ]
        means = np.asarray(
            [row["cv_mean_balanced_accuracy"] for row in seed_rows], dtype=float
        )
        rows.append(
            {
                "pair_scale": scale,
                "equal_seed_cv_mean": float(np.mean(means)),
                "worst_seed_cv_mean": float(np.min(means)),
                "between_seed_cv_std": float(np.std(means)),
                "per_seed": [
                    {
                        "seed": int(seed_result["seed"]),
                        "cv_mean_balanced_accuracy": float(candidate["cv_mean_balanced_accuracy"]),
                        "cv_worst_balanced_accuracy": float(candidate["cv_worst_balanced_accuracy"]),
                        "cv_std_balanced_accuracy": float(candidate["cv_std_balanced_accuracy"]),
                    }
                    for seed_result, candidate in zip(
                        per_seed, seed_rows, strict=True
                    )
                ],
            }
        )
    ranked = sorted(rows, key=_aggregate_rank_key)
    chosen = dict(ranked[0])
    pair_zero = next(row for row in rows if row["pair_scale"] == 0.0)
    improvement = float(
        chosen["equal_seed_cv_mean"] - pair_zero["equal_seed_cv_mean"]
    )
    nonzero_strictly_beats_zero = bool(
        float(chosen["pair_scale"]) > 0.0
        and round(improvement, 12) > 0.0
    )
    passes = bool(
        nonzero_strictly_beats_zero
        and chosen["equal_seed_cv_mean"] >= float(cv_mean_gate)
        and chosen["worst_seed_cv_mean"] >= float(cv_worst_seed_gate)
    )
    return {
        "ranking_rule": (
            "maximize equal-seed CV mean, then worst seed, then stability; "
            "prefer smaller pair scale on numerical ties"
        ),
        "candidates": sorted(rows, key=lambda row: float(row["pair_scale"])),
        "chosen": chosen,
        "pair_zero_reference": dict(pair_zero),
        "chosen_minus_pair_zero_cv_mean": improvement,
        "nonzero_strictly_beats_pair_zero": nonzero_strictly_beats_zero,
        "cv_mean_gate": float(cv_mean_gate),
        "cv_worst_seed_gate": float(cv_worst_seed_gate),
        "passes_fresh_confirmation_gate": passes,
    }


def _features_for_scale(
    encoded: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    pair_scale: float,
) -> np.ndarray:
    parameters = [q60.query_parameters(row) for row in encoded]
    return _scaled_features(
        parameters,
        mappings,
        configuration=configuration,
        pair_scale=pair_scale,
    )


def evaluate_fresh_pair_scale(
    data: Mapping[str, Any],
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    chosen_pair_scale: float,
    reference_pair_scale: float,
    shot_intent: int,
    z_quota: int,
    transverse_quota: int,
    multiqubit_quota: int,
    sensitivity_threshold: float,
) -> dict[str, Any]:
    """Select on fresh training data, then evaluate chosen and zero scales once."""

    reference_train = _features_for_scale(
        data["encoded_train"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=reference_pair_scale,
    )
    pair_zero_train = _features_for_scale(
        data["encoded_train"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=0.0,
    )
    selected, selection_scores, audit = balanced.select_balanced_train_only_features(
        reference_train,
        pair_zero_train,
        data["y_train"],
        mappings=mappings,
        z_quota=int(z_quota),
        transverse_quota=int(transverse_quota),
        multiqubit_quota=int(multiqubit_quota),
        shot_intent=int(shot_intent),
        sensitivity_threshold=float(sensitivity_threshold),
    )
    pair_zero_test = _features_for_scale(
        data["encoded_test"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=0.0,
    )
    if float(chosen_pair_scale) == 0.0:
        chosen_train = pair_zero_train
        chosen_test = pair_zero_test
    elif float(chosen_pair_scale) == float(reference_pair_scale):
        chosen_train = reference_train
        chosen_test = _features_for_scale(
            data["encoded_test"],
            mappings=mappings,
            configuration=configuration,
            pair_scale=reference_pair_scale,
        )
    else:
        chosen_train = _features_for_scale(
            data["encoded_train"],
            mappings=mappings,
            configuration=configuration,
            pair_scale=chosen_pair_scale,
        )
        chosen_test = _features_for_scale(
            data["encoded_test"],
            mappings=mappings,
            configuration=configuration,
            pair_scale=chosen_pair_scale,
        )
    chosen_sketch = balanced._sketch_features(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=chosen_pair_scale,
    )
    pair_zero_sketch = balanced._sketch_features(
        data["encoded_train"],
        data["y_train"],
        mappings=mappings,
        configuration=configuration,
        pair_scale=0.0,
    )
    chosen_train_scores, chosen_test_scores = balanced._fit_selected_scores(
        chosen_train,
        chosen_test,
        data["y_train"],
        chosen_sketch,
        selected,
    )
    _, zero_test_scores = balanced._fit_selected_scores(
        pair_zero_train,
        pair_zero_test,
        data["y_train"],
        pair_zero_sketch,
        selected,
    )
    classical_train, classical_test = q60._ridge_scores(
        data["encoded_train"], data["encoded_test"], data["y_train"]
    )
    chosen_accuracy = q60._balanced_accuracy(data["y_test"], chosen_test_scores)
    zero_accuracy = q60._balanced_accuracy(data["y_test"], zero_test_scores)
    classical_accuracy = q60._balanced_accuracy(data["y_test"], classical_test)
    seed = int(data["seed"])
    return {
        "configuration_fixed_before_fresh_split": True,
        "observable_selection_uses_fresh_training_only": True,
        "chosen_pair_scale": float(chosen_pair_scale),
        "selection_reference_pair_scale": float(reference_pair_scale),
        "representation_audit": audit,
        "selected_observables": [
            tuner._mapping_row(
                int(index), mappings[int(index)], selection_scores[int(index)]
            )
            for index in selected
        ],
        "chosen_scale_result": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], chosen_train_scores
            ),
            "test_balanced_accuracy": chosen_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], chosen_test_scores, seed=seed + 9400
            ),
        },
        "same_observables_pair_zero_result": {
            "test_balanced_accuracy": zero_accuracy,
            "chosen_minus_zero_test": chosen_accuracy - zero_accuracy,
        },
        "classical_60bin_reference": {
            "train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], classical_train
            ),
            "test_balanced_accuracy": classical_accuracy,
            "test_bootstrap_95": tuner._conditional_stratified_bootstrap(
                data["y_test"], classical_test, seed=seed + 9401
            ),
        },
        "chosen_minus_classical_test": chosen_accuracy - classical_accuracy,
        "paper_feasibility_signal": bool(chosen_accuracy >= 0.55),
    }


def _add_split_indices(excluded: set[int], report: Mapping[str, Any]) -> None:
    split = report["confirmation"]["split"]
    excluded.update(int(value) for value in split["train_indices"])
    excluded.update(int(value) for value in split["test_indices"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_paths = [Path(path) for path in args.tuning_reports]
    prior_path = Path(args.prior_confirmation)
    balanced_path = Path(args.balanced_report)
    prior_report = json.loads(prior_path.read_text(encoding="utf-8"))
    balanced_report = json.loads(balanced_path.read_text(encoding="utf-8"))
    configuration = freeze._configuration_only(prior_report["selection"]["chosen"])
    runs, raw_reports, shared_config, x_pair, y_pair = balanced._load_source_training_data(
        report_paths, cache_dir=Path(args.cache_dir)
    )
    mappings = toy.pauli_feature_mappings(
        int(shared_config["qubits"]), family="local"
    )
    per_seed: list[dict[str, Any]] = []
    for run_data in runs:
        result = cross_validate_pair_scales(
            run_data["encoded_train"],
            run_data["y_train"],
            mappings=mappings,
            configuration=configuration,
            pair_scales=args.pair_scales,
            reference_pair_scale=float(args.reference_pair_scale),
            cv_folds=int(shared_config["cv_folds"]),
            seed=int(run_data["seed"]),
            shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
            z_quota=int(args.z_quota),
            transverse_quota=int(args.transverse_quota),
            multiqubit_quota=int(args.multiqubit_quota),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        previous = next(
            row
            for row in balanced_report["training_only_cross_seed_validation"][
                "per_seed"
            ]
            if int(row["seed"]) == int(run_data["seed"])
        )
        reference_candidate = next(
            row
            for row in result["candidates"]
            if float(row["pair_scale"]) == float(args.reference_pair_scale)
        )
        if not np.isclose(
            reference_candidate["cv_mean_balanced_accuracy"],
            previous["balanced_cv_mean"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RunnerError(
                f"Balanced reference CV reproduction failed for seed {run_data['seed']}"
            )
        result["balanced_reference_metric_reproduced"] = True
        result["source_report"] = run_data["report_path"]
        result["source_report_sha256"] = run_data["report_sha256"]
        per_seed.append(result)

    aggregate = aggregate_pair_scale_cv(
        per_seed,
        cv_mean_gate=float(args.cv_mean_gate),
        cv_worst_seed_gate=float(args.cv_worst_seed_gate),
    )
    fresh_confirmation: dict[str, Any] = {
        "executed": False,
        "reason": (
            "no non-zero pair scale strictly beat pair=0 while passing the "
            "training-only gates"
        ),
    }
    if aggregate["passes_fresh_confirmation_gate"]:
        excluded = freeze._excluded_source_indices(raw_reports)
        _add_split_indices(excluded, prior_report)
        balanced_split = balanced_report["fresh_confirmation"]["split"]
        excluded.update(int(value) for value in balanced_split["train_indices"])
        excluded.update(int(value) for value in balanced_split["test_indices"])
        fresh = balanced._load_fresh_split(
            args,
            x_pair=x_pair,
            y_pair=y_pair,
            shared_config=shared_config,
            excluded_indices=excluded,
        )
        evaluation = evaluate_fresh_pair_scale(
            fresh,
            mappings=mappings,
            configuration=configuration,
            chosen_pair_scale=float(aggregate["chosen"]["pair_scale"]),
            reference_pair_scale=float(args.reference_pair_scale),
            shot_intent=int(shared_config["shot_intent_for_feature_ranking"]),
            z_quota=int(args.z_quota),
            transverse_quota=int(args.transverse_quota),
            multiqubit_quota=int(args.multiqubit_quota),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        fresh_indices = set(int(value) for value in fresh["train_indices"]) | set(
            int(value) for value in fresh["test_indices"]
        )
        fresh_confirmation = {
            "executed": True,
            "actual_seed": int(fresh["seed"]),
            "train_samples": len(fresh["encoded_train"]),
            "test_samples": len(fresh["encoded_test"]),
            "excluded_source_indices": len(excluded),
            "source_index_overlap": len(fresh_indices & excluded),
            "train_test_index_overlap": len(
                set(int(value) for value in fresh["train_indices"])
                & set(int(value) for value in fresh["test_indices"])
            ),
            "split": {
                "train_indices": [int(value) for value in fresh["train_indices"]],
                "test_indices": [int(value) for value in fresh["test_indices"]],
                "encoded_train_sha256": q60.q40_validate._array_sha256(
                    fresh["encoded_train"]
                ),
                "encoded_test_sha256": q60.q40_validate._array_sha256(
                    fresh["encoded_test"]
                ),
            },
            "evaluation": evaluation,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_balanced_pair_scale_tuning",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "pre_registered_protocol": {
            "pair_scales": [float(value) for value in args.pair_scales],
            "reference_pair_scale_for_observable_selection": float(
                args.reference_pair_scale
            ),
            "observable_quotas": {
                "single_z": int(args.z_quota),
                "pair_sensitive_local_xy": int(args.transverse_quota),
                "pair_sensitive_multiqubit": int(args.multiqubit_quota),
            },
            "scales_compared_with_identical_observables_within_each_fold": True,
            "prior_confirmation_report": str(prior_path.resolve()),
            "prior_confirmation_sha256": freeze._file_sha256(prior_path),
            "balanced_report": str(balanced_path.resolve()),
            "balanced_report_sha256": freeze._file_sha256(balanced_path),
        },
        "training_only_pair_scale_cv": {
            "source_test_results_used": False,
            "source_seeds": [int(run["seed"]) for run in runs],
            "per_seed": per_seed,
            "aggregate": aggregate,
        },
        "fresh_confirmation": fresh_confirmation,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Pair scale was selected using old training folds only. A fresh "
            "split is opened only for a non-zero scale that beats the pair-zero "
            "ablation in cross-validation. Results remain exact local model "
            "diagnostics, not hardware evidence or quantum advantage."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuning-reports",
        type=Path,
        nargs="+",
        default=balanced.DEFAULT_TUNING_REPORTS,
    )
    parser.add_argument(
        "--prior-confirmation",
        type=Path,
        default=balanced.DEFAULT_PRIOR_CONFIRMATION,
    )
    parser.add_argument(
        "--balanced-report", type=Path, default=DEFAULT_BALANCED_REPORT
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--pair-scales", type=tuner._parse_floats, default=DEFAULT_PAIR_SCALES
    )
    parser.add_argument(
        "--reference-pair-scale", type=float, default=DEFAULT_REFERENCE_PAIR_SCALE
    )
    parser.add_argument("--z-quota", type=int, default=balanced.DEFAULT_Z_QUOTA)
    parser.add_argument(
        "--transverse-quota", type=int, default=balanced.DEFAULT_TRANSVERSE_QUOTA
    )
    parser.add_argument(
        "--multiqubit-quota", type=int, default=balanced.DEFAULT_MULTIQUBIT_QUOTA
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=balanced.DEFAULT_SENSITIVITY_THRESHOLD,
    )
    parser.add_argument(
        "--cv-mean-gate", type=float, default=balanced.DEFAULT_CV_MEAN_GATE
    )
    parser.add_argument(
        "--cv-worst-seed-gate",
        type=float,
        default=balanced.DEFAULT_CV_WORST_SEED_GATE,
    )
    parser.add_argument(
        "--fresh-seed-start", type=int, default=DEFAULT_FRESH_SEED_START
    )
    parser.add_argument("--seed-scan", type=int, default=DEFAULT_SEED_SCAN)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=balanced.DEFAULT_CONFIRMATION_SAMPLES,
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=balanced.DEFAULT_CONFIRMATION_SAMPLES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        raise RunnerError(f"Refusing to overwrite existing artifact: {args.output}")
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
    aggregate = report["training_only_pair_scale_cv"]["aggregate"]
    chosen = aggregate["chosen"]
    print("PBMC68k q60 balanced pair-scale tuning")
    print(
        f"- chosen pair scale: {chosen['pair_scale']} "
        f"(CV {chosen['equal_seed_cv_mean']:.4f})"
    )
    print(
        f"- pair=0 CV: {aggregate['pair_zero_reference']['equal_seed_cv_mean']:.4f}"
    )
    print(
        f"- chosen minus pair=0 CV: "
        f"{aggregate['chosen_minus_pair_zero_cv_mean']:+.4f}"
    )
    print(
        f"- fresh confirmation gate: "
        f"{aggregate['passes_fresh_confirmation_gate']}"
    )
    confirmation = report["fresh_confirmation"]
    if confirmation["executed"]:
        evaluation = confirmation["evaluation"]
        print(f"- fresh seed: {confirmation['actual_seed']}")
        print(
            "- chosen held-out: "
            f"{evaluation['chosen_scale_result']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- same-observable pair=0 held-out: "
            f"{evaluation['same_observables_pair_zero_result']['test_balanced_accuracy']:.4f}"
        )
        print(
            "- classical held-out: "
            f"{evaluation['classical_60bin_reference']['test_balanced_accuracy']:.4f}"
        )
    else:
        print(f"- fresh confirmation skipped: {confirmation['reason']}")
    print("- provider calls: 0")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
