#!/usr/bin/env python3
"""Reconstruct and diagnose the frozen PBMC68k q60 confirmation split.

The script compares frozen-q60 and classical predictions cell by cell, reports
paired uncertainty, and audits whether the selected observables actually use
the RZZ interaction layer.  It is a post-hoc local diagnostic: it does not tune
on the confirmation split and has no provider or hardware execution path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_q60_shallow_cross_seed_freeze as freeze
import qiskit_qos_pbmc68k_q60_shallow_fireopal_validate as q60
import qiskit_qos_pbmc68k_q60_shallow_train_only_tune as tuner
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_CONFIRMATION_REPORT = Path(
    "fire_opal_pbmc68k_q60_shallow/"
    "pbmc68k_q60_cross_seed_frozen_confirmation.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_seed24_error_analysis.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "fire_opal_pbmc68k_q60_shallow/pbmc68k_q60_seed24_error_analysis.md"
)
DEFAULT_BOOTSTRAP_REPLICATES = 20_000

RunnerError = q60.RunnerError


def exact_mcnemar_p_value(classical_only: int, q60_only: int) -> float:
    """Two-sided exact McNemar p-value for the discordant predictions."""

    if classical_only < 0 or q60_only < 0:
        raise RunnerError("McNemar counts cannot be negative")
    discordant = int(classical_only + q60_only)
    if discordant == 0:
        return 1.0
    lower = min(int(classical_only), int(q60_only))
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def paired_stratified_bootstrap(
    labels: np.ndarray,
    q60_scores: np.ndarray,
    classical_scores: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap the paired balanced-accuracy difference within each class."""

    labels = np.asarray(labels, dtype=np.float64)
    q60_scores = np.asarray(q60_scores, dtype=np.float64)
    classical_scores = np.asarray(classical_scores, dtype=np.float64)
    if not (len(labels) == len(q60_scores) == len(classical_scores)):
        raise RunnerError("Paired bootstrap inputs differ in length")
    if replicates < 1:
        raise RunnerError("Paired bootstrap needs at least one replicate")
    groups = [np.flatnonzero(labels < 0.0), np.flatnonzero(labels > 0.0)]
    if any(len(group) == 0 for group in groups):
        raise RunnerError("Paired bootstrap needs both classes")
    rng = np.random.default_rng(int(seed))
    differences = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        differences[index] = q60._balanced_accuracy(
            labels[sampled], q60_scores[sampled]
        ) - q60._balanced_accuracy(labels[sampled], classical_scores[sampled])
    point = q60._balanced_accuracy(labels, q60_scores) - q60._balanced_accuracy(
        labels, classical_scores
    )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "point_difference": float(point),
        "bootstrap_mean_difference": float(np.mean(differences)),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_q60_strictly_better": float(np.mean(differences > 0.0)),
        "probability_tie": float(np.mean(differences == 0.0)),
    }


def _predict(scores: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(scores, dtype=np.float64) >= 0.0, 1.0, -1.0)


def paired_outcome_counts(
    labels: np.ndarray, q60_scores: np.ndarray, classical_scores: np.ndarray
) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.float64)
    q60_correct = _predict(q60_scores) == labels
    classical_correct = _predict(classical_scores) == labels
    return {
        "both_correct": int(np.sum(q60_correct & classical_correct)),
        "q60_only_correct": int(np.sum(q60_correct & ~classical_correct)),
        "classical_only_correct": int(np.sum(~q60_correct & classical_correct)),
        "both_wrong": int(np.sum(~q60_correct & ~classical_correct)),
    }


def _fixed_configuration_scores(
    encoded_train: np.ndarray,
    y_train: np.ndarray,
    encoded_test: np.ndarray,
    *,
    mappings: Sequence[Mapping[int, str]],
    configuration: Mapping[str, Any],
    shot_intent: int,
) -> dict[str, Any]:
    """Reproduce the train-only feature selection and fitted score vectors."""

    single_scale = float(configuration["single_scale"])
    phase_scale = float(configuration["phase_scale"])
    pair_scale = float(configuration["pair_scale"])
    selected_count = int(configuration["selected_feature_count"])
    train_parameters = [q60.query_parameters(row) for row in encoded_train]
    test_parameters = [q60.query_parameters(row) for row in encoded_test]
    train_features = tuner.scaled_feature_matrix(
        train_parameters,
        mappings,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
    )
    test_features = tuner.scaled_feature_matrix(
        test_parameters,
        mappings,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
    )
    sketch_linear, sketch_pair = q60.sketch_parameters(encoded_train, y_train)
    sketch_features = tuner.scaled_feature_matrix(
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
        shot_intent=int(shot_intent),
    )
    model = sketch_features[selected]
    head_train = q60._head_features(model, train_features[:, selected])
    head_test = q60._head_features(model, test_features[:, selected])
    train_scores, test_scores = q60._ridge_scores(head_train, head_test, y_train)
    return {
        "selected": selected,
        "selection_scores": selection_scores,
        "train_features": train_features,
        "test_features": test_features,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def _load_source_config(confirmation_report: Mapping[str, Any]) -> dict[str, Any]:
    source_runs = confirmation_report.get("selection", {}).get("source_runs", [])
    if not source_runs:
        raise RunnerError("Confirmation report has no source-run provenance")
    source_path = Path(str(source_runs[0]["source_path"]))
    if not source_path.exists():
        raise RunnerError(f"Recorded source tuning report is unavailable: {source_path}")
    expected_hash = str(source_runs[0]["source_sha256"])
    if freeze._file_sha256(source_path) != expected_hash:
        raise RunnerError("Source tuning report hash no longer matches provenance")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    return dict(source_report["config"])


def reconstruct_confirmation_data(
    confirmation_report: Mapping[str, Any], *, cache_dir: Path
) -> dict[str, Any]:
    """Re-encode the exact recorded train/test rows and verify both hashes."""

    if confirmation_report.get("kind") != (
        "pbmc68k_q60_shallow_cross_seed_frozen_confirmation"
    ):
        raise RunnerError("Unexpected confirmation report kind")
    source_config = _load_source_config(confirmation_report)
    confirmation = confirmation_report["confirmation"]
    split = confirmation["split"]
    seed = int(confirmation["actual_seed"])
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=str(source_config["positive_label"]),
        negative_label=str(source_config["negative_label"]),
    )
    encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[train_indices],
        feature_dim=int(source_config["qubits"]),
        hash_seed=seed,
        value_mode=str(source_config["value_mode"]),
        max_active_genes=int(source_config["max_active_genes"]),
    )
    encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
        x_pair[test_indices],
        feature_dim=int(source_config["qubits"]),
        hash_seed=seed,
        value_mode=str(source_config["value_mode"]),
        max_active_genes=int(source_config["max_active_genes"]),
    )
    actual_train_hash = q60.q40_validate._array_sha256(encoded_train)
    actual_test_hash = q60.q40_validate._array_sha256(encoded_test)
    if actual_train_hash != str(split["encoded_train_sha256"]):
        raise RunnerError("Reconstructed training matrix hash mismatch")
    if actual_test_hash != str(split["encoded_test_sha256"]):
        raise RunnerError("Reconstructed test matrix hash mismatch")
    return {
        "source_config": source_config,
        "encoded_train": encoded_train,
        "encoded_test": encoded_test,
        "y_train": y_pair[train_indices].astype(np.float64),
        "y_test": y_pair[test_indices].astype(np.float64),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_encoding_stats": train_stats,
        "test_encoding_stats": test_stats,
        "source": {**source_meta, **pair_meta},
    }


def _basis_label(mapping: Mapping[int, str]) -> str:
    paulis = sorted(set(str(value) for value in mapping.values()))
    return "+".join(paulis)


def _margin_summary(
    labels: np.ndarray, q60_scores: np.ndarray, classical_scores: np.ndarray
) -> dict[str, Any]:
    q60_correct = _predict(q60_scores) == labels
    classical_correct = _predict(classical_scores) == labels
    masks = {
        "all": np.ones(len(labels), dtype=bool),
        "both_correct": q60_correct & classical_correct,
        "q60_only_correct": q60_correct & ~classical_correct,
        "classical_only_correct": ~q60_correct & classical_correct,
        "both_wrong": ~q60_correct & ~classical_correct,
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        q_values = np.abs(q60_scores[mask])
        c_values = np.abs(classical_scores[mask])
        result[name] = {
            "count": int(np.sum(mask)),
            "q60_abs_margin_median": (
                None if not len(q_values) else float(np.median(q_values))
            ),
            "q60_abs_margin_mean": (
                None if not len(q_values) else float(np.mean(q_values))
            ),
            "classical_abs_margin_median": (
                None if not len(c_values) else float(np.median(c_values))
            ),
            "classical_abs_margin_mean": (
                None if not len(c_values) else float(np.mean(c_values))
            ),
        }
    return result


def _class_breakdown(
    labels: np.ndarray,
    q60_scores: np.ndarray,
    classical_scores: np.ndarray,
    *,
    positive_label: str,
    negative_label: str,
) -> list[dict[str, Any]]:
    q60_correct = _predict(q60_scores) == labels
    classical_correct = _predict(classical_scores) == labels
    rows: list[dict[str, Any]] = []
    for value, name in ((-1.0, negative_label), (1.0, positive_label)):
        mask = labels == value
        rows.append(
            {
                "label_value": value,
                "label_name": name,
                "samples": int(np.sum(mask)),
                "q60_recall": float(np.mean(q60_correct[mask])),
                "classical_recall": float(np.mean(classical_correct[mask])),
                "q60_minus_classical_recall": float(
                    np.mean(q60_correct[mask]) - np.mean(classical_correct[mask])
                ),
                "q60_only_correct": int(
                    np.sum(mask & q60_correct & ~classical_correct)
                ),
                "classical_only_correct": int(
                    np.sum(mask & ~q60_correct & classical_correct)
                ),
                "both_wrong": int(np.sum(mask & ~q60_correct & ~classical_correct)),
            }
        )
    return rows


def _cell_rows(
    labels: np.ndarray,
    indices: np.ndarray,
    q60_scores: np.ndarray,
    classical_scores: np.ndarray,
    *,
    positive_label: str,
    negative_label: str,
) -> list[dict[str, Any]]:
    q60_predictions = _predict(q60_scores)
    classical_predictions = _predict(classical_scores)
    rows: list[dict[str, Any]] = []
    for position, (label, source_index, q_score, c_score, q_pred, c_pred) in enumerate(
        zip(
            labels,
            indices,
            q60_scores,
            classical_scores,
            q60_predictions,
            classical_predictions,
            strict=True,
        )
    ):
        q_correct = bool(q_pred == label)
        c_correct = bool(c_pred == label)
        if q_correct and c_correct:
            outcome = "both_correct"
        elif q_correct:
            outcome = "q60_only_correct"
        elif c_correct:
            outcome = "classical_only_correct"
        else:
            outcome = "both_wrong"
        rows.append(
            {
                "test_position": int(position),
                "source_row_index": int(source_index),
                "true_label_value": float(label),
                "true_label_name": positive_label if label > 0.0 else negative_label,
                "q60_score": float(q_score),
                "classical_score": float(c_score),
                "q60_prediction": float(q_pred),
                "classical_prediction": float(c_pred),
                "q60_correct": q_correct,
                "classical_correct": c_correct,
                "q60_abs_margin": float(abs(q_score)),
                "classical_abs_margin": float(abs(c_score)),
                "paired_outcome": outcome,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    confirmation_path = Path(args.confirmation_report)
    confirmation_report = json.loads(confirmation_path.read_text(encoding="utf-8"))
    data = reconstruct_confirmation_data(
        confirmation_report, cache_dir=Path(args.cache_dir)
    )
    source_config = data["source_config"]
    frozen_configuration = freeze._configuration_only(
        confirmation_report["selection"]["chosen"]
    )
    mappings = toy.pauli_feature_mappings(
        int(source_config["qubits"]), family="local"
    )
    frozen = _fixed_configuration_scores(
        data["encoded_train"],
        data["y_train"],
        data["encoded_test"],
        mappings=mappings,
        configuration=frozen_configuration,
        shot_intent=int(source_config["shot_intent_for_feature_ranking"]),
    )
    selected = frozen["selected"]
    expected_selected = [
        int(row["candidate_index"])
        for row in confirmation_report["confirmation"]["frozen_q60"][
            "selected_observables"
        ]
    ]
    if selected.tolist() != expected_selected:
        raise RunnerError("Reconstructed selected observables differ from confirmation")
    y_test = data["y_test"]
    q60_scores = frozen["test_scores"]
    q60_accuracy = q60._balanced_accuracy(y_test, q60_scores)
    expected_q60_accuracy = float(
        confirmation_report["confirmation"]["frozen_q60"][
            "test_balanced_accuracy"
        ]
    )
    if not np.isclose(q60_accuracy, expected_q60_accuracy, atol=1e-12, rtol=0.0):
        raise RunnerError("Reconstructed q60 score differs from confirmation")

    classical_train_scores, classical_scores = q60._ridge_scores(
        data["encoded_train"], data["encoded_test"], data["y_train"]
    )
    classical_accuracy = q60._balanced_accuracy(y_test, classical_scores)
    expected_classical_accuracy = float(
        confirmation_report["confirmation"]["classical_60bin_reference"][
            "test_balanced_accuracy"
        ]
    )
    if not np.isclose(
        classical_accuracy, expected_classical_accuracy, atol=1e-12, rtol=0.0
    ):
        raise RunnerError("Reconstructed classical score differs from confirmation")

    counts = paired_outcome_counts(y_test, q60_scores, classical_scores)
    bootstrap = paired_stratified_bootstrap(
        y_test,
        q60_scores,
        classical_scores,
        seed=int(confirmation_report["confirmation"]["actual_seed"]) + 9200,
        replicates=int(args.bootstrap_replicates),
    )

    pair_zero_configuration = {**frozen_configuration, "pair_scale": 0.0}
    pair_zero = _fixed_configuration_scores(
        data["encoded_train"],
        data["y_train"],
        data["encoded_test"],
        mappings=mappings,
        configuration=pair_zero_configuration,
        shot_intent=int(source_config["shot_intent_for_feature_ranking"]),
    )
    pair_zero_accuracy = q60._balanced_accuracy(y_test, pair_zero["test_scores"])
    selected_bases = [_basis_label(mappings[int(index)]) for index in selected]
    basis_counts = dict(sorted(Counter(selected_bases).items()))
    original_all = np.vstack(
        [frozen["train_features"], frozen["test_features"]]
    )[:, selected]
    pair_zero_all = np.vstack(
        [pair_zero["train_features"], pair_zero["test_features"]]
    )[:, selected]
    feature_pair_deltas = np.max(np.abs(original_all - pair_zero_all), axis=0)
    sensitive = feature_pair_deltas > 1e-12

    cell_rows = _cell_rows(
        y_test,
        data["test_indices"],
        q60_scores,
        classical_scores,
        positive_label=str(source_config["positive_label"]),
        negative_label=str(source_config["negative_label"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q60_shallow_frozen_error_analysis",
        "status": "pass",
        "captured_at_utc": q60.q40_validate._utc_now(),
        "execution_attempted": False,
        "provider_calls": [],
        "quantum_seconds_used": 0,
        "provenance": {
            "confirmation_report": str(confirmation_path.resolve()),
            "confirmation_report_sha256": freeze._file_sha256(confirmation_path),
            "actual_seed": int(confirmation_report["confirmation"]["actual_seed"]),
            "configuration": frozen_configuration,
            "split_hashes_reproduced": True,
            "selected_observables_reproduced": True,
            "reported_metrics_reproduced": True,
            "test_samples": len(y_test),
        },
        "paired_performance": {
            "q60_balanced_accuracy": float(q60_accuracy),
            "classical_balanced_accuracy": float(classical_accuracy),
            "q60_minus_classical_balanced_accuracy": float(
                q60_accuracy - classical_accuracy
            ),
            "outcomes": counts,
            "discordant_predictions": int(
                counts["classical_only_correct"] + counts["q60_only_correct"]
            ),
            "exact_mcnemar_two_sided_p": exact_mcnemar_p_value(
                counts["classical_only_correct"], counts["q60_only_correct"]
            ),
            "paired_stratified_bootstrap": bootstrap,
        },
        "class_breakdown": _class_breakdown(
            y_test,
            q60_scores,
            classical_scores,
            positive_label=str(source_config["positive_label"]),
            negative_label=str(source_config["negative_label"]),
        ),
        "margin_summary": _margin_summary(y_test, q60_scores, classical_scores),
        "representation_audit": {
            "candidate_observables": len(mappings),
            "selected_observables": len(selected),
            "selected_candidate_indices": [int(value) for value in selected],
            "selected_measurement_basis_counts": basis_counts,
            "selected_single_qubit_observables": int(
                sum(len(mappings[int(index)]) == 1 for index in selected)
            ),
            "selected_multiqubit_observables": int(
                sum(len(mappings[int(index)]) > 1 for index in selected)
            ),
            "selected_features_numerically_sensitive_to_pair_scale": int(
                np.sum(sensitive)
            ),
            "selected_pair_scale_sensitivity_fraction": float(np.mean(sensitive)),
            "selected_feature_pair_scale_max_abs_deltas": [
                float(value) for value in feature_pair_deltas
            ],
            "pair_scale_zero_ablation": {
                "post_hoc_diagnostic_only": True,
                "test_balanced_accuracy": float(pair_zero_accuracy),
                "delta_from_frozen_q60": float(pair_zero_accuracy - q60_accuracy),
                "selected_candidate_indices": [
                    int(value) for value in pair_zero["selected"]
                ],
                "selected_set_changed_count": len(
                    set(int(value) for value in selected)
                    ^ set(int(value) for value in pair_zero["selected"])
                ),
            },
        },
        "cell_predictions": cell_rows,
        "train_score_reproduction": {
            "q60_train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], frozen["train_scores"]
            ),
            "classical_train_balanced_accuracy": q60._balanced_accuracy(
                data["y_train"], classical_train_scores
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This is post-hoc error analysis on an already evaluated local test "
            "split. It can diagnose representation failure and guide a future "
            "pre-registered model, but it is not new held-out performance evidence, "
            "hardware evidence, biological feature attribution, or evidence of "
            "quantum advantage."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    performance = report["paired_performance"]
    outcomes = performance["outcomes"]
    audit = report["representation_audit"]
    ablation = audit["pair_scale_zero_ablation"]
    bootstrap = performance["paired_stratified_bootstrap"]
    lines = [
        "# PBMC68k q60 frozen error analysis",
        "",
        "## Reproduction contract",
        "",
        f"- Confirmation seed: {report['provenance']['actual_seed']}",
        "- Recorded split hashes reproduced: yes",
        "- Recorded selected observables reproduced: yes",
        "- Provider calls: 0",
        "",
        "## Paired held-out result",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Frozen q60 balanced accuracy | {performance['q60_balanced_accuracy']:.4f} |",
        f"| Classical balanced accuracy | {performance['classical_balanced_accuracy']:.4f} |",
        f"| q60 minus classical | {performance['q60_minus_classical_balanced_accuracy']:+.4f} |",
        f"| Paired bootstrap 95% interval | [{bootstrap['lower_95']:+.4f}, {bootstrap['upper_95']:+.4f}] |",
        f"| Exact McNemar p-value | {performance['exact_mcnemar_two_sided_p']:.4f} |",
        "",
        "## Paired outcomes",
        "",
        f"- Both correct: {outcomes['both_correct']}",
        f"- q60 only correct: {outcomes['q60_only_correct']}",
        f"- Classical only correct: {outcomes['classical_only_correct']}",
        f"- Both wrong: {outcomes['both_wrong']}",
        "",
        "## Representation audit",
        "",
        f"- Selected measurement bases: {audit['selected_measurement_basis_counts']}",
        f"- Multiqubit observables selected: {audit['selected_multiqubit_observables']}",
        "- Features numerically sensitive to pair scale: "
        f"{audit['selected_features_numerically_sensitive_to_pair_scale']} / "
        f"{audit['selected_observables']}",
        f"- `pair_scale=0` post-hoc balanced accuracy: {ablation['test_balanced_accuracy']:.4f}",
        f"- Ablation delta from frozen q60: {ablation['delta_from_frozen_q60']:+.4f}",
        "",
        "## Decision",
        "",
        "The error analysis is diagnostic only. A future model should make "
        "interaction-sensitive X/Y or multiqubit observables part of the "
        "pre-registered representation before another fresh split is opened. "
        "No hardware step is justified by this artifact.",
        "",
        f"> {report['claim_boundary']}",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-report", type=Path, default=DEFAULT_CONFIRMATION_REPORT
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.json_output, args.markdown_output):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")
    report = run(args)
    _atomic_write(
        args.json_output,
        json.dumps(report, indent=2, sort_keys=True),
    )
    _atomic_write(args.markdown_output, render_markdown(report))
    performance = report["paired_performance"]
    outcomes = performance["outcomes"]
    audit = report["representation_audit"]
    print("PBMC68k q60 frozen error analysis")
    print(f"- q60 balanced accuracy: {performance['q60_balanced_accuracy']:.4f}")
    print(
        f"- classical balanced accuracy: "
        f"{performance['classical_balanced_accuracy']:.4f}"
    )
    print(
        "- paired outcomes: "
        f"q60-only={outcomes['q60_only_correct']}, "
        f"classical-only={outcomes['classical_only_correct']}, "
        f"both-wrong={outcomes['both_wrong']}"
    )
    print(
        "- pair-sensitive selected features: "
        f"{audit['selected_features_numerically_sensitive_to_pair_scale']}/"
        f"{audit['selected_observables']}"
    )
    print(
        "- pair=0 ablation balanced accuracy: "
        f"{audit['pair_scale_zero_ablation']['test_balanced_accuracy']:.4f}"
    )
    print("- provider calls: 0")
    print(f"- JSON: {args.json_output}")
    print(f"- Markdown: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
