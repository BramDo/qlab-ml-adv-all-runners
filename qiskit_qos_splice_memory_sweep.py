#!/usr/bin/env python3
"""Classical memory sweep for the Splice k-mer benchmark.

This stays separate from the earlier Splice quantum runners. The goal is to
measure the smallest classical model budget that matches a chosen quantum
target accuracy on the exact same train/test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_splice_kmer_utils as splice_utils
from qiskit_qos_hash_streaming_genomics_runner import benchmark_indices, human_bytes

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


FLOAT64_BYTES = 8
INT32_BYTES = 4


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    values = [int(item) for item in items]
    if any(item <= 0 for item in values):
        raise ValueError("all sweep values must be positive")
    return values


def linear_model_bytes(feature_dim: int) -> int:
    return (int(feature_dim) + 1) * FLOAT64_BYTES


def nb_model_bytes(feature_dim: int) -> int:
    return (2 * int(feature_dim) + 2) * FLOAT64_BYTES


def selector_bytes(feature_dim: int) -> int:
    return int(feature_dim) * INT32_BYTES


def load_quantum_target(path: str, *, target_qubits: int | None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    runs = payload.get("runs", [])
    if not runs:
        raise RuntimeError("No runs found in --quantum-json")
    if target_qubits is None:
        if len(runs) != 1:
            raise RuntimeError("Multiple runs found; specify --target-qubits")
        run = runs[0]
    else:
        matches = [run for run in runs if int(run.get("num_qubits", -1)) == int(target_qubits)]
        if not matches:
            raise RuntimeError(f"No run found for --target-qubits={target_qubits}")
        run = matches[0]
    return {
        "num_qubits": int(run["num_qubits"]),
        "quantum_test_accuracy": float(run["test_accuracy_quantum"]),
        "quantum_head_feature_count": int(run["quantum_head_feature_count"]),
        "readout_feature_count": int(run["readout_feature_count"]),
        "ambient_dense_weight_human": str(run["ambient_dense_weight_human"]),
        "streaming_sketch_state_human": str(run["streaming_sketch_state_human"]),
        "encoded_sample_human": str(run["encoded_sample_human"]),
    }


def record_result(
    rows: list[dict[str, Any]],
    *,
    family: str,
    feature_dim: int,
    accuracy: float,
    model_bytes: int,
    selector_feature_dim: int,
) -> None:
    selection_bytes = selector_bytes(selector_feature_dim)
    total_bytes = int(model_bytes) + int(selection_bytes)
    rows.append(
        {
            "name": f"{family}_k{feature_dim}",
            "family": family,
            "feature_dim": int(feature_dim),
            "test_accuracy": float(accuracy),
            "model_bytes": int(model_bytes),
            "selector_bytes": int(selection_bytes),
            "total_bytes": int(total_bytes),
            "total_bytes_human": human_bytes(total_bytes),
        }
    )


def render_plot(classical_rows: list[dict[str, Any]], quantum_target: dict[str, Any], *, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    families = sorted({row["family"] for row in classical_rows})
    markers = {
        "chi2_logreg": "o",
        "chi2_linearsvc": "s",
        "chi2_multinb": "^",
        "raw_linearsvc": "D",
        "raw_multinb": "v",
    }

    for family in families:
        family_rows = [row for row in classical_rows if row["family"] == family]
        family_rows.sort(key=lambda row: int(row["total_bytes"]))
        ax.plot(
            [int(row["total_bytes"]) for row in family_rows],
            [float(row["test_accuracy"]) for row in family_rows],
            marker=markers.get(family, "o"),
            label=family,
        )

    ax.axhline(
        float(quantum_target["quantum_test_accuracy"]),
        linestyle="--",
        alpha=0.6,
        label=f"quantum q={quantum_target['num_qubits']} ({quantum_target['quantum_test_accuracy']:.3f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Classical model + selector memory (bytes)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Splice classical memory sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classical memory budgets on Splice k-mer features.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=8)
    parser.add_argument("--budgets", default="8,16,32,64,128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--quantum-json", required=True)
    parser.add_argument("--target-qubits", type=int, default=20)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantum_target = load_quantum_target(args.quantum_json, target_qubits=args.target_qubits)
    if args.target_accuracy is not None:
        quantum_target["quantum_test_accuracy"] = float(args.target_accuracy)

    sequences, labels, seq_meta = splice_utils.load_splice_sequences(binary=args.binary)
    x_sparse, _, kmer_meta = splice_utils.compute_kmer_features(sequences, k=args.k)
    x_sparse, kept = splice_utils.filter_features_by_min_samples(x_sparse, min_samples=args.min_samples)
    kept = np.asarray(kept, dtype=np.int64)
    source_meta = {
        **seq_meta,
        **kmer_meta,
        "observed_feature_dim_after_min_samples": int(x_sparse.shape[1]),
        "min_samples": int(args.min_samples),
    }

    train_idx, test_idx = benchmark_indices(
        len(sequences),
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train = x_sparse[train_idx]
    x_test = x_sparse[test_idx]
    y_train = labels[train_idx].astype(int)
    y_test = labels[test_idx].astype(int)
    y_train_01 = (y_train > 0).astype(int)

    budgets = parse_int_list(args.budgets)
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        k = min(int(budget), int(x_train.shape[1]))
        selector = SelectKBest(score_func=chi2, k=k)
        x_train_sel = selector.fit_transform(x_train, y_train_01)
        x_test_sel = selector.transform(x_test)

        logreg = LogisticRegression(max_iter=5000, random_state=args.seed, solver="liblinear")
        logreg.fit(x_train_sel, y_train)
        record_result(
            rows,
            family="chi2_logreg",
            feature_dim=k,
            accuracy=float(np.mean(logreg.predict(x_test_sel) == y_test)),
            model_bytes=linear_model_bytes(k),
            selector_feature_dim=k,
        )

        svc = LinearSVC(random_state=args.seed)
        svc.fit(x_train_sel, y_train)
        record_result(
            rows,
            family="chi2_linearsvc",
            feature_dim=k,
            accuracy=float(np.mean(svc.predict(x_test_sel) == y_test)),
            model_bytes=linear_model_bytes(k),
            selector_feature_dim=k,
        )

        nb = MultinomialNB()
        nb.fit(x_train_sel, y_train)
        record_result(
            rows,
            family="chi2_multinb",
            feature_dim=k,
            accuracy=float(np.mean(nb.predict(x_test_sel) == y_test)),
            model_bytes=nb_model_bytes(k),
            selector_feature_dim=k,
        )

    raw_svc = LinearSVC(random_state=args.seed)
    raw_svc.fit(x_train, y_train)
    rows.append(
        {
            "name": "raw_linearsvc_full",
            "family": "raw_linearsvc",
            "feature_dim": int(x_train.shape[1]),
            "test_accuracy": float(np.mean(raw_svc.predict(x_test) == y_test)),
            "model_bytes": int(linear_model_bytes(x_train.shape[1])),
            "selector_bytes": 0,
            "total_bytes": int(linear_model_bytes(x_train.shape[1])),
            "total_bytes_human": human_bytes(linear_model_bytes(x_train.shape[1])),
        }
    )

    raw_nb = MultinomialNB()
    raw_nb.fit(x_train, y_train)
    rows.append(
        {
            "name": "raw_multinb_full",
            "family": "raw_multinb",
            "feature_dim": int(x_train.shape[1]),
            "test_accuracy": float(np.mean(raw_nb.predict(x_test) == y_test)),
            "model_bytes": int(nb_model_bytes(x_train.shape[1])),
            "selector_bytes": 0,
            "total_bytes": int(nb_model_bytes(x_train.shape[1])),
            "total_bytes_human": human_bytes(nb_model_bytes(x_train.shape[1])),
        }
    )

    rows.sort(key=lambda row: (int(row["total_bytes"]), -float(row["test_accuracy"]), row["family"]))
    candidates = [row for row in rows if float(row["test_accuracy"]) >= float(quantum_target["quantum_test_accuracy"])]
    best = min(candidates, key=lambda row: (int(row["total_bytes"]), row["family"])) if candidates else None

    match_summary = {
        "num_qubits": int(quantum_target["num_qubits"]),
        "quantum_test_accuracy": float(quantum_target["quantum_test_accuracy"]),
        "matched_by": None if best is None else best["name"],
        "matched_family": None if best is None else best["family"],
        "matched_feature_dim": None if best is None else int(best["feature_dim"]),
        "minimum_classical_bytes_to_match": None if best is None else int(best["total_bytes"]),
        "minimum_classical_bytes_to_match_human": None if best is None else best["total_bytes_human"],
    }

    payload = {
        "config": {
            "source": "splice-kmer-openml",
            "k": int(args.k),
            "binary": bool(args.binary),
            "min_samples": int(args.min_samples),
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "budgets": budgets,
            "quantum_json": args.quantum_json,
            "target_qubits": args.target_qubits,
            "target_accuracy": args.target_accuracy,
        },
        "source": source_meta,
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "class_balance_train": {
                "positive": int(np.sum(y_train > 0)),
                "negative": int(np.sum(y_train < 0)),
            },
            "class_balance_test": {
                "positive": int(np.sum(y_test > 0)),
                "negative": int(np.sum(y_test < 0)),
            },
        },
        "quantum_target": quantum_target,
        "classical_points": rows,
        "match_summary": match_summary,
        "notes": [
            "This runner measures classical model-plus-selector memory on raw Splice sparse k-mer features.",
            "The quantum side is loaded from an existing artifact so the target accuracy stays fixed.",
            "This is an empirical memory sweep, not a formal lower bound.",
        ],
    }

    stem = f"qiskit_qos_splice_memory_sweep_k{args.k}_{args.max_train_samples}x{args.max_test_samples}"
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, quantum_target, output_path=plot_out)

    print("Splice classical memory sweep")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- raw observed feature dim: {x_train.shape[1]}")
    print(f"- quantum target: q={quantum_target['num_qubits']} acc={quantum_target['quantum_test_accuracy']:.3f}")
    if best is None:
        print("- no classical point matched the quantum target")
    else:
        print(
            f"- matched by {best['name']} "
            f"at {best['total_bytes_human']} "
            f"with test_acc={best['test_accuracy']:.3f}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
