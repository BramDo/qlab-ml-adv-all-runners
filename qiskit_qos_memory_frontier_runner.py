#!/usr/bin/env python3
"""Compare quantum and classical accuracy-memory frontiers on the QOS sources.

This extension keeps the earlier toy/scaling/benchmark scripts intact and adds a
single place to answer the paper-style question:

"How much predictive performance do we get per unit of model memory?"

The current quantum side can either be loaded from an existing scaling artifact
or omitted. The classical side sweeps explicit memory budgets:
- hashing + SGD as a memory-bounded streaming-like reference
- TF-IDF + TruncatedSVD(k) + RidgeClassifier as a compressed dense baseline
- raw TF-IDF upper bounds for context

Important caveat:
- To stay comparable with the current quantum text pipeline, this runner fits
  TF-IDF/SVD on the full corpus before applying the train/test split. That
  mirrors the existing scaling artifact behavior, but it is not a clean
  train-only preprocessing benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_classical_benchmark as classical_bench
import qiskit_qos_memory_report as memory_report
import qiskit_qos_scaling_runner as scaling

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


FLOAT64_BYTES = 8


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    out = [int(item) for item in items]
    if any(item <= 0 for item in out):
        raise ValueError("all budget values must be positive")
    return out


def human_bytes(num_bytes: int) -> str:
    return memory_report.human_bytes(int(num_bytes))


def tfidf_idf_bytes(vocab_size: int) -> int:
    return int(vocab_size) * FLOAT64_BYTES


def svd_projector_bytes(vocab_size: int, svd_dim: int) -> int:
    return int(vocab_size) * int(svd_dim) * FLOAT64_BYTES


def linear_model_bytes(feature_dim: int) -> int:
    return classical_bench._linear_model_bytes(int(feature_dim))


def nb_model_bytes(feature_dim: int) -> int:
    return classical_bench._nb_model_bytes(int(feature_dim))


def pareto_frontier_indices(points: list[dict[str, Any]]) -> list[int]:
    indexed = list(enumerate(points))
    indexed.sort(key=lambda item: (float(item[1]["memory_axis_bytes"]), -float(item[1]["test_accuracy"])))
    frontier: list[int] = []
    best_acc = -np.inf
    for idx, point in indexed:
        acc = float(point["test_accuracy"])
        if acc > best_acc + 1e-12:
            frontier.append(idx)
            best_acc = acc
    frontier.sort(key=lambda idx: float(points[idx]["memory_axis_bytes"]))
    return frontier


def load_quantum_points(path: str, *, allowed_qubits: set[int] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    config = dict(payload["config"])
    source = dict(payload["source"])
    points: list[dict[str, Any]] = []
    for run in payload["runs"]:
        q = int(run["num_qubits"])
        if allowed_qubits is not None and q not in allowed_qubits:
            continue
        mem = memory_report.estimate_run_memory(run=dict(run), config=config, source=source)
        qmem = mem["quantum_memory"]
        axis_bytes = int(qmem["total_model_bytes_conceptual"])
        points.append(
            {
                "name": f"quantum_q{q}",
                "side": "quantum",
                "family": "quantum",
                "memory_axis_bytes": axis_bytes,
                "memory_axis_human": human_bytes(axis_bytes),
                "memory_label": f"{human_bytes(axis_bytes)} + {q} logical qubits",
                "logical_qubits": q,
                "test_accuracy": float(run["test_accuracy_quantum"]),
                "train_accuracy": float(run["train_accuracy_quantum"]),
                "runtime_seconds": float(run["elapsed_seconds"]),
                "readout_feature_count": int(run["readout_feature_count"]),
                "quantum_head_feature_count": int(run["quantum_head_feature_count"]),
                "source_artifact": str(path),
            }
        )
    return points, {"config": config, "source": source}


def load_text_source(args: argparse.Namespace) -> tuple[scaling.SourceBundle, pd.Series, np.ndarray]:
    source_args = argparse.Namespace(
        source=args.source,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
        svd_components=args.svd_components,
        seed=args.seed,
    )
    source = scaling.load_source(source_args)
    texts, labels = classical_bench._load_texts(args)
    if len(texts) != len(source.x) or not np.array_equal(labels.astype(float), source.y):
        raise RuntimeError("Text labels/features mismatch; source loading is inconsistent.")
    return source, texts, labels


def classical_budget_points(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source, texts, labels = load_text_source(args)
    train_idx, test_idx = classical_bench._benchmark_indices(
        source.x,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(1, args.tfidf_ngram_max),
    )
    x_tfidf_all = vectorizer.fit_transform(texts)
    raw_vocab_size = int(x_tfidf_all.shape[1])
    idf_bytes = tfidf_idf_bytes(raw_vocab_size)
    x_train_tfidf = x_tfidf_all[train_idx]
    x_test_tfidf = x_tfidf_all[test_idx]

    points: list[dict[str, Any]] = []

    for budget in parse_int_list(args.hash_budgets):
        hashing = HashingVectorizer(
            lowercase=True,
            n_features=budget,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, args.tfidf_ngram_max),
        )
        x_hash_all = hashing.transform(texts)
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=args.seed, max_iter=5000, tol=1e-4)
        clf.fit(x_hash_all[train_idx], y_train)
        acc = float(np.mean(clf.predict(x_hash_all[test_idx]) == y_test))
        total_bytes = linear_model_bytes(budget)
        points.append(
            {
                "name": f"hash_sgd_{budget}",
                "side": "classical",
                "family": "hashing",
                "memory_axis_bytes": total_bytes,
                "memory_axis_human": human_bytes(total_bytes),
                "memory_label": human_bytes(total_bytes),
                "feature_dim": int(budget),
                "test_accuracy": acc,
                "train_accuracy": float(np.mean(clf.predict(x_hash_all[train_idx]) == y_train)),
                "memory_breakdown": {
                    "model_bytes": int(total_bytes),
                    "feature_extractor_bytes": 0,
                    "total_bytes": int(total_bytes),
                },
            }
        )

    for budget in parse_int_list(args.svd_budgets):
        svd_k = min(int(budget), raw_vocab_size - 1, len(texts) - 1)
        if svd_k < 1:
            continue
        svd = TruncatedSVD(n_components=svd_k, random_state=args.seed)
        x_dense_all = svd.fit_transform(x_tfidf_all)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(x_dense_all[train_idx], y_train)
        acc = float(np.mean(clf.predict(x_dense_all[test_idx]) == y_test))
        model_bytes = linear_model_bytes(svd_k)
        feature_bytes = idf_bytes + svd_projector_bytes(raw_vocab_size, svd_k)
        total_bytes = feature_bytes + model_bytes
        points.append(
            {
                "name": f"svd_ridge_{svd_k}",
                "side": "classical",
                "family": "svd_ridge",
                "memory_axis_bytes": int(total_bytes),
                "memory_axis_human": human_bytes(total_bytes),
                "memory_label": human_bytes(total_bytes),
                "feature_dim": int(svd_k),
                "test_accuracy": acc,
                "train_accuracy": float(np.mean(clf.predict(x_dense_all[train_idx]) == y_train)),
                "memory_breakdown": {
                    "model_bytes": int(model_bytes),
                    "feature_extractor_bytes": int(feature_bytes),
                    "total_bytes": int(total_bytes),
                },
                "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            }
        )

    nb_raw = MultinomialNB()
    nb_raw.fit(x_train_tfidf, y_train)
    nb_total = idf_bytes + nb_model_bytes(raw_vocab_size)
    points.append(
        {
            "name": "raw_tfidf_multinb",
            "side": "classical",
            "family": "raw_upper_bound",
            "memory_axis_bytes": int(nb_total),
            "memory_axis_human": human_bytes(nb_total),
            "memory_label": human_bytes(nb_total),
            "feature_dim": int(raw_vocab_size),
            "test_accuracy": float(np.mean(nb_raw.predict(x_test_tfidf) == y_test)),
            "train_accuracy": float(np.mean(nb_raw.predict(x_train_tfidf) == y_train)),
            "memory_breakdown": {
                "model_bytes": int(nb_model_bytes(raw_vocab_size)),
                "feature_extractor_bytes": int(idf_bytes),
                "total_bytes": int(nb_total),
            },
        }
    )

    svc_raw = LinearSVC(random_state=args.seed)
    svc_raw.fit(x_train_tfidf, y_train)
    svc_total = idf_bytes + linear_model_bytes(raw_vocab_size)
    points.append(
        {
            "name": "raw_tfidf_linearsvc",
            "side": "classical",
            "family": "raw_upper_bound",
            "memory_axis_bytes": int(svc_total),
            "memory_axis_human": human_bytes(svc_total),
            "memory_label": human_bytes(svc_total),
            "feature_dim": int(raw_vocab_size),
            "test_accuracy": float(np.mean(svc_raw.predict(x_test_tfidf) == y_test)),
            "train_accuracy": float(np.mean(svc_raw.predict(x_train_tfidf) == y_train)),
            "memory_breakdown": {
                "model_bytes": int(linear_model_bytes(raw_vocab_size)),
                "feature_extractor_bytes": int(idf_bytes),
                "total_bytes": int(svc_total),
            },
        }
    )

    split = {
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
    }
    source_meta = {
        "source_name": source.source_name,
        "rows": int(source.metadata["rows"]),
        "raw_feature_dim": int(source.metadata["raw_feature_dim"]),
        "reduced_feature_dim": int(source.metadata["reduced_feature_dim"]),
    }
    return points, {"split": split, "source": source_meta}


def render_frontier_plot(points: list[dict[str, Any]], frontier_indices: list[int], *, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Model memory proxy (bytes, log scale)")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)

    styles = {
        "quantum": {"color": "#006d77", "marker": "o", "label": "quantum"},
        "hashing": {"color": "#6a994e", "marker": "s", "label": "classical hashing"},
        "svd_ridge": {"color": "#bc6c25", "marker": "^", "label": "classical SVD+ridge"},
        "raw_upper_bound": {"color": "#ae2012", "marker": "D", "label": "classical raw upper"},
    }

    used_labels: set[str] = set()
    for point in points:
        style = styles[point["family"]]
        label = style["label"] if style["label"] not in used_labels else None
        used_labels.add(style["label"])
        axes[0].scatter(
            float(point["memory_axis_bytes"]),
            float(point["test_accuracy"]),
            color=style["color"],
            marker=style["marker"],
            s=64,
            alpha=0.85,
            label=label,
        )

    frontier = [points[idx] for idx in frontier_indices]
    frontier.sort(key=lambda row: float(row["memory_axis_bytes"]))
    axes[1].plot(
        [float(point["memory_axis_bytes"]) for point in frontier],
        [float(point["test_accuracy"]) for point in frontier],
        color="#264653",
        linewidth=1.5,
        alpha=0.6,
    )
    for point in frontier:
        style = styles[point["family"]]
        axes[1].scatter(
            float(point["memory_axis_bytes"]),
            float(point["test_accuracy"]),
            color=style["color"],
            marker=style["marker"],
            s=70,
        )

    for idx in frontier_indices:
        point = points[idx]
        label = point["name"]
        if point["side"] == "quantum":
            label = f"q={point['logical_qubits']}"
        axes[0].annotate(
            label,
            (float(point["memory_axis_bytes"]), float(point["test_accuracy"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].annotate(
            label,
            (float(point["memory_axis_bytes"]), float(point["test_accuracy"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    axes[0].set_title("All accuracy-memory points")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Pareto frontier")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a quantum-vs-classical accuracy-memory frontier artifact.")
    parser.add_argument(
        "--source",
        default="20ng-atheism-vs-space",
        choices=["20ng-atheism-vs-space", "20ng-graphics-vs-baseball"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-test-samples", type=int, default=16)
    parser.add_argument("--tfidf-max-features", type=int, default=20000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--svd-budgets", default="16,32,64,128,256")
    parser.add_argument("--hash-budgets", default="64,128,256,512,1024,2048,4096,8192")
    parser.add_argument("--quantum-scaling-json", help="Optional quantum scaling artifact to load into the frontier.")
    parser.add_argument("--quantum-qubits", help="Optional comma-separated subset of quantum q values to keep.")
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    quantum_qubits = set(parse_int_list(args.quantum_qubits)) if args.quantum_qubits else None
    quantum_points: list[dict[str, Any]] = []
    quantum_meta: dict[str, Any] | None = None
    if args.quantum_scaling_json:
        quantum_points, quantum_meta = load_quantum_points(args.quantum_scaling_json, allowed_qubits=quantum_qubits)

    classical_points, classical_meta = classical_budget_points(args)
    all_points = quantum_points + classical_points
    frontier_indices = pareto_frontier_indices(all_points)
    for idx, point in enumerate(all_points):
        point["pareto_frontier"] = idx in frontier_indices

    payload: dict[str, Any] = {
        "config": {
            "source": args.source,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "tfidf_max_features": args.tfidf_max_features,
            "tfidf_min_df": args.tfidf_min_df,
            "tfidf_ngram_max": args.tfidf_ngram_max,
            "svd_components": args.svd_components,
            "svd_budgets": parse_int_list(args.svd_budgets),
            "hash_budgets": parse_int_list(args.hash_budgets),
            "quantum_scaling_json": args.quantum_scaling_json,
            "quantum_qubits": None if quantum_qubits is None else sorted(quantum_qubits),
        },
        "split": classical_meta["split"],
        "source": classical_meta["source"],
        "quantum_points": quantum_points,
        "classical_points": classical_points,
        "pareto_frontier": [all_points[idx] for idx in frontier_indices],
        "notes": [
            "The frontier x-axis uses a single scalar memory proxy in bytes.",
            "Quantum points use the conceptual classical-sidecar byte estimate from qiskit_qos_memory_report plus a separate logical-qubit label.",
            "Classical SVD points include both the TF-IDF idf vector and the dense SVD projector as memory-bearing preprocessing state.",
            "This runner mirrors the current quantum text pipeline by fitting TF-IDF/SVD on the full corpus before splitting; it is intended for in-repo frontier comparison, not for publication-grade train-only validation.",
        ],
    }
    if quantum_meta is not None:
        payload["quantum_artifact_meta"] = quantum_meta

    stem = args.source.replace("-", "_")
    json_out = args.json_out or f"qiskit_qos_memory_frontier_{stem}_{args.max_train_samples}x{args.max_test_samples}.json"
    plot_out = args.plot_out or f"qiskit_qos_memory_frontier_{stem}_{args.max_train_samples}x{args.max_test_samples}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_frontier_plot(all_points, frontier_indices, output_path=plot_out)

    print("QOS memory frontier")
    print(f"- source: {args.source}")
    print(f"- train/test: {classical_meta['split']['train_size']}/{classical_meta['split']['test_size']}")
    print("- frontier:")
    for point in payload["pareto_frontier"]:
        extra = f" ({point['memory_label']})"
        if point["side"] == "quantum":
            extra = f" ({point['memory_label']})"
        print(
            "  "
            f"{point['name']:<24} "
            f"acc={point['test_accuracy']:.3f} "
            f"mem={point['memory_axis_human']}{extra if point['side'] == 'quantum' else ''}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
