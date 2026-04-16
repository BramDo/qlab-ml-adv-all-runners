#!/usr/bin/env python3
"""Screen a harder PBMC68k pair with hashed gene-pair interaction features.

This is a separate extension from the plain-gene PBMC screen. It keeps the
real PBMC68k counts, but lifts each sparse cell vector into a hashed pairwise
interaction space so we can test whether the classical comfort screen becomes
materially harder before spending more quantum time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_pbmc68k_utils as pbmc
from qiskit_qos_hash_streaming_genomics_runner import benchmark_indices, human_bytes

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


FLOAT64_BYTES = 8
MASK64 = np.uint64((1 << 64) - 1)
HASH_CONST = np.uint64(0x9E3779B97F4A7C15)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    parsed = [int(item) for item in items]
    if any(item <= 1 for item in parsed):
        raise ValueError("all pairwise hash dimensions must be greater than 1")
    return parsed


def linear_model_bytes(feature_dim: int) -> int:
    return (int(feature_dim) + 1) * FLOAT64_BYTES


def nb_model_bytes(feature_dim: int) -> int:
    return (2 * int(feature_dim) + 2) * FLOAT64_BYTES


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    z = (values.astype(np.uint64) + HASH_CONST) & MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    return (z ^ (z >> np.uint64(31))) & MASK64


def row_pairwise_features(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    num_genes: int,
    feature_dim: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
) -> tuple[np.ndarray, dict[str, float]]:
    active_count_before_cap = int(len(indices))
    if active_count_before_cap < 2:
        return np.zeros(feature_dim, dtype=np.float32), {
            "active_genes_before_cap": float(active_count_before_cap),
            "active_genes_after_cap": float(active_count_before_cap),
            "pair_events": 0.0,
        }

    if max_active_genes is not None and active_count_before_cap > max_active_genes:
        keep = np.argpartition(values, -max_active_genes)[-max_active_genes:]
        keep.sort()
        indices = indices[keep]
        values = values[keep]

    active_count_after_cap = int(len(indices))
    if active_count_after_cap < 2:
        return np.zeros(feature_dim, dtype=np.float32), {
            "active_genes_before_cap": float(active_count_before_cap),
            "active_genes_after_cap": float(active_count_after_cap),
            "pair_events": 0.0,
        }

    ii, jj = np.triu_indices(active_count_after_cap, k=1)
    left = indices[ii].astype(np.uint64, copy=False)
    right = indices[jj].astype(np.uint64, copy=False)
    pair_keys = (left * np.uint64(num_genes) + right + np.uint64(hash_seed)) & MASK64
    hashed = splitmix64_array(pair_keys)
    bins = (hashed % np.uint64(feature_dim)).astype(np.int64, copy=False)

    if value_mode == "binary":
        pair_values = np.ones(len(ii), dtype=np.float64)
    elif value_mode == "log-product":
        logged = np.log1p(values.astype(np.float64, copy=False))
        pair_values = logged[ii] * logged[jj]
    else:
        raise ValueError(f"unsupported value mode: {value_mode}")

    row = np.zeros(feature_dim, dtype=np.float32)
    np.add.at(row, bins, pair_values.astype(np.float32, copy=False))
    norm = float(np.linalg.norm(row))
    if norm > 0.0:
        row /= norm

    return row, {
        "active_genes_before_cap": float(active_count_before_cap),
        "active_genes_after_cap": float(active_count_after_cap),
        "pair_events": float(len(ii)),
    }


def build_pairwise_hashed_matrix(
    x,
    *,
    feature_dim: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
) -> tuple[np.ndarray, dict[str, float]]:
    dense = np.zeros((x.shape[0], feature_dim), dtype=np.float32)
    active_before: list[float] = []
    active_after: list[float] = []
    pair_events: list[float] = []

    for row_idx in range(x.shape[0]):
        row = x.getrow(row_idx)
        encoded, stats = row_pairwise_features(
            indices=row.indices,
            values=row.data,
            num_genes=x.shape[1],
            feature_dim=feature_dim,
            hash_seed=hash_seed,
            value_mode=value_mode,
            max_active_genes=max_active_genes,
        )
        dense[row_idx] = encoded
        active_before.append(stats["active_genes_before_cap"])
        active_after.append(stats["active_genes_after_cap"])
        pair_events.append(stats["pair_events"])

    summary = {
        "avg_active_genes_before_cap": float(np.mean(active_before)) if active_before else 0.0,
        "avg_active_genes_after_cap": float(np.mean(active_after)) if active_after else 0.0,
        "avg_pair_events_per_sample": float(np.mean(pair_events)) if pair_events else 0.0,
        "max_pair_events_per_sample": int(max(pair_events)) if pair_events else 0,
    }
    return dense, summary


def evaluate_budget(
    x_train,
    x_test,
    y_train,
    y_test,
    *,
    feature_dim: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
    seed: int,
) -> dict[str, Any]:
    train_matrix, train_stats = build_pairwise_hashed_matrix(
        x_train,
        feature_dim=feature_dim,
        hash_seed=hash_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
    )
    test_matrix, test_stats = build_pairwise_hashed_matrix(
        x_test,
        feature_dim=feature_dim,
        hash_seed=hash_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
    )

    rows: list[dict[str, Any]] = []

    svc = LinearSVC(random_state=seed, max_iter=20000)
    svc.fit(train_matrix, y_train)
    rows.append(
        {
            "name": f"pairhash_linearsvc_d{feature_dim}",
            "family": "pairhash_linearsvc",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(svc.predict(test_matrix) == y_test)),
            "total_bytes": int(linear_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(linear_model_bytes(feature_dim)),
        }
    )

    logreg = LogisticRegression(max_iter=5000, random_state=seed, solver="liblinear")
    logreg.fit(train_matrix, y_train)
    rows.append(
        {
            "name": f"pairhash_logreg_d{feature_dim}",
            "family": "pairhash_logreg",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(logreg.predict(test_matrix) == y_test)),
            "total_bytes": int(linear_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(linear_model_bytes(feature_dim)),
        }
    )

    nb = MultinomialNB()
    nb.fit(train_matrix, y_train)
    rows.append(
        {
            "name": f"pairhash_multinb_d{feature_dim}",
            "family": "pairhash_multinb",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(nb.predict(test_matrix) == y_test)),
            "total_bytes": int(nb_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(nb_model_bytes(feature_dim)),
        }
    )

    cnb = ComplementNB()
    cnb.fit(train_matrix, y_train)
    rows.append(
        {
            "name": f"pairhash_complementnb_d{feature_dim}",
            "family": "pairhash_complementnb",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(cnb.predict(test_matrix) == y_test)),
            "total_bytes": int(nb_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(nb_model_bytes(feature_dim)),
        }
    )

    best_accuracy = max(row["test_accuracy"] for row in rows)
    best_smallest = min(
        [row for row in rows if row["test_accuracy"] == best_accuracy],
        key=lambda row: (int(row["total_bytes"]), row["name"]),
    )
    return {
        "feature_dim": int(feature_dim),
        "train_stats": train_stats,
        "test_stats": test_stats,
        "best_classical_accuracy": float(best_accuracy),
        "best_classical_name": best_smallest["name"],
        "best_classical_bytes": int(best_smallest["total_bytes"]),
        "best_classical_bytes_human": best_smallest["total_bytes_human"],
        "results": rows,
    }


def render_plot(budget_rows: list[dict[str, Any]], *, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.asarray([row["feature_dim"] for row in budget_rows], dtype=np.int64)
    y = np.asarray([row["best_classical_accuracy"] for row in budget_rows], dtype=np.float64)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hashed pairwise feature dimension")
    ax.set_ylabel("Best classical test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("PBMC68k pairwise comfort screen")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen a PBMC68k pair with hashed gene-pair features.")
    parser.add_argument("--cache-dir", default="data_cache/pbmc68k")
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--pairwise-budgets", default="256,1024,4096,16384,65536")
    parser.add_argument("--max-active-genes", type=int, default=256)
    parser.add_argument("--value-mode", choices=["binary", "log-product"], default="log-product")
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = parse_int_list(args.pairwise_budgets)

    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=args.cache_dir)
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    train_idx, test_idx = benchmark_indices(
        x_pair.shape[0],
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_pair[train_idx]
    y_test = y_pair[test_idx]

    budget_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for budget in budgets:
        evaluated = evaluate_budget(
            x_train,
            x_test,
            y_train,
            y_test,
            feature_dim=int(budget),
            hash_seed=args.hash_seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
            seed=args.seed,
        )
        budget_rows.append(
            {
                "feature_dim": int(evaluated["feature_dim"]),
                "best_classical_accuracy": float(evaluated["best_classical_accuracy"]),
                "best_classical_name": evaluated["best_classical_name"],
                "best_classical_bytes": int(evaluated["best_classical_bytes"]),
                "best_classical_bytes_human": evaluated["best_classical_bytes_human"],
                "train_stats": evaluated["train_stats"],
                "test_stats": evaluated["test_stats"],
            }
        )
        all_results.extend(evaluated["results"])

    pairwise_feature_dim = int(comb(int(x_pair.shape[1]), 2))
    pairwise_dense_weight_bytes = int(pairwise_feature_dim * FLOAT64_BYTES)
    projector_bytes = {
        f"q{qubits}": int(pairwise_feature_dim * qubits * FLOAT64_BYTES)
        for qubits in (20, 40, 60)
    }

    best_acc = max(row["best_classical_accuracy"] for row in budget_rows)
    best_smallest = min(
        [row for row in budget_rows if row["best_classical_accuracy"] == best_acc],
        key=lambda row: (int(row["best_classical_bytes"]), row["best_classical_name"]),
    )

    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "seed": args.seed,
            "hash_seed": args.hash_seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "pairwise_budgets": budgets,
            "max_active_genes": args.max_active_genes,
            "value_mode": args.value_mode,
        },
        "source": {
            **source_meta,
            **pair_meta,
            "pairwise_ambient_feature_dim": pairwise_feature_dim,
            "pairwise_ambient_dense_weight_bytes": pairwise_dense_weight_bytes,
            "pairwise_ambient_dense_weight_human": human_bytes(pairwise_dense_weight_bytes),
            "pairwise_ambient_projector_bytes": projector_bytes,
            "pairwise_ambient_projector_human": {key: human_bytes(value) for key, value in projector_bytes.items()},
        },
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
        "match_summary": {
            "best_classical_accuracy": float(best_acc),
            "best_classical_name": best_smallest["best_classical_name"],
            "minimum_bytes_for_best_accuracy": int(best_smallest["best_classical_bytes"]),
            "minimum_bytes_for_best_accuracy_human": best_smallest["best_classical_bytes_human"],
        },
        "budget_rows": budget_rows,
        "results": all_results,
        "notes": [
            "This is a hashed pairwise coexpression screen, not a full materialization of all gene pairs.",
            "The ambient pairwise memory numbers report the dense full-space classical proxy over all gene pairs.",
            "max_active_genes limits the per-cell interaction expansion to keep the screen laptop-feasible.",
        ],
    }

    stem = (
        "qiskit_qos_pbmc68k_pairwise_screen_"
        f"{args.max_train_samples}x{args.max_test_samples}"
    )
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(budget_rows, output_path=plot_out)

    print("PBMC68k pairwise screen")
    print(f"- pair: {args.positive_label} vs {args.negative_label}")
    print(f"- best_classical={best_acc:.3f} via {best_smallest['best_classical_name']} at {best_smallest['best_classical_bytes_human']}")
    print(f"- pairwise ambient dense weight: {human_bytes(pairwise_dense_weight_bytes)}")
    print(
        "- pairwise ambient projector bytes: "
        f"q20={human_bytes(projector_bytes['q20'])}, "
        f"q40={human_bytes(projector_bytes['q40'])}, "
        f"q60={human_bytes(projector_bytes['q60'])}"
    )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
