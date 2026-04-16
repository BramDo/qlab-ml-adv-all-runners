#!/usr/bin/env python3
"""Bounded classical screen on GSE132080 with hashed third-order interactions.

This is the explicit "last real-data escalation" for the subtle Perturb-seq
source: keep the hard within-gene guide pair, lift each sparse count vector into
hashed third-order gene interactions, and check whether comfortable classical
models still solve the task cheaply before spending more quantum time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_gse132080_utils as gse132080
from qiskit_qos_hash_streaming_genomics_runner import human_bytes

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
        raise ValueError("all third-order hash dimensions must be greater than 1")
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


@lru_cache(maxsize=None)
def triplet_position_arrays(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 3:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    triples = np.array(list(combinations(range(size), 3)), dtype=np.int64)
    return triples[:, 0], triples[:, 1], triples[:, 2]


def balanced_binary_split(
    y: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int,
    max_test_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_values = [1, -1]
    class_indices = {}
    for cls in class_values:
        idx = np.flatnonzero(y == cls)
        if len(idx) < 4:
            raise ValueError(f"class {cls} has too few rows ({len(idx)}) for a guarded split")
        rng.shuffle(idx)
        class_indices[cls] = idx

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_cap_per_class = max(1, max_train_samples // 2)
    test_cap_per_class = max(1, max_test_samples // 2)

    for cls in class_values:
        idx = class_indices[cls]
        n_train = int(round(train_fraction * len(idx)))
        n_train = max(2, min(n_train, len(idx) - 1))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        train_take = min(len(train_idx), train_cap_per_class)
        test_take = min(len(test_idx), test_cap_per_class)
        train_parts.append(np.sort(train_idx[:train_take]))
        test_parts.append(np.sort(test_idx[:test_take]))

    train_out = np.concatenate(train_parts)
    test_out = np.concatenate(test_parts)
    train_out.sort()
    test_out.sort()
    return train_out, test_out


def row_thirdorder_features(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    num_genes: int,
    feature_dim: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
    hash_repeats: int = 1,
    signed_hash: bool = False,
    activation_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    active_count_before_cap = int(len(indices))
    if active_count_before_cap < 3:
        return np.zeros(feature_dim, dtype=np.float32), {
            "active_genes_before_cap": float(active_count_before_cap),
            "active_genes_after_cap": float(active_count_before_cap),
            "third_order_events": 0.0,
        }

    if max_active_genes is not None and active_count_before_cap > max_active_genes:
        keep = np.argpartition(values, -max_active_genes)[-max_active_genes:]
        keep.sort()
        indices = indices[keep]
        values = values[keep]

    active_count_after_cap = int(len(indices))
    if active_count_after_cap < 3:
        return np.zeros(feature_dim, dtype=np.float32), {
            "active_genes_before_cap": float(active_count_before_cap),
            "active_genes_after_cap": float(active_count_after_cap),
            "third_order_events": 0.0,
        }

    ii, jj, kk = triplet_position_arrays(active_count_after_cap)
    left = indices[ii].astype(np.uint64, copy=False)
    mid = indices[jj].astype(np.uint64, copy=False)
    right = indices[kk].astype(np.uint64, copy=False)
    triplet_keys = (((left * np.uint64(num_genes) + mid) * np.uint64(num_genes) + right) + np.uint64(hash_seed)) & MASK64
    if value_mode == "binary":
        triplet_values = np.ones(len(ii), dtype=np.float64)
    elif value_mode == "log-product":
        logged = np.log1p(values.astype(np.float64, copy=False))
        triplet_values = logged[ii] * logged[jj] * logged[kk]
    else:
        raise ValueError(f"unsupported value mode: {value_mode}")

    row = np.zeros(feature_dim, dtype=np.float32)
    if signed_hash:
        repeat_count = max(1, int(hash_repeats))
        for repeat in range(repeat_count):
            repeat_offset = np.uint64((int(repeat) * int(HASH_CONST)) & int(MASK64))
            hashed = splitmix64_array((triplet_keys + repeat_offset) & MASK64)
            bins = (hashed % np.uint64(feature_dim)).astype(np.int64, copy=False)
            signs = np.where(((hashed >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float32, copy=False)
            updates = triplet_values.astype(np.float32, copy=False) * signs
            np.add.at(row, bins, updates)
        event_scale = math.sqrt(float(len(ii)) * repeat_count)
        if event_scale > 0.0:
            row /= event_scale
        if activation_scale != 1.0:
            row = np.tanh(float(activation_scale) * row).astype(np.float32, copy=False)
    else:
        hashed = splitmix64_array(triplet_keys)
        bins = (hashed % np.uint64(feature_dim)).astype(np.int64, copy=False)
        np.add.at(row, bins, triplet_values.astype(np.float32, copy=False))
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row /= norm

    return row, {
        "active_genes_before_cap": float(active_count_before_cap),
        "active_genes_after_cap": float(active_count_after_cap),
        "third_order_events": float(len(ii)),
    }


def build_thirdorder_hashed_matrix(
    x,
    *,
    feature_dim: int,
    hash_seed: int,
    value_mode: str,
    max_active_genes: int | None,
    hash_repeats: int = 1,
    signed_hash: bool = False,
    activation_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    dense = np.zeros((x.shape[0], feature_dim), dtype=np.float32)
    active_before: list[float] = []
    active_after: list[float] = []
    events: list[float] = []

    for row_idx in range(x.shape[0]):
        row = x.getrow(row_idx)
        encoded, stats = row_thirdorder_features(
            indices=row.indices,
            values=row.data,
            num_genes=x.shape[1],
            feature_dim=feature_dim,
            hash_seed=hash_seed,
            value_mode=value_mode,
            max_active_genes=max_active_genes,
            hash_repeats=hash_repeats,
            signed_hash=signed_hash,
            activation_scale=activation_scale,
        )
        dense[row_idx] = encoded
        active_before.append(stats["active_genes_before_cap"])
        active_after.append(stats["active_genes_after_cap"])
        events.append(stats["third_order_events"])

    return dense, {
        "avg_active_genes_before_cap": float(np.mean(active_before)) if active_before else 0.0,
        "avg_active_genes_after_cap": float(np.mean(active_after)) if active_after else 0.0,
        "avg_third_order_events_per_sample": float(np.mean(events)) if events else 0.0,
        "max_third_order_events_per_sample": int(max(events)) if events else 0,
        "hash_repeats": int(hash_repeats),
        "signed_hash": bool(signed_hash),
        "activation_scale": float(activation_scale),
    }


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
    hash_repeats: int,
    signed_hash: bool,
    activation_scale: float,
    seed: int,
) -> dict[str, Any]:
    train_matrix, train_stats = build_thirdorder_hashed_matrix(
        x_train,
        feature_dim=feature_dim,
        hash_seed=hash_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
        hash_repeats=hash_repeats,
        signed_hash=signed_hash,
        activation_scale=activation_scale,
    )
    test_matrix, test_stats = build_thirdorder_hashed_matrix(
        x_test,
        feature_dim=feature_dim,
        hash_seed=hash_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
        hash_repeats=hash_repeats,
        signed_hash=signed_hash,
        activation_scale=activation_scale,
    )

    rows: list[dict[str, Any]] = []

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_matrix, y_train)
    ridge_scores = np.asarray(ridge.predict(test_matrix), dtype=np.float64)
    ridge_threshold = 0.5 * (
        float(np.mean(ridge.predict(train_matrix)[y_train > 0])) + float(np.mean(ridge.predict(train_matrix)[y_train < 0]))
    )
    rows.append(
        {
            "name": f"thirdhash_ridge_d{feature_dim}",
            "family": "thirdhash_ridge",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(np.where(ridge_scores >= ridge_threshold, 1, -1) == y_test)),
            "total_bytes": int(linear_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(linear_model_bytes(feature_dim)),
        }
    )

    svc = LinearSVC(random_state=seed, max_iter=20000)
    svc.fit(train_matrix, y_train)
    rows.append(
        {
            "name": f"thirdhash_linearsvc_d{feature_dim}",
            "family": "thirdhash_linearsvc",
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
            "name": f"thirdhash_logreg_d{feature_dim}",
            "family": "thirdhash_logreg",
            "feature_dim": int(feature_dim),
            "test_accuracy": float(np.mean(logreg.predict(test_matrix) == y_test)),
            "total_bytes": int(linear_model_bytes(feature_dim)),
            "total_bytes_human": human_bytes(linear_model_bytes(feature_dim)),
        }
    )

    if float(np.min(train_matrix)) >= 0.0 and float(np.min(test_matrix)) >= 0.0:
        nb = MultinomialNB()
        nb.fit(train_matrix, y_train)
        rows.append(
            {
                "name": f"thirdhash_multinb_d{feature_dim}",
                "family": "thirdhash_multinb",
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
                "name": f"thirdhash_complementnb_d{feature_dim}",
                "family": "thirdhash_complementnb",
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
    ax.set_xlabel("Hashed third-order feature dimension")
    ax.set_ylabel("Best classical test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("GSE132080 third-order comfort screen")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen a GSE132080 guide pair with hashed third-order interactions.")
    parser.add_argument("--cache-dir", default="data_cache/gse132080")
    parser.add_argument("--positive-guide", default="POLR1D_+_28196016.23-P1_08")
    parser.add_argument("--negative-guide", default="POLR1D_+_28196016.23-P1_00")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--thirdorder-budgets", default="256,1024,4096,16384,65536")
    parser.add_argument("--max-active-genes", type=int, default=48)
    parser.add_argument("--value-mode", choices=["binary", "log-product"], default="log-product")
    parser.add_argument("--hash-repeats", type=int, default=1)
    parser.add_argument("--signed-hash", action="store_true")
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = parse_int_list(args.thirdorder_budgets)

    x, metadata, source_meta = gse132080.load_gse132080(cache_dir=args.cache_dir)
    x_pair, y_pair, pair_meta = gse132080.select_guide_pair(
        x,
        metadata,
        positive_guide=args.positive_guide,
        negative_guide=args.negative_guide,
        require_good_coverage=True,
    )

    train_idx, test_idx = balanced_binary_split(
        y_pair,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_pair[train_idx]
    y_test = y_pair[test_idx]

    ambient_feature_dim = int(comb(int(x_pair.shape[1]), 3))
    ambient_dense_weight_bytes = int(ambient_feature_dim * FLOAT64_BYTES)
    dense_projector_bytes = {
        "q20": int(ambient_feature_dim * 20 * FLOAT64_BYTES),
        "q40": int(ambient_feature_dim * 40 * FLOAT64_BYTES),
        "q60": int(ambient_feature_dim * 60 * FLOAT64_BYTES),
    }

    budget_rows: list[dict[str, Any]] = []
    for budget in budgets:
        evaluated = evaluate_budget(
            x_train,
            x_test,
            y_train,
            y_test,
            feature_dim=budget,
            hash_seed=args.hash_seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
            hash_repeats=args.hash_repeats,
            signed_hash=args.signed_hash,
            activation_scale=args.activation_scale,
            seed=args.seed,
        )
        budget_rows.append(evaluated)

    best_accuracy = max(row["best_classical_accuracy"] for row in budget_rows)
    best_smallest = min(
        [row for row in budget_rows if row["best_classical_accuracy"] == best_accuracy],
        key=lambda row: (int(row["best_classical_bytes"]), int(row["feature_dim"])),
    )

    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "positive_guide": args.positive_guide,
            "negative_guide": args.negative_guide,
            "seed": args.seed,
            "hash_seed": args.hash_seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "thirdorder_budgets": budgets,
            "max_active_genes": args.max_active_genes,
            "value_mode": args.value_mode,
            "hash_repeats": args.hash_repeats,
            "signed_hash": args.signed_hash,
            "activation_scale": args.activation_scale,
        },
        "source": {
            **source_meta,
            **pair_meta,
            "thirdorder_ambient_feature_dim": ambient_feature_dim,
            "thirdorder_ambient_dense_weight_bytes": ambient_dense_weight_bytes,
            "thirdorder_ambient_dense_weight_human": human_bytes(ambient_dense_weight_bytes),
            "dense_projector_bytes": dense_projector_bytes,
            "dense_projector_human": {key: human_bytes(value) for key, value in dense_projector_bytes.items()},
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
        "best_overall_accuracy": float(best_accuracy),
        "smallest_best_budget": {
            "feature_dim": int(best_smallest["feature_dim"]),
            "best_classical_name": best_smallest["best_classical_name"],
            "best_classical_bytes": int(best_smallest["best_classical_bytes"]),
            "best_classical_bytes_human": best_smallest["best_classical_bytes_human"],
        },
        "budget_rows": budget_rows,
        "notes": [
            "This is a bounded classical-only escalation on the hard POLR1D guide pair from GSE132080.",
            "The ambient third-order feature space is combinatorial and can reach TB-scale dense proxies, but the tested classical models still use hashed budgeted representations.",
            "If comfortable classical models remain strong here, this real-data route should be considered exhausted before spending more quantum time.",
        ],
    }

    json_out = args.json_out or "qiskit_qos_gse132080_thirdorder_screen_64x64.json"
    plot_out = args.plot_out or "qiskit_qos_gse132080_thirdorder_screen_64x64.png"
    Path(json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_plot(budget_rows, output_path=plot_out)

    print("GSE132080 third-order classical screen")
    print(f"- pair: {args.positive_guide} vs {args.negative_guide}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- third-order ambient dense classical weight memory: {human_bytes(ambient_dense_weight_bytes)}")
    for key, value in dense_projector_bytes.items():
        print(f"- dense projector {key}: {human_bytes(value)}")
    print(
        f"- best comfortable classical: {best_smallest['best_classical_name']} "
        f"acc={best_accuracy:.3f} bytes={best_smallest['best_classical_bytes_human']}"
    )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
