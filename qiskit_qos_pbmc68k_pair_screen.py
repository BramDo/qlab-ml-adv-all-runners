#!/usr/bin/env python3
"""Screen PBMC68k binary pairs for classical difficulty.

Goal: quickly find a harder real source before spending any quantum time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_pbmc68k_utils as pbmc
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


def parse_pairs(value: str) -> list[tuple[str, str]] | None:
    chunks = [item.strip() for item in value.split(",") if item.strip()]
    if not chunks:
        return None
    out: list[tuple[str, str]] = []
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError("pair specs must look like LabelA:LabelB")
        left, right = chunk.split(":", 1)
        out.append((left.strip(), right.strip()))
    return out


def linear_model_bytes(feature_dim: int) -> int:
    return (int(feature_dim) + 1) * FLOAT64_BYTES


def nb_model_bytes(feature_dim: int) -> int:
    return (2 * int(feature_dim) + 2) * FLOAT64_BYTES


def selector_bytes(feature_dim: int) -> int:
    return int(feature_dim) * INT32_BYTES


def evaluate_pair(
    x_pair,
    y_pair,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int,
    max_test_samples: int,
    budgets: list[int],
) -> dict[str, Any]:
    train_idx, test_idx = benchmark_indices(
        x_pair.shape[0],
        seed=seed,
        train_fraction=train_fraction,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_pair[train_idx]
    y_test = y_pair[test_idx]
    y_train_01 = (y_train > 0).astype(int)

    best_rows: list[dict[str, Any]] = []
    for budget in budgets:
        feat_dim = min(int(budget), int(x_train.shape[1]))
        selector = SelectKBest(score_func=chi2, k=feat_dim)
        x_train_sel = selector.fit_transform(x_train, y_train_01)
        x_test_sel = selector.transform(x_test)

        svc = LinearSVC(random_state=seed, max_iter=20000)
        svc.fit(x_train_sel, y_train)
        best_rows.append(
            {
                "name": f"chi2_linearsvc_k{feat_dim}",
                "accuracy": float(np.mean(svc.predict(x_test_sel) == y_test)),
                "bytes": int(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        logreg = LogisticRegression(max_iter=5000, random_state=seed, solver="liblinear")
        logreg.fit(x_train_sel, y_train)
        best_rows.append(
            {
                "name": f"chi2_logreg_k{feat_dim}",
                "accuracy": float(np.mean(logreg.predict(x_test_sel) == y_test)),
                "bytes": int(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        nb = MultinomialNB()
        nb.fit(x_train_sel, y_train)
        best_rows.append(
            {
                "name": f"chi2_multinb_k{feat_dim}",
                "accuracy": float(np.mean(nb.predict(x_test_sel) == y_test)),
                "bytes": int(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        cnb = ComplementNB()
        cnb.fit(x_train_sel, y_train)
        best_rows.append(
            {
                "name": f"chi2_complementnb_k{feat_dim}",
                "accuracy": float(np.mean(cnb.predict(x_test_sel) == y_test)),
                "bytes": int(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

    raw_svc = LinearSVC(random_state=seed, max_iter=20000)
    raw_svc.fit(x_train, y_train)
    best_rows.append(
        {
            "name": "raw_linearsvc_full",
            "accuracy": float(np.mean(raw_svc.predict(x_test) == y_test)),
            "bytes": int(linear_model_bytes(x_train.shape[1])),
        }
    )
    raw_cnb = ComplementNB()
    raw_cnb.fit(x_train, y_train)
    best_rows.append(
        {
            "name": "raw_complementnb_full",
            "accuracy": float(np.mean(raw_cnb.predict(x_test) == y_test)),
            "bytes": int(nb_model_bytes(x_train.shape[1])),
        }
    )

    best_acc = max(row["accuracy"] for row in best_rows)
    best_smallest = min(
        [row for row in best_rows if row["accuracy"] == best_acc],
        key=lambda row: (int(row["bytes"]), row["name"]),
    )
    return {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "best_classical_accuracy": float(best_acc),
        "best_classical_name": best_smallest["name"],
        "best_classical_bytes": int(best_smallest["bytes"]),
        "best_classical_bytes_human": human_bytes(best_smallest["bytes"]),
        "all_points": best_rows,
    }


def render_plot(rows: list[dict[str, Any]], *, output_path: str) -> None:
    labels = [row["pair_label"] for row in rows]
    scores = [float(row["best_classical_accuracy"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, scores)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Best classical test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("PBMC68k pair screening")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen PBMC68k binary pairs for classical difficulty.")
    parser.add_argument("--cache-dir", default="data_cache/pbmc68k")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument(
        "--pairs",
        default=(
            "CD8+ Cytotoxic T:CD8+/CD45RA+ Naive Cytotoxic,"
            "CD4+/CD45RO+ Memory:CD4+/CD45RA+/CD25- Naive T,"
            "CD4+/CD25 T Reg:CD4+/CD45RO+ Memory,"
            "CD14+ Monocyte:Dendritic"
        ),
        help="Comma-separated LabelA:LabelB specs",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = parse_int_list(args.budgets)
    pair_specs = parse_pairs(args.pairs)
    if not pair_specs:
        raise RuntimeError("No PBMC pairs requested")

    x, labels, meta = pbmc.load_pbmc68k(cache_dir=args.cache_dir)
    rows: list[dict[str, Any]] = []
    for positive_label, negative_label in pair_specs:
        x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
            x,
            labels,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        evaluation = evaluate_pair(
            x_pair,
            y_pair,
            seed=args.seed,
            train_fraction=args.train_fraction,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            budgets=budgets,
        )
        rows.append(
            {
                "pair_label": f"{positive_label} vs {negative_label}",
                **pair_meta,
                **evaluation,
            }
        )

    rows.sort(key=lambda row: (float(row["best_classical_accuracy"]), int(row["best_classical_bytes"])))
    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "budgets": budgets,
            "pairs": [{"positive": a, "negative": b} for a, b in pair_specs],
        },
        "source": meta,
        "pairs": rows,
        "notes": [
            "Lower best_classical_accuracy means a harder candidate pair for the current classical comfort screen.",
            "This is a screening pass before adding the source to the quantum runners.",
        ],
    }
    stem = f"qiskit_qos_pbmc68k_pair_screen_{args.max_train_samples}x{args.max_test_samples}"
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, output_path=plot_out)

    print("PBMC68k pair screen")
    for row in rows:
        print(
            f"- {row['pair_label']}: "
            f"best_classical={row['best_classical_accuracy']:.3f} "
            f"via {row['best_classical_name']} "
            f"at {row['best_classical_bytes_human']}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
