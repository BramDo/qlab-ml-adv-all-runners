#!/usr/bin/env python3
"""Classical difficulty screen for subtle within-gene guide pairs in GSE132080."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

import qiskit_qos_gse132080_utils as gse132080
from qiskit_qos_hash_streaming_genomics_runner import human_bytes

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
            raise ValueError("pair specs must look like guide_a:guide_b")
        left, right = chunk.split(":", 1)
        out.append((left.strip(), right.strip()))
    return out


def linear_model_bytes(feature_dim: int) -> int:
    return (int(feature_dim) + 1) * FLOAT64_BYTES


def nb_model_bytes(feature_dim: int) -> int:
    return (2 * int(feature_dim) + 2) * FLOAT64_BYTES


def selector_bytes(feature_dim: int) -> int:
    return int(feature_dim) * INT32_BYTES


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
    train_idx, test_idx = balanced_binary_split(
        y_pair,
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

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        feat_dim = min(int(budget), int(x_train.shape[1]))
        selector = SelectKBest(score_func=chi2, k=feat_dim)
        x_train_sel = selector.fit_transform(x_train, y_train_01)
        x_test_sel = selector.transform(x_test)

        svc = LinearSVC(random_state=seed, max_iter=20000)
        svc.fit(x_train_sel, y_train)
        rows.append(
            {
                "name": f"chi2_linearsvc_k{feat_dim}",
                "family": "chi2_linearsvc",
                "feature_dim": int(feat_dim),
                "test_accuracy": float(np.mean(svc.predict(x_test_sel) == y_test)),
                "total_bytes": int(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
                "total_bytes_human": human_bytes(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        logreg = LogisticRegression(max_iter=5000, random_state=seed, solver="liblinear")
        logreg.fit(x_train_sel, y_train)
        rows.append(
            {
                "name": f"chi2_logreg_k{feat_dim}",
                "family": "chi2_logreg",
                "feature_dim": int(feat_dim),
                "test_accuracy": float(np.mean(logreg.predict(x_test_sel) == y_test)),
                "total_bytes": int(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
                "total_bytes_human": human_bytes(linear_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        nb = MultinomialNB()
        nb.fit(x_train_sel, y_train)
        rows.append(
            {
                "name": f"chi2_multinb_k{feat_dim}",
                "family": "chi2_multinb",
                "feature_dim": int(feat_dim),
                "test_accuracy": float(np.mean(nb.predict(x_test_sel) == y_test)),
                "total_bytes": int(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
                "total_bytes_human": human_bytes(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

        cnb = ComplementNB()
        cnb.fit(x_train_sel, y_train)
        rows.append(
            {
                "name": f"chi2_complementnb_k{feat_dim}",
                "family": "chi2_complementnb",
                "feature_dim": int(feat_dim),
                "test_accuracy": float(np.mean(cnb.predict(x_test_sel) == y_test)),
                "total_bytes": int(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
                "total_bytes_human": human_bytes(nb_model_bytes(feat_dim) + selector_bytes(feat_dim)),
            }
        )

    raw_svc = LinearSVC(random_state=seed, max_iter=20000)
    raw_svc.fit(x_train, y_train)
    rows.append(
        {
            "name": "raw_linearsvc_full",
            "family": "raw_linearsvc",
            "feature_dim": int(x_train.shape[1]),
            "test_accuracy": float(np.mean(raw_svc.predict(x_test) == y_test)),
            "total_bytes": int(linear_model_bytes(x_train.shape[1])),
            "total_bytes_human": human_bytes(linear_model_bytes(x_train.shape[1])),
        }
    )

    raw_cnb = ComplementNB()
    raw_cnb.fit(x_train, y_train)
    rows.append(
        {
            "name": "raw_complementnb_full",
            "family": "raw_complementnb",
            "feature_dim": int(x_train.shape[1]),
            "test_accuracy": float(np.mean(raw_cnb.predict(x_test) == y_test)),
            "total_bytes": int(nb_model_bytes(x_train.shape[1])),
            "total_bytes_human": human_bytes(nb_model_bytes(x_train.shape[1])),
        }
    )

    best_accuracy = max(row["test_accuracy"] for row in rows)
    best_smallest = min(
        [row for row in rows if row["test_accuracy"] == best_accuracy],
        key=lambda row: (int(row["total_bytes"]), row["name"]),
    )
    return {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_positive": int(np.sum(y_train > 0)),
        "train_negative": int(np.sum(y_train < 0)),
        "test_positive": int(np.sum(y_test > 0)),
        "test_negative": int(np.sum(y_test < 0)),
        "best_classical_accuracy": float(best_accuracy),
        "best_classical_name": best_smallest["name"],
        "best_classical_bytes": int(best_smallest["total_bytes"]),
        "best_classical_bytes_human": best_smallest["total_bytes_human"],
        "all_points": rows,
    }


def render_plot(rows: list[dict[str, Any]], *, output_path: str) -> None:
    labels = [row["pair_label"] for row in rows]
    scores = [float(row["best_classical_accuracy"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, scores)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Best classical test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("GSE132080 subtle within-gene guide screening")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen subtle within-gene guide pairs in GSE132080.")
    parser.add_argument("--cache-dir", default="data_cache/gse132080")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument(
        "--pairs",
        default=(
            "ATP5E_-_57607036.23-P1P2_01:ATP5E_-_57607036.23-P1P2_00,"
            "POLR1D_+_28196016.23-P1_08:POLR1D_+_28196016.23-P1_00,"
            "CDC23_-_137548987.23-P1P2_08:CDC23_-_137548987.23-P1P2_04,"
            "GINS1_-_25388381.23-P1P2_08:GINS1_-_25388381.23-P1P2_06,"
            "DUT_+_48624411.23-P1P2_01:DUT_+_48624411.23-P1P2_08,"
            "HSPE1_+_198365089.23-P1P2_01:HSPE1_+_198365089.23-P1P2_03,"
            "COX11_+_53045977.23-P1P2_00:COX11_+_53045977.23-P1P2_03"
        ),
        help="Comma-separated guide_a:guide_b specs",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = parse_int_list(args.budgets)
    pair_specs = parse_pairs(args.pairs)
    if not pair_specs:
        raise RuntimeError("No GSE132080 guide pairs requested")

    x, metadata, source_meta = gse132080.load_gse132080(cache_dir=args.cache_dir)
    rows: list[dict[str, Any]] = []
    for positive_guide, negative_guide in pair_specs:
        x_pair, y_pair, pair_meta = gse132080.select_guide_pair(
            x,
            metadata,
            positive_guide=positive_guide,
            negative_guide=negative_guide,
            require_good_coverage=True,
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
                "pair_label": f"{positive_guide} vs {negative_guide}",
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
            "pairs": [{"positive_guide": a, "negative_guide": b} for a, b in pair_specs],
        },
        "source": source_meta,
        "pairs": rows,
        "notes": [
            "This source uses subtle within-gene guide-vs-guide differences from the Perturb-seq dataset GSE132080.",
            "Each pair compares two guides targeting the same gene with a small activity gap, rather than a broad lineage or stim-vs-ctrl shift.",
            "Lower best_classical_accuracy means a more promising source before spending quantum time.",
        ],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    if args.plot_out:
        render_plot(rows, output_path=args.plot_out)


if __name__ == "__main__":
    main()
