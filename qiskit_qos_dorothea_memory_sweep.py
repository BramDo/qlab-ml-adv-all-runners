#!/usr/bin/env python3
"""Classical memory sweep for the UCI Dorothea benchmark.

This stays separate from the earlier classical benchmark so the text workflows
remain unchanged. The goal is to answer a more paper-like question:

"How much classical model memory is needed to match the quantum accuracy?"
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

import qiskit_qos_dorothea_utils as dorothea_utils
import qiskit_qos_scaling_runner as scaling
import qiskit_qos_toy_model as toy

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


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def benchmark_indices(
    x_dense: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = len(x_dense)
    n_train = int(round(train_fraction * total))
    n_train = max(2, min(int(n_train), total - 1))
    train_idx, test_idx = toy.split_train_test(x_dense, n_train=n_train, rng=rng)
    if max_train_samples is not None and len(train_idx) > max_train_samples:
        keep_train = rng.choice(len(train_idx), size=max_train_samples, replace=False)
        keep_train.sort()
        train_idx = train_idx[keep_train]
    if max_test_samples is not None and len(test_idx) > max_test_samples:
        keep_test = rng.choice(len(test_idx), size=max_test_samples, replace=False)
        keep_test.sort()
        test_idx = test_idx[keep_test]
    return train_idx, test_idx


def load_quantum_targets(path: str, *, allowed_qubits: set[int] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    targets: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        q = int(run["num_qubits"])
        if allowed_qubits is not None and q not in allowed_qubits:
            continue
        targets.append(
            {
                "num_qubits": q,
                "test_accuracy_quantum": float(run["test_accuracy_quantum"]),
                "test_accuracy_classical_scaling": float(run["test_accuracy_classical"]),
                "readout_feature_count": int(run["readout_feature_count"]),
                "quantum_head_feature_count": int(run["quantum_head_feature_count"]),
            }
        )
    targets.sort(key=lambda item: item["num_qubits"])
    return targets


def linear_model_bytes(feature_dim: int) -> int:
    return (int(feature_dim) + 1) * FLOAT64_BYTES


def nb_model_bytes(feature_dim: int) -> int:
    return (2 * int(feature_dim) + 2) * FLOAT64_BYTES


def selector_bytes(feature_dim: int) -> int:
    return int(feature_dim) * INT32_BYTES


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


def render_plot(
    classical_rows: list[dict[str, Any]],
    quantum_targets: list[dict[str, Any]],
    *,
    output_path: str,
) -> None:
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

    for target in quantum_targets:
        ax.axhline(
            float(target["test_accuracy_quantum"]),
            linestyle="--",
            alpha=0.5,
            label=f"quantum q={target['num_qubits']} ({target['test_accuracy_quantum']:.3f})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Classical model + selector memory (bytes)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Dorothea classical memory sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classical memory budgets on UCI Dorothea.")
    parser.add_argument("--dorothea-cache-dir", default="data_cache/dorothea")
    parser.add_argument("--dorothea-train-only", action="store_true")
    parser.add_argument("--dorothea-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=128)
    parser.add_argument("--svd-components", type=int, default=512, help="Must match the scaling source config when using --quantum-scaling-json")
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--quantum-scaling-json", required=True)
    parser.add_argument("--quantum-qubits", default="10,20")
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qubit_filter = set(parse_int_list(args.quantum_qubits))
    quantum_targets = load_quantum_targets(args.quantum_scaling_json, allowed_qubits=qubit_filter)
    if not quantum_targets:
        raise RuntimeError("No matching quantum runs found in --quantum-scaling-json")

    source_args = argparse.Namespace(
        source="dorothea-uci",
        dorothea_cache_dir=args.dorothea_cache_dir,
        dorothea_train_only=args.dorothea_train_only,
        dorothea_balance=args.dorothea_balance,
        svd_components=args.svd_components,
        seed=args.seed,
    )
    dense_source = scaling.load_source(source_args)
    x_sparse, y, sparse_meta = dorothea_utils.load_dorothea_sparse(
        data_dir=args.dorothea_cache_dir,
        merge_valid=not args.dorothea_train_only,
    )
    if args.dorothea_balance:
        x_sparse, y, balance_meta = dorothea_utils.balance_binary_dataset(x_sparse, y, seed=args.seed)
        sparse_meta = {
            **sparse_meta,
            **balance_meta,
            "rows": int(x_sparse.shape[0]),
            "nnz": int(x_sparse.nnz),
            "density": float(x_sparse.nnz / (x_sparse.shape[0] * x_sparse.shape[1])),
            "positive_count": int(np.sum(y > 0)),
            "negative_count": int(np.sum(y < 0)),
        }
    train_idx, test_idx = benchmark_indices(
        dense_source.x,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    x_train = x_sparse[train_idx]
    x_test = x_sparse[test_idx]
    y_train = y[train_idx].astype(int)
    y_test = y[test_idx].astype(int)
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

    match_summary: list[dict[str, Any]] = []
    for target in quantum_targets:
        candidates = [row for row in rows if float(row["test_accuracy"]) >= float(target["test_accuracy_quantum"])]
        if candidates:
            best = min(candidates, key=lambda row: (int(row["total_bytes"]), row["family"]))
            match_summary.append(
                {
                    "num_qubits": int(target["num_qubits"]),
                    "quantum_test_accuracy": float(target["test_accuracy_quantum"]),
                    "matched_by": best["name"],
                    "matched_family": best["family"],
                    "matched_feature_dim": int(best["feature_dim"]),
                    "minimum_classical_bytes_to_match": int(best["total_bytes"]),
                    "minimum_classical_bytes_to_match_human": best["total_bytes_human"],
                }
            )
        else:
            match_summary.append(
                {
                    "num_qubits": int(target["num_qubits"]),
                    "quantum_test_accuracy": float(target["test_accuracy_quantum"]),
                    "matched_by": None,
                    "matched_family": None,
                    "matched_feature_dim": None,
                    "minimum_classical_bytes_to_match": None,
                    "minimum_classical_bytes_to_match_human": None,
                }
            )

    payload = {
        "config": {
            "source": "dorothea-uci",
            "dorothea_cache_dir": args.dorothea_cache_dir,
            "dorothea_train_only": bool(args.dorothea_train_only),
            "dorothea_balance": bool(args.dorothea_balance),
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "svd_components": args.svd_components,
            "budgets": budgets,
            "quantum_scaling_json": args.quantum_scaling_json,
            "quantum_qubits": sorted(qubit_filter),
        },
        "source": {
            **sparse_meta,
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
        "quantum_targets": quantum_targets,
        "classical_points": rows,
        "match_summary": match_summary,
        "notes": [
            "This runner measures classical model-plus-selector memory on raw Dorothea sparse features.",
            "The quantum side is loaded from an existing scaling artifact so the split and target accuracy stay fixed.",
            "This is an empirical memory sweep, not a formal lower bound.",
        ],
    }

    stem = f"qiskit_qos_dorothea_memory_sweep_{args.max_train_samples}x{args.max_test_samples}"
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, quantum_targets, output_path=plot_out)

    print("Dorothea classical memory sweep")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- raw feature dim: {x_train.shape[1]}")
    for match in match_summary:
        if match["matched_by"] is None:
            print(
                f"- q={match['num_qubits']}: "
                f"no classical point matched quantum_acc={match['quantum_test_accuracy']:.3f}"
            )
        else:
            print(
                f"- q={match['num_qubits']}: "
                f"quantum_acc={match['quantum_test_accuracy']:.3f} "
                f"matched by {match['matched_by']} at {match['minimum_classical_bytes_to_match_human']}"
            )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
