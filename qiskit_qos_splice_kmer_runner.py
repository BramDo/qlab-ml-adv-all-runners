#!/usr/bin/env python3
"""Split-aware Splice k-mer quantum benchmark.

This is the first real genomics k-mer route for the workspace. It keeps the
earlier Dorothea and 20NG experiments intact and reports both:

- actual observed k-mer feature count from the dataset
- full ambient 4^k feature-space size and the dense-memory implication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

from qiskit_qos_dorothea_chi2_quantum_runner import benchmark_indices, run_quantum_on_split
import qiskit_qos_scaling_runner as scaling
import qiskit_qos_splice_kmer_utils as splice_utils


def parse_run_specs(value: str) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("each run spec must look like chi2_k:num_qubits, e.g. 64:10")
        left, right = chunk.split(":", 1)
        chi2_k = int(left.strip())
        num_qubits = int(right.strip())
        if chi2_k <= 0 or num_qubits <= 0:
            raise ValueError("chi2_k and num_qubits must be positive")
        specs.append((chi2_k, num_qubits))
    if not specs:
        raise ValueError("no valid run specs parsed")
    return specs


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split-aware Splice k-mer quantum benchmarks.")
    parser.add_argument("--k", type=int, default=19, help="k-mer size; k=19 already implies ~2.2 TB ambient dense weight memory")
    parser.add_argument("--binary", action="store_true", default=True, help="Keep only EI vs IE")
    parser.add_argument("--min-samples", type=int, default=1, help="Drop observed k-mers that appear in fewer than this many samples")
    parser.add_argument("--runs", default="64:10,64:20", help="Comma-separated chi2_k:num_qubits specs")
    parser.add_argument("--readout-shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--encoder", default="ridge", choices=["block", "pca", "ridge", "lda"])
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--execution-mode", default="sampler-sim", choices=["statevector", "sampler-sim", "ibm-hardware"])
    parser.add_argument("--backend-name")
    parser.add_argument("--simulator-method", default="matrix_product_state", choices=["automatic", "statevector", "matrix_product_state"])
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--readout-mitigation", action="store_true")
    parser.add_argument("--cal-shots", type=int, default=512)
    parser.add_argument("--extra-error-suppression", action="store_true")
    parser.add_argument("--dd-sequence", default="XY4")
    parser.add_argument("--twirl-randomizations", type=int, default=8)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def render_plot(rows: list[dict[str, Any]], *, output_path: str) -> None:
    labels = [f"k={row['chi2_k']},q={row['num_qubits']}" for row in rows]
    x = np.arange(len(rows))
    quantum = [float(row["test_accuracy_quantum"]) for row in rows]
    ridge = [float(row["test_accuracy_classical_ridge"]) for row in rows]
    svc = [float(row["test_accuracy_classical_linearsvc"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, quantum, marker="o", label="quantum")
    ax.plot(x, ridge, marker="s", label="ridge baseline")
    ax.plot(x, svc, marker="^", label="LinearSVC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Splice k-mer quantum runs")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_specs = parse_run_specs(args.runs)
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

    dense_for_split = np.zeros((x_sparse.shape[0], 1), dtype=np.float64)
    train_idx, test_idx = benchmark_indices(
        dense_for_split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train_sparse = x_sparse[train_idx]
    x_test_sparse = x_sparse[test_idx]
    y_train = labels[train_idx].astype(np.float64)
    y_test = labels[test_idx].astype(np.float64)
    y_train_01 = (y_train > 0).astype(int)

    execution_config = scaling.build_execution_config(args)
    rows: list[dict[str, Any]] = []
    json_out = args.json_out or f"qiskit_qos_splice_kmer_k{args.k}.json"
    plot_out = args.plot_out or f"qiskit_qos_splice_kmer_k{args.k}.png"

    for index, (chi2_k, num_qubits) in enumerate(run_specs):
        selector = SelectKBest(score_func=chi2, k=min(chi2_k, int(x_train_sparse.shape[1])))
        x_train_sel = selector.fit_transform(x_train_sparse, y_train_01)
        x_test_sel = selector.transform(x_test_sparse)
        x_train_dense = np.asarray(x_train_sel.toarray(), dtype=np.float64)
        x_test_dense = np.asarray(x_test_sel.toarray(), dtype=np.float64)

        result = run_quantum_on_split(
            x_train_raw=x_train_dense,
            x_test_raw=x_test_dense,
            y_train=y_train,
            y_test=y_test,
            num_qubits=num_qubits,
            readout_shots=args.readout_shots,
            seed=args.seed + index,
            encoder_method=args.encoder,
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
        )
        rows.append(
            {
                "chi2_k": int(chi2_k),
                "num_qubits": int(num_qubits),
                "selected_feature_count": int(selector.get_support(indices=True).shape[0]),
                "ambient_feature_dim": int(source_meta["ambient_feature_dim"]),
                "ambient_dense_weight_bytes": int(source_meta["ambient_dense_weight_bytes"]),
                "ambient_dense_weight_human": human_bytes(int(source_meta["ambient_dense_weight_bytes"])),
                "observed_feature_dim_after_min_samples": int(source_meta["observed_feature_dim_after_min_samples"]),
                "observed_dense_weight_bytes": int(source_meta["observed_feature_dim_after_min_samples"] * 8),
                "observed_dense_weight_human": human_bytes(int(source_meta["observed_feature_dim_after_min_samples"] * 8)),
                **result,
            }
        )

    payload = {
        "config": {
            "source": "splice-kmer-openml",
            "k": int(args.k),
            "binary": bool(args.binary),
            "min_samples": int(args.min_samples),
            "runs": [{"chi2_k": int(k), "num_qubits": int(q)} for k, q in run_specs],
            "readout_shots": args.readout_shots,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "encoder": args.encoder,
            "quantum_head": args.quantum_head,
            "readout_family": args.readout_family,
            "execution_mode": args.execution_mode,
            "backend_name": args.backend_name,
            "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
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
        "runs": rows,
        "notes": [
            "ambient_dense_weight_bytes assumes a full dense classical weight vector over the canonical 4^k k-mer space",
            "observed_dense_weight_bytes uses only the observed vocabulary after min_samples filtering",
            "chi2 selection is fit on the training split only",
        ],
    }
    Path(json_out).write_text(json.dumps(payload, indent=2))
    plot_out = args.plot_out or f"qiskit_qos_splice_kmer_k{args.k}.png"
    render_plot(rows, output_path=plot_out)

    print("Splice k-mer quantum runner")
    print(f"- k: {args.k}")
    print(f"- binary rows: {source_meta['rows']}")
    print(f"- ambient feature dim: {source_meta['ambient_feature_dim']}")
    print(f"- ambient dense weight memory: {human_bytes(int(source_meta['ambient_dense_weight_bytes']))}")
    print(f"- observed feature dim after min_samples: {source_meta['observed_feature_dim_after_min_samples']}")
    print(f"- observed dense weight memory: {human_bytes(int(source_meta['observed_feature_dim_after_min_samples'] * 8))}")
    for row in rows:
        print(
            f"- k={row['chi2_k']}, q={row['num_qubits']}: "
            f"quantum={row['test_accuracy_quantum']:.3f} "
            f"ridge={row['test_accuracy_classical_ridge']:.3f} "
            f"linearsvc={row['test_accuracy_classical_linearsvc']:.3f}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
