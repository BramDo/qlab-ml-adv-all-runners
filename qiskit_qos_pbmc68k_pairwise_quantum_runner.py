#!/usr/bin/env python3
"""Quantum screen for a harder PBMC68k binary task with pairwise gene hashing."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from math import comb
from pathlib import Path

import numpy as np

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_utils as pbmc
from qiskit_qos_run_logger import log_run_event
import qiskit_qos_scaling_runner as scaling

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


def render_plot(rows: list[dict[str, object]], *, output_path: str) -> None:
    qubits = [int(row["num_qubits"]) for row in rows]
    quantum = [float(row["test_accuracy_quantum"]) for row in rows]
    ridge = [float(row["test_accuracy_classical_hashed_ridge"]) for row in rows]
    svc = [float(row["test_accuracy_classical_hashed_linearsvc"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(qubits, quantum, marker="o", label="quantum")
    ax.plot(qubits, ridge, marker="s", label="hashed ridge")
    ax.plot(qubits, svc, marker="^", label="hashed LinearSVC")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("PBMC68k pairwise quantum screen")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PBMC68k pairwise quantum benchmark.")
    parser.add_argument("--cache-dir", default="data_cache/pbmc68k")
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--qubits", default="10", help="Comma-separated qubit counts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--max-active-genes", type=int, default=256)
    parser.add_argument("--value-mode", choices=["binary", "log-product"], default="log-product")
    parser.add_argument("--readout-shots", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
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


def main() -> None:
    args = parse_args()
    qubits = genomics_runner.parse_int_list(args.qubits)
    execution_config = scaling.build_execution_config(args)
    log_run_event(
        "pbmc_pairwise_runner_start",
        cache_dir=args.cache_dir,
        qubits=qubits,
        execution_mode=args.execution_mode,
        backend_name=args.backend_name,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        query_batch_size=args.query_batch_size,
        readout_shots=args.readout_shots,
        feature_mapping_limit=execution_config.feature_mapping_limit,
        runtime_submit_batch_size=execution_config.runtime_submit_batch_size,
    )

    log_run_event(
        "pbmc_pairwise_data_load_start",
        cache_dir=args.cache_dir,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=args.cache_dir)
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    log_run_event(
        "pbmc_pairwise_data_load_done",
        source_rows=int(x.shape[0]),
        source_cols=int(x.shape[1]),
        pair_rows=int(x_pair.shape[0]),
        positive_count=int(np.sum(y_pair > 0.0)),
        negative_count=int(np.sum(y_pair < 0.0)),
    )
    train_idx, test_idx = genomics_runner.benchmark_indices(
        x_pair.shape[0],
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        labels=y_pair,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_pair[train_idx].astype(np.float64)
    y_test = y_pair[test_idx].astype(np.float64)
    log_run_event(
        "pbmc_pairwise_split_ready",
        total_rows=int(x_pair.shape[0]),
        train_size=int(len(train_idx)),
        test_size=int(len(test_idx)),
        train_positive=int(np.sum(y_train > 0.0)),
        train_negative=int(np.sum(y_train < 0.0)),
        test_positive=int(np.sum(y_test > 0.0)),
        test_negative=int(np.sum(y_test < 0.0)),
    )

    ambient_feature_dim = int(comb(int(x_pair.shape[1]), 2))
    ambient_dense_weight_bytes = int(ambient_feature_dim * 8)

    rows: list[dict[str, object]] = []
    json_out = args.json_out or f"qiskit_qos_pbmc68k_pairwise_quantum_q{args.qubits.replace(',', '_')}.json"
    plot_out = args.plot_out or f"qiskit_qos_pbmc68k_pairwise_quantum_q{args.qubits.replace(',', '_')}.png"

    run_start = time.perf_counter()
    for run_index, num_qubits in enumerate(qubits):
        log_run_event(
            "pbmc_pairwise_run_start",
            run_index=run_index,
            num_qubits=int(num_qubits),
            pairwise_hash_seed=int(args.hash_seed + run_index),
        )
        encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_train,
            feature_dim=int(num_qubits),
            hash_seed=int(args.hash_seed + run_index),
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_test,
            feature_dim=int(num_qubits),
            hash_seed=int(args.hash_seed + run_index),
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        log_run_event(
            "pbmc_pairwise_encoded_ready",
            run_index=run_index,
            num_qubits=int(num_qubits),
            train_stats=train_stats,
            test_stats=test_stats,
        )
        result = genomics_runner.run_quantum_on_encoded_split(
            encoded_train=np.asarray(encoded_train, dtype=np.float64),
            encoded_test=np.asarray(encoded_test, dtype=np.float64),
            y_train=y_train,
            y_test=y_test,
            readout_shots=args.readout_shots,
            seed=int(args.seed + run_index),
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
            query_batch_size=args.query_batch_size,
        )
        log_run_event(
            "pbmc_pairwise_run_done",
            run_index=run_index,
            num_qubits=int(num_qubits),
            test_accuracy_quantum=result["test_accuracy_quantum"],
            test_balanced_accuracy_quantum=result["test_balanced_accuracy_quantum"],
            test_accuracy_classical_hashed_ridge=result["test_accuracy_classical_hashed_ridge"],
            test_accuracy_classical_hashed_linearsvc=result["test_accuracy_classical_hashed_linearsvc"],
            execution_metadata=result["execution_metadata"],
        )
        rows.append(
            {
                "num_qubits": int(num_qubits),
                "pairwise_hash_seed": int(args.hash_seed + run_index),
                "max_active_genes": int(args.max_active_genes),
                "value_mode": args.value_mode,
                "ambient_feature_dim": ambient_feature_dim,
                "ambient_dense_weight_bytes": ambient_dense_weight_bytes,
                "ambient_dense_weight_human": genomics_runner.human_bytes(ambient_dense_weight_bytes),
                "streaming_sketch_state_bytes": int(8 * (num_qubits + max(num_qubits - 1, 0))),
                "streaming_sketch_state_human": genomics_runner.human_bytes(8 * (num_qubits + max(num_qubits - 1, 0))),
                "encoded_sample_bytes": int(8 * num_qubits),
                "encoded_sample_human": genomics_runner.human_bytes(8 * num_qubits),
                "dense_encoder_matrix_bytes_avoided": int(ambient_feature_dim * num_qubits * 8),
                "dense_encoder_matrix_human_avoided": genomics_runner.human_bytes(ambient_feature_dim * num_qubits * 8),
                "train_stats": train_stats,
                "test_stats": test_stats,
                **result,
            }
        )

    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "qubits": qubits,
            "seed": args.seed,
            "hash_seed": args.hash_seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "max_active_genes": args.max_active_genes,
            "value_mode": args.value_mode,
            "readout_shots": args.readout_shots,
            "query_batch_size": args.query_batch_size,
            "quantum_head": args.quantum_head,
            "readout_family": args.readout_family,
            "execution_mode": args.execution_mode,
            "backend_name": args.backend_name,
            "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
        },
        "source": {
            **source_meta,
            **pair_meta,
            "pairwise_ambient_feature_dim": ambient_feature_dim,
            "pairwise_ambient_dense_weight_bytes": ambient_dense_weight_bytes,
            "pairwise_ambient_dense_weight_human": genomics_runner.human_bytes(ambient_dense_weight_bytes),
        },
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "sampling_strategy": "label-stratified-balanced-capped",
            "class_balance_train": {
                "positive": int(np.sum(y_train > 0.0)),
                "negative": int(np.sum(y_train < 0.0)),
            },
            "class_balance_test": {
                "positive": int(np.sum(y_test > 0.0)),
                "negative": int(np.sum(y_test < 0.0)),
            },
        },
        "runs": rows,
        "elapsed_seconds": float(time.perf_counter() - run_start),
        "notes": [
            "The host expands each PBMC cell into hashed gene-pair interactions and only then feeds a q-dimensional encoded vector into the Qiskit sketch path.",
            "query_batch_size batches logical samples at the Pauli-readout stage; the raw pairwise ambient feature size stays on the host side.",
            "dense_encoder_matrix_bytes_avoided is the full dense q x pairwise_ambient_feature_dim matrix that this hashed route avoids storing.",
        ],
    }

    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, output_path=plot_out)
    log_run_event(
        "pbmc_pairwise_artifacts_written",
        json_out=json_out,
        plot_out=plot_out,
        elapsed_seconds=float(time.perf_counter() - run_start),
        run_count=len(rows),
    )

    print("PBMC68k pairwise quantum runner")
    print(f"- pair: {args.positive_label} vs {args.negative_label}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- pairwise ambient dense classical weight memory: {genomics_runner.human_bytes(ambient_dense_weight_bytes)}")
    for row in rows:
        print(
            f"- q={row['num_qubits']}: "
            f"quantum={row['test_accuracy_quantum']:.3f} "
            f"quantum_bal={row['test_balanced_accuracy_quantum']:.3f} "
            f"hashed_ridge={row['test_accuracy_classical_hashed_ridge']:.3f} "
            f"ridge_bal={row['test_balanced_accuracy_classical_hashed_ridge']:.3f} "
            f"hashed_svc={row['test_accuracy_classical_hashed_linearsvc']:.3f} "
            f"svc_bal={row['test_balanced_accuracy_classical_hashed_linearsvc']:.3f} "
            f"sketch={row['streaming_sketch_state_human']} "
            f"avoided_matrix={row['dense_encoder_matrix_human_avoided']} "
            f"query_batches={row['execution_metadata']['query']['batch_count']}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_run_event(
            "pbmc_pairwise_runner_failed",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise
