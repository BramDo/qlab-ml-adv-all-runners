#!/usr/bin/env python3
"""Quantum screen for the hard GSE132080 guide pair with hashed third-order interactions."""

from __future__ import annotations

import argparse
import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np

import qiskit_qos_gse132080_thirdorder_screen as thirdorder
import qiskit_qos_gse132080_utils as gse132080
import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_scaling_runner as scaling
import qiskit_qos_toy_model as toy

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
    ax.set_title("GSE132080 third-order quantum screen")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GSE132080 third-order quantum benchmark.")
    parser.add_argument("--cache-dir", default="data_cache/gse132080")
    parser.add_argument("--positive-guide", default="POLR1D_+_28196016.23-P1_08")
    parser.add_argument("--negative-guide", default="POLR1D_+_28196016.23-P1_00")
    parser.add_argument("--qubits", default="10,20", help="Comma-separated qubit counts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--max-active-genes", type=int, default=48)
    parser.add_argument("--value-mode", choices=["binary", "log-product"], default="log-product")
    parser.add_argument("--hash-repeats", type=int, default=1)
    parser.add_argument("--signed-hash", action="store_true")
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--encoder", default="direct", choices=["direct", "block", "pca", "ridge", "lda"])
    parser.add_argument("--pre-hash-dim", type=int)
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

    x, metadata, source_meta = gse132080.load_gse132080(cache_dir=args.cache_dir)
    x_pair, y_pair, pair_meta = gse132080.select_guide_pair(
        x,
        metadata,
        positive_guide=args.positive_guide,
        negative_guide=args.negative_guide,
        require_good_coverage=True,
    )

    train_idx, test_idx = thirdorder.balanced_binary_split(
        y_pair,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_pair[train_idx].astype(np.float64)
    y_test = y_pair[test_idx].astype(np.float64)

    ambient_feature_dim = int(comb(int(x_pair.shape[1]), 3))
    ambient_dense_weight_bytes = int(ambient_feature_dim * 8)

    rows: list[dict[str, object]] = []
    json_out = args.json_out or f"qiskit_qos_gse132080_thirdorder_quantum_q{args.qubits.replace(',', '_')}.json"
    plot_out = args.plot_out or f"qiskit_qos_gse132080_thirdorder_quantum_q{args.qubits.replace(',', '_')}.png"

    run_start = time.perf_counter()
    for run_index, num_qubits in enumerate(qubits):
        encoded_train, train_stats = thirdorder.build_thirdorder_hashed_matrix(
            x_train,
            feature_dim=int(args.pre_hash_dim or num_qubits),
            hash_seed=int(args.hash_seed + run_index),
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
            hash_repeats=args.hash_repeats,
            signed_hash=args.signed_hash,
            activation_scale=args.activation_scale,
        )
        encoded_test, test_stats = thirdorder.build_thirdorder_hashed_matrix(
            x_test,
            feature_dim=int(args.pre_hash_dim or num_qubits),
            hash_seed=int(args.hash_seed + run_index),
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
            hash_repeats=args.hash_repeats,
            signed_hash=args.signed_hash,
            activation_scale=args.activation_scale,
        )
        encoder_meta: dict[str, object]
        if args.encoder == "direct":
            final_train = np.asarray(encoded_train, dtype=np.float64)
            final_test = np.asarray(encoded_test, dtype=np.float64)
            encoder_meta = {
                "encoder_method": "direct",
                "pre_hash_dim": int(args.pre_hash_dim or num_qubits),
                "encoder_effective_components": int(num_qubits),
                "encoder_explained_variance_ratio_sum": None,
            }
        else:
            pre_train = np.asarray(encoded_train, dtype=np.float64)
            pre_test = np.asarray(encoded_test, dtype=np.float64)
            pre_train, pre_test = toy.standardize(pre_train, pre_test)
            encoder = toy.ToyEncoding.fit(
                pre_train,
                num_qubits=int(num_qubits),
                rng=np.random.default_rng(int(args.seed + run_index)),
                method=args.encoder,
                y_train=y_train,
            )
            final_train = encoder.encode(pre_train)
            final_test = encoder.encode(pre_test)
            encoder_meta = {
                "encoder_method": encoder.method,
                "pre_hash_dim": int(pre_train.shape[1]),
                "encoder_effective_components": int(encoder.effective_components),
                "encoder_explained_variance_ratio_sum": encoder.explained_variance_ratio_sum,
                "encoder_scale": float(encoder.scale),
            }
        result = genomics_runner.run_quantum_on_encoded_split(
            encoded_train=final_train,
            encoded_test=final_test,
            y_train=y_train,
            y_test=y_test,
            readout_shots=args.readout_shots,
            seed=int(args.seed + run_index),
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
            query_batch_size=args.query_batch_size,
        )
        rows.append(
            {
                "num_qubits": int(num_qubits),
                "thirdorder_hash_seed": int(args.hash_seed + run_index),
                "max_active_genes": int(args.max_active_genes),
                "value_mode": args.value_mode,
                "hash_repeats": int(args.hash_repeats),
                "signed_hash": bool(args.signed_hash),
                "activation_scale": float(args.activation_scale),
                "encoder": args.encoder,
                "pre_hash_dim": int(args.pre_hash_dim or num_qubits),
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
                **encoder_meta,
                **result,
            }
        )

    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "positive_guide": args.positive_guide,
            "negative_guide": args.negative_guide,
            "qubits": qubits,
            "seed": args.seed,
            "hash_seed": args.hash_seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "max_active_genes": args.max_active_genes,
            "value_mode": args.value_mode,
            "hash_repeats": args.hash_repeats,
            "signed_hash": args.signed_hash,
            "activation_scale": args.activation_scale,
            "encoder": args.encoder,
            "pre_hash_dim": args.pre_hash_dim,
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
            "thirdorder_ambient_feature_dim": ambient_feature_dim,
            "thirdorder_ambient_dense_weight_bytes": ambient_dense_weight_bytes,
            "thirdorder_ambient_dense_weight_human": genomics_runner.human_bytes(ambient_dense_weight_bytes),
        },
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
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
            "The host expands each GSE132080 cell into hashed third-order interactions and only then feeds a q-dimensional encoded vector into the Qiskit sketch path.",
            "query_batch_size batches logical samples at the Pauli-readout stage; the raw third-order ambient feature size stays on the host side.",
            "dense_encoder_matrix_bytes_avoided is the full dense q x thirdorder_ambient_feature_dim matrix that this hashed route avoids storing.",
        ],
    }

    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, output_path=plot_out)

    print("GSE132080 third-order quantum runner")
    print(f"- pair: {args.positive_guide} vs {args.negative_guide}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- third-order ambient dense classical weight memory: {genomics_runner.human_bytes(ambient_dense_weight_bytes)}")
    for row in rows:
        print(
            f"- q={row['num_qubits']}: "
            f"quantum={row['test_accuracy_quantum']:.3f} "
            f"hashed_ridge={row['test_accuracy_classical_hashed_ridge']:.3f} "
            f"hashed_svc={row['test_accuracy_classical_hashed_linearsvc']:.3f} "
            f"sketch={row['streaming_sketch_state_human']} "
            f"avoided_matrix={row['dense_encoder_matrix_human_avoided']} "
            f"query_batches={row['execution_metadata']['query']['batch_count']}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
