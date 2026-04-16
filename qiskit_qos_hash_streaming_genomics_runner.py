#!/usr/bin/env python3
"""Hash-streaming genomics quantum runner.

This is a separate extension for the ML_adv workspace. It avoids the dense
num_qubits x feature_dim encoder matrix from the generic toy path and instead
maps each raw k-mer stream directly into a compact q-dimensional signed hash
representation. The result is suitable for host-side streaming before a small
quantum sketch/readout stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC

import qiskit_qos_splice_kmer_utils as splice_utils
import qiskit_qos_scaling_runner as scaling
import qiskit_qos_toy_model as toy

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt

MASK64 = (1 << 64) - 1
REPEAT_CONST = 0x9E3779B97F4A7C15


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    out = [int(item) for item in items]
    if any(item <= 0 for item in out):
        raise ValueError("all integer values must be positive")
    return out


def human_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def batched(items: np.ndarray, batch_size: int | None) -> list[np.ndarray]:
    if batch_size is None or batch_size <= 0 or batch_size >= len(items):
        return [items]
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def splitmix64(value: int) -> int:
    z = (int(value) + REPEAT_CONST) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def benchmark_indices(
    total: int,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int | None,
    max_test_samples: int | None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if total < 4:
        raise ValueError("need at least 4 sequences")
    if labels is None:
        n_train = int(round(train_fraction * total))
        n_train = max(2, min(int(n_train), total - 1))
        dummy = np.zeros((total, 1), dtype=np.float64)
        train_idx, test_idx = toy.split_train_test(dummy, n_train=n_train, rng=rng)
    else:
        labels = np.asarray(labels)
        if len(labels) != total:
            raise ValueError("labels length must match total")
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        all_indices = np.arange(total, dtype=np.int64)
        for label in np.unique(labels):
            label_idx = all_indices[labels == label]
            if len(label_idx) < 2:
                raise ValueError("need at least 2 samples per class for stratified benchmark split")
            shuffled = rng.permutation(label_idx)
            n_train_label = int(round(train_fraction * len(label_idx)))
            n_train_label = max(1, min(int(n_train_label), len(label_idx) - 1))
            train_parts.append(np.asarray(shuffled[:n_train_label], dtype=np.int64))
            test_parts.append(np.asarray(shuffled[n_train_label:], dtype=np.int64))
        train_idx = np.sort(np.concatenate(train_parts))
        test_idx = np.sort(np.concatenate(test_parts))
    train_idx = balanced_subsample_indices(train_idx, labels=labels, max_samples=max_train_samples, rng=rng)
    test_idx = balanced_subsample_indices(test_idx, labels=labels, max_samples=max_test_samples, rng=rng)
    return train_idx, test_idx


def balanced_subsample_indices(
    indices: np.ndarray,
    *,
    labels: np.ndarray | None,
    max_samples: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples is None or len(indices) <= max_samples:
        return indices
    if labels is None:
        keep = rng.choice(len(indices), size=max_samples, replace=False)
        keep.sort()
        return np.asarray(indices[keep], dtype=np.int64)
    labels = np.asarray(labels)
    label_values = np.unique(labels[indices])
    if max_samples < len(label_values):
        raise ValueError("max_samples must be at least the number of classes for balanced subsampling")
    selected: list[np.ndarray] = []
    shuffled_by_label: dict[float | int, np.ndarray] = {}
    quotas: dict[float | int, int] = {}
    base = max_samples // len(label_values)
    for label in label_values:
        label_idx = indices[labels[indices] == label]
        shuffled = np.asarray(rng.permutation(label_idx), dtype=np.int64)
        shuffled_by_label[label] = shuffled
        quotas[label] = min(base, len(shuffled))
    remaining = max_samples - sum(quotas.values())
    while remaining > 0:
        candidates = [label for label in label_values if quotas[label] < len(shuffled_by_label[label])]
        if not candidates:
            break
        candidates.sort(key=lambda label: (len(shuffled_by_label[label]) - quotas[label], str(label)), reverse=True)
        quotas[candidates[0]] += 1
        remaining -= 1
    for label in label_values:
        quota = quotas[label]
        if quota <= 0:
            continue
        selected.append(shuffled_by_label[label][:quota])
    return np.sort(np.concatenate(selected))


def binary_prediction_metrics(y_true: np.ndarray, y_pred_positive: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.float64) > 0.0
    pred = np.asarray(y_pred_positive, dtype=bool)
    tp = int(np.sum(pred & truth))
    tn = int(np.sum((~pred) & (~truth)))
    fp = int(np.sum(pred & (~truth)))
    fn = int(np.sum((~pred) & truth))
    positive = int(np.sum(truth))
    negative = int(np.sum(~truth))
    positive_recall = float(tp / positive) if positive else 0.0
    negative_recall = float(tn / negative) if negative else 0.0
    if positive and negative:
        balanced_accuracy = 0.5 * (positive_recall + negative_recall)
    elif positive:
        balanced_accuracy = positive_recall
    elif negative:
        balanced_accuracy = negative_recall
    else:
        balanced_accuracy = 0.0
    return {
        "accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": float(balanced_accuracy),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


@dataclass
class HashStreamingEncoder:
    num_qubits: int
    k: int
    seed: int
    repeats: int = 2
    activation_scale: float = 1.0

    def encode_sequence(self, sequence: str) -> np.ndarray:
        encoded = np.zeros(self.num_qubits, dtype=np.float64)
        event_count = 0
        for key in splice_utils.iter_kmer_keys(sequence, k=self.k):
            event_count += 1
            mixed = splitmix64(key ^ self.seed)
            for repeat in range(self.repeats):
                probe = splitmix64(mixed + repeat * REPEAT_CONST)
                bucket = probe % self.num_qubits
                sign = 1.0 if ((probe >> 63) & 1) == 0 else -1.0
                encoded[bucket] += sign
        if event_count == 0:
            return encoded
        encoded /= math.sqrt(event_count * self.repeats)
        return np.tanh(self.activation_scale * encoded)

    def encode_sequences(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([self.encode_sequence(sequence) for sequence in sequences], dtype=np.float64)

    @property
    def sketch_state_bytes(self) -> int:
        return int(8 * (self.num_qubits + max(self.num_qubits - 1, 0)))

    @property
    def encoded_sample_bytes(self) -> int:
        return int(8 * self.num_qubits)

    def dense_encoder_matrix_bytes(self, ambient_feature_dim: int) -> int:
        return int(ambient_feature_dim) * int(self.num_qubits) * 8


def run_quantum_on_encoded_split(
    *,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    readout_shots: int | None,
    seed: int,
    quantum_head_method: str,
    readout_family: str,
    execution_config: toy.QuantumExecutionConfig,
    query_batch_size: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_qubits = int(encoded_train.shape[1])

    sketch = toy.WeightedStreamingSketch(num_qubits=num_qubits)
    for encoded_sample, label in zip(encoded_train, y_train, strict=True):
        sketch.update(encoded_sample, float(label))

    sketch_batch = toy.extract_pauli_features(
        [sketch.build_circuit()],
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    model_features = sketch_batch.features[0]

    all_encoded = np.concatenate([encoded_train, encoded_test], axis=0)
    query_features_batches: list[np.ndarray] = []
    query_metadata_batches: list[dict[str, Any]] = []
    for batch_index, encoded_batch in enumerate(batched(all_encoded, query_batch_size)):
        query_circuits = [
            toy.query_circuit(
                encoded_sample,
                single_scale=sketch.single_scale,
                phase_scale=sketch.phase_scale,
                pair_scale=sketch.pair_scale,
            )
            for encoded_sample in encoded_batch
        ]
        query_batch = toy.extract_pauli_features(
            query_circuits,
            shots=readout_shots,
            rng=rng,
            execution_config=execution_config,
            readout_family=readout_family,
        )
        query_features_batches.append(np.asarray(query_batch.features, dtype=np.float64))
        query_metadata_batches.append(
            {
                "batch_index": int(batch_index),
                "logical_query_circuit_count": int(len(query_circuits)),
                **query_batch.metadata,
            }
        )
    query_features_all = np.concatenate(query_features_batches, axis=0)
    query_train = np.asarray(query_features_all[: len(encoded_train)], dtype=np.float64)
    query_test = np.asarray(query_features_all[len(encoded_train) :], dtype=np.float64)
    cosine_train = np.asarray([toy.cosine_similarity(model_features, row) for row in query_train], dtype=np.float64)
    cosine_test = np.asarray([toy.cosine_similarity(model_features, row) for row in query_test], dtype=np.float64)

    if quantum_head_method == "cosine":
        q_scores_train = cosine_train
        q_scores_test = cosine_test
        query_feature_count = int(query_train.shape[1])
        head_feature_count = 1
    else:
        head_train_raw = np.asarray(
            [toy.quantum_head_feature_vector(model_features, row) for row in query_train],
            dtype=np.float64,
        )
        head_test_raw = np.asarray(
            [toy.quantum_head_feature_vector(model_features, row) for row in query_test],
            dtype=np.float64,
        )
        head_train, head_test = toy.standardize(head_train_raw, head_test_raw)
        if quantum_head_method == "ridge":
            head_w = toy.ridge_linear_classifier(head_train, y_train)
            q_scores_train = np.asarray(head_train @ head_w, dtype=np.float64)
            q_scores_test = np.asarray(head_test @ head_w, dtype=np.float64)
        elif quantum_head_method == "logistic":
            clf = toy.LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(head_train, (y_train > 0.0).astype(np.int64))
            q_scores_train = np.asarray(clf.decision_function(head_train), dtype=np.float64)
            q_scores_test = np.asarray(clf.decision_function(head_test), dtype=np.float64)
        else:
            raise ValueError(f"unsupported quantum head method: {quantum_head_method}")
        query_feature_count = int(query_train.shape[1])
        head_feature_count = int(head_train.shape[1])

    classical_train, classical_test = toy.standardize(encoded_train, encoded_test)
    ridge_w = toy.ridge_linear_classifier(classical_train, y_train)
    ridge_scores_train = classical_train @ ridge_w
    ridge_scores_test = classical_test @ ridge_w

    if toy.pearson_corr(q_scores_train, y_train) < 0.0:
        q_scores_train *= -1.0
        q_scores_test *= -1.0
        model_features = -model_features

    q_threshold = 0.5 * (
        float(np.mean(q_scores_train[y_train > 0.0])) + float(np.mean(q_scores_train[y_train < 0.0]))
    )
    ridge_threshold = 0.5 * (
        float(np.mean(ridge_scores_train[y_train > 0.0])) + float(np.mean(ridge_scores_train[y_train < 0.0]))
    )

    svc = LinearSVC(random_state=seed)
    svc.fit(classical_train, y_train.astype(int))
    svc_train_pred = np.asarray(svc.predict(classical_train) > 0, dtype=bool)
    svc_test_pred = np.asarray(svc.predict(classical_test) > 0, dtype=bool)

    q_train_metrics = binary_prediction_metrics(y_train, q_scores_train >= q_threshold)
    q_test_metrics = binary_prediction_metrics(y_test, q_scores_test >= q_threshold)
    ridge_train_metrics = binary_prediction_metrics(y_train, ridge_scores_train >= ridge_threshold)
    ridge_test_metrics = binary_prediction_metrics(y_test, ridge_scores_test >= ridge_threshold)
    svc_train_metrics = binary_prediction_metrics(y_train, svc_train_pred)
    svc_test_metrics = binary_prediction_metrics(y_test, svc_test_pred)

    return {
        "train_accuracy_quantum": q_train_metrics["accuracy"],
        "test_accuracy_quantum": q_test_metrics["accuracy"],
        "train_balanced_accuracy_quantum": q_train_metrics["balanced_accuracy"],
        "test_balanced_accuracy_quantum": q_test_metrics["balanced_accuracy"],
        "train_accuracy_classical_hashed_ridge": ridge_train_metrics["accuracy"],
        "test_accuracy_classical_hashed_ridge": ridge_test_metrics["accuracy"],
        "train_balanced_accuracy_classical_hashed_ridge": ridge_train_metrics["balanced_accuracy"],
        "test_balanced_accuracy_classical_hashed_ridge": ridge_test_metrics["balanced_accuracy"],
        "train_accuracy_classical_hashed_linearsvc": svc_train_metrics["accuracy"],
        "test_accuracy_classical_hashed_linearsvc": svc_test_metrics["accuracy"],
        "train_balanced_accuracy_classical_hashed_linearsvc": svc_train_metrics["balanced_accuracy"],
        "test_balanced_accuracy_classical_hashed_linearsvc": svc_test_metrics["balanced_accuracy"],
        "quantum_threshold": q_threshold,
        "classical_hashed_ridge_threshold": ridge_threshold,
        "signal_overlap_with_baseline": toy.pearson_corr(q_scores_test, ridge_scores_test),
        "readout_feature_count": int(len(model_features)),
        "query_feature_count": query_feature_count,
        "quantum_head_method": quantum_head_method,
        "quantum_head_feature_count": head_feature_count,
        "train_metrics_quantum": q_train_metrics,
        "test_metrics_quantum": q_test_metrics,
        "train_metrics_classical_hashed_ridge": ridge_train_metrics,
        "test_metrics_classical_hashed_ridge": ridge_test_metrics,
        "train_metrics_classical_hashed_linearsvc": svc_train_metrics,
        "test_metrics_classical_hashed_linearsvc": svc_test_metrics,
        "execution_metadata": {
            "sketch": sketch_batch.metadata,
            "query": {
                "batch_count": int(len(query_metadata_batches)),
                "logical_query_circuit_count_total": int(len(all_encoded)),
                "batches": query_metadata_batches,
            },
        },
    }


def render_plot(rows: list[dict[str, Any]], *, output_path: str) -> None:
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
    ax.set_title("Hash-streaming genomics benchmark")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hash-streaming genomics quantum benchmarks.")
    parser.add_argument("--source", default="splice-openml", choices=["splice-openml"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--qubits", default="10,20", help="Comma-separated qubit counts")
    parser.add_argument("--hash-repeats", type=int, default=2)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--readout-shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=None, help="Optional max number of logical query circuits per extract/submit batch")
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
    qubits = parse_int_list(args.qubits)
    execution_config = scaling.build_execution_config(args)
    sequences, labels, source_meta = splice_utils.load_splice_sequences(binary=True)
    train_idx, test_idx = benchmark_indices(
        len(sequences),
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        labels=labels,
    )
    train_sequences = [sequences[int(index)] for index in train_idx]
    test_sequences = [sequences[int(index)] for index in test_idx]
    y_train = labels[train_idx].astype(np.float64)
    y_test = labels[test_idx].astype(np.float64)

    rows: list[dict[str, Any]] = []
    json_out = args.json_out or f"qiskit_qos_hash_streaming_genomics_k{args.k}.json"
    plot_out = args.plot_out or f"qiskit_qos_hash_streaming_genomics_k{args.k}.png"

    run_start = time.perf_counter()
    ambient_feature_dim = splice_utils.ambient_kmer_dim(args.k)
    ambient_dense_weight_bytes = splice_utils.ambient_dense_weight_bytes(args.k)

    for run_index, num_qubits in enumerate(qubits):
        encoder = HashStreamingEncoder(
            num_qubits=num_qubits,
            k=args.k,
            seed=args.hash_seed + run_index,
            repeats=args.hash_repeats,
        )
        encoded_train = encoder.encode_sequences(train_sequences)
        encoded_test = encoder.encode_sequences(test_sequences)
        result = run_quantum_on_encoded_split(
            encoded_train=encoded_train,
            encoded_test=encoded_test,
            y_train=y_train,
            y_test=y_test,
            readout_shots=args.readout_shots,
            seed=args.seed + run_index,
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
            query_batch_size=args.query_batch_size,
        )
        rows.append(
            {
                "num_qubits": int(num_qubits),
                "hash_repeats": int(args.hash_repeats),
                "hash_seed": int(args.hash_seed + run_index),
                "k": int(args.k),
                "ambient_feature_dim": int(ambient_feature_dim),
                "ambient_dense_weight_bytes": int(ambient_dense_weight_bytes),
                "ambient_dense_weight_human": human_bytes(ambient_dense_weight_bytes),
                "streaming_sketch_state_bytes": encoder.sketch_state_bytes,
                "streaming_sketch_state_human": human_bytes(encoder.sketch_state_bytes),
                "encoded_sample_bytes": encoder.encoded_sample_bytes,
                "encoded_sample_human": human_bytes(encoder.encoded_sample_bytes),
                "dense_encoder_matrix_bytes_avoided": encoder.dense_encoder_matrix_bytes(ambient_feature_dim),
                "dense_encoder_matrix_human_avoided": human_bytes(encoder.dense_encoder_matrix_bytes(ambient_feature_dim)),
                "train_sequence_count": int(len(train_sequences)),
                "test_sequence_count": int(len(test_sequences)),
                "avg_train_kmer_events": float(
                    np.mean([max(0, len(sequence) - args.k + 1) for sequence in train_sequences])
                ),
                **result,
            }
        )

    payload = {
        "config": {
            "source": args.source,
            "k": int(args.k),
            "qubits": qubits,
            "hash_repeats": int(args.hash_repeats),
            "hash_seed": int(args.hash_seed),
            "readout_shots": args.readout_shots,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "quantum_head": args.quantum_head,
            "readout_family": args.readout_family,
            "execution_mode": args.execution_mode,
            "backend_name": args.backend_name,
            "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "query_batch_size": args.query_batch_size,
        },
        "source": {
            **source_meta,
            "ambient_feature_dim": int(ambient_feature_dim),
            "ambient_dense_weight_bytes": int(ambient_dense_weight_bytes),
            "ambient_dense_weight_human": human_bytes(ambient_dense_weight_bytes),
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
            "raw sequences are streamed into q hash bins; no num_qubits x ambient_feature_dim dense encoder matrix is stored",
            "ambient_dense_weight_bytes is the canonical dense classical weight size over the full 4^k k-mer space",
            "dense_encoder_matrix_bytes_avoided estimates the matrix storage avoided relative to the generic dense encoder path",
        ],
    }

    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, output_path=plot_out)

    print("Hash-streaming genomics runner")
    print(f"- source: {args.source}")
    print(f"- k: {args.k}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- ambient dense classical weight memory: {human_bytes(ambient_dense_weight_bytes)}")
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
            f"avoided_matrix={row['dense_encoder_matrix_human_avoided']}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
