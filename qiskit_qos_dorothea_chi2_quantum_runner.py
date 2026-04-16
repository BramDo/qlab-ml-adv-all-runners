#!/usr/bin/env python3
"""Split-aware Dorothea chi2 -> quantum runner.

This is a separate extension. It keeps the earlier scaling/memory runners intact
while adding the specific experiment:

1. split Dorothea into train/test
2. fit chi2 feature selection on train only
3. run the QOS quantum surrogate on the selected features
4. compare against selected-feature classical baselines
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

import qiskit_qos_dorothea_utils as dorothea_utils
import qiskit_qos_scaling_runner as scaling
import qiskit_qos_toy_model as toy

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


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


def run_quantum_on_split(
    *,
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
    readout_shots: int | None,
    seed: int,
    encoder_method: str,
    quantum_head_method: str,
    readout_family: str,
    execution_config: toy.QuantumExecutionConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x_train, x_test = toy.standardize(x_train_raw, x_test_raw)

    encoder = toy.ToyEncoding.fit(
        x_train,
        num_qubits=num_qubits,
        rng=rng,
        method=encoder_method,
        y_train=y_train,
    )
    encoded_train = encoder.encode(x_train)
    encoded_test = encoder.encode(x_test)

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

    head = toy.quantum_head_scores(
        model_features=model_features,
        encoded_train=encoded_train,
        encoded_test=encoded_test,
        y_train=y_train,
        head_method=quantum_head_method,
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        single_scale=sketch.single_scale,
        phase_scale=sketch.phase_scale,
        pair_scale=sketch.pair_scale,
        readout_family=readout_family,
    )
    q_scores_train = head["train_scores"]
    q_scores_test = head["test_scores"]

    ridge_w = toy.ridge_linear_classifier(x_train, y_train)
    ridge_scores_train = x_train @ ridge_w
    ridge_scores_test = x_test @ ridge_w

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
    svc.fit(x_train_raw, y_train.astype(int))
    svc_train_acc = float(np.mean(svc.predict(x_train_raw) == y_train.astype(int)))
    svc_test_acc = float(np.mean(svc.predict(x_test_raw) == y_test.astype(int)))

    return {
        "train_accuracy_quantum": toy.threshold_accuracy(q_scores_train, y_train, q_threshold),
        "test_accuracy_quantum": toy.threshold_accuracy(q_scores_test, y_test, q_threshold),
        "train_accuracy_classical_ridge": toy.threshold_accuracy(ridge_scores_train, y_train, ridge_threshold),
        "test_accuracy_classical_ridge": toy.threshold_accuracy(ridge_scores_test, y_test, ridge_threshold),
        "train_accuracy_classical_linearsvc": svc_train_acc,
        "test_accuracy_classical_linearsvc": svc_test_acc,
        "quantum_threshold": q_threshold,
        "classical_ridge_threshold": ridge_threshold,
        "signal_overlap_with_baseline": toy.pearson_corr(q_scores_test, ridge_scores_test),
        "encoder_method": encoder.method,
        "encoder_effective_components": encoder.effective_components,
        "encoder_explained_variance_ratio_sum": encoder.explained_variance_ratio_sum,
        "readout_feature_count": int(len(model_features)),
        "query_feature_count": head["query_feature_count"],
        "quantum_head_method": quantum_head_method,
        "quantum_head_feature_count": head["head_feature_count"],
        "execution_metadata": {
            "sketch": sketch_batch.metadata,
            "query": head["execution_metadata"],
        },
    }


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
    ax.set_title("Dorothea chi2-selected quantum runs")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split-aware chi2-selected Dorothea quantum benchmarks.")
    parser.add_argument("--dorothea-cache-dir", default="data_cache/dorothea")
    parser.add_argument("--dorothea-train-only", action="store_true")
    parser.add_argument("--dorothea-balance", action="store_true")
    parser.add_argument("--runs", default="64:10,64:20,128:20", help="Comma-separated chi2_k:num_qubits specs")
    parser.add_argument("--readout-shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--encoder", default="ridge", choices=["block", "pca", "ridge", "lda"])
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
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
    run_specs = parse_run_specs(args.runs)

    x_sparse, y, meta = dorothea_utils.load_dorothea_sparse(
        data_dir=args.dorothea_cache_dir,
        merge_valid=not args.dorothea_train_only,
    )
    if args.dorothea_balance:
        x_sparse, y, balance_meta = dorothea_utils.balance_binary_dataset(x_sparse, y, seed=args.seed)
        meta = {
            **meta,
            **balance_meta,
            "rows": int(x_sparse.shape[0]),
            "nnz": int(x_sparse.nnz),
            "density": float(x_sparse.nnz / (x_sparse.shape[0] * x_sparse.shape[1])),
            "positive_count": int(np.sum(y > 0)),
            "negative_count": int(np.sum(y < 0)),
        }

    dense_for_split = x_sparse[:, :1].toarray()
    train_idx, test_idx = benchmark_indices(
        dense_for_split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train_sparse = x_sparse[train_idx]
    x_test_sparse = x_sparse[test_idx]
    y_train = y[train_idx].astype(np.float64)
    y_test = y[test_idx].astype(np.float64)
    y_train_01 = (y_train > 0).astype(int)

    execution_config = scaling.build_execution_config(args)
    rows: list[dict[str, Any]] = []
    json_out = args.json_out or "qiskit_qos_dorothea_chi2_quantum_runner.json"
    plot_out = args.plot_out or "qiskit_qos_dorothea_chi2_quantum_runner.png"

    for index, (chi2_k, num_qubits) in enumerate(run_specs):
        selector = SelectKBest(score_func=chi2, k=min(chi2_k, int(x_train_sparse.shape[1])))
        x_train_sel = selector.fit_transform(x_train_sparse, y_train_01)
        x_test_sel = selector.transform(x_test_sparse)
        x_train_dense = np.asarray(x_train_sel.toarray(), dtype=np.float64)
        x_test_dense = np.asarray(x_test_sel.toarray(), dtype=np.float64)

        start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
        selected_indices = selector.get_support(indices=True)
        rows.append(
            {
                "chi2_k": int(chi2_k),
                "num_qubits": int(num_qubits),
                "selected_feature_count": int(len(selected_indices)),
                "selected_feature_indices_head": [int(idx) for idx in selected_indices[:20]],
                "n_train_used": int(len(train_idx)),
                "n_test_used": int(len(test_idx)),
                "elapsed_seconds": float(elapsed),
                **result,
            }
        )
        payload = {
            "config": {
                "source": "dorothea-uci",
                "dorothea_cache_dir": args.dorothea_cache_dir,
                "dorothea_train_only": bool(args.dorothea_train_only),
                "dorothea_balance": bool(args.dorothea_balance),
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
            "source": meta,
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
                "chi2 selection is fit on the training split only to avoid label leakage.",
                "The quantum core is the same QOS surrogate used elsewhere in this workspace.",
                "LinearSVC is included as a stronger selected-feature classical reference.",
            ],
        }
        Path(json_out).write_text(json.dumps(payload, indent=2))
        render_plot(rows, output_path=plot_out)

    print("Dorothea chi2 -> quantum runner")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- raw feature dim: {x_sparse.shape[1]}")
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
