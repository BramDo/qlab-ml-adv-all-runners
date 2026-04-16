#!/usr/bin/env python3
"""Synthetic astronomical-dimension source for paper-style memory experiments.

This runner creates an implicit sparse binary classification task with a huge raw
feature universe D = 2^k. The raw feature vectors are never materialized.
Instead, each sample is generated as a sparse set of active coordinates and then
compressed into a moderate dense feature vector via a CountSketch-style map.

Purpose:
- keep the existing toy/scaling/frontier runners untouched
- let us test a "paper-style" regime where raw classical model memory becomes
  astronomical while the quantum sketch still operates on a compact surrogate
  representation

Caveat:
- this is a synthetic surrogate, not the paper's formal hard instance
- the reported classical astronomical memory is a dense linear-model proxy on
  the raw feature universe, not a theorem-backed lower bound
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

import qiskit_qos_memory_report as memory_report
import qiskit_qos_toy_model as toy


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    out = [int(item) for item in items]
    if any(item <= 0 for item in out):
        raise ValueError("all values must be positive")
    return out


def human_bytes(num_bytes: int) -> str:
    return memory_report.human_bytes(int(num_bytes))


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    z = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z & UINT64_MASK


def sample_unique_coords(
    rng: np.random.Generator,
    *,
    high: int,
    size: int,
    forbidden: set[int] | None = None,
) -> np.ndarray:
    chosen: set[int] = set() if forbidden is None else set(forbidden)
    target = int(size)
    while len(chosen) < (0 if forbidden is None else len(forbidden)) + target:
        draw = int(rng.integers(0, high))
        chosen.add(draw)
    if forbidden is not None:
        chosen = chosen.difference(forbidden)
    out = np.fromiter(chosen, dtype=np.int64, count=target)
    rng.shuffle(out)
    return out[:target]


def sketch_sparse_sample(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    effective_dim: int,
    bin_seed: int,
    sign_seed: int,
) -> np.ndarray:
    bins = np.mod(splitmix64(coords, bin_seed), np.uint64(effective_dim)).astype(np.int64)
    signs = np.where((splitmix64(coords, sign_seed) & np.uint64(1)) == 0, 1.0, -1.0)
    out = np.zeros(effective_dim, dtype=np.float64)
    np.add.at(out, bins, signs * values)
    return out


def make_astronomical_source(
    *,
    raw_log2_dim: int,
    effective_dim: int,
    n_samples: int,
    prototype_nnz: int,
    signal_nnz: int,
    noise_nnz: int,
    signal_scale: float,
    noise_scale: float,
    gaussian_noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if raw_log2_dim < 4 or raw_log2_dim > 62:
        raise ValueError("raw_log2_dim must lie between 4 and 62 for this implicit generator")
    if effective_dim < 4:
        raise ValueError("effective_dim must be >= 4")
    if n_samples < 8:
        raise ValueError("need at least 8 samples")

    rng = np.random.default_rng(seed)
    raw_dim = 1 << raw_log2_dim

    proto_pos = sample_unique_coords(rng, high=raw_dim, size=prototype_nnz)
    proto_neg = sample_unique_coords(rng, high=raw_dim, size=prototype_nnz, forbidden=set(map(int, proto_pos.tolist())))

    x = np.zeros((n_samples, effective_dim), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.float64)

    for idx in range(n_samples):
        label = 1.0 if (idx % 2 == 0) else -1.0
        y[idx] = label
        prototype = proto_pos if label > 0 else proto_neg
        signal_coords = prototype[rng.integers(0, len(prototype), size=signal_nnz)]
        noise_coords = rng.integers(0, raw_dim, size=noise_nnz, dtype=np.int64)
        coords = np.concatenate([signal_coords.astype(np.int64), noise_coords.astype(np.int64)])
        values = np.concatenate(
            [
                np.full(signal_nnz, signal_scale, dtype=np.float64),
                np.full(noise_nnz, noise_scale, dtype=np.float64),
            ]
        )
        sample = sketch_sparse_sample(
            coords,
            values,
            effective_dim=effective_dim,
            bin_seed=0x1234ABCD,
            sign_seed=0x55AA7711,
        )
        if gaussian_noise > 0.0:
            sample = sample + rng.normal(0.0, gaussian_noise, size=effective_dim)
        x[idx] = sample

    metadata = {
        "dataset_kind": "implicit_sparse_astronomical",
        "rows": int(n_samples),
        "raw_feature_dim": int(raw_dim),
        "raw_feature_dim_log2": int(raw_log2_dim),
        "raw_feature_dim_human": f"2^{raw_log2_dim}",
        "effective_feature_dim": int(effective_dim),
        "prototype_nnz_per_class": int(prototype_nnz),
        "signal_nnz_per_sample": int(signal_nnz),
        "noise_nnz_per_sample": int(noise_nnz),
        "signal_scale": float(signal_scale),
        "noise_scale": float(noise_scale),
        "gaussian_noise": float(gaussian_noise),
        "positive_count": int(np.sum(y > 0.0)),
        "negative_count": int(np.sum(y < 0.0)),
    }
    return x, y, metadata


def build_execution_config(args: argparse.Namespace) -> toy.QuantumExecutionConfig:
    return toy.QuantumExecutionConfig(
        mode=args.execution_mode,
        backend_name=args.backend_name,
        optimization_level=args.optimization_level,
        simulator_method=args.simulator_method,
        readout_mitigation=args.readout_mitigation,
        cal_shots=args.cal_shots,
        extra_error_suppression=args.extra_error_suppression,
        dd_sequence=args.dd_sequence,
        twirl_randomizations=args.twirl_randomizations,
    )


def render_plot(rows: list[dict[str, object]], *, output_path: str, classical_raw_bytes: int) -> None:
    qubits = [int(row["num_qubits"]) for row in rows]
    q_acc = [float(row["test_accuracy_quantum"]) for row in rows]
    c_acc = [float(row["test_accuracy_classical_effective"]) for row in rows]
    q_bytes = [int(row["quantum_memory"]["total_model_bytes_conceptual"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(qubits, q_acc, marker="o", label="quantum toy")
    axes[0].plot(qubits, c_acc, marker="s", label="classical effective baseline")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Accuracy vs qubits")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(qubits, q_bytes, marker="o", label="quantum sidecar bytes")
    axes[1].axhline(classical_raw_bytes, color="#ae2012", linestyle="--", label="classical raw linear bytes")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("Bytes (log scale)")
    axes[1].set_title("Memory proxy vs qubits")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def checkpoint_artifacts(
    *,
    payload: dict[str, object],
    json_out: str,
    plot_out: str,
    classical_raw_bytes: int,
) -> None:
    Path(json_out).write_text(json.dumps(payload, indent=2))
    if payload["runs"]:
        render_plot(list(payload["runs"]), output_path=plot_out, classical_raw_bytes=classical_raw_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic astronomical-dimension QOS benchmark.")
    parser.add_argument("--raw-log2-dim", type=int, default=60, help="Implicit raw dimension D = 2^k")
    parser.add_argument("--effective-dim", type=int, default=256, help="Dense CountSketch dimension materialized in memory")
    parser.add_argument("--n-samples", type=int, default=512, help="Total synthetic samples")
    parser.add_argument("--prototype-nnz", type=int, default=2048)
    parser.add_argument("--signal-nnz", type=int, default=64)
    parser.add_argument("--noise-nnz", type=int, default=64)
    parser.add_argument("--signal-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--gaussian-noise", type=float, default=0.05)
    parser.add_argument("--qubits", default="10", help="Comma-separated qubit counts to sweep")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--encoder", default="ridge", choices=["block", "pca", "ridge", "lda"])
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=128)
    parser.add_argument("--execution-mode", default="sampler-sim", choices=["statevector", "sampler-sim", "ibm-hardware"])
    parser.add_argument("--backend-name")
    parser.add_argument("--simulator-method", default="matrix_product_state", choices=["automatic", "statevector", "matrix_product_state"])
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--readout-shots", type=int, default=32)
    parser.add_argument("--readout-mitigation", action="store_true")
    parser.add_argument("--cal-shots", type=int, default=512)
    parser.add_argument("--extra-error-suppression", action="store_true")
    parser.add_argument("--dd-sequence", default="XY4", choices=["XX", "XpXm", "XY4"])
    parser.add_argument("--twirl-randomizations", type=int, default=8)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    parser.add_argument("--source-out", help="Optional .npz path to save the generated effective source snapshot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qubits = parse_int_list(args.qubits)
    x, y, source_meta = make_astronomical_source(
        raw_log2_dim=args.raw_log2_dim,
        effective_dim=args.effective_dim,
        n_samples=args.n_samples,
        prototype_nnz=args.prototype_nnz,
        signal_nnz=args.signal_nnz,
        noise_nnz=args.noise_nnz,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        gaussian_noise=args.gaussian_noise,
        seed=args.seed,
    )
    if args.source_out:
        np.savez_compressed(args.source_out, x=x, y=y)

    config_for_memory = {
        "source": f"synthetic-astronomical-2pow{args.raw_log2_dim}",
        "encoder": args.encoder,
        "quantum_head": args.quantum_head,
        "readout_family": args.readout_family,
        "execution_mode": args.execution_mode,
        "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
        "readout_shots": args.readout_shots,
    }
    source_for_memory = {
        "raw_feature_dim": int(source_meta["raw_feature_dim"]),
        "reduced_feature_dim": int(source_meta["effective_feature_dim"]),
    }
    classical_raw_linear_bytes = memory_report.classical_linear_model_bytes(int(source_meta["raw_feature_dim"]))

    rows: list[dict[str, object]] = []
    stem = f"qiskit_qos_astronomical_2pow{args.raw_log2_dim}"
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"

    for q in qubits:
        start = time.perf_counter()
        result = toy.run_classification_from_arrays(
            x=x,
            y=y,
            num_qubits=q,
            readout_shots=args.readout_shots,
            seed=args.seed,
            train_fraction=args.train_fraction,
            encoder_method=args.encoder,
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=build_execution_config(args),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        elapsed = time.perf_counter() - start
        run_row = {
            "num_qubits": int(q),
            "train_accuracy_quantum": float(result["train_accuracy_quantum"]),
            "test_accuracy_quantum": float(result["test_accuracy_quantum"]),
            "train_accuracy_classical_effective": float(result["train_accuracy_classical"]),
            "test_accuracy_classical_effective": float(result["test_accuracy_classical"]),
            "n_train_used": int(len(result["train_labels"])),
            "n_test_used": int(len(result["test_labels"])),
            "readout_feature_count": int(result["readout_feature_count"]),
            "quantum_head_feature_count": int(result["quantum_head_feature_count"]),
            "query_feature_count": int(result["query_feature_count"]),
            "quantum_threshold": float(result["quantum_threshold"]),
            "classical_effective_threshold": float(result["classical_threshold"]),
            "signal_overlap_with_baseline": float(result["signal_overlap_with_baseline"]),
            "elapsed_seconds": float(elapsed),
            "execution_metadata": result["execution_metadata"],
        }
        mem = memory_report.estimate_run_memory(
            run={
                "num_qubits": int(q),
                "readout_feature_count": int(result["readout_feature_count"]),
                "quantum_head_feature_count": int(result["quantum_head_feature_count"]),
                "test_accuracy_quantum": float(result["test_accuracy_quantum"]),
                "test_accuracy_classical": float(result["test_accuracy_classical"]),
                "elapsed_seconds": float(elapsed),
            },
            config=config_for_memory,
            source=source_for_memory,
        )
        run_row["quantum_memory"] = mem["quantum_memory"]
        run_row["classical_memory_proxy"] = {
            "raw_linear_model_bytes": int(classical_raw_linear_bytes),
            "raw_linear_model_bytes_human": human_bytes(classical_raw_linear_bytes),
            "effective_linear_model_bytes": int(mem["classical_memory"]["effective_model_bytes"]),
            "effective_linear_model_bytes_human": mem["classical_memory"]["effective_model_bytes_human"],
            "raw_over_quantum_conceptual_ratio": mem["ratios"]["classical_raw_model_over_quantum_conceptual"],
        }
        rows.append(run_row)

        payload = {
            "config": {
                "raw_log2_dim": args.raw_log2_dim,
                "qubits": qubits,
                "effective_dim": args.effective_dim,
                "n_samples": args.n_samples,
                "prototype_nnz": args.prototype_nnz,
                "signal_nnz": args.signal_nnz,
                "noise_nnz": args.noise_nnz,
                "signal_scale": args.signal_scale,
                "noise_scale": args.noise_scale,
                "gaussian_noise": args.gaussian_noise,
                "seed": args.seed,
                "train_fraction": args.train_fraction,
                "encoder": args.encoder,
                "quantum_head": args.quantum_head,
                "readout_family": args.readout_family,
                "execution_mode": args.execution_mode,
                "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
                "readout_shots": args.readout_shots,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
            },
            "source": source_meta,
            "runs": rows,
            "notes": [
                "The raw feature universe is implicit and never materialized.",
                "The materialized dataset is a CountSketch-like dense surrogate of the raw sparse coordinates.",
                "The classical raw memory number is a dense linear-model proxy on the implicit raw feature universe, not a formal lower bound.",
            ],
        }
        checkpoint_artifacts(
            payload=payload,
            json_out=json_out,
            plot_out=plot_out,
            classical_raw_bytes=classical_raw_linear_bytes,
        )

    print("QOS astronomical runner")
    print(f"- raw dimension: 2^{args.raw_log2_dim} = {source_meta['raw_feature_dim']}")
    print(f"- effective dense dimension: {args.effective_dim}")
    print(f"- execution mode: {args.execution_mode}")
    if args.execution_mode == "sampler-sim":
        print(f"- simulator method: {args.simulator_method}")
    print(f"- classical raw linear memory proxy: {human_bytes(classical_raw_linear_bytes)}")
    print("- sweep:")
    for row in rows:
        print(
            f"  q={row['num_qubits']:>2}  "
            f"quantum test={row['test_accuracy_quantum']:.3f}  "
            f"classical effective test={row['test_accuracy_classical_effective']:.3f}  "
            f"quantum memory={row['quantum_memory']['total_model_bytes_conceptual_human']} + {row['num_qubits']} logical qubits  "
            f"raw/quantum ratio={row['classical_memory_proxy']['raw_over_quantum_conceptual_ratio']:.3e}x"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")
    if args.source_out:
        print(f"Saved source snapshot to: {args.source_out}")


if __name__ == "__main__":
    main()
