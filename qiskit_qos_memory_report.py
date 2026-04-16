#!/usr/bin/env python3
"""Post-process scaling summaries into explicit memory estimates.

This script is intentionally separate from the toy/scaling runners so we can
add memory accounting without changing earlier benchmark code paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FLOAT64_BYTES = 8
INT64_BYTES = 8
COMPLEX128_BYTES = 16


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit = units[0]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {unit}"


def block_encoder_spec_bytes(num_qubits: int) -> int:
    # Conservative conceptual storage: start/end boundary per block plus one scale.
    return (2 * num_qubits * INT64_BYTES) + FLOAT64_BYTES


def dense_encoder_bytes(num_qubits: int, feature_dim: int) -> int:
    return num_qubits * feature_dim * FLOAT64_BYTES + FLOAT64_BYTES


def quantum_sketch_sidecar_bytes(num_qubits: int) -> int:
    pair_terms = max(num_qubits - 1, 0)
    # linear_sum, pair_sum, three scales, weight_l1, count
    return (
        (num_qubits + pair_terms) * FLOAT64_BYTES
        + 3 * FLOAT64_BYTES
        + FLOAT64_BYTES
        + INT64_BYTES
    )


def quantum_readout_model_bytes(readout_feature_count: int) -> int:
    return readout_feature_count * FLOAT64_BYTES


def quantum_head_model_bytes(head_method: str, head_feature_count: int) -> int:
    if head_method in {"ridge", "logistic"}:
        return head_feature_count * FLOAT64_BYTES + FLOAT64_BYTES
    if head_method == "cosine":
        return FLOAT64_BYTES
    raise ValueError(f"unsupported head method: {head_method}")


def classical_linear_model_bytes(feature_dim: int) -> int:
    return feature_dim * FLOAT64_BYTES + FLOAT64_BYTES


def ridge_training_workspace_bytes(feature_dim: int) -> int:
    # Gram matrix + rhs vector. This is a lower-bound style proxy for the solve.
    return (feature_dim * feature_dim + feature_dim) * FLOAT64_BYTES


def quantum_head_training_workspace_bytes(head_feature_count: int) -> int:
    return (head_feature_count * head_feature_count + head_feature_count) * FLOAT64_BYTES


def statevector_bytes(num_qubits: int) -> int:
    return COMPLEX128_BYTES * (2 ** num_qubits)


def estimate_run_memory(
    *,
    run: dict[str, object],
    config: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    num_qubits = int(run["num_qubits"])
    readout_feature_count = int(run["readout_feature_count"])
    head_feature_count = int(run["quantum_head_feature_count"])
    effective_feature_dim = int(source.get("reduced_feature_dim", source["raw_feature_dim"]))
    raw_feature_dim = int(source["raw_feature_dim"])
    encoder_method = str(config["encoder"])
    head_method = str(config["quantum_head"])

    encoder_bytes_actual = dense_encoder_bytes(num_qubits, effective_feature_dim)
    if encoder_method == "block":
        encoder_bytes_conceptual = block_encoder_spec_bytes(num_qubits)
    else:
        encoder_bytes_conceptual = encoder_bytes_actual

    sketch_bytes = quantum_sketch_sidecar_bytes(num_qubits)
    readout_bytes = quantum_readout_model_bytes(readout_feature_count)
    head_bytes = quantum_head_model_bytes(head_method, head_feature_count)

    quantum_total_actual = encoder_bytes_actual + sketch_bytes + readout_bytes + head_bytes
    quantum_total_conceptual = encoder_bytes_conceptual + sketch_bytes + readout_bytes + head_bytes

    classical_model_effective = classical_linear_model_bytes(effective_feature_dim)
    classical_model_raw = classical_linear_model_bytes(raw_feature_dim)
    classical_train_effective = ridge_training_workspace_bytes(effective_feature_dim)
    classical_train_raw = ridge_training_workspace_bytes(raw_feature_dim)
    quantum_head_train = quantum_head_training_workspace_bytes(head_feature_count)

    ratios: dict[str, float | None] = {
        "classical_effective_model_over_quantum_actual": None,
        "classical_effective_model_over_quantum_conceptual": None,
        "classical_raw_model_over_quantum_conceptual": None,
        "classical_effective_training_over_quantum_conceptual": None,
        "classical_raw_training_over_quantum_conceptual": None,
    }
    if quantum_total_actual > 0:
        ratios["classical_effective_model_over_quantum_actual"] = classical_model_effective / quantum_total_actual
    if quantum_total_conceptual > 0:
        ratios["classical_effective_model_over_quantum_conceptual"] = classical_model_effective / quantum_total_conceptual
        ratios["classical_raw_model_over_quantum_conceptual"] = classical_model_raw / quantum_total_conceptual
        ratios["classical_effective_training_over_quantum_conceptual"] = classical_train_effective / quantum_total_conceptual
        ratios["classical_raw_training_over_quantum_conceptual"] = classical_train_raw / quantum_total_conceptual

    return {
        "quantum_logical_qubits": num_qubits,
        "feature_dims": {
            "effective_feature_dim": effective_feature_dim,
            "raw_feature_dim": raw_feature_dim,
        },
        "quantum_memory": {
            "encoder_method": encoder_method,
            "quantum_head_method": head_method,
            "encoder_bytes_actual": encoder_bytes_actual,
            "encoder_bytes_actual_human": human_bytes(encoder_bytes_actual),
            "encoder_bytes_conceptual": encoder_bytes_conceptual,
            "encoder_bytes_conceptual_human": human_bytes(encoder_bytes_conceptual),
            "sketch_sidecar_bytes": sketch_bytes,
            "sketch_sidecar_bytes_human": human_bytes(sketch_bytes),
            "readout_model_bytes": readout_bytes,
            "readout_model_bytes_human": human_bytes(readout_bytes),
            "head_model_bytes": head_bytes,
            "head_model_bytes_human": human_bytes(head_bytes),
            "total_model_bytes_actual": quantum_total_actual,
            "total_model_bytes_actual_human": human_bytes(quantum_total_actual),
            "total_model_bytes_conceptual": quantum_total_conceptual,
            "total_model_bytes_conceptual_human": human_bytes(quantum_total_conceptual),
            "head_training_workspace_bytes": quantum_head_train,
            "head_training_workspace_bytes_human": human_bytes(quantum_head_train),
        },
        "classical_memory": {
            "effective_model_bytes": classical_model_effective,
            "effective_model_bytes_human": human_bytes(classical_model_effective),
            "raw_model_bytes": classical_model_raw,
            "raw_model_bytes_human": human_bytes(classical_model_raw),
            "effective_ridge_training_workspace_bytes": classical_train_effective,
            "effective_ridge_training_workspace_bytes_human": human_bytes(classical_train_effective),
            "raw_ridge_training_workspace_bytes": classical_train_raw,
            "raw_ridge_training_workspace_bytes_human": human_bytes(classical_train_raw),
        },
        "simulator_memory": {
            "statevector_bytes": statevector_bytes(num_qubits),
            "statevector_bytes_human": human_bytes(statevector_bytes(num_qubits)),
        },
        "ratios": ratios,
        "notes": [
            "Quantum totals are reported as classical sidecar bytes plus a separate logical-qubit count.",
            "For block encoding, conceptual quantum encoder memory assumes implicit block boundaries, not a stored dense matrix.",
            "Actual current toy implementation still stores a dense compressor matrix, so actual bytes can exceed conceptual bytes.",
            "The classical raw-dimension numbers use the pre-SVD TF-IDF width when available; the effective numbers use the actually benchmarked input width.",
            "Statevector memory is a simulator-only cost and is not the hardware-memory claim from the paper.",
        ],
    }


def build_report(payload: dict[str, object]) -> dict[str, object]:
    config = dict(payload["config"])
    source = dict(payload["source"])
    rows = list(payload["runs"])
    report_rows = []
    for run in rows:
        run = dict(run)
        report_rows.append(
            {
                "num_qubits": int(run["num_qubits"]),
                "test_accuracy_quantum": float(run["test_accuracy_quantum"]),
                "test_accuracy_classical": float(run["test_accuracy_classical"]),
                "elapsed_seconds": float(run["elapsed_seconds"]),
                "memory": estimate_run_memory(run=run, config=config, source=source),
            }
        )
    return {
        "source": source,
        "config": config,
        "runs": report_rows,
    }


def print_summary(report: dict[str, object]) -> None:
    source = dict(report["source"])
    config = dict(report["config"])
    print("QOS memory report")
    print(f"- source: {config['source']}")
    print(f"- encoder: {config['encoder']}")
    print(f"- quantum head: {config['quantum_head']}")
    print(f"- effective feature dim: {source.get('reduced_feature_dim', source['raw_feature_dim'])}")
    print(f"- raw feature dim: {source['raw_feature_dim']}")
    print("- runs:")
    for row in report["runs"]:
        mem = row["memory"]
        qmem = mem["quantum_memory"]
        cmem = mem["classical_memory"]
        ratios = mem["ratios"]
        print(
            "  "
            f"q={row['num_qubits']:>2}  "
            f"quantum_test={row['test_accuracy_quantum']:.3f}  "
            f"classical_test={row['test_accuracy_classical']:.3f}  "
            f"quantum_model={qmem['total_model_bytes_conceptual_human']} + {mem['quantum_logical_qubits']} logical qubits  "
            f"classical_effective_model={cmem['effective_model_bytes_human']}  "
            f"classical_raw_model={cmem['raw_model_bytes_human']}  "
            f"raw/model ratio={ratios['classical_raw_model_over_quantum_conceptual']:.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate memory usage from an existing QOS scaling summary.")
    parser.add_argument("--scaling-json", required=True, help="Scaling summary JSON produced by qiskit_qos_scaling_runner.py")
    parser.add_argument("--json-out", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.scaling_json).read_text())
    report = build_report(payload)
    print_summary(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"Saved memory report to: {args.json_out}")


if __name__ == "__main__":
    main()
