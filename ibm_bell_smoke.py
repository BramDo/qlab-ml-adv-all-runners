#!/usr/bin/env python3
"""Minimal IBM Runtime Bell-state smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2

import qiskit_qos_toy_model as toy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Bell-state smoke test on IBM Runtime.")
    parser.add_argument("--backend-name", default="ibm_marrakesh")
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--json-out", default="ibm_bell_smoke.json")
    return parser.parse_args()


def build_bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def counts_from_result(result: object, *, shots: int) -> dict[str, int]:
    counts_list = toy._extract_counts_list_from_sampler_result(
        result,
        shots=shots,
        num_bits=2,
        n_items=1,
    )
    return counts_list[0]


def main() -> None:
    args = parse_args()
    json_out = Path(args.json_out)

    service = toy._build_runtime_service()
    backend = service.backend(args.backend_name)

    circuit = build_bell_circuit()
    transpiled = transpile(circuit, backend=backend, optimization_level=args.optimization_level)

    start = time.perf_counter()
    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=int(args.shots))
    result = job.result()
    elapsed = time.perf_counter() - start

    refreshed = False
    try:
        result = toy._refetch_runtime_job_result(str(job.job_id()))
        refreshed = True
    except Exception:
        pass

    counts = counts_from_result(result, shots=int(args.shots))
    payload = {
        "backend_name": toy._backend_name(backend),
        "job_id": str(job.job_id()),
        "shots": int(args.shots),
        "optimization_level": int(args.optimization_level),
        "elapsed_seconds": float(elapsed),
        "result_source": "fresh-service.job.result" if refreshed else "job.result",
        "circuit_depth": int(transpiled.depth()),
        "circuit_size": int(transpiled.size()),
        "counts": counts,
        "dominant_probability": max(counts.values()) / float(args.shots) if counts else 0.0,
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
