#!/usr/bin/env python3
"""Prepare the latest PBMC68k 40q batch and call Fire Opal validation only.

The provider path is deliberately limited to supported-device discovery and
``fireopal.validate``. It cannot submit circuits for hardware execution.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

import qiskit_qos_hash_streaming_genomics_runner as genomics_runner
import qiskit_qos_pbmc68k_pairwise_screen as pairwise_screen
import qiskit_qos_pbmc68k_utils as pbmc
import qiskit_qos_toy_model as toy


SCHEMA_VERSION = "1.0"
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SEEDS = (11, 13)
DEFAULT_QUBITS = 40
DEFAULT_TRAIN_SAMPLES = 16
DEFAULT_TEST_SAMPLES = 16
DEFAULT_ACTIVE_GENES = 256
DEFAULT_FEATURE_MAPPINGS = 2
DEFAULT_READOUT_SHOTS = 128
DEFAULT_SEED_TRANSPILER = 1729
FIRE_OPAL_MAX_BATCH = 300
FIRE_OPAL_PAYLOAD_BASIS = ("u1", "u2", "u3", "cx")


class RunnerError(RuntimeError):
    """Raise a sanitized runner error that is safe to persist."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "qiskit": _version("qiskit"),
        "qiskit_ibm_runtime": _version("qiskit-ibm-runtime"),
        "fire_opal": _version("fire-opal"),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return _sha256_bytes(contiguous.tobytes())


def _evaluate_qasm_number(expression: str) -> float:
    """Evaluate restricted OpenQASM arithmetic without using ``eval``."""

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise RunnerError(f"Unsupported OpenQASM numeric expression: {expression}")

    try:
        return visit(ast.parse(expression.strip(), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise RunnerError(f"Invalid OpenQASM numeric expression: {expression}") from exc


def _numericise_qasm_parameters(qasm: str) -> str:
    gate_call = re.compile(
        r"^(?P<prefix>\s*[A-Za-z_][A-Za-z0-9_]*)\((?P<args>[^()]*)\)(?P<suffix>.*)$"
    )
    output: list[str] = []
    for line in qasm.splitlines():
        match = gate_call.match(line)
        if match is None:
            output.append(line)
            continue
        numeric = ",".join(
            format(_evaluate_qasm_number(argument), ".17g")
            for argument in match.group("args").split(",")
        )
        output.append(f"{match.group('prefix')}({numeric}){match.group('suffix')}")
    return "\n".join(output) + "\n"


def _instruction_signatures(circuit: QuantumCircuit) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for item in circuit.data:
        params: list[float | str] = []
        for parameter in item.operation.params:
            try:
                params.append(round(float(parameter), 12))
            except (TypeError, ValueError):
                params.append(str(parameter))
        signatures.append(
            {
                "name": item.operation.name,
                "qubits": [circuit.find_bit(qubit).index for qubit in item.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in item.clbits],
                "params": params,
            }
        )
    return signatures


def _measurement_mapping(circuit: QuantumCircuit) -> list[dict[str, int]]:
    return [
        {
            "qubit": circuit.find_bit(item.qubits[0]).index,
            "clbit": circuit.find_bit(item.clbits[0]).index,
        }
        for item in circuit.data
        if item.operation.name == "measure"
    ]


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    return {
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "size": int(circuit.size()),
        "depth": int(circuit.depth()),
        "two_qubit_gates": int(sum(len(item.qubits) == 2 for item in circuit.data)),
        "count_ops": {
            name: int(count) for name, count in sorted(circuit.count_ops().items())
        },
    }


def export_numeric_qasm2(
    circuit: QuantumCircuit, *, seed_transpiler: int = DEFAULT_SEED_TRANSPILER
) -> tuple[str, dict[str, Any]]:
    """Export one virtual measured circuit as portable numeric OpenQASM 2."""

    payload = transpile(
        circuit,
        basis_gates=list(FIRE_OPAL_PAYLOAD_BASIS),
        optimization_level=1,
        seed_transpiler=seed_transpiler,
    )
    if len(payload.qregs) != 1 or len(payload.cregs) != 1:
        raise RunnerError("Fire Opal payload must have one qreg and one creg")
    if payload.num_qubits != circuit.num_qubits or payload.num_clbits != circuit.num_clbits:
        raise RunnerError("Portable transpilation changed register widths")
    expected_mapping = [
        {"qubit": index, "clbit": index} for index in range(circuit.num_qubits)
    ]
    actual_mapping = sorted(_measurement_mapping(payload), key=lambda row: row["clbit"])
    if actual_mapping != expected_mapping:
        raise RunnerError("Portable transpilation changed measurement mapping")

    qasm = _numericise_qasm_parameters(qasm2.dumps(payload))
    if re.search(r"\bpi\b", qasm):
        raise RunnerError("OpenQASM payload still contains symbolic pi")
    round_trip = qasm2.loads(qasm)
    if _instruction_signatures(round_trip) != _instruction_signatures(payload):
        raise RunnerError("OpenQASM round trip changed instruction structure")
    if sorted(_measurement_mapping(round_trip), key=lambda row: row["clbit"]) != actual_mapping:
        raise RunnerError("OpenQASM round trip changed measurement mapping")
    encoded = qasm.encode("utf-8")
    return qasm, {
        "qasm_sha256": _sha256_bytes(encoded),
        "qasm_bytes": len(encoded),
        "format": "OPENQASM 2.0",
        "basis_gates": list(FIRE_OPAL_PAYLOAD_BASIS),
        "optimization_level": 1,
        "seed_transpiler": seed_transpiler,
        "metrics": circuit_metrics(payload),
        "quantum_register_count": len(payload.qregs),
        "classical_register_count": len(payload.cregs),
        "measurement_mapping": actual_mapping,
        "virtual_qubits_only": True,
        "all_parameters_numeric": True,
        "round_trip_validated": True,
    }


def _normalized_mapping(mapping: Mapping[int, str]) -> list[dict[str, Any]]:
    return [
        {"qubit": int(qubit), "pauli": str(pauli)}
        for qubit, pauli in sorted(mapping.items())
    ]


def build_seed_circuits(
    *,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    seed: int,
    hash_seed: int,
    feature_mapping_limit: int,
    circuit_offset: int = 0,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build circuits in the exact existing feature-extraction order."""

    encoded_train = np.asarray(encoded_train, dtype=np.float64)
    encoded_test = np.asarray(encoded_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if encoded_train.ndim != 2 or encoded_test.ndim != 2:
        raise RunnerError("Encoded PBMC matrices must be two-dimensional")
    if encoded_train.shape[1] != encoded_test.shape[1]:
        raise RunnerError("Train/test encoded dimensions differ")
    if len(encoded_train) != len(y_train) or len(encoded_test) != len(y_test):
        raise RunnerError("Encoded matrices and labels differ in length")

    num_qubits = int(encoded_train.shape[1])
    sketch = toy.WeightedStreamingSketch(num_qubits=num_qubits)
    for encoded_sample, label in zip(encoded_train, y_train, strict=True):
        sketch.update(encoded_sample, float(label))

    base_rows: list[tuple[QuantumCircuit, dict[str, Any]]] = [
        (
            sketch.build_circuit(),
            {
                "role": "weighted_training_sketch",
                "split": "train",
                "sample_position": None,
                "source_row_index": None,
                "label": None,
            },
        )
    ]
    for split, encoded, labels, source_indices in (
        ("train", encoded_train, y_train, train_indices),
        ("test", encoded_test, y_test, test_indices),
    ):
        for position, (sample, label, source_index) in enumerate(
            zip(encoded, labels, source_indices, strict=True)
        ):
            base_rows.append(
                (
                    toy.query_circuit(
                        sample,
                        single_scale=sketch.single_scale,
                        phase_scale=sketch.phase_scale,
                        pair_scale=sketch.pair_scale,
                    ),
                    {
                        "role": "query",
                        "split": split,
                        "sample_position": int(position),
                        "source_row_index": int(source_index),
                        "label": float(label),
                    },
                )
            )

    mappings = toy.pauli_feature_mappings(num_qubits, family="local")
    mappings = mappings[:feature_mapping_limit]
    if len(mappings) != feature_mapping_limit:
        raise RunnerError("Requested more Pauli mappings than are available")

    qasms: list[str] = []
    manifest: list[dict[str, Any]] = []
    for base_index, (base_circuit, base_metadata) in enumerate(base_rows):
        for feature_index, mapping in enumerate(mappings):
            measured = toy.measurement_circuit_for_mapping(base_circuit, mapping)
            qasm, qasm_metadata = export_numeric_qasm2(
                measured, seed_transpiler=seed_transpiler
            )
            qasms.append(qasm)
            manifest.append(
                {
                    "circuit_index": circuit_offset + len(qasms) - 1,
                    "seed": int(seed),
                    "hash_seed": int(hash_seed),
                    "base_circuit_index": int(base_index),
                    "feature_index": int(feature_index),
                    "pauli_mapping": _normalized_mapping(mapping),
                    **base_metadata,
                    **qasm_metadata,
                }
            )

    return qasms, manifest, {
        "seed": int(seed),
        "hash_seed": int(hash_seed),
        "logical_base_circuit_count": len(base_rows),
        "feature_mapping_count": len(mappings),
        "measured_circuit_count": len(qasms),
        "pauli_mappings": [_normalized_mapping(mapping) for mapping in mappings],
        "encoded_train_sha256": _array_sha256(encoded_train),
        "encoded_test_sha256": _array_sha256(encoded_test),
    }


def _dataset_artifacts(cache_dir: Path) -> list[dict[str, Any]]:
    names = (
        "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz",
        "68k_pbmc_barcodes_annotation.tsv",
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        path = cache_dir / name
        if not path.is_file():
            raise RunnerError(f"PBMC68k cache artifact is missing: {path}")
        rows.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def prepare_batch(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    x, labels, source_meta = pbmc.load_pbmc68k(cache_dir=str(args.cache_dir))
    x_pair, y_pair, pair_meta = pbmc.select_binary_pair(
        x,
        labels,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    all_qasms: list[str] = []
    all_manifest: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_idx, test_idx = genomics_runner.benchmark_indices(
            x_pair.shape[0],
            seed=seed,
            train_fraction=args.train_fraction,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            labels=y_pair,
        )
        y_train = y_pair[train_idx].astype(np.float64)
        y_test = y_pair[test_idx].astype(np.float64)
        encoded_train, train_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_pair[train_idx],
            feature_dim=args.qubits,
            hash_seed=seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        encoded_test, test_stats = pairwise_screen.build_pairwise_hashed_matrix(
            x_pair[test_idx],
            feature_dim=args.qubits,
            hash_seed=seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
        )
        qasms, manifest, circuit_meta = build_seed_circuits(
            encoded_train=encoded_train,
            encoded_test=encoded_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=train_idx,
            test_indices=test_idx,
            seed=seed,
            hash_seed=seed,
            feature_mapping_limit=args.feature_mapping_limit,
            circuit_offset=len(all_qasms),
            seed_transpiler=args.seed_transpiler,
        )
        all_qasms.extend(qasms)
        all_manifest.extend(manifest)
        seed_rows.append(
            {
                **circuit_meta,
                "train_indices": [int(value) for value in train_idx],
                "test_indices": [int(value) for value in test_idx],
                "train_class_balance": {
                    "positive": int(np.sum(y_train > 0.0)),
                    "negative": int(np.sum(y_train < 0.0)),
                },
                "test_class_balance": {
                    "positive": int(np.sum(y_test > 0.0)),
                    "negative": int(np.sum(y_test < 0.0)),
                },
                "train_encoding_stats": train_stats,
                "test_encoding_stats": test_stats,
            }
        )

    if not all_qasms or len(all_qasms) > FIRE_OPAL_MAX_BATCH:
        raise RunnerError(
            f"Fire Opal batch size {len(all_qasms)} is outside 1..{FIRE_OPAL_MAX_BATCH}"
        )
    qasm_hashes = [row["qasm_sha256"] for row in all_manifest]
    aggregate_sha256 = _sha256_bytes(
        json.dumps(qasm_hashes, separators=(",", ":")).encode("utf-8")
    )
    config = {
        "dataset": "PBMC68k",
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "qubits": args.qubits,
        "seeds": list(args.seeds),
        "hash_seed_policy": "equal_to_split_seed",
        "train_fraction": args.train_fraction,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "max_active_genes": args.max_active_genes,
        "value_mode": args.value_mode,
        "feature_mapping_limit": args.feature_mapping_limit,
        "future_readout_shots": args.readout_shots,
        "comparison_baseline": {
            "backend": args.backend,
            "readout_shots": 128,
            "readout_calibration_shots": 2048,
            "readout_mitigation": True,
            "extra_error_suppression": True,
            "dd_sequence": "XY4",
            "twirl_randomizations": 8,
        },
    }
    return all_qasms, {
        "config": config,
        "source": {
            **source_meta,
            **pair_meta,
            "cache_artifacts": _dataset_artifacts(args.cache_dir),
        },
        "seeds": seed_rows,
        "manifest": all_manifest,
        "local_validation": {
            "circuit_count": len(all_qasms),
            "batch_limit": FIRE_OPAL_MAX_BATCH,
            "within_batch_limit": True,
            "aggregate_sha256": aggregate_sha256,
            "total_qasm_bytes": sum(int(row["qasm_bytes"]) for row in all_manifest),
            "all_target_qubits": all(
                row["metrics"]["num_qubits"] == args.qubits for row in all_manifest
            ),
            "all_one_classical_register": all(
                row["classical_register_count"] == 1 for row in all_manifest
            ),
            "all_virtual_qubits_only": all(
                row["virtual_qubits_only"] for row in all_manifest
            ),
            "all_parameters_numeric": all(
                row["all_parameters_numeric"] for row in all_manifest
            ),
            "all_round_trips_validated": all(
                row["round_trip_validated"] for row in all_manifest
            ),
        },
    }


def write_qasm_bundle(
    path: Path, qasms: Sequence[str], prepared: Mapping[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_numeric_qasm2_batch",
        "config": prepared["config"],
        "source": prepared["source"],
        "seeds": prepared["seeds"],
        "circuits": [
            {**row, "qasm": qasm}
            for row, qasm in zip(prepared["manifest"], qasms, strict=True)
        ],
    }
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    reloaded_circuits = reloaded.get("circuits", [])
    if len(reloaded_circuits) != len(qasms):
        raise RunnerError("QASM bundle round trip changed circuit count")
    if [
        _sha256_bytes(str(row["qasm"]).encode("utf-8")) for row in reloaded_circuits
    ] != [row["qasm_sha256"] for row in prepared["manifest"]]:
        raise RunnerError("QASM bundle round trip changed circuit content")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "circuits": len(qasms),
        "gzip_json_round_trip_passed": True,
    }


def _safe_provider_call(label: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise RunnerError(f"{label} failed ({type(exc).__name__})") from None


def _workflow_result(value: Any) -> dict[str, Any]:
    if hasattr(value, "result") and callable(value.result):
        value = _safe_provider_call("Fire Opal workflow result", value.result)
    if not isinstance(value, Mapping):
        raise RunnerError(
            f"Fire Opal workflow returned unexpected {type(value).__name__} payload"
        )
    return dict(value)


def _qctrl_api_key(qctrl_notebook: Path | None) -> tuple[str, str]:
    key = os.environ.get("QCTRL_API_KEY")
    if key:
        return key, "environment"
    if qctrl_notebook is None:
        raise RunnerError(
            "Set QCTRL_API_KEY or pass --qctrl-notebook for provider validation"
        )
    path = qctrl_notebook.expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"Q-CTRL notebook not found: {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    patterns = (
        r"authenticate_qctrl_account\s*\(\s*api_key\s*=\s*(['\"])(.+?)\1",
        r"qctrl_api_key\s*=\s*(['\"])(.+?)\1",
        r"api_key\s*=\s*(['\"])(.+?)\1",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            candidate = str(match.group(2)).strip()
            if candidate and "YOUR" not in candidate.upper():
                return candidate, "notebook_in_memory"
    raise RunnerError("No Q-CTRL API key candidate found in the supplied notebook")


def _read_qiskit_account(account_name: str | None) -> tuple[str, str, dict[str, Any]]:
    path = Path.home() / ".qiskit" / "qiskit-ibm.json"
    if not path.is_file():
        raise RunnerError("No saved Qiskit Runtime account file was found")
    accounts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(accounts, Mapping) or not accounts:
        raise RunnerError("No saved Qiskit Runtime accounts were found")
    selected = account_name or (
        "default-ibm-cloud" if "default-ibm-cloud" in accounts else next(iter(accounts))
    )
    account = accounts.get(selected)
    if (
        not isinstance(account, Mapping)
        or not account.get("token")
        or not account.get("instance")
    ):
        raise RunnerError(f"Saved IBM account {selected!r} lacks token or instance")
    return str(account["token"]), str(account["instance"]), {"account": selected}


def _ibm_credentials_source(
    account_name: str | None, instance_override: str | None
) -> tuple[str, str, dict[str, Any]]:
    environment_token = os.environ.get("IBM_CLOUD_API_KEY")
    environment_instance = os.environ.get("IBM_QUANTUM_CRN")
    saved_token: str | None = None
    saved_instance: str | None = None
    saved_source: dict[str, Any] = {"account": None}
    if not environment_token or not (instance_override or environment_instance):
        saved_token, saved_instance, saved_source = _read_qiskit_account(account_name)
    token = environment_token or saved_token
    instance = instance_override or environment_instance or saved_instance
    if not token or not instance:
        raise RunnerError("IBM credentials require both a token and an instance")
    return token, instance, {
        "token_source": "environment" if environment_token else "saved_qiskit_account",
        "instance_source": (
            "cli_override"
            if instance_override
            else "environment"
            if environment_instance
            else "saved_qiskit_account"
        ),
        "account": saved_source.get("account"),
        "instance_sha256": _sha256_bytes(instance.encode("utf-8")),
    }


def _fire_opal_credentials_from_source(
    account_name: str | None,
    qctrl_notebook: Path | None,
    instance_override: str | None,
) -> tuple[Any, Any, dict[str, Any], str]:
    qctrl_key, qctrl_source = _qctrl_api_key(qctrl_notebook)
    import fireopal

    _safe_provider_call(
        "Q-CTRL authentication", fireopal.authenticate_qctrl_account, api_key=qctrl_key
    )
    token, instance, credential_source = _ibm_credentials_source(
        account_name, instance_override
    )
    credentials = _safe_provider_call(
        "Fire Opal IBM credential construction",
        fireopal.credentials.make_credentials_for_ibm_cloud,
        token=token,
        instance=instance,
    )
    return fireopal, credentials, credential_source, qctrl_source


def validate_fireopal_batch(
    qasms: Sequence[str],
    *,
    backend: str,
    qiskit_account: str | None,
    qctrl_notebook: Path | None,
    instance: str | None,
) -> dict[str, Any]:
    if not qasms or len(qasms) > FIRE_OPAL_MAX_BATCH:
        raise RunnerError("Provider validation batch is outside the supported size")
    fireopal, credentials, credential_source, qctrl_source = (
        _fire_opal_credentials_from_source(
            qiskit_account, qctrl_notebook, instance
        )
    )
    devices = _workflow_result(
        _safe_provider_call(
            "Fire Opal supported-device discovery",
            fireopal.show_supported_devices,
            credentials=credentials,
        )
    )
    supported = [str(value) for value in devices.get("supported_devices", [])]
    if backend not in supported:
        return {
            "passed": False,
            "execution_attempted": False,
            "quantum_seconds_used": 0,
            "backend": backend,
            "backend_supported": False,
            "supported_device_count": len(supported),
            "errors": ["Backend is not in Fire Opal supported devices"],
            "warnings": [],
            "credential_source": credential_source,
            "qctrl_auth_source": qctrl_source,
            "api_calls": ["fireopal.show_supported_devices"],
        }
    with warnings.catch_warnings(record=True) as surfaced:
        warnings.simplefilter("always", RuntimeWarning)
        validation_job = _safe_provider_call(
            "Fire Opal PBMC68k batch validation",
            fireopal.validate,
            circuits=list(qasms),
            credentials=credentials,
            backend_name=backend,
        )
        action_id = getattr(validation_job, "action_id", None)
        validation = _workflow_result(validation_job)
    errors = [
        str(value) for value in validation.get("results", []) if value not in (None, "")
    ]
    returned_warnings = [
        str(value) for value in validation.get("warnings", []) if value not in (None, "")
    ]
    surfaced_warnings = [str(item.message) for item in surfaced]
    return {
        "passed": not errors,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "backend": backend,
        "backend_supported": True,
        "supported_device_count": len(supported),
        "circuits_validated": len(qasms),
        "validation_action_id": str(action_id) if action_id is not None else None,
        "errors": errors,
        "warnings": returned_warnings + surfaced_warnings,
        "returned_warning_count": len(returned_warnings),
        "surfaced_runtime_warning_count": len(surfaced_warnings),
        "credential_source": credential_source,
        "qctrl_auth_source": qctrl_source,
        "api_calls": ["fireopal.show_supported_devices", "fireopal.validate"],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative integers"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/pbmc68k"))
    parser.add_argument("--positive-label", default="CD4+/CD25 T Reg")
    parser.add_argument("--negative-label", default="CD4+/CD45RO+ Memory")
    parser.add_argument("--qubits", type=int, default=DEFAULT_QUBITS)
    parser.add_argument("--seeds", type=_parse_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--max-test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--max-active-genes", type=int, default=DEFAULT_ACTIVE_GENES)
    parser.add_argument(
        "--value-mode", choices=("binary", "log-product"), default="log-product"
    )
    parser.add_argument(
        "--feature-mapping-limit", type=int, default=DEFAULT_FEATURE_MAPPINGS
    )
    parser.add_argument("--readout-shots", type=int, default=DEFAULT_READOUT_SHOTS)
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--qiskit-account", default="default-ibm-cloud")
    parser.add_argument("--qctrl-notebook", type=Path)
    parser.add_argument("--instance")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q40/pbmc68k_q40_seed11_seed13_qasm2.json.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fire_opal_pbmc68k_q40/pbmc68k_q40_fireopal_validate.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.qubits < 1 or args.feature_mapping_limit < 1:
        raise RunnerError("Qubits and feature-mapping limit must be positive")
    for path in (args.bundle, args.output):
        if path.exists() and not args.force:
            raise RunnerError(f"Refusing to overwrite existing artifact: {path}")

    started = time.perf_counter()
    qasms, prepared = prepare_batch(args)
    bundle = write_qasm_bundle(args.bundle, qasms, prepared)
    provider: dict[str, Any] = {
        "requested": bool(args.validate),
        "passed": None,
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "api_calls": [],
    }
    if args.validate:
        try:
            provider = {
                "requested": True,
                **validate_fireopal_batch(
                    qasms,
                    backend=args.backend,
                    qiskit_account=args.qiskit_account,
                    qctrl_notebook=args.qctrl_notebook,
                    instance=args.instance,
                ),
            }
        except RunnerError as exc:
            provider = {
                "requested": True,
                "passed": False,
                "execution_attempted": False,
                "quantum_seconds_used": 0,
                "api_calls": [],
                "errors": [str(exc)],
                "warnings": [],
            }

    status = "pass" if not args.validate or provider.get("passed") is True else "fail"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pbmc68k_q40_fireopal_validate_only",
        "status": status,
        "captured_at_utc": _utc_now(),
        "environment": runtime_environment(),
        "execution_attempted": False,
        "quantum_seconds_used": 0,
        "allowed_provider_calls": [
            "fireopal.show_supported_devices",
            "fireopal.validate",
        ],
        "config": prepared["config"],
        "source": prepared["source"],
        "seeds": prepared["seeds"],
        "local_validation": prepared["local_validation"],
        "qasm_bundle": bundle,
        "provider_validation": provider,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "A passing provider validation establishes input compatibility only; "
            "it is not a hardware result or evidence of improved accuracy."
        ),
    }
    _atomic_write_json(args.output, report)
    print("PBMC68k 40q Fire Opal validation route")
    print(f"- circuits: {len(qasms)}")
    print("- local validation: pass")
    print(f"- provider validation requested: {args.validate}")
    if args.validate:
        print(f"- provider validation passed: {provider.get('passed')}")
        print(f"- validation action: {provider.get('validation_action_id')}")
        print(f"- warnings: {len(provider.get('warnings', []))}")
    print("- hardware execution attempted: False")
    print(f"- bundle: {args.bundle}")
    print(f"- report: {args.output}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
