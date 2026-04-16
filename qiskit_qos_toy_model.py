#!/usr/bin/env python3
"""Qiskit toy model inspired by quantum oracle sketching (QOS).

This script is intentionally modest in scope:
- It processes samples one by one and keeps only O(num_qubits) sketch state.
- It turns that sketch into a small Qiskit circuit.
- It reads out a compact classical feature vector from the resulting quantum state.

It demonstrates two tasks on synthetic data:
1. Classification via a label-weighted streaming sketch.
2. One-dimensional reduction via a guide-weighted streaming sketch that approximates Sigma g.

It does not implement the paper's full JAX QOS stack, interferometric classical shadow,
or the complexity-theoretic lower bounds. The goal is pedagogical: make the streaming
and compact-model ideas concrete inside the qlab Qiskit environment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_qos_run_logger import log_run_event

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

EPS = 1e-9


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return vec.copy()
    return vec / norm


def orthonormalize_rows(vectors: list[np.ndarray]) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for vec in vectors:
        work = np.asarray(vec, dtype=np.float64).copy()
        for existing in basis:
            work -= float(np.dot(work, existing)) * existing
        norm = float(np.linalg.norm(work))
        if norm > EPS:
            basis.append(work / norm)
    return basis


def residualize_against_basis(x: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    residual = np.asarray(x, dtype=np.float64).copy()
    for vec in basis:
        residual -= np.outer(residual @ vec, vec)
    return residual


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def split_train_test(x: np.ndarray, *, n_train: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n_train <= 0 or n_train >= len(x):
        raise ValueError("n_train must lie strictly between 0 and the dataset size")
    perm = rng.permutation(len(x))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return train_idx, test_idx


def subsample_rows(
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    max_rows: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_rows is None or len(x) <= max_rows:
        return x, y
    keep = rng.choice(len(x), size=max_rows, replace=False)
    keep.sort()
    x_sub = x[keep]
    y_sub = None if y is None else y[keep]
    return x_sub, y_sub


def pauli_label(num_qubits: int, mapping: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, gate in mapping.items():
        chars[num_qubits - 1 - qubit] = gate
    return "".join(chars)


def operator(num_qubits: int, mapping: dict[int, str]) -> SparsePauliOp:
    return SparsePauliOp.from_list([(pauli_label(num_qubits, mapping), 1.0)])


def expectation_with_shot_noise(
    state: Statevector,
    observable: SparsePauliOp,
    shots: int | None,
    rng: np.random.Generator,
) -> float:
    exact = float(np.real(state.expectation_value(observable)))
    if not shots or shots <= 0:
        return exact
    p_plus = 0.5 * (1.0 + np.clip(exact, -1.0, 1.0))
    counts = rng.binomial(shots, p_plus)
    return float((2.0 * counts / shots) - 1.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def signed_accuracy(scores: np.ndarray, labels_pm1: np.ndarray) -> float:
    preds = np.where(scores >= 0.0, 1.0, -1.0)
    return float(np.mean(preds == labels_pm1))


def threshold_accuracy(scores: np.ndarray, labels_pm1: np.ndarray, threshold: float) -> float:
    preds = np.where(scores >= threshold, 1.0, -1.0)
    return float(np.mean(preds == labels_pm1))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_center = a - np.mean(a)
    b_center = b - np.mean(b)
    denom = float(np.linalg.norm(a_center) * np.linalg.norm(b_center))
    if denom < EPS:
        return 0.0
    return float(np.dot(a_center, b_center) / denom)


@dataclass
class ClassificationData:
    x: np.ndarray
    y: np.ndarray
    signal_direction: np.ndarray


@dataclass
class ReductionData:
    x: np.ndarray
    guide: np.ndarray
    signal_direction: np.ndarray
    latent_score: np.ndarray


@dataclass
class CsvClassificationData:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    label_name: str
    label_mapping: dict[str, float]


@dataclass
class QuantumExecutionConfig:
    mode: str = "statevector"
    backend_name: str | None = None
    optimization_level: int = 1
    simulator_method: str = "automatic"
    readout_mitigation: bool = False
    cal_shots: int = 4096
    extra_error_suppression: bool = False
    dd_sequence: str = "XY4"
    twirl_randomizations: int = 8
    layout_strategy: str = "quality-chain"
    layout_beam_width: int = 32
    debug_runtime: bool = False
    runtime_submit_batch_size: int = 0
    feature_mapping_limit: int = 0
    runtime_cache: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass
class QuantumFeatureBatch:
    features: np.ndarray
    metadata: dict[str, object]


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _runtime_debug_enabled(execution_config: QuantumExecutionConfig | None = None) -> bool:
    if execution_config is not None and bool(getattr(execution_config, "debug_runtime", False)):
        return True
    return _env_flag("QISKIT_QOS_DEBUG_RUNTIME")


def _runtime_debug_log_path() -> str | None:
    path = os.environ.get("QISKIT_QOS_DEBUG_LOG")
    if path and path.strip():
        return path.strip()
    return None


def _runtime_debug(execution_config: QuantumExecutionConfig | None, stage: str, **fields: object) -> None:
    log_run_event(f"runtime::{stage}", **fields)
    if _runtime_debug_enabled(execution_config):
        line = " ".join([f"[qiskit-qos-debug] {stage}"] + [f"{key}={value!r}" for key, value in fields.items()])
        print(line, file=sys.stderr, flush=True)
        log_path = _runtime_debug_log_path()
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:
                pass


def _circuit_batch_summary(circuits: list[QuantumCircuit]) -> dict[str, object]:
    if not circuits:
        return {"circuit_count": 0}
    depths = [int(circuit.depth()) for circuit in circuits]
    sizes = [int(circuit.size()) for circuit in circuits]
    widths = [int(circuit.num_qubits) for circuit in circuits]
    return {
        "circuit_count": int(len(circuits)),
        "qubits": int(max(widths)),
        "max_depth": int(max(depths)),
        "mean_depth": float(np.mean(depths)),
        "max_size": int(max(sizes)),
        "mean_size": float(np.mean(sizes)),
    }


def _chunk_circuits(circuits: list[QuantumCircuit], batch_size: int) -> list[list[QuantumCircuit]]:
    if batch_size <= 0 or batch_size >= len(circuits):
        return [circuits]
    return [circuits[start : start + batch_size] for start in range(0, len(circuits), batch_size)]


def block_slices(n_features: int, num_blocks: int) -> list[np.ndarray]:
    if num_blocks <= 0 or num_blocks > n_features:
        raise ValueError("num_blocks must lie between 1 and n_features")
    return [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(n_features), num_blocks)]


def infer_csv_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col == label_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            feature_cols.append(col)
    return feature_cols


def parse_feature_columns_arg(feature_cols: str | None) -> list[str] | None:
    if feature_cols is None:
        return None
    cols = [col.strip() for col in feature_cols.split(",") if col.strip()]
    return cols or None


def map_binary_labels(raw_labels: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
    values = raw_labels.astype(str).str.strip()
    unique = list(dict.fromkeys(values.tolist()))
    if len(unique) != 2:
        raise ValueError(
            f"label column must contain exactly 2 unique values for binary classification, got {unique}"
        )

    canonical_map = {
        "-1": -1.0,
        "0": -1.0,
        "false": -1.0,
        "no": -1.0,
        "neg": -1.0,
        "negative": -1.0,
        "1": 1.0,
        "true": 1.0,
        "yes": 1.0,
        "pos": 1.0,
        "positive": 1.0,
    }
    lowered = [item.lower() for item in unique]
    if all(item in canonical_map for item in lowered):
        mapping = {orig: canonical_map[orig.lower()] for orig in unique}
    else:
        sorted_unique = sorted(unique)
        mapping = {sorted_unique[0]: -1.0, sorted_unique[1]: 1.0}

    return values.map(mapping).to_numpy(dtype=np.float64), mapping


def load_csv_classification_data(
    csv_path: str,
    *,
    label_col: str,
    feature_cols: list[str] | None,
) -> CsvClassificationData:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found in CSV")

    chosen_feature_cols = feature_cols or infer_csv_feature_columns(df, label_col)
    if not chosen_feature_cols:
        raise ValueError("no usable numeric feature columns found; pass --feature-cols explicitly")

    missing = [col for col in chosen_feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found in CSV: {missing}")

    keep_cols = chosen_feature_cols + [label_col]
    clean_df = df.loc[:, keep_cols].copy()
    for col in chosen_feature_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna(axis=0, how="any")
    if len(clean_df) < 4:
        raise ValueError("CSV has too few fully numeric labeled rows after dropping missing values")

    x = clean_df[chosen_feature_cols].to_numpy(dtype=np.float64)
    y, mapping = map_binary_labels(clean_df[label_col])
    return CsvClassificationData(
        x=x,
        y=y,
        feature_names=chosen_feature_cols,
        label_name=label_col,
        label_mapping={str(key): float(val) for key, val in mapping.items()},
    )


def make_block_directions(
    n_features: int,
    num_blocks: int,
    rng: np.random.Generator,
    *,
    micro_jitter: float = 0.06,
) -> np.ndarray:
    raw_blocks = rng.normal(size=(3, num_blocks))
    q, _ = np.linalg.qr(raw_blocks.T)
    coarse = q.T[:3]
    expanded = np.zeros((3, n_features), dtype=np.float64)

    for block_idx, feature_idx in enumerate(block_slices(n_features, num_blocks)):
        for row in range(3):
            expanded[row, feature_idx] = coarse[row, block_idx]

    expanded += micro_jitter * rng.normal(size=expanded.shape)
    return np.stack([normalize(row) for row in expanded], axis=0)


def make_classification_data(
    n_samples: int,
    n_features: int,
    num_blocks: int,
    rng: np.random.Generator,
    *,
    margin: float = 1.25,
    nuisance_scale: float = 0.7,
    noise_scale: float = 0.45,
) -> ClassificationData:
    signal, nuisance_a, nuisance_b = make_block_directions(n_features, num_blocks, rng)
    labels = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
    nuisance_1 = rng.normal(scale=nuisance_scale, size=n_samples)
    nuisance_2 = rng.normal(scale=0.55 * nuisance_scale, size=n_samples)
    noise = rng.normal(scale=noise_scale, size=(n_samples, n_features))

    x = (
        margin * labels[:, None] * signal[None, :]
        + nuisance_1[:, None] * nuisance_a[None, :]
        + nuisance_2[:, None] * nuisance_b[None, :]
        + noise
    )
    return ClassificationData(x=x.astype(np.float64), y=labels.astype(np.float64), signal_direction=signal)


def make_reduction_data(
    n_samples: int,
    n_features: int,
    num_blocks: int,
    rng: np.random.Generator,
    *,
    signal_scale: float = 1.45,
    nuisance_scale: float = 0.55,
    noise_scale: float = 0.35,
    guide_noise: float = 0.35,
) -> ReductionData:
    signal, nuisance_a, nuisance_b = make_block_directions(n_features, num_blocks, rng)
    latent = rng.normal(size=n_samples)
    nuisance_1 = rng.normal(scale=nuisance_scale, size=n_samples)
    nuisance_2 = rng.normal(scale=0.45 * nuisance_scale, size=n_samples)
    noise = rng.normal(scale=noise_scale, size=(n_samples, n_features))

    x = (
        signal_scale * latent[:, None] * signal[None, :]
        + nuisance_1[:, None] * nuisance_a[None, :]
        + nuisance_2[:, None] * nuisance_b[None, :]
        + noise
    )
    guide = normalize(signal + guide_noise * rng.normal(size=n_features))
    return ReductionData(
        x=x.astype(np.float64),
        guide=guide.astype(np.float64),
        signal_direction=signal.astype(np.float64),
        latent_score=latent.astype(np.float64),
    )


@dataclass
class ToyEncoding:
    compressor: np.ndarray
    scale: float
    method: str
    effective_components: int
    explained_variance_ratio_sum: float | None = None

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        *,
        num_qubits: int,
        rng: np.random.Generator,
        method: str = "block",
        y_train: np.ndarray | None = None,
    ) -> "ToyEncoding":
        if method == "block":
            del rng
            compressor = np.zeros((num_qubits, x_train.shape[1]), dtype=np.float64)
            for row, feature_idx in enumerate(block_slices(x_train.shape[1], num_qubits)):
                compressor[row, feature_idx] = 1.0 / np.sqrt(len(feature_idx))
            projected = x_train @ compressor.T
            explained = None
            effective_components = num_qubits
        elif method == "pca":
            del rng
            n_components = min(num_qubits, x_train.shape[0], x_train.shape[1])
            pca = PCA(n_components=n_components, svd_solver="full")
            projected = pca.fit_transform(x_train)
            compressor = np.zeros((num_qubits, x_train.shape[1]), dtype=np.float64)
            compressor[:n_components, :] = pca.components_
            if n_components < num_qubits:
                projected = np.pad(projected, ((0, 0), (0, num_qubits - n_components)))
            explained = float(np.sum(pca.explained_variance_ratio_))
            effective_components = n_components
        elif method == "spca":
            del rng
            if y_train is None:
                raise ValueError("encoder method 'spca' requires y_train")
            y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
            signed_x = x_train * y_train
            n_components = min(num_qubits, signed_x.shape[0], signed_x.shape[1])
            pca = PCA(n_components=n_components, svd_solver="full")
            projected = pca.fit_transform(signed_x)
            compressor = np.zeros((num_qubits, x_train.shape[1]), dtype=np.float64)
            compressor[:n_components, :] = pca.components_
            if n_components < num_qubits:
                projected = np.pad(projected, ((0, 0), (0, num_qubits - n_components)))
            explained = float(np.sum(pca.explained_variance_ratio_))
            effective_components = n_components
        elif method in {"ridge", "lda"}:
            del rng
            if y_train is None:
                raise ValueError(f"encoder method '{method}' requires y_train")
            y_train = np.asarray(y_train, dtype=np.float64)
            pos_mask = y_train > 0.0
            neg_mask = y_train < 0.0
            if not np.any(pos_mask) or not np.any(neg_mask):
                raise ValueError(f"encoder method '{method}' requires both classes in y_train")

            if method == "ridge":
                main_direction = ridge_linear_classifier(x_train, y_train)
            else:
                pos_x = x_train[pos_mask]
                neg_x = x_train[neg_mask]
                mean_diff = pos_x.mean(axis=0) - neg_x.mean(axis=0)
                pos_centered = pos_x - pos_x.mean(axis=0, keepdims=True)
                neg_centered = neg_x - neg_x.mean(axis=0, keepdims=True)
                within = pos_centered.T @ pos_centered + neg_centered.T @ neg_centered
                reg = 1e-2 * np.eye(x_train.shape[1])
                main_direction = np.linalg.solve(within + reg, mean_diff)

            basis = orthonormalize_rows([main_direction])
            residual = residualize_against_basis(x_train, basis)
            remaining = num_qubits - len(basis)
            if remaining > 0:
                n_resid = min(remaining, residual.shape[0], residual.shape[1])
                if n_resid > 0:
                    resid_pca = PCA(n_components=n_resid, svd_solver="full")
                    resid_pca.fit(residual)
                    basis.extend(orthonormalize_rows([*basis, *resid_pca.components_])[len(basis):])

            compressor = np.zeros((num_qubits, x_train.shape[1]), dtype=np.float64)
            if basis:
                compressor[: len(basis), :] = np.stack(basis, axis=0)
            projected = x_train @ compressor.T
            explained = None
            effective_components = len(basis)
        else:
            raise ValueError(f"unsupported encoder method: {method}")

        scale = float(np.quantile(np.abs(projected), 0.9))
        if scale < EPS:
            scale = 1.0
        return cls(
            compressor=compressor,
            scale=scale,
            method=method,
            effective_components=effective_components,
            explained_variance_ratio_sum=explained,
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        projected = np.asarray(x) @ self.compressor.T
        return np.tanh(projected / self.scale)


@dataclass
class WeightedStreamingSketch:
    num_qubits: int
    single_scale: float = 1.35
    phase_scale: float = 0.75
    pair_scale: float = 0.95
    linear_sum: np.ndarray = field(init=False)
    pair_sum: np.ndarray = field(init=False)
    weight_l1: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.linear_sum = np.zeros(self.num_qubits, dtype=np.float64)
        self.pair_sum = np.zeros(max(self.num_qubits - 1, 0), dtype=np.float64)

    def update(self, encoded_sample: np.ndarray, weight: float) -> None:
        encoded_sample = np.asarray(encoded_sample, dtype=np.float64)
        self.linear_sum += weight * encoded_sample
        if self.num_qubits > 1:
            self.pair_sum += weight * (encoded_sample[:-1] * encoded_sample[1:])
        self.weight_l1 += abs(weight)
        self.count += 1

    def build_circuit(self) -> QuantumCircuit:
        if self.count == 0 or self.weight_l1 < EPS:
            raise ValueError("sketch is empty")

        linear = self.linear_sum / self.weight_l1
        pair = self.pair_sum / self.weight_l1 if self.num_qubits > 1 else np.array([], dtype=np.float64)

        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        for qubit, value in enumerate(linear):
            qc.ry(self.single_scale * float(value), qubit)
            qc.rz(self.phase_scale * float(value), qubit)
        for qubit, value in enumerate(pair):
            qc.rzz(self.pair_scale * float(value), qubit, qubit + 1)
        return qc


def query_circuit(
    encoded_sample: np.ndarray,
    *,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
) -> QuantumCircuit:
    encoded_sample = np.asarray(encoded_sample, dtype=np.float64)
    num_qubits = len(encoded_sample)

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for qubit, value in enumerate(encoded_sample):
        qc.ry(single_scale * float(value), qubit)
        qc.rz(phase_scale * float(value), qubit)
    for qubit, value in enumerate(encoded_sample[:-1] * encoded_sample[1:]):
        qc.rzz(pair_scale * float(value), qubit, qubit + 1)
    return qc


def pauli_shadow_surrogate(
    state: Statevector,
    *,
    shots: int | None,
    rng: np.random.Generator,
    feature_mappings: list[dict[int, str]],
) -> np.ndarray:
    features: list[float] = []
    num_qubits = state.num_qubits
    for mapping in feature_mappings:
        features.append(expectation_with_shot_noise(state, operator(num_qubits, mapping), shots, rng))
    return np.asarray(features, dtype=np.float64)


def pauli_feature_mappings(num_qubits: int, family: str = "local") -> list[dict[int, str]]:
    mappings: list[dict[int, str]] = []
    for qubit in range(num_qubits):
        for gate in ("X", "Y", "Z"):
            mappings.append({qubit: gate})
    if family == "local":
        for qubit in range(num_qubits - 1):
            mappings.append({qubit: "X", qubit + 1: "X"})
            mappings.append({qubit: "Z", qubit + 1: "Z"})
        return mappings
    if family == "all-pairs":
        for left in range(num_qubits - 1):
            for right in range(left + 1, num_qubits):
                mappings.append({left: "X", right: "X"})
                mappings.append({left: "Z", right: "Z"})
        return mappings
    raise ValueError(f"unsupported readout family: {family}")


def local_pauli_shadow_surrogate(
    state: Statevector,
    *,
    shots: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    return pauli_shadow_surrogate(
        state,
        shots=shots,
        rng=rng,
        feature_mappings=pauli_feature_mappings(state.num_qubits, family="local"),
    )


def local_pauli_feature_mappings(num_qubits: int) -> list[dict[int, str]]:
    return pauli_feature_mappings(num_qubits, family="local")


def measurement_circuit_for_mapping(base_circuit: QuantumCircuit, mapping: dict[int, str]) -> QuantumCircuit:
    qc = base_circuit.copy()
    for qubit, gate in mapping.items():
        if gate == "X":
            qc.h(qubit)
        elif gate == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif gate != "Z":
            raise ValueError(f"unsupported Pauli gate for measurement: {gate}")
    qc.measure_all()
    return qc


def expectation_from_counts(
    counts: Mapping[Any, Any],
    *,
    mapping: dict[int, str],
    num_qubits: int,
) -> float:
    total = float(sum(int(round(float(value))) for value in counts.values()))
    if total <= 0.0:
        return 0.0

    signed_sum = 0.0
    for key, value in counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_qubits}b")
        else:
            bitstring = str(key).replace(" ", "").zfill(num_qubits)
        parity = 1.0
        for qubit in mapping:
            if bitstring[num_qubits - 1 - qubit] == "1":
                parity *= -1.0
        signed_sum += parity * float(value)
    return float(signed_sum / total)


def _bit_from_qiskit_bitstring(bitstring: str, qubit: int, n_qubits: int) -> int:
    cleaned = bitstring.replace(" ", "").zfill(n_qubits)
    return 1 if cleaned[n_qubits - 1 - qubit] == "1" else 0


def _aggregate_counts_on_qubits(
    counts: Mapping[Any, Any],
    *,
    n_qubits: int,
    qubits_desc: list[int],
) -> dict[str, int]:
    aggregated: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{n_qubits}b")
        else:
            bitstring = str(key).replace(" ", "").zfill(n_qubits)
        sub = "".join("1" if _bit_from_qiskit_bitstring(bitstring, q, n_qubits) else "0" for q in qubits_desc)
        aggregated[sub] = aggregated.get(sub, 0) + int(round(float(value)))
    return aggregated


def _counts_to_prob_vector(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    probs = np.zeros(2**n_qubits, dtype=float)
    total = 0.0
    for bitstring, value in counts.items():
        idx = int(bitstring.replace(" ", "").zfill(n_qubits), 2)
        probs[idx] += float(value)
        total += float(value)
    if total <= 0.0:
        raise ValueError("Histogram is leeg; kan geen kansen vormen.")
    probs /= total
    return probs


def _estimate_local_readout_params(
    counts_all0: Mapping[str, int],
    counts_all1: Mapping[str, int],
    *,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    shots0 = float(sum(counts_all0.values()))
    shots1 = float(sum(counts_all1.values()))
    if shots0 <= 0.0 or shots1 <= 0.0:
        raise ValueError("Calibration shots ontbreken.")

    ones_given_0 = np.zeros(n_qubits, dtype=float)
    zeros_given_1 = np.zeros(n_qubits, dtype=float)
    for bitstring, value in counts_all0.items():
        for qubit in range(n_qubits):
            if _bit_from_qiskit_bitstring(bitstring, qubit, n_qubits) == 1:
                ones_given_0[qubit] += float(value)
    for bitstring, value in counts_all1.items():
        for qubit in range(n_qubits):
            if _bit_from_qiskit_bitstring(bitstring, qubit, n_qubits) == 0:
                zeros_given_1[qubit] += float(value)

    p01 = np.clip(ones_given_0 / shots0, 0.0, 0.499999)
    p10 = np.clip(zeros_given_1 / shots1, 0.0, 0.499999)
    return p01, p10, {
        "shots_all0": int(shots0),
        "shots_all1": int(shots1),
        "p01_per_qubit": p01.tolist(),
        "p10_per_qubit": p10.tolist(),
        "p01_mean": float(np.mean(p01)),
        "p10_mean": float(np.mean(p10)),
    }


def _build_assignment_for_qubits(p01: np.ndarray, p10: np.ndarray, qubits_desc: list[int]) -> np.ndarray:
    if not qubits_desc:
        raise ValueError("mitigation qubit subset is leeg")
    mats: list[np.ndarray] = []
    for qubit in qubits_desc:
        mats.append(
            np.asarray(
                [
                    [1.0 - p01[qubit], p10[qubit]],
                    [p01[qubit], 1.0 - p10[qubit]],
                ],
                dtype=float,
            )
        )
    assignment = mats[0]
    for mat in mats[1:]:
        assignment = np.kron(assignment, mat)
    return assignment


def mitigated_expectation_from_counts(
    counts: Mapping[Any, Any],
    *,
    mapping: dict[int, str],
    num_qubits: int,
    p01: np.ndarray,
    p10: np.ndarray,
) -> tuple[float, dict[str, float]]:
    qubits_desc = sorted(mapping.keys(), reverse=True)
    sub_counts = _aggregate_counts_on_qubits(counts, n_qubits=num_qubits, qubits_desc=qubits_desc)
    p_obs = _counts_to_prob_vector(sub_counts, n_qubits=len(qubits_desc))
    assignment = _build_assignment_for_qubits(p01, p10, qubits_desc)
    p_true = np.linalg.pinv(assignment, rcond=1e-12) @ p_obs
    neg_mass = float(-np.minimum(p_true, 0.0).sum())
    p_true = np.clip(p_true, 0.0, None)
    norm = float(p_true.sum())
    if norm <= 0.0:
        raise ValueError("Mitigated distribution normalisatie faalde.")
    p_true /= norm

    expectation = 0.0
    for idx, prob in enumerate(p_true):
        bitstring = format(idx, f"0{len(qubits_desc)}b")
        parity = 1.0
        for bit in bitstring:
            if bit == "1":
                parity *= -1.0
        expectation += parity * float(prob)
    return float(expectation), {"negative_mass_before_clip": neg_mass}


def _normalize_counts(counts: Mapping[Any, Any], *, num_bits: int) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_bits}b")
        else:
            bitstring = str(key).replace(" ", "").zfill(num_bits)
        normalized[bitstring] = int(round(float(value)))
    return normalized


def _quasi_to_counts(quasi_dist: Mapping[Any, Any], *, shots: int, num_bits: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, probability in quasi_dist.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_bits}b")
        else:
            bitstring = str(key).replace(" ", "").zfill(num_bits)
        counts[bitstring] = int(round(float(probability) * shots))
    return counts


def _extract_counts_list_from_sampler_result(
    result: Any,
    *,
    shots: int,
    num_bits: int,
    n_items: int,
) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    item_extraction_failed = False
    for item_index in range(n_items):
        item = None
        if isinstance(result, (list, tuple)):
            if item_index < len(result):
                item = result[item_index]
        elif hasattr(result, "__getitem__"):
            try:
                item = result[item_index]
            except Exception:
                item = None
        if item is None:
            item_extraction_failed = True
            output = []
            break

        counts = None
        data = getattr(item, "data", None)
        if data is not None:
            for register_name in ("c", "meas"):
                register = getattr(data, register_name, None)
                if register is not None and hasattr(register, "get_counts"):
                    counts = register.get_counts()
                    break
            if counts is None and hasattr(data, "get_counts"):
                counts = data.get_counts()
        if counts is None and hasattr(item, "get_counts"):
            counts = item.get_counts()
        if counts is None:
            item_extraction_failed = True
            output = []
            break
        output.append(_normalize_counts(counts, num_bits=num_bits))
    if not item_extraction_failed and len(output) == n_items:
        return output

    if hasattr(result, "quasi_dists"):
        quasi_dists = getattr(result, "quasi_dists")
        if len(quasi_dists) < n_items:
            raise RuntimeError(f"Sampler result has {len(quasi_dists)} quasi distributions, expected >= {n_items}.")
        return [_quasi_to_counts(quasi_dist, shots=shots, num_bits=num_bits) for quasi_dist in quasi_dists[:n_items]]

    raise RuntimeError("Could not extract counts from sampler result.")


def _measurement_bit_count_for_circuits(
    original_circuits: list[QuantumCircuit],
    transpiled_circuits: list[QuantumCircuit] | None = None,
) -> int:
    candidates: list[int] = []
    if transpiled_circuits:
        candidates.extend(int(circuit.num_clbits) for circuit in transpiled_circuits if int(circuit.num_clbits) > 0)
    candidates.extend(int(circuit.num_clbits) for circuit in original_circuits if int(circuit.num_clbits) > 0)
    candidates.extend(int(circuit.num_qubits) for circuit in original_circuits if int(circuit.num_qubits) > 0)
    if not candidates:
        raise ValueError("Could not infer measurement bit-count for circuits.")
    return min(candidates)


def _backend_name(backend: Any) -> str:
    if hasattr(backend, "name"):
        name = backend.name
        if callable(name):
            try:
                return str(name())
            except Exception:
                pass
        return str(name)
    return backend.__class__.__name__


def _build_runtime_service() -> Any:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit-ibm-runtime ontbreekt. Installeer het in je qiskit-venv.") from exc

    token = (
        os.getenv("QCAPI_TOKEN")
        or os.getenv("QISKIT_IBM_TOKEN")
        or os.getenv("IBM_QUANTUM_TOKEN")
    )
    instance = os.getenv("QISKIT_IBM_INSTANCE")
    if token:
        kwargs: dict[str, str] = {"token": token}
        if instance:
            kwargs["instance"] = instance
        try:
            return QiskitRuntimeService(channel="ibm_quantum", **kwargs)
        except TypeError:
            return QiskitRuntimeService(**kwargs)
    return QiskitRuntimeService()


def _refetch_runtime_job_result(job_id: str, *, retries: int = 3, base_delay_seconds: float = 1.0) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            fresh_service = _build_runtime_service()
            return fresh_service.job(str(job_id)).result()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(base_delay_seconds * float(attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Could not refresh runtime result for job {job_id}.")


def _select_runtime_backend(service: Any, requested_backend: str | None, *, min_qubits: int) -> Any:
    if requested_backend:
        return service.backend(requested_backend)

    candidates = service.backends(simulator=False, operational=True, min_num_qubits=min_qubits)
    if not candidates:
        candidates = service.backends(simulator=False, min_num_qubits=min_qubits)
    if not candidates:
        raise RuntimeError(f"Geen geschikte hardware-backends gevonden voor >= {min_qubits} qubits.")
    try:
        from qiskit_ibm_runtime import least_busy

        return least_busy(candidates)
    except Exception:
        scored = []
        for backend in candidates:
            pending = 10**9
            try:
                pending = int(backend.status().pending_jobs)
            except Exception:
                pass
            scored.append((pending, _backend_name(backend), backend))
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2]


def _backend_coupling_edges(backend: Any) -> list[tuple[int, int]]:
    coupling_map = getattr(backend, "coupling_map", None)
    edges: list[tuple[int, int]] = []
    try:
        if coupling_map is not None and hasattr(coupling_map, "get_edges"):
            edges = [(int(left), int(right)) for left, right in coupling_map.get_edges()]
    except Exception:
        edges = []
    if edges:
        return edges
    try:
        config = backend.configuration()
        raw = getattr(config, "coupling_map", None) or []
        return [(int(left), int(right)) for left, right in raw]
    except Exception:
        return []


def _backend_readout_error_map(backend: Any) -> dict[int, float]:
    try:
        props = backend.properties()
    except Exception:
        return {}
    if props is None:
        return {}
    errors: dict[int, float] = {}
    for qubit in range(int(getattr(backend, "num_qubits", 0) or 0)):
        try:
            value = props.readout_error(qubit)
        except Exception:
            continue
        if value is None:
            continue
        errors[int(qubit)] = float(value)
    return errors


def _backend_two_qubit_error_map(backend: Any) -> dict[tuple[int, int], float]:
    try:
        props = backend.properties()
    except Exception:
        return {}
    if props is None:
        return {}
    errors: dict[tuple[int, int], float] = {}
    for gate in getattr(props, "gates", []):
        qubits = tuple(int(qubit) for qubit in getattr(gate, "qubits", []))
        if len(qubits) != 2:
            continue
        pair = tuple(sorted(qubits))
        gate_error = None
        for parameter in getattr(gate, "parameters", []):
            if getattr(parameter, "name", "") == "gate_error":
                try:
                    gate_error = float(parameter.value)
                except Exception:
                    gate_error = None
                break
        if gate_error is None:
            continue
        prev = errors.get(pair)
        if prev is None or gate_error < prev:
            errors[pair] = gate_error
    return errors


def _quality_chain_layout(
    backend: Any,
    *,
    num_qubits: int,
    beam_width: int,
) -> tuple[list[int] | None, dict[str, object]]:
    edges = _backend_coupling_edges(backend)
    adjacency: dict[int, set[int]] = {}
    for left, right in edges:
        if left == right:
            continue
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    if len(adjacency) < num_qubits:
        return None, {
            "layout_strategy": "quality-chain",
            "layout_found": False,
            "layout_reason": "insufficient_connected_qubits",
        }

    readout_error = _backend_readout_error_map(backend)
    edge_error = _backend_two_qubit_error_map(backend)
    default_node_cost = float(statistics.median(readout_error.values())) if readout_error else 0.02
    default_edge_cost = float(statistics.median(edge_error.values())) if edge_error else 0.02

    start_nodes = sorted(
        adjacency,
        key=lambda qubit: (
            float(readout_error.get(qubit, default_node_cost)),
            -len(adjacency.get(qubit, ())),
            int(qubit),
        ),
    )
    beam: list[tuple[float, tuple[int, ...], frozenset[int]]] = [
        (float(readout_error.get(qubit, default_node_cost)), (int(qubit),), frozenset([int(qubit)]))
        for qubit in start_nodes[: max(int(beam_width), 8)]
    ]

    for _ in range(1, num_qubits):
        next_beam: list[tuple[float, tuple[int, ...], frozenset[int]]] = []
        for cost, path, used in beam:
            last = int(path[-1])
            neighbors = [int(neighbor) for neighbor in adjacency.get(last, ()) if int(neighbor) not in used]
            neighbors.sort(
                key=lambda neighbor: (
                    float(edge_error.get(tuple(sorted((last, neighbor))), default_edge_cost))
                    + float(readout_error.get(neighbor, default_node_cost)),
                    -len(adjacency.get(neighbor, ())),
                    int(neighbor),
                )
            )
            for neighbor in neighbors[: max(int(beam_width), 8)]:
                pair = tuple(sorted((last, neighbor)))
                step_cost = float(edge_error.get(pair, default_edge_cost)) + float(
                    readout_error.get(neighbor, default_node_cost)
                )
                next_beam.append((cost + step_cost, path + (neighbor,), used | {neighbor}))
        if not next_beam:
            break
        next_beam.sort(key=lambda item: (float(item[0]), item[1]))
        beam = next_beam[: max(int(beam_width), 8)]

    complete = [item for item in beam if len(item[1]) == num_qubits]
    if not complete:
        return None, {
            "layout_strategy": "quality-chain",
            "layout_found": False,
            "layout_reason": "beam_search_failed",
        }

    _, best_path, _ = complete[0]
    layout = [int(qubit) for qubit in best_path]
    readout_vals = [float(readout_error.get(qubit, default_node_cost)) for qubit in layout]
    edge_vals = [
        float(edge_error.get(tuple(sorted((layout[index], layout[index + 1]))), default_edge_cost))
        for index in range(len(layout) - 1)
    ]
    return layout, {
        "layout_strategy": "quality-chain",
        "layout_found": True,
        "initial_layout": layout,
        "layout_mean_readout_error": float(np.mean(readout_vals)) if readout_vals else None,
        "layout_max_readout_error": float(np.max(readout_vals)) if readout_vals else None,
        "layout_mean_two_qubit_error": float(np.mean(edge_vals)) if edge_vals else None,
        "layout_max_two_qubit_error": float(np.max(edge_vals)) if edge_vals else None,
    }


def _select_runtime_layout(
    backend: Any,
    *,
    num_qubits: int,
    execution_config: QuantumExecutionConfig,
) -> tuple[list[int] | None, dict[str, object]]:
    if execution_config.layout_strategy == "none":
        return None, {"layout_strategy": "none", "layout_found": False}
    if execution_config.layout_strategy == "quality-chain":
        return _quality_chain_layout(
            backend,
            num_qubits=num_qubits,
            beam_width=max(int(execution_config.layout_beam_width), 8),
        )
    return None, {
        "layout_strategy": str(execution_config.layout_strategy),
        "layout_found": False,
        "layout_reason": "unknown_strategy",
    }


def runtime_sampler_options(execution_config: QuantumExecutionConfig) -> dict[str, object] | None:
    if not execution_config.extra_error_suppression:
        return None
    return {
        "dynamical_decoupling": {
            "enable": True,
            "sequence_type": execution_config.dd_sequence,
            "scheduling_method": "alap",
        },
        "twirling": {
            "enable_gates": True,
            "num_randomizations": int(execution_config.twirl_randomizations),
            "strategy": "active-accum",
        },
    }


def run_measurement_circuits_local(
    circuits: list[QuantumCircuit],
    *,
    shots: int,
    optimization_level: int,
    simulator_method: str = "automatic",
) -> tuple[list[dict[str, int]], dict[str, object]]:
    from qiskit import transpile

    backend = None
    backend_label = ""
    try:
        from qiskit_aer import AerSimulator

        if simulator_method == "automatic":
            backend = AerSimulator()
            backend_label = "AerSimulator"
        else:
            backend = AerSimulator(method=simulator_method)
            backend_label = f"AerSimulator[{simulator_method}]"
    except Exception:
        if simulator_method != "automatic":
            raise RuntimeError(
                f"Lokale simulatormethode '{simulator_method}' vereist qiskit-aer met die methode beschikbaar."
            )
        try:
            from qiskit.providers.basic_provider import BasicSimulator

            backend = BasicSimulator()
            backend_label = "BasicSimulator"
        except Exception as exc:
            raise RuntimeError("Geen lokale simulator gevonden. Installeer qiskit-aer of update qiskit.") from exc

    transpiled = transpile(circuits, backend=backend, optimization_level=optimization_level)
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    raw_counts = result.get_counts()
    if isinstance(raw_counts, dict):
        raw_counts_list = [raw_counts]
    else:
        raw_counts_list = list(raw_counts)
    counts_list = []
    for index, counts in enumerate(raw_counts_list):
        num_bits = int(transpiled[index].num_clbits) if int(transpiled[index].num_clbits) > 0 else int(circuits[index].num_clbits)
        if num_bits <= 0:
            num_bits = int(circuits[index].num_qubits)
        counts_list.append(_normalize_counts(counts, num_bits=num_bits))
    return counts_list, {
        "mode": "sampler-sim",
        "backend_name": backend_label,
        "simulator_method": simulator_method,
        "job_id": None,
        "circuit_count": len(circuits),
        "shots": shots,
    }


def run_measurement_circuits_ibm(
    circuits: list[QuantumCircuit],
    *,
    shots: int,
    backend_name: str | None,
    optimization_level: int,
    execution_config: QuantumExecutionConfig,
    sampler_options: dict[str, object] | None = None,
) -> tuple[list[dict[str, int]], dict[str, object]]:
    from qiskit import transpile

    if not circuits:
        raise ValueError("run_measurement_circuits_ibm requires at least one circuit")

    _runtime_debug(
        execution_config,
        "ibm_measure_start",
        shots=shots,
        backend_name=backend_name,
        optimization_level=optimization_level,
        **_circuit_batch_summary(circuits),
    )
    service = _build_runtime_service()
    min_qubits = max(circuit.num_qubits for circuit in circuits)
    backend = _select_runtime_backend(service, backend_name, min_qubits=min_qubits)
    _runtime_debug(
        execution_config,
        "ibm_backend_selected",
        backend=_backend_name(backend),
        min_qubits=min_qubits,
    )
    initial_layout, layout_metadata = _select_runtime_layout(
        backend,
        num_qubits=min_qubits,
        execution_config=execution_config,
    )
    _runtime_debug(
        execution_config,
        "ibm_layout_selected",
        **layout_metadata,
    )
    transpile_kwargs: dict[str, object] = {
        "backend": backend,
        "optimization_level": optimization_level,
    }
    if initial_layout is not None:
        transpile_kwargs["initial_layout"] = initial_layout
        transpile_kwargs["layout_method"] = "sabre"
        transpile_kwargs["routing_method"] = "sabre"
    try:
        transpiled = transpile(circuits, **transpile_kwargs)
    except Exception as exc:
        _runtime_debug(execution_config, "ibm_transpile_failed", error=repr(exc))
        traceback.print_exc(file=sys.stderr)
        raise
    _runtime_debug(
        execution_config,
        "ibm_transpile_done",
        **_circuit_batch_summary(transpiled),
    )
    measurement_bits = _measurement_bit_count_for_circuits(circuits, transpiled)
    _runtime_debug(
        execution_config,
        "ibm_measurement_bits",
        measurement_bits=measurement_bits,
    )
    submit_batch_size = max(int(getattr(execution_config, "runtime_submit_batch_size", 0) or 0), 0)
    transpiled_chunks = _chunk_circuits(list(transpiled), submit_batch_size)
    _runtime_debug(
        execution_config,
        "ibm_submit_plan",
        submit_batch_size=submit_batch_size,
        submit_batch_count=len(transpiled_chunks),
        transpiled_circuit_count=len(transpiled),
    )

    counts_list: list[dict[str, int]] = []
    job_ids: list[str] = []
    result_sources: list[str] = []
    primitive_label: str | None = None

    try:
        from qiskit_ibm_runtime import SamplerV2

        if sampler_options:
            sampler = SamplerV2(mode=backend, options=sampler_options)
        else:
            sampler = SamplerV2(mode=backend)
        _runtime_debug(execution_config, "ibm_sampler_v2_ready", sampler_options=sampler_options)
        primitive_label = "SamplerV2"
        for chunk_index, transpiled_chunk in enumerate(transpiled_chunks):
            _runtime_debug(
                execution_config,
                "ibm_submit_chunk_start",
                chunk_index=chunk_index,
                chunk_count=len(transpiled_chunks),
                **_circuit_batch_summary(transpiled_chunk),
            )
            try:
                job = sampler.run(transpiled_chunk, shots=shots)
            except Exception as exc:
                _runtime_debug(
                    execution_config,
                    "ibm_sampler_v2_run_failed",
                    error=repr(exc),
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )
                traceback.print_exc(file=sys.stderr)
                raise
            job_id = str(job.job_id())
            job_ids.append(job_id)
            _runtime_debug(
                execution_config,
                "ibm_sampler_v2_submitted",
                job_id=job_id,
                chunk_index=chunk_index,
                chunk_count=len(transpiled_chunks),
            )
            try:
                result = job.result()
            except Exception as exc:
                _runtime_debug(
                    execution_config,
                    "ibm_sampler_v2_result_failed",
                    job_id=job_id,
                    error=repr(exc),
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )
                traceback.print_exc(file=sys.stderr)
                raise
            result_source = "job.result"
            try:
                result = _refetch_runtime_job_result(job_id)
                result_source = "fresh-service.job.result"
            except Exception:
                pass
            result_sources.append(result_source)
            _runtime_debug(
                execution_config,
                "ibm_sampler_v2_result_done",
                job_id=job_id,
                result_source=result_source,
                chunk_index=chunk_index,
                chunk_count=len(transpiled_chunks),
            )
            try:
                counts_chunk = _extract_counts_list_from_sampler_result(
                    result,
                    shots=shots,
                    num_bits=measurement_bits,
                    n_items=len(transpiled_chunk),
                )
            except Exception as exc:
                _runtime_debug(
                    execution_config,
                    "ibm_extract_counts_failed",
                    error=repr(exc),
                    primitive=primitive_label,
                    result_source=result_source,
                    n_items=len(transpiled_chunk),
                    measurement_bits=measurement_bits,
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )
                traceback.print_exc(file=sys.stderr)
                raise
            counts_list.extend(counts_chunk)
            _runtime_debug(
                execution_config,
                "ibm_extract_counts_done",
                count_batches=len(counts_chunk),
                primitive=primitive_label,
                job_id=job_id,
                chunk_index=chunk_index,
                chunk_count=len(transpiled_chunks),
            )
    except ImportError:
        from qiskit_ibm_runtime import Sampler, Session

        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session)
            _runtime_debug(execution_config, "ibm_sampler_v1_ready")
            primitive_label = "SamplerV1"
            for chunk_index, transpiled_chunk in enumerate(transpiled_chunks):
                _runtime_debug(
                    execution_config,
                    "ibm_submit_chunk_start",
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                    **_circuit_batch_summary(transpiled_chunk),
                )
                try:
                    job = sampler.run(transpiled_chunk, shots=shots)
                except Exception as exc:
                    _runtime_debug(
                        execution_config,
                        "ibm_sampler_v1_run_failed",
                        error=repr(exc),
                        chunk_index=chunk_index,
                        chunk_count=len(transpiled_chunks),
                    )
                    traceback.print_exc(file=sys.stderr)
                    raise
                job_id = str(job.job_id())
                job_ids.append(job_id)
                _runtime_debug(
                    execution_config,
                    "ibm_sampler_v1_submitted",
                    job_id=job_id,
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )
                try:
                    result = job.result()
                except Exception as exc:
                    _runtime_debug(
                        execution_config,
                        "ibm_sampler_v1_result_failed",
                        job_id=job_id,
                        error=repr(exc),
                        chunk_index=chunk_index,
                        chunk_count=len(transpiled_chunks),
                    )
                    traceback.print_exc(file=sys.stderr)
                    raise
                result_source = "job.result"
                try:
                    result = _refetch_runtime_job_result(job_id)
                    result_source = "fresh-service.job.result"
                except Exception:
                    pass
                result_sources.append(result_source)
                _runtime_debug(
                    execution_config,
                    "ibm_sampler_v1_result_done",
                    job_id=job_id,
                    result_source=result_source,
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )
                try:
                    counts_chunk = _extract_counts_list_from_sampler_result(
                        result,
                        shots=shots,
                        num_bits=measurement_bits,
                        n_items=len(transpiled_chunk),
                    )
                except Exception as exc:
                    _runtime_debug(
                        execution_config,
                        "ibm_extract_counts_failed",
                        error=repr(exc),
                        primitive=primitive_label,
                        result_source=result_source,
                        n_items=len(transpiled_chunk),
                        measurement_bits=measurement_bits,
                        chunk_index=chunk_index,
                        chunk_count=len(transpiled_chunks),
                    )
                    traceback.print_exc(file=sys.stderr)
                    raise
                counts_list.extend(counts_chunk)
                _runtime_debug(
                    execution_config,
                    "ibm_extract_counts_done",
                    count_batches=len(counts_chunk),
                    primitive=primitive_label,
                    job_id=job_id,
                    chunk_index=chunk_index,
                    chunk_count=len(transpiled_chunks),
                )

    primary_job_id = job_ids[0] if job_ids else None
    primary_result_source = result_sources[0] if result_sources else None
    return counts_list, {
        "mode": "ibm-hardware",
        "backend_name": _backend_name(backend),
        "job_id": primary_job_id,
        "job_ids": job_ids,
        "primitive": primitive_label,
        "result_source": primary_result_source,
        "result_sources": result_sources,
        "circuit_count": len(transpiled),
        "measurement_bits": measurement_bits,
        "shots": shots,
        "submit_batch_count": len(transpiled_chunks),
        "submit_batch_size": submit_batch_size,
        **layout_metadata,
    }


def prepare_readout_calibration(
    *,
    num_qubits: int,
    execution_config: QuantumExecutionConfig,
) -> dict[str, object] | None:
    if not execution_config.readout_mitigation:
        return None

    cache_key = (
        f"readout_cal::{execution_config.mode}::{execution_config.backend_name or 'auto'}::{num_qubits}"
        f"::shots={execution_config.cal_shots}"
    )
    cached = execution_config.runtime_cache.get(cache_key)
    if cached is not None:
        _runtime_debug(execution_config, "readout_calibration_cache_hit", cache_key=cache_key)
        return cached
    _runtime_debug(
        execution_config,
        "readout_calibration_start",
        cache_key=cache_key,
        num_qubits=num_qubits,
        mode=execution_config.mode,
        cal_shots=execution_config.cal_shots,
    )

    cal0 = QuantumCircuit(num_qubits)
    cal0.measure_all()
    cal1 = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        cal1.x(qubit)
    cal1.measure_all()

    if execution_config.mode == "sampler-sim":
        counts_list, metadata = run_measurement_circuits_local(
            [cal0, cal1],
            shots=execution_config.cal_shots,
            optimization_level=execution_config.optimization_level,
            simulator_method=execution_config.simulator_method,
        )
    elif execution_config.mode == "ibm-hardware":
        counts_list, metadata = run_measurement_circuits_ibm(
            [cal0, cal1],
            shots=execution_config.cal_shots,
            backend_name=execution_config.backend_name,
            optimization_level=execution_config.optimization_level,
            execution_config=execution_config,
            sampler_options=None,
        )
        job_id = metadata.get("job_id")
        if job_id:
            try:
                refreshed_result = _refetch_runtime_job_result(str(job_id))
                counts_list = _extract_counts_list_from_sampler_result(
                    refreshed_result,
                    shots=execution_config.cal_shots,
                    num_bits=num_qubits,
                    n_items=2,
                )
            except Exception:
                pass
    else:
        return None

    p01, p10, cal_info = _estimate_local_readout_params(counts_list[0], counts_list[1], n_qubits=num_qubits)
    _runtime_debug(
        execution_config,
        "readout_calibration_done",
        p01_mean=float(np.mean(p01)),
        p10_mean=float(np.mean(p10)),
        job_id=metadata.get("job_id"),
    )
    calibration = {
        "p01": p01,
        "p10": p10,
        "metadata": {
            **metadata,
            **cal_info,
            "local_model": "independent per-qubit asymmetric readout",
        },
    }
    execution_config.runtime_cache[cache_key] = calibration
    return calibration


def extract_pauli_features(
    circuits: list[QuantumCircuit],
    *,
    shots: int | None,
    rng: np.random.Generator,
    execution_config: QuantumExecutionConfig,
    readout_family: str = "local",
) -> QuantumFeatureBatch:
    if not circuits:
        raise ValueError("at least one circuit is required")
    num_qubits = circuits[0].num_qubits
    if any(circuit.num_qubits != num_qubits for circuit in circuits):
        raise ValueError("all circuits must have the same qubit count")

    if execution_config.mode == "statevector":
        feature_mappings = pauli_feature_mappings(num_qubits, family=readout_family)
        features = np.asarray(
            [
                pauli_shadow_surrogate(
                    Statevector.from_instruction(circuit),
                    shots=shots,
                    rng=rng,
                    feature_mappings=feature_mappings,
                )
                for circuit in circuits
            ],
            dtype=np.float64,
        )
        return QuantumFeatureBatch(
            features=features,
            metadata={
                "mode": "statevector",
                "backend_name": "Statevector",
                "circuit_count": len(circuits),
                "readout_family": readout_family,
            },
        )

    if shots is None or shots <= 0:
        raise ValueError(f"execution mode '{execution_config.mode}' requires readout shots > 0")

    feature_mappings = pauli_feature_mappings(num_qubits, family=readout_family)
    feature_mapping_limit = max(int(getattr(execution_config, "feature_mapping_limit", 0) or 0), 0)
    if feature_mapping_limit > 0:
        feature_mappings = feature_mappings[:feature_mapping_limit]
    _runtime_debug(
        execution_config,
        "extract_pauli_features_start",
        num_circuits=len(circuits),
        num_qubits=num_qubits,
        shots=shots,
        readout_family=readout_family,
        feature_mapping_limit=feature_mapping_limit,
        feature_mapping_count=len(feature_mappings),
    )
    calibration = prepare_readout_calibration(num_qubits=num_qubits, execution_config=execution_config)
    measured_circuits = [
        measurement_circuit_for_mapping(circuit, mapping)
        for circuit in circuits
        for mapping in feature_mappings
    ]
    _runtime_debug(
        execution_config,
        "extract_pauli_features_measured_circuits",
        measured_circuit_count=len(measured_circuits),
        **_circuit_batch_summary(measured_circuits),
    )
    if execution_config.mode == "sampler-sim":
        counts_list, metadata = run_measurement_circuits_local(
            measured_circuits,
            shots=shots,
            optimization_level=execution_config.optimization_level,
            simulator_method=execution_config.simulator_method,
        )
    elif execution_config.mode == "ibm-hardware":
        counts_list, metadata = run_measurement_circuits_ibm(
            measured_circuits,
            shots=shots,
            backend_name=execution_config.backend_name,
            optimization_level=execution_config.optimization_level,
            execution_config=execution_config,
            sampler_options=runtime_sampler_options(execution_config),
        )
    else:
        raise ValueError(f"unsupported execution mode: {execution_config.mode}")

    n_features = len(feature_mappings)
    features = np.zeros((len(circuits), n_features), dtype=np.float64)
    for circuit_index in range(len(circuits)):
        offset = circuit_index * n_features
        for feature_index, mapping in enumerate(feature_mappings):
            counts = counts_list[offset + feature_index]
            if calibration is None:
                features[circuit_index, feature_index] = expectation_from_counts(
                    counts,
                    mapping=mapping,
                    num_qubits=num_qubits,
                )
            else:
                mitigated, _ = mitigated_expectation_from_counts(
                    counts,
                    mapping=mapping,
                    num_qubits=num_qubits,
                    p01=calibration["p01"],
                    p10=calibration["p10"],
                )
                features[circuit_index, feature_index] = mitigated
    metadata = dict(metadata)
    metadata["feature_count"] = n_features
    metadata["feature_mapping_limit"] = feature_mapping_limit
    metadata["feature_measurement_circuit_count"] = len(measured_circuits)
    metadata["readout_mitigation"] = bool(calibration is not None)
    metadata["readout_family"] = readout_family
    if calibration is not None:
        metadata["calibration"] = calibration["metadata"]
    sampler_options = runtime_sampler_options(execution_config)
    if sampler_options is not None:
        metadata["sampler_options"] = sampler_options
    _runtime_debug(
        execution_config,
        "extract_pauli_features_done",
        feature_count=n_features,
        measured_circuit_count=len(measured_circuits),
        metadata_mode=metadata.get("mode"),
        job_id=metadata.get("job_id"),
    )
    return QuantumFeatureBatch(features=features, metadata=metadata)


def extract_local_pauli_features(
    circuits: list[QuantumCircuit],
    *,
    shots: int | None,
    rng: np.random.Generator,
    execution_config: QuantumExecutionConfig,
) -> QuantumFeatureBatch:
    return extract_pauli_features(
        circuits,
        shots=shots,
        rng=rng,
        execution_config=execution_config,
        readout_family="local",
    )


def feature_score(
    model_features: np.ndarray,
    encoded_sample: np.ndarray,
    *,
    shots: int | None,
    rng: np.random.Generator,
    execution_config: QuantumExecutionConfig,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
    readout_family: str = "local",
) -> float:
    query_features = query_feature_vector(
        encoded_sample,
        shots=shots,
        rng=rng,
        execution_config=execution_config,
        single_scale=single_scale,
        phase_scale=phase_scale,
        pair_scale=pair_scale,
        readout_family=readout_family,
    )
    return cosine_similarity(model_features, query_features)


def query_feature_vector(
    encoded_sample: np.ndarray,
    *,
    shots: int | None,
    rng: np.random.Generator,
    execution_config: QuantumExecutionConfig,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
    readout_family: str = "local",
) -> np.ndarray:
    batch = extract_pauli_features(
        [
            query_circuit(
                encoded_sample,
                single_scale=single_scale,
                phase_scale=phase_scale,
                pair_scale=pair_scale,
            )
        ],
        shots=shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    return batch.features[0]


def quantum_head_feature_vector(model_features: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    cosine = cosine_similarity(model_features, query_features)
    interaction = model_features * query_features
    return np.concatenate([query_features, interaction, np.array([cosine], dtype=np.float64)])


def quantum_head_scores(
    *,
    model_features: np.ndarray,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train: np.ndarray,
    head_method: str,
    shots: int | None,
    rng: np.random.Generator,
    execution_config: QuantumExecutionConfig,
    single_scale: float,
    phase_scale: float,
    pair_scale: float,
    readout_family: str = "local",
) -> dict[str, object]:
    query_circuits = [
        query_circuit(
            encoded_sample,
            single_scale=single_scale,
            phase_scale=phase_scale,
            pair_scale=pair_scale,
        )
        for encoded_sample in np.concatenate([encoded_train, encoded_test], axis=0)
    ]
    query_batch = extract_pauli_features(
        query_circuits,
        shots=shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    query_train = np.asarray(query_batch.features[: len(encoded_train)], dtype=np.float64)
    query_test = np.asarray(query_batch.features[len(encoded_train) :], dtype=np.float64)
    cosine_train = np.asarray([cosine_similarity(model_features, row) for row in query_train], dtype=np.float64)
    cosine_test = np.asarray([cosine_similarity(model_features, row) for row in query_test], dtype=np.float64)

    if head_method == "cosine":
        return {
            "train_scores": cosine_train,
            "test_scores": cosine_test,
            "query_feature_count": int(query_train.shape[1]),
            "head_feature_count": 1,
            "execution_metadata": query_batch.metadata,
        }

    head_train_raw = np.asarray(
        [quantum_head_feature_vector(model_features, row) for row in query_train],
        dtype=np.float64,
    )
    head_test_raw = np.asarray(
        [quantum_head_feature_vector(model_features, row) for row in query_test],
        dtype=np.float64,
    )
    head_train, head_test = standardize(head_train_raw, head_test_raw)

    if head_method == "ridge":
        head_w = ridge_linear_classifier(head_train, y_train)
        train_scores = head_train @ head_w
        test_scores = head_test @ head_w
    elif head_method == "logistic":
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(head_train, (y_train > 0.0).astype(np.int64))
        train_scores = clf.decision_function(head_train)
        test_scores = clf.decision_function(head_test)
    else:
        raise ValueError(f"unsupported quantum head method: {head_method}")

    return {
        "train_scores": np.asarray(train_scores, dtype=np.float64),
        "test_scores": np.asarray(test_scores, dtype=np.float64),
        "query_feature_count": int(query_train.shape[1]),
        "head_feature_count": int(head_train.shape[1]),
        "execution_metadata": query_batch.metadata,
    }


def ridge_linear_classifier(train_x: np.ndarray, train_y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    gram = train_x.T @ train_x + ridge * np.eye(train_x.shape[1])
    rhs = train_x.T @ train_y
    return np.linalg.solve(gram, rhs)


def top_principal_component(train_x: np.ndarray) -> np.ndarray:
    cov = train_x.T @ train_x / max(len(train_x), 1)
    evals, evecs = np.linalg.eigh(cov)
    return normalize(evecs[:, np.argmax(evals)])


def plot_summary(
    out_path: Path,
    *,
    cls_scores_q: np.ndarray,
    cls_scores_c: np.ndarray,
    cls_labels: np.ndarray,
    pca_scores_q: np.ndarray | None,
    pca_scores_c: np.ndarray | None,
    raw_feature_count: int,
    num_qubits: int,
    readout_feature_count: int,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = np.where(cls_labels > 0, "tab:blue", "tab:orange")

    axes[0, 0].scatter(cls_scores_c, cls_scores_q, c=colors, s=26, alpha=0.8, edgecolors="none")
    axes[0, 0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 0].set_title("Classification Scores")
    axes[0, 0].set_xlabel("Classical ridge score")
    axes[0, 0].set_ylabel("Quantum-toy score")

    axes[0, 1].hist(cls_scores_q[cls_labels > 0], bins=18, alpha=0.7, label="label +1")
    axes[0, 1].hist(cls_scores_q[cls_labels < 0], bins=18, alpha=0.7, label="label -1")
    axes[0, 1].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 1].set_title("Quantum Classification Score Histogram")
    axes[0, 1].set_xlabel("Toy score")
    axes[0, 1].legend()

    if pca_scores_q is not None and pca_scores_c is not None:
        axes[1, 0].scatter(pca_scores_c, pca_scores_q, c="tab:green", s=26, alpha=0.8, edgecolors="none")
        axes[1, 0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
        axes[1, 0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
        axes[1, 0].set_title("1D Reduction Scores")
        axes[1, 0].set_xlabel("Classical PCA projection")
        axes[1, 0].set_ylabel("Quantum-toy projection")
    else:
        axes[1, 0].text(0.5, 0.5, "classification-only run", ha="center", va="center")
        axes[1, 0].set_title("1D Reduction Scores")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

    axes[1, 1].bar(
        ["raw features", "qubits", "readout features"],
        [raw_feature_count, num_qubits, readout_feature_count],
        color=["tab:gray", "tab:purple", "tab:red"],
    )
    axes[1, 1].set_title("Model Size Proxy")
    axes[1, 1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_label_weight_diagnostics(
    out_path: Path,
    *,
    pos_mean_encoded: np.ndarray,
    neg_mean_encoded: np.ndarray,
    signed_gap_encoded: np.ndarray,
    pos_mean_pair: np.ndarray,
    neg_mean_pair: np.ndarray,
    signed_gap_pair: np.ndarray,
    cumulative_signed_mean: np.ndarray,
    quantum_scores: np.ndarray,
    labels: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    qubits = np.arange(len(pos_mean_encoded))
    width = 0.25
    axes[0, 0].bar(qubits - width, pos_mean_encoded, width=width, label="mean | y=+1")
    axes[0, 0].bar(qubits, neg_mean_encoded, width=width, label="mean | y=-1")
    axes[0, 0].bar(qubits + width, signed_gap_encoded, width=width, label="signed gap")
    axes[0, 0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 0].set_title("Per-Qubit Label-Weighted Contribution")
    axes[0, 0].set_xlabel("encoded qubit/block")
    axes[0, 0].set_ylabel("mean encoded value")
    axes[0, 0].set_xticks(qubits)
    axes[0, 0].legend()

    pair_idx = np.arange(len(pos_mean_pair))
    if len(pair_idx) > 0:
        axes[0, 1].bar(pair_idx - width, pos_mean_pair, width=width, label="mean pair | y=+1")
        axes[0, 1].bar(pair_idx, neg_mean_pair, width=width, label="mean pair | y=-1")
        axes[0, 1].bar(pair_idx + width, signed_gap_pair, width=width, label="signed gap")
        axes[0, 1].set_xticks(pair_idx)
        axes[0, 1].set_xlabel("adjacent pair")
    else:
        axes[0, 1].text(0.5, 0.5, "no pair terms for 1 qubit", ha="center", va="center")
        axes[0, 1].set_xticks([])
    axes[0, 1].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 1].set_title("Pair-Term Label Weighting")
    axes[0, 1].set_ylabel("mean pair value")
    axes[0, 1].legend(loc="best")

    steps = np.arange(1, cumulative_signed_mean.shape[0] + 1)
    for qubit in range(cumulative_signed_mean.shape[1]):
        axes[1, 0].plot(steps, cumulative_signed_mean[:, qubit], lw=1.5, label=f"q{qubit}")
    axes[1, 0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[1, 0].set_title("Cumulative Signed Sketch During Stream")
    axes[1, 0].set_xlabel("samples seen")
    axes[1, 0].set_ylabel("running mean of y * encoded_sample")
    axes[1, 0].legend(ncol=2, fontsize=8)

    axes[1, 1].hist(quantum_scores[labels > 0], bins=18, alpha=0.7, label="y=+1")
    axes[1, 1].hist(quantum_scores[labels < 0], bins=18, alpha=0.7, label="y=-1")
    axes[1, 1].set_title("Final Quantum Score by Label")
    axes[1, 1].set_xlabel("quantum toy score")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_classification_task(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    num_qubits: int,
    readout_shots: int | None,
    seed: int,
    encoder_method: str = "block",
    quantum_head_method: str = "cosine",
    readout_family: str = "local",
    execution_config: QuantumExecutionConfig | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    execution_config = execution_config or QuantumExecutionConfig()
    data = make_classification_data(n_train + n_test, n_features, num_qubits, rng)
    train_idx, test_idx = split_train_test(data.x, n_train=n_train, rng=rng)

    x_train_raw, x_test_raw = data.x[train_idx], data.x[test_idx]
    y_train, y_test = data.y[train_idx], data.y[test_idx]
    x_train_raw, y_train = subsample_rows(x_train_raw, y_train, max_rows=max_train_samples, rng=rng)
    x_test_raw, y_test = subsample_rows(x_test_raw, y_test, max_rows=max_test_samples, rng=rng)
    x_train, x_test = standardize(x_train_raw, x_test_raw)

    encoder = ToyEncoding.fit(
        x_train,
        num_qubits=num_qubits,
        rng=rng,
        method=encoder_method,
        y_train=y_train,
    )
    encoded_train = encoder.encode(x_train)
    encoded_test = encoder.encode(x_test)
    signed_train = y_train[:, None] * encoded_train
    cumulative_signed_mean = np.cumsum(signed_train, axis=0) / np.arange(1, len(signed_train) + 1)[:, None]
    pos_mean_encoded = encoded_train[y_train > 0].mean(axis=0)
    neg_mean_encoded = encoded_train[y_train < 0].mean(axis=0)
    signed_gap_encoded = pos_mean_encoded - neg_mean_encoded
    if num_qubits > 1:
        train_pair = encoded_train[:, :-1] * encoded_train[:, 1:]
        pos_mean_pair = train_pair[y_train > 0].mean(axis=0)
        neg_mean_pair = train_pair[y_train < 0].mean(axis=0)
        signed_gap_pair = pos_mean_pair - neg_mean_pair
    else:
        pos_mean_pair = np.array([], dtype=np.float64)
        neg_mean_pair = np.array([], dtype=np.float64)
        signed_gap_pair = np.array([], dtype=np.float64)

    sketch = WeightedStreamingSketch(num_qubits=num_qubits)
    for encoded_sample, label in zip(encoded_train, y_train, strict=True):
        sketch.update(encoded_sample, float(label))

    sketch_batch = extract_pauli_features(
        [sketch.build_circuit()],
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    model_features = sketch_batch.features[0]

    head = quantum_head_scores(
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

    raw_w = ridge_linear_classifier(x_train, y_train)
    c_scores_train = x_train @ raw_w
    c_scores_test = x_test @ raw_w

    if pearson_corr(q_scores_train, y_train) < 0.0:
        q_scores_train *= -1.0
        q_scores_test *= -1.0
        model_features = -model_features

    q_threshold = 0.5 * (
        float(np.mean(q_scores_train[y_train > 0.0])) + float(np.mean(q_scores_train[y_train < 0.0]))
    )
    c_threshold = 0.5 * (
        float(np.mean(c_scores_train[y_train > 0.0])) + float(np.mean(c_scores_train[y_train < 0.0]))
    )

    return {
        "model_features": model_features,
        "train_scores_quantum": q_scores_train,
        "test_scores_quantum": q_scores_test,
        "train_scores_classical": c_scores_train,
        "test_scores_classical": c_scores_test,
        "train_labels": y_train,
        "test_labels": y_test,
        "train_accuracy_quantum": threshold_accuracy(q_scores_train, y_train, q_threshold),
        "test_accuracy_quantum": threshold_accuracy(q_scores_test, y_test, q_threshold),
        "train_accuracy_classical": threshold_accuracy(c_scores_train, y_train, c_threshold),
        "test_accuracy_classical": threshold_accuracy(c_scores_test, y_test, c_threshold),
        "quantum_threshold": q_threshold,
        "classical_threshold": c_threshold,
        "signal_overlap_with_baseline": pearson_corr(q_scores_test, c_scores_test),
        "encoder_scale": encoder.scale,
        "encoder_method": encoder.method,
        "encoder_effective_components": encoder.effective_components,
        "encoder_explained_variance_ratio_sum": encoder.explained_variance_ratio_sum,
        "readout_feature_count": int(len(model_features)),
        "query_feature_count": head["query_feature_count"],
        "quantum_head_method": quantum_head_method,
        "readout_family": readout_family,
        "quantum_head_feature_count": head["head_feature_count"],
        "execution_metadata": {
            "sketch": sketch_batch.metadata,
            "query": head["execution_metadata"],
        },
        "pos_mean_encoded": pos_mean_encoded,
        "neg_mean_encoded": neg_mean_encoded,
        "signed_gap_encoded": signed_gap_encoded,
        "pos_mean_pair": pos_mean_pair,
        "neg_mean_pair": neg_mean_pair,
        "signed_gap_pair": signed_gap_pair,
        "cumulative_signed_mean": cumulative_signed_mean,
    }


def run_classification_from_arrays(
    *,
    x: np.ndarray,
    y: np.ndarray,
    num_qubits: int,
    readout_shots: int | None,
    seed: int,
    n_train: int | None = None,
    train_fraction: float = 0.67,
    encoder_method: str = "block",
    quantum_head_method: str = "cosine",
    readout_family: str = "local",
    execution_config: QuantumExecutionConfig | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    execution_config = execution_config or QuantumExecutionConfig()
    total = len(x)
    if total < 4:
        raise ValueError("need at least 4 samples")

    if n_train is None:
        n_train = int(round(train_fraction * total))
    n_train = max(2, min(int(n_train), total - 1))
    train_idx, test_idx = split_train_test(x, n_train=n_train, rng=rng)

    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train_raw, y_train = subsample_rows(x_train_raw, y_train, max_rows=max_train_samples, rng=rng)
    x_test_raw, y_test = subsample_rows(x_test_raw, y_test, max_rows=max_test_samples, rng=rng)
    x_train, x_test = standardize(x_train_raw, x_test_raw)

    encoder = ToyEncoding.fit(
        x_train,
        num_qubits=num_qubits,
        rng=rng,
        method=encoder_method,
        y_train=y_train,
    )
    encoded_train = encoder.encode(x_train)
    encoded_test = encoder.encode(x_test)
    signed_train = y_train[:, None] * encoded_train
    cumulative_signed_mean = np.cumsum(signed_train, axis=0) / np.arange(1, len(signed_train) + 1)[:, None]
    pos_mean_encoded = encoded_train[y_train > 0].mean(axis=0)
    neg_mean_encoded = encoded_train[y_train < 0].mean(axis=0)
    signed_gap_encoded = pos_mean_encoded - neg_mean_encoded
    if num_qubits > 1:
        train_pair = encoded_train[:, :-1] * encoded_train[:, 1:]
        pos_mean_pair = train_pair[y_train > 0].mean(axis=0)
        neg_mean_pair = train_pair[y_train < 0].mean(axis=0)
        signed_gap_pair = pos_mean_pair - neg_mean_pair
    else:
        pos_mean_pair = np.array([], dtype=np.float64)
        neg_mean_pair = np.array([], dtype=np.float64)
        signed_gap_pair = np.array([], dtype=np.float64)

    sketch = WeightedStreamingSketch(num_qubits=num_qubits)
    for encoded_sample, label in zip(encoded_train, y_train, strict=True):
        sketch.update(encoded_sample, float(label))

    sketch_batch = extract_pauli_features(
        [sketch.build_circuit()],
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    model_features = sketch_batch.features[0]

    head = quantum_head_scores(
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

    raw_w = ridge_linear_classifier(x_train, y_train)
    c_scores_train = x_train @ raw_w
    c_scores_test = x_test @ raw_w

    if pearson_corr(q_scores_train, y_train) < 0.0:
        q_scores_train *= -1.0
        q_scores_test *= -1.0
        model_features = -model_features

    q_threshold = 0.5 * (
        float(np.mean(q_scores_train[y_train > 0.0])) + float(np.mean(q_scores_train[y_train < 0.0]))
    )
    c_threshold = 0.5 * (
        float(np.mean(c_scores_train[y_train > 0.0])) + float(np.mean(c_scores_train[y_train < 0.0]))
    )

    return {
        "model_features": model_features,
        "train_scores_quantum": q_scores_train,
        "test_scores_quantum": q_scores_test,
        "train_scores_classical": c_scores_train,
        "test_scores_classical": c_scores_test,
        "train_labels": y_train,
        "test_labels": y_test,
        "train_accuracy_quantum": threshold_accuracy(q_scores_train, y_train, q_threshold),
        "test_accuracy_quantum": threshold_accuracy(q_scores_test, y_test, q_threshold),
        "train_accuracy_classical": threshold_accuracy(c_scores_train, y_train, c_threshold),
        "test_accuracy_classical": threshold_accuracy(c_scores_test, y_test, c_threshold),
        "quantum_threshold": q_threshold,
        "classical_threshold": c_threshold,
        "signal_overlap_with_baseline": pearson_corr(q_scores_test, c_scores_test),
        "encoder_scale": encoder.scale,
        "encoder_method": encoder.method,
        "encoder_effective_components": encoder.effective_components,
        "encoder_explained_variance_ratio_sum": encoder.explained_variance_ratio_sum,
        "readout_feature_count": int(len(model_features)),
        "query_feature_count": head["query_feature_count"],
        "quantum_head_method": quantum_head_method,
        "readout_family": readout_family,
        "quantum_head_feature_count": head["head_feature_count"],
        "execution_metadata": {
            "sketch": sketch_batch.metadata,
            "query": head["execution_metadata"],
        },
        "pos_mean_encoded": pos_mean_encoded,
        "neg_mean_encoded": neg_mean_encoded,
        "signed_gap_encoded": signed_gap_encoded,
        "pos_mean_pair": pos_mean_pair,
        "neg_mean_pair": neg_mean_pair,
        "signed_gap_pair": signed_gap_pair,
        "cumulative_signed_mean": cumulative_signed_mean,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def run_reduction_task(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    num_qubits: int,
    readout_shots: int | None,
    seed: int,
    encoder_method: str = "block",
    readout_family: str = "local",
    execution_config: QuantumExecutionConfig | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    execution_config = execution_config or QuantumExecutionConfig()
    data = make_reduction_data(n_train + n_test, n_features, num_qubits, rng)
    train_idx, test_idx = split_train_test(data.x, n_train=n_train, rng=rng)

    x_train_raw, x_test_raw = data.x[train_idx], data.x[test_idx]
    latent_train, latent_test = data.latent_score[train_idx], data.latent_score[test_idx]
    x_train_raw, latent_train = subsample_rows(x_train_raw, latent_train, max_rows=max_train_samples, rng=rng)
    x_test_raw, latent_test = subsample_rows(x_test_raw, latent_test, max_rows=max_test_samples, rng=rng)
    x_train, x_test = standardize(x_train_raw, x_test_raw)

    guide = normalize(data.guide)
    guide = normalize(guide / (x_train.std(axis=0) + EPS))
    guide_proj_train = x_train @ guide
    guide_scale = float(np.quantile(np.abs(guide_proj_train), 0.9))
    if guide_scale < EPS:
        guide_scale = 1.0

    if encoder_method in {"ridge", "lda"}:
        encoder_method = "block"
    encoder = ToyEncoding.fit(x_train, num_qubits=num_qubits, rng=rng, method=encoder_method)
    encoded_train = encoder.encode(x_train)
    encoded_test = encoder.encode(x_test)

    sketch = WeightedStreamingSketch(num_qubits=num_qubits)
    for x_row, encoded_sample in zip(x_train, encoded_train, strict=True):
        weight = float(np.tanh(np.dot(x_row, guide) / guide_scale))
        sketch.update(encoded_sample, weight)

    sketch_batch = extract_pauli_features(
        [sketch.build_circuit()],
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    model_features = sketch_batch.features[0]

    query_circuits = [
        query_circuit(
            encoded_sample,
            single_scale=sketch.single_scale,
            phase_scale=sketch.phase_scale,
            pair_scale=sketch.pair_scale,
        )
        for encoded_sample in np.concatenate([encoded_train, encoded_test], axis=0)
    ]
    query_batch = extract_pauli_features(
        query_circuits,
        shots=readout_shots,
        rng=rng,
        execution_config=execution_config,
        readout_family=readout_family,
    )
    query_train = np.asarray(query_batch.features[: len(encoded_train)], dtype=np.float64)
    query_test = np.asarray(query_batch.features[len(encoded_train) :], dtype=np.float64)
    q_scores_train = np.asarray([cosine_similarity(model_features, row) for row in query_train], dtype=np.float64)
    q_scores_test = np.asarray([cosine_similarity(model_features, row) for row in query_test], dtype=np.float64)

    pca_vec = top_principal_component(x_train)
    if np.dot(pca_vec, guide) < 0.0:
        pca_vec = -pca_vec

    c_scores_train = x_train @ pca_vec
    c_scores_test = x_test @ pca_vec

    if pearson_corr(q_scores_train, c_scores_train) < 0.0:
        q_scores_train *= -1.0
        q_scores_test *= -1.0
        model_features = -model_features

    return {
        "model_features": model_features,
        "train_scores_quantum": q_scores_train,
        "test_scores_quantum": q_scores_test,
        "train_scores_classical": c_scores_train,
        "test_scores_classical": c_scores_test,
        "train_corr_quantum_vs_pca": pearson_corr(q_scores_train, c_scores_train),
        "test_corr_quantum_vs_pca": pearson_corr(q_scores_test, c_scores_test),
        "test_corr_quantum_vs_latent": pearson_corr(q_scores_test, latent_test),
        "guide_overlap_with_signal": float(np.dot(guide, data.signal_direction)),
        "encoder_scale": encoder.scale,
        "encoder_method": encoder.method,
        "encoder_effective_components": encoder.effective_components,
        "encoder_explained_variance_ratio_sum": encoder.explained_variance_ratio_sum,
        "readout_feature_count": int(len(model_features)),
        "readout_family": readout_family,
        "execution_metadata": {
            "sketch": sketch_batch.metadata,
            "query": query_batch.metadata,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qiskit toy model for quantum oracle sketching ideas")
    parser.add_argument("--csv", help="CSV path for real binary classification data")
    parser.add_argument("--label-col", help="Label column name for --csv mode")
    parser.add_argument(
        "--feature-cols",
        help="Comma-separated feature columns for --csv mode; default is all numeric columns except the label",
    )
    parser.add_argument(
        "--csv-train-fraction",
        type=float,
        default=0.67,
        help="Train fraction for --csv mode when no explicit sample count is provided",
    )
    parser.add_argument("--num-qubits", type=int, default=4, help="Quantum sketch size")
    parser.add_argument("--n-features", type=int, default=32, help="Raw classical feature dimension")
    parser.add_argument("--n-train", type=int, default=192)
    parser.add_argument("--n-test", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--readout-shots",
        type=int,
        default=512,
        help="Per-observable shot budget for the local shadow surrogate; use 0 for exact expectations",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["statevector", "sampler-sim", "ibm-hardware"],
        default="statevector",
        help="How quantum observables are evaluated: exact statevector, local shot-based sim, or IBM hardware",
    )
    parser.add_argument("--backend-name", help="Optional IBM backend name for --execution-mode ibm-hardware")
    parser.add_argument(
        "--simulator-method",
        default="automatic",
        choices=["automatic", "statevector", "matrix_product_state"],
        help="Backend method for --execution-mode sampler-sim; use matrix_product_state for larger 1D circuit ladders",
    )
    parser.add_argument("--readout-mitigation", action="store_true", help="Enable local-tensored readout mitigation")
    parser.add_argument("--cal-shots", type=int, default=4096, help="Calibration shots for readout mitigation")
    parser.add_argument(
        "--extra-error-suppression",
        action="store_true",
        help="Enable IBM Runtime dynamical decoupling and gate twirling when supported",
    )
    parser.add_argument(
        "--dd-sequence",
        type=str,
        default="XY4",
        choices=["XX", "XpXm", "XY4"],
        help="Dynamical decoupling sequence for --extra-error-suppression",
    )
    parser.add_argument(
        "--twirl-randomizations",
        type=int,
        default=8,
        help="Gate-twirling randomizations for --extra-error-suppression",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional cap on the number of train samples evaluated after the split; useful for hardware smoke runs",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        help="Optional cap on the number of test samples evaluated after the split; useful for hardware smoke runs",
    )
    parser.add_argument(
        "--encoder",
        choices=["block", "pca", "spca", "ridge", "lda"],
        default="block",
        help="Feature compression used before the quantum sketch",
    )
    parser.add_argument(
        "--quantum-head",
        choices=["cosine", "ridge", "logistic"],
        default="cosine",
        help="Small classical head applied to per-sample quantum readout features",
    )
    parser.add_argument(
        "--readout-family",
        choices=["local", "all-pairs"],
        default="local",
        help="Observable family used for quantum readout features",
    )
    parser.add_argument("--plot", action="store_true", help="Write a summary plot")
    parser.add_argument("--plot-out", default="qiskit_qos_toy_model.png")
    parser.add_argument("--label-plot-out", default="qiskit_qos_label_weight_diagnostics.png")
    parser.add_argument("--json-out", default="qiskit_qos_toy_model_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    readout_shots = args.readout_shots if args.readout_shots > 0 else None
    csv_mode = args.csv is not None
    if args.cal_shots < 1:
        raise ValueError("--cal-shots must be >= 1")
    if args.twirl_randomizations < 1:
        raise ValueError("--twirl-randomizations must be >= 1")
    execution_config = QuantumExecutionConfig(
        mode=args.execution_mode,
        backend_name=args.backend_name,
        simulator_method=args.simulator_method,
        readout_mitigation=args.readout_mitigation,
        cal_shots=args.cal_shots,
        extra_error_suppression=args.extra_error_suppression,
        dd_sequence=args.dd_sequence,
        twirl_randomizations=args.twirl_randomizations,
    )

    if csv_mode:
        if not args.label_col:
            raise ValueError("--label-col is required when using --csv")
        csv_data = load_csv_classification_data(
            args.csv,
            label_col=args.label_col,
            feature_cols=parse_feature_columns_arg(args.feature_cols),
        )
        classification = run_classification_from_arrays(
            x=csv_data.x,
            y=csv_data.y,
            num_qubits=args.num_qubits,
            readout_shots=readout_shots,
            seed=args.seed,
            n_train=None,
            train_fraction=args.csv_train_fraction,
            encoder_method=args.encoder,
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        reduction = None
    else:
        csv_data = None
        classification = run_classification_task(
            n_train=args.n_train,
            n_test=args.n_test,
            n_features=args.n_features,
            num_qubits=args.num_qubits,
            readout_shots=readout_shots,
            seed=args.seed,
            encoder_method=args.encoder,
            quantum_head_method=args.quantum_head,
            readout_family=args.readout_family,
            execution_config=execution_config,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        reduction = run_reduction_task(
            n_train=args.n_train,
            n_test=args.n_test,
            n_features=args.n_features,
            num_qubits=args.num_qubits,
            readout_shots=readout_shots,
            seed=args.seed + 101,
            encoder_method=args.encoder,
            readout_family=args.readout_family,
            execution_config=execution_config,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )

    summary = {
        "config": {
            "num_qubits": args.num_qubits,
            "n_features": csv_data.x.shape[1] if csv_mode else args.n_features,
            "n_train": classification["n_train"] if csv_mode else args.n_train,
            "n_test": classification["n_test"] if csv_mode else args.n_test,
            "seed": args.seed,
            "readout_shots": 0 if readout_shots is None else readout_shots,
            "execution_mode": args.execution_mode,
            "backend_name": args.backend_name,
            "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
            "readout_mitigation": args.readout_mitigation,
            "cal_shots": args.cal_shots,
            "extra_error_suppression": args.extra_error_suppression,
            "dd_sequence": args.dd_sequence if args.extra_error_suppression else None,
            "twirl_randomizations": args.twirl_randomizations if args.extra_error_suppression else None,
            "encoder": args.encoder,
            "quantum_head": args.quantum_head,
            "readout_family": args.readout_family,
            "shadow_feature_count": classification["readout_feature_count"],
            "query_feature_count": classification["query_feature_count"],
            "quantum_head_feature_count": classification["quantum_head_feature_count"],
            "csv": args.csv if csv_mode else None,
        },
        "classification": {
            "train_accuracy_quantum": classification["train_accuracy_quantum"],
            "test_accuracy_quantum": classification["test_accuracy_quantum"],
            "train_accuracy_classical": classification["train_accuracy_classical"],
            "test_accuracy_classical": classification["test_accuracy_classical"],
            "quantum_threshold": classification["quantum_threshold"],
            "classical_threshold": classification["classical_threshold"],
            "test_score_corr_quantum_vs_classical": classification["signal_overlap_with_baseline"],
            "pos_mean_encoded": classification["pos_mean_encoded"].tolist(),
            "neg_mean_encoded": classification["neg_mean_encoded"].tolist(),
            "signed_gap_encoded": classification["signed_gap_encoded"].tolist(),
            "pos_mean_pair": classification["pos_mean_pair"].tolist(),
            "neg_mean_pair": classification["neg_mean_pair"].tolist(),
            "signed_gap_pair": classification["signed_gap_pair"].tolist(),
            "cumulative_signed_mean_tail": classification["cumulative_signed_mean"][-8:].tolist(),
            "encoder_method": classification["encoder_method"],
            "encoder_effective_components": classification["encoder_effective_components"],
            "encoder_explained_variance_ratio_sum": classification["encoder_explained_variance_ratio_sum"],
            "quantum_head_method": classification["quantum_head_method"],
            "readout_family": classification["readout_family"],
            "quantum_head_feature_count": classification["quantum_head_feature_count"],
            "execution_metadata": classification["execution_metadata"],
        },
        "dimension_reduction": None
        if csv_mode
        else {
            "train_corr_quantum_vs_pca": reduction["train_corr_quantum_vs_pca"],
            "test_corr_quantum_vs_pca": reduction["test_corr_quantum_vs_pca"],
            "test_corr_quantum_vs_latent": reduction["test_corr_quantum_vs_latent"],
            "guide_overlap_with_signal": reduction["guide_overlap_with_signal"],
            "execution_metadata": reduction["execution_metadata"],
        },
        "notes": [
            "Streaming sketch keeps O(num_qubits) statistics and processes each sample once.",
            "Readout uses a compact local-Pauli shadow surrogate, not the paper's full interferometric classical shadow.",
            "This is a pedagogical Qiskit analogue, not a reproduction of the JAX codebase or the lower-bound proofs.",
        ],
    }
    if csv_mode and csv_data is not None:
        summary["classification"]["label_name"] = csv_data.label_name
        summary["classification"]["feature_names"] = csv_data.feature_names
        summary["classification"]["label_mapping"] = csv_data.label_mapping

    print("Qiskit QOS-inspired toy model")
    if csv_mode and csv_data is not None:
        print(f"- data source: CSV {args.csv}")
        print(f"- label column: {csv_data.label_name}")
        print(f"- rows used: {csv_data.x.shape[0]}")
        print(f"- numeric feature columns: {len(csv_data.feature_names)}")
    print(f"- raw feature dimension: {csv_data.x.shape[1] if csv_mode and csv_data is not None else args.n_features}")
    print(f"- encoder: {args.encoder}")
    print(f"- quantum sketch size: {args.num_qubits} qubits")
    print(f"- compact readout size: {classification['readout_feature_count']} features")
    print(f"- quantum head: {classification['quantum_head_method']} ({classification['quantum_head_feature_count']} features)")
    print(f"- readout family: {classification['readout_family']}")
    print(f"- execution mode: {args.execution_mode}")
    if args.backend_name:
        print(f"- requested backend: {args.backend_name}")
    if args.execution_mode == "sampler-sim":
        print(f"- simulator method: {args.simulator_method}")
    if args.readout_mitigation:
        print(f"- readout mitigation: on (cal shots={args.cal_shots})")
    if args.extra_error_suppression:
        print(
            f"- extra suppression: on (DD={args.dd_sequence}, twirl_randomizations={args.twirl_randomizations})"
        )
    print(f"- readout shots: {summary['config']['readout_shots']}")
    print()
    print("Classification")
    print(f"  quantum toy  train acc: {classification['train_accuracy_quantum']:.3f}")
    print(f"  quantum toy   test acc: {classification['test_accuracy_quantum']:.3f}")
    print(f"  classical    train acc: {classification['train_accuracy_classical']:.3f}")
    print(f"  classical     test acc: {classification['test_accuracy_classical']:.3f}")
    print(
        "  score corr(q-toy, classical): "
        f"{classification['signal_overlap_with_baseline']:.3f}"
    )
    print(
        "  signed per-qubit gap: "
        f"{[round(v, 3) for v in classification['signed_gap_encoded'].tolist()]}"
    )
    if csv_mode and csv_data is not None:
        print(f"  label mapping: {csv_data.label_mapping}")
    else:
        print()
        print("Dimension Reduction")
        print(f"  corr(q-toy, PCA)   train: {reduction['train_corr_quantum_vs_pca']:.3f}")
        print(f"  corr(q-toy, PCA)    test: {reduction['test_corr_quantum_vs_pca']:.3f}")
        print(f"  corr(q-toy, latent) test: {reduction['test_corr_quantum_vs_latent']:.3f}")
        print(f"  guide overlap with signal: {reduction['guide_overlap_with_signal']:.3f}")

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {json_out}")

    if args.plot:
        plot_summary(
            Path(args.plot_out),
            cls_scores_q=classification["test_scores_quantum"],
            cls_scores_c=classification["test_scores_classical"],
            cls_labels=classification["test_labels"],
            pca_scores_q=None if reduction is None else reduction["test_scores_quantum"],
            pca_scores_c=None if reduction is None else reduction["test_scores_classical"],
            raw_feature_count=summary["config"]["n_features"],
            num_qubits=args.num_qubits,
            readout_feature_count=classification["readout_feature_count"],
        )
        print(f"Saved plot to: {args.plot_out}")
        plot_label_weight_diagnostics(
            Path(args.label_plot_out),
            pos_mean_encoded=classification["pos_mean_encoded"],
            neg_mean_encoded=classification["neg_mean_encoded"],
            signed_gap_encoded=classification["signed_gap_encoded"],
            pos_mean_pair=classification["pos_mean_pair"],
            neg_mean_pair=classification["neg_mean_pair"],
            signed_gap_pair=classification["signed_gap_pair"],
            cumulative_signed_mean=classification["cumulative_signed_mean"],
            quantum_scores=classification["test_scores_quantum"],
            labels=classification["test_labels"],
        )
        print(f"Saved label-weight diagnostics to: {args.label_plot_out}")


if __name__ == "__main__":
    main()
