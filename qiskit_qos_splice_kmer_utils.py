#!/usr/bin/env python3
"""Splice junction k-mer dataset utilities.

This mirrors the paper-repo idea of using binary Splice EI vs IE classification
with k-mer frequency features, but uses OpenML because it is available in the
current qiskit venv.
"""

from __future__ import annotations

from collections import Counter
import gzip
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import fetch_openml


NUCLEOTIDES = "ACGT"
NUCLEOTIDE_TO_BITS = {char: idx for idx, char in enumerate(NUCLEOTIDES)}


def ambient_kmer_dim(k: int) -> int:
    return 4 ** int(k)


def ambient_dense_weight_bytes(k: int, *, float_bytes: int = 8) -> int:
    return ambient_kmer_dim(k) * int(float_bytes)


def clean_sequence(sequence: str) -> str:
    return "".join(char for char in str(sequence).upper() if char in NUCLEOTIDES)


def iter_kmers(sequence: str, *, k: int) -> Iterator[str]:
    cleaned = clean_sequence(sequence)
    for start in range(max(0, len(cleaned) - k + 1)):
        kmer = cleaned[start : start + k]
        if len(kmer) == k:
            yield kmer


def iter_kmer_keys(sequence: str, *, k: int) -> Iterator[int]:
    cleaned = clean_sequence(sequence)
    if k <= 0 or len(cleaned) < k:
        return
    mask = (1 << (2 * k)) - 1
    key = 0
    for char in cleaned[:k]:
        key = (key << 2) | NUCLEOTIDE_TO_BITS[char]
    yield key
    for char in cleaned[k:]:
        key = ((key << 2) & mask) | NUCLEOTIDE_TO_BITS[char]
        yield key


def _open_text_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def detect_sequence_file_format(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
        return "fasta"
    if lower_name.endswith((".fq", ".fastq", ".fq.gz", ".fastq.gz")):
        return "fastq"

    with _open_text_maybe_gzip(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                return "fasta"
            if line.startswith("@"):
                return "fastq"
            break
    raise ValueError(f"Could not infer FASTA/FASTQ format for {path}")


def iter_sequences_from_file(path: Path) -> Iterator[str]:
    fmt = detect_sequence_file_format(path)
    with _open_text_maybe_gzip(path) as handle:
        if fmt == "fasta":
            parts: list[str] = []
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if parts:
                        yield clean_sequence("".join(parts))
                        parts = []
                    continue
                parts.append(line)
            if parts:
                yield clean_sequence("".join(parts))
            return

        while True:
            header = handle.readline()
            if not header:
                break
            if not header.strip():
                continue
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()
            if not sequence or not plus or not quality:
                break
            yield clean_sequence(sequence.strip())


def load_splice_sequences(*, binary: bool = True) -> tuple[list[str], np.ndarray, dict[str, object]]:
    dataset = fetch_openml(name="splice", version=1, as_frame=True)
    x_df = dataset.data
    y = dataset.target.astype(str)
    sequences = [clean_sequence("".join(str(value).upper() for value in row.values)) for _, row in x_df.iterrows()]

    if binary:
        mask = (y == "EI") | (y == "IE")
        sequences = [seq for seq, keep in zip(sequences, mask) if keep]
        y = y[mask]

    labels = np.where(np.asarray(y) == "IE", 1.0, -1.0)
    metadata = {
        "dataset_name": "splice",
        "rows": int(len(sequences)),
        "sequence_length_raw": int(len(sequences[0])) if sequences else 0,
        "binary": bool(binary),
        "positive_label": "IE",
        "negative_label": "EI",
        "positive_count": int(np.sum(labels > 0)),
        "negative_count": int(np.sum(labels < 0)),
    }
    return sequences, labels.astype(np.float64), metadata


def compute_kmer_features(sequences: list[str], *, k: int) -> tuple[sp.csr_matrix, dict[str, int], dict[str, object]]:
    kmer_to_idx: dict[str, int] = {}
    idx = 0

    cleaned_sequences = [clean_sequence(seq) for seq in sequences]

    for seq in cleaned_sequences:
        for start in range(max(0, len(seq) - k + 1)):
            kmer = seq[start : start + k]
            if len(kmer) != k:
                continue
            if kmer not in kmer_to_idx:
                kmer_to_idx[kmer] = idx
                idx += 1

    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[float] = []
    sample_nnz: list[int] = []

    for row, seq in enumerate(cleaned_sequences):
        counts: Counter[str] = Counter()
        for start in range(max(0, len(seq) - k + 1)):
            kmer = seq[start : start + k]
            if len(kmer) == k and kmer in kmer_to_idx:
                counts[kmer] += 1
        total = sum(counts.values())
        sample_nnz.append(len(counts))
        if total == 0:
            continue
        for kmer, count in counts.items():
            row_ind.append(row)
            col_ind.append(kmer_to_idx[kmer])
            data_val.append(count / total)

    x = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(len(sequences), len(kmer_to_idx)), dtype=np.float64)
    metadata = {
        "kmer_k": int(k),
        "observed_feature_dim": int(x.shape[1]),
        "ambient_feature_dim": int(ambient_kmer_dim(k)),
        "ambient_dense_weight_bytes": int(ambient_dense_weight_bytes(k)),
        "nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])) if x.shape[0] and x.shape[1] else 0.0,
        "avg_nnz_per_sample": float(np.mean(sample_nnz)) if sample_nnz else 0.0,
        "max_nnz_per_sample": int(max(sample_nnz)) if sample_nnz else 0,
    }
    return x, kmer_to_idx, metadata


def filter_features_by_min_samples(x: sp.csr_matrix, *, min_samples: int) -> tuple[sp.csr_matrix, np.ndarray]:
    if min_samples <= 1:
        keep = np.arange(x.shape[1], dtype=np.int64)
        return x, keep
    feature_counts = np.asarray((x != 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(feature_counts >= int(min_samples))
    return x[:, keep], keep
