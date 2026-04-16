#!/usr/bin/env python3
"""Utilities for loading the official 10x PBMC68k counts plus annotations."""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


PBMC68K_MATRIX_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/"
    "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
)
PBMC68K_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/"
    "pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv"
)


def ensure_download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Codex/1.0; +https://openai.com)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()
    path.write_bytes(data)
    return path


def _read_member_text(archive_path: Path, suffix: str) -> list[str]:
    with tarfile.open(archive_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"Could not find {suffix} inside {archive_path}")
        with tar.extractfile(member) as fh:
            if fh is None:
                raise FileNotFoundError(f"Could not extract {suffix} from {archive_path}")
            return fh.read().decode("utf-8").strip().splitlines()


def _read_member_matrix(archive_path: Path, suffix: str) -> sparse.csr_matrix:
    with tarfile.open(archive_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"Could not find {suffix} inside {archive_path}")
        with tar.extractfile(member) as fh:
            if fh is None:
                raise FileNotFoundError(f"Could not extract {suffix} from {archive_path}")
            raw = fh.read()
    matrix = mmread(io.BytesIO(raw))
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    return matrix.tocsr()


def _parse_features(lines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append({"gene_id": parts[0], "gene_name": parts[1]})
        elif len(parts) == 1:
            rows.append({"gene_id": parts[0], "gene_name": parts[0]})
    return pd.DataFrame(rows)


def load_pbmc68k(
    *,
    cache_dir: str = "data_cache/pbmc68k",
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_dir)
    matrix_tar = ensure_download(PBMC68K_MATRIX_URL, cache_path / "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz")
    annotation_path = ensure_download(PBMC68K_ANNOTATION_URL, cache_path / "68k_pbmc_barcodes_annotation.tsv")

    barcodes = _read_member_text(matrix_tar, "barcodes.tsv")
    feature_lines = _read_member_text(matrix_tar, "genes.tsv")
    features = _parse_features(feature_lines)
    matrix_gene_by_cell = _read_member_matrix(matrix_tar, "matrix.mtx")
    matrix_cell_by_gene = matrix_gene_by_cell.transpose().tocsr()

    if matrix_cell_by_gene.shape[0] != len(barcodes):
        raise RuntimeError("Barcode count does not match matrix rows")
    if matrix_cell_by_gene.shape[1] != len(features):
        raise RuntimeError("Gene count does not match matrix columns")

    annotations = pd.read_csv(annotation_path, sep="\t")
    annotations["barcode_norm"] = annotations["barcodes"].astype(str).str.replace(r"-1$", "", regex=True)
    barcode_norm = pd.Index([barcode.replace("-1", "") if barcode.endswith("-1") else barcode for barcode in barcodes], name="barcode_norm")
    ann_indexed = annotations.drop_duplicates("barcode_norm").set_index("barcode_norm")
    matched = ann_indexed.reindex(barcode_norm)
    valid_mask = matched["celltype"].notna().to_numpy()

    x = matrix_cell_by_gene[valid_mask]
    labels = matched.loc[valid_mask, "celltype"].astype(str).to_numpy()

    meta = {
        "dataset_name": "pbmc68k",
        "rows_total": int(matrix_cell_by_gene.shape[0]),
        "rows_annotated": int(x.shape[0]),
        "genes": int(x.shape[1]),
        "nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])),
        "celltype_counts": matched.loc[valid_mask, "celltype"].value_counts().to_dict(),
        "matrix_url": PBMC68K_MATRIX_URL,
        "annotation_url": PBMC68K_ANNOTATION_URL,
        "cache_dir": str(cache_path),
        "gene_preview": features["gene_name"].head(10).tolist(),
    }
    return x.tocsr(), labels, meta


def select_binary_pair(
    x: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    positive_label: str,
    negative_label: str,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    keep = np.isin(labels, [positive_label, negative_label])
    if int(np.sum(keep)) == 0:
        raise ValueError("No rows matched the requested PBMC labels")
    x_pair = x[keep]
    y_pair = np.where(labels[keep] == positive_label, 1, -1).astype(np.int64)
    meta = {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "rows": int(x_pair.shape[0]),
        "positive_count": int(np.sum(y_pair > 0)),
        "negative_count": int(np.sum(y_pair < 0)),
        "genes": int(x_pair.shape[1]),
        "nnz": int(x_pair.nnz),
        "density": float(x_pair.nnz / (x_pair.shape[0] * x_pair.shape[1])),
    }
    return x_pair.tocsr(), y_pair, meta
