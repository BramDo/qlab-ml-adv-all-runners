#!/usr/bin/env python3
"""Utilities for the subtle Perturb-seq source GSE132080."""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


GSE132080_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132080/suppl"
GSE132080_MATRIX_URL = f"{GSE132080_BASE_URL}/GSE132080_10X_matrix.mtx.gz"
GSE132080_BARCODES_URL = f"{GSE132080_BASE_URL}/GSE132080_10X_barcodes.tsv.gz"
GSE132080_GENES_URL = f"{GSE132080_BASE_URL}/GSE132080_10X_genes.tsv.gz"
GSE132080_CELL_IDENTITIES_URL = f"{GSE132080_BASE_URL}/GSE132080_cell_identities.csv.gz"
GSE132080_SGRNA_PHENOTYPES_URL = f"{GSE132080_BASE_URL}/GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz"


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
        path.write_bytes(response.read())
    return path


def normalize_guide_identity(value: str) -> str:
    parts = str(value).split("_")
    if len(parts) >= 2 and parts[0] == parts[1]:
        return "_".join(parts[1:])
    return str(value)


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_genes_tsv_gz(path: Path) -> np.ndarray:
    names: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                names.append(parts[1])
            elif parts:
                names.append(parts[0])
    return np.array(names, dtype=object)


def _read_matrix_mtx_gz(path: Path) -> sparse.csr_matrix:
    with gzip.open(path, "rb") as handle:
        matrix = mmread(handle)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    return matrix.tocsc().transpose().tocsr()


def load_gse132080(
    *,
    cache_dir: str = "data_cache/gse132080",
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    cache_path = Path(cache_dir)
    matrix_path = ensure_download(GSE132080_MATRIX_URL, cache_path / "GSE132080_10X_matrix.mtx.gz")
    barcodes_path = ensure_download(GSE132080_BARCODES_URL, cache_path / "GSE132080_10X_barcodes.tsv.gz")
    genes_path = ensure_download(GSE132080_GENES_URL, cache_path / "GSE132080_10X_genes.tsv.gz")
    identities_path = ensure_download(GSE132080_CELL_IDENTITIES_URL, cache_path / "GSE132080_cell_identities.csv.gz")
    phenotypes_path = ensure_download(
        GSE132080_SGRNA_PHENOTYPES_URL,
        cache_path / "GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz",
    )

    x = _read_matrix_mtx_gz(matrix_path)
    barcodes = np.array(_read_lines_gz(barcodes_path), dtype=object)
    genes = _read_genes_tsv_gz(genes_path)
    if x.shape[0] != len(barcodes):
        raise RuntimeError("Barcode count does not match matrix rows")
    if x.shape[1] != len(genes):
        raise RuntimeError("Gene count does not match matrix columns")

    identities = pd.read_csv(identities_path)
    identities["cell_barcode"] = identities["cell_barcode"].astype(str)
    identities["guide_norm"] = identities["guide_identity"].map(normalize_guide_identity)
    phenotypes = pd.read_csv(phenotypes_path)
    phenotypes["sgRNA_name"] = phenotypes["sgRNA_name"].astype(str)

    merged = identities.merge(phenotypes, left_on="guide_norm", right_on="sgRNA_name", how="left")
    merged = merged.set_index("cell_barcode")

    positions = pd.Index(barcodes, name="cell_barcode").get_indexer(merged.index)
    if np.any(positions < 0):
        missing = merged.index[positions < 0][:10].tolist()
        raise RuntimeError(f"Could not align all GSE132080 barcodes; examples: {missing}")

    x_aligned = x[positions].tocsr()
    merged = merged.copy()
    merged["cell_barcode"] = merged.index

    meta = {
        "dataset_name": "GSE132080",
        "rows_total": int(x_aligned.shape[0]),
        "genes": int(x_aligned.shape[1]),
        "nnz": int(x_aligned.nnz),
        "density": float(x_aligned.nnz / (x_aligned.shape[0] * x_aligned.shape[1])),
        "good_coverage_cells": int(merged["good_coverage"].sum()),
        "matched_guides": int(merged["gene"].notna().sum()),
        "guide_count": int(merged["guide_norm"].nunique()),
        "target_gene_count": int(merged["gene"].dropna().nunique()),
        "guide_preview": merged["guide_norm"].dropna().astype(str).value_counts().head(10).to_dict(),
        "matrix_url": GSE132080_MATRIX_URL,
        "cell_identities_url": GSE132080_CELL_IDENTITIES_URL,
        "phenotypes_url": GSE132080_SGRNA_PHENOTYPES_URL,
        "cache_dir": str(cache_path),
        "gene_preview": [str(item) for item in genes[:10]],
    }
    return x_aligned, merged, meta


def select_guide_pair(
    x: sparse.csr_matrix,
    metadata: pd.DataFrame,
    *,
    positive_guide: str,
    negative_guide: str,
    require_good_coverage: bool = True,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    keep = metadata["guide_norm"].isin([positive_guide, negative_guide]).to_numpy()
    if require_good_coverage:
        keep &= metadata["good_coverage"].to_numpy(dtype=bool)
    if int(np.sum(keep)) == 0:
        raise ValueError("No rows matched the requested GSE132080 guide pair")

    metadata_pair = metadata.loc[keep].copy()
    x_pair = x[keep]
    y_pair = np.where(metadata_pair["guide_norm"].to_numpy() == positive_guide, 1, -1).astype(np.int64)

    pair_gene = metadata_pair["gene"].dropna().astype(str).unique().tolist()
    meta = {
        "positive_guide": positive_guide,
        "negative_guide": negative_guide,
        "gene_labels": pair_gene,
        "rows": int(x_pair.shape[0]),
        "positive_count": int(np.sum(y_pair > 0)),
        "negative_count": int(np.sum(y_pair < 0)),
        "genes": int(x_pair.shape[1]),
        "nnz": int(x_pair.nnz),
        "density": float(x_pair.nnz / (x_pair.shape[0] * x_pair.shape[1])),
        "positive_activity_day10": float(
            metadata_pair.loc[metadata_pair["guide_norm"] == positive_guide, "relative_activity_day10"].dropna().iloc[0]
        ),
        "negative_activity_day10": float(
            metadata_pair.loc[metadata_pair["guide_norm"] == negative_guide, "relative_activity_day10"].dropna().iloc[0]
        ),
    }
    return x_pair.tocsr(), y_pair, meta
