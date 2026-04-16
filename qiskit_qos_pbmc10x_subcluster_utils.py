#!/usr/bin/env python3
"""Utilities for the finer-label PBMC 10x multiome RNA source.

This source is separate from the coarse PBMC68k labels. It uses the
SingleCellMultiModal `pbmc_10x` resources:

- v1.0.1 RNA counts in 10x HDF5 form
- v1.0.0 `pbmc_colData.rda` sidecar with finer `celltype` labels
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import rdata
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "qiskit_qos_pbmc10x_subcluster_utils requires the `rdata` package. "
        "Install it in the qiskit venv, for example: "
        "`../quantum-math-lab/scripts/run-in-qiskit-venv.sh python -m pip install rdata`."
    ) from exc


PBMC10X_RNA_H5_URL = (
    "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/"
    "SingleCellMultiModal/pbmc_10x/v1.0.1/pbmc_rna_tenx.h5"
)
PBMC10X_COLDATA_RDA_URL = (
    "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/"
    "SingleCellMultiModal/pbmc_10x/v1.0.0/pbmc_colData.rda"
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
        path.write_bytes(response.read())
    return path


def _decode_bytes_array(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values
        ],
        dtype=object,
    )


def _resolve_tenx_group(handle: h5py.File) -> h5py.Group:
    if "matrix" in handle:
        return handle["matrix"]
    top_level_groups = [obj for obj in handle.values() if isinstance(obj, h5py.Group)]
    if len(top_level_groups) == 1:
        return top_level_groups[0]
    raise RuntimeError("Could not resolve a 10x-style count group inside the HDF5 file")


def _load_tenx_counts(h5_path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        group = _resolve_tenx_group(handle)
        shape = tuple(int(value) for value in group["shape"][:])
        matrix_gene_by_cell = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )
        matrix_cell_by_gene = matrix_gene_by_cell.transpose().tocsr()

        barcodes = _decode_bytes_array(group["barcodes"][:])
        barcodes = np.array(
            [barcode[:-2] if barcode.endswith("-1") else barcode for barcode in barcodes],
            dtype=object,
        )

        if "features" in group and "name" in group["features"]:
            genes = _decode_bytes_array(group["features"]["name"][:])
        elif "genes" in group:
            genes = _decode_bytes_array(group["genes"][:])
        else:
            raise RuntimeError("Could not find gene names inside the 10x HDF5 file")

    if matrix_cell_by_gene.shape[0] != len(barcodes):
        raise RuntimeError("Barcode count does not match RNA matrix rows")
    if matrix_cell_by_gene.shape[1] != len(genes):
        raise RuntimeError("Gene count does not match RNA matrix columns")
    return matrix_cell_by_gene, barcodes, genes


def _load_coldata(rda_path: Path) -> pd.DataFrame:
    payload = rdata.read_rda(str(rda_path))
    coldata = payload["pbmc_colData"]
    frame = pd.DataFrame(coldata.listData, index=pd.Index(coldata.rownames, name="barcode"))
    frame.index = frame.index.astype(str)
    frame["celltype"] = frame["celltype"].astype(str)
    frame["broad_celltype"] = frame["broad_celltype"].astype(str)
    return frame


def load_pbmc10x_subclusters(
    *,
    cache_dir: str = "data_cache/pbmc10x_subclusters",
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_dir)
    h5_path = ensure_download(PBMC10X_RNA_H5_URL, cache_path / "pbmc_rna_tenx.h5")
    coldata_path = ensure_download(PBMC10X_COLDATA_RDA_URL, cache_path / "pbmc_colData.rda")

    x, barcodes, genes = _load_tenx_counts(h5_path)
    coldata = _load_coldata(coldata_path)

    positions = pd.Index(barcodes, name="barcode").get_indexer(coldata.index)
    if np.any(positions < 0):
        missing = coldata.index[positions < 0][:10].tolist()
        raise RuntimeError(f"Could not align all PBMC10x barcodes; examples: {missing}")

    x_aligned = x[positions].tocsr()
    labels = coldata["celltype"].to_numpy(dtype=object)

    meta = {
        "dataset_name": "pbmc10x_subclusters",
        "rows_total": int(x_aligned.shape[0]),
        "genes": int(x_aligned.shape[1]),
        "nnz": int(x_aligned.nnz),
        "density": float(x_aligned.nnz / (x_aligned.shape[0] * x_aligned.shape[1])),
        "celltype_counts": coldata["celltype"].value_counts().to_dict(),
        "broad_celltype_counts": coldata["broad_celltype"].value_counts().to_dict(),
        "rna_h5_url": PBMC10X_RNA_H5_URL,
        "coldata_url": PBMC10X_COLDATA_RDA_URL,
        "cache_dir": str(cache_path),
        "gene_preview": [str(item) for item in genes[:10]],
    }
    return x_aligned, labels, meta


def select_binary_pair(
    x: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    positive_label: str,
    negative_label: str,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    keep = np.isin(labels, [positive_label, negative_label])
    if int(np.sum(keep)) == 0:
        raise ValueError("No rows matched the requested PBMC10x subcluster labels")
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
