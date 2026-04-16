#!/usr/bin/env python3
"""Utilities for loading the UCI Dorothea dataset.

This stays separate from the earlier text/20NG paths. The dataset is cached
locally so repeat runs do not depend on OpenML availability.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


DOROTHEA_URL = "https://archive.ics.uci.edu/static/public/169/dorothea.zip"
DOROTHEA_FEATURE_DIM = 100000
REQUIRED_FILES = (
    "dorothea_train.data",
    "dorothea_train.labels",
    "dorothea_valid.data",
    "dorothea_valid.labels",
)


def ensure_dorothea_cache(data_dir: str | os.PathLike[str], *, download_url: str = DOROTHEA_URL) -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    if all((path / name).exists() for name in REQUIRED_FILES):
        return path

    with urllib.request.urlopen(download_url, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            if not filename.startswith("dorothea_"):
                continue
            target_path = path / filename
            with zf.open(member) as src, target_path.open("wb") as dst:
                dst.write(src.read())

    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Dorothea extraction incomplete; missing files: {missing}")
    return path


def _parse_sparse_lines(data_path: Path, *, feature_dim: int) -> sp.csr_matrix:
    row_ind: list[int] = []
    col_ind: list[int] = []
    data_val: list[int] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            indices = [int(token) for token in line.strip().split() if token]
            for index in indices:
                row_ind.append(row)
                col_ind.append(index - 1)
                data_val.append(1)

    num_rows = row + 1 if "row" in locals() else 0
    return sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(num_rows, feature_dim), dtype=np.float64)


def _load_subset(data_dir: Path, subset: str, *, feature_dim: int) -> tuple[sp.csr_matrix, np.ndarray]:
    data_path = data_dir / f"dorothea_{subset}.data"
    labels_path = data_dir / f"dorothea_{subset}.labels"
    if not data_path.exists():
        raise FileNotFoundError(f"missing Dorothea data file: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing Dorothea labels file: {labels_path}")

    x = _parse_sparse_lines(data_path, feature_dim=feature_dim)
    y = np.loadtxt(labels_path, dtype=np.int64)
    return x, y


def load_dorothea_sparse(
    *,
    data_dir: str | os.PathLike[str],
    merge_valid: bool = True,
    feature_dim: int = DOROTHEA_FEATURE_DIM,
) -> tuple[sp.csr_matrix, np.ndarray, dict[str, object]]:
    cache_dir = ensure_dorothea_cache(data_dir)
    x_train, y_train = _load_subset(cache_dir, "train", feature_dim=feature_dim)
    if merge_valid:
        x_valid, y_valid = _load_subset(cache_dir, "valid", feature_dim=feature_dim)
        x = sp.vstack([x_train, x_valid], format="csr")
        y = np.concatenate([y_train, y_valid])
    else:
        x = x_train
        y = y_train

    metadata = {
        "rows": int(x.shape[0]),
        "raw_feature_dim": int(x.shape[1]),
        "nnz": int(x.nnz),
        "density": float(x.nnz / (x.shape[0] * x.shape[1])),
        "positive_count": int(np.sum(y > 0)),
        "negative_count": int(np.sum(y < 0)),
        "merge_valid": bool(merge_valid),
        "cache_dir": str(cache_dir),
    }
    return x, y.astype(np.float64), metadata


def balance_binary_dataset(
    x: sp.csr_matrix | np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[sp.csr_matrix | np.ndarray, np.ndarray, dict[str, object]]:
    pos_idx = np.flatnonzero(y > 0)
    neg_idx = np.flatnonzero(y < 0)
    keep_per_class = int(min(len(pos_idx), len(neg_idx)))
    if keep_per_class == 0:
        raise ValueError("cannot balance dataset with an empty class")
    rng = np.random.default_rng(seed)
    pos_pick = np.sort(rng.choice(pos_idx, size=keep_per_class, replace=False))
    neg_pick = np.sort(rng.choice(neg_idx, size=keep_per_class, replace=False))
    keep = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(keep)
    return x[keep], y[keep], {
        "balanced": True,
        "balanced_rows": int(len(keep)),
        "balanced_positive_count": int(keep_per_class),
        "balanced_negative_count": int(keep_per_class),
    }


def load_dorothea_svd(
    *,
    data_dir: str | os.PathLike[str],
    svd_components: int,
    seed: int,
    merge_valid: bool = True,
    balance_classes: bool = False,
    feature_dim: int = DOROTHEA_FEATURE_DIM,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_sparse, y, metadata = load_dorothea_sparse(
        data_dir=data_dir,
        merge_valid=merge_valid,
        feature_dim=feature_dim,
    )
    if balance_classes:
        x_sparse, y, balance_meta = balance_binary_dataset(x_sparse, y, seed=seed)
        metadata = {
            **metadata,
            **balance_meta,
            "rows": int(x_sparse.shape[0]),
            "nnz": int(x_sparse.nnz),
            "density": float(x_sparse.nnz / (x_sparse.shape[0] * x_sparse.shape[1])),
            "positive_count": int(np.sum(y > 0)),
            "negative_count": int(np.sum(y < 0)),
        }
    max_rank = max(1, min(x_sparse.shape[0] - 1, x_sparse.shape[1] - 1))
    n_components = min(int(svd_components), max_rank)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    x_dense = np.asarray(svd.fit_transform(x_sparse), dtype=np.float64)
    metadata = {
        **metadata,
        "reduced_feature_dim": int(x_dense.shape[1]),
        "svd_components_used": int(n_components),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    return x_dense, y, metadata
