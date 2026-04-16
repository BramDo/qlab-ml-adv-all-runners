#!/usr/bin/env python3
"""Semi-synthetic task helpers on the real GSE132080 source."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

import qiskit_qos_gse132080_thirdorder_screen as thirdorder
import qiskit_qos_gse132080_utils as gse132080


def load_hard_polr1d_pair(
    *,
    cache_dir: str = "data_cache/gse132080",
    positive_guide: str = "POLR1D_+_28196016.23-P1_08",
    negative_guide: str = "POLR1D_+_28196016.23-P1_00",
):
    x, metadata, source_meta = gse132080.load_gse132080(cache_dir=cache_dir)
    x_pair, guide_y, pair_meta = gse132080.select_guide_pair(
        x,
        metadata,
        positive_guide=positive_guide,
        negative_guide=negative_guide,
        require_good_coverage=True,
    )
    return x_pair, guide_y.astype(np.int64), source_meta, pair_meta


def _balanced_labels_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    labels = np.ones(scores.shape[0], dtype=np.int64)
    labels[order[: scores.shape[0] // 2]] = -1
    return labels


def build_residualized_semisynth_labels(
    x_pair,
    guide_y: np.ndarray,
    *,
    teacher_dim: int = 65536,
    shortcut_dim: int = 4096,
    hash_seed: int = 7,
    shortcut_hash_seed: int | None = None,
    value_mode: str = "log-product",
    max_active_genes: int | None = 48,
    hash_repeats: int = 2,
    signed_hash: bool = True,
    activation_scale: float = 2.0,
    seed: int = 7,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    shortcut_seed = int(hash_seed if shortcut_hash_seed is None else shortcut_hash_seed)
    teacher_matrix, teacher_stats = thirdorder.build_thirdorder_hashed_matrix(
        x_pair,
        feature_dim=int(teacher_dim),
        hash_seed=hash_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
        hash_repeats=hash_repeats,
        signed_hash=signed_hash,
        activation_scale=activation_scale,
    )
    shortcut_matrix, shortcut_stats = thirdorder.build_thirdorder_hashed_matrix(
        x_pair,
        feature_dim=int(shortcut_dim),
        hash_seed=shortcut_seed,
        value_mode=value_mode,
        max_active_genes=max_active_genes,
        hash_repeats=hash_repeats,
        signed_hash=signed_hash,
        activation_scale=activation_scale,
    )

    rng = np.random.default_rng(seed)
    teacher_weights = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=int(teacher_dim))
    raw_scores = np.asarray(teacher_matrix, dtype=np.float64) @ teacher_weights
    raw_scores /= max(1.0, np.sqrt(float(teacher_dim)))
    raw_scores -= float(np.mean(raw_scores))

    guide_projection = guide_y.astype(np.float64, copy=False)
    guide_norm_sq = float(np.dot(guide_projection, guide_projection))
    if guide_norm_sq > 0.0:
        guide_coef = float(np.dot(guide_projection, raw_scores) / guide_norm_sq)
        guide_component = guide_coef * guide_projection
    else:
        guide_coef = 0.0
        guide_component = np.zeros_like(raw_scores)
    guide_residual = raw_scores - guide_component

    shortcut_model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    shortcut_model.fit(np.asarray(shortcut_matrix, dtype=np.float64), guide_residual)
    shortcut_prediction = shortcut_model.predict(np.asarray(shortcut_matrix, dtype=np.float64))
    residual_scores = guide_residual - shortcut_prediction
    residual_scores -= float(np.mean(residual_scores))

    labels = _balanced_labels_from_scores(residual_scores)
    positive_mask = labels > 0
    negative_mask = labels < 0

    raw_var = float(np.var(raw_scores))
    guide_residual_var = float(np.var(guide_residual))
    residual_var = float(np.var(residual_scores))

    meta = {
        "task_type": "residualized_semisynth_thirdorder",
        "teacher_dim": int(teacher_dim),
        "shortcut_dim_projected_out": int(shortcut_dim),
        "teacher_weight_bytes": int(thirdorder.linear_model_bytes(teacher_dim)),
        "teacher_weight_human": thirdorder.human_bytes(thirdorder.linear_model_bytes(teacher_dim)),
        "teacher_stats": teacher_stats,
        "shortcut_stats": shortcut_stats,
        "teacher_hash_seed": int(hash_seed),
        "shortcut_hash_seed": int(shortcut_seed),
        "hash_repeats": int(hash_repeats),
        "signed_hash": bool(signed_hash),
        "activation_scale": float(activation_scale),
        "value_mode": value_mode,
        "ridge_alpha": float(ridge_alpha),
        "raw_score_std": float(np.std(raw_scores)),
        "guide_projection_coef": guide_coef,
        "guide_projection_r2": 0.0 if raw_var <= 0.0 else float(1.0 - (guide_residual_var / raw_var)),
        "shortcut_projection_r2": 0.0 if guide_residual_var <= 0.0 else float(1.0 - (residual_var / guide_residual_var)),
        "guide_label_correlation_final": float(np.corrcoef(guide_projection, residual_scores)[0, 1]),
        "positive_count": int(np.sum(positive_mask)),
        "negative_count": int(np.sum(negative_mask)),
        "positive_score_mean": float(np.mean(residual_scores[positive_mask])),
        "negative_score_mean": float(np.mean(residual_scores[negative_mask])),
        "score_gap": float(np.mean(residual_scores[positive_mask]) - np.mean(residual_scores[negative_mask])),
    }
    return labels, meta
