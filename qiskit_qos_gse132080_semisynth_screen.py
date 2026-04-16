#!/usr/bin/env python3
"""Classical budget screen on a semi-synthetic hard task over real GSE132080 cells."""

from __future__ import annotations

import argparse
import json
import os
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

import qiskit_qos_gse132080_semisynth_utils as semisynth
import qiskit_qos_gse132080_thirdorder_screen as thirdorder
from qiskit_qos_hash_streaming_genomics_runner import human_bytes

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a semi-synthetic GSE132080 classical budget screen.")
    parser.add_argument("--cache-dir", default="data_cache/gse132080")
    parser.add_argument("--positive-guide", default="POLR1D_+_28196016.23-P1_08")
    parser.add_argument("--negative-guide", default="POLR1D_+_28196016.23-P1_00")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hash-seed", type=int, default=7)
    parser.add_argument("--shortcut-hash-seed", type=int)
    parser.add_argument("--feature-dims", default="256,1024,4096,16384,65536")
    parser.add_argument("--teacher-dim", type=int, default=65536)
    parser.add_argument("--shortcut-dim", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--max-active-genes", type=int, default=48)
    parser.add_argument("--value-mode", choices=["binary", "log-product"], default="log-product")
    parser.add_argument("--hash-repeats", type=int, default=2)
    parser.add_argument("--signed-hash", action="store_true", default=True)
    parser.add_argument("--activation-scale", type=float, default=2.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def render_plot(budget_rows: list[dict[str, Any]], *, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.asarray([row["feature_dim"] for row in budget_rows], dtype=np.int64)
    y = np.asarray([row["best_classical_accuracy"] for row in budget_rows], dtype=np.float64)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hashed third-order feature dimension")
    ax.set_ylabel("Best classical test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("GSE132080 semi-synthetic comfort screen")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    feature_dims = thirdorder.parse_int_list(args.feature_dims)
    x_pair, guide_y, source_meta, pair_meta = semisynth.load_hard_polr1d_pair(
        cache_dir=args.cache_dir,
        positive_guide=args.positive_guide,
        negative_guide=args.negative_guide,
    )
    y_semisynth, task_meta = semisynth.build_residualized_semisynth_labels(
        x_pair,
        guide_y,
        teacher_dim=args.teacher_dim,
        shortcut_dim=args.shortcut_dim,
        hash_seed=args.hash_seed,
        shortcut_hash_seed=args.shortcut_hash_seed,
        value_mode=args.value_mode,
        max_active_genes=args.max_active_genes,
        hash_repeats=args.hash_repeats,
        signed_hash=args.signed_hash,
        activation_scale=args.activation_scale,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
    )

    train_idx, test_idx = thirdorder.balanced_binary_split(
        y_semisynth,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    x_train = x_pair[train_idx]
    x_test = x_pair[test_idx]
    y_train = y_semisynth[train_idx]
    y_test = y_semisynth[test_idx]

    budget_rows = [
        thirdorder.evaluate_budget(
            x_train,
            x_test,
            y_train,
            y_test,
            feature_dim=feature_dim,
            hash_seed=args.hash_seed,
            value_mode=args.value_mode,
            max_active_genes=args.max_active_genes,
            hash_repeats=args.hash_repeats,
            signed_hash=args.signed_hash,
            activation_scale=args.activation_scale,
            seed=args.seed,
        )
        for feature_dim in feature_dims
    ]

    best_accuracy = max(row["best_classical_accuracy"] for row in budget_rows)
    best_smallest = min(
        [row for row in budget_rows if row["best_classical_accuracy"] == best_accuracy],
        key=lambda row: (int(row["best_classical_bytes"]), int(row["feature_dim"])),
    )

    ambient_feature_dim = int(comb(int(x_pair.shape[1]), 3))
    ambient_dense_weight_bytes = int(ambient_feature_dim * 8)
    payload = {
        "config": {
            "cache_dir": args.cache_dir,
            "positive_guide": args.positive_guide,
            "negative_guide": args.negative_guide,
            "seed": args.seed,
            "hash_seed": args.hash_seed,
            "shortcut_hash_seed": args.shortcut_hash_seed,
            "feature_dims": feature_dims,
            "teacher_dim": args.teacher_dim,
            "shortcut_dim": args.shortcut_dim,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "max_active_genes": args.max_active_genes,
            "value_mode": args.value_mode,
            "hash_repeats": args.hash_repeats,
            "signed_hash": args.signed_hash,
            "activation_scale": args.activation_scale,
            "ridge_alpha": args.ridge_alpha,
        },
        "source": {
            **source_meta,
            **pair_meta,
            "thirdorder_ambient_feature_dim": ambient_feature_dim,
            "thirdorder_ambient_dense_weight_bytes": ambient_dense_weight_bytes,
            "thirdorder_ambient_dense_weight_human": human_bytes(ambient_dense_weight_bytes),
        },
        "semisynth_task": task_meta,
        "split": {
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "class_balance_train": {
                "positive": int(np.sum(y_train > 0)),
                "negative": int(np.sum(y_train < 0)),
            },
            "class_balance_test": {
                "positive": int(np.sum(y_test > 0)),
                "negative": int(np.sum(y_test < 0)),
            },
        },
        "best_overall_accuracy": float(best_accuracy),
        "smallest_best_budget": {
            "feature_dim": int(best_smallest["feature_dim"]),
            "best_classical_name": best_smallest["best_classical_name"],
            "best_classical_bytes": int(best_smallest["best_classical_bytes"]),
            "best_classical_bytes_human": best_smallest["best_classical_bytes_human"],
        },
        "budget_rows": budget_rows,
        "notes": [
            "Labels are generated on real GSE132080 cells by a hidden high-dimensional third-order teacher.",
            "The hidden teacher score is residualized against the original guide label and a smaller hashed shortcut before thresholding into binary labels.",
            "If small classical models still solve this task cheaply, the semi-synthetic route did not actually remove the shortcut.",
        ],
    }

    json_out = args.json_out or "qiskit_qos_gse132080_semisynth_screen_64x64.json"
    plot_out = args.plot_out or "qiskit_qos_gse132080_semisynth_screen_64x64.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(budget_rows, output_path=plot_out)

    print("GSE132080 semi-synthetic screen")
    print(f"- pair: {args.positive_guide} vs {args.negative_guide}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"- ambient dense classical weight memory: {human_bytes(ambient_dense_weight_bytes)}")
    print(
        f"- teacher_dim={args.teacher_dim} shortcut_dim_projected_out={args.shortcut_dim} "
        f"shortcut_hash_seed={task_meta['shortcut_hash_seed']} "
        f"guide_projection_r2={task_meta['guide_projection_r2']:.3f} "
        f"shortcut_projection_r2={task_meta['shortcut_projection_r2']:.3f}"
    )
    for row in budget_rows:
        print(
            f"- d={row['feature_dim']}: best={row['best_classical_accuracy']:.3f} "
            f"via {row['best_classical_name']} ({row['best_classical_bytes_human']})"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
