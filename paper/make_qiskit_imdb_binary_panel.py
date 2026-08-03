#!/usr/bin/env python3
"""Render a paper-style IMDb binary classification panel for the Qiskit bridge."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
ASSET_DIR = PAPER_DIR / "assets"
OFFICIAL_ACCURACY_JSON = ROOT / "official_qos" / "real_datasets" / "imdb_size_vs_accuracy.json"
QISKIT_SWEEP_GLOB = "qiskit_official_qos_imdb_classifier_proof_min2_d*_256x256_abs.json"


plt.rcParams.update(
    {
        "font.family": "sans",
        "font.serif": ["Google Sans"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (5.6, 4.2),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "legend.frameon": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)


COLORS = {
    "official_streaming": "#2657AF",
    "official_sparse": "#707070",
    "qiskit_quantum": "#CD591A",
    "bounded_raw": "#2C8E5A",
}


def load_official_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(OFFICIAL_ACCURACY_JSON.read_text(encoding="utf-8"))
    raw = payload["raw_data_by_min_df"]
    min_dfs = sorted(int(k) for k in raw)

    streaming_acc = []
    streaming_space = []
    sparse_acc = []
    sparse_space = []

    for min_df in min_dfs:
        point = raw[str(min_df)]
        streaming_acc.append(float(point["streaming"]["accuracy_mean"]))
        streaming_space.append(float(point["streaming"]["space"]))
        sparse_acc.append(float(point["sparse"]["accuracy_mean"]))
        sparse_space.append(float(point["sparse"]["space"]))

    return (
        np.asarray(streaming_acc),
        np.asarray(streaming_space),
        np.asarray(sparse_acc),
        np.asarray(sparse_space),
    )


def load_qiskit_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = []
    for path in sorted(ROOT.glob(QISKIT_SWEEP_GLOB)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            (
                int(payload["effective_bridge_dim"]),
                float(payload["raw_baseline"]["linearsvc_accuracy"]),
                float(payload["quantum_feature_classifier"]["linearsvc_accuracy"]),
            )
        )

    if not records:
        raise FileNotFoundError(f"No classifier-proof sweep files found for {QISKIT_SWEEP_GLOB}")

    records.sort(key=lambda x: x[0])
    dims = np.asarray([r[0] for r in records], dtype=float)
    raw_acc = np.asarray([r[1] for r in records], dtype=float)
    quantum_acc = np.asarray([r[2] for r in records], dtype=float)
    return dims, raw_acc, quantum_acc


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    streaming_acc, streaming_space, sparse_acc, sparse_space = load_official_curves()
    dims, raw_acc, quantum_acc = load_qiskit_sweep()

    fig, ax = plt.subplots()

    ax.plot(
        sparse_acc,
        sparse_space,
        color=COLORS["official_sparse"],
        linewidth=1.5,
        alpha=0.9,
    )
    ax.scatter(
        sparse_acc,
        sparse_space,
        marker="X",
        color=COLORS["official_sparse"],
        s=42,
        label="Official sparse / QRAM",
        alpha=0.9,
    )

    ax.plot(
        streaming_acc,
        streaming_space,
        color=COLORS["official_streaming"],
        linewidth=1.5,
        alpha=0.9,
    )
    ax.scatter(
        streaming_acc,
        streaming_space,
        marker="P",
        color=COLORS["official_streaming"],
        s=46,
        label="Official streaming",
        alpha=0.9,
    )

    ax.plot(raw_acc, dims, color=COLORS["bounded_raw"], linewidth=1.6, alpha=0.95)
    ax.scatter(
        raw_acc,
        dims,
        marker="o",
        color=COLORS["bounded_raw"],
        s=34,
        label="Bounded raw baseline",
        alpha=0.95,
    )

    ax.plot(quantum_acc, dims, color=COLORS["qiskit_quantum"], linewidth=1.6, alpha=0.95)
    ax.scatter(
        quantum_acc,
        dims,
        marker="D",
        color=COLORS["qiskit_quantum"],
        s=34,
        label="Bounded Qiskit quantum",
        alpha=0.95,
    )

    halo = [pe.withStroke(linewidth=3, foreground="white")]
    ax.text(
        0.80,
        2.4e6,
        "Official sparse / QRAM",
        color=COLORS["official_sparse"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.845,
        1.4e4,
        "Official streaming",
        color=COLORS["official_streaming"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.475,
        18,
        "Bounded Qiskit quantum",
        color=COLORS["qiskit_quantum"],
        fontsize=10,
        path_effects=halo,
    )
    ax.text(
        0.475,
        55,
        "Bounded raw baseline",
        color=COLORS["bounded_raw"],
        fontsize=10,
        path_effects=halo,
    )

    ax.set_yscale("log")
    ax.set_ylim(8, 1e7)
    ax.set_xlim(0.45, 0.91)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Machine size")
    ax.set_title("Binary classification")
    ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])
    ax.set_xticklabels(["45%", "50%", "55%", "60%", "70%", "80%", "90%"])
    ax.tick_params(direction="in", which="both", top=False, right=True)
    ax.grid(True, which="major", ls="-", alpha=0.1)
    fig.tight_layout()

    png_path = ASSET_DIR / "qiskit_imdb_binary_classification_panel.png"
    pdf_path = ASSET_DIR / "qiskit_imdb_binary_classification_panel.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
