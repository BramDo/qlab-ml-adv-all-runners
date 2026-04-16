#!/usr/bin/env python3
"""Multi-pair 20 Newsgroups benchmark for the QOS surrogate.

This is a separate extension that keeps the earlier toy/scaling runners intact.
It evaluates the same quantum classifier on multiple real 20NG category pairs,
then summarizes mean/std accuracy and feature-space size across pairs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import qiskit_qos_scaling_runner as scaling
import qiskit_qos_text_runner as text_runner
import qiskit_qos_toy_model as toy

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


FLOAT64_BYTES = 8


@dataclass
class PairBundle:
    categories: list[str]
    texts: pd.Series
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, object]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    out = [int(item) for item in items]
    if any(item <= 0 for item in out):
        raise ValueError("all values must be positive")
    return out


def parse_pair_specs(value: str) -> list[list[str]]:
    pairs: list[list[str]] = []
    for block in value.split(";"):
        block = block.strip()
        if not block:
            continue
        cats = [item.strip() for item in block.split(",") if item.strip()]
        if len(cats) != 2:
            raise ValueError("each --category-pairs entry must contain exactly two comma-separated categories")
        pairs.append(cats)
    if not pairs:
        raise ValueError("no valid category pairs parsed")
    return pairs


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def select_pairs(args: argparse.Namespace) -> list[list[str]]:
    if args.category_pairs:
        return parse_pair_specs(args.category_pairs)

    target_names = fetch_20newsgroups(subset="train").target_names
    all_pairs = list(itertools.combinations(target_names, 2))
    rng = np.random.default_rng(args.pair_seed)
    if args.n_pairs > len(all_pairs):
        raise ValueError(f"requested {args.n_pairs} pairs but only {len(all_pairs)} unique pairs exist")
    pick = rng.choice(len(all_pairs), size=args.n_pairs, replace=False)
    pick.sort()
    return [list(all_pairs[idx]) for idx in pick]


def load_pair_bundle(
    *,
    categories: list[str],
    max_features: int,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    analyzer: str,
    svd_components: int,
    seed: int,
) -> PairBundle:
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    texts = text_runner.clean_text(pd.Series(dataset.data), strip_literals=[])
    x_dense, feature_meta = text_runner.tfidf_svd_features(
        texts,
        max_features=max_features,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        analyzer=analyzer,
        svd_components=svd_components,
        seed=seed,
    )
    y = np.where(np.asarray(dataset.target, dtype=np.int64) == 1, 1.0, -1.0)
    metadata = {
        "source_name": f"20ng-{categories[0]}-vs-{categories[1]}",
        "rows": int(len(x_dense)),
        "raw_feature_dim": int(feature_meta["vocab_size"]),
        "reduced_feature_dim": int(x_dense.shape[1]),
        "target_names": [str(name) for name in dataset.target_names],
        "text_features": feature_meta,
    }
    return PairBundle(
        categories=categories,
        texts=texts,
        x=np.asarray(x_dense, dtype=np.float64),
        y=y.astype(np.float64),
        metadata=metadata,
    )


def split_like_toy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = len(x)
    n_train = int(round(train_fraction * total))
    n_train = max(2, min(int(n_train), total - 1))
    train_idx, test_idx = toy.split_train_test(x, n_train=n_train, rng=rng)

    train_dummy = np.asarray(x[train_idx], dtype=np.float64)
    test_dummy = np.asarray(x[test_idx], dtype=np.float64)
    y_train = np.asarray(y[train_idx], dtype=np.float64)
    y_test = np.asarray(y[test_idx], dtype=np.float64)
    train_sub, y_train_sub = toy.subsample_rows(train_dummy, y_train, max_rows=max_train_samples, rng=rng)
    test_sub, y_test_sub = toy.subsample_rows(test_dummy, y_test, max_rows=max_test_samples, rng=rng)

    def recover_indices(original: np.ndarray, subset: np.ndarray) -> np.ndarray:
        if len(subset) == 0:
            return np.array([], dtype=int)
        index_map: dict[bytes, list[int]] = {}
        for idx, row in enumerate(original):
            index_map.setdefault(np.ascontiguousarray(row).tobytes(), []).append(idx)
        out: list[int] = []
        for row in subset:
            key = np.ascontiguousarray(row).tobytes()
            out.append(index_map[key].pop(0))
        return np.asarray(out, dtype=int)

    local_train = recover_indices(train_dummy, train_sub)
    local_test = recover_indices(test_dummy, test_sub)
    return train_idx[local_train], test_idx[local_test]


def hashing_baseline_accuracy(
    *,
    texts_train: pd.Series,
    texts_test: pd.Series,
    y_train: np.ndarray,
    y_test: np.ndarray,
    analyzer: str,
    ngram_min: int,
    ngram_max: int,
    hash_features: int,
    seed: int,
) -> tuple[float, int]:
    hashing = HashingVectorizer(
        lowercase=True,
        n_features=hash_features,
        alternate_sign=False,
        norm="l2",
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
    )
    x_train_hash = hashing.transform(texts_train)
    x_test_hash = hashing.transform(texts_test)
    sgd_hash = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed, max_iter=5000, tol=1e-4)
    sgd_hash.fit(x_train_hash, y_train)
    return float(np.mean(sgd_hash.predict(x_test_hash) == y_test)), int((hash_features + 1) * FLOAT64_BYTES)


def summarize_runs(rows: list[dict[str, object]], *, qubits: list[int]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for q in qubits:
        q_rows = [row for row in rows if int(row["num_qubits"]) == q]
        if not q_rows:
            continue

        def collect(key: str) -> list[float]:
            return [float(row[key]) for row in q_rows]

        summary.append(
            {
                "num_qubits": int(q),
                "pair_count": int(len(q_rows)),
                "quantum_test_accuracy_mean": float(np.mean(collect("test_accuracy_quantum"))),
                "quantum_test_accuracy_std": float(np.std(collect("test_accuracy_quantum"))),
                "classical_scaling_test_accuracy_mean": float(np.mean(collect("test_accuracy_classical_scaling"))),
                "classical_scaling_test_accuracy_std": float(np.std(collect("test_accuracy_classical_scaling"))),
                "hashing_test_accuracy_mean": float(np.mean(collect("test_accuracy_hashing"))),
                "hashing_test_accuracy_std": float(np.std(collect("test_accuracy_hashing"))),
                "raw_feature_dim_mean": float(np.mean(collect("raw_feature_dim"))),
                "reduced_feature_dim_mean": float(np.mean(collect("reduced_feature_dim"))),
                "quantum_beats_scaling_count": int(sum(float(row["test_accuracy_quantum"]) > float(row["test_accuracy_classical_scaling"]) for row in q_rows)),
                "quantum_beats_hashing_count": int(sum(float(row["test_accuracy_quantum"]) > float(row["test_accuracy_hashing"]) for row in q_rows)),
            }
        )
    return summary


def render_plot(summary_rows: list[dict[str, object]], *, output_path: str) -> None:
    qubits = [int(row["num_qubits"]) for row in summary_rows]
    q_mean = [float(row["quantum_test_accuracy_mean"]) for row in summary_rows]
    q_std = [float(row["quantum_test_accuracy_std"]) for row in summary_rows]
    c_mean = [float(row["classical_scaling_test_accuracy_mean"]) for row in summary_rows]
    c_std = [float(row["classical_scaling_test_accuracy_std"]) for row in summary_rows]
    h_mean = [float(row["hashing_test_accuracy_mean"]) for row in summary_rows]
    h_std = [float(row["hashing_test_accuracy_std"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(qubits, q_mean, yerr=q_std, marker="o", label="quantum toy")
    ax.errorbar(qubits, c_mean, yerr=c_std, marker="s", label="old scaling baseline")
    ax.errorbar(qubits, h_mean, yerr=h_std, marker="^", label="hashing baseline")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("20NG pair average accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-pair 20 Newsgroups QOS benchmarks.")
    parser.add_argument("--qubits", default="10", help="Comma-separated qubit counts to sweep")
    parser.add_argument("--n-pairs", type=int, default=3, help="How many random category pairs to sample when --category-pairs is omitted")
    parser.add_argument("--pair-seed", type=int, default=17)
    parser.add_argument("--category-pairs", help="Semicolon-separated explicit pairs, e.g. 'alt.atheism,sci.space;rec.autos,sci.med'")
    parser.add_argument("--readout-shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--encoder", default="ridge", choices=["block", "pca", "ridge", "lda"])
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--execution-mode", default="sampler-sim", choices=["statevector", "sampler-sim", "ibm-hardware"])
    parser.add_argument("--backend-name")
    parser.add_argument("--simulator-method", default="matrix_product_state", choices=["automatic", "statevector", "matrix_product_state"])
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--readout-mitigation", action="store_true")
    parser.add_argument("--cal-shots", type=int, default=512)
    parser.add_argument("--extra-error-suppression", action="store_true")
    parser.add_argument("--dd-sequence", default="XY4")
    parser.add_argument("--twirl-randomizations", type=int, default=8)
    parser.add_argument("--tfidf-max-features", type=int, default=120000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--tfidf-ngram-min", type=int, default=3)
    parser.add_argument("--tfidf-ngram-max", type=int, default=5)
    parser.add_argument("--tfidf-analyzer", default="char_wb", choices=["word", "char", "char_wb"])
    parser.add_argument("--svd-components", type=int, default=512)
    parser.add_argument("--hash-features", type=int, default=4096)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qubits = parse_int_list(args.qubits)
    pairs = select_pairs(args)
    rows: list[dict[str, object]] = []

    execution_config = scaling.build_execution_config(args)

    for pair_index, categories in enumerate(pairs):
        pair_seed = int(args.seed + pair_index)
        bundle = load_pair_bundle(
            categories=categories,
            max_features=args.tfidf_max_features,
            min_df=args.tfidf_min_df,
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            analyzer=args.tfidf_analyzer,
            svd_components=args.svd_components,
            seed=pair_seed,
        )
        train_idx, test_idx = split_like_toy(
            bundle.x,
            bundle.y,
            seed=pair_seed,
            train_fraction=args.train_fraction,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        texts_train = bundle.texts.iloc[train_idx]
        texts_test = bundle.texts.iloc[test_idx]
        y_train = bundle.y[train_idx].astype(int)
        y_test = bundle.y[test_idx].astype(int)
        hash_acc, hash_bytes = hashing_baseline_accuracy(
            texts_train=texts_train,
            texts_test=texts_test,
            y_train=y_train,
            y_test=y_test,
            analyzer=args.tfidf_analyzer,
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            hash_features=args.hash_features,
            seed=pair_seed,
        )

        for q in qubits:
            start = time.perf_counter()
            result = toy.run_classification_from_arrays(
                x=bundle.x,
                y=bundle.y,
                num_qubits=q,
                readout_shots=args.readout_shots,
                seed=pair_seed,
                train_fraction=args.train_fraction,
                encoder_method=args.encoder,
                quantum_head_method=args.quantum_head,
                readout_family=args.readout_family,
                execution_config=execution_config,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "pair_index": int(pair_index),
                    "pair_label": f"{categories[0]} vs {categories[1]}",
                    "categories": categories,
                    "num_qubits": int(q),
                    "rows": int(bundle.metadata["rows"]),
                    "raw_feature_dim": int(bundle.metadata["raw_feature_dim"]),
                    "reduced_feature_dim": int(bundle.metadata["reduced_feature_dim"]),
                    "test_accuracy_quantum": float(result["test_accuracy_quantum"]),
                    "test_accuracy_classical_scaling": float(result["test_accuracy_classical"]),
                    "test_accuracy_hashing": float(hash_acc),
                    "hashing_model_bytes": int(hash_bytes),
                    "hashing_model_bytes_human": human_bytes(hash_bytes),
                    "readout_feature_count": int(result["readout_feature_count"]),
                    "query_feature_count": int(result["query_feature_count"]),
                    "quantum_head_feature_count": int(result["quantum_head_feature_count"]),
                    "n_train_used": int(len(result["train_labels"])),
                    "n_test_used": int(len(result["test_labels"])),
                    "elapsed_seconds": float(elapsed),
                }
            )

    summary = summarize_runs(rows, qubits=qubits)
    payload = {
        "config": {
            "qubits": qubits,
            "n_pairs": int(len(pairs)),
            "pair_seed": args.pair_seed,
            "readout_shots": args.readout_shots,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "encoder": args.encoder,
            "quantum_head": args.quantum_head,
            "readout_family": args.readout_family,
            "execution_mode": args.execution_mode,
            "backend_name": args.backend_name,
            "simulator_method": args.simulator_method if args.execution_mode == "sampler-sim" else None,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "tfidf_max_features": args.tfidf_max_features,
            "tfidf_min_df": args.tfidf_min_df,
            "tfidf_ngram_min": args.tfidf_ngram_min,
            "tfidf_ngram_max": args.tfidf_ngram_max,
            "tfidf_analyzer": args.tfidf_analyzer,
            "svd_components": args.svd_components,
            "hash_features": args.hash_features,
        },
        "pairs": [{"categories": pair, "label": f"{pair[0]} vs {pair[1]}"} for pair in pairs],
        "pair_runs": rows,
        "summary_by_qubits": summary,
        "notes": [
            "This runner averages over multiple real 20 Newsgroups category pairs.",
            "The hashing baseline is a memory-bounded classical reference on the same text split.",
            "The old scaling baseline is the dense SVD ridge path already used by qiskit_qos_scaling_runner.py.",
        ],
    }

    stem = f"qiskit_qos_20ng_pair_runner_{len(pairs)}pairs"
    json_out = args.json_out or f"{stem}.json"
    plot_out = args.plot_out or f"{stem}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    if summary:
        render_plot(summary, output_path=plot_out)

    print("QOS 20NG pair runner")
    print(f"- pairs: {len(pairs)}")
    print(f"- execution mode: {args.execution_mode}")
    print(f"- tfidf analyzer: {args.tfidf_analyzer}")
    print(f"- tfidf ngram range: {args.tfidf_ngram_min}-{args.tfidf_ngram_max}")
    print("- summary:")
    for row in summary:
        print(
            "  "
            f"q={row['num_qubits']:<2d} "
            f"quantum={row['quantum_test_accuracy_mean']:.3f}±{row['quantum_test_accuracy_std']:.3f} "
            f"scaling_classical={row['classical_scaling_test_accuracy_mean']:.3f}±{row['classical_scaling_test_accuracy_std']:.3f} "
            f"hashing={row['hashing_test_accuracy_mean']:.3f}±{row['hashing_test_accuracy_std']:.3f} "
            f"raw_dim_mean={row['raw_feature_dim_mean']:.1f}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
