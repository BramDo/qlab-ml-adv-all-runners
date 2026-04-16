#!/usr/bin/env python3
"""Scaling runner for the Qiskit QOS surrogate on larger built-in datasets.

This file intentionally leaves qiskit_qos_toy_model.py unchanged and wraps it as a
separate scaling workflow. It supports a larger built-in numeric source and a larger
text source so we can probe qubit scaling without overwriting the earlier toy runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_digits

import qiskit_qos_dorothea_utils as dorothea_utils
import qiskit_qos_text_runner as text_runner
import qiskit_qos_toy_model as toy

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


@dataclass
class SourceBundle:
    x: np.ndarray
    y: np.ndarray
    source_name: str
    label_mapping: dict[str, float]
    metadata: dict[str, object]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    out = [int(item) for item in items]
    if any(item <= 0 for item in out):
        raise ValueError("qubit counts must be positive")
    return out


def default_output_stem(source: str) -> str:
    return source.replace("/", "_").replace(":", "_").replace("-", "_")


def resolve_20ng_categories(source_name: str, custom_categories: str | None = None) -> list[str]:
    if source_name == "20ng-atheism-vs-space":
        return ["alt.atheism", "sci.space"]
    if source_name == "20ng-graphics-vs-baseball":
        return ["comp.graphics", "rec.sport.baseball"]
    if source_name == "20ng-custom":
        if not custom_categories:
            raise ValueError("--20ng-categories is required for --source 20ng-custom")
        categories = [item.strip() for item in custom_categories.split(",") if item.strip()]
        if len(categories) != 2:
            raise ValueError("--20ng-categories must contain exactly two comma-separated categories")
        return categories
    raise ValueError(f"unsupported 20ng source: {source_name}")


def load_digits_even_vs_odd() -> SourceBundle:
    dataset = load_digits()
    x = np.asarray(dataset.data, dtype=np.float64)
    even_mask = (dataset.target % 2) == 0
    y = np.where(even_mask, 1.0, -1.0)
    metadata = {
        "dataset_kind": "numeric",
        "rows": int(len(x)),
        "raw_feature_dim": int(x.shape[1]),
        "target_name": "digit parity",
        "positive_count": int(np.sum(y > 0.0)),
        "negative_count": int(np.sum(y < 0.0)),
        "class_names": [str(name) for name in dataset.target_names.tolist()],
    }
    return SourceBundle(
        x=x,
        y=y,
        source_name="digits-even-vs-odd",
        label_mapping={"even": 1.0, "odd": -1.0},
        metadata=metadata,
    )


def load_20ng_binary(
    *,
    categories: list[str],
    max_features: int,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    analyzer: str,
    svd_components: int,
    seed: int,
) -> SourceBundle:
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
        "dataset_kind": "text",
        "rows": int(len(x_dense)),
        "raw_feature_dim": int(feature_meta["vocab_size"]),
        "reduced_feature_dim": int(x_dense.shape[1]),
        "target_names": [str(name) for name in dataset.target_names],
        "positive_count": int(np.sum(y > 0.0)),
        "negative_count": int(np.sum(y < 0.0)),
        "text_features": feature_meta,
    }
    return SourceBundle(
        x=np.asarray(x_dense, dtype=np.float64),
        y=y,
        source_name=f"20ng-{categories[0]}-vs-{categories[1]}",
        label_mapping={dataset.target_names[1]: 1.0, dataset.target_names[0]: -1.0},
        metadata=metadata,
    )


def load_dorothea_uci(
    *,
    cache_dir: str,
    svd_components: int,
    seed: int,
    merge_valid: bool = True,
    balance_classes: bool = False,
) -> SourceBundle:
    x_dense, y, metadata = dorothea_utils.load_dorothea_svd(
        data_dir=cache_dir,
        svd_components=svd_components,
        seed=seed,
        merge_valid=merge_valid,
        balance_classes=balance_classes,
    )
    metadata = {
        "dataset_kind": "sparse_binary",
        "target_name": "drug activity",
        **metadata,
    }
    return SourceBundle(
        x=np.asarray(x_dense, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        source_name="dorothea-uci",
        label_mapping={"active": 1.0, "inactive": -1.0},
        metadata=metadata,
    )


def load_source(args: argparse.Namespace) -> SourceBundle:
    if args.source == "digits-even-vs-odd":
        return load_digits_even_vs_odd()
    if args.source in {"20ng-atheism-vs-space", "20ng-graphics-vs-baseball", "20ng-custom"}:
        return load_20ng_binary(
            categories=resolve_20ng_categories(args.source, getattr(args, "twenty_ng_categories", None)),
            max_features=args.tfidf_max_features,
            min_df=args.tfidf_min_df,
            ngram_min=args.tfidf_ngram_min,
            ngram_max=args.tfidf_ngram_max,
            analyzer=args.tfidf_analyzer,
            svd_components=args.svd_components,
            seed=args.seed,
        )
    if args.source == "dorothea-uci":
        return load_dorothea_uci(
            cache_dir=args.dorothea_cache_dir,
            svd_components=args.svd_components,
            seed=args.seed,
            merge_valid=not args.dorothea_train_only,
            balance_classes=bool(getattr(args, "dorothea_balance", False)),
        )
    raise ValueError(f"unsupported source: {args.source}")


def build_execution_config(args: argparse.Namespace) -> toy.QuantumExecutionConfig:
    layout_strategy = str(os.environ.get("QISKIT_QOS_LAYOUT_STRATEGY", "quality-chain") or "quality-chain").strip().lower()
    if layout_strategy not in {"quality-chain", "none"}:
        raise ValueError(
            "QISKIT_QOS_LAYOUT_STRATEGY must be one of: quality-chain, none"
        )
    return toy.QuantumExecutionConfig(
        mode=args.execution_mode,
        backend_name=args.backend_name,
        optimization_level=args.optimization_level,
        simulator_method=args.simulator_method,
        readout_mitigation=args.readout_mitigation,
        cal_shots=args.cal_shots,
        extra_error_suppression=args.extra_error_suppression,
        dd_sequence=args.dd_sequence,
        twirl_randomizations=args.twirl_randomizations,
        layout_strategy=layout_strategy,
        debug_runtime=toy._env_flag("QISKIT_QOS_DEBUG_RUNTIME"),
        runtime_submit_batch_size=max(int(os.environ.get("QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE", "0") or 0), 0),
        feature_mapping_limit=max(int(os.environ.get("QISKIT_QOS_FEATURE_MAPPING_LIMIT", "0") or 0), 0),
    )


def render_scaling_plot(rows: list[dict[str, object]], *, output_path: str) -> None:
    qubits = [int(row["num_qubits"]) for row in rows]
    q_acc = [float(row["test_accuracy_quantum"]) for row in rows]
    c_acc = [float(row["test_accuracy_classical"]) for row in rows]
    runtimes = [float(row["elapsed_seconds"]) for row in rows]
    feature_counts = [int(row["readout_feature_count"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(qubits, q_acc, marker="o", label="quantum toy")
    axes[0].plot(qubits, c_acc, marker="s", label="classical baseline")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Accuracy vs qubits")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(qubits, runtimes, marker="o", label="runtime (s)")
    axes[1].plot(qubits, feature_counts, marker="s", label="readout features")
    axes[1].set_xlabel("Qubits")
    axes[1].set_title("Cost proxies vs qubits")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_payload(
    *,
    args: argparse.Namespace,
    source: SourceBundle,
    qubits: list[int],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "config": {
            "source": source.source_name,
            "qubits": qubits,
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
            "readout_mitigation": bool(args.readout_mitigation),
            "extra_error_suppression": bool(args.extra_error_suppression),
            "tfidf_max_features": args.tfidf_max_features,
            "tfidf_min_df": args.tfidf_min_df,
            "tfidf_ngram_min": args.tfidf_ngram_min,
            "tfidf_ngram_max": args.tfidf_ngram_max,
            "tfidf_analyzer": args.tfidf_analyzer,
            "twenty_ng_categories": getattr(args, "twenty_ng_categories", None),
            "dorothea_cache_dir": getattr(args, "dorothea_cache_dir", None),
            "dorothea_train_only": bool(getattr(args, "dorothea_train_only", False)),
            "dorothea_balance": bool(getattr(args, "dorothea_balance", False)),
        },
        "source": {
            "label_mapping": source.label_mapping,
            **source.metadata,
        },
        "runs": rows,
        "notes": [
            "This file is a separate scaling extension and does not replace qiskit_qos_toy_model.py.",
            "For larger qubit counts, prefer sampler-sim or ibm-hardware over statevector.",
            "Artifacts are checkpointed after each completed qubit run.",
        ],
    }


def checkpoint_artifacts(
    *,
    args: argparse.Namespace,
    source: SourceBundle,
    qubits: list[int],
    rows: list[dict[str, object]],
    json_out: str,
    plot_out: str,
) -> None:
    payload = build_payload(args=args, source=source, qubits=qubits, rows=rows)
    Path(json_out).write_text(json.dumps(payload, indent=2))
    if rows:
        render_scaling_plot(rows, output_path=plot_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qubit-scaling sweeps on larger built-in sources.")
    parser.add_argument(
        "--source",
        default="digits-even-vs-odd",
        choices=[
            "digits-even-vs-odd",
            "20ng-atheism-vs-space",
            "20ng-graphics-vs-baseball",
            "20ng-custom",
            "dorothea-uci",
        ],
        help="Built-in source to benchmark without touching earlier toy scripts.",
    )
    parser.add_argument("--20ng-categories", dest="twenty_ng_categories", help="Comma-separated custom 20NG categories when --source 20ng-custom")
    parser.add_argument("--dorothea-cache-dir", default="data_cache/dorothea", help="Local cache dir for UCI Dorothea download/extraction")
    parser.add_argument("--dorothea-train-only", action="store_true", help="Use only Dorothea train rows instead of merging train+valid before the benchmark split")
    parser.add_argument("--dorothea-balance", action="store_true", help="Downsample Dorothea to a balanced subset before the benchmark split so accuracy stays meaningful")
    parser.add_argument("--qubits", default="6,10,14", help="Comma-separated qubit counts to sweep.")
    parser.add_argument("--readout-shots", type=int, default=128, help="Readout shots for sampler-based modes.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--encoder", default="block", choices=["block", "pca", "ridge", "lda"])
    parser.add_argument("--quantum-head", default="ridge", choices=["cosine", "ridge", "logistic"])
    parser.add_argument("--readout-family", default="local", choices=["local", "all-pairs"])
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--execution-mode", default="sampler-sim", choices=["statevector", "sampler-sim", "ibm-hardware"])
    parser.add_argument("--backend-name")
    parser.add_argument(
        "--simulator-method",
        default="automatic",
        choices=["automatic", "statevector", "matrix_product_state"],
        help="Backend method for --execution-mode sampler-sim; matrix_product_state is preferable for larger 1D ladders",
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--readout-mitigation", action="store_true")
    parser.add_argument("--cal-shots", type=int, default=512)
    parser.add_argument("--extra-error-suppression", action="store_true")
    parser.add_argument("--dd-sequence", default="XY4")
    parser.add_argument("--twirl-randomizations", type=int, default=8)
    parser.add_argument("--tfidf-max-features", type=int, default=20000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-analyzer", default="word", choices=["word", "char", "char_wb"])
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qubits = parse_int_list(args.qubits)
    source = load_source(args)
    rows: list[dict[str, object]] = []
    stem = default_output_stem(source.source_name)
    json_out = args.json_out or f"qiskit_qos_scaling_{stem}.json"
    plot_out = args.plot_out or f"qiskit_qos_scaling_{stem}.png"

    for num_qubits in qubits:
        start = time.perf_counter()
        execution_config = build_execution_config(args)
        result = toy.run_classification_from_arrays(
            x=source.x,
            y=source.y,
            num_qubits=num_qubits,
            readout_shots=args.readout_shots,
            seed=args.seed,
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
                "num_qubits": int(num_qubits),
                "train_accuracy_quantum": float(result["train_accuracy_quantum"]),
                "test_accuracy_quantum": float(result["test_accuracy_quantum"]),
                "train_accuracy_classical": float(result["train_accuracy_classical"]),
                "test_accuracy_classical": float(result["test_accuracy_classical"]),
                "readout_feature_count": int(result["readout_feature_count"]),
                "query_feature_count": int(result["query_feature_count"]),
                "quantum_head_feature_count": int(result["quantum_head_feature_count"]),
                "n_train_used": int(len(result["train_labels"])),
                "n_test_used": int(len(result["test_labels"])),
                "quantum_threshold": float(result["quantum_threshold"]),
                "classical_threshold": float(result["classical_threshold"]),
                "signal_overlap_with_baseline": float(result["signal_overlap_with_baseline"]),
                "elapsed_seconds": float(elapsed),
                "execution_metadata": result["execution_metadata"],
            }
        )
        checkpoint_artifacts(
            args=args,
            source=source,
            qubits=qubits,
            rows=rows,
            json_out=json_out,
            plot_out=plot_out,
        )

    print("Qiskit QOS scaling runner")
    print(f"- source: {source.source_name}")
    print(f"- rows: {source.metadata['rows']}")
    if "raw_feature_dim" in source.metadata:
        print(f"- raw feature dim: {source.metadata['raw_feature_dim']}")
    if "reduced_feature_dim" in source.metadata:
        print(f"- reduced feature dim: {source.metadata['reduced_feature_dim']}")
    print(f"- execution mode: {args.execution_mode}")
    if args.execution_mode == "sampler-sim":
        print(f"- simulator method: {args.simulator_method}")
    print(f"- readout shots: {args.readout_shots}")
    print(f"- encoder: {args.encoder}")
    print(f"- quantum head: {args.quantum_head}")
    print(f"- readout family: {args.readout_family}")
    print("- sweep:")
    for row in rows:
        print(
            f"  q={row['num_qubits']:>2}  "
            f"quantum test={row['test_accuracy_quantum']:.3f}  "
            f"classical test={row['test_accuracy_classical']:.3f}  "
            f"features={row['readout_feature_count']}"
        )
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
