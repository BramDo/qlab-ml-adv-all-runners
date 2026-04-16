#!/usr/bin/env python3
"""Classical benchmark runner for the QOS scaling sources.

This is a separate extension so we can compare stronger and memory-bounded
classical baselines against the existing quantum scaling artifacts without
rewriting the toy or scaling runners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import qiskit_qos_scaling_runner as scaling
import qiskit_qos_text_runner as text_runner
import qiskit_qos_toy_model as toy


FLOAT64_BYTES = 8


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark stronger classical baselines on the QOS scaling datasets.")
    parser.add_argument(
        "--source",
        default="20ng-atheism-vs-space",
        choices=[
            "20ng-atheism-vs-space",
            "20ng-graphics-vs-baseball",
            "20ng-custom",
        ],
        help="Built-in text source to benchmark.",
    )
    parser.add_argument("--20ng-categories", dest="twenty_ng_categories", help="Comma-separated custom 20NG categories when --source 20ng-custom")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=32)
    parser.add_argument("--tfidf-max-features", type=int, default=20000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-analyzer", default="word", choices=["word", "char", "char_wb"])
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=4096)
    parser.add_argument("--quantum-scaling-json", help="Optional scaling artifact to compare against.")
    parser.add_argument("--quantum-qubits", type=int, default=14, help="Which quantum q to compare if --quantum-scaling-json is set.")
    parser.add_argument("--json-out")
    return parser.parse_args()


def _load_texts(args: argparse.Namespace) -> tuple[pd.Series, np.ndarray]:
    categories = scaling.resolve_20ng_categories(args.source, args.twenty_ng_categories)
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    texts = text_runner.clean_text(pd.Series(dataset.data), strip_literals=[])
    labels = np.where(np.asarray(dataset.target, dtype=np.int64) == 1, 1, -1)
    return texts, labels


def _benchmark_indices(
    x_dense: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = len(x_dense)
    n_train = int(round(train_fraction * total))
    n_train = max(2, min(int(n_train), total - 1))
    train_idx, test_idx = toy.split_train_test(x_dense, n_train=n_train, rng=rng)
    if max_train_samples is not None and len(train_idx) > max_train_samples:
        keep_train = rng.choice(len(train_idx), size=max_train_samples, replace=False)
        keep_train.sort()
        train_idx = train_idx[keep_train]
    if max_test_samples is not None and len(test_idx) > max_test_samples:
        keep_test = rng.choice(len(test_idx), size=max_test_samples, replace=False)
        keep_test.sort()
        test_idx = test_idx[keep_test]
    return train_idx, test_idx


def _linear_model_bytes(feature_dim: int) -> int:
    return (feature_dim + 1) * FLOAT64_BYTES


def _nb_model_bytes(feature_dim: int, n_classes: int = 2) -> int:
    return (feature_dim * n_classes + n_classes) * FLOAT64_BYTES


def _record_result(
    results: list[dict[str, Any]],
    *,
    name: str,
    family: str,
    feature_space: str,
    feature_dim: int,
    accuracy: float,
    model_bytes: int,
) -> None:
    results.append(
        {
            "name": name,
            "family": family,
            "feature_space": feature_space,
            "feature_dim": int(feature_dim),
            "test_accuracy": float(accuracy),
            "model_bytes": int(model_bytes),
            "model_bytes_human": human_bytes(model_bytes),
        }
    )


def _load_quantum_comparison(path: str, *, q: int) -> dict[str, Any] | None:
    payload = json.loads(Path(path).read_text())
    for run in payload.get("runs", []):
        if int(run["num_qubits"]) == q:
            return {
                "num_qubits": int(run["num_qubits"]),
                "test_accuracy_quantum": float(run["test_accuracy_quantum"]),
                "test_accuracy_classical_baseline_from_scaling": float(run["test_accuracy_classical"]),
                "readout_feature_count": int(run["readout_feature_count"]),
                "quantum_head_feature_count": int(run["quantum_head_feature_count"]),
            }
    return None


def main() -> None:
    args = parse_args()

    source_args = argparse.Namespace(
        source=args.source,
        twenty_ng_categories=args.twenty_ng_categories,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_analyzer=args.tfidf_analyzer,
        svd_components=args.svd_components,
        seed=args.seed,
    )
    source = scaling.load_source(source_args)
    texts, labels = _load_texts(args)
    if len(texts) != len(source.x) or not np.array_equal(labels.astype(float), source.y):
        raise RuntimeError("Text labels/features mismatch; source loading is inconsistent.")

    train_idx, test_idx = _benchmark_indices(
        source.x,
        seed=args.seed,
        train_fraction=args.train_fraction,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    x_train_dense = np.asarray(source.x[train_idx], dtype=np.float64)
    x_test_dense = np.asarray(source.x[test_idx], dtype=np.float64)
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    texts_train = texts.iloc[train_idx]
    texts_test = texts.iloc[test_idx]

    results: list[dict[str, Any]] = []

    # Match the current scaling baseline exactly.
    x_train_std, x_test_std = toy.standardize(x_train_dense, x_test_dense)
    w = toy.ridge_linear_classifier(x_train_std, y_train.astype(np.float64))
    train_scores = x_train_std @ w
    test_scores = x_test_std @ w
    threshold = 0.5 * (
        float(np.mean(train_scores[y_train > 0])) + float(np.mean(train_scores[y_train < 0]))
    )
    pred = np.where(test_scores >= threshold, 1, -1)
    _record_result(
        results,
        name="ridge_manual_svd256",
        family="ridge",
        feature_space="svd256",
        feature_dim=x_train_dense.shape[1],
        accuracy=float(np.mean(pred == y_test)),
        model_bytes=_linear_model_bytes(x_train_dense.shape[1]),
    )

    ridge_clf = RidgeClassifier(alpha=1.0)
    ridge_clf.fit(x_train_dense, y_train)
    _record_result(
        results,
        name="ridgeclassifier_svd256",
        family="ridge",
        feature_space="svd256",
        feature_dim=x_train_dense.shape[1],
        accuracy=float(np.mean(ridge_clf.predict(x_test_dense) == y_test)),
        model_bytes=_linear_model_bytes(x_train_dense.shape[1]),
    )

    logreg_dense = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=args.seed))
    logreg_dense.fit(x_train_dense, y_train)
    _record_result(
        results,
        name="logreg_svd256",
        family="logreg",
        feature_space="svd256",
        feature_dim=x_train_dense.shape[1],
        accuracy=float(np.mean(logreg_dense.predict(x_test_dense) == y_test)),
        model_bytes=_linear_model_bytes(x_train_dense.shape[1]),
    )

    svc_dense = make_pipeline(StandardScaler(), LinearSVC(random_state=args.seed, dual=False, max_iter=10000))
    svc_dense.fit(x_train_dense, y_train)
    _record_result(
        results,
        name="linearsvc_svd256",
        family="svm",
        feature_space="svd256",
        feature_dim=x_train_dense.shape[1],
        accuracy=float(np.mean(svc_dense.predict(x_test_dense) == y_test)),
        model_bytes=_linear_model_bytes(x_train_dense.shape[1]),
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
        analyzer=args.tfidf_analyzer,
    )
    x_train_tfidf = vectorizer.fit_transform(texts_train)
    x_test_tfidf = vectorizer.transform(texts_test)
    raw_vocab_size = int(x_train_tfidf.shape[1])

    logreg_raw = LogisticRegression(max_iter=5000, random_state=args.seed)
    logreg_raw.fit(x_train_tfidf, y_train)
    _record_result(
        results,
        name="logreg_raw_tfidf",
        family="logreg",
        feature_space="raw_tfidf",
        feature_dim=raw_vocab_size,
        accuracy=float(np.mean(logreg_raw.predict(x_test_tfidf) == y_test)),
        model_bytes=_linear_model_bytes(raw_vocab_size),
    )

    svc_raw = LinearSVC(random_state=args.seed)
    svc_raw.fit(x_train_tfidf, y_train)
    _record_result(
        results,
        name="linearsvc_raw_tfidf",
        family="svm",
        feature_space="raw_tfidf",
        feature_dim=raw_vocab_size,
        accuracy=float(np.mean(svc_raw.predict(x_test_tfidf) == y_test)),
        model_bytes=_linear_model_bytes(raw_vocab_size),
    )

    nb_raw = MultinomialNB()
    nb_raw.fit(x_train_tfidf, y_train)
    _record_result(
        results,
        name="multinb_raw_tfidf",
        family="naive_bayes",
        feature_space="raw_tfidf",
        feature_dim=raw_vocab_size,
        accuracy=float(np.mean(nb_raw.predict(x_test_tfidf) == y_test)),
        model_bytes=_nb_model_bytes(raw_vocab_size),
    )

    hashing = HashingVectorizer(
        lowercase=True,
        n_features=args.hash_features,
        alternate_sign=False,
        norm="l2",
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
        analyzer=args.tfidf_analyzer,
    )
    x_train_hash = hashing.transform(texts_train)
    x_test_hash = hashing.transform(texts_test)

    sgd_hash = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=args.seed, max_iter=5000, tol=1e-4)
    sgd_hash.fit(x_train_hash, y_train)
    _record_result(
        results,
        name=f"sgd_log_hashing_{args.hash_features}",
        family="sgd",
        feature_space="hashing",
        feature_dim=args.hash_features,
        accuracy=float(np.mean(sgd_hash.predict(x_test_hash) == y_test)),
        model_bytes=_linear_model_bytes(args.hash_features),
    )

    results.sort(key=lambda row: (-float(row["test_accuracy"]), int(row["model_bytes"])))
    payload: dict[str, Any] = {
        "config": {
            "source": args.source,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "tfidf_max_features": args.tfidf_max_features,
            "tfidf_min_df": args.tfidf_min_df,
            "tfidf_ngram_min": args.tfidf_ngram_min,
            "tfidf_ngram_max": args.tfidf_ngram_max,
            "tfidf_analyzer": args.tfidf_analyzer,
            "svd_components": args.svd_components,
            "hash_features": args.hash_features,
            "twenty_ng_categories": args.twenty_ng_categories,
        },
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
        "source": {
            "source_name": source.source_name,
            "raw_feature_dim": int(source.metadata["raw_feature_dim"]),
            "reduced_feature_dim": int(source.metadata["reduced_feature_dim"]),
        },
        "results": results,
        "notes": [
            "This artifact is a classical-side extension and does not replace the earlier QOS toy/scaling runs.",
            "The raw TF-IDF baselines preserve far more lexical signal than the dense SVD(256) representation.",
            "The hashing baseline is intended as a memory-bounded classical reference, not as the strongest possible classifier.",
        ],
    }
    if args.quantum_scaling_json:
        payload["quantum_reference"] = _load_quantum_comparison(args.quantum_scaling_json, q=args.quantum_qubits)

    json_out = args.json_out or f"qiskit_qos_classical_benchmark_{args.source.replace('-', '_')}_{args.max_train_samples}x{args.max_test_samples}.json"
    Path(json_out).write_text(json.dumps(payload, indent=2))

    print("QOS classical benchmark")
    print(f"- source: {args.source}")
    print(f"- train/test: {len(train_idx)}/{len(test_idx)}")
    print("- results:")
    for row in results:
        print(
            "  "
            f"{row['name']:<26} "
            f"acc={row['test_accuracy']:.3f} "
            f"mem={row['model_bytes_human']}"
        )
    if payload.get("quantum_reference") is not None:
        qref = payload["quantum_reference"]
        print(
            f"- quantum reference q={qref['num_qubits']}: "
            f"quantum_acc={qref['test_accuracy_quantum']:.3f} "
            f"old_classical_baseline={qref['test_accuracy_classical_baseline_from_scaling']:.3f}"
        )
    print(f"Saved summary to: {json_out}")


if __name__ == "__main__":
    main()
