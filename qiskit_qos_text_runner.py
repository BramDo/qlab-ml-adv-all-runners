#!/usr/bin/env python3
"""Run the QOS-inspired Qiskit toy model on labeled or weakly labeled text data.

This wrapper converts text to compact dense features via TF-IDF + TruncatedSVD,
then reuses the numeric streaming-sketch classifier from qiskit_qos_toy_model.

For the sarcasm workbook in this folder's workflow:
- the sheet has tweet text but no direct binary sarcasm/non-sarcasm label column
- so a weak proxy task can be defined, e.g. presence of '#not'
- direct label tokens can optionally be stripped from the text to avoid leakage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from qiskit.quantum_info import Statevector

import qiskit_qos_toy_model as toy


def parse_strip_literals(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_label_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_class_groups(value: str | None) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    if value is None or not value.strip():
        return groups
    for chunk in value.split(";"):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid class group spec '{part}', expected name=label1|label2")
        name, labels_part = part.split("=", 1)
        name = name.strip()
        labels = [item.strip() for item in labels_part.split("|") if item.strip()]
        if not name or not labels:
            raise ValueError(f"invalid class group spec '{part}'")
        groups[name] = labels
    return groups


def load_table(input_path: str, sheet_name: str | None) -> pd.DataFrame:
    suffix = Path(input_path).suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(input_path, sheet_name=sheet_name or 0)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(input_path, sep=sep)
    raise ValueError(f"unsupported input suffix: {suffix}")


def build_labels_from_column(series: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
    labels, mapping = toy.map_binary_labels(series)
    return labels, {str(k): float(v) for k, v in mapping.items()}


def build_labels_from_groups(
    series: pd.Series,
    *,
    positive_labels: list[str],
    negative_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw = series.astype(str).str.strip()
    lowered = raw.str.lower()
    pos_set = {item.lower() for item in positive_labels}
    neg_set = {item.lower() for item in negative_labels}
    if not pos_set or not neg_set:
        raise ValueError("both positive_labels and negative_labels must be non-empty")

    keep_mask = lowered.isin(pos_set | neg_set).to_numpy(dtype=bool)
    kept = lowered[keep_mask]
    labels = np.where(kept.isin(pos_set).to_numpy(dtype=bool), 1.0, -1.0)
    mapping = {item: 1.0 for item in positive_labels}
    mapping.update({item: -1.0 for item in negative_labels})
    return labels, keep_mask, mapping


def build_multiclass_from_groups(
    series: pd.Series,
    *,
    class_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]], list[str]]:
    raw = series.astype(str).str.strip()
    lowered = raw.str.lower()
    label_to_group: dict[str, str] = {}
    for class_name, labels in class_groups.items():
        for label in labels:
            key = label.lower()
            if key in label_to_group:
                raise ValueError(f"label '{label}' appears in more than one class group")
            label_to_group[key] = class_name

    keep_mask = lowered.isin(label_to_group.keys()).to_numpy(dtype=bool)
    mapped = lowered[keep_mask].map(label_to_group).to_numpy(dtype=object)
    return mapped, keep_mask, class_groups, list(class_groups.keys())


def build_multiclass_from_labels(
    series: pd.Series,
    *,
    class_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]], list[str]]:
    groups = {label: [label] for label in class_labels}
    return build_multiclass_from_groups(series, class_groups=groups)


def build_labels_from_pattern(texts: pd.Series, pattern: str) -> tuple[np.ndarray, dict[str, float]]:
    positive = texts.str.contains(pattern, case=False, regex=False, na=False).to_numpy(dtype=bool)
    labels = np.where(positive, 1.0, -1.0)
    mapping = {f"contains:{pattern}": 1.0, f"not_contains:{pattern}": -1.0}
    return labels, mapping


def clean_text(texts: pd.Series, *, strip_literals: list[str]) -> pd.Series:
    cleaned = texts.astype(str)
    cleaned = cleaned.str.replace(r"@USER\d+", " USER ", regex=True)
    cleaned = cleaned.str.replace(r"http\S+", " URL ", regex=True)
    for literal in strip_literals:
        cleaned = cleaned.str.replace(literal, " ", case=False, regex=False)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def balanced_subset_indices(labels_pm1: np.ndarray, rng: np.random.Generator, max_per_class: int | None) -> np.ndarray:
    pos_idx = np.flatnonzero(labels_pm1 > 0)
    neg_idx = np.flatnonzero(labels_pm1 < 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("need both positive and negative samples for classification")

    target = min(len(pos_idx), len(neg_idx))
    if max_per_class is not None:
        target = min(target, max_per_class)

    pos_pick = rng.choice(pos_idx, size=target, replace=False) if len(pos_idx) > target else pos_idx
    neg_pick = rng.choice(neg_idx, size=target, replace=False) if len(neg_idx) > target else neg_idx
    idx = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(idx)
    return idx


def balanced_subset_indices_multiclass(labels: np.ndarray, rng: np.random.Generator, max_per_class: int | None) -> np.ndarray:
    classes = list(dict.fromkeys(labels.tolist()))
    counts = [int(np.sum(labels == cls)) for cls in classes]
    target = min(counts)
    if max_per_class is not None:
        target = min(target, max_per_class)

    picks: list[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        if len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
        picks.append(idx)
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def tfidf_svd_features(
    texts: pd.Series,
    *,
    max_features: int,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    analyzer: str,
    svd_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("require 1 <= ngram_min <= ngram_max")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=min_df,
        ngram_range=(ngram_min, ngram_max),
        analyzer=analyzer,
    )
    x_sparse = vectorizer.fit_transform(texts)
    vocab_size = int(x_sparse.shape[1])
    if vocab_size < 2:
        raise ValueError("text vocabulary too small after TF-IDF filtering")

    svd_k = min(svd_components, vocab_size - 1)
    svd = TruncatedSVD(n_components=svd_k, random_state=seed)
    x_dense = svd.fit_transform(x_sparse)
    return x_dense, {
        "tfidf_shape": [int(x_sparse.shape[0]), int(x_sparse.shape[1])],
        "tfidf_analyzer": analyzer,
        "tfidf_ngram_range": [int(ngram_min), int(ngram_max)],
        "svd_components": int(svd_k),
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "vocab_size": vocab_size,
        "sample_terms": vectorizer.get_feature_names_out()[:20].tolist(),
    }


def stratified_train_test_indices(labels: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in list(dict.fromkeys(labels.tolist())):
        idx = np.flatnonzero(labels == cls)
        if len(idx) < 2:
            raise ValueError(f"class '{cls}' has fewer than 2 samples")
        idx = idx.copy()
        rng.shuffle(idx)
        n_train = int(round(train_fraction * len(idx)))
        n_train = max(1, min(n_train, len(idx) - 1))
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def fit_binary_head(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    y_train_pm1: np.ndarray,
    y_test_pm1: np.ndarray,
    readout_shots: int | None,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    num_qubits = encoded_train.shape[1]
    signed_train = y_train_pm1[:, None] * encoded_train
    cumulative_signed_mean = np.cumsum(signed_train, axis=0) / np.arange(1, len(signed_train) + 1)[:, None]
    pos_mean_encoded = encoded_train[y_train_pm1 > 0].mean(axis=0)
    neg_mean_encoded = encoded_train[y_train_pm1 < 0].mean(axis=0)
    signed_gap_encoded = pos_mean_encoded - neg_mean_encoded
    if num_qubits > 1:
        train_pair = encoded_train[:, :-1] * encoded_train[:, 1:]
        pos_mean_pair = train_pair[y_train_pm1 > 0].mean(axis=0)
        neg_mean_pair = train_pair[y_train_pm1 < 0].mean(axis=0)
        signed_gap_pair = pos_mean_pair - neg_mean_pair
    else:
        pos_mean_pair = np.array([], dtype=np.float64)
        neg_mean_pair = np.array([], dtype=np.float64)
        signed_gap_pair = np.array([], dtype=np.float64)

    sketch = toy.WeightedStreamingSketch(num_qubits=num_qubits)
    for encoded_sample, label in zip(encoded_train, y_train_pm1, strict=True):
        sketch.update(encoded_sample, float(label))

    sketch_state = Statevector.from_instruction(sketch.build_circuit())
    model_features = toy.local_pauli_shadow_surrogate(sketch_state, shots=readout_shots, rng=rng)

    q_scores_train = np.array(
        [
            toy.feature_score(
                model_features,
                encoded_sample,
                shots=readout_shots,
                rng=rng,
                single_scale=sketch.single_scale,
                phase_scale=sketch.phase_scale,
                pair_scale=sketch.pair_scale,
            )
            for encoded_sample in encoded_train
        ]
    )
    q_scores_test = np.array(
        [
            toy.feature_score(
                model_features,
                encoded_sample,
                shots=readout_shots,
                rng=rng,
                single_scale=sketch.single_scale,
                phase_scale=sketch.phase_scale,
                pair_scale=sketch.pair_scale,
            )
            for encoded_sample in encoded_test
        ]
    )

    raw_w = toy.ridge_linear_classifier(x_train, y_train_pm1)
    c_scores_train = x_train @ raw_w
    c_scores_test = x_test @ raw_w

    if toy.pearson_corr(q_scores_train, y_train_pm1) < 0.0:
        q_scores_train *= -1.0
        q_scores_test *= -1.0
        model_features = -model_features
        signed_gap_encoded = -signed_gap_encoded
        signed_gap_pair = -signed_gap_pair
        cumulative_signed_mean = -cumulative_signed_mean

    q_pos_mean = float(np.mean(q_scores_train[y_train_pm1 > 0]))
    q_neg_mean = float(np.mean(q_scores_train[y_train_pm1 < 0]))
    c_pos_mean = float(np.mean(c_scores_train[y_train_pm1 > 0]))
    c_neg_mean = float(np.mean(c_scores_train[y_train_pm1 < 0]))

    return {
        "train_scores_quantum": q_scores_train,
        "test_scores_quantum": q_scores_test,
        "train_scores_classical": c_scores_train,
        "test_scores_classical": c_scores_test,
        "signed_gap_encoded": signed_gap_encoded,
        "signed_gap_pair": signed_gap_pair,
        "cumulative_signed_mean": cumulative_signed_mean,
        "pos_mean_encoded": pos_mean_encoded,
        "neg_mean_encoded": neg_mean_encoded,
        "pos_mean_pair": pos_mean_pair,
        "neg_mean_pair": neg_mean_pair,
        "readout_feature_count": int(len(model_features)),
        "q_pos_mean": q_pos_mean,
        "q_neg_mean": q_neg_mean,
        "c_pos_mean": c_pos_mean,
        "c_neg_mean": c_neg_mean,
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> list[list[int]]:
    mapping = {cls: i for i, cls in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=True):
        counts[mapping[str(truth)], mapping[str(pred)]] += 1
    return counts.tolist()


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in classes:
        mask = y_true == cls
        out[cls] = float(np.mean(y_pred[mask] == y_true[mask])) if np.any(mask) else 0.0
    return out


def plot_multiclass_summary(
    out_path: Path,
    *,
    classes: list[str],
    quantum_confusion: list[list[int]],
    classical_confusion: list[list[int]],
    quantum_per_class_acc: dict[str, float],
    classical_per_class_acc: dict[str, float],
    signed_gap_matrix: np.ndarray,
    raw_feature_count: int,
    num_qubits: int,
    readout_feature_count: int,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    q_mat = np.asarray(quantum_confusion, dtype=float)
    c_mat = np.asarray(classical_confusion, dtype=float)

    for ax, mat, title in [
        (axes[0, 0], q_mat, "Quantum Confusion"),
        (axes[0, 1], c_mat, "Classical Confusion"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    x = np.arange(len(classes))
    width = 0.36
    axes[1, 0].bar(x - width / 2, [quantum_per_class_acc[c] for c in classes], width=width, label="quantum")
    axes[1, 0].bar(x + width / 2, [classical_per_class_acc[c] for c in classes], width=width, label="classical")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha="right")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_title("Per-Class Test Accuracy")
    axes[1, 0].legend()

    im2 = axes[1, 1].imshow(signed_gap_matrix, cmap="coolwarm", aspect="auto")
    axes[1, 1].set_yticks(range(len(classes)))
    axes[1, 1].set_yticklabels(classes)
    axes[1, 1].set_xticks(range(num_qubits))
    axes[1, 1].set_xticklabels([f"q{i}" for i in range(num_qubits)])
    axes[1, 1].set_title(
        f"Label-Weighted Gap Heatmap\nraw={raw_feature_count}, qubits={num_qubits}, readout={readout_feature_count}"
    )
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_multiclass_label_diagnostics(
    out_path: Path,
    *,
    classes: list[str],
    signed_gap_matrix: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4 + 0.45 * len(classes)))
    im = ax.imshow(signed_gap_matrix, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(range(signed_gap_matrix.shape[1]))
    ax.set_xticklabels([f"q{i}" for i in range(signed_gap_matrix.shape[1])])
    ax.set_title("One-vs-Rest Label-Weighted Gap per Class")
    ax.set_xlabel("qubit/block")
    ax.set_ylabel("class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_multiclass_from_arrays(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    num_qubits: int,
    readout_shots: int | None,
    seed: int,
    train_fraction: float,
) -> dict[str, object]:
    train_idx, test_idx = stratified_train_test_indices(labels, train_fraction, seed)
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    x_train, x_test = toy.standardize(x_train_raw, x_test_raw)
    rng = np.random.default_rng(seed)
    encoder = toy.ToyEncoding.fit(x_train, num_qubits=num_qubits, rng=rng)
    encoded_train = encoder.encode(x_train)
    encoded_test = encoder.encode(x_test)

    q_train_cols: list[np.ndarray] = []
    q_test_cols: list[np.ndarray] = []
    c_train_cols: list[np.ndarray] = []
    c_test_cols: list[np.ndarray] = []
    signed_gaps: list[np.ndarray] = []
    class_details: dict[str, dict[str, object]] = {}

    for i, class_name in enumerate(class_names):
        y_train_pm1 = np.where(y_train == class_name, 1.0, -1.0)
        y_test_pm1 = np.where(y_test == class_name, 1.0, -1.0)
        head = fit_binary_head(
            x_train=x_train,
            x_test=x_test,
            encoded_train=encoded_train,
            encoded_test=encoded_test,
            y_train_pm1=y_train_pm1,
            y_test_pm1=y_test_pm1,
            readout_shots=readout_shots,
            seed=seed + 1009 * (i + 1),
        )
        q_mid = 0.5 * (head["q_pos_mean"] + head["q_neg_mean"])
        q_scale = max(0.5 * (head["q_pos_mean"] - head["q_neg_mean"]), 1e-6)
        c_mid = 0.5 * (head["c_pos_mean"] + head["c_neg_mean"])
        c_scale = max(0.5 * (head["c_pos_mean"] - head["c_neg_mean"]), 1e-6)

        q_train_cols.append((head["train_scores_quantum"] - q_mid) / q_scale)
        q_test_cols.append((head["test_scores_quantum"] - q_mid) / q_scale)
        c_train_cols.append((head["train_scores_classical"] - c_mid) / c_scale)
        c_test_cols.append((head["test_scores_classical"] - c_mid) / c_scale)
        signed_gaps.append(head["signed_gap_encoded"])
        class_details[class_name] = {
            "signed_gap_encoded": head["signed_gap_encoded"].tolist(),
            "signed_gap_pair": head["signed_gap_pair"].tolist(),
            "quantum_score_mid": q_mid,
            "quantum_score_scale": q_scale,
            "classical_score_mid": c_mid,
            "classical_score_scale": c_scale,
        }

    q_train = np.stack(q_train_cols, axis=1)
    q_test = np.stack(q_test_cols, axis=1)
    c_train = np.stack(c_train_cols, axis=1)
    c_test = np.stack(c_test_cols, axis=1)

    q_pred_train = np.asarray(class_names, dtype=object)[np.argmax(q_train, axis=1)]
    q_pred_test = np.asarray(class_names, dtype=object)[np.argmax(q_test, axis=1)]
    c_pred_train = np.asarray(class_names, dtype=object)[np.argmax(c_train, axis=1)]
    c_pred_test = np.asarray(class_names, dtype=object)[np.argmax(c_test, axis=1)]

    return {
        "train_labels": y_train,
        "test_labels": y_test,
        "quantum_pred_train": q_pred_train,
        "quantum_pred_test": q_pred_test,
        "classical_pred_train": c_pred_train,
        "classical_pred_test": c_pred_test,
        "train_accuracy_quantum": float(np.mean(q_pred_train == y_train)),
        "test_accuracy_quantum": float(np.mean(q_pred_test == y_test)),
        "train_accuracy_classical": float(np.mean(c_pred_train == y_train)),
        "test_accuracy_classical": float(np.mean(c_pred_test == y_test)),
        "quantum_confusion_test": confusion_counts(y_test, q_pred_test, class_names),
        "classical_confusion_test": confusion_counts(y_test, c_pred_test, class_names),
        "quantum_per_class_accuracy": per_class_accuracy(y_test, q_pred_test, class_names),
        "classical_per_class_accuracy": per_class_accuracy(y_test, c_pred_test, class_names),
        "signed_gap_matrix": np.stack(signed_gaps, axis=0),
        "class_details": class_details,
        "readout_feature_count": int(3 * num_qubits + 2 * max(num_qubits - 1, 0)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text wrapper for the Qiskit QOS toy classifier")
    parser.add_argument("--input", required=True, help="Excel or CSV file")
    parser.add_argument("--sheet", help="Sheet name for Excel input")
    parser.add_argument("--text-col", required=True, help="Text column name")
    parser.add_argument("--label-col", help="Label column name")
    parser.add_argument(
        "--class-labels",
        help="Comma-separated labels to keep as separate classes for multiclass runs",
    )
    parser.add_argument(
        "--class-groups",
        help="Grouped multiclass spec, e.g. pos=positief;neg=licht negatief|zwaar negatief;neutral=neutraal|informatie",
    )
    parser.add_argument(
        "--positive-labels",
        help="Comma-separated label values mapped to +1 when using --label-col",
    )
    parser.add_argument(
        "--negative-labels",
        help="Comma-separated label values mapped to -1 when using --label-col",
    )
    parser.add_argument("--label-pattern", help="Fixed substring used as a weak positive label when no label column exists")
    parser.add_argument(
        "--strip-literals",
        default="",
        help="Comma-separated literals removed from the text before vectorization, e.g. '#sarcasme,#not'",
    )
    parser.add_argument("--balance-classes", action="store_true", help="Subsample to equal class counts")
    parser.add_argument("--max-per-class", type=int, help="Optional per-class cap after balancing")
    parser.add_argument("--tfidf-max-features", type=int, default=3000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-ngram-min", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--tfidf-analyzer", default="word", choices=["word", "char", "char_wb"])
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument("--csv-train-fraction", type=float, default=0.67)
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--readout-shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-prefix", default="qiskit_qos_text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.label_col and not args.label_pattern:
        raise ValueError("provide either --label-col or --label-pattern")

    df = load_table(args.input, args.sheet)
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found")
    valid_text_mask = df[args.text_col].notna().to_numpy(dtype=bool)
    texts = df.loc[valid_text_mask, args.text_col].astype(str).reset_index(drop=True)
    class_labels = parse_label_list(args.class_labels)
    class_groups = parse_class_groups(args.class_groups)
    multiclass_mode = bool(class_labels) or bool(class_groups)

    if args.label_col:
        if args.label_col not in df.columns:
            raise ValueError(f"label column '{args.label_col}' not found")
        raw_labels = df.loc[valid_text_mask, args.label_col].reset_index(drop=True)
        if multiclass_mode:
            if class_groups:
                labels, keep_mask, label_mapping, class_names = build_multiclass_from_groups(
                    raw_labels,
                    class_groups=class_groups,
                )
                label_source = f"class_groups:{args.label_col}"
            else:
                labels, keep_mask, label_mapping, class_names = build_multiclass_from_labels(
                    raw_labels,
                    class_labels=class_labels,
                )
                label_source = f"class_labels:{args.label_col}"
            texts = texts.iloc[keep_mask].reset_index(drop=True)
        else:
            positive_labels = parse_label_list(args.positive_labels)
            negative_labels = parse_label_list(args.negative_labels)
            if positive_labels or negative_labels:
                labels, keep_mask, label_mapping = build_labels_from_groups(
                    raw_labels,
                    positive_labels=positive_labels,
                    negative_labels=negative_labels,
                )
                texts = texts.iloc[keep_mask].reset_index(drop=True)
                label_source = f"column_groups:{args.label_col}"
            else:
                labels, label_mapping = build_labels_from_column(raw_labels)
                label_source = f"column:{args.label_col}"
            class_names = None
    else:
        if multiclass_mode:
            raise ValueError("multiclass options require --label-col")
        labels, label_mapping = build_labels_from_pattern(texts, args.label_pattern)
        label_source = f"pattern:{args.label_pattern}"
        class_names = None

    strip_literals = parse_strip_literals(args.strip_literals)
    texts_clean = clean_text(texts, strip_literals=strip_literals)

    rng = np.random.default_rng(args.seed)
    if args.balance_classes:
        keep_idx = (
            balanced_subset_indices_multiclass(labels, rng, args.max_per_class)
            if multiclass_mode
            else balanced_subset_indices(labels, rng, args.max_per_class)
        )
        texts_clean = texts_clean.iloc[keep_idx].reset_index(drop=True)
        labels = labels[keep_idx]

    x_dense, feature_meta = tfidf_svd_features(
        texts_clean,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        ngram_min=args.tfidf_ngram_min,
        ngram_max=args.tfidf_ngram_max,
        analyzer=args.tfidf_analyzer,
        svd_components=args.svd_components,
        seed=args.seed,
    )

    readout_shots = args.readout_shots if args.readout_shots > 0 else None
    if multiclass_mode and class_names is not None:
        classification = run_multiclass_from_arrays(
            x=x_dense,
            labels=np.asarray(labels, dtype=object),
            class_names=class_names,
            num_qubits=args.num_qubits,
            readout_shots=readout_shots,
            seed=args.seed,
            train_fraction=args.csv_train_fraction,
        )
    else:
        classification = toy.run_classification_from_arrays(
            x=x_dense,
            y=labels,
            num_qubits=args.num_qubits,
            readout_shots=readout_shots,
            seed=args.seed,
            train_fraction=args.csv_train_fraction,
        )

    plot_prefix = Path(args.plot_prefix)
    summary_path = plot_prefix.with_name(plot_prefix.name + "_summary.json")
    summary_plot = plot_prefix.with_name(plot_prefix.name + "_summary.png")
    label_plot = plot_prefix.with_name(plot_prefix.name + "_label_weight.png")

    summary = {
        "input": {
            "path": args.input,
            "sheet": args.sheet,
            "text_col": args.text_col,
            "label_source": label_source,
            "label_mapping": label_mapping,
            "strip_literals": strip_literals,
            "rows_used": int(len(texts_clean)),
            "balance_classes": bool(args.balance_classes),
            "max_per_class": args.max_per_class,
        },
        "features": feature_meta,
        "model": {
            "num_qubits": args.num_qubits,
            "readout_shots": 0 if readout_shots is None else int(readout_shots),
            "readout_feature_count": classification["readout_feature_count"],
            "train_rows": classification["n_train"],
            "test_rows": classification["n_test"],
        },
        "classification": (
            {
                "mode": "multiclass",
                "classes": class_names,
                "train_accuracy_quantum": classification["train_accuracy_quantum"],
                "test_accuracy_quantum": classification["test_accuracy_quantum"],
                "train_accuracy_classical": classification["train_accuracy_classical"],
                "test_accuracy_classical": classification["test_accuracy_classical"],
                "quantum_confusion_test": classification["quantum_confusion_test"],
                "classical_confusion_test": classification["classical_confusion_test"],
                "quantum_per_class_accuracy": classification["quantum_per_class_accuracy"],
                "classical_per_class_accuracy": classification["classical_per_class_accuracy"],
                "class_details": classification["class_details"],
            }
            if multiclass_mode and class_names is not None
            else {
                "mode": "binary",
                "train_accuracy_quantum": classification["train_accuracy_quantum"],
                "test_accuracy_quantum": classification["test_accuracy_quantum"],
                "train_accuracy_classical": classification["train_accuracy_classical"],
                "test_accuracy_classical": classification["test_accuracy_classical"],
                "quantum_threshold": classification["quantum_threshold"],
                "classical_threshold": classification["classical_threshold"],
                "test_score_corr_quantum_vs_classical": classification["signal_overlap_with_baseline"],
                "signed_gap_encoded": classification["signed_gap_encoded"].tolist(),
            }
        ),
        "notes": [
            "If label_source is pattern-based, this is a weakly supervised proxy task rather than a ground-truth sarcasm benchmark.",
            "Text is converted to TF-IDF and then compressed with TruncatedSVD before the Qiskit sketch.",
            "Multiclass mode uses one-vs-rest quantum sketches on a shared train/test split.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if multiclass_mode and class_names is not None:
        plot_multiclass_summary(
            summary_plot,
            classes=class_names,
            quantum_confusion=classification["quantum_confusion_test"],
            classical_confusion=classification["classical_confusion_test"],
            quantum_per_class_acc=classification["quantum_per_class_accuracy"],
            classical_per_class_acc=classification["classical_per_class_accuracy"],
            signed_gap_matrix=classification["signed_gap_matrix"],
            raw_feature_count=x_dense.shape[1],
            num_qubits=args.num_qubits,
            readout_feature_count=classification["readout_feature_count"],
        )
        plot_multiclass_label_diagnostics(
            label_plot,
            classes=class_names,
            signed_gap_matrix=classification["signed_gap_matrix"],
        )
    else:
        toy.plot_summary(
            summary_plot,
            cls_scores_q=classification["test_scores_quantum"],
            cls_scores_c=classification["test_scores_classical"],
            cls_labels=classification["test_labels"],
            pca_scores_q=None,
            pca_scores_c=None,
            raw_feature_count=x_dense.shape[1],
            num_qubits=args.num_qubits,
            readout_feature_count=classification["readout_feature_count"],
        )
        toy.plot_label_weight_diagnostics(
            label_plot,
            pos_mean_encoded=classification["pos_mean_encoded"],
            neg_mean_encoded=classification["neg_mean_encoded"],
            signed_gap_encoded=classification["signed_gap_encoded"],
            pos_mean_pair=classification["pos_mean_pair"],
            neg_mean_pair=classification["neg_mean_pair"],
            signed_gap_pair=classification["signed_gap_pair"],
            cumulative_signed_mean=classification["cumulative_signed_mean"],
            quantum_scores=classification["test_scores_quantum"],
            labels=classification["test_labels"],
        )

    print("Qiskit QOS text runner")
    print(f"- input: {args.input}")
    print(f"- text column: {args.text_col}")
    print(f"- label source: {label_source}")
    print(f"- rows used: {len(texts_clean)}")
    print(f"- tfidf shape: {tuple(feature_meta['tfidf_shape'])}")
    print(f"- dense shape: {x_dense.shape}")
    print(f"- svd explained variance: {feature_meta['svd_explained_variance_ratio_sum']:.3f}")
    print(f"- quantum sketch size: {args.num_qubits} qubits")
    print()
    print("Classification")
    print(f"  quantum toy  train acc: {classification['train_accuracy_quantum']:.3f}")
    print(f"  quantum toy   test acc: {classification['test_accuracy_quantum']:.3f}")
    print(f"  classical    train acc: {classification['train_accuracy_classical']:.3f}")
    print(f"  classical     test acc: {classification['test_accuracy_classical']:.3f}")
    if multiclass_mode and class_names is not None:
        print(f"  classes: {class_names}")
        print(f"  quantum per-class acc: {classification['quantum_per_class_accuracy']}")
        print(f"  classical per-class acc: {classification['classical_per_class_accuracy']}")
    else:
        print(f"  score corr(q-toy, classical): {classification['signal_overlap_with_baseline']:.3f}")
        print(f"  signed per-qubit gap: {[round(v, 3) for v in classification['signed_gap_encoded'].tolist()]}")
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved plot to: {summary_plot}")
    print(f"Saved label-weight diagnostics to: {label_plot}")


if __name__ == "__main__":
    main()
