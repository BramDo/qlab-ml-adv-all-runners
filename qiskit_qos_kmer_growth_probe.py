#!/usr/bin/env python3
"""Streaming k-mer growth probe for large genomics corpora.

This runner is intentionally separate from the quantum benchmark scripts. Its
goal is to answer one question first:

- does the observed k-mer vocabulary keep growing fast enough that a classical
  dense model might become very large?

It supports:
- exact counting for small smoke tests
- HyperLogLog estimation for larger local pilots
- `splice-openml` as a built-in real source
- local FASTA/FASTQ(.gz) files or globs
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

import qiskit_qos_splice_kmer_utils as splice_utils


def human_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


class HyperLogLog:
    def __init__(self, *, precision: int = 14) -> None:
        if precision < 4 or precision > 20:
            raise ValueError("precision must be between 4 and 20")
        self.precision = int(precision)
        self.m = 1 << self.precision
        self.registers = np.zeros(self.m, dtype=np.uint8)

    @staticmethod
    def _hash_u64(value: str) -> int:
        digest = hashlib.blake2b(value.encode("ascii"), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    @staticmethod
    def _rho(value: int, *, width: int) -> int:
        if value == 0:
            return width + 1
        return width - value.bit_length() + 1

    def add(self, value: str) -> None:
        hashed = self._hash_u64(value)
        idx = hashed >> (64 - self.precision)
        width = 64 - self.precision
        remainder_mask = (1 << width) - 1
        remainder = hashed & remainder_mask
        rank = self._rho(remainder, width=width)
        if rank > int(self.registers[idx]):
            self.registers[idx] = rank

    def estimate(self) -> float:
        m = float(self.m)
        registers = self.registers.astype(np.float64)
        indicator = np.sum(np.power(2.0, -registers))
        if self.m == 16:
            alpha = 0.673
        elif self.m == 32:
            alpha = 0.697
        elif self.m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / m)
        estimate = alpha * m * m / indicator

        zeros = int(np.sum(self.registers == 0))
        if estimate <= 2.5 * m and zeros > 0:
            estimate = m * math.log(m / zeros)
        elif estimate > (1.0 / 30.0) * (1 << 32):
            estimate = -(1 << 32) * math.log(1.0 - estimate / (1 << 32))
        return float(estimate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming k-mer growth probe for large corpora.")
    parser.add_argument("--source", default="splice-openml", choices=["splice-openml", "files"])
    parser.add_argument("--input-glob", help="Glob for FASTA/FASTQ(.gz) files when --source files")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--binary", action="store_true", default=True, help="For splice-openml keep EI vs IE only")
    parser.add_argument("--mode", default="hll", choices=["hll", "exact", "both"])
    parser.add_argument("--hll-precision", type=int, default=14)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--chunk-bases", type=int, default=0, help="If >0, split long sequences into fixed-size chunks before counting")
    parser.add_argument("--limit-sequences", type=int)
    parser.add_argument("--min-sequence-length", type=int, default=0)
    parser.add_argument("--json-out")
    parser.add_argument("--plot-out")
    return parser.parse_args()


def iter_chunked_sequences(sequences: Iterable[str], *, chunk_bases: int) -> Iterator[str]:
    if chunk_bases <= 0:
        yield from sequences
        return
    for sequence in sequences:
        if len(sequence) <= chunk_bases:
            yield sequence
            continue
        for start in range(0, len(sequence), chunk_bases):
            chunk = sequence[start : start + chunk_bases]
            if chunk:
                yield chunk


def iter_source_sequences(args: argparse.Namespace) -> tuple[Iterator[str], dict[str, object]]:
    if args.source == "splice-openml":
        sequences, labels, metadata = splice_utils.load_splice_sequences(binary=args.binary)
        del labels
        return iter_chunked_sequences(iter(sequences), chunk_bases=args.chunk_bases), {
            "source": "splice-openml",
            **metadata,
        }

    if not args.input_glob:
        raise ValueError("--input-glob is required when --source files")
    matched = sorted(glob.glob(args.input_glob))
    if not matched:
        raise ValueError(f"No files matched glob: {args.input_glob}")

    def iterator() -> Iterator[str]:
        for raw_path in matched:
            path = Path(raw_path)
            yield from splice_utils.iter_sequences_from_file(path)

    return iter_chunked_sequences(iterator(), chunk_bases=args.chunk_bases), {
        "source": "files",
        "input_glob": args.input_glob,
        "matched_files": matched,
        "matched_file_count": len(matched),
    }


def shard_rows(
    sequences: Iterable[str],
    *,
    k: int,
    shard_size: int,
    limit_sequences: int | None,
    min_sequence_length: int,
    mode: str,
    hll_precision: int,
) -> list[dict[str, object]]:
    hll = HyperLogLog(precision=hll_precision) if mode in {"hll", "both"} else None
    exact_hashes: set[int] | None = set() if mode in {"exact", "both"} else None
    rows: list[dict[str, object]] = []
    total_sequences = 0
    total_bases = 0
    total_kmer_events = 0
    shard_kmer_events = 0
    shard_sequences = 0

    for raw_sequence in sequences:
        sequence = splice_utils.clean_sequence(raw_sequence)
        if len(sequence) < max(k, min_sequence_length):
            continue
        total_sequences += 1
        shard_sequences += 1
        total_bases += len(sequence)
        kmers_this_seq = 0
        for kmer in splice_utils.iter_kmers(sequence, k=k):
            kmers_this_seq += 1
            if hll is not None:
                hll.add(kmer)
            if exact_hashes is not None:
                exact_hashes.add(HyperLogLog._hash_u64(kmer))
        total_kmer_events += kmers_this_seq
        shard_kmer_events += kmers_this_seq

        if shard_sequences >= shard_size:
            rows.append(
                snapshot_row(
                    total_sequences=total_sequences,
                    total_bases=total_bases,
                    total_kmer_events=total_kmer_events,
                    shard_kmer_events=shard_kmer_events,
                    exact_hashes=exact_hashes,
                    hll=hll,
                )
            )
            shard_sequences = 0
            shard_kmer_events = 0

        if limit_sequences is not None and total_sequences >= limit_sequences:
            break

    if shard_sequences > 0:
        rows.append(
            snapshot_row(
                total_sequences=total_sequences,
                total_bases=total_bases,
                total_kmer_events=total_kmer_events,
                shard_kmer_events=shard_kmer_events,
                exact_hashes=exact_hashes,
                hll=hll,
            )
        )
    return rows


def snapshot_row(
    *,
    total_sequences: int,
    total_bases: int,
    total_kmer_events: int,
    shard_kmer_events: int,
    exact_hashes: set[int] | None,
    hll: HyperLogLog | None,
) -> dict[str, object]:
    exact_unique = len(exact_hashes) if exact_hashes is not None else None
    hll_estimate = hll.estimate() if hll is not None else None
    return {
        "sequences_seen": int(total_sequences),
        "bases_seen": int(total_bases),
        "kmer_events_seen": int(total_kmer_events),
        "kmer_events_last_shard": int(shard_kmer_events),
        "exact_unique_kmers": int(exact_unique) if exact_unique is not None else None,
        "hll_estimated_unique_kmers": float(hll_estimate) if hll_estimate is not None else None,
        "exact_dense_weight_bytes": int(exact_unique * 8) if exact_unique is not None else None,
        "hll_dense_weight_bytes_estimate": float(hll_estimate * 8.0) if hll_estimate is not None else None,
    }


def add_growth_deltas(rows: list[dict[str, object]], *, mode: str) -> None:
    prev_exact = None
    prev_hll = None
    prev_sequences = 0
    for row in rows:
        sequences_seen = int(row["sequences_seen"])
        delta_sequences = sequences_seen - prev_sequences
        row["delta_sequences"] = int(delta_sequences)
        prev_sequences = sequences_seen

        if mode in {"exact", "both"}:
            exact = row["exact_unique_kmers"]
            row["delta_exact_unique_kmers"] = None if exact is None or prev_exact is None else int(exact - prev_exact)
            prev_exact = exact if exact is not None else prev_exact
        if mode in {"hll", "both"}:
            estimate = row["hll_estimated_unique_kmers"]
            row["delta_hll_unique_kmers_estimate"] = None if estimate is None or prev_hll is None else float(estimate - prev_hll)
            prev_hll = estimate if estimate is not None else prev_hll


def render_plot(rows: list[dict[str, object]], *, output_path: str, mode: str) -> None:
    x = [int(row["sequences_seen"]) for row in rows]
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    if mode in {"exact", "both"}:
        ax_top.plot(x, [row["exact_unique_kmers"] for row in rows], marker="o", label="exact unique kmers")
    if mode in {"hll", "both"}:
        ax_top.plot(x, [row["hll_estimated_unique_kmers"] for row in rows], marker="s", label="HLL estimate")
    ax_top.set_ylabel("Unique k-mers")
    ax_top.grid(alpha=0.25)
    ax_top.legend()

    if mode in {"exact", "both"}:
        ax_bottom.plot(x, [None if row["exact_dense_weight_bytes"] is None else row["exact_dense_weight_bytes"] / (1024 ** 3) for row in rows], marker="o", label="exact dense GB")
    if mode in {"hll", "both"}:
        ax_bottom.plot(x, [None if row["hll_dense_weight_bytes_estimate"] is None else row["hll_dense_weight_bytes_estimate"] / (1024 ** 3) for row in rows], marker="s", label="HLL dense GB est")
    ax_bottom.set_ylabel("Dense weight memory (GB)")
    ax_bottom.set_xlabel("Sequences seen")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def final_summary(rows: list[dict[str, object]], *, k: int) -> dict[str, object]:
    if not rows:
        return {
            "rows_recorded": 0,
            "ambient_feature_dim": splice_utils.ambient_kmer_dim(k),
            "ambient_dense_weight_bytes": splice_utils.ambient_dense_weight_bytes(k),
        }
    last = rows[-1]
    summary = {
        "rows_recorded": len(rows),
        "ambient_feature_dim": splice_utils.ambient_kmer_dim(k),
        "ambient_dense_weight_bytes": splice_utils.ambient_dense_weight_bytes(k),
        "ambient_dense_weight_human": human_bytes(splice_utils.ambient_dense_weight_bytes(k)),
        "sequences_seen": last["sequences_seen"],
        "bases_seen": last["bases_seen"],
        "kmer_events_seen": last["kmer_events_seen"],
    }
    if last.get("exact_unique_kmers") is not None:
        summary["exact_unique_kmers"] = int(last["exact_unique_kmers"])
        summary["exact_dense_weight_bytes"] = int(last["exact_dense_weight_bytes"])
        summary["exact_dense_weight_human"] = human_bytes(int(last["exact_dense_weight_bytes"]))
    if last.get("hll_estimated_unique_kmers") is not None:
        summary["hll_estimated_unique_kmers"] = float(last["hll_estimated_unique_kmers"])
        summary["hll_dense_weight_bytes_estimate"] = float(last["hll_dense_weight_bytes_estimate"])
        summary["hll_dense_weight_human_estimate"] = human_bytes(float(last["hll_dense_weight_bytes_estimate"]))
    return summary


def main() -> None:
    args = parse_args()
    sequences, source_meta = iter_source_sequences(args)
    rows = shard_rows(
        sequences,
        k=args.k,
        shard_size=args.shard_size,
        limit_sequences=args.limit_sequences,
        min_sequence_length=args.min_sequence_length,
        mode=args.mode,
        hll_precision=args.hll_precision,
    )
    add_growth_deltas(rows, mode=args.mode)

    payload = {
        "config": {
            "source": args.source,
            "input_glob": args.input_glob,
            "k": int(args.k),
            "binary": bool(args.binary),
            "mode": args.mode,
            "hll_precision": int(args.hll_precision),
            "shard_size": int(args.shard_size),
            "chunk_bases": int(args.chunk_bases),
            "limit_sequences": int(args.limit_sequences) if args.limit_sequences is not None else None,
            "min_sequence_length": int(args.min_sequence_length),
        },
        "source": source_meta,
        "summary": final_summary(rows, k=args.k),
        "rows": rows,
        "notes": [
            "ambient_dense_weight_bytes is the full canonical 4^k dense k-mer model size",
            "exact counts are only practical for smaller pilots",
            "HLL estimates are intended for large-corpus growth screening on limited hardware",
        ],
    }

    json_out = args.json_out or f"qiskit_qos_kmer_growth_probe_k{args.k}.json"
    plot_out = args.plot_out or f"qiskit_qos_kmer_growth_probe_k{args.k}.png"
    Path(json_out).write_text(json.dumps(payload, indent=2))
    render_plot(rows, output_path=plot_out, mode=args.mode)

    print("Streaming k-mer growth probe")
    print(f"- source: {args.source}")
    print(f"- k: {args.k}")
    print(f"- ambient dense weight memory: {human_bytes(splice_utils.ambient_dense_weight_bytes(args.k))}")
    if payload["summary"].get("exact_dense_weight_human"):
        print(f"- exact observed dense weight memory: {payload['summary']['exact_dense_weight_human']}")
    if payload["summary"].get("hll_dense_weight_human_estimate"):
        print(f"- HLL observed dense weight memory estimate: {payload['summary']['hll_dense_weight_human_estimate']}")
    print(f"Saved summary to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()
