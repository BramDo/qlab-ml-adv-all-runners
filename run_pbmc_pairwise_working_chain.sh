#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 6 ]]; then
  echo "usage: $0 [qubits] [seed] [hash_seed] [train_samples] [test_samples] [backend]" >&2
  exit 2
fi

QUBITS="${1:-40}"
SEED="${2:-7}"
HASH_SEED="${3:-$SEED}"
TRAIN_SAMPLES="${4:-8}"
TEST_SAMPLES="${5:-8}"
BACKEND="${6:-ibm_fez}"
READOUT_SHOTS="${READOUT_SHOTS:-32}"
CAL_SHOTS="${CAL_SHOTS:-512}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-2}"
DD_SEQUENCE="${DD_SEQUENCE:-XY4}"
TWIRL_RANDOMIZATIONS="${TWIRL_RANDOMIZATIONS:-8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEM="qiskit_qos_pbmc68k_pairwise_quantum_q${QUBITS}_hw_${TRAIN_SAMPLES}x${TEST_SAMPLES}_${BACKEND}_legacylayout_seed${SEED}"
RUN_LOG="/tmp/${STEM}.jsonl"

export QISKIT_QOS_LAYOUT_STRATEGY="${QISKIT_QOS_LAYOUT_STRATEGY:-none}"
export QISKIT_QOS_FEATURE_MAPPING_LIMIT="${QISKIT_QOS_FEATURE_MAPPING_LIMIT:-2}"
export QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE="${QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE:-1}"
export QISKIT_QOS_RUN_LOG="${QISKIT_QOS_RUN_LOG:-$RUN_LOG}"
export QISKIT_QOS_DEBUG_RUNTIME="${QISKIT_QOS_DEBUG_RUNTIME:-1}"

exec ../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_pbmc68k_pairwise_quantum_runner.py \
  --cache-dir /home/bram/.cache/qiskit_qos/pbmc68k \
  --qubits "$QUBITS" \
  --seed "$SEED" \
  --hash-seed "$HASH_SEED" \
  --execution-mode ibm-hardware \
  --backend-name "$BACKEND" \
  --max-train-samples "$TRAIN_SAMPLES" \
  --max-test-samples "$TEST_SAMPLES" \
  --readout-shots "$READOUT_SHOTS" \
  --query-batch-size "$QUERY_BATCH_SIZE" \
  --readout-mitigation \
  --cal-shots "$CAL_SHOTS" \
  --extra-error-suppression \
  --dd-sequence "$DD_SEQUENCE" \
  --twirl-randomizations "$TWIRL_RANDOMIZATIONS" \
  --json-out "${STEM}.json" \
  --plot-out "${STEM}.png"
