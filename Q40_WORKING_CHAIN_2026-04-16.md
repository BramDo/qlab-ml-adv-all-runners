# Q40 Working Chain

Date: 2026-04-16

This note captures the currently working `q=40` hardware route for the PBMC68k pairwise benchmark.

## Working route

- backend: `ibm_fez`
- qubits: `40`
- split: `8/8`
- readout shots: `32`
- readout mitigation: enabled
- extra error suppression: enabled
- DD sequence: `XY4`
- twirling randomizations: `8`
- layout strategy: `none`
- runtime submit batch size: `1`
- feature mapping limit: `2`

## Why this matters

The newer `quality-chain` payload route repeatedly failed on the first real sketch submit.

The legacy-style route below completed end-to-end:
- disable the quality-chain layout
- submit one transpiled circuit per Runtime job chunk
- keep the bounded `feature_mapping_limit=2`

## Environment knobs

```bash
QISKIT_QOS_LAYOUT_STRATEGY=none
QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE=1
QISKIT_QOS_FEATURE_MAPPING_LIMIT=2
QISKIT_QOS_DEBUG_RUNTIME=1
```

## Command that worked

```bash
QISKIT_QOS_LAYOUT_STRATEGY=none \
QISKIT_QOS_FEATURE_MAPPING_LIMIT=2 \
QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE=1 \
QISKIT_QOS_RUN_LOG=/tmp/q40_legacy_none_layout_8x8_probe.jsonl \
QISKIT_QOS_DEBUG_RUNTIME=1 \
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
python qiskit_qos_pbmc68k_pairwise_quantum_runner.py \
  --cache-dir /home/bram/.cache/qiskit_qos/pbmc68k \
  --qubits 40 \
  --execution-mode ibm-hardware \
  --backend-name ibm_fez \
  --max-train-samples 8 \
  --max-test-samples 8 \
  --readout-shots 32 \
  --query-batch-size 2 \
  --readout-mitigation \
  --cal-shots 512 \
  --extra-error-suppression \
  --dd-sequence XY4 \
  --twirl-randomizations 8 \
  --json-out qiskit_qos_pbmc68k_pairwise_quantum_q40_hw_8x8_ibm_fez_legacylayout_probe.json \
  --plot-out qiskit_qos_pbmc68k_pairwise_quantum_q40_hw_8x8_ibm_fez_legacylayout_probe.png
```

## Result

Artifact:
- `qiskit_qos_pbmc68k_pairwise_quantum_q40_hw_8x8_ibm_fez_legacylayout_probe.json`

Key bounded result:
- quantum test accuracy: `0.75`
- quantum balanced accuracy: `0.75`
- hashed ridge test accuracy: `0.375`
- hashed LinearSVC test accuracy: `0.375`
- ambient dense classical weight: `3.99 GB`
- quantum sketch state: `632 B`
- avoided dense encoder matrix: `159.70 GB`

## Practical rule

If `q=40` starts failing again on the current hardware path, first retry this exact chain before changing shots, mitigation, or model details.
