# QLab ML_adv All Runners

This repository is a clean export of the active `ML_adv` workspace with the full runner/code surface preserved.

What is included:
- the Qiskit toy and scaling pipeline
- dataset-specific quantum/classical runners and screens
- the `official_qos` paper-route subtree
- IBM Runtime issue notes and hardware working-chain notes
- small source files needed to rerun examples such as `breast_cancer_wisconsin.csv`

What is intentionally excluded:
- generated JSON and PNG artifacts
- cached datasets
- local venvs, nested `.git` metadata, and `__pycache__`
- temporary probes, logs, and progress files

## Layout

- `qiskit_qos_toy_model.py`
  Main toy QOS implementation with statevector, sampler simulator, and IBM hardware paths.
- `qiskit_qos_scaling_runner.py`
  Generic scaling runner used across text and bounded hardware experiments.
- `qiskit_qos_pbmc68k_pairwise_quantum_runner.py`
  Main PBMC68k bounded hardware/classical comparison runner.
- `qiskit_qos_hash_streaming_genomics_runner.py`
  Shared genomics-style hashing, split, and evaluation layer.
- `qiskit_official_qos_*.py`
  Qiskit ports and classifier-proof runners aligned to the paper route.
- `official_qos/`
  Paper-route subtree with core modules and `real_datasets` scripts.
- `ibm_runtime_issue/`
  Markdown package for the IBM Runtime payload issue.
- `Q40_WORKING_CHAIN_2026-04-16.md`
  The currently working bounded `q=40` IBM hardware chain.
- `docs/repo_overview_en.md`
  English orientation guide for this exported workspace.
- `OFFICIAL_QOS_REPRO_STATUS_2026-04-15.md`
  Status notebook for the reproduction track.
- `RUNNERS.md`
  Quick index of the runner and screen entrypoints in this export.

## Running

The active workflow in this project uses the qlab Qiskit venv helper:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh python <script>.py ...
```

Typical examples:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py --plot
```

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_scaling_runner.py \
    --source 20ng-atheism-vs-space \
    --qubits 20 \
    --max-train-samples 8 \
    --max-test-samples 8
```

```bash
./run_pbmc_pairwise_working_chain.sh 40 11 11 16 16 ibm_fez
```

## Current hardware note

The currently working bounded `q=40` hardware chain uses:
- `ibm_fez`
- `QISKIT_QOS_LAYOUT_STRATEGY=none`
- `QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE=1`
- `QISKIT_QOS_FEATURE_MAPPING_LIMIT=2`

The launcher for that path is:
- `run_pbmc_pairwise_working_chain.sh`

The detailed note is:
- `Q40_WORKING_CHAIN_2026-04-16.md`

## Paper-route scope

This repository is broader than the narrow `official_qos` export. It keeps:
- the paper-route subtree itself under `official_qos/`
- the Qiskit bridge scripts and classifier-proof scripts in the repo root
- the broader `ML_adv` runner family used for bounded hardware and scaling experiments

## Regenerating artifacts

This export does not track generated result artifacts. Recreate them locally from the included runners and notes.
