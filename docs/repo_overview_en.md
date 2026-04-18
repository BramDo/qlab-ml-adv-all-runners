# Repository Overview

This repository is a clean export of the active `ML_adv` workspace.
It is not organized as a single installable package with one obvious entrypoint.
The real working surface is the set of runner scripts in the repository root, plus the `official_qos/` reference subtree.

## What Is In This Repository

The export keeps:

- the Qiskit toy and scaling pipeline
- dataset-specific quantum, classical, and screening runners
- the `official_qos/` paper-route JAX implementation
- IBM Runtime issue notes and hardware working-chain notes
- small source files needed to rerun representative examples

The export intentionally leaves out generated plots, cached datasets, local virtual environments, logs, and other temporary artifacts.

## Main Layers

### 1. `official_qos/`

This is the official paper-route subtree.
It contains the JAX implementation described in the QOS paper, including:

- `qos.py`
- `qos_sampling.py`
- `qsvt.py`
- `benchmark.py`
- `real_datasets/*.py`

Use this subtree when you want the reference implementation, the original benchmark path, or the real-dataset scripts that match the paper route.

### 2. Root `qiskit_qos_*.py` scripts

These are the active Qiskit-side runners, ports, and helpers for the broader `ML_adv` workspace.
This is the main place to look if you want to understand what is actually being run in this export.

The most important entrypoints are:

- `qiskit_qos_toy_model.py`
  Core toy QOS implementation with statevector, sampler simulator, and IBM hardware paths.
- `qiskit_qos_scaling_runner.py`
  Generic scaling wrapper for larger text and numeric sources.
- `qiskit_qos_hash_streaming_genomics_runner.py`
  Genomics-oriented hash-streaming path that avoids a dense encoder matrix.
- `qiskit_qos_pbmc68k_pairwise_quantum_runner.py`
  Main bounded PBMC68k quantum versus classical comparison runner.
- `qiskit_official_qos_sampling_port.py`
  Faithful Qiskit bridge for the smallest official QOS sampling kernels.
- `ibm_bell_smoke.py`
  Minimal IBM Runtime Bell-state smoke test.

### 3. IBM Runtime notes

The repository also keeps operational notes for IBM Runtime and hardware-specific behavior:

- `ibm_runtime_issue/`
  Prepared Markdown material for the bounded `q=40` `SamplerV2.run()` payload issue.
- `Q40_WORKING_CHAIN_2026-04-16.md`
  The currently working bounded `q=40` hardware chain and its exact environment settings.
- `run_pbmc_pairwise_working_chain.sh`
  Stable launcher for the current PBMC hardware path.

## How The Repository Is Best Read

If you are new to the repo, use this order:

1. `README.md` for scope and environment.
2. `RUNNERS.md` for the practical runner index.
3. The runner that matches your dataset or experiment type.
4. `official_qos/` when you need the paper-route reference implementation.

That order matters because this export is broader than the narrow `official_qos/` subtree.
If you start inside `official_qos/`, you can miss the Qiskit-side runner layer that drives many of the active experiments in this checkout.

## Practical Map By Use Case

### Toy or surrogate QOS experiments

Start with `qiskit_qos_toy_model.py`.
This file is intentionally modest and pedagogical: it keeps only a compact streaming sketch state and turns that into a small Qiskit circuit.

### Scaling experiments on text or numeric data

Start with `qiskit_qos_scaling_runner.py`.
It wraps the toy model instead of changing it directly, so larger-source experiments stay separate from the base sketch implementation.

### Genomics and PBMC experiments

Start with `qiskit_qos_hash_streaming_genomics_runner.py` and the PBMC runners.
These files implement the hash-streaming route used to avoid the dense `num_qubits x feature_dim` encoder matrix.

### Official QOS parity and bridge work

Start with `qiskit_official_qos_sampling_port.py`.
Its job is to establish kernel-level parity with the official sampling implementation before attaching the port to a full dataset pipeline.

### IBM hardware checks

Start with `ibm_bell_smoke.py` for a minimal sanity check, then move to `run_pbmc_pairwise_working_chain.sh` and `Q40_WORKING_CHAIN_2026-04-16.md` for the bounded PBMC hardware route.

## How Scripts Are Usually Run

The active workflow uses the qlab Qiskit virtual-environment helper:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh python <script>.py ...
```

Representative examples:

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

## Bottom Line

Treat this repository as an exported experiment workspace with a few stable entrypoints, not as a conventional library.

If you only want the shortest path:

- `README.md` explains scope
- `RUNNERS.md` shows where to start
- `qiskit_qos_toy_model.py` is the core Qiskit sketch implementation
- `official_qos/` is the paper-reference subtree
- `Q40_WORKING_CHAIN_2026-04-16.md` is the current hardware runbook
