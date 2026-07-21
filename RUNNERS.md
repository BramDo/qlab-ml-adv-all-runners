# Runner Index

This file is a quick entrypoint map for the exported `ML_adv` runner surface.

## Core toy and scaling

- `qiskit_qos_toy_model.py`
  Main toy QOS implementation and IBM hardware entrypoint.
- `qiskit_qos_scaling_runner.py`
  Generic scaling runner for text and bounded hardware experiments.
- `qiskit_qos_text_runner.py`
  Text-oriented runner wrapper.
- `qiskit_qos_classical_benchmark.py`
  Matched classical baselines for quantum scaling artifacts.
- `qiskit_qos_memory_report.py`
  Post-process scaling summaries into memory accounting.
- `ibm_bell_smoke.py`
  Minimal IBM Runtime hardware smoke check.

## PBMC and genomics

- `qiskit_qos_hash_streaming_genomics_runner.py`
  Shared split, hashing, encoding, and evaluation layer.
- `qiskit_qos_pbmc68k_pairwise_quantum_runner.py`
  Main bounded PBMC68k pairwise quantum/classical runner.
- `qiskit_qos_pbmc68k_pairwise_screen.py`
  PBMC68k pairwise screen runner.
- `qiskit_qos_pbmc68k_pair_screen.py`
  PBMC pair screen.
- `qiskit_qos_pbmc10x_subcluster_screen.py`
  PBMC10x subcluster screen.
- `qiskit_qos_pbmc68k_q60_module_pipeline.py`
  Frozen 60-module PBMC68k data allocation, five-split MPS gate, and matched
  local/hardware analysis. Contains no provider calls.
- `qiskit_qos_pbmc68k_q60_module_fireopal_validate.py`
  Numeric QASM export and Fire Opal validate-only path for the 32/32 sentinel
  and gated 256/256 phase. Cannot submit hardware.
- `qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py`
  Separately confirmed sentinel/large submission and retrieval boundary with a
  declared 450-quantum-second full-study cap and no automatic resubmission.
- `qiskit_qos_gse132080_guide_screen.py`
  GSE132080 guide screen.
- `qiskit_qos_gse132080_semisynth_quantum_runner.py`
  GSE132080 semisynthetic quantum runner.
- `qiskit_qos_gse132080_semisynth_screen.py`
  GSE132080 semisynthetic screen.
- `qiskit_qos_gse132080_thirdorder_quantum_runner.py`
  GSE132080 third-order quantum runner.
- `qiskit_qos_gse132080_thirdorder_screen.py`
  GSE132080 third-order screen.
- `run_pbmc_pairwise_working_chain.sh`
  Stable launcher for the current bounded PBMC hardware chain.

## Text and categorical datasets

- `qiskit_qos_20ng_pair_runner.py`
  20 Newsgroups pair runner.
- `qiskit_qos_splice_kmer_runner.py`
  Splice k-mer quantum runner.
- `qiskit_qos_splice_classical_extended.py`
  Extended classical baselines for splice.
- `qiskit_qos_splice_memory_sweep.py`
  Splice memory sweep runner.
- `qiskit_qos_dorothea_chi2_quantum_runner.py`
  Dorothea chi-squared quantum runner.
- `qiskit_qos_dorothea_memory_sweep.py`
  Dorothea memory sweep runner.

## Frontier and probes

- `qiskit_qos_astronomical_runner.py`
  Astronomical-size synthetic source runner.
- `qiskit_qos_memory_frontier_runner.py`
  Memory frontier runner.
- `qiskit_qos_kmer_growth_probe.py`
  K-mer growth probe.

## Official QOS bridge and classifier-proof scripts

- `qiskit_official_qos_sampling_port.py`
  Qiskit-side sampling port for the official route.
- `qiskit_official_qos_20news_bridge.py`
- `qiskit_official_qos_20news_classifier_proof.py`
- `qiskit_official_qos_imdb_bridge.py`
- `qiskit_official_qos_imdb_classifier_proof.py`
- `qiskit_official_qos_pbmc_bridge.py`
- `qiskit_official_qos_pbmc_classifier_proof.py`
- `qiskit_official_qos_splice_bridge.py`
- `qiskit_official_qos_splice_classifier_proof.py`

## Official paper-route subtree

- `official_qos/benchmark.py`
- `official_qos/qos.py`
- `official_qos/qos_sampling.py`
- `official_qos/primitives.py`
- `official_qos/qsvt.py`
- `official_qos/data_generation.py`
- `official_qos/real_datasets/*.py`

Use the root `README.md` for environment and scope,
`Q40_WORKING_CHAIN_2026-04-16.md` for the stable bounded 40q hardware path, and
`Q60_MODULE_B4_RUNBOOK.md` for the leakage-safe 60q follow-up.
