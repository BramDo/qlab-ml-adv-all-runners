# Appendix: q=40 Runtime Payload Failure Evidence

## Workspace

- `/mnt/c/Users/Lenna/SynologyDrive/qlab/ML_adv`

## Main drafts

- `IBM_RUNTIME_GITHUB_ISSUE_q40_sampler403.md`
- `IBM_RUNTIME_SUPPORT_TICKET_q40_sampler403.md`

## Failing bounded q=40 runs

### `ibm_marrakesh`

- structured run log:
  - `/tmp/q40_runtime_minisketch2_qb1_8x8.jsonl`
- runtime debug log:
  - `/tmp/q40_runtime_minisketch2_qb1_8x8.debug.log`
- successful calibration job ID:
  - `d7gb9vkj168s73f29pag`
- failing Cloudflare Ray ID:
  - `9ed26368c834167e`

### `ibm_fez`

- structured run log:
  - `/tmp/q40_runtime_minisketch2_qb1_8x8_ibm_fez.jsonl`
- runtime debug log:
  - `/tmp/q40_runtime_minisketch2_qb1_8x8_ibm_fez.debug.log`
- successful calibration job ID:
  - `d7gbg5ea0v2s738al52g`
- failing Cloudflare Ray ID:
  - `9ed27695d9b8d592`

## Historical reference artifact

- old q=40 hardware artifact that completed on `ibm_fez`:
  - `qiskit_qos_pbmc68k_pairwise_quantum_q40_hw_4x4_ibm_fez.json`

## Working comparison cases

- bounded q=20 hardware route that completes:
  - `qiskit_qos_pbmc68k_pairwise_quantum_q20_hw_8x8_ibm_marrakesh_minisketch2_qb1.json`
- minimal Bell smoke that completes:
  - `ibm_bell_smoke_ibm_marrakesh.json`

## Current bounded route settings

- `feature_mapping_limit = 2`
- `query_batch_size = 1`
- `runtime_submit_batch_size = 4`
- `readout_mitigation = true`
- `readout_shots = 64`
- `cal_shots = 2048`

## Observed q=40 failure point

- calibration succeeds
- the first real sketch submit fails
- failing sketch submit characteristics:
  - `2` measured circuits
  - transpiled `max_depth ~396`
  - transpiled `max_size ~632`
