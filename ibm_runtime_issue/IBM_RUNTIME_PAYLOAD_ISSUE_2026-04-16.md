# IBM Runtime Payload Issue Handoff

## Summary

We now have a reproducible IBM Runtime payload boundary on `ibm_marrakesh`.

- small non-QOS Runtime submits work
- very small QOS submits also work
- slightly larger QOS submits are blocked by Cloudflare with `403 Forbidden`

This is not a generic backend/auth failure. It appears tied to the payload structure of the transpiled QOS circuits.

## Working Cases

1. Minimal Bell-state Runtime smoke
- backend: `ibm_marrakesh`
- artifact: `ibm_bell_smoke_ibm_marrakesh.json`
- result: success
- example job id: `d7g8opkj168s73f271c0`

2. Bounded QOS mini-sketch on PBMC68k
- route: `qiskit_qos_pbmc68k_pairwise_quantum_runner.py`
- settings:
  - `q=20`
  - `feature_mapping_limit=2`
  - `query_batch_size=1`
  - `submit_batch_size=4`
  - no DD/twirling
- artifact: `qiskit_qos_pbmc68k_pairwise_quantum_q20_hw_4x4_ibm_marrakesh_minisketch2_qb1.json`
- result: full end-to-end success
- observed behavior:
  - sketch submit with `2` measured circuits succeeds
  - repeated query submits with `2` measured circuits each also succeed

## Failing Cases

1. Full QOS sketch on PBMC68k
- route: `qiskit_qos_pbmc68k_pairwise_quantum_runner.py`
- settings:
  - `q=20`
  - default local readout family
  - full feature mapping count `98`
  - no DD/twirling in the cleanest failure probes
- behavior:
  - calibration submit succeeds
  - first real sketch submit fails at `SamplerV2.run(...)` with Cloudflare `403`

2. Same full sketch after IBM submit splitting
- still fails with:
  - `submit_batch_size=32`
  - `submit_batch_size=8`
  - `submit_batch_size=4`

3. Mini-sketch query payload with `4` measured circuits
- mini-sketch with `feature_mapping_limit=2` and `query_batch_size=2`
- sketch submit with `2` measured circuits succeeds
- next query submit with `4` measured circuits fails with Cloudflare `403`

## Current Best Boundary

Current evidence suggests:

- `2` measured circuits in this QOS pipeline: succeeds
- `4` measured circuits in this QOS pipeline: fails

This boundary held even when:

- DD/twirling were disabled
- submit batching was reduced to `4`
- backend/layout selection was stable and good

## Representative Evidence

- Bell smoke artifact: `ibm_bell_smoke_ibm_marrakesh.json`
- full-sketch failure log:
  - `/tmp/q20_runtime_debug_batched4_nosuppression.log`
- mini-sketch success/failure boundary log:
  - `/tmp/q20_runtime_debug_minisketch2.log`
- bounded successful mini-sketch artifact:
  - `qiskit_qos_pbmc68k_pairwise_quantum_q20_hw_4x4_ibm_marrakesh_minisketch2_qb1.json`

## Interpretation

The issue does not look like:

- generic IBM Runtime downtime
- invalid credentials
- a backend outage
- DD/twirling option incompatibility
- simple "too many circuits per submit" in the naive sense

The issue does look like:

- a Cloudflare/edge rejection tied to the structure or serialized size/shape of the QOS transpiled payload

## Recommended Escalation Ask

Ask IBM Runtime support / internal owners to inspect why these QOS `SamplerV2.run(...)` payloads are rejected at the edge while tiny Bell-state and tiny `2`-circuit QOS payloads are accepted.

Minimum request:

- confirm whether there is an undocumented request-size or payload-shape rule at the Cloudflare/edge layer
- confirm whether specific transpiled circuit payload structure can trigger the block even for very small circuit counts
- advise whether there is a supported way to package these submits so they bypass the edge rejection
