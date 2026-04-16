# Reproducible IBM Runtime `SamplerV2.run()` failure: Cloudflare `403 Forbidden` on valid `q=40` sketch submissions after successful calibration

We are reporting a reproducible IBM Runtime issue affecting `SamplerV2.run()`.

In our workflow, a bounded QOS hardware route works at smaller settings, and `q=40` readout calibration also succeeds. However, the first real `q=40` sketch submission is blocked by Cloudflare with `403 Forbidden`.

This occurs on multiple backends:

- `ibm_marrakesh`
- `ibm_fez`

This does not appear to be a general authentication or backend-availability issue, because:

- a minimal Bell-state hardware submission succeeds
- bounded `q=20` QOS runs succeed end-to-end
- `q=40` readout calibration succeeds on both backends before the failure

## Reproduction summary

- Runtime primitive: `SamplerV2`
- execution mode: IBM hardware
- readout mitigation enabled
- bounded route settings:
  - `feature_mapping_limit = 2`
  - `query_batch_size = 1`
  - `runtime_submit_batch_size = 4`

## Failing condition

- the first real sketch submission after calibration
- payload contains only `2` measured circuits
- transpiled payload metadata is approximately:
  - `max_depth ~396`
  - `max_size ~632`

## Observed behavior

- calibration submit succeeds
- calibration result is returned successfully
- immediate next `sampler.run(...)` for the sketch payload fails with:
  - `IBMRuntimeError`
  - underlying `403 Client Error: Forbidden`
  - Cloudflare block page

## Relevant identifiers

- `ibm_marrakesh` failing Ray ID: `9ed26368c834167e`
- `ibm_fez` calibration job ID: `d7gbg5ea0v2s738al52g`
- `ibm_fez` failing Ray ID: `9ed27695d9b8d592`

## Request

Could you please confirm whether there is any undocumented edge-layer constraint related to request size, payload structure, or transpiled circuit complexity that could cause valid `SamplerV2` submissions to be blocked by Cloudflare, even when calibration on the same backend succeeds?

We can provide:

- structured JSONL event logs
- traceback
- exact runtime sequence
- transpiled payload metadata for the failing submission
