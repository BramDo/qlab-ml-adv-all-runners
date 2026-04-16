# IBM Runtime `SamplerV2.run()` returns Cloudflare `403 Forbidden` for valid QOS sketch payloads after successful calibration

We are seeing a reproducible IBM Runtime failure where `SamplerV2.run()` is blocked by Cloudflare with `403 Forbidden` for valid QOS sketch payloads.

## What works

- minimal Bell-state hardware submission
- bounded QOS route at `q=20`
- `q=40` readout calibration on both `ibm_marrakesh` and `ibm_fez`

## What fails

- bounded `q=40`
- first real sketch submission after successful calibration
- failure occurs on both `ibm_marrakesh` and `ibm_fez`

The failing submit is already small in circuit count:

- `2` measured circuits
- `submit_batch_count = 1`
- `submit_batch_size = 4`

The sketch payload is deeper/larger than calibration:

- `max_depth ~396`
- `max_size ~632`

This suggests the failure is tied to payload structure or transpiled circuit complexity, not basic auth, backend availability, or calibration.

## Observed sequence

1. runner starts normally
2. dataset load / split / encoding complete
3. calibration job submits and completes successfully
4. first sketch submit fails at `sampler.run(...)` with `IBMRuntimeError` wrapping Cloudflare `403 Forbidden`

## Known failing Ray IDs

- `ibm_marrakesh`: `9ed26368c834167e`
- `ibm_fez`: `9ed27695d9b8d592`

## Question

- Is there an undocumented Runtime edge rule or payload-shape constraint that can block valid `SamplerV2` submissions like this?

We can provide structured logs, traceback, job IDs, and payload metadata if needed.
