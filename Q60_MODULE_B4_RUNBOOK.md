# PBMC68k 60q module-B4 runbook

This route keeps the successful q40 seed-11 task and Fire Opal protocol, while
replacing the pair-hash input by 60 label-free coexpression modules. It is a
task-bound QML experiment, not a formal quantum-advantage proof.

## Frozen design

- PBMC68k: `CD4+/CD25 T Reg` versus `CD4+/CD45RO+ Memory`.
- Seed-11 32/32 rows remain the sentinel.
- A separate 512-cell pool learns 60 modules from 1,200 variable genes.
- Five mutually disjoint 256/256 development splits precede a separate 256/256
  final split.
- Four module statistics become four 60-value upload blocks. Median/IQR
  scaling is fitted on each training partition only.
- The q60 circuit uses a 6x10 grid, `B=4`, multiplier `sqrt(60)`, logical depth
  20, 134 two-qubit gates, 627 observables, X/Y/Z measurements, and 128 shots.

## Known runtime

Use the established WSL Qiskit environment:

```bash
cd /mnt/c/Users/Lenna/SynologyDrive/qlab/ML_adv
export PYTHONPATH=.
PY=/home/bram/.venvs/qiskit/bin/python
```

Freeze modules and every index before evaluating models:

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py prepare
```

Run the MPS ladder and five development splits locally. This is expected to be
the expensive classical step:

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py local-screen
```

If all MPS probes converge, ideal q60 must beat both training-only-selected
linear and RBF baselines on at least four splits. If chi=256 does not agree
with chi=512 within `1e-3`, the large phase may proceed only as a hardware
feasibility experiment. If the MPS probes converge but the 4/5 performance
gate fails, the large phase is blocked.

## Sentinel outcome — 21 July 2026

The frozen sentinel completed on `ibm_fez` through Fire Opal as action
`2335848`:

- 192 circuits at 128 shots, covering 32 training and 32 held-out test cells;
- hardware balanced accuracy `0.53125` (`17/32`);
- linear baseline `0.50000` (`16/32`);
- RBF baseline `0.43750` (`14/32`);
- quantum training-only CV mean `0.59375`;
- exact McNemar `p=1.0` versus the stronger linear baseline;
- paired-bootstrap 95% interval `[-0.1875, 0.25]`;
- Fire Opal dashboard quantum time: 26 seconds;
- submission-to-retrieval interval: approximately 8 minutes 33 seconds.

The archived `get_result` payload omitted the quantum-seconds field, so the
26-second measurement is attributed to the Fire Opal dashboard. The local MPS
attempt was stopped after 42 minutes 57 seconds after completing the `chi=64`
probe and one of eight `chi=128` samples; it did not establish convergence.

This is a task-specific local time-to-feature-generation advantage under the
declared resources. The 26-second quantum task is more than 99.1x faster than
the incomplete 2,577-second MPS attempt; the complete 513-second route through
retrieval is still more than 5.0x faster. The hardware point score also exceeds
both predeclared baselines. MPS did not converge, so the timing ratios are lower
bounds rather than a matched-numerical-error result. This is not an end-to-end,
general, or asymptotic advantage claim.

## Fire Opal boundary

Prepare the sentinel locally without a provider call:

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_validate.py --phase sentinel
```

Provider compatibility validation is a distinct, zero-quantum-second action:

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_validate.py \
  --phase sentinel --validate --force
```

The runner reads credentials only at that boundary. Prefer `QCTRL_API_KEY`,
`IBM_CLOUD_API_KEY`, and `IBM_QUANTUM_CRN`; an explicitly supplied legacy
notebook is read in memory and never copied into an artifact.

After a passing provider validation, create a provider-free hardware plan:

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py plan --phase sentinel
```

Submission is intentionally not shown as a copy-paste default. It requires the
phase-specific confirmation literal printed in the plan and explicit user
authorization at that time. The large plan additionally requires a successful
retrieved sentinel result. Retrieval uses persisted action IDs and cannot
resubmit work.

## Quantum-second budget

| Phase | Circuits | Shots | Estimate | Declared cap |
|---|---:|---:|---:|---:|
| Validate-only | varies | 128 | 0 | 0 |
| Sentinel | 192 | 24,576 total | 30-40, central 35 | 50 |
| Large | 1,536 | 196,608 total | 240-320, central 280 | 400 |
| Full study | 1,728 | 221,184 total | 270-360, central 315 | 450 |

The estimate was calibrated against the provider-reported 26 quantum seconds
for the matching q40 192x128 run. The completed q60 sentinel was also reported
as 26 quantum seconds in the Fire Opal dashboard, below its 50-second cap. The
caps are declared research controls; the Fire Opal API does not enforce them.
No retry, shot increase, or backend switch is automatic.

## Result analysis and claim tiers

After retrieval, run the local analysis without additional quantum time:

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py hardware-analysis \
  --hardware-result PATH_TO_RESULT.json \
  --output PATH_TO_ANALYSIS.json
```

- A successful sentinel is a hardware-feasibility milestone.
- Non-converged MPS plus a large hardware run remains feasibility-only.
- A frozen large hardware classifier that strictly beats both classical
  baselines is an empirical hardware-advantage candidate for this task.
- No result from this workflow establishes general or asymptotic quantum
  advantage.
