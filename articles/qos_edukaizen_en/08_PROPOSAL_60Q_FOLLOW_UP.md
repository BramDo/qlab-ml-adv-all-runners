# The 60-qubit result: hardware 17, linear 16, RBF 14

On 21 July 2026, the previously proposed 60-qubit pilot was executed through Fire Opal on `ibm_fez`. This time, adding width did not merely mean using a larger hash. We replaced the old input with sixty label-free coexpression modules and deliberately kept the circuit shallow.

The fixed held-out test produced the strongest hardware point result in this series:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 60-qubit hardware | 0.53125 | 17/32 |
| classical linear baseline | 0.50000 | 16/32 |
| classical RBF baseline | 0.43750 | 14/32 |

This is a positive intermediate result: on this test, the frozen hardware classifier finished ahead of both predeclared classical references. Because the test set is small and the uncertainty is wide, it is not a general or asymptotic quantum-advantage claim.

## Why the first 60-qubit route failed

Our earlier 60-qubit route obtained 15/32 on hardware, compared with 17/32 ideally and 19/32 classically. The extra qubits mainly carried more hashed input channels. That added width, but not necessarily more stable biological structure.

The new pilot therefore changed the representation, not just the number of qubits:

- sixty coexpression modules learned from a fixed, label-free pool of 512 cells;
- 1,200 variable genes with detection frequencies between 1% and 95%;
- deterministic KMeans with `random_state=6110` and `n_init=20`;
- four summaries per module: mean `log1p`, detection fraction, RMS and top-quartile mean;
- median/IQR scaling learned from training cells only;
- `tanh(z/3)` and per-block L2 normalisation.

The module pool, training set and test set are mutually disjoint. Test labels played no role in module construction, scaling, model selection or hyperparameter selection.

## The 60-qubit circuit

The sixty qubits form a logical `6×10` topology. The feature map uses four input blocks, a `sqrt(60)` multiplier, logical depth 20 and 134 two-qubit interactions. X, Y and Z measurements yield 627 ordered observables per cell.

Three measurement circuits were built for each of 32 training and 32 test cells:

- 192 circuits;
- 128 shots per circuit;
- 24,576 shots in total;
- backend `ibm_fez`;
- Fire Opal action `2335848`.

The Fire Opal dashboard reported **26 quantum seconds** for this task. That is strikingly short for 192 circuits on sixty qubits. The archived `get_result` response did not contain this field, so we explicitly identify 26 seconds as the dashboard measurement.

## Training-only selection and blind test

Within the training set, cross-validation selected an RBF SVC for the quantum route with `C=10` and `gamma=0.1`. Mean training-only CV was 0.59375; the worst fold remained at 0.50000. The frozen route was then evaluated once on the 32 protected test cells.

On that test, hardware scored 17/32, the linear baseline 16/32 and the RBF baseline 14/32. The one-cell lead over the strongest baseline is small, but for the first time the direction is positive on real hardware.

## Why the turnaround time is also interesting

According to the Fire Opal dashboard, the quantum task itself took only 26 seconds. Submission to fully retrieved results took approximately 8 minutes and 33 seconds, including orchestration, compilation, queueing and retrieval. The local MPS check of exactly the same 60-qubit representation ran for 42 minutes and 57 seconds without converging: bond dimension 64 had completed, while at 128 only one of eight required parts had finished.

This is a relevant practical indication: for generating these particular wide quantum features, hardware produced a complete result faster than our attempted classical MPS simulation.

It is explicitly not an end-to-end time advantage over ordinary classical ML. The linear and RBF models can train directly on classically prepared data without simulating the 60-qubit circuit. The fair claim is narrower: **the quantum feature generator was executable, achieved the highest held-out point score and became available faster than our incomplete classical simulation of that same quantum representation**.

## Statistical boundary

With 32 test cells, one prediction equals 3.125 percentage points. The exact two-sided McNemar p-value against the stronger linear baseline is 1.0. The 95% bootstrap interval for `hardware minus linear` ranges from -0.1875 to +0.25. A classical lead, a tie and a hardware lead all remain compatible with this small sample.

We therefore report a **task-specific empirical indication of practical partial quantum advantage**, not proven general quantum advantage.

## The next decision gate

The next scientific step is not to spend more quantum time automatically. First, the design must be frozen and the complete classical frontier established for a larger 256/256 split. A large hardware phase would require 1,536 circuits at 128 shots and will receive separate approval only if its information value justifies the Fire Opal budget.

## Sources and reproducibility

- [60-qubit module pipeline](qiskit_qos_pbmc68k_q60_module_pipeline.py)
- [Fire Opal pilot runner](qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py)
- [60-qubit runbook](Q60_MODULE_B4_RUNBOOK.md)
- [QOS paper](https://arxiv.org/abs/2604.07639)
- [Complete repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
