# The result: hardware 16, classical 17

After days of data selection, representation tests, circuit validation, Fire Opal execution and local analysis, the fixed test reduces to one simple numerical difference:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 40-qubit hardware features | 0.50000 | 16/32 |
| classical raw-gene frontier | 0.53125 | 17/32 |
| hardware minus classical | -0.03125 | -1 cell |

Hardware therefore did not win. But “classical won” would also be too strong. With 32 test cells, one prediction equals 3.125 percentage points. The result is effectively a tie around chance level.

## What happened to the training signal?

During training-only cross-validation, the selected hardware classifier obtained a mean score of 0.59375. The classical winner scored 0.53125. This was a small positive sign: the measured quantum features apparently contained enough structure to rise above chance in some training folds.

On the preprotected test set, hardware dropped to 0.50000. The signal therefore did not demonstrably generalise. Because both final models fit their complete training data perfectly, the pattern is consistent with overfitting or unstable small-sample geometry.

We therefore call the CV lead an **exploratory training signal**, not partial quantum advantage.

## The paired comparison

Among the 32 test cells there were:

- 7 for which only the hardware classifier was correct;
- 8 for which only the classical classifier was correct;
- 17 for which both were correct or both were wrong.

The exact two-sided McNemar p-value is 1.0. The 95% bootstrap interval for `hardware minus classical` ranges from -0.25 to +0.1875. Both a material classical advantage and a hardware advantage remain compatible with the uncertainty of this pilot.

The separate Wilson intervals are also wide:

- hardware: approximately 0.336 to 0.664;
- classical: approximately 0.364 to 0.691.

## What was achieved?

The predictive result is weak, but the execution produced four concrete outcomes:

1. A real PBMC68k cell can be reproducibly translated from 32,738 gene columns into a four-block 40-qubit circuit.
2. A batch of 192 circuits can be validated, executed and retrieved without resubmission through Fire Opal on IBM Fez.
3. From only three global measurement bases, 405 ordered features can be reconstructed for every cell.
4. Hardware features can be evaluated in a predeclared, test-leakage-free ML pipeline against a recomputed classical frontier.

This is a **hardware-feasibility milestone**: evidence that the pipeline is executable. It is not evidence that the pipeline already generalises well enough to be useful.

## What about the 26 quantum seconds?

The provider reported 26 QPU seconds for the 192 circuits. This is strikingly compact compared with some complete classical simulations of wide quantum circuits. It is nevertheless not a speedup measurement for the ML task.

A fair timing claim must also count:

- classical gene selection and hashing;
- circuit construction and QASM export;
- provider compilation and queueing;
- Fire Opal processing;
- data retrieval and observable calculation;
- classifier training;
- the time of the best classical end-to-end pipeline.

Moreover, the classical raw-gene classifier obtained a better point estimate here. A short QPU kernel is not enough when the final task is not solved better or more cheaply.

## Why this is still publishable

Negative and near-neutral hardware measurements are scientifically useful when the protocol is fixed in advance. This pilot shows exactly where a theoretical space benefit can disappear on its way to NISQ hardware:

- aggressive feature compression can remove biological signal;
- hash collisions can mix interactions;
- shallow circuits may be insufficiently expressive;
- noise and 128 shots distort correlations;
- 32 training examples are too few for a stable 405-dimensional classifier.

The correct conclusion is not “QOS does not work.” The complete QOS algorithm was not tested. The conclusion is: **this shallow 40-qubit QOS-inspired feature map was executable, but on the selected 32/32 PBMC split it did not generalise better than the classical frontier**.

## The claim in one sentence

> We demonstrated an end-to-end 40-qubit hardware-feasible QOS-inspired single-cell classification pipeline; it produced an exploratory training-CV signal but no held-out predictive quantum advantage.

In [part 7](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/route-to-quantum-advantage-qml/) we describe the experiments required to turn this intermediate result into a serious advantage test.

## Reproducibility

- [Hardware analysis runner](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis.py)
- [GitHub pull request containing the complete runner and test suite](https://github.com/BramDo/qlab-ml-adv-all-runners/pull/1)
