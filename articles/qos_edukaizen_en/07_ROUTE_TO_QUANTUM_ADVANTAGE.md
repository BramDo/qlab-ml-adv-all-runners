# What is still required for quantum advantage?

The 60-qubit pilot of 21 July 2026 provides the first positive hardware point result: 17/32 versus 16/32 for linear and 14/32 for RBF. This is a serious reason to continue testing, but not proven advantage. A credible quantum-ML claim still requires positive answers to three different questions at once: does the model generalise, is the quantum route demonstrably hard to match within the selected classical resource bound, and is the end-to-end resource accounting correct?

## 1. Demonstrate generalisation first

A follow-up study needs more independent cells. That means not only a larger test set, but also several predeclared splits or donors. A useful minimum programme is:

- select models and observables using training data only;
- use several fixed, balanced train/test splits;
- retain a completely untouched final test cohort;
- report balanced accuracy, AUC, calibration and per-class errors;
- report confidence intervals and a preselected paired test;
- check biological stability across donors or batches.

With 32 test cells, one cell can reverse the conclusion. With hundreds of test cells, a small but consistent difference can be assessed much more reliably.

## 2. More hardware data, not merely more qubits

Forty or sixty qubits sound impressive, but width alone does not solve sample scarcity. Both 405 features in the 40-qubit route and 627 observables in the new 60-qubit route face only 32 training cells. This remains statistically unfavourable. Possible improvements include:

- run more training cells on hardware;
- reduce the observable panel using training data only;
- execute several independent 128-shot runs;
- measure calibration drift across days;
- compare shot budgets;
- incorporate uncertainty in quantum features into the classifier.

A sixty-qubit circuit can be kept shallower, but it still requires careful readout and batch design. The small sentinel was allowed as a feasibility test when the planned MPS convergence check could not be completed within the available time. A large hardware phase still requires training-only stability, a frozen design and separate approval.

## 3. Move closer to the complete QOS algorithm

The theoretical classification separation applies to an oracle-sketching and quantum-linear-algebra protocol with formal sample access. Our feature map uses four hashed blocks, short rotation layers and local Pauli readout. The next theoretical bridge must make explicit:

- which part of the QOS oracle the circuit approximates;
- how approximation error scales with blocks, shots and depth;
- which QSVT or linear-solver steps are missing;
- whether the 405-feature classifier approximates the same decision function as the theorem’s LS-SVM;
- how much classical side information is required for hashing and circuit control.

Without that bridge, “QOS-inspired” is more accurate than “hardware implementation of Theorem 3.”

## 4. A stronger classical frontier

The classical opponent must be assessed both practically and under explicit resource constraints. At minimum we need:

- sparse logistic regression and LinearSVC on all genes;
- feature selection using training data only;
- hashing, sparse JL and streaming models with measured memory;
- PCA/SVD and kernel routes;
- biological marker baselines;
- tensor-network or causal-cone analysis of the specific quantum circuit;
- runtime, RAM, model size and energy as separate columns.

The official QOS repository added sparse JL projections in May 2026. This matters because the classical frontier moves. An advantage claim must be repeated against the latest strong baseline rather than the baseline available when the project started.

## 5. Which kind of advantage do we mean?

“Quantum advantage” can refer to different claims:

| Claim | Required evidence |
| --- | --- |
| predictive advantage | significantly better held-out performance |
| space advantage | the same task performance using demonstrably less working memory |
| time advantage | lower fair end-to-end time at the same error tolerance |
| sample advantage | fewer data points needed for the same generalisation |
| scaling advantage | more favourable measured growth with problem size |

QOS theory mainly concerns machine size and, in dynamic cases, sample complexity. Our pilot mainly measures hardware feasibility and predictive accuracy. These remain different axes.

## A realistic experimental ladder

After the successful 60-qubit sentinel, the route forward has five new gates:

1. **Freeze the representation:** retain the sixty label-free gene modules, 627 observables and classifier selection without test feedback.
2. **Broaden the classical frontier:** add sparse linear, kernel, marker, JL and streaming baselines with measured time and memory.
3. **Larger local splits:** test 256/256 and several preselected seeds before using more hardware time.
4. **Large hardware confirmation:** execute the frozen design on more cells only after separate approval and record every batch cost.
5. **Final blind test:** evaluate one untouched cohort and publish a null or negative outcome as well.

Only when the quantum route performs better on the final blind test, or matches performance with convincingly lower measured resources, does an empirical advantage claim emerge.

## What can we already say?

The current series now ends with a more positive but still bounded intermediate result:

> Our 60-qubit QOS-inspired feature map ran on real hardware, scored one cell above the strongest predeclared classical baseline on the fixed 32-cell test and produced complete features faster than our incomplete MPS simulation of the same representation. This is a task-specific practical indication, not a general quantum-advantage claim.

[Part 8](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/proposal-60-qubit-qml-follow-up-study/) gives the executed 60-qubit protocol, the 17/32 outcome, timing and statistical boundary in full.

## Sources and code

- [QOS paper](https://arxiv.org/abs/2604.07639)
- [Official QOS repository](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our Qiskit/Fire Opal repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Hardware milestone pull request](https://github.com/BramDo/qlab-ml-adv-all-runners/pull/1)
