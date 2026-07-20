# Proposal: a 60-qubit QML follow-up study

Sixty qubits can carry more of a gene-expression profile at once than forty qubits. This does not mean that a 60-qubit model automatically classifies better. Extra width can also create more noise, more observables and more opportunities for overfitting. This chapter is therefore **a research proposal, not an announced execution**. We are not submitting new circuits to Fire Opal now and are not starting a large classical sweep.

## What the existing 60-qubit pilot already taught us

We have already executed an exploratory 60-qubit route on the same PBMC68k task. On the fixed small test set, the results were:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 60-qubit hardware | 0.46875 | 15/32 |
| ideal 60-qubit representation | 0.53125 | 17/32 |
| classical reference | 0.59375 | 19/32 |

More qubits did not improve performance here. In a later local blind test with 256 training and 256 test cells, the ideal 60-qubit representation remained at 0.53125, compared with 0.55859 for a linear model and 0.54688 for an RBF model on the same hashed input. This is useful negative information: the current 60-qubit design should not be sent to hardware again without modification.

## The real bottleneck: preparation before hardware

Most research time is not spent in quantum seconds. It is spent freezing data, splits, gene representations, classical references, observable selection and statistical tests. Every new dataset can reopen that chain. A sensible 60-qubit project should therefore reuse as much of the existing PBMC68k infrastructure as possible:

- the same two CD4 T-cell classes;
- the same controlled data loader and normalisation;
- predeclared larger splits;
- the already implemented linear and RBF references;
- the same balanced-accuracy and paired-test code;
- the same Fire Opal validation and retrieval route.

We are therefore not proposing a new biological dataset. The research task remains cell classification; only the quantum representation changes.

## What sixty qubits must add

A new route is useful only if the twenty additional qubits preserve new information. The proposal uses the extra width for stable gene modules or additional hash channels, not a simple stretching of the 40-qubit circuit. Four blocks of sixty qubits could, for example, carry 240 compact input channels instead of 160. A shallow interaction layer could then mix correlations between those modules.

The design limits remain deliberately strict:

- no more than sixty physical qubits;
- low circuit depth and hardware-friendly connectivity;
- a small observable panel selected on training data only;
- no use of the test set for feature, model or hyperparameter selection;
- a shot-noise projection before hardware is considered.

More measurable correlations are not automatically better. With hundreds of possible observables and few training cells, the final classical classifier can easily learn accidents. The proposal therefore prefers a few dozen stable observables over every possible qubit pair.

## A short go/no-go ladder

To limit classical preparation, the study has four sequential gates:

1. **Freeze the task.** Reuse PBMC68k, the existing labels and five preselected larger splits. Do not start another dataset search.
2. **Compare representations.** Test only the present 40-qubit anchor and one new 60-qubit design locally against the existing linear and RBF frontier.
3. **Project hardware effects.** Add 128-shot noise and a realistically limited readout. Stop if the advantage disappears.
4. **Decide on Fire Opal.** Propose a small frozen hardware pilot only if the ideal 60-qubit route beats both classical references on at least four of five splits.

These gates do not prove quantum advantage. Their main purpose is to prevent quantum time being spent on a representation that is already uncompetitive locally.

## What a later hardware pilot would measure

If the local gates ever pass, the first hardware question is modest: does the local 60-qubit signal remain visible after transpilation, noise and finite shots? The pilot need not yet be a definitive advantage test. Circuit, observables, classifier and test cells must nevertheless be frozen before submission. Reporting would then compare four levels: ideal quantum, projected shot noise, Fire Opal hardware and the frozen classical frontier.

Only a much larger independent blind test could then establish predictive advantage. A claim about computational or space advantage would additionally require separate resource accounting; higher accuracy alone would not prove it.

## Decision: document and postpone

A redesigned 60-qubit route may outperform our 40-qubit pilot because it can preserve more relevant gene information and interactions. The existing results nevertheless show that width alone is insufficient. The largest expected improvement must first come from representation and enough training data.

The present decision is therefore: **record the proposal, but do not execute the study yet**. The route remains available for a later time when classical preparation effort and Fire Opal budget can be allocated deliberately.

## Sources and results

- [QOS paper](https://arxiv.org/abs/2604.07639)
- [Official QOS repository](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our Qiskit/Fire Opal repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Published 40-qubit result](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/result-hardware-versus-classical/)
