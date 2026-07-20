# From gene expression to 40 qubits

Quantum machine learning is often described as if a quantum computer searches an entire database at once. Our experiment does something more precise and more interesting: it attempts to **predict the cell type of a single cell from its gene-expression profile**. The input is a long, sparse vector of RNA counts; the output is one of two immune-cell classes.

This eight-part series connects three layers that are easily confused. The first is the theory of *Quantum Oracle Sketching* (QOS), published in April 2026. That theory concerns a small quantum model processing massive classical data streams without retaining the entire matrix. The second layer is the official JAX code and its numerical PBMC68k experiments. The third is our own shallow, QOS-inspired translation to a circuit that was actually executed on 40 physical qubits of IBM Fez through Fire Opal. The additional eighth part proposes a possible 60-qubit follow-up study, while deliberately postponing its execution.

## The series

1. [What is the QML task? Classifying cells, not looking up genes](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/what-is-the-qml-task-cell-classification/)
2. [The theory of Quantum Oracle Sketching](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-oracle-sketching-theory/)
3. [From PBMC68k gene expression to 40 qubits](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/pbmc68k-from-gene-expression-to-qubits/)
4. [From JAX to a 40-qubit hardware circuit](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/qos-to-40-qubit-hardware-fire-opal/)
5. [405 observables and a leakage-free classifier](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-readout-405-observables-classifier/)
6. [The result: hardware 16, classical 17](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/result-hardware-versus-classical/)
7. [What is still required for quantum advantage?](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/route-to-quantum-advantage-qml/)
8. [Proposal: a 60-qubit QML follow-up study](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/proposal-60-qubit-qml-follow-up-study/)

## The experiment in one table

| Component | Value |
| --- | --- |
| Dataset | PBMC68k / Zheng68k |
| Full input per cell | 32,738 genes |
| Binary task | regulatory CD4 T cell versus CD4 memory T cell |
| Pilot | 32 training and 32 test cells |
| Quantum representation | 4 blocks of 40 qubits |
| Hardware | IBM Fez through Fire Opal |
| Circuits | 192 circuits, 128 shots per circuit |
| Readout | 405 Pauli observables per cell |
| Hardware test | 0.50000 — 16 of 32 correct |
| Classical test | 0.53125 — 17 of 32 correct |

## What this series does and does not claim

The execution shows that the complete route—from real single-cell RNA data, through a compact quantum feature map, to measured hardware features and a predeclared classifier—is technically executable. That is a concrete intermediate hardware result.

It is not yet empirical quantum advantage. Hardware lost the fixed test by one cell, the uncertainty interval is wide, and the physical qubits used here are not equivalent to the error-corrected logical qubits in the theory. Our shallow feature map is also a hardware-oriented approximation rather than the complete QOS/QSVT algorithm.

That is exactly why the series is useful. It explains not only how the theory works, but also where the difficult transition to real hardware lies: data access, circuit depth, readout, shot noise, generalisation, and a fair classical comparison.

## Primary sources

- [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639)
- [Official Quantum Oracle Sketching code](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our Qiskit and Fire Opal runners](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [10x PBMC68k dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
