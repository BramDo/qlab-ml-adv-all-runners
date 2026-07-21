# From gene expression to 60 qubits

Quantum machine learning is often described as if a quantum computer searches an entire database at once. Our experiment does something more precise and more interesting: it attempts to **predict the cell type of a single cell from its gene-expression profile**. The input is a long, sparse vector of RNA counts; the output is one of two immune-cell classes.

This eight-part series connects three layers that are easily confused. The first is the theory of *Quantum Oracle Sketching* (QOS), published in April 2026. That theory concerns a small quantum model processing massive classical data streams without retaining the entire matrix. The second layer is the official JAX code and its numerical PBMC68k experiments. The third is our own hardware translation: first a 40-qubit pilot, then a redesigned 60-qubit route based on label-free gene modules. Both circuits were actually executed on IBM Fez through Fire Opal.

The new 60-qubit run is the strongest result in the series. On the predeclared held-out test, hardware scored 17/32, compared with 16/32 for the linear and 14/32 for the RBF baseline. The Fire Opal dashboard reported only 26 quantum seconds, and the complete hardware feature output was retrieved after about 8 minutes 33 seconds. Our classical MPS attempt had not produced a converged reference after 42 minutes 57 seconds.

## The series

1. [What is the QML task? Classifying cells, not looking up genes](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/what-is-the-qml-task-cell-classification/)
2. [The theory of Quantum Oracle Sketching](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-oracle-sketching-theory/)
3. [From PBMC68k gene expression to 40 qubits](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/pbmc68k-from-gene-expression-to-qubits/)
4. [From JAX to a 40-qubit hardware circuit](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/qos-to-40-qubit-hardware-fire-opal/)
5. [405 observables and a leakage-free classifier](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-readout-405-observables-classifier/)
6. [The 40-qubit result: hardware 16, classical 17](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/result-hardware-versus-classical/)
7. [What is still required for quantum advantage?](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/route-to-quantum-advantage-qml/)
8. [The 60-qubit result: hardware 17, linear 16, RBF 14](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/proposal-60-qubit-qml-follow-up-study/)

## The experiment in one table

| Component | Value |
| --- | --- |
| Dataset | PBMC68k / Zheng68k |
| Full input per cell | 32,738 genes |
| Binary task | regulatory CD4 T cell versus CD4 memory T cell |
| Pilot | 32 training and 32 test cells |
| Quantum representation | 60 label-free coexpression modules, 4 statistical blocks |
| Hardware | IBM Fez through Fire Opal |
| Circuits | 192 circuits, 128 shots per circuit |
| Circuit | 60 qubits, logical depth 20, 134 two-qubit interactions |
| Readout | 627 Pauli observables per cell |
| 60q hardware test | 0.53125 — 17 of 32 correct |
| Linear test | 0.50000 — 16 of 32 correct |
| RBF test | 0.43750 — 14 of 32 correct |
| Quantum time reported by the Fire Opal dashboard | 26 seconds |
| Submission to retrieval | about 8 minutes 33 seconds |
| Classical MPS attempt | stopped after 42 minutes 57 seconds without a converged reference |

## What this series does and does not claim

The execution shows that the complete route—from real single-cell RNA data, through label-free gene modules and a compact quantum feature map, to measured hardware features and a predeclared classifier—is technically executable. The 60-qubit route also had the best held-out point score of the three preselected models.

This is a strong practical, partial quantum-advantage indication: higher held-out accuracy together with much faster feature generation than our attempt to classically simulate the same 60q quantum circuit. It is not a general or asymptotic quantum-advantage claim. The linear and RBF classifiers themselves remain inexpensive, the test contains only 32 cells, and the uncertainty interval is wide. The 26 quantum seconds come from the Fire Opal dashboard; the archived API result left that field empty. Our shallow feature map is also a hardware-oriented approximation rather than the complete QOS/QSVT algorithm.

That is exactly why the series is useful. It explains not only how the theory works, but also where the difficult transition to real hardware lies: data access, circuit depth, readout, shot noise, generalisation, and a fair classical comparison.

## Primary sources

- [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639)
- [Official Quantum Oracle Sketching code](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our Qiskit and Fire Opal runners](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [10x PBMC68k dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
