# From gene expression to 60 qubits

Quantum machine learning is often described as if a quantum computer searches an entire database at once. Our experiment does something more precise and more interesting: it attempts to **predict the cell type of a single cell from its gene-expression profile**. The input is a long, sparse vector of RNA counts; the output is one of two immune-cell classes.

This eight-part series connects four layers that are easily confused. The first is the theory of *Quantum Oracle Sketching* (QOS), published in April 2026. That theory concerns a small quantum model processing massive classical data streams without retaining the entire matrix. The second layer is the official JAX code and its numerical PBMC68k experiments. The third is our **literal flat-QOS sketch on four qubits**: a bounded port of the official sampling kernel, physically executed on IBM Fez. The fourth consists of our 40- and 60-qubit PBMC68k routes. Those are QOS-inspired NISQ feature maps, explicitly not literal implementations of the complete QOS/QSVT algorithm.

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
| Local time separation for the same feature target | greater than 99.1x at kernel scope; greater than 5.0x including retrieval |

## Which components are literal QOS?

| Route | Relationship to the QOS paper |
| --- | --- |
| 4q flat-QOS toy/pilot | Literal port of the official `q_state_sketch_flat` sampling kernel for $D=16$ and $M=64$; 66 circuits on IBM Fez, Fire Opal action `2334156`; mean Hellinger fidelity 0.990104 over 64 random kernels |
| 40q/60q PBMC68k | QOS-inspired hardware feature maps using classically computed rotation angles and Pauli readout; no literal sampling oracle, QSVT or classical-shadow chain |
| Complete paper route | Streaming sample access, oracle construction, quantum linear algebra and controlled readout; not implemented end to end on hardware in this project |

The 4q run therefore shows that a genuine QOS sketch building block works on hardware. The 60q run shows something different: a wide, shallow and biologically structured feature map for real PBMC68k data is executable and reaches an interesting local timing and point score. Neither result by itself is a hardware proof of Theorem 3.

## What this series does and does not claim

The execution shows that the complete route—from real single-cell RNA data, through label-free gene modules and a compact quantum feature map, to measured hardware features and a predeclared classifier—is technically executable. The 60-qubit route also had the best held-out point score of the three preselected models.

Under the declared local resources, this is a measured **time-to-feature-generation advantage**. The 26-second quantum task is more than 99.1x faster than the MPS attempt that remained incomplete after 2,577 seconds; even the complete 513-second route through retrieval is more than 5.0x faster. The feature target was the same, but MPS did not converge and therefore produced no matched numerical error. This is not a general or asymptotic quantum-advantage claim. The linear and RBF classifiers themselves remain inexpensive, the test contains only 32 cells, and the uncertainty interval is wide. The 26 quantum seconds come from the Fire Opal dashboard; the archived API result left that field empty. The 60q feature map is a hardware-oriented QOS-inspired adaptation, **not a literal QOS implementation**, and not the complete QOS/QSVT algorithm.

That is exactly why the series is useful. It explains not only how the theory works, but also where the difficult transition to real hardware lies: data access, circuit depth, readout, shot noise, generalisation, and a fair classical comparison.

## Primary sources

- [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639)
- [Official Quantum Oracle Sketching code](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our Qiskit and Fire Opal runners](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Pro Student Quantum Advantage List](https://edukaizen.nl/pro-student-quantum-advantage-list/)
- [10x PBMC68k dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
