# The theory of Quantum Oracle Sketching

Quantum Oracle Sketching attempts to bypass a longstanding problem in quantum algorithms: how can a quantum computer access an enormous classical dataset without first building equally enormous quantum memory or QRAM?

The [paper by Zhao and colleagues](https://arxiv.org/abs/2604.07639), posted on 8 April 2026, formulates an answer in the streaming model. The official repository now contains JAX implementations, tutorials, QSVT routines and experiments on datasets including PBMC68k. At the time of our check, the latest public main branch reached commit `10c092c` from 21 May 2026, which added sparse Johnson–Lindenstrauss projections to the real-data route.

## The data-access problem

Many quantum algorithms assume an oracle: a coherent operation that makes information about a vector, matrix or function accessible in superposition. Such an oracle is not free for classical data. If the entire dataset first has to be loaded into QRAM, the memory benefit can disappear before quantum computation begins.

QOS reverses the order. It receives random classical samples one by one. Each sample controls a small quantum rotation. Accumulating many such updates produces a quantum channel approximating the desired oracle operation. The sample can then be discarded.

Schematically:

```text
sample z1 → small rotation ┐
sample z2 → small rotation ├→ compact quantum sketch → quantum algorithm
sample z3 → small rotation ┘
```

The quantum state does not store every data point separately. It stores a coherent summary suitable for a subsequent quantum query.

## The classification theory

For binary classification, the paper writes the training data as a sparse matrix $X \in \mathbb{R}^{N \times D}$ with labels $y_i \in \{-1,+1\}$. The classical reference is a regularised least-squares support vector machine, equivalent to a ridge-like linear classifier:

```math
w = \operatorname*{argmin}_{w}\;\lVert Xw-y\rVert_2^2 + \lambda\lVert w\rVert_2^2.
```

A new feature vector $x^{\prime}$ receives label $\operatorname{sign}(x^{\prime} \cdot w)$. QOS constructs quantum oracles with which a quantum linear-algebra algorithm can approximate the relevant decision information without storing the complete $D$-dimensional parameter space classically.

Under the formal assumptions of the model, Theorem 3 states that a quantum machine of size `poly(log D)` can solve the classification task with approximately linearly many samples in $N$, whereas a classical machine of size $O(D^{0.99})$ cannot. The dynamic variant adds a separation in sample efficiency when the data stream changes but the decision rule remains approximately constant.

## Where exactly is the advantage?

The claimed advantage is primarily a **space or machine-size advantage**. The paper compares logical qubits with classical floating-point memory units. For PBMC68k and other datasets, the numerical study indicates that a QOS curve using fewer than sixty logical qubits can retain high performance while general classical streaming and sparse-matrix routes require much more storage.

This does not automatically mean:

- that a current physical QPU is faster in wall-clock time;
- that data loading is free;
- that every domain-specific classical heuristic has been excluded;
- that sixty noisy physical qubits equal sixty logical qubits;
- that better accuracy defines the theoretical advantage.

The paper itself describes the real-data figures as numerical experiments. The implementation is a JAX simulation, and dataset-specific classical heuristics are listed as future work. The asymptotic theorem and the practical PBMC graph support one another, but they are not the same object of proof.

## Why the Born rule matters

A striking element of the theory is the quadratic relationship between amplitudes and probabilities. Samples control small unitary updates; convergence to the expected oracle operation is related to that probabilistic structure. The paper proves that the quadratic sample scaling required by this construction is optimal.

QSVT and classical shadows are then needed to compute useful functions of vectors and matrices from the sketch and read them out compactly in classical form. This goes far beyond “putting a data point into rotation angles.” The complete theoretical protocol consists of data access, oracle construction, quantum linear algebra and controlled readout.

## From official code to two different hardware routes

The [official repository](https://github.com/haimengzhao/quantum-oracle-sketching) contains two numerical routes:

- explicit random sampling in `qos_sampling.py`;
- an expected-unitary route in `qos.py` for more efficient benchmarking.

In the smallest hardware route we did port one component literally: the official flat-QOS sampling kernel `q_state_sketch_flat`. For $D=16$, $M=64$ and four qubits, each random sample contributes to a phase

```math
\phi_j=\frac{\pi D}{M}\sum_{t=1}^{M}\mathbf{1}[i_t=j]\frac{1-v_{i_t}}{2},
\qquad
U_{\mathrm{sketch}}=\sum_j e^{i\phi_j}|j\rangle\langle j|.
```

After preparing $|+\rangle^{\otimes 4}$, the circuit applies this sample-dependent diagonal. A final Hadamard layer makes the phase spectrum measurable. The Fire Opal run on IBM Fez contained 64 random kernels and two controls; action `2334156` reached mean Hellinger fidelity 0.990104. This is a phase-sensitive hardware result for a genuine QOS building block, but not a complete QML classifier or an advantage proof.

Our 40q and 60q PBMC68k routes are deliberately different. They construct shallow QOS-inspired feature maps that fit present hardware, using classically computed rotation angles and local Pauli readout. They contain no literal random-sampling oracle, reusable coherent query oracle, QSVT/linear solver or exact classical-shadow readout. This lets us physically test the transition from large classical input to a small quantum machine, but it does not automatically inherit the complete guarantee of Theorem 3.

That distinction is the central rule of this article series: **the 4q flat-QOS pilot implements one literal sketch kernel; the 40q/60q PBMC68k pilots are hardware adaptations, not literal implementations of the complete algorithm**.

In [part 3](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/pbmc68k-from-gene-expression-to-qubits/) we follow exactly how 32,738 genes become four blocks of forty numbers.

## Sources

- [QOS paper, arXiv:2604.07639](https://arxiv.org/abs/2604.07639)
- [Official QOS code and real-data experiments](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Latest official commit checked](https://github.com/haimengzhao/quantum-oracle-sketching/commit/10c092cefcfdff9951bf5729bd2ffb4c25fe2254)
