# From JAX to a 40-qubit hardware circuit

The original QOS paper provides JAX code and numerical simulations. Our next question was simpler but physically concrete: can we execute a recognisable streaming-style single-cell feature map on forty qubits and retrieve enough information for a classifier?

## First: a literal QOS building block on four qubits

Before the wide PBMC68k route, we ported the official flat-QOS sampling sketch separately to Qiskit. The 4q toy model uses $D=16$ positions and $M=64$ random samples. The samples build literally the same phase vector as `q_state_sketch_flat`; an additional Hadamard layer converts otherwise invisible phases into measurable interference.

On IBM Fez, 64 random sketch circuits plus two controls ran through Fire Opal action `2334156`. Mean Hellinger fidelity with the ideal distributions was 0.990104, the median was 0.991417 and the minimum was 0.980411. This is a literal flat-QOS kernel on hardware. It is not yet a complete QOS classifier: QSVT, the linear solver and the complete readout chain are missing.

## Then: a hardware-oriented PBMC68k translation

The circuit processes four feature blocks in the same register. A block is uploaded through single-qubit rotations, after which qubit pairs are connected using a fixed interaction structure. A final rotation layer follows the fourth block.

At logical circuit level, each cell uses:

- 40 qubits;
- depth 20 before measurement;
- 87 two-qubit gates;
- fully numerical parameters;
- no mid-circuit measurements or resets.

This is deliberately shallow. The complete theoretical QOS/QSVT protocol requires more complex oracles and error-corrected logical operations. Unlike the 4q sketch above, this 40q circuit is **not a literal QOS implementation**: its rotation angles are prepared classically and there is no sample-by-sample oracle construction. This variant investigates the next experimental boundary: a physical quantum feature map wide enough to be more difficult to simulate classically, yet shallow enough to survive on current hardware.

## Why three measurement circuits per cell?

A quantum state cannot be read out completely in a single measurement. We choose a fixed panel of homogeneous Pauli observables. Each base cell therefore produces three versions:

- measure every qubit in the X basis;
- measure every qubit in the Y basis;
- measure every qubit in the Z basis.

Several single-qubit and two-qubit correlations can be calculated from one global basis measurement. With 64 cells this gives:

```text
64 cells × 3 measurement bases = 192 circuits.
```

Every circuit received 128 shots, for a total shot budget of 24,576. The readout contained 40 single-qubit supports and 95 predeclared qubit-pair supports per cell, each in X, Y and Z. This produces 405 features.

## Validate first, execute later

Because Fire Opal runs are scarce, the workflow used separate safety phases:

1. reconstruct the dataset, split, encoding, circuit shape, QASM and hashes locally;
2. pass all 192 payloads through Fire Opal `validate` without execution;
3. freeze a hardware plan containing backend, shots and readout;
4. submit a single batch only after explicit confirmation;
5. store the action ID and never allow retrieval to resubmit;
6. perform classification locally only after the hardware file was pinned.

Provider validation accepted every circuit for `ibm_fez`. Logical depth was always 20 and exported payload depth was 22 before provider compilation. Calibration warnings from validation were retained rather than filtered out.

## Fire Opal and IBM Fez

[Fire Opal](https://q-ctrl.com/fire-opal) is a performance-management layer from Q-CTRL. It compiles and optimises circuits for the selected device and applies error-suppression or mitigation steps. That does not make the measured result error-free, but it can improve the amount of usable algorithmic information obtained from noisy hardware.

The run received Fire Opal action ID `2334162` and used IBM Fez. The provider reported 26 quantum seconds. This is QPU usage time rather than complete turnaround time: data preparation, compilation, queueing, classical post-processing and model selection are not included.

## What came back?

Retrieval returned exactly 192 probability distributions in manifest order. Every value was finite and non-negative. Normalisation differed from one by no more than approximately $2.2\times 10^{-15}$. Each circuit produced between 114 and 128 distinct bit strings.

From these results we reconstructed 64 rows of 405 expectation values. Values ranged from -0.90625 to 1.0. Bit order is explicitly Qiskit little-endian: the rightmost bit corresponds to qubit zero.

This is the main intermediate hardware result: not only were the circuits accepted, but the entire chain from real cell data to usable, ordered quantum features was completed.

## Is this the first single-cell QML run?

We should not make such a broad claim. Quantum-kernel studies on biological cell classification and real IBM hardware already existed, including a 2023 study of neuronal M-types. The defensible novelty is narrower: an early and, to the best of our present knowledge, first physical 40-qubit feasibility execution of this specific **QOS-inspired coherent PBMC68k route**.

Even that wording should be preceded by a systematic literature review in a formal publication. This series therefore primarily uses “from JAX to 40-qubit hardware,” not “the first QML experiment ever.”

In [part 5](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-readout-405-observables-classifier/) we examine how the 405 features are used without leaking the test set into model selection.

## Sources

- [Official QOS repository: JAX implementation](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Our literal 4q flat-QOS pilot](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/main/qiskit_official_qos_flat_fireopal_pilot.py)
- [Fire Opal](https://q-ctrl.com/fire-opal)
- [Our guarded hardware pilot](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_pilot.py)
- [Earlier cell classification with quantum kernels on IBM hardware](https://www.nature.com/articles/s41598-023-38558-z)
