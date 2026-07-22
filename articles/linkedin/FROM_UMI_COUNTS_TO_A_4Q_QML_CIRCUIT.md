# From UMI Counts to a Four-Qubit Circuit: A Beginner's Route into Real-Data QML

Quantum machine learning is often introduced from the circuit outward: start with qubits, add gates, measure, and only later ask what the numbers mean.

For beginners, I think the opposite direction is more useful.

Start with one real data point. Follow every transformation. Only then draw the circuit.

I have published a new beginner's guide that does exactly this for single-cell RNA data from the PBMC68k dataset. It follows one immune cell from raw UMI counts to four rotation angles, a four-qubit Qiskit circuit, eight measured quantum features, and a classical classifier.

The guide is available here:

https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qml-beginnershandleiding-umi-naar-4-qubit-circuit/

The complete reproducible code and downloadable DOCX version are here:

https://github.com/BramDo/qlab-ml-adv-all-runners/tree/main/docs/beginner

## What is the machine-learning task?

PBMC68k is a single-cell gene-expression dataset. Each row represents a cell, each column represents a gene, and each value is a molecule count approximated through Unique Molecular Identifiers, or UMIs.

The small educational task distinguishes two immune-cell classes:

- CD4+/CD25 T regulatory cells;
- CD4+/CD45RO+ memory cells.

The full loader contains 68,579 annotated cells and 32,738 genes. The tutorial deliberately uses only 16 training cells and 16 fully separated test cells. This is not intended to set a performance record. Its purpose is to make the complete data-to-circuit route small enough to inspect.

## Step 1: understand the UMI count

RNA molecules are amplified before sequencing. Without a molecular identifier, multiple sequencing reads can be copies of the same original molecule. A UMI gives the original molecule a short barcode, allowing duplicated reads to be collapsed.

If nine reads contain only two distinct UMIs, the expression count is two, not nine.

This matters because the circuit does not receive an abstract floating-point vector from nowhere. It receives angles derived from sparse, noisy, biologically measured molecule counts.

## Step 2: protect the test set

Before selecting genes or learning a scale, the cells are split with a fixed seed.

All data-dependent choices are learned from the training cells only:

1. select four variable genes without using labels;
2. learn the normalization and scaling parameters;
3. generate training quantum features;
4. train the classical classifier;
5. evaluate the held-out cells once.

For the fixed split, the four selected genes are IER2, ACTG1, LIMD2, and GLTSCR2.

This order is important. A visually impressive circuit cannot repair leakage introduced before the circuit is built.

## Step 3: turn counts into angles

Cells have different total numbers of detected molecules, so each gene count is normalized by the cell's total UMI count, scaled to 10,000, and transformed with log1p.

In plain-text notation:

n_ig = 10,000 x_ig / sum_h(x_ih)

l_ig = log(1 + n_ig)

The training mean and standard deviation then produce a z-score. After clipping the z-score to the interval [-3, 3], the rotation angle is:

theta_ig = pi x clip(z_ig, -3, 3) / 3

This maps each selected gene to one angle between -pi and pi.

For one real training cell, the four raw UMI values are 0, 3, 1, and 2. After normalization and scaling, they become the four circuit angles:

- -1.218720 radians;
- 0.385290 radians;
- 0.278651 radians;
- 0.098712 radians.

That is the concrete bridge from biology to quantum gates.

## Step 4: build the circuit

The circuit starts in |0000>. Each angle controls one RY rotation. Four CNOT gates then connect the qubits in a ring:

0 -> 1 -> 2 -> 3 -> 0

An RY gate has the matrix:

RY(theta) = [[cos(theta/2), -sin(theta/2)],
             [sin(theta/2),  cos(theta/2)]]

The four-qubit rotation layer is a tensor product of four such matrices. The CNOT ring multiplies that layer into one 16 x 16 unitary matrix, because four qubits have 2^4 = 16 computational basis states.

The tutorial exports the full matrix, statevector, circuit drawing, and numerical unitarity check. For the example cell, the Frobenius error in U-dagger U = I is approximately 1.11 x 10^-15.

## Step 5: turn measurements into features

The circuit is sampled with 512 shots. The measured bitstrings are converted into four single-qubit Z expectation values and four neighbouring ZZ correlations.

The result is an eight-dimensional feature vector:

f(x) = (Z0, Z1, Z2, Z3, Z0Z1, Z1Z2, Z2Z3, Z3Z0)

A classical logistic-regression model then learns the final cell-type decision from those eight features.

This hybrid structure is worth emphasizing. The quantum circuit generates features; the classical model still performs the supervised fit.

## What was the result?

On the 16 held-out cells:

- four-qubit quantum features: 7/16 correct;
- classical model using the same four genes: 9/16 correct.

That is not quantum advantage, and the guide says so explicitly.

I consider this a useful result because the objective is education and reproducibility. A beginner should see the complete method, including an honest classical comparison, rather than being shown only a quantum score without context.

## How does this relate to Quantum Oracle Sketching?

This four-qubit PBMC68k tutorial is not a literal implementation of Quantum Oracle Sketching. It uses classically prepared rotation angles, a fixed CNOT ring, and Z/ZZ readout. It is a conventional small quantum feature map.

The same repository contains a separate four-qubit hardware experiment that does port the official q_state_sketch_flat sampling primitive. It also contains a QOS-inspired 60-qubit PBMC68k hardware route. These are three different experimental layers:

- the beginner model explains data-to-circuit encoding;
- the flat-QOS pilot validates one literal QOS building block on hardware;
- the 60-qubit route studies a wide, shallow real-data hardware feature map and a bounded local time-to-feature result.

None of them should be described as the complete QOS/QSVT classifier or as a general end-to-end quantum-ML advantage.

## Why publish a deliberately small example?

Because data encoding is where many QML explanations become vague.

Saying that a vector is "loaded into a quantum state" hides several practical questions:

- Where did the vector come from?
- Which statistics were learned from training data?
- How were values mapped into legal gate parameters?
- What matrix did the circuit implement?
- Which observables became machine-learning features?
- What did the classical baseline do with the same information?

The four-qubit scale lets us answer every one of those questions with real numbers.

My next educational goal is to keep this transparent route intact while gradually connecting it to larger hardware experiments. More qubits are scientifically interesting only when the representation, validation boundary, and classical comparison remain equally visible.

Beginner's guide:
https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qml-beginnershandleiding-umi-naar-4-qubit-circuit/

Code, data snapshots, circuit images, matrices, tests, and DOCX:
https://github.com/BramDo/qlab-ml-adv-all-runners/tree/main/docs/beginner

QOS preprint:
https://arxiv.org/abs/2604.07639

#QuantumComputing #QuantumMachineLearning #QML #Qiskit #Bioinformatics #SingleCellRNA #Education #ReproducibleResearch
