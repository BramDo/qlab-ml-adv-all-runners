# From PBMC68k gene expression to 40 qubits

A quantum processor with forty qubits cannot simply store 32,738 gene columns as 32,738 individual qubits. The crucial step is therefore the encoding: a deterministic, reproducible translation from a sparse gene-expression vector to four compact blocks of forty numbers each.

## Step 1: one cell as a sparse vector

PBMC68k contains UMI counts per gene for each cell. In the two selected cell types, approximately six hundred genes are active in an average pilot cell. Zero values do not need to be processed. We retain the 48 most active genes in each cell so circuit construction remains bounded and equally sized for every cell.

This is a practical hardware choice. It can remove informative weak genes and is therefore part of the model definition rather than merely a technical optimisation.

## Step 2: gene pairs instead of individual genes

Individual gene expression can distinguish cell types, but interactions can carry additional structure. Forty-eight active genes produce:

```math
\binom{48}{2}=1128
```

gene pairs. The weight of a pair is the product of the log-transformed expression values. This reduces the dominance of extremely large raw counts while preserving co-activation.

This pairwise feature space is much larger than the qubit count. We do not materialise a complete matrix containing every possible gene pair. Each pair that is actually observed is processed immediately.

## Step 3: deterministic hashing into 160 buckets

Each gene pair is assigned to one of 160 buckets using hash seed 7. Values landing in the same bucket are added. Feature hashing avoids a large explicit projection matrix but causes collisions: different pairs can share a bucket.

The 160 values are rearranged into four blocks:

```text
160 hash buckets = 4 coherent blocks × 40 qubits
```

Each non-empty block is L2-normalised separately. All four blocks had norm one in the pilot. The encoding is label-free: the same transformation is applied regardless of whether the cell is regulatory or memory T cell.

## Step 4: streaming through the same register

The four blocks are not placed on four separate quantum chips. They are uploaded sequentially into the same forty-qubit register. An interaction layer follows each upload, so a later block changes the state built by earlier blocks.

This is the QOS-inspired element: samples or feature blocks are processed sequentially in a compact quantum state. It is not, however, an exact implementation of the complete QOS oracle, QSVT and classical-shadow chain from the paper.

The interaction strength in the selected architecture scales with the square root of register width:

```math
\sqrt{40}=6.324555\ldots
```

This `sqrt(q)` choice came from an earlier label-free resource preflight. That preflight checked whether the circuit family remained structurally interesting and numerically manageable as width increased. Test labels were not used to select the architecture.

## What is lost?

Compression is not magic. Four kinds of information are lost or mixed:

- genes outside a cell’s top 48 disappear;
- hash collisions merge different gene pairs;
- four normalised blocks lose absolute scale information;
- with 128 shots, quantum expectation values are estimated only approximately.

In return, the circuit can produce a nonlinear, interfering representation of the 160 hashed values. The hope is that relevant class structure enters measurable correlations.

## Is this biologically interpretable?

Not at gene level without additional bookkeeping. To determine which gene pairs dominate a bucket, a reverse mapping from gene pairs to buckets must be retained for each cell or cohort. That is possible, but it enlarges the classical sidecar and is not the primary readout in this pilot.

The current representation is designed for classification and hardware feasibility. It is not a pathway database or biomarker analysis. A follow-up study could add collision audits, marker enrichment and stability tests across several hash seeds.

## Why not simply use PCA?

PCA, sparse linear models and feature hashing are strong classical baselines and must always be included. The quantum route becomes interesting only if a compact state retains predictive structure that is difficult to preserve classically under the same resource constraint.

We therefore use two classical comparisons in this pilot: a full raw-gene log1p model and a model on exactly the same 160 hashed features. In [part 6](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/result-hardware-versus-classical/) we see that the raw-gene logistic classifier ultimately classified one more test cell correctly than the hardware feature map.

## Reproducible parameters

| Parameter | Value |
| --- | --- |
| genes per cell in source matrix | 32,738 |
| maximum active genes | 48 |
| pair events per cell | 1,128 |
| hash seed | 7 |
| hash buckets | 160 |
| blocks | 4 |
| qubits per block | 40 |
| interaction scale | `sqrt(q)` |

In [part 4](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/qos-to-40-qubit-hardware-fire-opal/) we turn these blocks into 192 measurement circuits for IBM Fez.

## Sources

- [Hardware validation runner](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_validate.py)
- [PBMC68k loader and annotations](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/main/qiskit_qos_pbmc68k_utils.py)
- [Official QOS real-dataset code](https://github.com/haimengzhao/quantum-oracle-sketching/tree/main/real_datasets)
