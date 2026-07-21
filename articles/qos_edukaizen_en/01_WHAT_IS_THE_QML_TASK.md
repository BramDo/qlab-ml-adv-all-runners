# What is the QML task? Classifying cells, not looking up genes

The short answer is: **we are not building a gene-search engine**. We perform supervised binary classification. The model receives the gene-expression profile of a cell and predicts which of two closely related CD4 T-cell types it belongs to.

## What is in PBMC68k?

PBMC stands for *peripheral blood mononuclear cells*: white blood cells from peripheral blood, including T cells, B cells, natural killer cells and monocytes. Single-cell RNA sequencing counts how many RNA molecules from each gene were observed in every cell. The result is a large matrix:

```math
X \in \mathbb{R}^{N \times D}.
```

One row represents one cell. One column represents one gene. In our local 10x version, each row has 32,738 possible genes. Most values are zero because only a small fraction of all genes is observed in any one cell. The matrix is therefore both high-dimensional and sparse.

An annotation file accompanies the measurement matrix. For some cells it contains a cell type assigned by the original analysis. During supervised learning, this file supplies the labels. It is a database, but the model is not asked a query such as “look up gene IL7R.” It learns a decision rule from complete cell profiles.

## Our specific binary question

We select two classes from the annotated dataset:

- positive label: `CD4+/CD25 T Reg`, 6,187 available cells;
- negative label: `CD4+/CD45RO+ Memory`, 3,061 available cells.

Together they form 9,248 candidate cells. The hardware pilot used a fixed balanced subset: 32 training cells and 32 test cells, with sixteen cells from each class in both partitions.

The prediction for a new cell is therefore:

```text
gene-expression vector of one cell
        ↓
quantum feature map and measurements
        ↓
classical decision rule
        ↓
regulatory T cell or CD4 memory T cell
```

This is a difficult small-data problem. The two classes are biologically related, and the experiment uses far fewer training examples than genes. With 32 training cells against 32,738 raw features, a flexible model can easily memorise the training data without generalising to new cells.

## Where does quantum machine learning enter?

The quantum processor does not replace the complete analysis. It acts as a **feature map**. Classical gene expression is compressed into circuit parameters, after which the circuit prepares a quantum state and introduces interactions between qubits. Measurements then produce 405 numbers per cell. An ordinary classical classifier learns from those measured feature vectors.

This is therefore a hybrid QML pipeline:

| Step | Classical or quantum? |
| --- | --- |
| Load RNA matrix and labels | classical |
| Select active genes and hash pairs | classical |
| Prepare the state and let qubits interact | quantum |
| Measure Pauli observables | quantum |
| Train the classifier and predict cell type | classical |

The scientific question is not only whether this combination provides reasonable classification. The deeper QOS question is whether a quantum representation can retain information from a very large feature space with much less internal memory than a general classical streaming model.

## Classification is not biomarker discovery

Interpretation is an important limitation. Because gene pairs are deterministically hashed into 160 buckets, several pairs can land in the same bucket. The model may learn a useful pattern without every measured quantum feature mapping to one specific gene or biological mechanism.

Our current task therefore does not directly answer:

- which individual gene causes the difference;
- which pathway is biologically decisive;
- whether the label is clinically reliable;
- whether the classifier generalises to other donors.

Those questions require separate interpretation, validation and cohort studies. The present pilot is a computational experiment with a biologically real, high-dimensional input.

## Why the task is still relevant

Single-cell data is growing rapidly in both cell count and the number of measurable features. The challenge is not that a laptop cannot find one gene, but that learning from large, changing and sparse matrices requires memory and data movement. QOS formulates a streaming model for precisely such situations: a sample is processed, the internal state is updated, and the sample need not remain permanently in working memory.

In [part 2](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/quantum-oracle-sketching-theory/) we examine that theory. It will become clear why the paper discusses machine size and sample access, and why higher test accuracy alone does not establish quantum advantage.

## Sources

- [10x PBMC68k / Fresh 68k PBMCs](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
- [QOS paper: dataset and classification design](https://arxiv.org/abs/2604.07639)
- [Official PBMC68k code in the QOS repository](https://github.com/haimengzhao/quantum-oracle-sketching/tree/main/real_datasets)
