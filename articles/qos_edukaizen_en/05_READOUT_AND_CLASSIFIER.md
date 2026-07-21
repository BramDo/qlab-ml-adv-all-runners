# 405 observables and a leakage-free classifier

A hardware experiment becomes a machine-learning experiment only when the roles of training data, model selection and testing are clear in advance. With just 32 training and 32 test cells, a single careless choice can strongly distort the score.

## The quantum processor supplies features, not the final label

For every cell, we calculate 405 expectation values from the X, Y and Z measurements. A single feature has the form:

```math
\langle P_i \rangle \quad\text{or}\quad \langle P_iP_j \rangle,
```

where $P$ is an X, Y or Z Pauli operator. Together they describe local qubit states and selected correlations. The hardware stage ends with this feature vector.

An ordinary supervised classifier then begins. This matters for interpretation: the prediction is produced by the combination of a quantum feature map and a classical decision model. A difference from a classical baseline may originate in either component.

## The predeclared protocol

The hardware analysis was fixed as follows:

- primary metric: balanced accuracy on the fixed test set;
- model selection: four-fold stratified cross-validation;
- CV seed: 6011;
- model selection using only the 32 training rows;
- use the test set once, after selecting the winner;
- recompute the classical frontier on exactly the same 32/32 split.

The selector function intentionally has no test argument. It therefore cannot technically use test features or labels. A unit test checks this boundary.

## Which models were tried?

Within the training data, we compared the following models on the 405 hardware features:

- ridge classification;
- logistic regression with L2 regularisation;
- linear support-vector classification;
- RBF support-vector classification.

Regularisation strengths and a small number of RBF gamma values formed a modest grid. Standardisation was relearned inside each CV fold. Selection ranked mean balanced accuracy first, followed by the worst fold, dispersion and finally simplicity.

The winner was an RBF SVC with `C=1` and `gamma=0.01`. Fold scores were 0.625, 0.375, 0.750 and 0.625. The mean was 0.59375, but the standard deviation of 0.136 and worst fold of 0.375 show that the signal was unstable.

## Why balanced accuracy?

The full candidate population contains more regulatory than memory cells, although the pilot split is perfectly balanced. Balanced accuracy first calculates sensitivity for each class and then averages them:

```math
\operatorname{BA}=\frac{1}{2}(\operatorname{TPR}+\operatorname{TNR}).
```

On our balanced test set this equals ordinary accuracy numerically, but the definition remains useful when future splits are not perfectly equal.

## The classical frontier

A quantum route should not be compared only with a weak model using the same compressed input. We therefore used two classical representations:

1. all 32,738 raw genes, normalised per cell to a library size of 10,000 and then transformed with `log1p`;
2. exactly the same four hashed B=4 blocks, flattened to 160 classical features.

Model selection again used training CV only. The classical winner was logistic regression on the raw-gene `log1p` matrix with `C=0.01`. This gives the classical model access to information that the top-48 and hashing steps may have removed on the quantum side. It is strict but relevant: a practical advantage claim must compete with the best reasonable classical route to the same biological prediction.

## Making overfitting visible

After fitting on all 32 training cells, both the selected hardware classifier and the classical model obtained a training score of 1.0. Both could perfectly separate the small training set. Their CV scores were much lower, and the fixed test scores lower still.

That difference is the central lesson, not a detail. A model can find impressive geometry in its training features while failing to learn a stable rule for new cells. This agrees with the broader QML literature: expressivity and trainability do not guarantee generalisation.

## What the uncertainty analysis measures

We compare the two predictors pairwise on the same 32 test cells. An exact McNemar test examines cells for which only one model is correct. We also resample test cells within each class 10,000 times to obtain an interval for the accuracy difference.

This bootstrap measures uncertainty from the small test sample. It does not include:

- a new hardware measurement with independent shots;
- calibration drift on another day;
- a new train/test split;
- uncertainty caused by selecting the model grid;
- biological variation between donors.

In [part 6](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/result-hardware-versus-classical/) we reveal the fixed test score and discuss what 16 versus 17 actually means.

## Sources

- [Local hardware analysis with training-only selection](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis.py)
- [QOS paper: LS-SVM classification task](https://arxiv.org/abs/2604.07639)
