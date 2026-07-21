# ML Adv Toy: Qiskit QOS Surrogate

This folder contains a small Qiskit toy model for the ideas in arXiv:2604.07639, not a reproduction of the full paper implementation.

What it demonstrates:
- One-pass streaming updates: each sample is processed once and discarded.
- A compact quantum sketch on a small number of qubits.
- A compact classical readout extracted from the sketch state.
- Multiple execution paths for that readout: exact statevector, shot-based simulator, and IBM hardware.
- Two tasks:
  - Binary classification from a label-weighted sketch.
  - 1D reduction from a guide-weighted sketch that approximates `Sigma g`.

What it does not demonstrate:
- The full `quantum-oracle-sketching` JAX codebase.
- The formal lower bounds via NOPE / Forrelation.
- The paper's exact interferometric classical shadow protocol.

The point is to make the paper's streaming-memory idea tangible in the existing qlab Qiskit venv.

## 60-Qubit Fire Opal Hardware Milestone

The latest milestone, retrieved on 21 July 2026, is a real end-to-end run of a
new 60-qubit PBMC68k feature map on `ibm_fez` through Fire Opal. Unlike the
earlier 60-qubit hash experiment, this route uses 60 label-free coexpression
modules learned from a separate 512-cell pool. Four statistics per module are
scaled on training rows only and encoded into a shallow `6 x 10` circuit.

The frozen seed-11 sentinel used:

- PBMC68k, `CD4+/CD25 T Reg` versus `CD4+/CD45RO+ Memory`
- a `32` train / `32` held-out test split
- logical depth `20` with `134` two-qubit interactions per base circuit
- `192` measured circuits on `ibm_fez` through Fire Opal
- `128` shots per circuit (`24,576` requested shots total)
- `627` recovered one- and two-qubit Pauli features per cell
- Fire Opal action `2335848`

The predeclared training-only analysis produced:

- 60q hardware training-CV balanced accuracy: `0.59375`
- 60q hardware held-out balanced accuracy: `0.53125` (`17/32`)
- matched raw-gene linear SVC: `0.50000` (`16/32`)
- matched raw-gene RBF SVC: `0.43750` (`14/32`)
- exact McNemar `p=1.0` against the stronger linear baseline
- stratified paired-bootstrap 95% interval: `[-0.1875, 0.25]`

The full Fire Opal submission-to-retrieval interval was about 8 minutes 33
seconds. The Fire Opal dashboard reported only **26 quantum seconds**, an
especially short QPU task for 192 circuits on 60 qubits. By comparison, the
local MPS convergence attempt was stopped after 42 minutes 57 seconds after
completing only the `chi=64` probe and one `chi=128` sample. The archived
`get_result` payload omitted the quantum-seconds field, so the 26-second value
is explicitly attributed to the dashboard rather than the retrieval JSON.

This is a measured **local time-to-feature-generation advantage** for the
declared resources and target. The 26-second quantum task is more than `99.1x`
faster than the incomplete 2,577-second MPS attempt; even the complete
513-second submission-to-retrieval route is more than `5.0x` faster. The
measured quantum features also gave the best held-out point score. The MPS
route did not converge, so these ratios are lower bounds for producing the same
specified 60q feature target, not a matched-numerical-error comparison. This is
not an end-to-end advantage over the inexpensive classical classifiers, and
the 32-cell test is too small for a general predictive or asymptotic claim.

The earlier 40q milestone remains the historical anchor: hardware scored
`16/32` versus `17/32` for its matched classical frontier and used 26
provider-reported QPU seconds.

Relevant 60q entry points:

- `qiskit_qos_pbmc68k_q60_module_pipeline.py`: label-free modules, frozen
  splits, local simulation gate, and hardware analysis
- `qiskit_qos_pbmc68k_q60_module_fireopal_validate.py`: local and provider
  validate-only gate
- `qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py`: guarded plan, one-shot
  submission, retrieval, ordering checks, and quantum-time accounting
- `Q60_MODULE_B4_RUNBOOK.md`: exact study protocol and claim boundaries

Generated provider payloads and raw result JSON remain local and are
intentionally excluded from Git. The repository tracks the runners, validation
logic, tests, and the numerical result summary above.

The theory and hardware article series has one combined
[Edukaizen project page](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/)
with Dutch and English language buttons. The language-specific article URLs,
including the existing [English index](https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/),
remain available.

The 60q result is also registered as a local runtime lower bound in the
[Pro Student Quantum Advantage List](https://edukaizen.nl/pro-student-quantum-advantage-list/).
The Markdown sources and publication records live in `articles/qos_edukaizen/`
and `articles/qos_edukaizen_en/`.

The synthetic datasets are deliberately block-structured: the raw feature dimension can be much larger than the qubit count, but the useful signal lives in a small number of coarse modes. That keeps the toy honest to the "small machine on large classical data" story without pretending to prove the paper's separation.

## Memory Accounting

Accuracy is not the only metric here. For the paper-style claim, the relevant quantity is often model/workspace size.

This repo now keeps three separate notions of memory:
- `quantum logical memory`: the number of logical qubits `q`
- `quantum classical sidecar memory`: encoder parameters, sketch sidecar arrays, readout model features, and the small learned head
- `classical memory`: the weight vector or training workspace for a linear baseline

Important distinction:
- `statevector` memory is only simulator memory and scales like `16 * 2^q` bytes for `complex128`
- hardware memory claims should not use that statevector number

To post-process an existing scaling summary into explicit memory estimates:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_memory_report.py \
    --scaling-json qiskit_qos_scaling_20ng_atheism_space_sim_q10_q14_q20_8x8_64.json \
    --json-out qiskit_qos_memory_report_20ng_atheism_space_8x8_64.json
```

What the report contains:
- `effective_feature_dim`: the feature width actually benchmarked by the toy
- `raw_feature_dim`: the pre-SVD width when available, which is closer to the paper-style high-dimensional input story
- `quantum total model bytes actual`: current implementation cost, including a dense stored compressor matrix
- `quantum total model bytes conceptual`: fairer block-streaming estimate for `--encoder block`
- `classical effective/raw model bytes`: linear model size on the effective or raw feature width
- `classical effective/raw ridge training workspace bytes`: a lower-bound proxy for the Gram solve
- `statevector_bytes`: simulator-only reference, not a hardware claim

This matters on text datasets like `20ng-atheism-vs-space`: the current toy already uses a classical `TF-IDF -> SVD(256)` front-end, so the effective classical dimension is only `256`, while the raw sparse vocabulary width is `20000`. If you compare only against the reduced `256`-dimensional baseline, much of the paper-style memory separation has already been compressed away classically.

## Stronger Classical Baselines

To compare the toy against stronger and more memory-aware classical baselines on the same split, use the separate runner:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_classical_benchmark.py \
    --source 20ng-atheism-vs-space \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --quantum-scaling-json qiskit_qos_scaling_20ng_atheism_space_q10_q14_32x32.json \
    --quantum-qubits 14 \
    --json-out qiskit_qos_classical_benchmark_20ng_atheism_vs_space_32x32.json
```

This benchmarks:
- the original manual ridge baseline on `SVD(256)`
- stronger dense baselines on `SVD(256)` such as `RidgeClassifier`, `LogisticRegression`, and `LinearSVC`
- stronger sparse baselines on raw `TF-IDF`
- a memory-bounded classical reference: `HashingVectorizer + SGDClassifier`

This benchmark is intentionally separate from the quantum scaling runner so earlier artifacts stay unchanged.

To test a richer quantum readout on the same scaling source without changing earlier artifacts:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_scaling_runner.py \
    --source 20ng-atheism-vs-space \
    --qubits 14 \
    --encoder block \
    --quantum-head ridge \
    --readout-family all-pairs \
    --readout-shots 128 \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --json-out qiskit_qos_scaling_20ng_atheism_space_q14_32x32_all_pairs.json
```

This keeps the circuit ansatz unchanged and only enriches the observable family used for sketch/query readout.

## Run

From this folder:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py --plot
```

Exact readout instead of shot-noisy readout:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py --readout-shots 0 --plot
```

Larger raw feature space with the same small sketch:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py --n-features 64 --num-qubits 5 --plot
```

CSV with labels:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py \
    --csv /absolute/path/data.csv \
    --label-col label \
    --num-qubits 4 \
    --plot
```

Notes for CSV mode:
- The label column must be binary.
- By default the script uses all fully numeric columns except the label as features.
- If needed, pass `--feature-cols f1,f2,f3`.
- Train/test split is controlled by `--csv-train-fraction` and defaults to `0.67`.
- You can switch the pre-sketch encoder with `--encoder block`, `--encoder pca`, `--encoder ridge`, or `--encoder lda`.
- `ridge` and `lda` are supervised binary encoders; the reduction demo falls back to `block`.
- You can switch the post-readout head with `--quantum-head cosine`, `--quantum-head ridge`, or `--quantum-head logistic`.
- `cosine` is the original scalar similarity score; `ridge` and `logistic` train a small binary classifier on per-sample quantum readout features.
- You can switch the observable family with `--readout-family local` or `--readout-family all-pairs`.
- `local` uses per-qubit `X/Y/Z` plus nearest-neighbor `XX/ZZ`; `all-pairs` keeps the single-qubit terms and expands pair correlators to every qubit pair.
- You can switch the execution path with `--execution-mode statevector`, `--execution-mode sampler-sim`, or `--execution-mode ibm-hardware`.
- For `sampler-sim` and `ibm-hardware`, `--readout-shots` must be greater than `0`.
- For larger nearest-neighbor ladders under `sampler-sim`, prefer `--simulator-method matrix_product_state`; the default dense local simulation becomes impractical well before `q=40`.
- For hardware smoke runs, use `--max-train-samples` and `--max-test-samples` aggressively; the number of executed circuits scales like `samples * O(num_qubits)`.
- Hardware mitigation controls:
  - `--readout-mitigation --cal-shots N` fits a local-tensored asymmetric readout model from all-zero and all-one calibration circuits.
  - `--extra-error-suppression --dd-sequence XY4 --twirl-randomizations 8` enables IBM Runtime dynamical decoupling and gate twirling.

Known binary benchmark: Breast Cancer Wisconsin

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py \
    --csv breast_cancer_wisconsin.csv \
    --label-col label \
    --num-qubits 6 \
    --quantum-head ridge \
    --plot \
    --plot-out breast_cancer_qos_head_ridge_summary.png \
    --label-plot-out breast_cancer_qos_head_ridge_label_weight.png \
    --json-out breast_cancer_qos_head_ridge_summary.json
```

This uses the standard scikit-learn breast cancer dataset:
- 569 samples
- 30 numeric features
- binary label: `malignant` vs `benign`
- For the encoder itself, `block` compression remained best on this dataset.
- For the quantum head, the trainable heads were much stronger than raw cosine similarity.
- Measured with `512` readout shots and `block` encoder:
  - `4 qubits`: `cosine 0.729`, `ridge 0.947`, `logistic 0.947`
  - `6 qubits`: `cosine 0.729`, `ridge 0.968`, `logistic 0.957`
- Full encoder comparison artifact: `breast_cancer_encoder_comparison.json`
- Full head comparison artifact: `breast_cancer_quantum_head_comparison.json`
- Best measured run in this toy: `block` encoder + `ridge` quantum head + `6` qubits, matching the classical baseline test accuracy on this split.
- On this standardized dataset, `ridge` and `lda` collapsed to nearly the same first direction, so they produced effectively identical scores.

Execution-mode smoke check on the same guarded Breast Cancer subset (`6` qubits, `block`, `ridge` head, `64/32` sample cap for sim; `4/4` for hardware):
- `statevector`, `256` shots: quantum test acc `0.938`
- `sampler-sim`, `256` shots: quantum test acc `0.969`
- `ibm-hardware` on `ibm_kingston`, `128` shots, `4` train + `4` test: quantum test acc `0.500`
- Smoke comparison artifact: `breast_cancer_execution_mode_smoke.json`
- IBM hardware smoke artifact: `breast_cancer_ibm_smoke_summary.json`

IBM mitigation smoke on the same guarded `4/4` hardware subset:
- With `--readout-mitigation --cal-shots 512 --extra-error-suppression --dd-sequence XY4 --twirl-randomizations 8`, the run completed cleanly on `ibm_kingston`.
- The mitigated smoke still scored `0.500` on this tiny subset, so the dominant bottleneck at this scale is sample starvation, not just raw device noise.
- Mitigated hardware artifact: `breast_cancer_ibm_mitigated_smoke_summary_v4.json`
- Hardware comparison artifact: `breast_cancer_ibm_smoke_comparison.json`
- The mitigated run records calibration job ids and fitted per-qubit `p01/p10` values in the execution metadata.

Minimal IBM hardware smoke command:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py \
    --csv breast_cancer_wisconsin.csv \
    --label-col label \
    --num-qubits 6 \
    --encoder block \
    --quantum-head ridge \
    --execution-mode ibm-hardware \
    --backend-name ibm_kingston \
    --readout-shots 128 \
    --max-train-samples 4 \
    --max-test-samples 4 \
    --json-out breast_cancer_ibm_smoke_summary.json
```

Mitigated IBM hardware smoke command:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_toy_model.py \
    --csv breast_cancer_wisconsin.csv \
    --label-col label \
    --num-qubits 6 \
    --encoder block \
    --quantum-head ridge \
    --execution-mode ibm-hardware \
    --backend-name ibm_kingston \
    --readout-shots 128 \
    --readout-mitigation \
    --cal-shots 512 \
    --extra-error-suppression \
    --dd-sequence XY4 \
    --twirl-randomizations 8 \
    --max-train-samples 4 \
    --max-test-samples 4 \
    --json-out breast_cancer_ibm_mitigated_smoke_summary_v4.json
```

Current limitation:
- The shot-based path removes the `Statevector` bottleneck, but the number of measurement circuits still grows with `number_of_samples * number_of_local_observables`.
- So this is now hardware-compatible and no longer blocked at `2^q`, but it is not yet a production-grade `60` logical-qubit implementation of the paper claim.

Separate scaling extension:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_scaling_runner.py \
    --source digits-even-vs-odd \
    --qubits 6,10,14 \
    --readout-shots 128 \
    --encoder block \
    --quantum-head ridge \
    --execution-mode sampler-sim
```

This uses a separate runner and leaves `qiskit_qos_toy_model.py` intact.

Built-in larger sources:
- `digits-even-vs-odd`: 1797 samples, 64 raw features.
- `20ng-atheism-vs-space`: 20 Newsgroups text -> TF-IDF + TruncatedSVD dense features.
- `20ng-graphics-vs-baseball`: second text option with the same pipeline.
- `20ng-custom`: any two 20 Newsgroups categories via `--20ng-categories cat_a,cat_b`.

Useful scaling notes:
- For larger qubit sweeps, prefer `--execution-mode sampler-sim` first.
- The scaling runner accepts the same mitigation flags for hardware spot-checks.
- For text sources, it vectorizes first and then reuses the numeric Qiskit toy classifier.
- To raise raw classical memory on a real text source, prefer `--tfidf-analyzer char_wb` with wider n-grams and a larger `--tfidf-max-features`.
- Outputs are written to separate `qiskit_qos_scaling_*.json` and `qiskit_qos_scaling_*.png` artifacts.

Real-data high-memory text smoke with a broader raw feature space:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_scaling_runner.py \
    --source 20ng-custom \
    --20ng-categories rec.autos,sci.med \
    --qubits 10 \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --readout-shots 128 \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --tfidf-analyzer char_wb \
    --tfidf-ngram-min 3 \
    --tfidf-ngram-max 5 \
    --tfidf-max-features 120000 \
    --tfidf-min-df 2 \
    --svd-components 512
```

Measured smoke on that setting:
- Source: `20ng-rec.autos-vs-sci.med`
- Raw feature dim: `87133`
- Reduced feature dim: `512`
- `q=10`: quantum test `0.656`, old classical scaling baseline `0.562`

Matching classical benchmark on the same split:
- `sgd_log_hashing_4096`: `0.844` at `32.01 KB`
- `linearsvc_raw_tfidf`: `0.781` at `61.17 KB`
- `logreg_svd256`: `0.750` at `4.01 KB`

Multi-pair 20NG pilot, closer to the paper's "many category pairs" idea:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_20ng_pair_runner.py \
    --qubits 10 \
    --n-pairs 3 \
    --pair-seed 17 \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --readout-shots 128 \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --tfidf-analyzer char_wb \
    --tfidf-ngram-min 3 \
    --tfidf-ngram-max 5 \
    --tfidf-max-features 120000 \
    --tfidf-min-df 2 \
    --svd-components 512 \
    --hash-features 4096
```

Measured pilot:
- `q=10`: quantum `0.562 ± 0.051`
- old scaling baseline: `0.594 ± 0.102`
- hashing baseline: `0.760 ± 0.029`
- mean raw feature dim across pairs: `98,286`

Text workbook / tweets:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_text_runner.py \
    --input "/mnt/c/Users/Lenna/OneDrive - Universiteit Utrecht/corpora en tools/onderzoek/sarcasme_tweets (1).xlsx" \
    --sheet Sheet1 \
    --text-col Text \
    --label-pattern "#not" \
    --strip-literals "#sarcasme,#not" \
    --balance-classes \
    --num-qubits 6 \
    --plot-prefix sarcasme_not_proxy
```

Important:
- This workbook does not appear to contain a direct binary sarcasm/non-sarcasm label column.
- The command above therefore defines a proxy task: tweets with explicit `#not` versus sarcastic tweets without that marker.
- Because `#not` itself would otherwise trivially leak the target, it is stripped from the text before vectorization.

Political tweets: positive versus negative spectrum from category labels:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_text_runner.py \
    --input "/mnt/c/Users/Lenna/OneDrive - Universiteit Utrecht/corpora en tools/onderzoek/output/workflow_322_overzicht/04_output/tabel_322_tweets_scores_en_categorie.xlsx" \
    --sheet 322_tweets \
    --text-col verkorte_tekst \
    --label-col evaluatieve_categorie \
    --positive-labels positief \
    --negative-labels "licht negatief,zwaar negatief,impliciet kritisch" \
    --num-qubits 6 \
    --plot-prefix politieke_pos_neg
```

Interpretation for that run:
- `positief` is mapped to `+1`.
- `licht negatief`, `zwaar negatief`, and `impliciet kritisch` are mapped to `-1`.
- `neutraal` and `informatie` are excluded from the binary classification set.

Political tweets: 3-way grouped model:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_text_runner.py \
    --input "/mnt/c/Users/Lenna/OneDrive - Universiteit Utrecht/corpora en tools/onderzoek/output/workflow_322_overzicht/04_output/tabel_322_tweets_scores_en_categorie.xlsx" \
    --sheet 322_tweets \
    --text-col verkorte_tekst \
    --label-col evaluatieve_categorie \
    --class-groups "positief=positief;negatief=licht negatief|zwaar negatief|impliciet kritisch;overig=neutraal|informatie" \
    --num-qubits 6 \
    --plot-prefix politieke_3weg
```

Political tweets: 6-way model on all categories:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_text_runner.py \
    --input "/mnt/c/Users/Lenna/OneDrive - Universiteit Utrecht/corpora en tools/onderzoek/output/workflow_322_overzicht/04_output/tabel_322_tweets_scores_en_categorie.xlsx" \
    --sheet 322_tweets \
    --text-col verkorte_tekst \
    --label-col evaluatieve_categorie \
    --class-labels "positief,licht negatief,zwaar negatief,impliciet kritisch,neutraal,informatie" \
    --num-qubits 6 \
    --plot-prefix politieke_6weg
```

Accuracy-memory frontier on the `20ng-atheism-vs-space` scaling source, reusing an existing quantum scaling artifact:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_memory_frontier_runner.py \
    --source 20ng-atheism-vs-space \
    --max-train-samples 16 \
    --max-test-samples 16 \
    --quantum-scaling-json qiskit_qos_scaling_20ng_atheism_space_q20_q40_q60_16x16_32_mps.json \
    --quantum-qubits 20,40,60
```

Synthetic astronomical source with implicit raw dimension `2^60`, first sanity-tested at `q=10`:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_astronomical_runner.py \
    --raw-log2-dim 60 \
    --qubits 10 \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --readout-shots 32 \
    --n-samples 512 \
    --max-train-samples 128 \
    --max-test-samples 128
```

Real sparse high-dimensional source from the paper family: UCI Dorothea.

Important caveat:
- Dorothea is heavily imbalanced in raw form (`112` positive vs `1038` negative after merging train+valid).
- For local QOS accuracy checks, use `--dorothea-balance` so the reported accuracy stays interpretable.

First staged Dorothea sweep, `q=10` then `q=20`, on a balanced `32/32` split:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_scaling_runner.py \
    --source dorothea-uci \
    --dorothea-cache-dir data_cache/dorothea \
    --dorothea-balance \
    --qubits 10,20 \
    --readout-shots 32 \
    --seed 7 \
    --train-fraction 0.67 \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --svd-components 256
```

Measured staged results on that balanced split:
- `q=10`: quantum `0.406`, old scaling baseline `0.594`
- `q=20`: quantum `0.625`, old scaling baseline `0.594`
- raw feature dim: `100000`
- reduced feature dim after SVD on the balanced subset: `223`

Matching classical memory sweep on the exact same split:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_dorothea_memory_sweep.py \
    --dorothea-cache-dir data_cache/dorothea \
    --dorothea-balance \
    --seed 7 \
    --train-fraction 0.67 \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --svd-components 256 \
    --quantum-scaling-json qiskit_qos_scaling_dorothea_balanced_q10_q20_32x32_32shots.json \
    --quantum-qubits 10,20
```

Measured Dorothea memory-sweep result:
- `q=10` quantum target `0.406` is matched by `chi2_linearsvc_k64` at `776 B`
- `q=20` quantum target `0.625` is also matched by `chi2_linearsvc_k64` at `776 B`
- stronger classical points reach `0.8125` at about `12 KB`

Split-aware supervised preselection before the quantum pipeline:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_dorothea_chi2_quantum_runner.py \
    --dorothea-cache-dir data_cache/dorothea \
    --dorothea-balance \
    --runs 64:10,64:20,128:20 \
    --readout-shots 32 \
    --seed 7 \
    --train-fraction 0.67 \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --max-train-samples 32 \
    --max-test-samples 32
```

Measured split-aware Dorothea chi2 results:
- `k=64, q=10`: quantum `0.656`, selected-feature ridge `0.531`, selected-feature `LinearSVC` `0.625`
- `k=64, q=20`: quantum `0.500`, selected-feature ridge `0.531`, selected-feature `LinearSVC` `0.625`
- `k=128, q=20`: quantum `0.562`, selected-feature ridge `0.750`, selected-feature `LinearSVC` `0.750`

Interpretation:
- The main gain comes from train-only supervised feature selection before the quantum map.
- On this split, `k=64, q=10` is the best of the tested quantum points and clearly beats the earlier Dorothea quantum baseline.
- Simply increasing qubits to `q=20` did not help here; the quality of the selected feature subspace matters more than raw qubit count.

Real genomic k-mer route: binary Splice Junction (`EI` vs `IE`) with explicit ambient `4^k` accounting.

This route is useful when the goal is to reason about very large *ambient* classical model memory, because the canonical dense k-mer space grows as `4^k`.

First `k=19` smoke:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_splice_kmer_runner.py \
    --k 19 \
    --binary \
    --min-samples 1 \
    --runs 64:10,64:20 \
    --readout-shots 32 \
    --seed 7 \
    --train-fraction 0.67 \
    --encoder ridge \
    --quantum-head ridge \
    --readout-family local \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --max-train-samples 64 \
    --max-test-samples 64
```

Measured `Splice k=19` result:
- ambient feature dim: `274,877,906,944`
- ambient dense classical weight memory: `2.00 TB`
- observed feature dim on the actual dataset: `52,652`
- observed dense classical weight memory: `411.34 KB`
- `chi2 k=64, q=10`: quantum `0.609`, selected-feature ridge `0.422`, selected-feature `LinearSVC` `0.578`
- `chi2 k=64, q=20`: quantum `0.547`, selected-feature ridge `0.422`, selected-feature `LinearSVC` `0.578`

Important nuance:
- the `2.00 TB` figure is the dense ambient `4^19` memory of a canonical full k-mer linear model;
- the actually observed vocabulary on this dataset is much smaller, so a sparse or selected-feature classical model does **not** automatically need terabytes here.
- This makes `Splice` a good route for explicit ambient-memory accounting, but not yet proof that the *minimum* classical memory to match quantum is terabyte-scale.

Interpretation:
- Dorothea is a better paper-near source than the earlier synthetic `2^60` surrogate because the raw classical feature space is genuinely large.
- On the current QOS surrogate, it still does not show a memory advantage yet: a tiny classical selected-feature model already matches the `q=20` quantum accuracy.
- So Dorothea is now integrated and usable, but not yet the empirical memory-gap case we are looking for.

PBMC68k real single-cell source:

The repo now also supports the official 10x PBMC68k count matrix plus the
official barcode annotations from the 10x paper repo. A first classical
difficulty screen over several binary cell-type pairs is available via:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_pbmc68k_pair_screen.py \
    --cache-dir data_cache/pbmc68k \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --json-out qiskit_qos_pbmc68k_pair_screen_64x64.json \
    --plot-out qiskit_qos_pbmc68k_pair_screen_64x64.png
```

Measured first PBMC68k pair-screen result:
- hardest screened pair: `CD4+/CD25 T Reg` vs `CD4+/CD45RO+ Memory`
- best comfortable classical score on the plain-gene screen: `0.734`
- smallest model reaching that score: `chi2_logreg_k1024` at `12.01 KB`

To push the classical ambient feature space into the GB regime before spending
quantum time, use the hashed gene-pair screen on that harder PBMC pair:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_pbmc68k_pairwise_screen.py \
    --cache-dir data_cache/pbmc68k \
    --positive-label 'CD4+/CD25 T Reg' \
    --negative-label 'CD4+/CD45RO+ Memory' \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --pairwise-budgets 256,1024,4096,16384,65536 \
    --max-active-genes 256 \
    --value-mode log-product \
    --json-out qiskit_qos_pbmc68k_pairwise_screen_64x64.json \
    --plot-out qiskit_qos_pbmc68k_pairwise_screen_64x64.png
```

Measured first PBMC68k pairwise result:
- pairwise ambient dense weight proxy: `3.99 GB`
- pairwise ambient dense projector proxy: `q20 79.85 GB`, `q40 159.70 GB`, `q60 239.55 GB`
- best comfortable classical pairwise score in this first screen: `0.609`
- smallest model reaching that score: `pairhash_linearsvc_d256` at `2.01 KB`

Interpretation:
- PBMC68k pairwise interactions are a cleaner path to a genuinely large classical ambient space than `Splice`.
- The first pairwise comfort screen is now materially harder than the plain-gene PBMC screen, but it still does not by itself prove a GB-scale *minimum* classical memory requirement.
- The next quantum step should therefore use this harder PBMC pairwise route locally before any hardware attempt.

Finer single-cell source with subcluster labels:

The repo now also supports the finer-label `SingleCellMultiModal::pbmc_10x`
RNA source. This uses:
- the official `pbmc_rna_tenx.h5` RNA counts from the `pbmc_10x` multiome resource
- the official `pbmc_colData.rda` sidecar with `celltype` and `broad_celltype`

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_pbmc10x_subcluster_screen.py \
    --cache-dir data_cache/pbmc10x_subclusters \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --json-out qiskit_qos_pbmc10x_subcluster_screen_64x64.json \
    --plot-out qiskit_qos_pbmc10x_subcluster_screen_64x64.png
```

Measured first finer-label PBMC10x screen:
- source size: `10,032` cells, `36,549` genes, `19,324,570` nonzeros
- fine labels include `naive CD4 T cells`, `memory CD4 T cells`, `intermediate monocytes`, `MAIT T cells`, `plasmacytoid DC`
- hardest screened pair in this first pass: `naive B cells` vs `memory B cells`
- best comfortable classical score on that pair: `0.953125`
- smallest model reaching that score: `chi2_complementnb_k1024` at `20.02 KB`

Interpretation:
- finer subcluster labels alone are **not** enough here to make the source hard for classical baselines;
- several closely related subtype pairs are still solved very cheaply;
- this makes `pbmc_10x` useful as a fine-label real-data source, but not yet the hard memory-gap source we are looking for.

Subtler single-cell perturbation source: `GSE132080`

The repo now also supports the official GEO `GSE132080` Perturb-seq source from
Jost et al. 2020, using:
- the official `GSE132080_10X_matrix.mtx.gz`, `GSE132080_10X_barcodes.tsv.gz`, and `GSE132080_10X_genes.tsv.gz`
- the official `GSE132080_cell_identities.csv.gz` cell metadata
- the official `GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz` guide activity sidecar

This source is screened differently from the PBMC runs: instead of broad lineage
or `stim`/`ctrl` labels, it compares two guides targeting the **same gene** with
a small activity gap.

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_gse132080_guide_screen.py \
    --cache-dir data_cache/gse132080 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --json-out qiskit_qos_gse132080_guide_screen_64x64.json \
    --plot-out qiskit_qos_gse132080_guide_screen_64x64.png
```

Measured first `GSE132080` guide-vs-guide screen:
- source size: `23,608` cells, `33,694` genes, `112,033,131` nonzeros
- screened `7` subtle within-gene guide pairs with day-10 activity deltas around `0.016` to `0.038`
- hardest pair in this first pass: `POLR1D_+_28196016.23-P1_08` vs `POLR1D_+_28196016.23-P1_00`
- best comfortable classical score on that pair: `0.5625`
- smallest model reaching that score: `chi2_linearsvc_k128` at `1.51 KB`

Interpretation:
- this is materially harder than the earlier PBMC subtype and perturbation screens;
- the hardest first-pass pairs now push comfortable classical baselines down into the `0.56` to `0.61` range instead of `0.95+`;
- this makes `GSE132080` the current best real single-cell candidate before spending new quantum time.

Bounded third-order escalation on the hard `GSE132080` pair:

If you want a real-data test that actually reaches a `TB`-class ambient
classical space, the next step is not more raw genes but higher-order
interactions. The repo now has a separate classical-only third-order runner for
the hard `POLR1D` guide pair:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_gse132080_thirdorder_screen.py \
    --cache-dir data_cache/gse132080 \
    --positive-guide POLR1D_+_28196016.23-P1_08 \
    --negative-guide POLR1D_+_28196016.23-P1_00 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --max-active-genes 48 \
    --thirdorder-budgets 256,1024,4096,16384,65536 \
    --json-out qiskit_qos_gse132080_thirdorder_screen_64x64.json \
    --plot-out qiskit_qos_gse132080_thirdorder_screen_64x64.png
```

Measured first `GSE132080` third-order result:
- ambient third-order feature dimension: `6,374,818,071,644`
- ambient dense classical weight proxy: `46.38 TB`
- dense projector proxy: `q20 927.66 TB`, `q40 1.81 PB`, `q60 2.72 PB`
- best comfortable classical score across the tested hashed budgets: `0.53125`
- smallest model reaching that score: `thirdhash_linearsvc_d65536` at `512.01 KB`

Interpretation:
- this is the first real-data screen in this repo where the ambient classical proxy is unambiguously `TB`-class;
- even after lifting to third-order interactions, the tested comfortable classical models still do **not** become strong;
- this makes the hard `POLR1D` pair the cleanest current candidate for a first quantum run on a real-data source with a genuine `TB`-class classical ambient comparison.

First local quantum screen on the hard `GSE132080` third-order route:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_gse132080_thirdorder_quantum_runner.py \
    --cache-dir data_cache/gse132080 \
    --positive-guide POLR1D_+_28196016.23-P1_08 \
    --negative-guide POLR1D_+_28196016.23-P1_00 \
    --qubits 10,20 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --max-active-genes 48 \
    --readout-shots 32 \
    --query-batch-size 8 \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --json-out qiskit_qos_gse132080_thirdorder_quantum_q10_q20_64x64.json \
    --plot-out qiskit_qos_gse132080_thirdorder_quantum_q10_q20_64x64.png
```

Measured first `GSE132080` third-order quantum result:
- ambient dense classical weight proxy remains `46.38 TB`
- `q=10`: quantum `0.453125`, hashed ridge `0.515625`, hashed `LinearSVC` `0.5`
- `q=20`: quantum `0.390625`, hashed ridge `0.453125`, hashed `LinearSVC` `0.4375`
- avoided dense encoder matrix: `463.83 TB` at `q=10`, `927.66 TB` at `q=20`

Interpretation:
- on this first local quantum pass, the `TB`-class ambient comparison is finally there, but the quantum side is not yet competitive;
- `q=10` and `q=20` both underperform the bounded hashed classical baselines on the same split;
- this means the source is promising as a hard classical ambient benchmark, but the current quantum encoding/head is not yet good enough to justify pushing straight to `q=40`.

Bounded encoding-tuning follow-up on the same `q=10` route:
- tried signed repeated hashing on the host-side third-order encoder with `repeats in {2,4}` and `activation_scale in {0.5,1.0,2.0}`
- best quantum point in this sweep: `repeats=2`, `activation_scale=2.0`, quantum `0.546875`
- matching hashed baselines on that same encoding still stayed higher: ridge `0.59375`, `LinearSVC` `0.578125`

Interpretation:
- the third-order encoding is indeed tunable, and it can improve the quantum side materially from `0.453125` to `0.546875`;
- but on this source and split, encoding tweaks alone still did **not** push quantum above the bounded classical baselines.

Apples-to-apples signed classical sweep on the same hard third-order source:
- reran the `POLR1D` third-order classical screen with the same signed hashing family used in the tuned quantum host-side encoder
- extended the hashed classical budgets up to `1,048,576`
- best comfortable classical point: `thirdhash_logreg_d1024`
- accuracy `0.59375`
- model size `8.01 KB`

Interpretation:
- the ambient classical comparison is still `TB`-class, but the *comfortable hashed* classical route does **not** need large explicit memory here to become competitive;
- so the honest claim on this source is the large ambient classical space, not a proven large *minimum* classical memory requirement.

Semi-synthetic hardening on the same real `GSE132080` source:

If the natural guide labels still leave a small classical shortcut, the next
step is not another biological labelset but a semi-synthetic task on the same
real cells. The repo now supports a residualized hidden-teacher route:
- build a large hidden third-order teacher on the real `POLR1D` cells
- explicitly project out the original guide-label direction
- explicitly project out a smaller hashed classical shortcut
- then threshold the residual teacher score into a balanced binary label

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_gse132080_semisynth_screen.py \
    --cache-dir data_cache/gse132080 \
    --positive-guide POLR1D_+_28196016.23-P1_08 \
    --negative-guide POLR1D_+_28196016.23-P1_00 \
    --teacher-dim 65536 \
    --shortcut-dim 4096 \
    --feature-dims 256,1024,4096,16384,65536 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --hash-repeats 2 \
    --signed-hash \
    --activation-scale 2.0 \
    --json-out qiskit_qos_gse132080_semisynth_screen_64x64.json \
    --plot-out qiskit_qos_gse132080_semisynth_screen_64x64.png
```

Measured first `GSE132080` semi-synthetic result:
- source ambient third-order proxy remains `46.38 TB`
- hidden teacher dimension: `65536`
- small hashed shortcut projected out during label construction: `4096`
- shortcut projection `R^2`: `0.9999992`
- smallest best classical point in the post-residual task: `d=16384`
- best comfortable classical accuracy: `0.515625`
- model size at that point: `128.01 KB`
- key low-budget points:
  `d=256 -> 0.484375`, `d=1024 -> 0.46875`, `d=4096 -> 0.5`

Interpretation:
- this is the first route in the repo where the small hashed classical shortcut is intentionally removed on a real-data source;
- the semi-synthetic task behaves the way we wanted: `d=1024` and even `d=4096` stay at or near chance;
- this is therefore a much better candidate for a quantum follow-up than the natural-label `POLR1D` task.

Broadened classical upper bound on the same semi-synthetic task:
- reran the semi-synthetic screen with `ridge` added and budgets extended to `262144` and `1048576`
- best classical score in the broader sweep: `0.53125`
- smallest budget reaching that score: `thirdhash_ridge_d65536`
- model size at that point: `512.01 KB`
- larger budgets did not improve further:
  `d=262144 -> 0.53125`, `d=1048576 -> 0.53125`

Interpretation:
- even after adding the missing `ridge` family and pushing the budget into MB-scale hashed models, the broad classical upper bound still stays below the first `q=10` quantum point;
- that makes the semi-synthetic task a much cleaner quantum-vs-classical comparison than the natural-label route.

First local quantum screen on the semi-synthetic `GSE132080` task:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_gse132080_semisynth_quantum_runner.py \
    --cache-dir data_cache/gse132080 \
    --positive-guide POLR1D_+_28196016.23-P1_08 \
    --negative-guide POLR1D_+_28196016.23-P1_00 \
    --teacher-dim 65536 \
    --shortcut-dim 4096 \
    --qubits 10 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --hash-repeats 2 \
    --signed-hash \
    --activation-scale 2.0 \
    --readout-shots 32 \
    --query-batch-size 8 \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --json-out qiskit_qos_gse132080_semisynth_quantum_q10_64x64.json \
    --plot-out qiskit_qos_gse132080_semisynth_quantum_q10_64x64.png
```

Measured first `GSE132080` semi-synthetic quantum result:
- ambient dense classical proxy remains `46.38 TB`
- `q=10`: quantum `0.578125`
- matching small hashed classical baselines on the same split:
  ridge `0.546875`, `LinearSVC` `0.546875`
- sketch state: `152 B`
- avoided dense encoder matrix: `463.83 TB`

Interpretation:
- this is the first local run in the repo where the quantum side on a real-data-derived hard task moves above the small hashed classical baselines;
- the gain is not huge, but it is finally in the right direction on the right kind of task;
- if we continue on this route, the next clean step is `q=20` on the same semi-synthetic labels, not another dataset search.

Follow-up `q=20` screen on the same semi-synthetic labels:
- `q=20`: quantum `0.5`
- matching small hashed classical baselines in the q-run:
  ridge `0.65625`, `LinearSVC` `0.65625`
- avoided dense encoder matrix: `927.66 TB`

Interpretation:
- `q=20` does **not** help on the current direct encoder; it is materially worse than `q=10`;
- on this route, `q=10` is therefore the best quantum point so far, while the broad classical upper bound remains `0.53125`.

First local quantum screen on the harder PBMC pairwise route:

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_pbmc68k_pairwise_quantum_runner.py \
    --cache-dir data_cache/pbmc68k \
    --positive-label 'CD4+/CD25 T Reg' \
    --negative-label 'CD4+/CD45RO+ Memory' \
    --qubits 10 \
    --max-train-samples 64 \
    --max-test-samples 64 \
    --max-active-genes 256 \
    --value-mode log-product \
    --readout-shots 32 \
    --query-batch-size 8 \
    --quantum-head ridge \
    --readout-family local \
    --execution-mode sampler-sim \
    --simulator-method matrix_product_state \
    --json-out qiskit_qos_pbmc68k_pairwise_quantum_q10_64x64.json \
    --plot-out qiskit_qos_pbmc68k_pairwise_quantum_q10_64x64.png
```

Measured first PBMC pairwise quantum result:
- `q=10`: quantum `0.562`, hashed ridge `0.531`, hashed `LinearSVC` `0.578`
- pairwise ambient dense classical weight proxy: `3.99 GB`
- avoided dense `q x D_pairwise` encoder matrix at `q=10`: `39.93 GB`
- query batching used: `16` logical batches at `8` samples per batch

Interpretation:
- the raw pairwise source is now in a real GB-class ambient regime;
- but the first `q=10` quantum point is not yet above the best comfortable classical pairwise baseline on the same split;
- the next sensible step is `q=20` on this same route, not hardware yet.

Streaming k-mer growth probe:

Use this before any hardware step when the real question is whether the *observed* k-mer vocabulary keeps growing fast enough to justify a very large classical model.

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_kmer_growth_probe.py \
    --source splice-openml \
    --k 20 \
    --mode both \
    --shard-size 128 \
    --limit-sequences 512 \
    --json-out qiskit_qos_kmer_growth_probe_splice_k20_512.json \
    --plot-out qiskit_qos_kmer_growth_probe_splice_k20_512.png
```

Key idea:
- `ambient_dense_weight_bytes` is the canonical full `4^k` dense k-mer model size.
- `exact` and `HLL` rows tell you how the *observed* vocabulary grows with more sequences.
- If that growth curve flattens early, then this dataset is a bad candidate for a terabyte-class *minimum* classical memory claim.
- for chromosome-style FASTA shards, use `--chunk-bases` so one long contig becomes many growth points instead of one giant sequence.

Hash-streaming genomics runner:

Use this when the goal is not more corpus accounting but an IBM-ready architecture:
the host streams raw k-mers, keeps only a compact `q`-bin hash encoding plus the
small `WeightedStreamingSketch`, and avoids the dense `num_qubits x 4^k`
encoder matrix entirely.

```bash
../quantum-math-lab/scripts/run-in-qiskit-venv.sh \
  python qiskit_qos_hash_streaming_genomics_runner.py \
    --source splice-openml \
    --k 20 \
    --qubits 10,20 \
    --hash-repeats 2 \
    --readout-shots 32 \
    --max-train-samples 32 \
    --max-test-samples 32 \
    --json-out qiskit_qos_hash_streaming_genomics_k20.json \
    --plot-out qiskit_qos_hash_streaming_genomics_k20.png
```

Key idea:
- raw sequences are hashed directly into `q` bins with deterministic signed updates
- the quantum sketch state stays `O(q)` on the host
- the output reports the dense encoder-matrix bytes avoided relative to the generic toy path
- this is the right stepping stone before any IBM hardware run on large genomics shards
- for IBM hardware at higher `q`, use `--query-batch-size` to split the query phase into smaller submit batches

## Files

- `qiskit_qos_toy_model.py`
  - Main standalone script.
  - Uses Qiskit circuits plus `Statevector` readout.
  - Writes a JSON summary and, with `--plot`, both a general PNG summary and a dedicated label-weight diagnostics PNG.
- `qiskit_qos_text_runner.py`
  - Text-data wrapper around the same Qiskit sketch classifier.
  - Converts text to TF-IDF + SVD features, then runs the streaming sketch model.
- `qiskit_qos_memory_frontier_runner.py`
  - Separate accuracy-vs-memory comparison runner.
  - Combines quantum points from an existing scaling artifact with explicit classical memory-budget sweeps.
- `qiskit_qos_astronomical_runner.py`
  - Separate synthetic source for implicit raw dimensions like `2^60`.
  - Reports quantum accuracy together with a dense raw-classical memory proxy that can become astronomical.
- `qiskit_qos_dorothea_utils.py`
  - UCI Dorothea download/cache and sparse parser utilities.
  - Also supports a balanced subset path for accuracy-oriented local sweeps.
- `qiskit_qos_dorothea_memory_sweep.py`
  - Separate Dorothea memory-budget runner on raw sparse features.
  - Reports the minimum classical selector+model memory that matches a target quantum accuracy.
- `qiskit_qos_dorothea_chi2_quantum_runner.py`
  - Split-aware Dorothea experiment with train-only `chi2` feature selection before the quantum pipeline.
  - Useful when the goal is to improve the quantum score without label leakage.
- `qiskit_qos_splice_kmer_utils.py`
  - Splice Junction sequence loader and observed/ambient k-mer feature accounting.
- `qiskit_qos_splice_kmer_runner.py`
  - Split-aware genomics k-mer benchmark with explicit `4^k` ambient memory reporting.
- `qiskit_qos_hash_streaming_genomics_runner.py`
  - Matrix-free k-mer benchmark that streams raw sequences into a compact `q`-bin hash encoding before the Qiskit sketch/readout path.

## Interpretation

The classification toy uses a single signed streaming sketch with label weights `y in {-1, +1}`.

Its final decision uses a threshold calibrated from the train scores, not a hardcoded zero cut. That matters because the toy readout is compact but not centered by construction.

The reduction toy uses a guide vector `g` and streams weights proportional to `x . g`, so the sketch approximates a compressed, one-pass version of `Sigma g`. That mirrors the paper's use of a guiding vector for dimension reduction, but only at the level of a toy surrogate.

The compact readout is a local-Pauli feature vector taken from the sketch state. That is deliberately smaller and simpler than the paper's full readout machinery.
