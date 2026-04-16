# Official QOS Repro Status 2026-04-15

## Environment

- Official repo clone: `/mnt/c/Users/Lenna/SynologyDrive/qlab/ML_adv/official_qos`
- JAX venv: `/home/bram/.venvs/qos-paper`
- Verified import baseline: `jax 0.8.1`, `jaxlib 0.8.1`

## Synthetic Benchmark

Completed from official repo root with:

```bash
MPLCONFIGDIR=/tmp/mpl-qos-paper /home/bram/.venvs/qos-paper/bin/python benchmark.py
```

Generated artifacts:

- `benchmark_flat_vector.pdf`
- `benchmark_general_vector.pdf`
- `benchmark_boolean_function.pdf`
- `benchmark_matrix_element.pdf`
- `benchmark_matrix_row_index.pdf`

## Real Datasets Completed

### Splice

Completed:

- `splice_svm.py`
- `splice_pca.py`
- `splice_combine_fig.py`

Artifacts:

- `splice_size_vs_accuracy.json/.pdf`
- `splice_size_vs_variance.json/.pdf`
- `splice_size_vs_accuracy_and_variance.pdf`

Key points:

- `min_samples=1`: accuracy `0.8593`, quantum space `40`, streaming `4064`, sparse `82858`
- `min_samples=1`: PCA variance recovery ~`1.0`, quantum space `38`

### 20 Newsgroups

Completed:

- `20news_svm.py --n_pairs 5`
- `20news_pca.py --n_pairs 5`
- `20news_combine_fig.py`
- `20news_svm.py --n_pairs 100`
- `20news_pca.py --n_pairs 100`
- `20news_combine_fig.py` rerun on the full `100`-pair artifacts

Artifacts:

- `20newsgroups_size_vs_accuracy.json/.pdf`
- `20newsgroups_size_vs_variance.json/.pdf`
- `20newsgroups_combine.pdf`

Key full-run points:

- `min_df=2`: SVM accuracy `0.93631`, streaming `11036.31`, sparse `108537.05`, quantum `45.15`
- `min_df=100`: SVM accuracy `0.83373`, streaming `107.17`, sparse `17588.77`, quantum `36.64`
- `min_df=2`: PCA variance recovery `0.99614`, streaming `11373.76`, sparse `113623.44`, quantum `43.27`
- `min_df=100`: PCA variance recovery `0.56618`, streaming `115.53`, sparse `19044.69`, quantum `36.26`

### PBMC68k

Completed:

- `pbmc68k_svm.py`
- `pbmc68k_pca.py`
- `pbmc68k_combine_fig.py`

Artifacts:

- `pbmc68k_size_vs_accuracy.json/.pdf`
- `pbmc68k_size_vs_variance.json/.pdf`
- `pbmc68k_combine.pdf`

Key points:

- `min_samples=1`: accuracy `0.90146`, quantum space `49.8`, streaming `16122`, sparse `1443697.03`
- `min_samples=20000`: accuracy `0.81100`, quantum space `45.64`, streaming `74`, sparse `554839.08`
- `min_samples=1`: PCA variance recovery `1.00000014`, quantum space `54.0`
- `min_samples=20000`: PCA variance recovery `0.92029`, quantum space `54.0`

### IMDb

Completed:

- `imdb_svm.py`
- `imdb_pca.py`
- `imdb_combine_fig.py`

Artifacts:

- `imdb_size_vs_accuracy.json/.pdf`
- `imdb_size_vs_variance.json/.pdf`
- `imdb_combine.pdf`

Key points:

- `min_df=2`: accuracy `0.891`, quantum space `55.0`, streaming `59535`, sparse `4392452`
- `min_df=5000`: accuracy `0.7139`, quantum space `51.0`, streaming `79`, sparse `740450`
- `min_df=2`: PCA variance recovery `0.99998187`, quantum space `53.0`
- `min_df=5000`: PCA variance recovery `0.76397827`, quantum space `51.0`

## Large-Route Comparison Snapshot

The cleanest large-route comparison is now PBMC68k plus IMDb:

- PBMC68k keeps strong official quality while staying in the small quantum-space regime:
  - `min_samples=1`: SVM accuracy `0.90146`, streaming `16122.0`, sparse `1443697.03`, quantum `49.8`
  - `min_samples=1`: PCA variance recovery `1.00000014`, streaming `16122`, sparse `7111339`, quantum `54.0`
  - `min_samples=20000`: SVM accuracy `0.81100`, streaming `74.0`, sparse `554839.08`, quantum `45.64`
  - `min_samples=20000`: PCA variance recovery `0.92029`, streaming `74`, sparse `2740930`, quantum `54.0`

- IMDb shows the same space-gap pattern on a very different modality:
  - `min_df=2`: SVM accuracy `0.891`, streaming `59535`, sparse `4392452`, quantum `55.0`
  - `min_df=2`: PCA variance recovery `0.99998187`, streaming `59535`, sparse `4392452`, quantum `53.0`
  - `min_df=5000`: SVM accuracy `0.7139`, streaming `79`, sparse `740450`, quantum `51.0`
  - `min_df=5000`: PCA variance recovery `0.76397827`, streaming `79`, sparse `740450`, quantum `51.0`

Interpretation:

- PBMC68k and IMDb now give two official large-route references with the same core pattern:
  explicit classical streaming / sparse space stays large while the paper-model quantum
  workspace stays in the mid-40s to mid-50s.
- This is the right level for the main claim: a model-based large-dataset space gap,
  not a literal end-to-end claim that the full dataset is physically stored in that many qubits.

## Local Compatibility Patch

PBMC68k was not runnable on this host with the upstream loader because normalization
densified the full matrix before `log1p`. A local compatibility patch keeps sparse
input sparse by applying `np.log1p` directly to `X.data`.

Patched file:

- `official_qos/real_datasets/pbmc68k_utils.py`

This is a host-memory compatibility patch, not a conceptual algorithm change.

IMDb also needed a local host-path compatibility improvement:

- `official_qos/real_datasets/imdb_utils.py` now honors `OFFICIAL_QOS_IMDB_DATA_ROOT`
  and prefers the explicit local cache root before relative paths
- `official_qos/real_datasets/imdb_svm.py` and `official_qos/real_datasets/imdb_pca.py`
  now emit JSONL progress logs so long runs are auditable on this host

## Ready Reference Before Qiskit Port

What now counts as stable reference:

- Official synthetic benchmark: done
- Official Splice path: done
- Official PBMC68k path: done, with sparse normalization patch
- Official IMDb path: done, with local-cache preference and progress logging
- Official 20NG path: full `20news_svm.py --n_pairs 100` and `20news_pca.py --n_pairs 100` done

Recommended next order:

1. Keep PBMC68k, IMDb, and now full 20NG as the fixed official JAX references
2. Start or extend a faithful Qiskit port on the smallest controlled target, preferably Splice or the already-bounded IMDb/PBMC bridges
3. Use the official JSON outputs above as the comparison baseline for the Qiskit port

## Qiskit Port Bootstrap

Important implementation note:

- The official real-dataset scripts do not call the QOS kernels directly.
- They report the paper's space/accuracy or space/variance curves.
- So the correct first Qiskit port is kernel-level parity against `official_qos/qos_sampling.py`, before wiring anything to Splice or PBMC.

Current Qiskit bootstrap status:

- `qiskit==1.4.5` and `qiskit-aer==0.16.4` installed in `/home/bram/.venvs/qos-paper`
- first small Qiskit kernel port added in `qiskit_official_qos_sampling_port.py`
- verified parity on two official sampling kernels at `dim=16`, `num_samples=64`:
  - boolean oracle sketch: max abs error vs JAX `5.66e-16`
  - flat state sketch: max abs error vs JAX `1.99e-16`, state infidelity `8.88e-16`
- first saved artifact: `qiskit_official_qos_sampling_port_dim16_m64.json`
- general state-sketch parity now also checked at `dim=8`, `num_samples=64`, `degree=4`:
  - max abs error vs JAX `5.03e-17`
  - raw l2 error vs JAX `8.52e-17`
  - normalized state infidelity vs JAX ~ `0`
- extended saved artifact: `qiskit_official_qos_sampling_port_dim8_m64_deg4.json`
- first narrow Splice bridge added in `qiskit_official_qos_splice_bridge.py`
- first Splice bridge run completed at `min_samples=20`, `bridge_dim=8`, `num_samples=64`
  - selected a real mean-difference vector from the official Splice loader
  - general-state parity stayed tight: max abs error vs JAX `3.89e-16`, raw l2 `4.47e-16`
  - note: this first top-|mean diff| slice was all one sign, so the boolean/flat views are degenerate there; the useful result is the general-state parity
- first saved Splice bridge artifact: `qiskit_official_qos_splice_bridge_m20_d8_m64.json`
- Splice bridge selection rule then refined to a balanced signed slice (`4` negative + `4` positive features at `bridge_dim=8`)
  - selection mode: `balanced_signed_top_abs`
  - boolean and flat kernels are now non-degenerate on real Splice data
  - general-state parity remains tight: max abs error vs JAX `2.22e-16`, raw l2 `3.09e-16`
- first classifier-facing Splice proof added in `qiskit_official_qos_splice_classifier_proof.py`
  - `128/128`, `min_samples=12`, effective bridge `8`: raw baseline `0.7344` ridge / `0.6953` logistic, quantum-feature classifier `0.4375` for both, with many zero rows
  - `128/128`, `min_samples=8`, effective bridge `16`: raw baseline `0.75`, quantum-feature classifier `0.5391` ridge / `0.5313` logistic
  - `128/128`, `min_samples=5`, effective bridge `32`: raw baseline `0.6875` ridge / `0.7422` logistic, quantum-feature classifier `0.5625` ridge / `0.5313` logistic
  - interpretation: widening the real-data slice reduces zero-row collapse and improves the quantum-feature proof, but it is still below the raw selected-feature baseline
- readout/head upgrade added to the Splice proof runner:
  - explicit quantum feature views: `raw`, `abs`, `sq`, `raw_abs`, `raw_sq`, `all`
  - extra heads: `LinearSVC`, `kNN-3`, `NearestCentroid`
  - then extended again with prototype-style heads: `cosine_proto` and `corr_proto`
  - on the strongest current Splice setting (`128/128`, `min_samples=5`, effective bridge `32`), the best quantum-side result now comes from `abs(q_state)` with a cosine prototype head:
    - quantum `0.7344`
    - raw baseline cosine prototype on the same selected raw features is `0.7891`
    - this is a substantial improvement over the earlier quantum ridge/logistic proof (`0.5625` / `0.5313`) on the same bridge
  - saved artifacts: `qiskit_official_qos_splice_classifier_proof_m5_d32_abs_128x128.json`, `qiskit_official_qos_splice_classifier_proof_m5_d32_rawabs_128x128.json`

Recommended next Qiskit step:

1. If we stay on Splice, keep iterating on feature/readout design; the first nontrivial wins came from `abs(q_state)` and prototype-style heads rather than widening alone.
2. Otherwise, keep the current scripts as the minimal faithful Qiskit bridge and pivot back to another dataset or kernel target.

## Large-Route Qiskit Bridge

- first official PBMC68k bridge added in `qiskit_official_qos_pbmc_bridge.py`
- this bridge reports three things in one artifact:
  - the filtered PBMC68k real-data shape and class balance
  - pair-specific paper-space metrics (`streaming`, `sparse`, `quantum`)
  - Qiskit-vs-JAX kernel parity on a small selected slice
- first run completed at `min_samples=1`, `bridge_dim=32`, `num_samples=64`
  - binary top-2 PBMC classes selected by the official loader: `CD8+ Cytotoxic T` vs `CD8+/CD45RA+ Naive Cytotoxic`
  - pair-specific space metrics on the filtered binary matrix:
    - streaming `14156`
    - sparse `3861314`
    - quantum `52.0`
  - official averaged PBMC SVM curve reference at the same `min_samples=1`:
    - accuracy mean `0.90146`
    - streaming `16122.0`
    - sparse `1443697.03`
    - quantum `49.8`
  - general-state parity on the selected `32`-gene slice remains tight:
    - max abs error vs JAX `1.94e-16`
    - raw l2 error `3.25e-16`
- saved artifact: `qiskit_official_qos_pbmc_bridge_m1_d32_m64.json`
- first bounded PBMC68k classifier proof added in `qiskit_official_qos_pbmc_classifier_proof.py`
  - official top-2 PBMC binary task, `256/256`, `min_samples=1`, effective bridge `32`
  - official curve reference at the same `min_samples=1` remains:
    - accuracy mean `0.90146`
    - streaming `16122.0`
    - sparse `1443697.03`
    - quantum `49.8`
  - bounded proof comparison on the selected `32`-gene slice:
    - best raw baseline `0.7266` (`logistic`)
    - initial `abs` quantum view reached `0.5820` (`LinearSVC`)
    - improved `raw_abs` quantum view reaches `0.6211` (`LinearSVC`)
    - zero-row collapse is gone here (`0` train, `0` test), so the remaining gap is not caused by empty quantum rows
- saved artifacts: `qiskit_official_qos_pbmc_classifier_proof_m1_d32_256x256_abs.json`, `qiskit_official_qos_pbmc_classifier_proof_m1_d32_256x256_rawabs.json`

## IMDb Qiskit Bridge

- first official IMDb bridge added in `qiskit_official_qos_imdb_bridge.py`
- this bridge reports:
  - the filtered IMDb real-data shape and class balance
  - pair-specific paper-space metrics at a chosen `min_df`
  - official curve references for both accuracy and variance at that same `min_df`
  - Qiskit-vs-JAX kernel parity on a selected real-data feature slice
- first run completed at `min_df=2`, `bridge_dim=32`, `num_samples=64`
  - binary sentiment task with balanced classes: `25000` negative / `25000` positive
  - pair-specific space metrics on the filtered TF-IDF matrix:
    - streaming `59535`
    - sparse `4392452`
    - quantum `55.0`
  - official IMDb curve references at the same `min_df=2`:
    - accuracy mean `0.891`
    - variance recovery `0.99998187`
    - streaming `59535.0`
    - sparse `4392452.0`
    - quantum `55.0` on the SVM curve and `53.0` on the PCA curve
  - selected bridge slice uses `balanced_signed_top_abs` over `32` real IMDb features
  - general-state parity on that slice remains tight:
    - max abs error vs JAX `1.53e-16`
    - raw l2 error `2.81e-16`
    - normalized state infidelity vs JAX `2.22e-16`
- saved artifact: `qiskit_official_qos_imdb_bridge_d32_m64_min2.json`
- first bounded IMDb classifier proof added in `qiskit_official_qos_imdb_classifier_proof.py`
  - `min_df=2`, `256/256`, effective bridge `32`, `quantum_feature_view=raw_abs`
  - official curve references at the same `min_df=2` remain:
    - accuracy mean `0.891`
    - variance recovery `0.99998187`
    - streaming `59535.0`
    - sparse `4392452.0`
    - quantum `55.0` on the SVM curve and `53.0` on the PCA curve
  - bounded proof comparison on the selected `32`-feature slice:
    - best raw baseline `0.6445` (`centroid`)
    - best quantum-feature classifier `0.5547` (`ridge`)
    - quantum zero-row collapse is low but not zero (`3` train, `3` test)
- saved artifact: `qiskit_official_qos_imdb_classifier_proof_min2_d32_256x256_rawabs.json`
- small IMDb bridge-width sweep on `quantum_feature_view=abs`:
  - `d=8`: best quantum `0.5820` (`linearsvc` / `logistic`), slightly above the best raw baseline on the same `8`-feature slice (`0.5703`)
  - `d=16`: best quantum `0.5586`, below the best raw baseline `0.6211`
  - `d=32`: best quantum `0.5586`, below the best raw baseline `0.6445`
  - `d=64`: best quantum `0.4883`, below the best raw baseline `0.6445`
  - `d=128`: best quantum `0.5547`, below the best raw baseline `0.7188`
  - interpretation: IMDb currently has a narrow small-bridge sweet spot at `d=8`; widening the bridge does not monotonically help
- saved artifacts: `qiskit_official_qos_imdb_classifier_proof_min2_d8_256x256_abs.json`, `qiskit_official_qos_imdb_classifier_proof_min2_d16_256x256_abs.json`, `qiskit_official_qos_imdb_classifier_proof_min2_d64_256x256_abs.json`, `qiskit_official_qos_imdb_classifier_proof_min2_d128_256x256_abs.json`

## 20 Newsgroups Qiskit Bridge

- first official 20NG bridge added in `qiskit_official_qos_20news_bridge.py`
- this bridge reports:
  - the filtered 20NG real-data shape and class balance for one fixed binary pair
  - pair-specific paper-space metrics at a chosen `min_df`
  - official full `100`-pair curve references for both accuracy and variance at that same `min_df`
  - Qiskit-vs-JAX kernel parity on a selected real-data feature slice
- first run completed on the fixed pair `talk.politics.mideast` vs `sci.crypt` at `min_df=2`, `bridge_dim=16`, `num_samples=64`
  - binary pair class balance: `991` vs `940`
  - pair-specific space metrics on the filtered TF-IDF matrix:
    - streaming `14034`
    - sparse `168515`
    - quantum `45.0`
  - official full `100`-pair 20NG references at the same `min_df=2`:
    - SVM accuracy mean `0.93631`
    - PCA variance recovery `0.99614`
    - streaming means `11036.31` on the SVM curve and `11373.76` on the PCA curve
    - sparse means `108537.05` on the SVM curve and `113623.44` on the PCA curve
    - quantum means `45.15` on the SVM curve and `43.27` on the PCA curve
  - selected bridge slice uses `balanced_signed_top_abs` over `16` real TF-IDF features
  - general-state parity on that slice remains tight:
    - max abs error vs JAX `1.11e-16`
    - raw l2 error `2.48e-16`
    - normalized state infidelity vs JAX `0.0`
- saved artifact: `qiskit_official_qos_20news_bridge_mindf2_d16_m64_mideast_crypt.json`
- first bounded 20NG classifier proof added in `qiskit_official_qos_20news_classifier_proof.py`
  - fixed pair `talk.politics.mideast` vs `sci.crypt`, `min_df=2`, `256/256`, effective bridge `16`, `quantum_feature_view=abs`
  - official full `100`-pair references at the same `min_df=2` remain:
    - SVM accuracy mean `0.93631`
    - PCA variance recovery `0.99614`
    - streaming means `11036.31` on the SVM curve and `11373.76` on the PCA curve
    - sparse means `108537.05` on the SVM curve and `113623.44` on the PCA curve
    - quantum means `45.15` on the SVM curve and `43.27` on the PCA curve
  - bounded proof comparison on the selected `16`-feature slice:
    - best raw baseline `0.7969` (`centroid`, `corr_proto`, `cosine_proto`)
    - best quantum-feature classifier `0.7695` (`logistic`)
    - quantum zero-row collapse is substantial here (`73` train, `87` test), so the remaining gap is likely still feature-map sparsity rather than head quality
- saved artifact: `qiskit_official_qos_20news_classifier_proof_mindf2_d16_256x256_abs_mideast_crypt.json`
- small 20NG bridge-width sweep on `quantum_feature_view=abs`:
  - `d=32`: best quantum `0.8008` (`corr_proto` / `cosine_proto`), but raw also rises to `0.8516`
  - `d=64`: best quantum `0.7422` (`cosine_proto`), while raw rises to `0.8594`
  - zero-row collapse does shrink as the bridge widens:
    - `d=16`: `73` train / `87` test
    - `d=32`: `49` train / `58` test
    - `d=64`: `28` train / `43` test
  - interpretation: wider bridges reduce zero rows, but on this pair they do not improve the bounded quantum-vs-raw comparison; `d=16` remains the cleanest current bounded 20NG point
- saved artifacts: `qiskit_official_qos_20news_classifier_proof_mindf2_d32_256x256_abs_mideast_crypt.json`, `qiskit_official_qos_20news_classifier_proof_mindf2_d64_256x256_abs_mideast_crypt.json`

## IBM Hardware Payload Split

- two-path split is now explicit:
  - `Path A: keep QOS alive` by continuing with bounded hardware-feasible QOS routes
  - `Path B: escalate IBM Runtime payload issue` with a concrete reproducible handoff

### Path A: Keep QOS Alive

- the IBM hardware route is healthy for truly small submits:
  - a minimal Bell-state Runtime smoke on `ibm_marrakesh` completed successfully
  - saved artifact: `ibm_bell_smoke_ibm_marrakesh.json`
- the QOS route itself is not fundamentally broken on hardware:
  - a bounded `q=20` PBMC run with `feature_mapping_limit=2` and `query_batch_size=1` completed end-to-end on IBM hardware
  - this keeps every sketch/query submit at exactly `2` measured circuits
  - result on that bounded route:
    - quantum test accuracy `0.50`
    - hashed ridge `0.75`
    - hashed `LinearSVC` `0.75`
    - readout feature count `2`
    - query batches `8`
  - saved artifact: `qiskit_qos_pbmc68k_pairwise_quantum_q20_hw_4x4_ibm_marrakesh_minisketch2_qb1.json`
- implication:
  - the project can continue on real IBM hardware, but only in a constrained QOS regime where per-submit payloads stay very small
  - next live-QOS work should stay on this bounded path unless the payload issue is resolved

### Path B: IBM Runtime Payload Issue

- the problem is now narrowed to a reproducible payload boundary on `ibm_marrakesh`
- working submits:
  - Bell-state smoke: `1` tiny circuit, success
  - bounded mini-sketch with `feature_mapping_limit=2`:
    - sketch submit with `2` measured circuits: success
    - query submits with `2` measured circuits each: success when `query_batch_size=1`
- failing submits:
  - full `q=20` sketch with `98` measured circuits: blocked
  - still blocked after `submit_batch_size=32`
  - still blocked after `submit_batch_size=8`
  - still blocked after `submit_batch_size=4`
  - mini-sketch query submit with `4` measured circuits: also blocked
- implication:
  - this is not a generic IBM auth/backend failure
  - not caused by DD/twirling
  - not fixed by naive batch splitting alone
  - the current best hypothesis is a payload-structure threshold tied to the transpiled QOS circuits themselves

### Immediate Next Actions

1. keep hardware QOS work alive on the bounded mini-sketch path (`feature_mapping_limit=2`, `query_batch_size=1`)
2. hand off the IBM Runtime payload issue with the exact failing/succeeding boundaries
3. only widen the live hardware QOS path after the payload issue is understood or avoided
