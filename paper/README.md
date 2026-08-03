# Paper Draft

Files:

- `main.tex`
- `references.bib`

Compile from this directory with:

```powershell
./build-paper.ps1
```

Equivalent manual sequence:

```powershell
pdflatex -interaction=nonstopmode -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
```

The verified PDF path is:

- `build/main.pdf`

This draft is intentionally claim-disciplined:

- bounded IBM hardware evidence only
- matched hashed classical baselines only
- no general quantum advantage claim
- memory claims separated from accuracy claims
- `q=40` reported as the mean of two `16/16` repeats

Current quantitative backbone:

- `q=10`: `0.50` quantum, `0.50` ridge, `0.50` SVC
- `q=20`: `0.625` quantum, `0.25` ridge, `0.25` SVC
- `q=40` mean over two `16/16` repeats: `0.50` quantum, `0.46875` ridge, `0.50` SVC
- companion simulator example on IMDb sentiment:
  - `d=8`: quantum `0.58203125` vs best raw baseline `0.5703125`

Current stable hardware route:

- backend `ibm_fez`
- `QISKIT_QOS_LAYOUT_STRATEGY=none`
- `QISKIT_QOS_RUNTIME_SUBMIT_BATCH_SIZE=1`
- `QISKIT_QOS_FEATURE_MAPPING_LIMIT=2`
