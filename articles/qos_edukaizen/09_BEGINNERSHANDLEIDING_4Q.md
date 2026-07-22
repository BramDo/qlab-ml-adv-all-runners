# Beginnershandleiding QML: van UMI-telling naar een 4-qubitcircuit

Hoe verandert een rij met genexpressiedata in een uitvoerbaar quantumcircuit? In deze handleiding volgen we één echte PBMC68k-cel van ruwe UMI-tellingen naar vier rotatiehoeken, een klein Qiskit-circuit, acht quantumfeatures en een klassieke classifier.

Dit is bewust een beginnersmodel. Het draait op een vier-qubitsimulator, gebruikt slechts 16 trainings- en 16 testcellen en maakt **geen quantumvoordeelclaim**. Het doel is de volledige vertaalketen zichtbaar en reproduceerbaar maken.

## 1. De taak: twee typen immuuncellen onderscheiden

De PBMC68k-dataset bevat in onze loader 68.579 cellen en 32.738 genen. Een rij is één cel, een kolom is één gen en iedere matrixwaarde is een molecuultelling. We kiezen twee celtypen:

- `CD4+/CD25 T Reg`;
- `CD4+/CD45RO+ Memory`.

Het gekozen binaire deel bevat 9.248 cellen. Voor het educatieve experiment trekken we met vaste seed 11 acht trainings- en acht testcellen per klasse. Training en test overlappen niet.

De taak is dus niet een gen opzoeken. Zij is:

```text
genexpressieprofiel van een cel -> quantumfeaturemap -> voorspeld celtype
```

## 2. Wat is een UMI-telling?

UMI betekent *Unique Molecular Identifier*. Voor vermenigvuldiging in het laboratorium krijgt ieder opgevangen RNA-molecuul een korte barcode. Reads met dezelfde celbarcode, hetzelfde gen en dezelfde UMI zijn waarschijnlijk kopieën van hetzelfde oorspronkelijke molecuul. Zij tellen daarom samen als één molecuul.

![Van sequencing reads naar twee unieke UMI-tellingen](docs/beginner/assets/umi_counting.png)

Een cel met vijf reads van UMI `ACGT` en vier reads van UMI `TGCA` heeft voor dat gen dus negen reads, maar slechts twee onderscheiden UMI's. De uiteindelijke countmatrix is dunbezet en bevat niet-negatieve gehele getallen. Een nul betekent dat geen molecule is gedetecteerd; het bewijst niet dat het gen biologisch volledig afwezig was.

## 3. Eerst splitsen, daarna keuzes leren

Alle datagedreven keuzes worden uitsluitend op de zestien trainingscellen gemaakt:

1. vier genen selecteren zonder labels te gebruiken;
2. normalisatie- en schaalparameters leren;
3. quantumfeatures voor de training berekenen;
4. de klassieke classifier trainen;
5. pas daarna de zestien afgeschermde testcellen evalueren.

Uit genen met een bruikbare detectiefrequentie selecteert het script de vier grootste trainingsvarianties. Voor seed 11 zijn dat:

| Qubit | Gen | Detectie in training | Trainingsgemiddelde | Trainingsstandaardafwijking |
| ---: | --- | ---: | ---: | ---: |
| 0 | IER2 | 0,6250 | 1,724827 | 1,482075 |
| 1 | ACTG1 | 0,7500 | 2,261162 | 1,377323 |
| 2 | LIMD2 | 0,5625 | 1,427313 | 1,354013 |
| 3 | GLTSCR2 | 0,7500 | 2,266142 | 1,349629 |

Deze volgorde is belangrijk: genen kiezen met kennis van testlabels of de schaal fitten op testwaarden zou datalekken veroorzaken.

## 4. Van ruwe UMI naar een rotatiehoek

Cellen bevatten niet allemaal evenveel gemeten RNA. Daarom delen we de telling voor gen `g` eerst door alle UMI's van cel `i`, schalen naar 10.000 en nemen `log1p`:

```math
n_{ig}=10.000\frac{x_{ig}}{\sum_h x_{ih}},
\qquad
\ell_{ig}=\log(1+n_{ig}).
```

Daarna gebruiken we uitsluitend het trainingsgemiddelde en de trainingsstandaardafwijking:

```math
z_{ig}=\frac{\ell_{ig}-\mu_g^{\mathrm{train}}}
{\sigma_g^{\mathrm{train}}}.
```

De z-score wordt begrensd tot `[-3,3]` en omgezet naar een hoek:

```math
\theta_{ig}=\pi\frac{\operatorname{clip}(z_{ig},-3,3)}{3}.
```

Daardoor ligt iedere hoek tussen `-pi` en `pi`.

De eerste echte trainingscel heeft label `CD4+/CD45RO+ Memory`, bevat in totaal 2.010 UMI's en geeft:

| Gen | Ruwe UMI | `log1p` | z-score | Hoek in radialen |
| --- | ---: | ---: | ---: | ---: |
| IER2 | 0 | 0,000000 | -1,163792 | -1,218720 |
| ACTG1 | 3 | 2,767914 | 0,367925 | 0,385290 |
| LIMD2 | 1 | 1,787605 | 0,266092 | 0,278651 |
| GLTSCR2 | 2 | 2,393362 | 0,094263 | 0,098712 |

![De complete route van UMI-data naar een hybride voorspelling](docs/beginner/assets/qml_pipeline.png)

## 5. Het vier-qubitcircuit

We starten in basisstaat `|0000>`. Iedere hoek bestuurt één `RY`-poort. Daarna verbinden vier CNOT-poorten de qubits in een ring: `0->1`, `1->2`, `2->3` en `3->0`.

![Het vier-qubitcircuit voor de voorbeeldcel](docs/beginner/assets/pbmc68k_q4_first_cell_circuit.png)

De Y-rotatie heeft matrix:

```math
R_Y(\theta)=
\begin{pmatrix}
\cos(\theta/2)&-\sin(\theta/2)\\
\sin(\theta/2)& \cos(\theta/2)
\end{pmatrix}.
```

Voor vier qubits vormt de rotatielaag het tensorproduct:

```math
R=R_Y(\theta_3)\otimes R_Y(\theta_2)\otimes
R_Y(\theta_1)\otimes R_Y(\theta_0).
```

De volledige celafhankelijke bewerking is:

```math
U(\theta)=\operatorname{CNOT}_{3\rightarrow0}
\operatorname{CNOT}_{2\rightarrow3}
\operatorname{CNOT}_{1\rightarrow2}
\operatorname{CNOT}_{0\rightarrow1}R.
```

Vier qubits hebben `2^4 = 16` basisstaten. Daarom is `U` een `16 x 16` matrix. De numerieke controle geeft `||U^dagger U-I||_F = 1,11 x 10^-15`: binnen afrondingsfout is de matrix unitair.

![Absolute waarden van de 16 x 16 unitaire matrix](docs/beginner/assets/pbmc68k_q4_first_cell_unitary_magnitude.png)

## 6. Van toestand naar meetkansen

Het circuit maakt de toestand:

```math
|\psi(\theta)\rangle=U(\theta)|0000\rangle
=\sum_{b=0}^{15}a_b|b\rangle.
```

Volgens de Born-regel is de kans op bitstring `b` gelijk aan:

```math
p_b=|a_b|^2,
\qquad
\sum_b p_b=1.
```

Voor de voorbeeldcel zijn de grootste ideale kansen:

| Basisstaat `q3q2q1q0` | Kans |
| --- | ---: |
| `0000` | 0,633736 |
| `1110` | 0,308730 |
| `1111` | 0,024114 |
| `1101` | 0,012463 |

Qiskit toont bitstrings als `q3 q2 q1 q0`; qubit 0 staat dus rechts.

## 7. Acht quantumfeatures uit bitstrings

Voor een gemeten bit `b_q` gebruiken we de Z-eigenwaarde `(-1)^{b_q}`. De losse en gepaarde verwachtingswaarden zijn:

```math
\langle Z_q\rangle=\sum_b p_b(-1)^{b_q},
```

```math
\langle Z_qZ_r\rangle=\sum_b p_b(-1)^{b_q+b_r}.
```

Per cel krijgen we vier losse Z-features en vier naburige ZZ-features:

```math
f(x)=(\langle Z_0\rangle,\langle Z_1\rangle,
\langle Z_2\rangle,\langle Z_3\rangle,
\langle Z_0Z_1\rangle,\langle Z_1Z_2\rangle,
\langle Z_2Z_3\rangle,\langle Z_3Z_0\rangle).
```

![Ideale quantumfeatures en schattingen uit 512 shots](docs/beginner/assets/quantum_features.png)

| Feature | Ideaal | 512 shots |
| --- | ---: | ---: |
| Z0 | 0,886607 | 0,914063 |
| Z1 | 0,319566 | 0,300781 |
| Z2 | 0,307240 | 0,281250 |
| Z3 | 0,305744 | 0,281250 |
| Z0Z1 | 0,329931 | 0,308594 |
| Z1Z2 | 0,961427 | 0,964844 |
| Z2Z3 | 0,995132 | 1,000000 |
| Z3Z0 | 0,344847 | 0,328125 |

Het verschil tussen ideaal en 512 shots is gewone steekproefruis. Meer shots verkleinen gemiddeld de fout, maar vragen meer circuituitvoeringen.

## 8. Waarom er nog een klassieke classifier nodig is

Het quantumcircuit produceert features; het kiest niet zelf het uiteindelijke label. Een klassieke logistische regressie leert met de zestien trainingslabels:

```math
P(y=+1\mid f)=\sigma(w^Tf+b),
\qquad
\sigma(a)=\frac{1}{1+e^{-a}}.
```

Op de zestien afgeschermde testcellen is het resultaat:

| Model | Correct | Balanced accuracy |
| --- | ---: | ---: |
| 4q quantumfeatures | 7/16 | 0,4375 |
| Klassiek, dezelfde vier genen | 9/16 | 0,5625 |

Dat is een geslaagde reproduceerbaarheids- en onderwijsdemonstratie, maar geen prestatievoordeel. Een goede QML-handleiding moet ook een negatief vergelijkingsresultaat gewoon tonen.

## 9. Zelf uitvoeren

De simulatorroute gebruikt geen IBM- of Fire Opal-quantumtijd:

```bash
/home/bram/.venvs/qiskit/bin/python \
  qiskit_qos_pbmc68k_q4_educational.py \
  --shots 512 \
  --json-out output/pbmc68k_q4_educational.json
```

De tabellen, circuittekening en matrices zijn opnieuw te maken met:

```bash
/home/bram/.venvs/qiskit/bin/python \
  qiskit_qos_pbmc68k_q4_explain.py \
  --shots 512 \
  --output-dir docs/beginner/assets
```

De [volledige broncode en alle vaste artefacten](https://github.com/BramDo/qlab-ml-adv-all-runners/tree/main/docs/beginner) staan op GitHub. Daar is ook een [downloadbare DOCX-handleiding](https://github.com/BramDo/qlab-ml-adv-all-runners/raw/main/docs/beginner/qml-van-umi-naar-circuit.docx) beschikbaar.

## 10. Relatie tot Quantum Oracle Sketching

Dit PBMC68k-beginnersmodel is **geen letterlijke QOS-implementatie**. Het gebruikt vier klassiek voorbereide rotatiehoeken, een vaste CNOT-ring en Z/ZZ-readout. Het is een gewone kleine quantumfeaturemap.

De repository bevat daarnaast een afzonderlijke vier-qubit hardwarepilot die wel de officiële `q_state_sketch_flat` sampling-kern naar Qiskit port. Ook die pilot implementeert slechts één QOS-bouwsteen en niet de volledige QOS/QSVT-classificatieketen.

| Route | Doel | Letterlijke QOS-kern? | Claimgrens |
| --- | --- | --- | --- |
| 4q PBMC68k-beginnersmodel | UMI-data naar circuit en classifier leren vertalen | Nee | Geen advantageclaim |
| 4q flat-QOS-hardwarepilot | Officiële sampling-sketch op hardware testen | Ja, één primitive | Geen volledige QOS-classifier |
| 60q PBMC68k-pilot | Brede real-data hardwarefeaturemap testen | Nee, QOS-geïnspireerd | Alleen afgebakende lokale timingclaim |

## Bronnen

- Kivioja et al., [Counting absolute numbers of molecules using unique molecular identifiers](https://doi.org/10.1038/nmeth.1778), Nature Methods 9, 72-74 (2012).
- Zheng et al., [Massively parallel digital transcriptional profiling of single cells](https://doi.org/10.1038/ncomms14049), Nature Communications 8, 14049 (2017).
- [IBM Quantum-documentatie over de RY-poort](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYGate).
- [IBM Quantum-documentatie over bitvolgorde](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering).
- Havlicek et al., [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2), Nature 567, 209-212 (2019).
- Zhao et al., [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639), arXiv-preprint (2026).
- [10x Genomics PBMC68k countmatrix](https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz).
