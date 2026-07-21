# De theorie van Quantum Oracle Sketching

Quantum Oracle Sketching probeert een oud probleem van quantumalgoritmen te omzeilen: hoe krijgt een quantumcomputer toegang tot een enorme klassieke dataset zonder eerst een even enorm quantumgeheugen of QRAM te bouwen?

De [paper van Zhao en collega’s](https://arxiv.org/abs/2604.07639), geplaatst op 8 april 2026, formuleert een antwoord in het streamingmodel. De officiële repository bevat inmiddels JAX-implementaties, tutorials, QSVT-routines en experimenten met onder andere PBMC68k. De nieuwste publieke hoofdbranch liep bij onze controle tot commit `10c092c` van 21 mei 2026, waarin sparse Johnson–Lindenstrauss-projecties aan de real-data-route waren toegevoegd.

## Het datatoegangsprobleem

Veel quantumalgoritmen veronderstellen een oracle: een coherente operatie die informatie over een vector, matrix of functie toegankelijk maakt in superpositie. Voor klassieke data is zo’n oracle niet gratis. Als de hele dataset eerst in QRAM moet worden geplaatst, kan de geheugenwinst verdwijnen voordat de quantumrekening begint.

QOS draait de volgorde om. Het ontvangt willekeurige klassieke samples één voor één. Ieder sample stuurt een kleine quantumrotatie aan. Door veel van zulke updates op te stapelen ontstaat een quantumkanaal dat een gewenste oracle-operatie benadert. Het sample kan daarna worden weggegooid.

Schematisch:

```text
sample z1 → kleine rotatie ┐
sample z2 → kleine rotatie ├→ compacte quantumsketch → quantumalgoritme
sample z3 → kleine rotatie ┘
```

De quantumtoestand bewaart niet ieder datapunt afzonderlijk. Hij bewaart een coherente samenvatting die geschikt is voor een volgende quantumquery.

## De classificatietheorie

Voor binaire classificatie schrijft de paper de trainingdata als een sparse matrix $X \in \mathbb{R}^{N \times D}$ en labels $y_i \in \{-1,+1\}$. De klassieke referentie is een geregulariseerde least-squares support vector machine, equivalent aan een ridge-achtige lineaire classifier:

```math
w = \operatorname*{argmin}_{w}\;\lVert Xw-y\rVert_2^2 + \lambda\lVert w\rVert_2^2.
```

Een nieuwe featurevector $x'$ krijgt het label $\operatorname{sign}(x' \cdot w)$. QOS bouwt de quantumoracles waarmee een quantum lineair-algebra-algoritme de relevante beslisinformatie kan benaderen zonder de volledige $D$-dimensionale parameterwereld klassiek op te slaan.

Theorem 3 van de paper stelt, onder de formele voorwaarden van het model, dat een quantummachine van grootte `poly(log D)` de classificatietaak kan oplossen met ongeveer lineair veel samples in $N$, terwijl een klassieke machine met grootte $O(D^{0.99})$ dat niet kan. De dynamische variant voegt een scheiding in sample-efficiëntie toe wanneer de datastroom verandert maar de beslisregel ongeveer gelijk blijft.

## Waar zit het voordeel precies?

Het geclaimde voordeel is primair een **ruimte- of machinegroottevoordeel**. De paper vergelijkt logische qubits met klassieke floating-point geheugeneenheden. Voor PBMC68k en andere datasets laat de numerieke studie zien dat de QOS-curve bij minder dan zestig logische qubits een hoge prestatie kan behouden terwijl algemene klassieke streaming- en sparse-matrixroutes veel meer opslag gebruiken.

Dat betekent niet automatisch:

- dat een huidige fysieke QPU sneller is in wandkloktijd;
- dat de dataloading gratis is;
- dat iedere klassieke, domeinspecifieke heuristic is uitgesloten;
- dat zestig lawaaiige fysieke qubits gelijkstaan aan zestig logische qubits;
- dat betere accuracy de definitie van het theoretische voordeel is.

De paper zegt zelf dat de real-datafiguren numerieke experimenten zijn. De implementatie is JAX-simulatie, en dataset-specifieke klassieke heuristieken worden als toekomstig werk genoemd. De asymptotische stelling en de praktische PBMC-grafiek ondersteunen elkaar, maar zijn niet hetzelfde bewijsobject.

## Waarom de Born-regel belangrijk is

Een opvallend onderdeel van de theorie is de kwadratische relatie tussen amplitudes en kansen. De samples sturen kleine unitaires updates aan; de convergentie naar de verwachte oracle-operatie hangt samen met die probabilistische structuur. De paper bewijst dat de vereiste kwadratische sample-scaling voor deze constructie optimaal is.

Daarna zijn QSVT en classical shadows nodig om nuttige functies van vectoren en matrices uit de sketch te berekenen en compact klassiek uit te lezen. Dit is veel meer dan “een datapunt in rotatiehoeken stoppen”. Het volledige theoretische protocol bestaat uit datatoegang, oracle-opbouw, quantum lineaire algebra en gecontroleerde readout.

## De officiële code versus onze hardwarevertaling

De [officiële repository](https://github.com/haimengzhao/quantum-oracle-sketching) bevat twee numerieke routes:

- expliciete random sampling in `qos_sampling.py`;
- een expected-unitary-route in `qos.py` voor efficiëntere benchmarking.

Onze Qiskit-route is doelbewust anders. Wij bouwen een ondiepe, QOS-geïnspireerde featuremap die op huidige hardware past. Daardoor kunnen we de overgang van grote klassieke invoer naar een kleine quantummachine fysiek testen, maar we erven niet automatisch de volledige theorem-3-garantie.

Die scheiding is de belangrijkste regel van de artikelenserie: **de theorie motiveert de route; het hardware-experiment test een beperkte implementatie ervan**.

In [deel 3](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/pbmc68k-van-genexpressie-naar-qubits/) volgen we exact hoe 32.738 genen veranderen in vier blokken van veertig getallen.

## Bronnen

- [QOS-paper, arXiv:2604.07639](https://arxiv.org/abs/2604.07639)
- [Officiële QOS-code en real-data-experimenten](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Laatste gecontroleerde officiële commit](https://github.com/haimengzhao/quantum-oracle-sketching/commit/10c092cefcfdff9951bf5729bd2ffb4c25fe2254)

