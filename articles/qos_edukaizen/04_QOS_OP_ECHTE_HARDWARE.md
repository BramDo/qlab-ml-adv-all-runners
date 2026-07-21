# Van JAX naar een 40-qubit hardwarecircuit

De oorspronkelijke QOS-paper levert JAX-code en numerieke simulaties. Onze volgende vraag was eenvoudiger maar fysiek concreet: kunnen we een herkenbare, streamingachtige single-cell featuremap daadwerkelijk op veertig qubits uitvoeren en er genoeg informatie uit teruglezen voor een classifier?

## Eerst: een letterlijke QOS-bouwsteen op vier qubits

Vóór de brede PBMC68k-route hebben we de officiële flat-QOS sampling-sketch apart naar Qiskit geport. Het 4q-toymodel gebruikt $D=16$ posities en $M=64$ willekeurige samples. De samples bouwen letterlijk dezelfde fasevector als `q_state_sketch_flat`; een extra Hadamardlaag zet de anders onzichtbare fasen om in meetbare interferentie.

Op IBM Fez draaiden 64 willekeurige sketchcircuits plus twee controles via Fire Opal-action `2334156`. De gemiddelde Hellinger-fideliteit met de ideale distributies was 0,990104, de mediaan 0,991417 en het minimum 0,980411. Dit is een letterlijke flat-QOS-kern op hardware. Het is nog geen complete QOS-classifier: QSVT, de lineaire solver en de volledige readoutketen ontbreken.

## Daarna: een hardwaregerichte PBMC68k-vertaling

Het circuit verwerkt vier featureblokken in hetzelfde register. Een blok wordt met enkelqubitrotaties geüpload, waarna qubitparen met een vaste interactiestructuur worden gekoppeld. Na het vierde blok volgt een laatste rotatielaag.

Op logisch circuitniveau heeft iedere cel:

- 40 qubits;
- diepte 20 vóór de metingen;
- 87 twee-qubitpoorten;
- volledig numerieke parameters;
- geen mid-circuitmetingen of resets.

Dat is bewust ondiep. Het volledige QOS/QSVT-protocol uit de theorie vraagt complexere oracles en foutgecorrigeerde logische operaties. Anders dan de 4q-sketch hierboven is dit 40q-circuit **geen letterlijke QOS-implementatie**: de rotatiehoeken worden klassiek voorbereid en er is geen sample-voor-sample oracleconstructie. Deze variant onderzoekt de eerstvolgende experimentele grens: een fysieke quantumfeaturemap die breed genoeg is om klassiek lastiger te simuleren, maar ondiep genoeg om op huidige hardware te overleven.

## Waarom drie meetcircuits per cel?

Een quantumtoestand kan niet in één meting volledig worden uitgelezen. Wij kiezen een vast paneel van homogene Pauli-observabelen. Voor iedere basiscel maken we daarom drie versies:

- alle qubits meten in de X-basis;
- alle qubits meten in de Y-basis;
- alle qubits meten in de Z-basis.

Uit één globale basismeting kunnen meerdere enkelqubit- en twee-qubitcorrelaties worden berekend. Met 64 cellen levert dit:

```text
64 cellen × 3 meetbases = 192 circuits.
```

Ieder circuit kreeg 128 shots. Het totale shotbudget was dus 24.576. De readout omvatte per cel 40 enkelqubitsupports en 95 vooraf vastgelegde qubitpaarsupports, ieder in X, Y en Z. Dat geeft 405 features.

## Eerst valideren, dan pas uitvoeren

Omdat Fire Opal-runs schaars zijn, bestond de workflow uit gescheiden veiligheidsfasen:

1. lokaal de dataset, split, encoding, circuitvorm, QASM en hashes reconstrueren;
2. alle 192 payloads via Fire Opal `validate` controleren zonder uitvoering;
3. een hardwareplan met backend, shots en readout vastleggen;
4. slechts na een expliciete bevestiging één batch indienen;
5. het action-ID opslaan en retrieval nooit laten resubmittereren;
6. de classificatie uitsluitend lokaal uitvoeren nadat het hardwarebestand was gepind.

De provider-validatie accepteerde alle circuits voor `ibm_fez`. De logische diepte was steeds 20 en de geëxporteerde payloads hadden diepte 22 vóór providercompilatie. De validatiewaarschuwingen over calibratie zijn bewaard in plaats van weggefilterd.

## Fire Opal en IBM Fez

[Fire Opal](https://q-ctrl.com/fire-opal) is een performance-managementlaag van Q-CTRL. Zij compileert en optimaliseert circuits voor het gekozen apparaat en voert foutonderdrukking of mitigatiestappen uit. Dat maakt het gemeten resultaat niet foutloos, maar kan de bruikbare algoritmische informatie op lawaaiige hardware verbeteren.

De run kreeg Fire Opal-action-ID `2334162` en gebruikte IBM Fez. De provider rapporteerde 26 quantumseconden. Dat is de QPU-gebruikstijd, niet de volledige doorlooptijd: datavoorbereiding, compilatie, wachtrij, klassieke postprocessing en modelselectie vallen daarbuiten.

## Wat kwam terug?

De retrieval leverde exact 192 kansverdelingen, in dezelfde volgorde als het manifest. Alle waarden waren eindig en niet-negatief. De normalisaties weken maximaal ongeveer $2.2\times 10^{-15}$ van één af. Per circuit kwamen 114 tot 128 verschillende bitstrings voor.

Daaruit reconstrueerden we 64 rijen van ieder 405 verwachtingswaarden. De waarden lagen tussen -0,90625 en 1,0. De bitvolgorde is expliciet Qiskit little-endian: de meest rechtse bit hoort bij qubit nul.

Dat is het belangrijkste hardwaretussenresultaat: niet alleen circuits werden geaccepteerd, maar de hele keten van echte celdata naar bruikbare, geordende quantumfeatures is voltooid.

## Is dit de eerste single-cell QML-run?

Zo breed mogen we het niet formuleren. Er bestonden al quantum-kernelstudies op biologische celclassificatie en echte IBM-hardware, bijvoorbeeld een studie naar neuronale M-typen uit 2023. De verdedigbare nieuwigheid is smaller: een vroege, voor zover wij nu weten eerste fysieke 40-qubit feasibility-uitvoering van deze specifieke **QOS-geïnspireerde coherente PBMC68k-route**.

Zelfs die formulering hoort bij publicatie een systematische literatuurcontrole te krijgen. In deze reeks gebruiken we daarom primair “van JAX naar 40-qubit hardware” en niet “eerste QML ooit”.

In [deel 5](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-readout-405-observabelen-classifier/) bekijken we hoe de 405 features worden gebruikt zonder de testset tijdens modelkeuze te laten lekken.

## Bronnen

- [Officiële QOS-repository: JAX-implementatie](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze letterlijke 4q flat-QOS-pilot](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/main/qiskit_official_qos_flat_fireopal_pilot.py)
- [Fire Opal](https://q-ctrl.com/fire-opal)
- [Onze guarded hardwarepilot](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_pilot.py)
- [Eerdere celclassificatie met quantumkernels op IBM-hardware](https://www.nature.com/articles/s41598-023-38558-z)
