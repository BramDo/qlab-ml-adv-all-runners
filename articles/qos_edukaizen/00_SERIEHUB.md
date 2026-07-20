# Van genexpressie naar 40 qubits

Quantum machine learning klinkt vaak alsof een quantumcomputer een complete database in één keer doorzoekt. Ons experiment doet iets preciezers en interessanters: het probeert **het celtype van één cel te voorspellen uit haar genexpressieprofiel**. De invoer is een lange, dunbezette vector met RNA-tellingen; de uitvoer is een van twee immuuncelklassen.

Deze achtdelige reeks verbindt drie lagen die gemakkelijk door elkaar raken. De eerste laag is de theorie van *Quantum Oracle Sketching* (QOS), gepubliceerd in april 2026. Die theorie gaat over een klein quantummodel dat enorme klassieke datastromen verwerkt zonder de hele matrix te bewaren. De tweede laag is de officiële JAX-code en de numerieke PBMC68k-experimenten. De derde laag is onze eigen, ondiepe en QOS-geïnspireerde vertaling naar een circuit dat werkelijk op 40 fysieke qubits van IBM Fez is uitgevoerd via Fire Opal. Het extra achtste deel beschrijft een mogelijke 60-qubitvervolgstudie, maar stelt die uitvoering bewust uit.

## De reeks

1. [Wat is de QML-taak? Cellen classificeren, geen genen opzoeken](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/wat-is-de-qml-taak-cellen-classificeren/)
2. [De theorie van Quantum Oracle Sketching](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-oracle-sketching-theorie/)
3. [Van PBMC68k-genexpressie naar 40 qubits](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/pbmc68k-van-genexpressie-naar-qubits/)
4. [Van JAX naar een 40-qubit hardwarecircuit](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qos-naar-40-qubit-hardware-fire-opal/)
5. [405 observabelen en een lekvrije classifier](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-readout-405-observabelen-classifier/)
6. [Het resultaat: hardware 16, klassiek 17](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/)
7. [Wat is nog nodig voor quantumvoordeel?](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/route-naar-quantumvoordeel-qml/)
8. [Voorstel: een 60-qubit QML-vervolgstudie](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/voorstel-60-qubit-qml-vervolgstudie/)

## Het experiment in één tabel

| Onderdeel | Waarde |
| --- | --- |
| Dataset | PBMC68k / Zheng68k |
| Volledige invoer per cel | 32.738 genen |
| Binaire taak | regulatoire CD4-T-cel versus CD4-geheugencel |
| Pilot | 32 training- en 32 testcellen |
| Quantumrepresentatie | 4 blokken van 40 qubits |
| Hardware | IBM Fez via Fire Opal |
| Circuits | 192 circuits, 128 shots per circuit |
| Readout | 405 Pauli-observabelen per cel |
| Hardwaretest | 0,50000 — 16 van 32 correct |
| Klassieke test | 0,53125 — 17 van 32 correct |

## Wat deze reeks wel en niet claimt

De uitvoering laat zien dat de volledige route—van echte single-cell RNA-data, via een compacte quantumfeaturemap, naar gemeten hardwarefeatures en een vooraf vastgelegde classifier—technisch uitvoerbaar is. Dat is een concreet hardwaretussenresultaat.

Het is nog geen empirisch quantumvoordeel. De hardware verloor de vaste test met één cel, het onzekerheidsinterval is breed en de gebruikte fysieke qubits zijn niet hetzelfde als de foutgecorrigeerde logische qubits uit de theorie. Ook is onze ondiepe featuremap een hardwaregerichte benadering, niet het volledige QOS/QSVT-algoritme.

Juist daardoor is de reeks nuttig. Zij laat niet alleen zien hoe de theorie werkt, maar ook waar de moeilijke overgang naar echte hardware zit: datatoegang, circuitdiepte, readout, shotruis, generalisatie en een eerlijke klassieke vergelijking.

## Primaire bronnen

- [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639)
- [Officiële Quantum Oracle Sketching-code](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze Qiskit- en Fire Opal-runners](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [10x PBMC68k-dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
