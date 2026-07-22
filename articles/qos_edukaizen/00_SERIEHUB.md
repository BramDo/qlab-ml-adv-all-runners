# Van genexpressie naar 60 qubits

Quantum machine learning klinkt vaak alsof een quantumcomputer een complete database in één keer doorzoekt. Ons experiment doet iets preciezers en interessanters: het probeert **het celtype van één cel te voorspellen uit haar genexpressieprofiel**. De invoer is een lange, dunbezette vector met RNA-tellingen; de uitvoer is een van twee immuuncelklassen.

Deze achtdelige reeks verbindt vier lagen die gemakkelijk door elkaar raken. De eerste laag is de theorie van *Quantum Oracle Sketching* (QOS), gepubliceerd in april 2026. Die theorie gaat over een klein quantummodel dat enorme klassieke datastromen verwerkt zonder de hele matrix te bewaren. De tweede laag is de officiële JAX-code en de numerieke PBMC68k-experimenten. De derde laag is onze **letterlijke flat-QOS-sketch op vier qubits**: een afgebakende port van de officiële sampling-kern, fysiek uitgevoerd op IBM Fez. De vierde laag bestaat uit onze 40- en 60-qubit PBMC68k-routes. Die zijn QOS-geïnspireerde NISQ-featuremaps en nadrukkelijk geen letterlijke implementatie van het volledige QOS/QSVT-algoritme.

De nieuwste 60-qubitrun is het sterkste resultaat in de reeks. Op de vooraf afgeschermde testset scoorde hardware 17/32, tegenover 16/32 voor de lineaire en 14/32 voor de RBF-baseline. Het Fire Opal-dashboard rapporteerde slechts 26 quantumseconden en de volledige hardwarefeature-uitvoer was na ongeveer 8 minuten en 33 seconden opgehaald. Onze klassieke MPS-poging had na 42 minuten en 57 seconden nog geen convergente referentie opgeleverd.

## De reeks

1. [Wat is de QML-taak? Cellen classificeren, geen genen opzoeken](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/wat-is-de-qml-taak-cellen-classificeren/)
2. [De theorie van Quantum Oracle Sketching](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-oracle-sketching-theorie/)
3. [Van PBMC68k-genexpressie naar 40 qubits](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/pbmc68k-van-genexpressie-naar-qubits/)
4. [Van JAX naar een 40-qubit hardwarecircuit](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qos-naar-40-qubit-hardware-fire-opal/)
5. [405 observabelen en een lekvrije classifier](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-readout-405-observabelen-classifier/)
6. [Het 40-qubitresultaat: hardware 16, klassiek 17](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/)
7. [Wat is nog nodig voor quantumvoordeel?](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/route-naar-quantumvoordeel-qml/)
8. [Het 60-qubitresultaat: hardware 17, lineair 16, RBF 14](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/voorstel-60-qubit-qml-vervolgstudie/)
9. [Beginnershandleiding QML: van UMI-telling naar een 4-qubitcircuit](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qml-beginnershandleiding-umi-naar-4-qubit-circuit/)

## Het experiment in één tabel

| Onderdeel | Waarde |
| --- | --- |
| Dataset | PBMC68k / Zheng68k |
| Volledige invoer per cel | 32.738 genen |
| Binaire taak | regulatoire CD4-T-cel versus CD4-geheugencel |
| Pilot | 32 training- en 32 testcellen |
| Quantumrepresentatie | 60 labelvrije coexpressiemodules, 4 statistische blokken |
| Hardware | IBM Fez via Fire Opal |
| Circuits | 192 circuits, 128 shots per circuit |
| Circuit | 60 qubits, logische diepte 20, 134 tweequbitinteracties |
| Readout | 627 Pauli-observabelen per cel |
| 60q-hardwaretest | 0,53125 — 17 van 32 correct |
| Lineaire test | 0,50000 — 16 van 32 correct |
| RBF-test | 0,43750 — 14 van 32 correct |
| Quantumtijd volgens Fire Opal-dashboard | 26 seconden |
| Submit tot retrieval | ongeveer 8 minuten 33 seconden |
| Klassieke MPS-poging | na 42 minuten 57 seconden zonder convergente referentie gestopt |
| Lokale tijdseparatie voor dezelfde featuretarget | meer dan 99,1× op kerneltijd; meer dan 5,0× inclusief retrieval |

## Welke onderdelen zijn letterlijk QOS?

| Route | Relatie tot de QOS-paper |
| --- | --- |
| 4q flat-QOS-toy/pilot | Letterlijke port van de officiële `q_state_sketch_flat` sampling-kern voor $D=16$ en $M=64$; 66 circuits op IBM Fez, Fire Opal-action `2334156`; gemiddelde Hellinger-fideliteit 0,990104 over 64 willekeurige kernels |
| 40q/60q PBMC68k | QOS-geïnspireerde hardware-featuremaps met klassiek berekende rotatiehoeken en Pauli-readout; geen letterlijke sampling-oracle, QSVT of classical-shadowketen |
| Volledige paperroute | Streaming sampletoegang, oracle-opbouw, quantum lineaire algebra en gecontroleerde readout; niet end-to-end op hardware geïmplementeerd in dit project |

De 4q-run laat dus zien dat een echte QOS-sketchbouwsteen op hardware werkt. De 60q-run laat iets anders zien: dat een brede, ondiepe en biologisch gestructureerde featuremap op echte PBMC68k-data uitvoerbaar is en lokaal een interessante timing en puntenscore bereikt. Geen van beide resultaten is op zichzelf een hardwarebewijs van theorem 3.

## Wat deze reeks wel en niet claimt

De uitvoering laat zien dat de volledige route—van echte single-cell RNA-data, via labelvrij geleerde genmodules en een compacte quantumfeaturemap, naar gemeten hardwarefeatures en een vooraf vastgelegde classifier—technisch uitvoerbaar is. Bovendien had de 60-qubitroute op deze vaste testset de beste puntenscore van de drie vooraf gekozen modellen.

Dat is binnen de gedeclareerde lokale resources een gemeten **time-to-feature-generation advantage**. De 26-seconden-quantumtaak is meer dan 99,1× sneller dan de na 2.577 seconden nog onvoltooide MPS-poging; zelfs de volledige 513-secondenroute tot retrieval is meer dan 5,0× sneller. De featuretarget was dezelfde, maar MPS convergeerde niet en leverde dus geen gematchte numerieke fout op. Het is daarom geen algemene of asymptotische quantumvoordeelclaim. De klassieke lineaire en RBF-classifiers zelf zijn goedkoop, de test bevat slechts 32 cellen en het onzekerheidsinterval is breed. De 26 quantumseconden komen uit het Fire Opal-dashboard; het gearchiveerde API-resultaat liet dat veld leeg. Ook is de 60q-featuremap een hardwaregerichte QOS-geïnspireerde benadering, **geen letterlijke QOS-implementatie** en niet het volledige QOS/QSVT-algoritme.

Juist daardoor is de reeks nuttig. Zij laat niet alleen zien hoe de theorie werkt, maar ook waar de moeilijke overgang naar echte hardware zit: datatoegang, circuitdiepte, readout, shotruis, generalisatie en een eerlijke klassieke vergelijking.

## Primaire bronnen

- [Exponential quantum advantage in processing massive classical data](https://arxiv.org/abs/2604.07639)
- [Officiële Quantum Oracle Sketching-code](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze Qiskit- en Fire Opal-runners](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Pro Student Quantum Advantage List](https://edukaizen.nl/pro-student-quantum-advantage-list/)
- [10x PBMC68k-dataset](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
