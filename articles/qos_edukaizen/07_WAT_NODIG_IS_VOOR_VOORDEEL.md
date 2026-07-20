# Wat is nog nodig voor quantumvoordeel?

De 40-qubitpilot bewijst uitvoerbaarheid, niet voordeel. Om een geloofwaardige quantum-ML-claim te maken moeten drie verschillende vragen tegelijk positief worden beantwoord: generaliseert het model, is de quantumroute aantoonbaar moeilijk te evenaren binnen de gekozen klassieke resourcegrens, en klopt de end-to-end resourceboekhouding?

## 1. Eerst aantoonbare generalisatie

Een volgende studie heeft meer onafhankelijke cellen nodig. Niet alleen een grotere testset, maar ook meerdere vooraf vastgelegde splits of donors. Een bruikbaar minimumprogramma is:

- model- en observablekeuze uitsluitend op trainingdata;
- meerdere vaste, gebalanceerde train/testsplits;
- een volledig onaangeroerde finale testcohort;
- rapportage van balanced accuracy, AUC, calibratie en per-klassefouten;
- betrouwbaarheidsintervallen en een vooraf gekozen paired test;
- biologische controle over verschillende donoren of batches.

Met 32 testcellen kan één cel de conclusie omdraaien. Met honderden testcellen wordt een klein maar consistent verschil veel beter beoordeelbaar.

## 2. Meer hardwaredata, niet alleen meer qubits

Veertig of zestig qubits klinkt indrukwekkend, maar breedte alleen lost samplearmoede niet op. Onze huidige 405 features tegenover 32 trainingscellen is statistisch ongunstig. Mogelijke verbeteringen zijn:

- meer trainingscellen op hardware uitvoeren;
- het observablepaneel training-only verkleinen;
- meerdere onafhankelijke 128-shotruns uitvoeren;
- calibratiedrift over dagen meten;
- shotbudgetten vergelijken;
- onzekerheid van de quantumfeatures meenemen in de classifier.

Een zestig-qubitcircuit kan minder diep worden gehouden, maar vraagt nog steeds een zorgvuldig readout- en batchontwerp. Fire Opal-budget moet daarom pas worden besteed nadat lokale simulatie en training-only stabiliteit voldoende sterk zijn.

## 3. Dichter bij het volledige QOS-algoritme

De theoretische classificatiescheiding geldt voor een oracle-sketching- en quantum-lineair-algebraprotocol met formele sampletoegang. Onze featuremap gebruikt vier gehashte blokken, korte rotatielagen en lokale Pauli-readout. De volgende theoretische brug moet expliciet maken:

- welk deel van de QOS-oracle door het circuit wordt benaderd;
- hoe de approximatiefout schaalt met blokken, shots en diepte;
- welke QSVT- of lineaire-solverstappen ontbreken;
- of de 405-featureclassifier dezelfde beslisfunctie benadert als de LS-SVM uit de theorem;
- hoeveel klassieke sidecarinformatie nodig is voor hashing en circuitaansturing.

Zonder die brug is “QOS-inspired” correcter dan “hardware-implementatie van theorem 3”.

## 4. Een sterkere klassieke frontier

De klassieke tegenstander moet zowel praktisch als resourcebeperkt worden bekeken. Minstens nodig zijn:

- sparse logistische regressie en LinearSVC op alle genen;
- feature selection met uitsluitend trainingdata;
- hashing, sparse JL en streamingmodellen met gemeten geheugen;
- PCA/SVD- en kernelroutes;
- biologische markerbaselines;
- tensor-network- of causal-coneanalyse van het specifieke quantumcircuit;
- runtime, RAM, modelgrootte en energie als afzonderlijke kolommen.

De officiële QOS-repository voegde in mei 2026 sparse JL-projecties toe. Dat is belangrijk: de klassieke frontier beweegt mee. Een advantage-claim moet tegen de nieuwste sterke baseline worden herhaald, niet tegen de baseline waarmee het project begon.

## 5. Welke advantage bedoelen we?

“Quantum advantage” kan verschillende dingen betekenen:

| Claim | Benodigd bewijs |
| --- | --- |
| voorspellend voordeel | significant betere held-out prestatie |
| ruimtevoordeel | dezelfde taakprestatie met aantoonbaar minder werkgeheugen |
| tijdvoordeel | lagere eerlijke end-to-endtijd bij dezelfde fouttolerantie |
| samplevoordeel | minder datapunten nodig voor dezelfde generalisatie |
| scaling advantage | gunstiger gemeten groei met probleemgrootte |

De QOS-theorie richt zich vooral op machinegrootte en in dynamische gevallen samplecomplexiteit. Onze pilot meet vooral hardware-uitvoerbaarheid en predictive accuracy. Dat zijn nog verschillende assen.

## Een realistische experimentele ladder

De veiligste route vooruit bestaat uit vijf poorten:

1. **Lokale stabiliteit:** een ideale of MPS-route moet op bijna alle vooraf gekozen splits boven een brede klassieke frontier liggen.
2. **Representatiecontrole:** verwijder observabelen die alleen trainingsruis vangen en test hash-seedstabiliteit.
3. **Kleine hardwarebevestiging:** herhaal de huidige 40-qubitroute met onafhankelijke shots en meer cellen.
4. **Breedtescaling:** vergelijk 20, 40 en 60 qubits met dezelfde taak, accuracydoelstelling en resourceboekhouding.
5. **Finale blindtest:** bevries alles en evalueer één nog nooit bekeken cohort.

Pas wanneer de quantumroute op de finale blindtest beter scoort of dezelfde score met een overtuigend lagere gemeten resource bereikt, ontstaat een empirische advantage-claim.

## Wat kunnen we nu al zeggen?

De huidige serie eindigt met een bescheiden maar concrete conclusie:

> De QOS-theorie heeft een route naar exponentieel ruimtevoordeel voor leren uit enorme klassieke datastromen. De officiële resultaten zijn numeriek. Onze 40-qubitpilot brengt een QOS-geïnspireerde single-cell featuremap voor het eerst in onze werkbank volledig naar echte hardware. Die uitvoering werkt technisch, maar levert op de vaste 32-cellentest nog geen generalisatie- of resourcevoordeel op.

Dat is geen eindpunt. Het is een scherp gedefinieerd startpunt voor de volgende experimenten. In [deel 8](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/voorstel-60-qubit-qml-vervolgstudie/) werken we één mogelijke 60-qubitvervolgstudie uit als voorstel. Die studie wordt nu nog niet uitgevoerd.

## Bronnen en code

- [QOS-paper](https://arxiv.org/abs/2604.07639)
- [Officiële QOS-repository](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze Qiskit/Fire Opal-repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [40-qubit milestone pull request](https://github.com/BramDo/qlab-ml-adv-all-runners/pull/1)
