# Wat is nog nodig voor quantumvoordeel?

De 60-qubitpilot van 21 juli 2026 geeft voor het eerst een positief hardwarepuntresultaat: 17/32 tegenover 16/32 lineair en 14/32 RBF. Dat is een serieuze aanwijzing om verder te testen, maar nog geen bewezen voordeel. Voor een geloofwaardige quantum-ML-claim moeten drie verschillende vragen tegelijk positief worden beantwoord: generaliseert het model, is de quantumroute aantoonbaar moeilijk te evenaren binnen de gekozen klassieke resourcegrens, en klopt de end-to-end resourceboekhouding?

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

Veertig of zestig qubits klinkt indrukwekkend, maar breedte alleen lost samplearmoede niet op. Zowel 405 features in de 40-qubitroute als 627 observabelen in de nieuwe 60-qubitroute staan tegenover slechts 32 trainingscellen. Dat blijft statistisch ongunstig. Mogelijke verbeteringen zijn:

- meer trainingscellen op hardware uitvoeren;
- het observablepaneel training-only verkleinen;
- meerdere onafhankelijke 128-shotruns uitvoeren;
- calibratiedrift over dagen meten;
- shotbudgetten vergelijken;
- onzekerheid van de quantumfeatures meenemen in de classifier.

Een zestig-qubitcircuit kan minder diep worden gehouden, maar vraagt nog steeds een zorgvuldig readout- en batchontwerp. De kleine sentinel mocht als feasibility-test door toen de geplande MPS-convergentiecontrole niet binnen de beschikbare tijd kon worden voltooid. Voor een grote hardwarefase blijven training-only stabiliteit, een bevroren ontwerp en afzonderlijke toestemming verplicht.

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

Na de geslaagde 60-qubitsentinel bestaat de route vooruit uit vijf nieuwe poorten:

1. **Bevries de representatie:** behoud de zestig labelvrije genmodules, 627 observabelen en classifierselectie zonder testfeedback.
2. **Verbreed de klassieke frontier:** voeg sparse lineaire, kernel-, marker-, JL- en streamingbaselines toe met gemeten tijd en geheugen.
3. **Grotere lokale splits:** test 256/256 en meerdere vooraf gekozen seeds voordat nieuwe hardwaretijd wordt ingezet.
4. **Grote hardwarebevestiging:** voer alleen na afzonderlijke toestemming het bevroren ontwerp uit op meer cellen en registreer alle batchkosten.
5. **Finale blindtest:** evalueer één nog nooit bekeken cohort en publiceer ook een nul- of negatief resultaat.

Pas wanneer de quantumroute op de finale blindtest beter scoort of dezelfde score met een overtuigend lagere gemeten resource bereikt, ontstaat een empirische advantage-claim.

## Wat kunnen we nu al zeggen?

De huidige serie eindigt nu met een lokaal gemeten, maar nog steeds afgebakend voordeel:

> Onze 60-qubit QOS-geïnspireerde featuremap was uitvoerbaar op echte hardware, scoorde op de vaste 32-cellentest één cel beter dan de sterkste vooraf gekozen klassieke baseline en bereikte dezelfde gespecificeerde featuretarget in 26 quantumseconden. De MPS-poging was na 2.577 seconden nog onvoltooid: een lokale kernel-tijdseparatie groter dan 99,1×. Dit is een task-specific time-to-feature-generation advantage binnen de gedeclareerde resources, geen algemene quantum-advantageclaim.

In [deel 8](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/voorstel-60-qubit-qml-vervolgstudie/) staan het uitgevoerde 60-qubitprotocol, de 17/32-uitkomst, de timing en de statistische grens volledig beschreven.

## Bronnen en code

- [QOS-paper](https://arxiv.org/abs/2604.07639)
- [Officiële QOS-repository](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze Qiskit/Fire Opal-repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Hardware milestone pull request](https://github.com/BramDo/qlab-ml-adv-all-runners/pull/1)
