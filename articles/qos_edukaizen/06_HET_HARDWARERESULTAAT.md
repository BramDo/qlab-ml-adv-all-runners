# Het resultaat: hardware 16, klassiek 17

Na dagen van dataselectie, representatietests, circuitvalidatie, Fire Opal-uitvoering en lokale analyse komt de vaste test neer op één eenvoudig getalverschil:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 40-qubit hardwarefeatures | 0,50000 | 16/32 |
| klassieke raw-gene frontier | 0,53125 | 17/32 |
| hardware minus klassiek | -0,03125 | -1 cel |

De hardware won dus niet. Maar “klassiek won” is eveneens te groot geformuleerd. Met 32 testcellen is één voorspelling gelijk aan 3,125 procentpunt. De uitkomst is praktisch een gelijkspel rond kansniveau.

## Wat gebeurde met het trainingssignaal?

Tijdens training-only cross-validatie scoorde de gekozen hardwareclassifier gemiddeld 0,59375. De klassieke winnaar zat op 0,53125. Dat was een klein lichtpunt: de gemeten quantumfeatures bevatten blijkbaar genoeg structuur om binnen sommige trainingfolds boven kansniveau uit te komen.

Op de vooraf afgeschermde testset zakte hardware echter naar 0,50000. Het signaal generaliseerde dus niet aantoonbaar. Omdat beide eindmodellen hun volledige training perfect fitten, past het patroon bij overfitting of instabiele kleine-samplegeometrie.

We noemen de CV-voorsprong daarom een **exploratief trainingssignaal**, niet een gedeeltelijk quantumvoordeel.

## De paarsgewijze vergelijking

Van de 32 testcellen waren er:

- 7 waarvoor alleen de hardwareclassifier correct was;
- 8 waarvoor alleen de klassieke classifier correct was;
- 17 waarvoor beide correct of beide fout waren.

De exacte tweezijdige McNemar-p-waarde is 1,0. De 95%-bootstrapinterval voor `hardware minus klassiek` loopt van -0,25 tot +0,1875. Zowel een materieel klassiek voordeel als een hardwarevoordeel past dus nog binnen de onzekerheid van deze pilot.

Ook de afzonderlijke Wilson-intervallen zijn breed:

- hardware: ongeveer 0,336 tot 0,664;
- klassiek: ongeveer 0,364 tot 0,691.

## Wat is wél bereikt?

De voorspellende uitkomst is zwak, maar de uitvoering heeft vier concrete resultaten opgeleverd:

1. Een echte PBMC68k-cel kan reproduceerbaar van 32.738 genkolommen naar een vierbloks 40-qubitcircuit worden vertaald.
2. Een batch van 192 circuits kan via Fire Opal op IBM Fez worden gevalideerd, uitgevoerd en zonder resubmissie teruggehaald.
3. Uit slechts drie globale meetbases kunnen 405 geordende features per cel worden gereconstrueerd.
4. De hardwarefeatures kunnen in een vooraf vastgelegde, testlekvrije ML-pijplijn worden geëvalueerd tegen een opnieuw berekende klassieke frontier.

Dat is een **hardware-feasibility milestone**: een bewijs dat de pijplijn uitvoerbaar is. Het is geen bewijs dat de pijplijn al nuttig genoeg generaliseert.

## Hoe zit het met de 26 quantumseconden?

De provider rapporteerde 26 QPU-seconden voor de 192 circuits. Dit is opvallend compact vergeleken met sommige volledige klassieke simulaties van brede quantumcircuits. Toch is het geen speedupmeting voor de ML-taak.

Voor een eerlijke tijdclaim moeten we ook tellen:

- klassieke selectie en hashing van genen;
- circuitconstructie en QASM-export;
- providercompilatie en queue;
- Fire Opal-verwerking;
- dataretrieval en observableberekening;
- classifiertraining;
- de tijd van de beste klassieke end-to-endpipeline.

Bovendien gaf de klassieke raw-geneclassifier hier een betere puntenscore. Een korte QPU-kern is niet genoeg wanneer de eindtaak niet beter of goedkoper wordt opgelost.

## Waarom dit toch publiceerbaar is

Negatieve en bijna-neutrale hardwaremetingen zijn wetenschappelijk nuttig wanneer het protocol vooraf is vastgelegd. Deze pilot laat precies zien waar een theoretische ruimtewinst onderweg naar NISQ-hardware kan verdwijnen:

- aggressive featurecompressie kan biologisch signaal verwijderen;
- hashbotsingen kunnen interacties vermengen;
- ondiepe circuits kunnen te weinig expressief zijn;
- lawaai en 128 shots vervormen correlaties;
- 32 trainingsvoorbeelden zijn te weinig voor een stabiele 405-dimensionale classifier.

De juiste conclusie is daarom niet “QOS werkt niet”. Het volledige QOS-algoritme is niet getest. De conclusie is: **deze ondiepe 40-qubit QOS-geïnspireerde featuremap was uitvoerbaar, maar generaliseerde op de gekozen 32/32-PBMCsplit niet beter dan de klassieke frontier**.

## De claim in één zin

> We demonstrated an end-to-end 40-qubit hardware-feasible QOS-inspired single-cell classification pipeline; it produced an exploratory training-CV signal but no held-out predictive quantum advantage.

In [deel 7](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/route-naar-quantumvoordeel-qml/) zetten we uiteen welke experimenten nodig zijn om van dit tussenresultaat naar een serieuze advantage-test te gaan.

## Reproduceerbaarheid

- [Hardware-analyserunner](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis.py)
- [GitHub pull request met volledige runner- en testsuite](https://github.com/BramDo/qlab-ml-adv-all-runners/pull/1)

