# Voorstel: een 60-qubit QML-vervolgstudie

Zestig qubits kunnen meer van een genexpressieprofiel tegelijk dragen dan veertig qubits. Dat betekent nog niet dat een 60-qubitmodel automatisch beter classificeert. Extra breedte kan ook meer ruis, meer observabelen en meer mogelijkheden tot overfitting opleveren. Daarom is dit hoofdstuk **een onderzoeksvoorstel, geen aangekondigde uitvoering**. We dienen nu geen nieuwe circuits in bij Fire Opal en starten geen grote klassieke sweep.

## Wat de bestaande 60-qubitproef ons al leerde

We hebben al een verkennende 60-qubitroute op dezelfde PBMC68k-taak uitgevoerd. Op de vaste kleine testset waren de uitkomsten:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 60-qubit hardware | 0,46875 | 15/32 |
| ideale 60-qubitrepresentatie | 0,53125 | 17/32 |
| klassieke referentie | 0,59375 | 19/32 |

Meer qubits waren hier dus geen verbetering. Ook in een latere lokale blindtest met 256 trainings- en 256 testcellen bleef de ideale 60-qubitrepresentatie op 0,53125, tegenover 0,55859 voor een lineair model en 0,54688 voor een RBF-model op dezelfde gehashte invoer. Dat is nuttige negatieve informatie: het huidige 60-qubitontwerp hoeft niet nogmaals ongewijzigd naar hardware.

## Het echte knelpunt: voorbereiding vóór hardware

De meeste onderzoekstijd zit niet in de quantumseconden. Zij zit in het vastzetten van data, splits, genrepresentaties, klassieke referenties, observablekeuze en statistische toetsen. Iedere nieuwe dataset kan die keten opnieuw openen. Een verstandig 60-qubitproject moet daarom zoveel mogelijk van de bestaande PBMC68k-infrastructuur hergebruiken:

- dezelfde twee CD4-T-celklassen;
- dezelfde gecontroleerde datalader en normalisatie;
- vooraf vastgelegde grotere splits;
- de al geïmplementeerde lineaire en RBF-referenties;
- dezelfde balanced-accuracy- en paired-testcode;
- dezelfde Fire Opal-validatie- en retrievalroute.

We stellen dus geen nieuwe biologische dataset voor. Het onderzoeksvraagstuk blijft celclassificatie; alleen de quantumrepresentatie verandert.

## Wat zestig qubits inhoudelijk moeten toevoegen

Een nieuwe route is alleen zinvol als de twintig extra qubits nieuwe informatie bewaren. Het voorstel is om de breedte te gebruiken voor stabiele genmodules of aanvullende hashkanalen, niet voor een simpele uitrekking van het 40-qubitcircuit. Vier blokken van zestig qubits kunnen bijvoorbeeld 240 compacte invoerkanalen dragen in plaats van 160. Een ondiepe interactielaag kan vervolgens correlaties tussen die modules mengen.

De ontwerpgrenzen blijven bewust streng:

- maximaal zestig fysieke qubits;
- geringe circuitdiepte en hardwarevriendelijke koppelingen;
- een klein, training-only gekozen observablepaneel;
- geen testsetgebruik voor feature-, model- of hyperparameterkeuze;
- een shotruisprojectie voordat hardware wordt overwogen.

Meer meetbare correlaties zijn niet vanzelf beter. Met honderden mogelijke observabelen en weinig trainingscellen leert de klassieke eindclassifier gemakkelijk toeval. Het voorstel beperkt daarom de readout liever tot enkele tientallen stabiele observabelen dan alle mogelijke qubitparen te gebruiken.

## Een korte go/no-go-ladder

Om de klassieke voorbereiding begrensd te houden, krijgt de studie vier opeenvolgende poorten:

1. **Bevries de taak.** Hergebruik PBMC68k, de bestaande labels en vijf vooraf gekozen grotere splits. Geen nieuwe datasetzoektocht.
2. **Vergelijk representaties.** Test lokaal alleen het huidige 40-qubitanker en één nieuw 60-qubitontwerp tegen de bestaande lineaire en RBF-frontier.
3. **Projecteer hardware-effecten.** Voeg 128-shotruis en een realistische beperkte readout toe. Stop als het voordeel dan verdwijnt.
4. **Beslis over Fire Opal.** Alleen wanneer de ideale 60-qubitroute op minstens vier van vijf splits boven beide klassieke referenties ligt, wordt een kleine bevroren hardwarepilot voorgesteld.

Deze poorten zijn geen bewijs van quantumvoordeel. Ze voorkomen vooral dat quantumtijd wordt besteed aan een representatie die lokaal al geen kans maakt.

## Wat een latere hardwarepilot zou meten

Als de lokale poorten ooit slagen, is de eerste hardwarevraag bescheiden: blijft het lokale 60-qubitsignaal na transpilation, ruis en eindige shots zichtbaar? De pilot hoeft nog geen definitieve advantage-test te zijn. Wel moeten circuit, observabelen, classifier en testcellen vóór indiening vaststaan. De rapportage vergelijkt dan vier niveaus: ideaal quantum, geprojecteerde shotruis, Fire Opal-hardware en de bevroren klassieke frontier.

Pas een veel grotere onafhankelijke blindtest kan daarna iets zeggen over voorspellend voordeel. Een claim over computationeel of ruimtevoordeel vraagt bovendien afzonderlijke resourceboekhouding; een hogere accuracy alleen bewijst dat niet.

## Besluit: documenteren en uitstellen

Een opnieuw ontworpen 60-qubitroute kan beter worden dan onze 40-qubitpilot, omdat zij meer relevante geninformatie en interacties kan bewaren. De bestaande resultaten tonen echter dat breedte alleen niet genoeg is. De grootste verwachte winst moet eerst uit representatie en voldoende trainingsdata komen.

Daarom is de huidige beslissing: **het voorstel vastleggen, maar de studie nog niet uitvoeren**. Zo blijft de route beschikbaar voor een later moment waarop klassieke voorbereidingstijd en Fire Opal-budget bewust kunnen worden vrijgemaakt.

## Bronnen en resultaten

- [QOS-paper](https://arxiv.org/abs/2604.07639)
- [Officiële QOS-repository](https://github.com/haimengzhao/quantum-oracle-sketching)
- [Onze Qiskit/Fire Opal-repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
- [Het gepubliceerde 40-qubitresultaat](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/)
