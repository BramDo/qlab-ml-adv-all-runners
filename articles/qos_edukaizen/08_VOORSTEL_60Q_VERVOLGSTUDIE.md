# Het 60-qubitresultaat: hardware 17, lineair 16, RBF 14

Op 21 juli 2026 is de eerder voorgestelde 60-qubitpilot daadwerkelijk uitgevoerd via Fire Opal op `ibm_fez`. Ditmaal betekende meer breedte niet simpelweg een grotere hash. We vervingen de oude invoer door zestig labelvrije coexpressiemodules en hielden het circuit bewust ondiep.

De vaste held-out test gaf het beste hardwarepuntresultaat in deze reeks:

| Route | Balanced accuracy | Correct |
| --- | ---: | ---: |
| 60-qubit hardware | 0,53125 | 17/32 |
| klassieke lineaire baseline | 0,50000 | 16/32 |
| klassieke RBF-baseline | 0,43750 | 14/32 |

Dit is een positief resultaat op twee assen: de bevroren hardwareclassifier eindigde op deze test vóór beide vooraf vastgelegde klassieke referenties, en de quantumfeatureberekening gaf lokaal een duidelijke tijdseparatie tegenover MPS. Door de kleine testset en de ontbrekende MPS-convergentie is het geen algemene of asymptotische quantum-advantageclaim.

## Waarom de eerste 60-qubitroute niet werkte

Onze eerdere 60-qubitroute haalde 15/32 op hardware, tegenover 17/32 ideaal en 19/32 klassiek. De extra qubits droegen toen vooral meer gehashte invoerkanalen. Dat voegde breedte toe, maar niet noodzakelijk stabielere biologische structuur.

De nieuwe pilot veranderde daarom de representatie, niet alleen het aantal qubits:

- zestig coexpressiemodules, geleerd uit een vaste en labelvrije pool van 512 cellen;
- 1.200 variabele genen met detectiefrequentie tussen 1% en 95%;
- deterministische KMeans met `random_state=6110` en `n_init=20`;
- per module vier samenvattingen: gemiddeld `log1p`, detectiefractie, RMS en het gemiddelde van het bovenste kwartiel;
- mediaan/IQR-schaling uitsluitend geleerd op de trainingscellen;
- `tanh(z/3)` en L2-normalisatie per blok.

De modulepool, training en test zijn onderling gescheiden. De testlabels speelden geen rol bij modulevorming, schaling, modelkeuze of hyperparameterkeuze.

## Het 60-qubitcircuit

De zestig qubits liggen in een logische `6×10`-topologie. De featuremap gebruikt vier invoerblokken, een multiplier van `sqrt(60)`, een logische diepte van 20 en 134 tweequbitinteracties. X-, Y- en Z-metingen leveren samen 627 geordende observabelen per cel.

Voor 32 trainings- en 32 testcellen zijn drie meetcircuits per cel gemaakt:

- 192 circuits;
- 128 shots per circuit;
- 24.576 shots totaal;
- backend `ibm_fez`;
- Fire Opal action `2335848`.

Het Fire Opal-dashboard rapporteerde **26 quantumseconden** voor deze taak. Dat is opvallend kort voor 192 circuits op zestig qubits. Het gearchiveerde `get_result`-antwoord bevatte dit veld niet; daarom vermelden we expliciet dat 26 seconden de dashboardmeting is.

## Training-only selectie en blinde test

Binnen de trainingsset koos de quantumroute via cross-validation een RBF-SVC met `C=10` en `gamma=0,1`. De gemiddelde training-only CV-score was 0,59375; de slechtste fold bleef op 0,50000. Pas daarna is één keer op de 32 afgeschermde testcellen geëvalueerd.

Op die test scoorde hardware 17/32, de lineaire baseline 16/32 en de RBF-baseline 14/32. Dat verschil van één cel ten opzichte van de sterkste baseline is klein, maar de richting is voor het eerst positief op echte hardware.

## Waarom ook de doorlooptijd interessant is

De eigenlijke quantumtaak duurde volgens het Fire Opal-dashboard slechts 26 seconden. Van indiening tot volledig opgehaald resultaat verstreken ongeveer 8 minuten en 33 seconden; daarin zitten ook orchestratie, compilatie, wachttijd en retrieval. De lokale MPS-controle van exact dezelfde 60-qubitrepresentatie liep 42 minuten en 57 seconden en was toen nog niet geconvergeerd: bond dimension 64 was voltooid, maar bij 128 was slechts één van acht benodigde delen klaar.

Voor dezelfde gespecificeerde 627-featuretarget is 2.577 seconden gedeeld door 26 seconden gelijk aan 99,1. Omdat MPS toen nog niet klaar was, is **meer dan 99,1×** een gemeten lokale ondergrens voor de kernel-tijdseparatie. Als we de volledige Fire Opal-route van 513 seconden gebruiken, blijft de ondergrens **meer dan 5,0×**.

Dit is dus een **lokale time-to-feature-generation advantage binnen de gedeclareerde resources**. Het is nadrukkelijk geen end-to-end tijdvoordeel tegenover gewone klassieke ML: de lineaire en RBF-modellen kunnen rechtstreeks op klassiek voorbereide data worden getraind zonder het 60-qubitcircuit te simuleren. Ook convergeerde MPS niet, zodat geen gematchte numerieke featurefout beschikbaar is.

## Statistische grens van dit resultaat

Met 32 testcellen is één fout gelijk aan 3,125 procentpunt. De tweezijdige exacte McNemar-p-waarde tegenover de sterkste lineaire baseline is 1,0. Het 95%-bootstrapinterval voor `hardware minus lineair` loopt van -0,1875 tot +0,25. Een klassiek voordeel, gelijkspel en hardwarevoordeel blijven dus allemaal verenigbaar met deze kleine steekproef.

We melden het tijdresultaat daarom als een **taakgebonden lokale quantum advantage voor featuregeneratie**, met een gemeten ondergrens van 99,1× op kerneltijd en 5,0× inclusief retrieval. De voorspellende 17/32-score blijft door de kleine test een empirische aanwijzing, niet bewezen algemene quantum advantage.

## De volgende beslispoort

De volgende wetenschappelijke stap is niet automatisch meer quantumtijd gebruiken. Eerst moeten we het ontwerp bevriezen en de volledige klassieke frontier voor een grotere 256/256-split vastleggen. Een grote hardwarefase zou 1.536 circuits van 128 shots vergen en krijgt alleen afzonderlijke toestemming wanneer de extra informatiewaarde opweegt tegen het Fire Opal-budget.

## Bronnen en reproduceerbaarheid

- [60-qubit modulepipeline](qiskit_qos_pbmc68k_q60_module_pipeline.py)
- [Fire Opal-pilotrunner](qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py)
- [60-qubit runbook](Q60_MODULE_B4_RUNBOOK.md)
- [Vermelding in de Pro Student Quantum Advantage List](https://edukaizen.nl/pro-student-quantum-advantage-list/)
- [QOS-paper](https://arxiv.org/abs/2604.07639)
- [Volledige repository](https://github.com/BramDo/qlab-ml-adv-all-runners)
