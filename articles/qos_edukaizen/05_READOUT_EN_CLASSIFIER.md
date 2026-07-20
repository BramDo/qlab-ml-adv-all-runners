# 405 observabelen en een lekvrije classifier

Een hardware-experiment wordt pas een machine-learningexperiment wanneer vooraf duidelijk is welke data voor training, modelkeuze en toetsing worden gebruikt. Met slechts 32 trainings- en 32 testcellen kan één onzorgvuldige keuze de score sterk vertekenen.

## De quantumprocessor levert features, geen eindlabel

Voor iedere cel berekenen we uit de X-, Y- en Z-metingen 405 verwachtingswaarden. Een enkele feature heeft de vorm:

```math
\langle P_i \rangle \quad\text{of}\quad \langle P_iP_j \rangle,
```

waar $P$ een X-, Y- of Z-Paulioperator is. De verzameling beschrijft lokale qubittoestanden en geselecteerde correlaties. Het hardwaredeel eindigt bij deze featurevector.

Daarna start een gewone supervised classifier. Dat is belangrijk voor de interpretatie: de voorspelling komt uit de combinatie van quantumfeaturemap en klassiek beslismodel. Een verschil met een klassieke baseline kan aan beide onderdelen liggen.

## Het vooraf vastgelegde protocol

Voor de hardware-uitkomst werd de analyse als volgt vastgelegd:

- primaire maat: balanced accuracy op de vaste testset;
- modelkeuze: vier-fold gestratificeerde cross-validatie;
- CV-seed: 6011;
- modelkeuze uitsluitend op 32 trainingsrijen;
- testset één keer gebruiken nadat de winnaar vaststaat;
- klassieke frontier opnieuw berekenen op exact dezelfde 32/32-split.

De selectorfunctie in de code heeft opzettelijk geen testargument. Daardoor kan zij technisch geen testfeatures of testlabels gebruiken. Een unit test controleert die grens.

## Welke modellen zijn geprobeerd?

Op de 405 hardwarefeatures vergeleken we binnen de training:

- ridge-classificatie;
- logistische regressie met L2-regularisatie;
- lineaire support-vectorclassificatie;
- RBF-support-vectorclassificatie.

Regelsterktes en enkele RBF-gammawaarden vormden een bescheiden grid. Iedere standaardisatie werd opnieuw binnen de betreffende CV-fold geleerd. De selectie rangschikte eerst de gemiddelde balanced accuracy, daarna de slechtste fold, de spreiding en ten slotte eenvoud.

De winnaar was een RBF-SVC met `C=1` en `gamma=0.01`. De fold-scores waren 0,625; 0,375; 0,750 en 0,625. Het gemiddelde was 0,59375, maar de standaardafwijking 0,136 en de slechtste fold 0,375 laten zien dat het signaal instabiel was.

## Waarom balanced accuracy?

De volledige kandidaatpopulatie bevat meer regulatoire dan geheugencellen. De pilot-split is wel exact gebalanceerd. Balanced accuracy berekent eerst de gevoeligheid per klasse en neemt daarna het gemiddelde:

```math
\operatorname{BA}=\frac{1}{2}(\operatorname{TPR}+\operatorname{TNR}).
```

Op onze gebalanceerde testset is dat numeriek gelijk aan gewone accuracy, maar de definitie blijft bruikbaar wanneer toekomstige splits niet perfect gelijk zijn.

## De klassieke frontier

Een quantumroute mag niet alleen tegen een zwak model op dezelfde gecomprimeerde input worden vergeleken. Daarom gebruikten we twee klassieke representaties:

1. alle 32.738 ruwe genen, per cel genormaliseerd naar een library size van 10.000 en daarna `log1p`;
2. exact dezelfde vier gehashte B=4-blokken, platgemaakt tot 160 klassieke features.

Ook hier gebeurde modelkeuze uitsluitend via training-CV. De klassieke winnaar werd logistische regressie op de raw-gene `log1p`-matrix met `C=0.01`. Daarmee krijgt klassiek toegang tot informatie die door de top-48- en hashstappen aan de quantumkant kan zijn verwijderd. Dat is streng, maar relevant: een praktische voordeelclaim moet concurreren met de beste redelijke klassieke route naar dezelfde biologische voorspelling.

## Overfitting zichtbaar maken

Na fitten op alle 32 trainingscellen haalden zowel de gekozen hardwareclassifier als het klassieke model een trainingsscore van 1,0. Beide konden de kleine training dus volledig scheiden. Hun CV-scores waren veel lager, en de vaste test nog lager.

Dat verschil is geen detail maar de hoofdles. Een model kan een indrukwekkende geometrie in zijn trainingsfeatures vinden en toch geen stabiele regel voor nieuwe cellen hebben. Dit sluit aan bij bredere QML-literatuur: expressiviteit en trainbaarheid garanderen geen generalisatie.

## Wat de onzekerheidsanalyse wel meet

We vergelijken de twee voorspellers paarsgewijs op dezelfde 32 testcellen. Een exacte McNemar-test kijkt naar cellen waarop slechts één van beide modellen correct is. Daarnaast resamplen we binnen iedere klasse 10.000 keer de testcellen om een interval voor het accuracyverschil te krijgen.

Die bootstrap meet onzekerheid door de kleine teststeekproef. Zij omvat niet:

- een nieuwe hardwaremeting met onafhankelijke shots;
- calibratiedrift op een andere dag;
- een nieuwe train/testsplit;
- onzekerheid door het kiezen van de modelgrid;
- biologische variatie tussen donoren.

In [deel 6](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/) openen we de vaste testscore en bespreken we wat 16 tegen 17 werkelijk betekent.

## Bronnen

- [Lokale hardwareanalyse met training-only selectie](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_hardware_analysis.py)
- [QOS-paper: LS-SVM-classificatietaak](https://arxiv.org/abs/2604.07639)

