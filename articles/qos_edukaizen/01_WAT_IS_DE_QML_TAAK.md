# Wat is de QML-taak? Cellen classificeren, geen genen opzoeken

Het korte antwoord is: **wij bouwen geen genenzoekmachine**. We voeren een supervised binaire classificatietaak uit. Het model krijgt het genexpressieprofiel van een cel en voorspelt tot welke van twee sterk verwante typen CD4-T-cellen die cel behoort.

## Wat staat er in PBMC68k?

PBMC staat voor *peripheral blood mononuclear cells*: witte bloedcellen uit perifeer bloed, waaronder T-cellen, B-cellen, natural-killercellen en monocyten. Bij single-cell RNA-sequencing wordt voor iedere cel geteld hoeveel RNA-moleculen van elk gen zijn waargenomen. Het resultaat is een grote matrix:

```math
X \in \mathbb{R}^{N \times D}.
```

Een rij is één cel. Een kolom is één gen. In onze lokale 10x-versie heeft iedere rij 32.738 mogelijke genen. De meeste waarden zijn nul: in één cel wordt maar een klein deel van alle genen waargenomen. Daardoor is de matrix hoogdimensionaal én sparse.

Naast de meetmatrix bestaat een annotatiebestand. Daarin staat voor een deel van de cellen een door de oorspronkelijke analyse toegekend celtype. Dat bestand functioneert tijdens supervised learning als de bron van de labels. Het is dus wel een database, maar het model doet geen vraag als “zoek gen IL7R op”. Het leert een beslisregel uit complete celprofielen.

## Onze concrete binaire vraag

We selecteren uit de geannoteerde dataset twee klassen:

- positief label: `CD4+/CD25 T Reg`, 6.187 beschikbare cellen;
- negatief label: `CD4+/CD45RO+ Memory`, 3.061 beschikbare cellen.

Samen vormen zij 9.248 kandidaatcellen. Voor de hardwarepilot is daaruit een vastgelegde, gebalanceerde subset gebruikt: 32 trainingscellen en 32 testcellen, in beide delen zestien cellen per klasse.

De voorspelling voor een nieuwe cel is dus:

```text
genexpressievector van één cel
        ↓
quantumfeaturemap en metingen
        ↓
klassieke beslisregel
        ↓
T-regulerende cel of CD4-geheugencel
```

Dit is een moeilijke kleine-data-opgave. De twee klassen zijn biologisch verwant en het experiment gebruikt veel minder trainingsvoorbeelden dan genen. Met 32 trainingscellen tegenover 32.738 ruwe kenmerken kan een flexibel model de training gemakkelijk onthouden zonder op nieuwe cellen te generaliseren.

## Waar komt quantum machine learning binnen?

De quantumprocessor vervangt niet de hele analyse. Hij fungeert als **featuremap**. De klassieke genexpressie wordt gecomprimeerd tot circuitparameters, waarna het circuit een quantumtoestand voorbereidt en interacties tussen qubits aanbrengt. Metingen leveren vervolgens 405 getallen per cel. Een gewone klassieke classifier leert op die gemeten featurevectoren.

Dit is daarom een hybride QML-pijplijn:

| Stap | Klassiek of quantum? |
| --- | --- |
| RNA-matrix en labels laden | klassiek |
| actieve genen selecteren en paren hashen | klassiek |
| toestand voorbereiden en qubits laten interageren | quantum |
| Pauli-observabelen meten | quantum |
| classifier trainen en celtype voorspellen | klassiek |

De wetenschappelijke vraag is niet alleen of deze combinatie een redelijke classificatie oplevert. De diepere QOS-vraag is of de quantumrepresentatie informatie uit een zeer grote featurewereld kan behouden met een veel kleiner intern geheugen dan een algemeen klassiek streamingmodel.

## Classificatie is iets anders dan biomarkerontdekking

Een belangrijke beperking is interpretatie. Omdat genparen deterministisch in 160 buckets worden gehasht, kunnen verschillende genparen in dezelfde bucket belanden. Het model kan daardoor een bruikbaar patroon leren zonder dat iedere gemeten quantumfeature aan één specifiek gen of biologisch mechanisme kan worden gekoppeld.

Onze huidige taak beantwoordt dus niet rechtstreeks:

- welk individueel gen veroorzaakt het verschil;
- welke pathway is biologisch beslissend;
- of het label klinisch betrouwbaar is;
- of de classifier naar andere donoren generaliseert.

Daarvoor zijn aparte interpretatie-, validatie- en cohortstudies nodig. De huidige pilot is een computationeel experiment met een biologisch reële, hoogdimensionale invoer.

## Waarom deze taak toch relevant is

Single-cell data groeit snel in zowel het aantal cellen als het aantal meetbare kenmerken. De uitdaging is niet dat een laptop één gen niet kan vinden, maar dat leren uit grote, veranderende en sparse matrices geheugen en dataverkeer vereist. QOS formuleert precies voor zulke situaties een streamingmodel: een sample wordt verwerkt, de interne toestand wordt bijgewerkt en het sample hoeft niet permanent in het werkgeheugen te blijven.

In [deel 2](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-oracle-sketching-theorie/) bekijken we die theorie. Dan wordt duidelijk waarom de paper over machinegrootte en sampletoegang spreekt, en waarom een hogere testaccuracy op zichzelf nog geen quantumvoordeel bewijst.

## Bronnen

- [10x PBMC68k / Fresh 68k PBMCs](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
- [QOS-paper: dataset- en classificatieopzet](https://arxiv.org/abs/2604.07639)
- [Officiële PBMC68k-code in de QOS-repository](https://github.com/haimengzhao/quantum-oracle-sketching/tree/main/real_datasets)

