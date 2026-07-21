# Van PBMC68k-genexpressie naar 40 qubits

Een quantumprocessor met veertig qubits kan niet simpelweg 32.738 genkolommen als 32.738 afzonderlijke qubits opslaan. De cruciale stap is daarom de encoding: een deterministische, reproduceerbare vertaling van een sparse genexpressievector naar vier compacte blokken van elk veertig getallen.

## Stap 1: één cel als sparse vector

Voor iedere cel bevat PBMC68k UMI-tellingen per gen. In de geselecteerde twee celtypen zijn gemiddeld ongeveer zeshonderd genen per pilotcel actief. De nulwaarden hoeven niet te worden verwerkt. We behouden per cel de 48 actiefste genen, zodat de circuitbouw begrensd en voor iedere cel even groot blijft.

Dit is een praktische hardwarekeuze. Zij kan informatieve zwakke genen verwijderen en is dus onderdeel van de modeldefinitie, niet slechts een technische optimalisatie.

## Stap 2: genparen in plaats van losse genen

Losse genexpressie kan een celtype onderscheiden, maar interacties kunnen aanvullende structuur dragen. Uit 48 actieve genen ontstaan:

```math
\binom{48}{2}=1128
```

genparen. Het gewicht van een paar is het product van de log-getransformeerde expressiewaarden. Daardoor domineren extreem grote ruwe tellingen minder sterk, terwijl co-activatie behouden blijft.

Deze pairwise featurewereld is veel groter dan het aantal qubits. We materialiseren geen volledige matrix met alle mogelijke genparen. Ieder daadwerkelijk waargenomen paar wordt direct verwerkt.

## Stap 3: deterministisch hashen naar 160 buckets

Elk genpaar wordt met hash-seed 7 toegewezen aan een van 160 buckets. Waarden die in dezelfde bucket vallen worden opgeteld. Feature hashing vermijdt een grote expliciete projectiematrix, maar veroorzaakt botsingen: verschillende paren kunnen dezelfde bucket delen.

De 160 waarden worden herschikt tot vier blokken:

```text
160 hashbuckets = 4 coherente blokken × 40 qubits
```

Elk niet-leeg blok wordt afzonderlijk L2-genormaliseerd. In de pilot hadden alle vier de blokken norm één. De encoding is labelvrij: dezelfde transformatie wordt toegepast ongeacht of de cel een regulatoire of geheugen-T-cel is.

## Stap 4: streaming door hetzelfde register

De vier blokken worden niet op vier verschillende quantumchips gezet. Ze worden achtereenvolgens in hetzelfde register van veertig qubits geüpload. Na iedere upload volgt een interactielaag. Zo beïnvloedt een later blok de toestand die door eerdere blokken is opgebouwd.

Dit is het QOS-geïnspireerde deel: samples of featureblokken worden sequentieel in een compacte quantumtoestand verwerkt. Het is echter geen exacte implementatie van de volledige QOS-oracle, QSVT en classical-shadowketen uit de paper.

De interactiesterkte schaalt in de gekozen architectuur als de wortel van de registerbreedte:

```math
\sqrt{40}=6.324555\ldots
```

Deze `sqrt(q)`-keuze kwam uit een voorafgaande labelvrije resourcepreflight. Daarbij werd gekeken of de circuitfamilie structureel interessant en numeriek beheersbaar bleef wanneer de breedte toenam. De testlabels zijn niet gebruikt om deze architectuur te kiezen.

## Wat gaat verloren?

Compressie is geen magie. Er gaan vier soorten informatie verloren of raken vermengd:

- genen buiten de top 48 van een cel verdwijnen;
- hashbotsingen voegen verschillende genparen samen;
- vier genormaliseerde blokken verliezen absolute schaalinformatie;
- met 128 shots worden quantumverwachtingswaarden slechts beperkt nauwkeurig geschat.

Daar staat tegenover dat het circuit een niet-lineaire, interfererende representatie van de 160 gehashte waarden kan produceren. De hoop is dat relevante klassestructuur in meetbare correlaties terechtkomt.

## Is dit biologisch interpreteerbaar?

Niet op genniveau zonder extra administratie. Om later te kunnen zeggen welke genparen een bucket domineren, moeten we per cel of per cohort een reverse map van genparen naar buckets bewaren. Dat is mogelijk, maar vergroot de klassieke sidecar en is in deze pilot niet de primaire readout.

De huidige representatie is ontworpen voor classificatie en hardware-uitvoerbaarheid. Zij is geen pathway-database en geen biomarkeranalyse. Een vervolgstudie kan collision-audits, marker-enrichment en stabiliteit over meerdere hash-seeds toevoegen.

## Waarom niet gewoon PCA?

PCA, sparse lineaire modellen en feature hashing zijn sterke klassieke baselines. Ze moeten altijd worden meegerekend. De quantumroute wordt pas interessant wanneer zij met een compacte toestand voorspellende structuur bewaart die onder dezelfde resourcegrens moeilijk klassiek te behouden is.

In deze pilot gebruiken we daarom twee klassieke vergelijkingen: een volledig raw-gene log1p-model en een model op exact dezelfde 160 gehashte features. In [deel 6](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/) zien we dat de raw-gene logistische classifier uiteindelijk één testcel meer correct had dan de hardwarefeaturemap.

## Reproduceerbare parameters

| Parameter | Waarde |
| --- | --- |
| genen per cel in bronmatrix | 32.738 |
| maximaal actieve genen | 48 |
| paar-events per cel | 1.128 |
| hash-seed | 7 |
| hashbuckets | 160 |
| blokken | 4 |
| qubits per blok | 40 |
| interactieschaal | `sqrt(q)` |

In [deel 4](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qos-naar-40-qubit-hardware-fire-opal/) zetten we deze blokken om in 192 meetcircuits voor IBM Fez.

## Bronnen

- [Hardware-validatierunner](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/agent/add-q40-fire-opal-hardware-milestone/qiskit_qos_pbmc_q40_sqrtq_b4_fireopal_validate.py)
- [PBMC68k-loader en annotaties](https://github.com/BramDo/qlab-ml-adv-all-runners/blob/main/qiskit_qos_pbmc68k_utils.py)
- [Officiële QOS real-datasetcode](https://github.com/haimengzhao/quantum-oracle-sketching/tree/main/real_datasets)

