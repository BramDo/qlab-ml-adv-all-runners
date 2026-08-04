# Beginnershandleiding: ML_adv installeren en uitvoeren

Deze handleiding begint bij nul: de repository downloaden, een losse
Python-omgeving maken, de installatie controleren en de aanbevolen hoofdroute
uitvoeren. De commando's zijn bedoeld voor Windows 10/11 met WSL2 en Ubuntu.

## Welke route moet ik uitvoeren?

Begin met:

```text
qiskit_qos_pbmc68k_q4_educational.py
```

Dit is de aanbevolen hoofdroute voor beginners omdat zij:

- echte PBMC68k single-cell RNA-data gebruikt;
- een complete keten van data naar vier qubits en een classifier laat zien;
- lokaal met Qiskit Aer draait;
- geen IBM-account, Fire Opal-account of betaalde quantumtijd gebruikt;
- aansluit op de Edukaizen-beginnershandleiding.

De 40- en 60-qubitroutes zijn onderzoeksroutes. Start die niet als eerste en
dien nooit hardwarejobs in zonder het bijbehorende runbook, geldige accounts
en expliciete toestemming voor het quantumtijdverbruik.

## 1. Benodigdheden

Je hebt nodig:

- Windows 10 of 11;
- WSL2 met Ubuntu;
- een internetverbinding voor Git, Python-pakketten en de PBMC68k-data;
- minimaal ongeveer 2 GB vrije schijfruimte;
- geen quantumaccount voor de beginnersroute.

Open PowerShell als gewone gebruiker en controleer WSL:

```powershell
wsl --status
wsl --list --verbose
```

Als Ubuntu nog ontbreekt:

```powershell
wsl --install -d Ubuntu
```

Herstart Windows als daarom wordt gevraagd. Open daarna Ubuntu vanuit het
Startmenu en rond de eerste gebruikersconfiguratie af.

## 2. Git en Python in Ubuntu installeren

Voer dit uit in de Ubuntu-terminal:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Controleer de installatie:

```bash
git --version
python3 --version
```

Python 3.12 is de momenteel gecontroleerde versie. Een andere recente
Python 3-versie kan werken, maar is niet de vaste referentie van deze gids.

## 3. Repository downloaden

Kies een map in je Linux-homefolder. Dat voorkomt problemen met Windows-
regeleinden en Synology-bestandsattributen.

```bash
cd ~
mkdir -p qlab
cd qlab
git clone https://github.com/BramDo/qlab-ml-adv-all-runners.git ML_adv
cd ML_adv
git switch main
git pull --ff-only origin main
```

Controleer dat je op de juiste plaats staat:

```bash
pwd
git status --short --branch
```

De verwachte branch is `main`. Direct na het klonen hoort de werkmap schoon
te zijn.

## 4. Losse Python-omgeving maken

Maak de omgeving in de repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-beginner.txt
```

Iedere nieuwe terminal activeert de omgeving opnieuw met:

```bash
cd ~/qlab/ML_adv
source .venv/bin/activate
```

Controleer de belangrijkste pakketten:

```bash
python -c "import qiskit, qiskit_aer; print('Qiskit', qiskit.__version__); print('Aer', qiskit_aer.__version__)"
```

Voor deze gids worden `Qiskit 2.5.0` en `Aer 0.17.2` verwacht.

## 5. Korte installatiecontrole

Voer eerst de kleine regressietests uit:

```bash
python -m pytest -q \
  tests/test_qiskit_qos_pbmc68k_q4_educational.py \
  tests/test_qiskit_qos_pbmc68k_q4_explain.py \
  --basetemp output/pytest-beginner
```

De tests moeten eindigen met `passed`. Een waarschuwing is niet automatisch
een fout; regels met `FAILED` of `ERROR` zijn dat wel.

## 6. De hoofdroute uitvoeren

Voer vanuit de root van de repository uit:

```bash
python qiskit_qos_pbmc68k_q4_educational.py \
  --shots 512 \
  --json-out output/pbmc68k_q4_educational.json
```

Bij de eerste uitvoering downloadt het programma automatisch:

- de officiële 10x PBMC68k-countmatrix, ongeveer 124 MB gecomprimeerd;
- de bijbehorende celtypeannotaties, ongeveer 5 MB.

Deze bestanden komen in `data_cache/pbmc68k/`. Volgende uitvoeringen gebruiken
de cache en hoeven ze niet opnieuw te downloaden.

## 7. Verwachte uitvoer

Met de vaste seed en 512 shots hoort de uitvoer onder meer te tonen:

```text
PBMC68k four-qubit educational simulator
Selected genes: IER2, ACTG1, LIMD2, GLTSCR2
Quantum features: 0.438 (7/16)
Classical same four genes: 0.562 (9/16)
```

Ook wordt het vier-qubitcircuit in de terminal getekend. Het volledige
machineleesbare resultaat staat daarna in:

```text
output/pbmc68k_q4_educational.json
```

Controleer het resultaat zonder het bestand handmatig te doorzoeken:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output/pbmc68k_q4_educational.json")
data = json.loads(path.read_text(encoding="utf-8"))
result = data["results"]
print("quantum:", result["quantum_correct"], "/", result["test_cells"])
print("classical:", result["classical_correct"], "/", result["test_cells"])
print("claim:", data["claim_boundary"])
PY
```

Dit resultaat is een reproduceerbare onderwijsdemonstratie. De quantumfeatures
scoren hier lager dan dezelfde vier klassiek gebruikte genfeatures. Het is dus
geen quantumvoordeel.

## 8. Figuren en uitleg opnieuw maken

Na een geslaagde hoofdroute kun je de tabellen, toestand, unitaire matrix en
figuren uit de beginnershandleiding opnieuw genereren:

```bash
python qiskit_qos_pbmc68k_q4_explain.py \
  --shots 512 \
  --output-dir output/pbmc68k_q4_explainer
```

De conceptuele uitleg staat in:

- `docs/beginner/README.md`;
- `docs/beginner/qml-van-umi-naar-circuit.docx`;
- de [Edukaizen-beginnershandleiding](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qml-beginnershandleiding-umi-naar-4-qubit-circuit/).

## 9. Daarna: de letterlijke QOS-kern bestuderen

De beginnersroute hierboven is een gewone vier-qubitfeaturemap en geen
letterlijke QOS-implementatie. De volgende inhoudelijke stap is daarom
`qiskit_official_qos_sampling_port.py`, die kleine Qiskit- en JAX-kernen
vergelijkt. Deze gevorderde route heeft aanvullende JAX/QSP-dependencies. Lees
voor de installatie en de reeds geverifieerde referentiewaarden eerst
`OFFICIAL_QOS_REPRO_STATUS_2026-04-15.md`; verander de werkende
beginnersomgeving niet voordat de hoofdroute is geslaagd.

## 10. Gevorderd: IBM Fez via Fire Opal

Dit is de gecontroleerde 60-qubit onderzoeksroute. Fire Opal is hierbij de
laag die de circuits voor de gekozen IBM-backend valideert, uitvoert en later
via een opgeslagen action-ID terughaalt. Deze route is **niet nodig** voor de
beginnersroute uit hoofdstuk 6.

Gebruik voor de hieronder lokaal gecontroleerde combinatie bij voorkeur de
bestaande Qiskit-omgeving met:

```text
fire-opal 11.1.0
qiskit-ibm-runtime 0.48.0
```

In de projectomgeving installeer je die aanvullingen zo:

```bash
source .venv/bin/activate
python -m pip install "fire-opal==11.1.0" "qiskit-ibm-runtime==0.48.0"
export PYTHONPATH=.
PY=python
```

Zet credentials alleen als omgevingsvariabelen of gebruik een reeds veilig
opgeslagen Qiskit-account. Zet sleutels nooit in een commando, notebook,
JSON-resultaat of Git-bestand:

```bash
export QCTRL_API_KEY='...'
export IBM_CLOUD_API_KEY='...'
export IBM_QUANTUM_CRN='...'
```

Controleer uitsluitend of de variabelen aanwezig zijn, zonder hun inhoud af
te drukken:

```bash
test -n "$QCTRL_API_KEY" && echo "QCTRL_API_KEY is ingesteld"
test -n "$IBM_CLOUD_API_KEY" && echo "IBM_CLOUD_API_KEY is ingesteld"
test -n "$IBM_QUANTUM_CRN" && echo "IBM_QUANTUM_CRN is ingesteld"
```

### Stap A: dataset, modules en splits lokaal vastzetten

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py prepare
```

Daarna kan de lokale MPS-screen worden uitgevoerd:

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py local-screen
```

Deze simulatie kan uren duren en veel geheugen gebruiken. Ga pas verder als
de gates in `Q60_MODULE_B4_RUNBOOK.md` zijn gehaald. Een onvoltooide MPS-run
is geen bewijs dat de ideale 60-qubitroute beter is.

### Stap B: sentinelbundle bouwen, nog zonder provider

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_validate.py --phase sentinel
```

Dit maakt de bevroren OpenQASM-bundle voor de kleine sentinelproef. Er wordt
nog niets naar IBM of Q-CTRL gestuurd.

### Stap C: Fire Opal validate-only

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_validate.py \
  --phase sentinel --validate --force
```

Dit doet alleen device-discovery en `fireopal.validate`. Het script bevat
bewust geen execute- of retrievalpad; de verwachte quantumtijd is dus nul.
Een geslaagde validatie betekent alleen dat de bundle providercompatibel is,
niet dat er toestemming is gegeven om hardware te gebruiken.

### Stap D: provider-vrij uitvoerplan maken

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py plan --phase sentinel
```

Controleer in het plan backend, aantal circuits, shots en maximum budget. De
sentinelfase is vastgezet op 192 circuits maal 128 shots: 24.576 shots, circa
30--40 quantumseconden en een stopgrens van 50 quantumseconden. Die grens is
een menselijke budgetcontrole; de externe API handhaaft haar niet automatisch.

### Stap E: alleen na afzonderlijke expliciete toestemming indienen

Het `submit`-subcommando kan werkelijk `fireopal.execute` aanroepen. Het
vereist daarom de exacte bevestigingstekst die het plan toont. Kopieer die
niet vooraf naar scripts en behandel een geslaagde validatie nooit als
indientoestemming. Na indienen wordt de action-ID direct in een receipt
opgeslagen; er is geen automatische retry, verhoging van shots of wisseling
van backend.

De grote fase komt pas na analyse en goedkeuring van de sentinel. Zij bevat
1.536 circuits maal 128 shots (196.608 shots), met circa 240--320 geschatte
quantumseconden en een stopgrens van 400 quantumseconden.

### Stap F: bestaand resultaat ophalen en analyseren

Als er al een receipt met action-ID bestaat, haalt dit commando uitsluitend
die actie op en kan het niets opnieuw indienen:

```bash
$PY qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py retrieve --phase sentinel
```

Analyseer daarna het opgehaalde resultaat met de precieze paden uit de receipt:

```bash
$PY qiskit_qos_pbmc68k_q60_module_pipeline.py hardware-analysis \
  --hardware-result PATH_NAAR_RESULT.json \
  --output PATH_NAAR_ANALYSE.json
```

De eerder voltooide sentinel op IBM Fez via Fire Opal gebruikte action
`2335848`: 192 circuits, 128 shots en 26 gerapporteerde quantumseconden. Op de
bevroren taak scoorde q60 `17/32`, tegenover lineair `16/32` en RBF `14/32`.
Dat is een klein taakgebonden resultaat, geen algemeen quantumvoordeel en ook
geen volledig gematchte foutvergelijking. Raadpleeg voor ieder hardwaregebruik
altijd eerst `Q60_MODULE_B4_RUNBOOK.md`.

## 11. Welke routes niet als eerste uitvoeren?

Voer als beginner niet meteen de volgende scripts uit:

- `qiskit_qos_pbmc68k_q60_module_fireopal_pilot.py`;
- `qiskit_qos_pbmc68k_q40_fireopal_pilot.py`;
- scripts met `ibm-hardware`, `--submit` of een Fire Opal-actie.

Daarvoor zijn credentials, providerconfiguratie, budgetbewaking en aanvullende
validatie nodig. De veilige volgorde is:

```text
installatie -> tests -> 4q simulator -> 4q uitleg -> QOS-reproductiestatus
             -> pas daarna validate-only -> afzonderlijk geautoriseerde hardware
```

Voor de onderzoeksroutes gebruik je:

- `RUNNERS.md` voor het overzicht;
- `Q40_WORKING_CHAIN_2026-04-16.md` voor de begrensde 40q-route;
- `Q60_MODULE_B4_RUNBOOK.md` voor de 60q-route.

## 12. Veelvoorkomende problemen

### `ModuleNotFoundError`

Activeer eerst de omgeving en installeer de requirements opnieuw:

```bash
source .venv/bin/activate
python -m pip install -r requirements-beginner.txt
```

### Download van PBMC68k mislukt

Controleer de internetverbinding en verwijder geen volledige cache als slechts
één bestand ontbreekt. De twee verwachte bestanden zijn:

```text
data_cache/pbmc68k/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
data_cache/pbmc68k/68k_pbmc_barcodes_annotation.tsv
```

Een bestand van nul bytes is een mislukte download en mag afzonderlijk worden
verwijderd voordat je het commando opnieuw uitvoert.

### Git toont honderden wijzigingen op een Synology-map

Gebruik bij voorkeur de Linux-homefolder uit deze handleiding. Staat de repo
toch onder `/mnt/c/.../SynologyDrive`, controleer de status dan met WSL Git:

```bash
git status --short --branch
git diff --ignore-cr-at-eol --stat
```

### Het programma vraagt om IBM-credentials

Dan voer je niet de aanbevolen beginnersroute uit. Stop het proces en gebruik
precies het commando uit hoofdstuk 6; dat gebruikt alleen de lokale simulator.

## 13. Samenvatting

Voor een eerste volledige uitvoering heb je uiteindelijk slechts dit nodig:

```bash
git clone https://github.com/BramDo/qlab-ml-adv-all-runners.git ML_adv
cd ML_adv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-beginner.txt
python -m pytest -q tests/test_qiskit_qos_pbmc68k_q4_educational.py \
  tests/test_qiskit_qos_pbmc68k_q4_explain.py --basetemp output/pytest-beginner
python qiskit_qos_pbmc68k_q4_educational.py --shots 512 \
  --json-out output/pbmc68k_q4_educational.json
```

De hoofdroute is dus de **vier-qubit PBMC68k educational simulator**. Zij is
klein, echt-data-gedreven, controleerbaar en veilig om lokaal uit te voeren.
