# Edukaizen-publicatie

Gepubliceerd op 20 juli 2026 als een pagina-hub met acht onderliggende
artikelen. Op 21 juli 2026 zijn de bestaande pagina's in-place bijgewerkt met
het uitgevoerde 60-qubitresultaat. Op 22 juli is een negende, educatief
4-qubithoofdstuk toegevoegd; er zijn geen bestaande pagina's gedupliceerd:

- Gecombineerde projectpagina: https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/
- Engelse spiegel: https://edukaizen.nl/quantum-oracle-sketching-qml-gene-expression/
- WordPress-hubpagina: `498`
- WordPress-artikelpagina's: `499` tot en met `505`, plus `516` en `610`
- Hoofdmenu-item `QOS QML`: `506`
- Taalkeuzes onder het menu-item: `Nederlands` (`563`) en `English` (`564`)
- Extra menu-link `Beginnershandleiding 4q`: `614`

De artikelen staan in deze volgorde:

1. [Wat is de QML-taak? Cellen classificeren, geen genen opzoeken](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/wat-is-de-qml-taak-cellen-classificeren/) — pagina `504`
2. [De theorie van Quantum Oracle Sketching](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-oracle-sketching-theorie/) — pagina `503`
3. [Van PBMC68k-genexpressie naar 40 qubits](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/pbmc68k-van-genexpressie-naar-qubits/) — pagina `500`
4. [Van JAX naar een 40-qubit hardwarecircuit](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qos-naar-40-qubit-hardware-fire-opal/) — pagina `501`
5. [405 observabelen en een lekvrije classifier](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/quantum-readout-405-observabelen-classifier/) — pagina `499`
6. [Het 40-qubitresultaat: hardware 16, klassiek 17](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/resultaat-hardware-versus-klassiek/) — pagina `505`
7. [Wat is nog nodig voor quantumvoordeel?](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/route-naar-quantumvoordeel-qml/) — pagina `502`
8. [Het 60-qubitresultaat: hardware 17, lineair 16, RBF 14](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/voorstel-60-qubit-qml-vervolgstudie/) — pagina `516`
9. [Beginnershandleiding QML: van UMI-telling naar een 4-qubitcircuit](https://edukaizen.nl/quantum-oracle-sketching-qml-genexpressie/qml-beginnershandleiding-umi-naar-4-qubit-circuit/) — pagina `610`

Na publicatie zijn alle negen publieke URL's met HTTP-status `200` gecontroleerd.
De seriemarkeringen, vorige/volgende-navigatie, de kerngetallen `16/32` en
`17/32`, de actieve MathJax-verwerking en de nieuwe hoofdnavigatie zijn live
geverifieerd. Op 21 juli is hoofdstuk 8 vervangen door het daadwerkelijk
uitgevoerde 60-qubitexperiment: hardware `17/32`, lineair `16/32`, RBF `14/32`,
Fire Opal-action `2335848`, 26 quantumseconden volgens het dashboard en circa
8 minuten 33 seconden van indiening tot retrieval. Tegenover de onvoltooide
MPS-poging van 2.577 seconden is dit een lokale ondergrens van 99,1× op
kerneltijd en 5,0× inclusief retrieval. De acht bijgewerkte
Nederlandse en Engelse URL's gaven daarna HTTP-status `200`; alle vereiste
resultaatmarkeringen en navigatie waren publiek aanwezig, zonder ruwe
Markdown-links.

Het QML-resultaat is op 21 juli tevens als `local_runtime_lower_bound` toegevoegd
aan de bestaande [Pro Student Quantum Advantage List](https://edukaizen.nl/pro-student-quantum-advantage-list/),
WordPress-pagina `451`. De publieke pagina vermeldt de 99,1× kernelondergrens,
de 5,0× end-to-endondergrens en alle voorspellende en MPS-beperkingen.
Op dezelfde datum is een volledige Engelse spiegel gepubliceerd. Iedere hub en
ieder hoofdstuk bevat een directe taalwissel naar de overeenkomstige pagina.
De projectpagina is daarna samengevoegd tot één tweetalige pagina met twee
keuzeknoppen en beide artikellijsten. De losse Nederlandse artikelmenu-items
`507`–`513` en `517` en de Engelse menutak `527`–`535` zijn verwijderd; de
bijbehorende artikelpagina's zijn behouden.

Op 21 juli 2026 is de volledige tweetalige reeks opnieuw in-place bijgewerkt om
de relatie tot de QOS-paper preciezer te maken. De projectpagina's en de
relevante theorie-, hardware-, route- en 60q-hoofdstukken onderscheiden nu:

- de letterlijke 4q flat-QOS sampling-kern (`D=16`, `M=64`, Fire Opal-action
  `2334156`, gemiddelde Hellinger-fideliteit `0,990104`);
- de 40q/60q PBMC68k-circuits als QOS-geïnspireerde NISQ-featuremaps, zonder
  letterlijke sampling-oracle, QSVT of exacte classical-shadowreadout;
- de nog niet end-to-end op hardware uitgevoerde volledige paperroute.

Alle achttien bestaande publieke URL's zijn daarna opnieuw met HTTP-status
`200`, serienavigatie en de nieuwe Nederlandstalige en Engelstalige
claimmarkeringen geverifieerd. Er zijn geen pagina's of slugs toegevoegd.

Op 22 juli 2026 is de beginnershandleiding als hoofdstuk 9 gepubliceerd. De
pagina volgt één echte PBMC68k-cel van UMI-telling naar vier rotatiehoeken, een
4-qubitcircuit, een `16 x 16` unitaire matrix, acht Z/ZZ-features en een
klassieke classifier. Het resultaat `7/16` tegenover `9/16` voor dezelfde vier
klassieke genfeatures wordt expliciet als onderwijsresultaat zonder
advantageclaim vermeld. De gecombineerde QML-projectpagina en hoofdstuk 8
linken naar het nieuwe hoofdstuk. Hoofdmenu-item `614` hangt onder `QOS QML`
(`506`). De pagina, hub, voorgaande-hoofdstuklink, het homepage-menu, twaalf
MathJax-formules en vijf GitHub-afbeeldingen zijn publiek met HTTP-status `200`
gecontroleerd. De broncode, DOCX, WordPress-bron en het LinkedIn-artikel zijn
via GitHub-PR `#3` opgenomen in `main` met mergecommit `4c1e50c`.
