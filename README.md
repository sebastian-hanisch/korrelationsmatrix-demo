# Korrelationsmatrix bei vielen Variablen: was ist noch wahr? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-korrelationsmatrix-demo.streamlit.app/)** -->

Fünftes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: eine Korrelation ist eine Zahl, eine Matrix aus vielen Kennzahlen ist keine Menge unabhängiger Zahlen), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo) und [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo). Es ist die **erste Demo des Portfolios mit dem Spektrum einer geschätzten Korrelationsmatrix**: **p Kennzahlen** eines Standorts über **n Wochen**, die wahre Korrelationsmatrix ist **bekannt** (Einheit, Kette, ein Faktor, fünf Blöcke). Die Fragen: **Wie viele Paare sind echt, wenn man p(p − 1)/2 zugleich betrachtet? Was ist an den Eigenwerten und der Konditionszahl der Stichprobenmatrix noch wahr? Welcher Schätzer verbessert die Matrix wann?**

## Warum dieses Problem

Eine Korrelationsmatrix aus n Wochen sieht wie ein verkleinertes Abbild der Wahrheit aus und ist es nicht, sobald p nicht klein gegen n ist. Befunde: **bei reinem Rauschen (wahre Einheitsmatrix) hat bei 50 Kennzahlen und 30 Wochen jeder Datensatz mindestens ein naiv „signifikantes“ Paar** (100,0 %), das größte |r| liegt im Mittel bei **0,597** (Schwelle des einzelnen Tests 0,361); **die Eigenwerte** einer Stichprobenmatrix der Einheit liegen nicht bei 1: bei p/n = 0,9 ist der kleinste 0,003 und die **Konditionszahl 1219,9** statt 1; bei p > n hat sie höchstens den Rang n − 1; **kein Schätzer gewinnt überall**: die lineare Schrumpfung halbiert den Fehler bei der Kette (9,08 → 4,86 bei n = 30) und **nützt beim Faktor nichts** (7,97 → 7,59), das Abschneiden der Eigenwerte halbiert ihn beim Faktor (7,97 → 4,17) und **schadet bei der Kette** (5,20 gegen 4,48 bei n = 120), die nichtlineare Schrumpfung liegt bei den Blöcken vorn (6,33 gegen 8,84 bei n = 30) und ist bei der Einheit schlechter als die lineare (0,64 gegen 0,04 bei n = 120).

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-5-korrelationsmatrix.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und weniger Läufen. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben, mit diesen Ausnahmen:

- **Maximum-Schwelle bei p = 50, n = 30:** 0,687 mit 3,1 % Fehlalarm (Plan: 0,676 mit 5,1 %); im Raster hält sie 3,1 bis 7,8 %, Bonferroni 4,1 bis 6,4 % (Plan: 4,4 bis 8,2 %).
- **Fehler bei der Einheit, n = 30:** Abschneiden 0,26 und lineare Schrumpfung 0,35 (Plan: 0,22 und 0,35); das Abschneiden liegt hier vorn.
- **Faktor, n = 30:** lineare Schrumpfung 7,59 (Plan: 7,77), Abschneiden 4,17 (4,33), nichtlinear 4,65 (4,84).
- **Inverse bei n = 60:** relativer Fehler der Stichprobenmatrix 17,31 bis 19,61 (Plan: 16,4 bis 18,1); die Zahl hängt stark vom Datensatz ab (die Matrix ist fast singulär).
- **Konditionszahl bei q = 0,9:** 1219,9 (Plan: 1 199).
- **Nichtlineare Schrumpfung bei p > n:** braucht mindestens 13 Wochen (die Bandbreite n^(−1/3) muss unter 1/√5 liegen); die Regler erlauben 20 bis 1 000.
- **Ohne Gegenstück im Plan:** die Wahrheit „Kette“ heißt in der Oberfläche „Kette (0,5 hoch Abstand)“; die Preset-Hilfetexte nennen Zahlen des Rasters.

## Modell

Eine wahre Korrelationsmatrix R (Erzeuger `kmx_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`: `prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py`): **Einheit** (alle Kennzahlen unabhängig); **Kette** ρ^|i−j| mit ρ = 0,5; **ein Faktor** (alle Paare 0,36, gemeinsame Auslastung mit Ladung 0,6); **Blöcke** zu zehn Kennzahlen mit 0,5 innerhalb (bei p = 50 fünf Blöcke). Die n Wochen sind unabhängig und normalverteilt mit der Matrix R (Cholesky-Faktor). Die **Stichprobenmatrix** R̂ ist die Korrelationsmatrix der n Wochen.

## Methodik

- **Schwellen für das größte |r|** (`kmx_algorithm.py`, ohne scipy zur Laufzeit): die **naive** des einzelnen t-Tests, die **Bonferroni-Schwelle** (Niveau 5 %/Paare) und die **Maximum-Schwelle**: das Quantil des größten |r| aus B Datensätzen, in denen jede Spalte für sich zufällig umgeordnet wurde (die Daten bleiben, die Abhängigkeit zwischen den Spalten geht verloren).
- **Spektrum:** Eigenwerte der Stichprobenmatrix gegen die **Marchenko–Pastur-Verteilung** mit den Rändern (1 ± √(p/n))², Rang und Konditionszahl, Zahl der Eigenwerte über dem oberen Rand.
- **Fünf Schätzer** der Korrelationsmatrix: **Stichprobenmatrix**, **lineare Schrumpfung** nach Ledoit–Wolf (mit der Einheitsmatrix mischen, Gewicht aus den Daten), **Abschneiden der Eigenwerte** (alle bis zum oberen Marchenko–Pastur-Rand durch ihr Mittel ersetzen), **analytische nichtlineare Schrumpfung** nach Ledoit–Wolf (Kerndichte der Eigenwerte und Hilbert-Transformierte) und das **Orakel** (Eigenvektoren der Stichprobenmatrix, Eigenwerte uᵀRu mit der wahren Matrix, nur zum Vergleich). Fehler: Frobenius-Norm der Differenz zur wahren Matrix; dazu der relative Fehler der Inversen.
- **Experimente:** (1) Rauschmaximum, (2) Spektrum und Konditionszahl, (3) wie viele Eigenwerte echt sind, (4) welcher Schätzer wann, (5) mehr Kennzahlen als Wochen.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/matrix_grid.json`, 388 Sekunden auf 10 Prozessen): Rauschmaximum (9 Zellen: p = 10, 50, 200 · n = 30, 100, 500, je 4 · 250 Datensätze, die Maximum-Schwelle aus 600 Referenzmatrizen), Spektrum (4 Zellen: p = 100, p/n = 0,1 / 0,5 / 0,9 / 1,5, je 4 · 100), Eigenwerte über dem Rand (16 Zellen: vier Wahrheiten · n = 30, 60, 120, 500 bei p = 50, je 4 · 100) und Fehler der Schätzer (19 Zellen: dieselben 16 mit je 4 · 50 Datensätzen, dazu die Kette bei p = 100 mit n = 30, 60, 120). Vier Seeds je Zelle (101 bis 104), die Streuung über die Seeds steht im Raster. **Live:** eine Matrix mit allen Schätzern und der Maximum-Schwelle.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die Tabellen Zeile für Zeile in der Rundung der App). Die Streuung über die Seeds liegt beim mittleren größten |r| bei höchstens 0,005, bei den Fehlern der Schätzer bei höchstens 0,22 (Faktor, n = 30).

**Rauschmaximum bei wahrer Einheitsmatrix** (Experiment 1; Anteil der 1 000 Datensätze, in denen mindestens ein Paar über der Schwelle liegt):

| p | n | Paare | mittleres größtes \|r\| | Schwelle naiv | Datensätze mit ≥ 1 Paar | Schwelle Bonferroni | Anteil | Schwelle Maximum (95 %) | Anteil |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 30 | 45 | 0,445 | 0,361 | 90,9 % | 0,566 | 6,4 % | 0,572 | 5,2 % |
| 10 | 100 | 45 | 0,246 | 0,197 | 89,3 % | 0,321 | 4,6 % | 0,327 | 4,1 % |
| 10 | 500 | 45 | 0,111 | 0,088 | 89,4 % | 0,145 | 5,3 % | 0,141 | 6,9 % |
| 50 | 30 | 1225 | 0,597 | 0,361 | 100,0 % | 0,676 | 5,2 % | 0,687 | 3,1 % |
| 50 | 100 | 1225 | 0,340 | 0,197 | 100,0 % | 0,398 | 4,8 % | 0,398 | 4,8 % |
| 50 | 500 | 1225 | 0,156 | 0,088 | 100,0 % | 0,182 | 6,4 % | 0,180 | 7,8 % |
| 200 | 30 | 19900 | 0,683 | 0,361 | 100,0 % | 0,743 | 5,1 % | 0,737 | 6,7 % |
| 200 | 100 | 19900 | 0,406 | 0,197 | 100,0 % | 0,451 | 6,2 % | 0,448 | 6,9 % |
| 200 | 500 | 19900 | 0,186 | 0,088 | 100,0 % | 0,209 | 4,1 % | 0,209 | 4,1 % |

**Spektrum der Stichprobenmatrix bei wahrer Einheitsmatrix, p = 100** (Experiment 2; 400 Datensätze je Zeile):

| p/n | n | größter Eigenwert (Rand) | kleinster positiver (Rand) | Rang | Konditionszahl (Formel) |
|---|---|---|---|---|---|
| 0,1 | 1000 | 1,70 (1,73) | 0,482 (0,468) | 100 | 3,5 (3,7) |
| 0,5 | 200 | 2,79 (2,91) | 0,095 (0,086) | 100 | 29,6 (34,0) |
| 0,9 | 111 | 3,60 (3,80) | 0,003 (0,003) | 100 | 1219,9 (1442,0) |
| 1,5 | 67 | 4,64 (4,95) | 0,066 (0,051) | 66 | unendlich |

**Eigenwerte über dem Marchenko–Pastur-Rand, p = 50** (Experiment 3; Mittel über 400 Datensätze, rechts die wahre Zahl):

| Wahrheit | n = 30 | n = 60 | n = 120 | n = 500 | wahr (n = 30 / 60 / 120 / 500) |
|---|---|---|---|---|---|
| Einheit (alle unabhängig) | 0,06 | 0,06 | 0,07 | 0,06 | 0 / 0 / 0 / 0 |
| Kette (0,5 hoch Abstand) | 1,52 | 2,80 | 4,83 | 9,78 | 0 / 0 / 3 / 10 |
| ein Faktor (alle Paare 0,36) | 1,00 | 1,00 | 1,00 | 1,00 | 1 / 1 / 1 / 1 |
| fünf Blöcke zu zehn (innen 0,5) | 3,07 | 4,70 | 5,00 | 5,00 | 5 / 5 / 5 / 5 |

**Fehler der Schätzer, p = 50** (Experiment 4; Frobenius-Norm gegen die Wahrheit, Mittel über 200 Datensätze; Reihenfolge: Stichprobe / lineare Schrumpfung / Abschneiden / nichtlineare Schrumpfung / Orakel):

| Wahrheit | n = 30 | n = 60 | n = 120 | n = 500 |
|---|---|---|---|---|
| Einheit (alle unabhängig) | 9,21 / 0,35 / 0,26 / 1,37 / 0,00 | 6,44 / 0,09 / 0,11 / 1,39 / 0,00 | 4,54 / 0,04 / 0,08 / 0,64 / 0,00 | 2,22 / 0,01 / 0,05 / 0,28 / 0,00 |
| Kette (0,5 hoch Abstand) | 9,08 / 4,86 / 6,88 / 4,99 / 4,73 | 6,36 / 4,29 / 6,14 / 4,36 / 4,15 | 4,48 / 3,55 / 5,20 / 3,52 / 3,40 | 2,18 / 2,04 / 3,05 / 2,01 / 1,94 |
| ein Faktor (alle Paare 0,36) | 7,97 / 7,59 / 4,17 / 4,65 / 3,35 | 5,55 / 5,44 / 2,81 / 3,14 / 2,18 | 3,94 / 3,94 / 2,01 / 2,15 / 1,47 | 1,89 / 1,90 / 0,92 / 0,97 / 0,68 |
| fünf Blöcke zu zehn (innen 0,5) | 8,84 / 6,89 / 9,20 / 6,33 / 5,87 | 6,15 / 5,41 / 5,64 / 4,47 / 4,18 | 4,34 / 4,05 / 3,73 / 3,11 / 2,92 | 2,11 / 2,08 / 1,80 / 1,54 / 1,42 |

**Mehr Kennzahlen als Wochen, p = 100, Kette** (Experiment 5; je Zeile 200 Datensätze):

| n | Fehler: Stichprobe / linear / Abschneiden / nichtlinear / Orakel | Konditionszahl linear / nichtlinear |
|---|---|---|
| 30 | 18,35 / 7,47 / 11,11 / 7,65 / 7,34 | 3,1 / 4,2 |
| 60 | 12,88 / 6,88 / 10,15 / 6,97 / 6,78 | 3,6 / 4,8 |
| 120 | 9,07 / 6,06 / 8,93 / 6,09 / 5,94 | 4,9 / 9,5 |

| Frage | Befund |
|---|---|
| Wie viele Paare sind „signifikant“, wenn nichts da ist? (Experiment 1) | Bei **p = 50** liegt die Zahl der naiv „signifikanten“ Paare bei 61,2 bis 61,4 von 1 225 (5 % sind 61,25), bei **p = 200** bei 993,6 bis 994,8 von 19 900; **jeder** Datensatz hat ab p = 50 mindestens eines (100,0 %), bei p = 10 sind es 89,3 bis 90,9 %. Bonferroni und die Maximum-Schwelle halten 4,1 bis 6,4 % bzw. 3,1 bis 7,8 % der Datensätze; ihre Schwellen unterscheiden sich um höchstens 0,011. |
| Was ist an den Eigenwerten noch wahr? (Experiment 2) | **Nichts:** der größte Eigenwert der Stichprobenmatrix einer Einheitsmatrix liegt **knapp unter** dem oberen Rand (1,70 gegen 1,73; 3,60 gegen 3,80), der kleinste knapp über dem unteren; die Konditionszahl wächst von 3,5 auf **1219,9**; bei p/n = 1,5 ist der Rang **66** statt 100. |
| Zählt „Eigenwerte über dem Rand“ die echten? (Experiment 3) | **Bei Einheit, Faktor und Blöcken ja** (ab n = 120: 0,07, 1,00 und 5,00 gegen wahr 0, 1 und 5); **bei der Kette nein:** bei n = 30 und 60 stehen 1,52 und 2,80 Eigenwerte über dem Rand, obwohl keiner der wahren (größter 2,98) darüber liegt, bei n = 120 sind es 4,83 statt 3. |
| Welcher Schätzer gewinnt? (Experiment 4) | **Keiner überall:** von 16 Zellen (vier Wahrheiten · vier n) gewinnt unter den berechenbaren Schätzern die Stichprobenmatrix keine, die lineare Schrumpfung 5, das Abschneiden 5, die nichtlineare 6. Das **Orakel** ist in jeder Zelle am besten. |
| Hilft die lineare Schrumpfung immer? | **Nein:** beim Faktor kaum (7,59 gegen 7,97 bei n = 30, 1,90 gegen 1,89 bei n = 500, Gewicht 0,26 bis 0,02); bei der Kette und der Einheit viel (Gewicht 0,73 bis 0,14 bzw. 0,96 bis 1,00). |
| Schadet das Abschneiden? | **Ja, wo die Struktur klein ist:** Kette bei n = 120 und 500 (5,20 gegen 4,48 und 3,05 gegen 2,18), Blöcke bei n = 30 (9,20 gegen 8,84); beim Faktor halbiert es den Fehler (4,17 gegen 7,97 bei n = 30, 0,92 gegen 1,89 bei n = 500). |
| Wie gut ist die nichtlineare Schrumpfung? | Bei den **Blöcken in jeder Zelle vorn** (6,33 / 4,47 / 3,11 / 1,54) und bei Kette und Faktor **nahe am Orakel** (Kette n = 120: 3,52 gegen 3,40); bei der Einheit **schlechter als die lineare** (0,64 gegen 0,04 bei n = 120). |
| Was kostet die Inverse? (Experiment 4) | Bei n = 60 (p = 50) ist die Inverse der Stichprobenmatrix **relativ um 17,31 bis 19,61 daneben**, die Konditionszahl liegt bei 389 bis 3 314; nach linearer Schrumpfung sind es 0,01 bis 1,43 (Konditionszahl 1,1 bis 120,5), nach nichtlinearer 0,29 bis 0,51. |
| Was geschieht bei p > n? (Experiment 5) | Die Stichprobenmatrix hat den Rang n − 1 (29 bei n = 30, 59 bei n = 60); beide Schrumpfungen liefern eine volle Matrix (Konditionszahl 3,1 bis 9,5) und **weniger als die Hälfte des Fehlers** (7,47 und 7,65 gegen 18,35 bei n = 30). |
| Handwerte | R = [[1; 0,3], [0,3; 1]] hat die Eigenwerte 0,7 und 1,3; drei Kennzahlen mit allen Paaren 0,5 haben 2 und 0,5 (doppelt); Marchenko–Pastur bei q = 1/4: Ränder 0,25 und 2,25, Konditionszahl 9; vier Wochen mit x = (1, 2, 3, 4), y = (2, 1, 4, 3): r = 0,6; der kritische Wert von r bei n = 12 ist 0,576. |

## Ehrliche Grenzen

- **Die vier Wahrheiten sind Beispiele.** Echte Matrizen sind Mischungen; wie häufig welche Form vorkommt, zeigt die Demo nicht. Die Daten sind normalverteilt und über die Wochen unabhängig (schwere Ränder und Autokorrelation: Stücke 1 und 4).
- **Der Fehler der Matrix ist nicht der Schaden einer Anwendung.** Gemessen wird die Frobenius-Norm (und der Fehler der Inversen), nicht etwa der Verlust einer Prognose oder eines Portfolios.
- **Die Maximum-Schwelle** beruht auf Umordnung der Spalten; sie ist hier für unabhängige Kennzahlen geprüft (sie fällt mit Bonferroni zusammen). Bei echter Abhängigkeit gilt die Aussage „kein Paar hängt zusammen“ nicht mehr, die Schwelle ist dann eine Schwelle für Rauschen, nicht für die echten Paare.
- **Die Marchenko–Pastur-Aussagen** gelten für p, n → ∞ bei festem p/n; bei p = 100 liegen die Ränder bis zu 0,3 neben den Messungen (der größte Eigenwert liegt systematisch darunter).
- **Die nichtlineare Schrumpfung** ist aus der veröffentlichten Beschreibung nachgebaut (Ledoit und Wolf 2020, analytische Fassung) und gegen eine Schleifenfassung, die Einheitsmatrix und das Orakel geprüft; ein Bibliotheksorakel gibt es im Test-venv nicht. Sie braucht bei p > n − 1 mindestens 13 Wochen.
- **Das Orakel** nutzt die Wahrheit und ist kein Schätzer.
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_korrelationsmatrix.py`): Stichprobenmatrix gegen `numpy.corrcoef`, Eigenwerte gegen `scipy.linalg.eigh`, Schwellen gegen `scipy.stats.t`; **lineare Schrumpfung gegen `sklearn.covariance.ledoit_wolf`** (Gewicht und Matrix auf 10⁻¹²); **Marchenko–Pastur-Dichte durch numerische Integration** (Masse 1, Mittel 1, zweites Moment 1 + q; für q > 1 die Masse 1/q); die **nichtlineare Schrumpfung gegen eine Schleifenfassung** der Formel und gegen Eigenschaften (Einheit bleibt nahe 1, volle Matrix bei p > n); das Orakel gegen die direkte Schleife; die Maximum-Schwelle gegen die Simulation unabhängiger Matrizen; Bonferroni hält die Fehlerrate.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`), kleinste Fälle, konstante Spalten, Rang, Fehlerwege, Grenzen der Regler.
- **Eigenschaften** (`tests/test_eigenschaften_korrelationsmatrix.py`): jeder Schätzer symmetrisch, positiv semidefinit, Diagonale 1, Spur p; Invarianz gegen Verschieben und Skalieren; Rang min(p, n − 1); die Zweige (die Einheit gehört der linearen Schrumpfung und dem Abschneiden, der Faktor dem Abschneiden, die Blöcke der nichtlinearen), über viele Datensätze mit Abständen.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `kmx_algorithm`: 139 Stellen, alle gesichert; 9 davon sind als gleichwertig festgehalten (Gleichheit an Schwellen mit Gleitkommawerten, die Wahl `<=` gegen `<` bei p = n − 1, wo beide Formeln der nichtlinearen Schrumpfung übereinstimmen, und eine Grenze, die für irrationale Werte nie erreicht wird). Der erste Lauf fand 14 Überlebende und führte zu Tests für die Schleifenfassung bei p > n − 1, die Ordnungsstatistik der Maximum-Schwelle und die relative Rangtoleranz. Formen, Auswertung und Verdikte prüfen die übrigen Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `kmx_algorithm.py` | Kern: wahre Matrizen, Stichprobenmatrix, Schwellen, Spektrum, vier Schätzer und das Orakel |
| `kmx_constants.py` | Regler-Grenzen, Wahrheiten und Schätzer, Raster der Messreihen, Farben, Presets |
| `kmx_evaluation.py` | Einzelmatrix, Zellen des Rasters, Verdikte |
| `kmx_layout.py` | Fester Kopf (Stylesheet) |
| `kmx_pdf_export.py` | PDF-Export der Hauptansicht |
| `kmx_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `kmx_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/matrix_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und scikit-learn für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Gleichzeitiges Band um die ganze Matrix** (Bootstrap nach Fan, Shao und Zhou, Zeilen-Bootstrap): in der Vormessung nicht schmaler als ein Bonferroni–Fisher-Band, deshalb nicht gebaut.
- **Teilkorrelation und Korrelationsnetz** (wer hängt mit wem nach Abzug der übrigen zusammen): das nächste Stück der Reihe.
- **Faktormodelle und Graphical Lasso als Schätzer, Portfolio-Anwendungen, nicht normale Daten, QuEST:** nur als Hinweis.
- **Ursachen:** Linie *Kausale Diagramme und Adjustierung*.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (rund sechs Minuten auf zehn Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
