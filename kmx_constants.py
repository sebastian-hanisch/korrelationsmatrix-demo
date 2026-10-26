"""Konstanten: Regler-Grenzen, Wahrheiten und Schätzer, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson-Kern kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): eine Matrix mit p = 200 und n = 1 000 braucht für Stichprobenmatrix, Eigenzerlegung und alle Schätzer Sekundenbruchteile; die
Maximum-Schwelle per Spaltenpermutation kostet B · n · p² (höchstens 2·10⁹, rund eine Sekunde). Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/matrix_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

P_MIN, P_MAX, DEFAULT_P = 10, 200, 50
N_MIN, N_MAX, DEFAULT_N = 20, 1000, 30
PERM_MIN, PERM_MAX, DEFAULT_PERM = 99, 499, 199
DEFAULT_SEED = 35
ALPHA = 0.05
PERM_BUDGET = 2 * 10**9           # B · n · p² höchstens

# Wahrheiten: Schlüssel -> Anzeigename
TRUTHS = {
    "einheit": "Einheit (alle unabhängig)",
    "kette": "Kette (0,5 hoch Abstand)",
    "faktor": "ein Faktor (alle Paare 0,36)",
    "bloecke": "fünf Blöcke zu zehn (innen 0,5)",
}
BLOCK_SIZE, BLOCK_RHO, FACTOR_RHO, CHAIN_RHO = 10, 0.5, 0.36, 0.5

# Schätzer: Schlüssel -> Anzeigename
ESTIMATORS = {
    "stichprobe": "Stichprobenmatrix",
    "linear": "lineare Schrumpfung",
    "clip": "Eigenwerte abschneiden",
    "nl": "nichtlineare Schrumpfung",
    "orakel": "Orakel (kennt die Wahrheit)",
}
ESTIMATOR_SHORT = {"stichprobe": "Stichprobe", "linear": "linear", "clip": "Abschneiden", "nl": "nichtlinear", "orakel": "Orakel"}

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
NOISE_PS, NOISE_NS, NOISE_RUNS, NOISE_REF = (10, 50, 200), (30, 100, 500), 250, 600     # Läufe je Seed; Referenzmatrizen der Maximum-Schwelle
SPEC_P, SPEC_QS, SPEC_RUNS = 100, (0.1, 0.5, 0.9, 1.5), 100
ABOVE_P, ABOVE_NS, ABOVE_RUNS = 50, (30, 60, 120, 500), 100
ERR_P, ERR_NS, ERR_RUNS = 50, (30, 60, 120, 500), 50
WIDE_P, WIDE_NS, WIDE_RUNS = 100, (30, 60, 120), 50                                    # p > n und p ≈ n

COLOR_SAMPLE = "#4c78a8"
COLOR_TRUE = "#222222"
COLOR_EDGE = "#d95f02"
ESTIMATOR_COLORS = {"stichprobe": "#9d9d9d", "linear": "#4c78a8", "clip": "#f58518", "nl": "#54a24b", "orakel": "#222222"}
TRUTH_COLORS = {"einheit": "#9d9d9d", "kette": "#4c78a8", "faktor": "#e45756", "bloecke": "#54a24b"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"truth": "einheit", "p": 50, "n": 30, "perms": 199, "seed": DEFAULT_SEED},
    "Fast so viele Kennzahlen wie Wochen": {"truth": "einheit", "p": 100, "n": 111, "perms": 199, "seed": DEFAULT_SEED},
    "Mehr Kennzahlen als Wochen": {"truth": "kette", "p": 100, "n": 60, "perms": 199, "seed": DEFAULT_SEED},
    "Ein gemeinsamer Faktor": {"truth": "faktor", "p": 50, "n": 60, "perms": 199, "seed": DEFAULT_SEED},
    "Fünf Blöcke": {"truth": "bloecke", "p": 50, "n": 60, "perms": 199, "seed": DEFAULT_SEED},
    "Kette": {"truth": "kette", "p": 50, "n": 120, "perms": 199, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/matrix_grid.json); tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Wahre Einheitsmatrix, 50 Kennzahlen, 30 Wochen: das größte |r| liegt in der Messreihe im Mittel bei 0,597; in 100,0 % der Datensätze ist mindestens ein Paar naiv „signifikant“, bei der Bonferroni-Schwelle in 5,2 %.",
    "Fast so viele Kennzahlen wie Wochen": "Wahre Einheitsmatrix, 100 Kennzahlen, 111 Wochen (p/n = 0,9): der kleinste Eigenwert der Stichprobenmatrix ist im Mittel 0,003 und die Konditionszahl im Mittel 1 220 bei starker Streuung (Formel für q = 0,9: 1 442), statt 1.",
    "Mehr Kennzahlen als Wochen": "Kette, 100 Kennzahlen, 60 Wochen: die Stichprobenmatrix hat höchstens den Rang 59 und den Fehler 12,88; lineare Schrumpfung 6,88, nichtlineare 6,97, Orakel 6,78.",
    "Ein gemeinsamer Faktor": "Ein Faktor, 50 Kennzahlen, 60 Wochen: Fehler Stichprobenmatrix 5,55, lineare Schrumpfung 5,44 (kaum besser), Abschneiden der Eigenwerte 2,81, nichtlineare Schrumpfung 3,14.",
    "Fünf Blöcke": "Fünf Blöcke, 50 Kennzahlen, 60 Wochen: Fehler Stichprobenmatrix 6,15, lineare Schrumpfung 5,41, Abschneiden 5,64, nichtlineare Schrumpfung 4,47, Orakel 4,18.",
    "Kette": "Kette, 50 Kennzahlen, 120 Wochen: Fehler Stichprobenmatrix 4,48, lineare Schrumpfung 3,55, Abschneiden 5,20 (schlechter als die Stichprobe), nichtlineare Schrumpfung 3,52, Orakel 3,40.",
}
