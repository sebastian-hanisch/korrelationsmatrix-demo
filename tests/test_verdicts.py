"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds. Damit prüft der Test die Grenzen (Rang < p, ≥ 120 Wochen, Gleichstand) und jede genannte Zahl direkt."""

import numpy as np

import kmx_constants as C
import kmx_evaluation as ev


def analysis(truth="einheit", p=50, n=30, counts=(1225, 61, 0, 0), rank=29, cond=float("inf"), above=0, true_above=0, errors=None):
    e = errors or {"stichprobe": 9.2, "linear": 0.35, "clip": 0.26, "nl": 1.37, "orakel": 0.0}
    eig = np.array([0.0, 0.5, 1.0, 5.0])
    return ev.Analysis(ev.Settings(truth, p, n, 199, 1), np.eye(2), np.eye(2), eig, np.array([1.0, 1.0]), 5.25, 0.5967, {"naiv": 0.361, "bonferroni": 0.676, "max": 0.687},
                       {"pairs": counts[0], "naiv": counts[1], "bonferroni": counts[2], "max": counts[3]}, 199, rank, cond, above, true_above, {}, {k: {"frob": v, "spectral": v, "cond": 1.0} for k, v in e.items()}, 0.96)


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------

def test_main_for_the_identity_with_only_the_naive_threshold_firing_is_a_success():
    kind, text = ev.verdict_main(analysis())
    assert kind == "success" and "größtes |r| = **0.60**" in text and "naiv 0.36, Bonferroni 0.68, Maximum 0.69" in text and "(von 1225): naiv **61**, Bonferroni **0**, Maximum **0**." in text
    assert "jedes naiv „signifikante“ Paar ist ein Fehlalarm" in text


def test_main_for_the_identity_warns_if_a_strict_threshold_fires_too():
    assert ev.verdict_main(analysis(counts=(1225, 61, 1, 0)))[0] == "warning"
    kind, text = ev.verdict_main(analysis(counts=(1225, 61, 0, 2)))
    assert kind == "warning" and "etwa 5 % der Datensätze" in text


def test_main_for_a_structured_truth_is_info_and_names_the_truth():
    kind, text = ev.verdict_main(analysis("kette", counts=(1225, 200, 40, 41)))
    assert kind == "info" and "Kette (0,5 hoch Abstand)" in text and "mischen echte Paare und Rauschen" in text


def test_spectrum_verdict_reports_rank_condition_and_counts():
    kind, text = ev.verdict_spectrum(analysis(above=2, true_above=1))
    assert kind == "warning" and "p/n = 1.67" in text and "größter Eigenwert **5.00**" in text and "Rand (1 + √(p/n))² = 5.25" in text and "kleinster **0.000**" in text
    assert "Rang **29** von 50" in text and "Konditionszahl **unendlich (die Matrix ist singulär)**" in text and "Über dem Rand stehen **2** Eigenwerte, die Wahrheit hat **1**." in text


def test_spectrum_verdict_is_info_for_full_rank_and_formats_large_condition_numbers():
    kind, text = ev.verdict_spectrum(analysis(p=10, n=50, rank=10, cond=1219.9))
    assert kind == "info" and "Rang **10** von 10" in text and "Konditionszahl **1 219.9**" in text


def test_estimators_verdict_names_the_best_computable_and_the_gain():
    kind, text = ev.verdict_estimators(analysis())
    assert kind == "success" and "Stichprobe **9.20**, linear **0.35**, Abschneiden **0.26**, nichtlinear **1.37**, Orakel **0.00**" in text
    assert "Am besten ohne Kenntnis der Wahrheit: **Eigenwerte abschneiden**" in text and "97 % weniger Fehler" in text and "das Orakel erreicht 0.00" in text


def test_estimators_verdict_warns_if_the_sample_matrix_wins():
    kind, text = ev.verdict_estimators(analysis(errors={"stichprobe": 1.0, "linear": 1.5, "clip": 2.0, "nl": 1.2, "orakel": 0.5}))
    assert kind == "warning" and "**Stichprobenmatrix**" in text and "Keine Schrumpfung verbessert" in text
    assert ev.verdict_estimators(analysis(errors={"stichprobe": 1.0, "linear": 1.0, "clip": 2.0, "nl": 1.2, "orakel": 0.5}))[0] == "warning"     # Gleichstand: die erste bleibt vorn


# --- E1 bis E5 --------------------------------------------------------------------------------------------------------------------------

def _noise(p=50, n=30, **kw):
    row = {"p": p, "n": n, "runs": 1000, "pairs": p * (p - 1) // 2, "mean_max": 0.5967, "r_naive": 0.361, "r_bonferroni": 0.676, "r_max": 0.6871, "any_naive": 1.0, "any_bonferroni": 0.052, "any_max": 0.031}
    row.update(kw)
    return row


def test_noise_verdict_names_the_three_shares():
    kind, text = ev.verdict_noise([_noise(), _noise(10, 30)], 50, 30)
    assert kind == "info" and "p = 50 Kennzahlen und n = 30 Wochen" in text and "**0.597**" in text and "bei 0.361" in text and "**100.0 %** der 1000 Datensätze" in text
    assert "Bonferroni-Schwelle (0.676) 5.2 %" in text and "Maximum-Schwelle (0.687) 3.1 %" in text


def _spec(q, n, cond, rank=100.0, **kw):
    row = {"q": q, "p": 100, "n": n, "largest": 1.7, "smallest": 0.48, "rank": rank, "cond": cond, "edge_upper": 1.73, "edge_lower": 0.47, "cond_formula": None if q >= 1 else 3.7}
    row.update(kw)
    return row


def test_spectrum_grid_verdict_names_the_extreme_conditions_and_the_rank():
    rows = [_spec(1.5, 67, None, 66.0), _spec(0.9, 111, 1219.9), _spec(0.1, 1000, 3.5), _spec(0.5, 200, 29.6)]
    kind, text = ev.verdict_spectrum_grid(rows)
    assert kind == "warning" and "p/n = 0.1 die Konditionszahl **3.5**" in text and "p/n = 0.9 **1 220**" in text and "bei p/n = 1.5 (67 Wochen, 100 Kennzahlen) ist der Rang **66** statt 100" in text
    kind, text = ev.verdict_spectrum_grid(rows[1:])
    assert text.endswith("**1 220**.") and "ist der Rang" not in text


def _above(truth, n, mean, true, edge=2.71):
    return {"truth": truth, "n": n, "p": 50, "edge": edge, "mean_above": mean, "true_above": true, "largest_true": 5.5, "runs": 400}


def test_above_verdict_is_a_success_if_the_rule_hits_from_120_weeks_on():
    rows = [_above("bloecke", 30, 3.07, 5, 5.25), _above("bloecke", 120, 5.0, 5), _above("bloecke", 500, 5.0, 5, 1.73)]
    kind, text = ev.verdict_above(rows, "bloecke")
    assert kind == "success" and "bei n = 30 stehen im Mittel **3.07** Eigenwerte über dem Rand (5.25), bei n = 500 **5.00**; wahr sind 5 und 5." in text and "Die Zählregel trifft ab n = 120." in text
    assert "fünf Blöcke zu zehn" in text


def test_above_verdict_threshold_is_exclusive_at_half_an_eigenvalue():
    ok = [_above("kette", 120, 3.49, 3), _above("kette", 500, 10.0, 10)]
    assert ev.verdict_above(ok, "kette")[0] == "success"
    bad = [_above("kette", 120, 3.5, 3), _above("kette", 500, 10.0, 10)]
    kind, text = ev.verdict_above(bad, "kette")
    assert kind == "warning" and "verfehlt die Wahrheit" in text


def _err(truth, n, s, lin, clip, nl, orakel=1.0, p=50, inv=None):
    return {"truth": truth, "p": p, "n": n, "runs": 200, "frob": {"stichprobe": s, "linear": lin, "clip": clip, "nl": nl, "orakel": orakel}, "frob_sd": {}, "cond": {}, "weight": 0.5,
            "inverse": inv or {"stichprobe": None, "linear": None, "nl": None}}


def test_errors_verdict_names_the_five_errors_and_the_best():
    rows = [_err("faktor", 60, 5.55, 5.44, 2.81, 3.14, 2.24)]
    kind, text = ev.verdict_errors(rows, "faktor", 60)
    assert kind == "info" and "ein Faktor (alle Paare 0,36), p = 50, n = 60" in text and "Stichprobe **5.55**, linear **5.44**, Abschneiden **2.81**, nichtlinear **3.14**, Orakel **2.24**" in text
    assert "Am besten ohne Kenntnis der Wahrheit: **Eigenwerte abschneiden**" in text
    assert ev.verdict_errors([_err("kette", 60, 1.0, 2.0, 3.0, 4.0)], "kette", 60)[0] == "warning"


def test_winners_verdict_counts_cells_and_says_that_no_estimator_wins_everywhere():
    rows = [_err("a", 30, 5, 1, 2, 3), _err("b", 30, 5, 3, 1, 2), _err("c", 30, 5, 3, 2, 1), _err("d", 30, 5, 3, 2, 1), _err("e", 30, 5, 1, 2, 3, p=100)]
    kind, text = ev.verdict_winners(rows)
    assert kind == "warning" and "Von 4 Zellen" in text and "Stichprobe 0, linear 1, Abschneiden 1, nichtlinear 2" in text and "**Kein Schätzer gewinnt überall.**" in text
    kind, text = ev.verdict_winners([_err("a", 30, 5, 1, 2, 3), _err("b", 30, 5, 1, 2, 3)])
    assert kind == "success" and "Ein Schätzer gewinnt überall." in text and "linear 2" in text


def test_inverse_verdict_picks_the_worst_truth_and_names_the_three_errors():
    rows = [_err("kette", 60, 1, 1, 1, 1, inv={"stichprobe": 17.31, "linear": 0.47, "nl": 0.51}), _err("faktor", 60, 1, 1, 1, 1, inv={"stichprobe": 19.41, "linear": 1.43, "nl": 0.32}),
            _err("einheit", 30, 1, 1, 1, 1), _err("kette", 120, 1, 1, 1, 1, inv={"stichprobe": 1.5, "linear": 0.39, "nl": 0.38})]
    kind, text = ev.verdict_inverse(rows, 60)
    assert kind == "warning" and "Bei n = 60 (p = 50)" in text and "um bis zu **1941 %** falsch (ein Faktor (alle Paare 0,36))" in text and "nach linearer Schrumpfung um 143 %, nach nichtlinearer um 32 %" in text


def test_wide_verdict_reads_the_smallest_n_of_the_wide_cells():
    rows = [_err("kette", 60, 12.88, 6.88, 10.15, 6.97, 6.78, p=100), _err("kette", 30, 18.35, 7.47, 11.11, 7.65, 7.34, p=100), _err("kette", 30, 9.0, 4.8, 6.9, 5.0, 4.7, p=50)]
    kind, text = ev.verdict_wide(rows)
    assert kind == "warning" and "p = 100 Kennzahlen und n = 30 Wochen" in text and "höchstens den Rang 29" in text and "**18.35**" in text and "**7.47**" in text and "**7.65**" in text and "(Orakel 7.34)" in text
