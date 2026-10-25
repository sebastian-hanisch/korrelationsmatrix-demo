"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/matrix_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft, sonst gilt eine Toleranz von einem
halben Tausendstel bzw. Hundertstel."""

from pathlib import Path

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import kmx_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _c(v, d):
    return f"{v:.{d}f}".replace(".", ",")


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def test_claim_the_noise_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Rauschmaximum bei wahrer Einheitsmatrix“ (neun Zeilen)."""
    for r in grid["noise"]:
        row = (f"| {r['p']} | {r['n']} | {r['pairs']} | {_c(r['mean_max'], 3)} | {_c(r['r_naive'], 3)} | {_p(r['any_naive'])} % | {_c(r['r_bonferroni'], 3)} | {_p(r['any_bonferroni'])} % | "
               f"{_c(r['r_max'], 3)} | {_p(r['any_max'])} % |")
        assert row in README, row


def test_claim_the_spectrum_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Spektrum der Stichprobenmatrix“ (vier Zeilen)."""
    for r in grid["spec"]:
        cond = "unendlich" if r["cond"] is None else f"{_c(r['cond'], 1)} ({_c(r['cond_formula'], 1)})"
        row = (f"| {_c(r['q'], 1)} | {r['n']} | {_c(r['largest'], 2)} ({_c(r['edge_upper'], 2)}) | {_c(r['smallest'], 3)} ({_c(r['edge_lower'], 3)}) | {r['rank']:.0f} | {cond} |")
        assert row in README, row


def test_claim_the_above_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Eigenwerte über dem Marchenko–Pastur-Rand“ (vier Zeilen)."""
    for t in C.TRUTHS:
        cells = [ev.find(grid["above"], truth=t, n=n) for n in C.ABOVE_NS]
        row = f"| {C.TRUTHS[t]} | " + " | ".join(_c(x["mean_above"], 2) for x in cells) + " | " + " / ".join(str(x["true_above"]) for x in cells) + " |"
        assert row in README, row


def test_claim_the_error_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Fehler der Schätzer, p = 50“ (vier Zeilen zu je vier Zellen mit fünf Zahlen)."""
    for t in C.TRUTHS:
        cells = [ev.find(grid["err"], truth=t, p=50, n=n) for n in C.ERR_NS]
        row = f"| {C.TRUTHS[t]} | " + " | ".join(" / ".join(_c(x["frob"][k], 2) for k in C.ESTIMATORS) for x in cells) + " |"
        assert row in README, row


def test_claim_the_wide_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Mehr Kennzahlen als Wochen, p = 100“ (drei Zeilen)."""
    for n in C.WIDE_NS:
        x = ev.find(grid["err"], truth="kette", p=100, n=n)
        row = f"| {n} | " + " / ".join(_c(x["frob"][k], 2) for k in C.ESTIMATORS) + f" | {_c(x['cond']['linear'], 1)} / {_c(x['cond']['nl'], 1)} |"
        assert row in README, row
    assert [K.matrix_rank(K.sample_corr(K.draw("kette", n, 100, __import__("prs_scenario").VRng(1)))) for n in (30, 60)] == [29, 59]


def test_claim_pure_noise_statements(grid):
    """README: naive Paare bei p = 50: 61,2 bis 61,4 (von 1 225), bei p = 200: 993,6 bis 994,8; ≥ 1 Paar bei p = 10: 89,3 bis 90,9 %; Bonferroni 4,1 bis 6,4 %, Maximum 3,1 bis 7,8 %; Schwellen höchstens 0,011 auseinander."""
    f = lambda p: [r["naive_pairs"] for r in grid["noise"] if r["p"] == p]          # noqa: E731
    assert (min(f(50)), max(f(50))) == pytest.approx((61.2, 61.4), abs=0.05) and (min(f(200)), max(f(200))) == pytest.approx((993.6, 994.8), abs=0.05)
    ten = [r["any_naive"] for r in grid["noise"] if r["p"] == 10]
    assert (min(ten), max(ten)) == pytest.approx((0.893, 0.909), abs=0.0006) and all(r["any_naive"] == 1.0 for r in grid["noise"] if r["p"] >= 50)
    bon, mx = [r["any_bonferroni"] for r in grid["noise"]], [r["any_max"] for r in grid["noise"]]
    assert (min(bon), max(bon)) == pytest.approx((0.041, 0.064), abs=0.0006) and (min(mx), max(mx)) == pytest.approx((0.031, 0.078), abs=0.0006)
    assert max(abs(r["r_bonferroni"] - r["r_max"]) for r in grid["noise"]) == pytest.approx(0.011, abs=0.0006)
    assert 0.05 * 1225 == pytest.approx(61.25)


def test_claim_seed_spread(grid):
    """README: Streuung über die Seeds beim mittleren größten |r| höchstens 0,005, bei den Fehlern der Schätzer höchstens 0,22."""
    assert max(r["mean_max_sd"] for r in grid["noise"]) == pytest.approx(0.005, abs=0.0006)
    assert max(max(r["frob_sd"].values()) for r in grid["err"]) == pytest.approx(0.22, abs=0.006)


def test_claim_the_spectrum_statements(grid):
    """README: größter Eigenwert 1,70 gegen 1,73 und 3,60 gegen 3,80, Konditionszahl 3,5 → 1219,9 (Formel 1442,0), Rang 66 bei p/n = 1,5."""
    q = {r["q"]: r for r in grid["spec"]}
    assert (q[0.1]["largest"], q[0.1]["edge_upper"], q[0.9]["largest"], q[0.9]["edge_upper"]) == pytest.approx((1.70, 1.73, 3.60, 3.80), abs=0.006)
    assert (q[0.1]["cond"], q[0.9]["cond"], q[0.9]["cond_formula"]) == pytest.approx((3.5, 1219.9, 1442.0), abs=0.06) and q[1.5]["rank"] == 66


def test_claim_the_counting_rule(grid):
    """README: Kette 1,52 / 2,80 / 4,83 gegen wahr 0 / 0 / 3, größter wahrer Eigenwert 2,98; ab n = 120: Einheit 0,07, Faktor 1,00, Blöcke 5,00."""
    kette = [ev.find(grid["above"], truth="kette", n=n) for n in C.ABOVE_NS]
    assert [c["mean_above"] for c in kette[:3]] == pytest.approx([1.52, 2.80, 4.83], abs=0.006) and [c["true_above"] for c in kette] == [0, 0, 3, 10] and kette[0]["largest_true"] == pytest.approx(2.98, abs=0.006)
    at120 = [ev.find(grid["above"], truth=t, n=120)["mean_above"] for t in ("einheit", "faktor", "bloecke")]
    assert at120 == pytest.approx([0.07, 1.00, 5.00], abs=0.006)


def test_claim_the_estimator_wins_and_the_oracle(grid):
    """README: von 16 Zellen gewinnt die Stichprobenmatrix keine, die lineare Schrumpfung 5, das Abschneiden 5, die nichtlineare 6; das Orakel ist in jeder Zelle am besten."""
    cells = [r for r in grid["err"] if r["p"] == 50]
    wins = {k: sum(ev.best_computable(r) == k for r in cells) for k in ev.COMPUTABLE}
    assert len(cells) == 16 and wins == {"stichprobe": 0, "linear": 5, "clip": 5, "nl": 6}
    assert all(r["frob"]["orakel"] <= min(r["frob"][k] for k in ev.COMPUTABLE) for r in grid["err"])


def test_claim_linear_shrinkage_and_clipping(grid):
    """README: Faktor: lineare Schrumpfung 7,59 gegen 7,97 (n = 30), 1,90 gegen 1,89 (n = 500), Gewicht 0,26 bis 0,02; Kette 0,73 bis 0,14, Einheit 0,96 bis 1,00; Abschneiden: Kette 5,20 gegen 4,48 (n = 120),
    3,05 gegen 2,18 (n = 500), Blöcke 9,20 gegen 8,84 (n = 30), Faktor 4,17 gegen 7,97 und 0,92 gegen 1,89."""
    e = lambda t, n: ev.find(grid["err"], truth=t, p=50, n=n)          # noqa: E731
    assert (e("faktor", 30)["frob"]["linear"], e("faktor", 30)["frob"]["stichprobe"], e("faktor", 500)["frob"]["linear"], e("faktor", 500)["frob"]["stichprobe"]) == pytest.approx((7.59, 7.97, 1.90, 1.89), abs=0.006)
    assert (e("faktor", 30)["weight"], e("faktor", 500)["weight"], e("kette", 30)["weight"], e("kette", 500)["weight"], e("einheit", 30)["weight"], e("einheit", 500)["weight"]) == pytest.approx((0.26, 0.02, 0.73, 0.14, 0.96, 1.00), abs=0.006)
    assert (e("kette", 120)["frob"]["clip"], e("kette", 120)["frob"]["stichprobe"], e("kette", 500)["frob"]["clip"], e("kette", 500)["frob"]["stichprobe"]) == pytest.approx((5.20, 4.48, 3.05, 2.18), abs=0.006)
    assert (e("bloecke", 30)["frob"]["clip"], e("bloecke", 30)["frob"]["stichprobe"], e("faktor", 30)["frob"]["clip"], e("faktor", 500)["frob"]["clip"], e("faktor", 500)["frob"]["stichprobe"]) == pytest.approx((9.20, 8.84, 4.17, 0.92, 1.89), abs=0.006)


def test_claim_nonlinear_shrinkage(grid):
    """README: Blöcke 6,33 / 4,47 / 3,11 / 1,54 (in jeder Zelle vorn); Kette n = 120: 3,52 gegen Orakel 3,40; Einheit n = 120: 0,64 gegen lineare 0,04; Kette bei n = 30: 9,08 → 4,86."""
    blocks = [ev.find(grid["err"], truth="bloecke", p=50, n=n) for n in C.ERR_NS]
    assert [b["frob"]["nl"] for b in blocks] == pytest.approx([6.33, 4.47, 3.11, 1.54], abs=0.006) and all(ev.best_computable(b) == "nl" for b in blocks)
    k = ev.find(grid["err"], truth="kette", p=50, n=120)["frob"]
    assert (k["nl"], k["orakel"]) == pytest.approx((3.52, 3.40), abs=0.006)
    e = ev.find(grid["err"], truth="einheit", p=50, n=120)["frob"]
    assert (e["nl"], e["linear"]) == pytest.approx((0.64, 0.04), abs=0.006)
    k30 = ev.find(grid["err"], truth="kette", p=50, n=30)["frob"]
    assert (k30["stichprobe"], k30["linear"]) == pytest.approx((9.08, 4.86), abs=0.006)


def test_claim_the_inverse_at_n_60(grid):
    """README: relativer Fehler der Inversen bei n = 60: Stichprobenmatrix 17,31 bis 19,61, linear 0,01 bis 1,43, nichtlinear 0,29 bis 0,51; Konditionszahl 389 bis 3 314, nach linearer Schrumpfung 1,1 bis 120,5."""
    cells = [ev.find(grid["err"], truth=t, p=50, n=60) for t in C.TRUTHS]
    inv = lambda k: [c["inverse"][k] for c in cells]          # noqa: E731
    assert (min(inv("stichprobe")), max(inv("stichprobe"))) == pytest.approx((17.31, 19.61), abs=0.006)
    assert (min(inv("linear")), max(inv("linear"))) == pytest.approx((0.01, 1.43), abs=0.006) and (min(inv("nl")), max(inv("nl"))) == pytest.approx((0.29, 0.51), abs=0.006)
    cs = [c["cond"]["stichprobe"] for c in cells]
    assert (min(cs), max(cs)) == pytest.approx((389, 3314), abs=0.6)
    cl = [c["cond"]["linear"] for c in cells]
    assert (min(cl), max(cl)) == pytest.approx((1.1, 120.5), abs=0.06)


def test_claim_more_variables_than_weeks(grid):
    """README: Konditionszahl der Schrumpfungen 3,1 bis 9,5; bei n = 30: Stichprobe 18,35, linear 7,47, nichtlinear 7,65."""
    cells = [ev.find(grid["err"], truth="kette", p=100, n=n) for n in C.WIDE_NS]
    conds = [c["cond"][k] for c in cells for k in ("linear", "nl")]
    assert (min(conds), max(conds)) == pytest.approx((3.1, 9.5), abs=0.06)
    assert (cells[0]["frob"]["stichprobe"], cells[0]["frob"]["linear"], cells[0]["frob"]["nl"]) == pytest.approx((18.35, 7.47, 7.65), abs=0.006)


def test_claim_the_intro_numbers(grid):
    """README, Einleitung: p = 50, n = 30: größtes |r| 0,597 gegen die naive Schwelle 0,361, jeder Datensatz ≥ 1 naives Paar; Konditionszahl bei p/n = 0,9 ist 1219,9."""
    c = ev.find(grid["noise"], p=50, n=30)
    assert (c["mean_max"], c["r_naive"], c["any_naive"]) == pytest.approx((0.597, 0.361, 1.0), abs=0.0006)


def test_claim_hand_values(pair_matrix, equi3, four_weeks):
    """README: Eigenwerte 0,7 und 1,3; 2 und 0,5 (doppelt); Marchenko–Pastur bei q = 1/4: Ränder 0,25 und 2,25, Konditionszahl 9; r = 0,6; kritischer Wert bei n = 12: 0,576."""
    assert K.spectrum(pair_matrix) == pytest.approx([0.7, 1.3]) and K.spectrum(equi3) == pytest.approx([0.5, 0.5, 2.0])
    assert K.mp_edges(0.25) == pytest.approx((0.25, 2.25)) and K.mp_condition(0.25) == pytest.approx(9.0)
    assert K.sample_corr(four_weeks)[0, 1] == pytest.approx(0.6) and K.r_threshold(12) == pytest.approx(0.576, abs=0.0006)


def test_claim_preset_help_numbers(grid):
    """Hilfetexte der Presets: jede Zahl steht im Raster (Zelle und Größe des Presets) in der Rundung der App."""
    noise = ev.find(grid["noise"], p=50, n=30)
    spec = ev.find(grid["spec"], q=0.9)
    wide = ev.find(grid["err"], truth="kette", p=100, n=60)
    cells = {name: ev.find(grid["err"], truth=C.PRESETS[name]["truth"], p=50, n=C.PRESETS[name]["n"]) for name in ("Ein gemeinsamer Faktor", "Fünf Blöcke", "Kette")}
    expect = {
        "Standardfall (Voreinstellung)": [_c(noise["mean_max"], 3), _p(noise["any_naive"]), _p(noise["any_bonferroni"])],
        "Fast so viele Kennzahlen wie Wochen": [_c(spec["smallest"], 3), f"{round(spec['cond']):,}".replace(",", " "), f"{round(spec['cond_formula']):,}".replace(",", " ")],
        "Mehr Kennzahlen als Wochen": [_c(wide["frob"][k], 2) for k in ("stichprobe", "linear", "nl", "orakel")],
        "Ein gemeinsamer Faktor": [_c(cells["Ein gemeinsamer Faktor"]["frob"][k], 2) for k in ("stichprobe", "linear", "clip", "nl")],
        "Fünf Blöcke": [_c(cells["Fünf Blöcke"]["frob"][k], 2) for k in ("stichprobe", "linear", "clip", "nl", "orakel")],
        "Kette": [_c(cells["Kette"]["frob"][k], 2) for k in ("stichprobe", "linear", "clip", "nl", "orakel")],
    }
    assert set(expect) == set(C.PRESETS)
    for name, values in expect.items():
        for v in values:
            assert v in C.PRESET_HELP[name], (name, v)
    assert "Rang 59" in C.PRESET_HELP["Mehr Kennzahlen als Wochen"] and "(Formel: 1 442)" in C.PRESET_HELP["Fast so viele Kennzahlen wie Wochen"]


def test_claim_readme_mentions_its_key_numbers():
    """Verbindet die README mit den Tests: die Zahlen, die dort als Kernbefunde stehen, kommen im Text vor."""
    for needle in ("0,597", "0,361", "1219,9", "9,08 → 4,86", "7,97 → 7,59", "7,97 → 4,17", "6,33 gegen 8,84", "0,64 gegen 0,04", "388 Sekunden", "17,31 bis 19,61", "gewinnt unter den berechenbaren Schätzern die Stichprobenmatrix keine, die lineare Schrumpfung 5, das Abschneiden 5, die nichtlineare 6"):
        assert needle in README, needle


def test_claim_grid_build_time(grid):
    """README: 388 Sekunden auf 10 Prozessen."""
    assert grid["meta"]["seconds"] == 388
    assert np.isfinite(grid["meta"]["seconds"])
