"""App: jedes Preset, der Permalink, jeder Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kmx_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["p_slider"] == preset["p"] and at.session_state["n_slider"] == preset["n"] and at.session_state["truth_select"] == preset["truth"]
    assert len(at.get("plotly_chart")) == 10                  # Hauptansicht 4, E1: 1, E2: 2, E3: 1, E4: 1, E5: 1


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["truth"] = "bloecke"
    at.query_params["n"] = "99999"
    at.query_params["p"] = "3"
    at.query_params["b"] = "abc"                # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["truth_select"] == "bloecke" and at.session_state["n_slider"] == C.N_MAX and at.session_state["p_slider"] == C.P_MIN and at.session_state["perm_slider"] == C.DEFAULT_PERM


def test_five_metrics_show_the_matrix_summary():
    at = _app()
    assert [m.label for m in at.metric] == ["Größtes |r|", "Paare naiv (von 1225)", "Paare (streng)", "Rang", "Konditionszahl"]
    assert 20 < int(at.metric[1].value) < 120 and at.metric[3].value == "29 von 50" and at.metric[4].value == "unendlich"          # p = 50, n = 30: Rang n − 1, singulär


def test_the_headline_verdict_depends_on_the_truth():
    at = _app()
    assert "Die Wahrheit ist die Einheitsmatrix" in _texts(at)
    at.sidebar.selectbox(key="truth_select").set_value("kette").run()
    assert not at.exception and "Die Wahrheit hat Struktur" in _texts(at) and "Die Wahrheit ist die Einheitsmatrix" not in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_n", 100), ("e1_n", 500), ("e1_p", 10), ("e1_p", 200), ("e3_truth", "einheit"), ("e3_truth", "faktor"), ("e4_truth", "kette"), ("e4_truth", "einheit"),
                       ("e4_truth", "bloecke"), ("e4_n", 30), ("e4_n", 500)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Bei p = 50 Kennzahlen und n = 30 Wochen liegt das größte |r| im Mittel bei", "Bei wahrer Einheitsmatrix hat die Stichprobenmatrix", "Eigenwerte über dem Rand", "Am besten ohne Kenntnis der Wahrheit",
                   "Von 16 Zellen (vier Wahrheiten, vier n) gewinnt", "Bei n = 60 (p = 50) ist die Inverse der Stichprobenmatrix", "Bei p = 100 Kennzahlen und n = 30 Wochen hat die Stichprobenmatrix höchstens den Rang 29"):
        assert needle in text, needle


def test_large_matrices_run_with_the_permutation_budget():
    at = _app()
    at.sidebar.slider(key="p_slider").set_value(C.P_MAX)
    at.sidebar.slider(key="n_slider").set_value(C.N_MAX)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.metric[3].value == "200 von 200"


def test_corner_combination_of_all_minima_and_all_maxima():
    for edge in ("min", "max"):
        at = _app()
        for key in ("p_slider", "n_slider", "perm_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.selectbox(key="truth_select").set_value("bloecke" if edge == "min" else "faktor")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
