"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import kmx_constants as C
import kmx_evaluation as ev
import kmx_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert preset["truth"] in C.TRUTHS and C.P_MIN <= preset["p"] <= C.P_MAX and C.N_MIN <= preset["n"] <= C.N_MAX and C.PERM_MIN <= preset["perms"] <= C.PERM_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("truth_select", "p_slider", "n_slider", "perm_slider", "seed_input")] == [s.truth, s.p, s.n, s.perms, s.seed]


def test_every_preset_differs_from_every_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_the_presets_that_name_grid_cells_use_exactly_those_cells():
    """Die Hilfetexte lesen Zellen des Rasters: p und n der Presets müssen dort vorkommen."""
    assert (C.PRESETS["Standardfall (Voreinstellung)"]["p"], C.PRESETS["Standardfall (Voreinstellung)"]["n"]) in {(p, n) for p in C.NOISE_PS for n in C.NOISE_NS}
    assert C.PRESETS["Fast so viele Kennzahlen wie Wochen"]["n"] == round(C.SPEC_P / 0.9) and C.PRESETS["Fast so viele Kennzahlen wie Wochen"]["p"] == C.SPEC_P
    assert (C.PRESETS["Mehr Kennzahlen als Wochen"]["p"], C.PRESETS["Mehr Kennzahlen als Wochen"]["n"]) == (C.WIDE_P, 60)
    for name in ("Ein gemeinsamer Faktor", "Fünf Blöcke", "Kette"):
        assert C.PRESETS[name]["p"] == C.ERR_P and C.PRESETS[name]["n"] in C.ERR_NS


@pytest.mark.parametrize("raw,expected", [("99999", C.N_MAX), ("1", C.N_MIN), ("150", 150), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["n_slider"], raw) == expected


def test_clean_value_accepts_only_known_choices_and_clamps_the_seed():
    spec = P.SETTING_SPECS["truth_select"]
    assert P.clean_value(spec, "faktor") == "faktor" and P.clean_value(spec, "kreis") is None and P.clean_value(spec, "") is None
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
