"""Auswertung: Einzelmatrix, Zellen, Seeds, Zugriff auf das Raster (die Verdikte prüft test_verdicts.py)."""

import math

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import kmx_evaluation as ev
import prs_scenario as S


def test_analysis_is_deterministic_and_internally_consistent():
    s = ev.Settings("kette", 30, 40, 99, 11)
    a = ev.analyse(s)
    assert a.settings == s and a.R_hat.shape == (30, 30) and a.perms_used == 99
    data_seed, perm_seed = S.substream_seeds(11, 2)
    X = K.draw("kette", 40, 30, S.VRng(data_seed))
    assert (a.R_hat == K.sample_corr(X)).all() and (a.R_true == K.true_matrix("kette", 30)).all()
    assert a.thresholds["max"] == K.max_threshold(X, 99, S.VRng(perm_seed)) and a.max_abs == K.max_abs_offdiag(a.R_hat)
    assert a.counts == {"pairs": 435, **{k: int((np.abs(K.offdiag(a.R_hat)) > a.thresholds[k]).sum()) for k in ("naiv", "bonferroni", "max")}}
    assert a.rank == K.matrix_rank(a.R_hat) == 30 and a.cond == K.condition_number(a.R_hat) and a.above == int((a.eig > a.edge).sum()) and a.edge == pytest.approx((1 + math.sqrt(30 / 40)) ** 2)
    assert a.true_above == int((a.eig_true > a.edge).sum())
    assert a.thresholds["naiv"] < a.thresholds["bonferroni"] and set(a.estimates) == set(C.ESTIMATORS) and set(a.errors) == set(C.ESTIMATORS)
    assert a.errors["stichprobe"]["frob"] == pytest.approx(K.frobenius_error(a.R_hat, a.R_true)) and a.errors["orakel"]["frob"] <= a.errors["stichprobe"]["frob"]


def test_the_settings_compare_and_hash_by_value():
    assert ev.Settings("kette", 30, 40, 99, 11) == ev.Settings("kette", 30, 40, 99, 11) and hash(ev.Settings()) == hash(ev.Settings())
    assert ev.Settings("kette", 30, 40, 99, 11) != ev.Settings("kette", 30, 40, 99, 12) and ev.Settings() != "x"


@pytest.mark.parametrize("field,value", [("truth", "faktor"), ("p", 31), ("n", 41), ("seed", 12)])
def test_every_setting_changes_the_analysis(field, value):
    base = ev.Settings("kette", 30, 40, 99, 11)
    a, b = ev.analyse(base), ev.analyse(ev.Settings(**{**base.__dict__, field: value}))
    assert a.R_hat.shape != b.R_hat.shape or not np.array_equal(a.R_hat, b.R_hat)


def test_the_number_of_permutations_changes_only_the_maximum_threshold_budget():
    a, b = ev.analyse(ev.Settings("kette", 30, 40, 99, 11)), ev.analyse(ev.Settings("kette", 30, 40, 299, 11))
    assert a.perms_used == 99 and b.perms_used == 299 and (a.R_hat == b.R_hat).all() and a.thresholds["naiv"] == b.thresholds["naiv"]
    assert a.thresholds["max"] != b.thresholds["max"]                                           # mit mehr Umordnungen ändert sich die geschätzte Schwelle


def test_cell_seeds_are_distinct_across_kinds_truths_p_n_and_seed():
    seeds = {ev.cell_seed(kind, truth, p, n, s) for kind in ("noise", "ref", "spec", "above", "err", "probe") for truth in C.TRUTHS for p in (10, 50, 100) for n in (30, 60, 500) for s in C.GRID_SEEDS}
    assert len(seeds) == 6 * 4 * 3 * 3 * 4


def test_noise_cell_structure_reproducibility_and_bounds():
    a = ev.noise_cell(8, 20, 30, seeds=(1, 2), ref=50)
    assert a == ev.noise_cell(8, 20, 30, seeds=(1, 2), ref=50) and a["runs"] == 60 and a["pairs"] == 28 and a["p"] == 8 and a["n"] == 20
    assert a["r_naive"] == K.r_threshold(20) and a["r_bonferroni"] == K.r_threshold(20, tests=28) and a["r_naive"] < a["r_max"] < 1
    assert 0 < a["mean_max"] < 1 and a["mean_max"] > a["r_naive"] and a["any_naive"] >= a["any_bonferroni"] and 0 <= a["any_max"] <= 1
    assert all(v * 60 == pytest.approx(round(v * 60)) for v in (a["any_naive"], a["any_bonferroni"], a["any_max"]))


def test_noise_cell_naive_pairs_follow_five_percent_of_the_pairs():
    a = ev.noise_cell(30, 100, 100, seeds=(3,), ref=50)
    assert a["naive_pairs"] == pytest.approx(0.05 * 435, rel=0.08) and a["any_naive"] > 0.95


def test_spectrum_cell_matches_the_formulas_and_handles_q_above_one():
    c = ev.spectrum_cell(0.5, 40, 10, seeds=(1, 2))
    assert c == ev.spectrum_cell(0.5, 40, 10, seeds=(1, 2)) and c["n"] == 80 and c["rank"] == 40 and c["cond"] is not None and c["cond_formula"] == pytest.approx(K.mp_condition(0.5))
    assert c["largest"] < c["edge_upper"] * 1.1 and c["smallest"] > 0
    w = ev.spectrum_cell(1.5, 30, 6, seeds=(1,))
    assert w["n"] == 20 and w["rank"] == 19 and w["cond"] is None and w["cond_formula"] is None


def test_above_cell_counts_and_true_counts():
    f = ev.above_cell("faktor", 120, 20, 10, seeds=(1,))
    assert f["true_above"] == 1 and f["mean_above"] == 1.0 and f["edge"] == pytest.approx((1 + math.sqrt(20 / 120)) ** 2) and f["largest_true"] == pytest.approx(1 + 19 * 0.36)
    e = ev.above_cell("einheit", 120, 20, 10, seeds=(1,))
    assert e["true_above"] == 0 and e["mean_above"] <= 0.5


def test_error_cell_structure_reproducibility_and_ordering():
    c = ev.error_cell("kette", 20, 40, 8, seeds=(1, 2))
    assert c == ev.error_cell("kette", 20, 40, 8, seeds=(1, 2)) and c["runs"] == 16 and set(c["frob"]) == set(C.ESTIMATORS) and set(c["frob_sd"]) == set(C.ESTIMATORS)
    assert c["frob"]["orakel"] <= min(c["frob"][k] for k in ev.COMPUTABLE) and c["frob"]["linear"] < c["frob"]["stichprobe"] and 0 < c["weight"] < 1
    assert set(c["cond"]) == set(ev.COMPUTABLE) and set(c["inverse"]) == {"stichprobe", "linear", "nl"} and all(v is not None for v in c["inverse"].values())      # n > p: Inverse berechenbar


def test_error_cell_has_no_inverse_and_no_condition_for_a_singular_sample_matrix():
    c = ev.error_cell("kette", 30, 20, 6, seeds=(1,))
    assert all(v is None for v in c["inverse"].values()) and c["cond"]["stichprobe"] is None and c["cond"]["linear"] is not None


def test_best_computable_names_the_smallest_error_without_the_oracle():
    assert ev.best_computable({"frob": {"stichprobe": 5.0, "linear": 3.0, "clip": 4.0, "nl": 3.5, "orakel": 1.0}}) == "linear"


def test_find_matches_exact_and_float_keys_and_returns_none_otherwise():
    rows = [{"q": 0.5, "n": 3}, {"q": 0.9, "n": 3}]
    assert ev.find(rows, q=0.9) is rows[1] and ev.find(rows, q=0.9 + 1e-12) is rows[1] and ev.find(rows, q=0.91) is None and ev.find(rows, n=3) is rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    broken = tmp_path / "kaputt.json"
    broken.write_text("{nicht json", encoding="utf-8")
    assert ev.load_grid(broken) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": [1, 2]}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": [1, 2]}
