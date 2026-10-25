"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Fälle, inklusive Grenzen, Rang, konstante Spalten, Fehlerwege."""

import math

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import kmx_evaluation as ev
import prs_scenario as S


def test_two_variables_is_the_smallest_matrix():
    for truth in C.TRUTHS:
        R = K.true_matrix(truth, 2)
        assert R.shape == (2, 2) and R[0, 0] == 1.0
    with pytest.raises(ValueError):
        K.true_matrix("einheit", 1)
    with pytest.raises(ValueError):
        K.true_matrix("würfel", 5)


def test_three_weeks_is_the_smallest_data_set_and_two_are_rejected():
    X = K.draw("kette", 3, 4, S.VRng(1))
    assert X.shape == (3, 4) and K.matrix_rank(K.sample_corr(X)) == 2                       # n − 1
    with pytest.raises(ValueError):
        K.draw("kette", 2, 4, S.VRng(1))


def test_a_constant_column_is_an_error_for_the_shrinkage_and_nan_for_the_correlation():
    X = S.VRng(2).normal((20, 4))
    X[:, 1] = 3.0
    assert np.isnan(K.sample_corr(X)[:, 1]).any()
    with pytest.raises(ValueError):
        K.ledoit_wolf(X)
    with pytest.raises(ValueError):
        K.standardize(X)


def test_the_threshold_needs_four_weeks_and_a_positive_family():
    assert K.r_threshold(4) == pytest.approx(0.9500, abs=1e-3)                                # 2 Freiheitsgrade: r = 0,95
    with pytest.raises(ValueError):
        K.r_threshold(3)
    with pytest.raises(ValueError):
        K.mp_edges(0.0)
    with pytest.raises(ValueError):
        K.mp_edges(-1.0)


def test_p_equals_n_gives_rank_n_minus_one_and_infinite_condition_number():
    R = K.sample_corr(S.VRng(3).normal((12, 12)))
    assert K.matrix_rank(R) == 11 and K.condition_number(R) == math.inf
    assert K.matrix_rank(K.ledoit_wolf(S.VRng(3).normal((12, 12)))[0]) == 12


def test_the_nonlinear_shrinkage_runs_at_p_equal_to_n_and_at_p_equal_to_n_minus_one():
    for n, p in ((13, 13), (14, 13), (13, 12), (30, 80)):
        R = K.sample_corr(S.VRng(4).normal((n, p)))
        out = K.nonlinear_shrinkage(R, n)
        assert out.shape == (p, p) and np.isfinite(out).all() and np.diag(out) == pytest.approx(np.ones(p))


def test_the_nonlinear_shrinkage_for_p_above_n_needs_at_least_13_weeks():
    with pytest.raises(ValueError):
        K.nonlinear_shrinkage(K.sample_corr(S.VRng(4).normal((12, 12))), 12)
    assert K.nonlinear_shrinkage(K.sample_corr(S.VRng(4).normal((12, 6))), 12).shape == (6, 6)               # p ≤ n − 1 braucht diese Grenze nicht


def test_block_matrix_with_fewer_than_ten_variables_is_one_block():
    R = K.true_matrix("bloecke", 7)
    assert R[0, 6] == 0.5 and R[3, 4] == 0.5 and np.diag(R).tolist() == [1.0] * 7


def test_the_permutation_budget_is_inclusive_and_has_a_floor():
    assert ev.affordable_perms(499, 1000, 200) == C.PERM_BUDGET // (1000 * 200**2) == 50                       # p = 200 und n = 1 000 erlauben nur 50 Permutationen
    assert ev.affordable_perms(499, 100, 50, budget=499 * 100 * 50 * 50) == 499
    assert ev.affordable_perms(499, 100, 50, budget=499 * 100 * 50 * 50 - 1) == 498
    assert ev.affordable_perms(499, 100, 50, budget=1) == 49
    assert ev.affordable_perms(99, 20, 10) == 99


def test_settings_at_all_limits_run():
    for truth, p, n in (("einheit", C.P_MIN, C.N_MIN), ("kette", C.P_MAX, C.N_MIN), ("faktor", C.P_MIN, C.N_MAX), ("bloecke", 60, 70)):
        a = ev.analyse(ev.Settings(truth, p, n, C.PERM_MIN, 0))
        assert a.R_hat.shape == (p, p) and a.rank == min(p, n - 1)
        assert set(a.estimates) == set(C.ESTIMATORS) and all(np.isfinite(e["frob"]) for e in a.errors.values())


def test_every_truth_runs_through_the_analysis():
    for truth in C.TRUTHS:
        a = ev.analyse(ev.Settings(truth, 30, 50, 99, 4))
        assert a.counts["pairs"] == 435 and a.counts["bonferroni"] <= a.counts["naiv"] <= 435 and a.counts["max"] <= a.counts["naiv"]


def test_the_inverse_error_is_infinite_for_a_singular_estimate_and_zero_for_the_truth():
    R = K.true_matrix("kette", 6)
    assert K.inverse_error(R, R) == pytest.approx(0.0, abs=1e-12)
    assert K.inverse_error(K.sample_corr(S.VRng(5).normal((4, 6))), R) == math.inf


def test_mp_density_is_zero_outside_the_support_for_every_q():
    x = np.array([0.0, 100.0, -1.0])
    for q in (0.1, 1.0, 3.0):
        assert (K.mp_density(x, q) == 0.0).all()
