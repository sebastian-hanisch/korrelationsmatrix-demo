"""Eigenschaften der Schätzer: Symmetrie, Positivität, Spur, Invarianz, Rang und die Zweige (jeder Schätzer gewinnt irgendwo und verliert irgendwo).

Einzel-Seed-Tests auf „Schätzer A schlägt B“ sind fragil (Lehre aus Stück 3): hier zählen Mittel über viele Datensätze mit Abständen."""

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import prs_scenario as S


def _errors(truth, n, p=30, runs=12, seed=100):
    R_true = K.true_matrix(truth, p)
    acc = {k: [] for k in C.ESTIMATORS}
    for i in range(runs):
        est = K.estimate_all(K.draw(truth, n, p, S.VRng(seed + i)), R_true)
        for k in acc:
            acc[k].append(K.frobenius_error(est[k], R_true))
    return {k: float(np.mean(v)) for k, v in acc.items()}


@pytest.mark.parametrize("truth", ["kette", "faktor"])
@pytest.mark.parametrize("n,p", [(60, 20), (20, 30)])
def test_every_estimator_is_symmetric_positive_semidefinite_with_unit_diagonal(truth, n, p):
    est = K.estimate_all(K.draw(truth, n, p, S.VRng(1)), K.true_matrix(truth, p))
    for k in C.ESTIMATORS:
        M = est[k]
        assert M == pytest.approx(M.T, abs=1e-10) and np.diag(M) == pytest.approx(np.ones(p), abs=1e-9) and K.spectrum(M)[0] > -1e-9, k


@pytest.mark.parametrize("seed", range(4))
def test_the_sample_matrix_is_invariant_to_shifting_and_scaling_the_columns(seed):
    X = K.draw("kette", 25, 8, S.VRng(seed))
    R = K.sample_corr(X)
    assert K.sample_corr(X * np.arange(1.0, 9.0) + 100.0) == pytest.approx(R, abs=1e-10)
    assert K.ledoit_wolf(X * 5.0 - 2.0)[0] == pytest.approx(K.ledoit_wolf(X)[0], abs=1e-10)


@pytest.mark.parametrize("n,p", [(10, 30), (29, 30), (30, 30), (200, 30)])
def test_the_sample_matrix_has_rank_min_p_n_minus_one(n, p):
    assert K.matrix_rank(K.sample_corr(S.VRng(2).normal((n, p)))) == min(p, n - 1)


def test_the_linear_shrinkage_makes_every_matrix_invertible_and_better_conditioned():
    R = K.sample_corr(S.VRng(3).normal((20, 40)))
    lw = K.ledoit_wolf(S.VRng(3).normal((20, 40)))[0]
    assert K.condition_number(R) == float("inf") and np.isfinite(K.condition_number(lw)) and K.condition_number(lw) < 20


def test_the_eigenvalue_sum_is_p_for_every_estimator():
    est = K.estimate_all(K.draw("bloecke", 50, 20, S.VRng(4)), K.true_matrix("bloecke", 20))
    for k in C.ESTIMATORS:
        assert K.spectrum(est[k]).sum() == pytest.approx(20.0, abs=1e-8), k


def test_the_identity_is_best_served_by_the_linear_shrinkage_or_the_clipping_not_by_the_nonlinear():
    e = _errors("einheit", 60)
    assert e["linear"] < 0.5 and e["clip"] < 0.5 and e["nl"] > 2 * e["linear"] and e["stichprobe"] > 10 * e["linear"]


def test_the_factor_is_best_served_by_clipping_and_not_by_the_linear_shrinkage():
    e = _errors("faktor", 60)
    assert e["clip"] < 0.7 * e["stichprobe"] and e["linear"] > 0.9 * e["stichprobe"] and e["orakel"] < e["clip"]


def test_the_blocks_are_best_served_by_the_nonlinear_shrinkage():
    e = _errors("bloecke", 120, p=50, runs=8)
    assert e["nl"] < min(e["stichprobe"], e["linear"], e["clip"]) and e["orakel"] < e["nl"]


def test_clipping_hurts_where_the_structure_is_small_and_shrinkage_helps_only_a_little_at_large_n():
    e = _errors("kette", 120, p=50, runs=8)
    assert e["clip"] > e["stichprobe"] and e["linear"] < e["stichprobe"] and e["nl"] < e["stichprobe"]
    big = _errors("kette", 500, p=20, runs=6)
    assert big["linear"] > 0.85 * big["stichprobe"]


def test_the_largest_sample_eigenvalue_of_noise_stays_below_the_edge_on_average():
    n, p = 200, 50
    tops = [K.spectrum(K.sample_corr(S.VRng(10 + i).normal((n, p))))[-1] for i in range(20)]
    assert np.mean(tops) < K.mp_edges(p / n)[1] and np.mean(tops) > 0.9 * K.mp_edges(p / n)[1]


def test_the_naive_threshold_fires_at_five_percent_of_the_pairs_and_the_maximum_almost_always_exceeds_it():
    n, p = 60, 40
    counts, peaks = [], []
    for i in range(60):
        r = np.abs(K.offdiag(K.sample_corr(S.VRng(200 + i).normal((n, p)))))
        counts.append((r > K.r_threshold(n)).sum())
        peaks.append(r.max())
    assert np.mean(counts) == pytest.approx(0.05 * 780, rel=0.1) and min(peaks) > K.r_threshold(n)
