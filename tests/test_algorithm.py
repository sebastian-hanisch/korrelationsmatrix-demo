"""Einheiten des Kerns an Handrechnungen (Mini-Instanzen in conftest.py) und ihre Grundeigenschaften."""

import math

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import prs_scenario as S


def test_sample_correlation_by_hand(four_weeks):
    R = K.sample_corr(four_weeks)
    assert R == pytest.approx(np.array([[1.0, 0.6], [0.6, 1.0]]), abs=1e-14)
    assert K.spectrum(R) == pytest.approx([0.4, 1.6], abs=1e-14)


def test_spectrum_of_the_small_matrices_by_hand(pair_matrix, equi3):
    assert K.spectrum(pair_matrix) == pytest.approx([0.7, 1.3], abs=1e-14)
    assert K.spectrum(equi3) == pytest.approx([0.5, 0.5, 2.0], abs=1e-14)
    assert K.condition_number(pair_matrix) == pytest.approx(13 / 7) and K.condition_number(equi3) == pytest.approx(4.0)
    assert K.matrix_rank(pair_matrix) == 2


def test_marchenko_pastur_edges_and_condition_by_hand():
    assert K.mp_edges(0.25) == pytest.approx((0.25, 2.25)) and K.mp_edges(1.0) == pytest.approx((0.0, 4.0)) and K.mp_edges(4.0) == pytest.approx((1.0, 9.0))
    assert K.mp_condition(0.25) == pytest.approx(9.0) and K.mp_condition(0.0001) == pytest.approx(((1 + 0.01) / (1 - 0.01)) ** 2) and K.mp_condition(1.0) == math.inf and K.mp_condition(2.0) == math.inf


def test_marchenko_pastur_density_by_hand():
    x = np.array([0.1, 0.25, 1.0, 2.25, 3.0])
    d = K.mp_density(x, 0.25)
    assert d[0] == 0.0 and d[1] == 0.0 and d[3] == 0.0 and d[4] == 0.0                         # außerhalb und auf den Rändern
    assert d[2] == pytest.approx(math.sqrt(1.25 * 0.75) / (2 * math.pi * 0.25 * 1.0))          # √((b − x)(x − a))/(2π q x) bei x = 1


def test_r_threshold_by_hand():
    assert K.r_threshold(12) == pytest.approx(0.5760, abs=5e-5)                                 # kritischer Wert r bei 10 Freiheitsgraden, zweiseitig 5 %
    assert K.r_threshold(12, tests=10) > K.r_threshold(12) and K.r_threshold(1000) < 0.07
    with pytest.raises(ValueError):
        K.r_threshold(3)


def test_pair_counts_by_hand():
    R = np.array([[1.0, 0.7, 0.1], [0.7, 1.0, 0.4], [0.1, 0.4, 1.0]])
    c = K.pair_counts(R, 12)                                                                    # Schwelle 0,576: nur 0,7; Bonferroni (3 Tests) etwa 0,68: ebenfalls nur 0,7
    assert c == {"pairs": 3, "naiv": 1, "bonferroni": 1}
    assert K.max_abs_offdiag(R) == 0.7 and K.offdiag(R).tolist() == [0.7, 0.1, 0.4]


def test_true_matrices_by_hand():
    assert K.true_matrix("kette", 3).tolist() == [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
    assert K.true_matrix("faktor", 3).tolist() == [[1.0, 0.36, 0.36], [0.36, 1.0, 0.36], [0.36, 0.36, 1.0]]
    assert (K.true_matrix("einheit", 4) == np.eye(4)).all()
    R = K.true_matrix("bloecke", 25)
    assert R[0, 9] == 0.5 and R[0, 10] == 0.0 and R[19, 20] == 0.0 and R[20, 24] == 0.5 and np.diag(R).tolist() == [1.0] * 25           # letzter Block hat fünf Kennzahlen
    assert K.spectrum(K.true_matrix("faktor", 50))[-1] == pytest.approx(1 + 49 * 0.36) and K.spectrum(K.true_matrix("bloecke", 50))[-5:] == pytest.approx([5.5] * 5)


def test_linear_shrinkage_endpoints_and_weight_range():
    X = K.draw("kette", 40, 10, S.VRng(1))
    R, a = K.ledoit_wolf(X)
    S_ = K.sample_corr(X)
    assert 0.0 <= a <= 1.0 and R == pytest.approx((1 - a) * S_ + a * np.eye(10), abs=1e-12) and np.trace(R) == pytest.approx(10.0)


def test_clipping_by_hand(equi3):
    out = K.clip_eigenvalues(equi3, n=100)                                                      # q = 0,03: Rand 1,39: nur der Eigenwert 0,5 wird ersetzt (durch 0,5): keine Änderung
    assert out == pytest.approx(equi3, abs=1e-12)
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert K.clip_eigenvalues(R, n=2) == pytest.approx(np.eye(2))                               # alle Eigenwerte unter dem Rand (4 bei q = 1): Mittel 1


def test_clipping_keeps_the_large_eigenvalue_and_averages_the_rest():
    """R = Block [[1; 0,8], [0,8; 1]] und [[1; 0,1], [0,1; 1]]: Eigenwerte 1,8 / 0,2 / 1,1 / 0,9. Bei n = 1 000 (q = 0,004, Rand 1,13) bleibt 1,8, die übrigen werden durch ihr Mittel 0,7333 ersetzt:
    C = 1,8 uuᵀ + 0,7333 (I − uuᵀ) mit u = (1, 1, 0, 0)/√2, also C₁₁ = 1,2667, C₁₂ = 0,5333, C₃₃ = 0,7333; auf Diagonale 1 normiert: r₁₂ = 8/19."""
    R = np.array([[1.0, 0.8, 0.0, 0.0], [0.8, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]])
    out = K.clip_eigenvalues(R, 1000)
    assert out[0, 1] == pytest.approx(8 / 19, abs=1e-12) and out[2, 3] == pytest.approx(0.0, abs=1e-12) and np.diag(out).tolist() == pytest.approx([1.0] * 4)
    assert K.clip_eigenvalues(R, 16) == pytest.approx(np.eye(4), abs=1e-12)                    # n = 16: Rand 2,25, alle Eigenwerte darunter, Mittel 1


def test_oracle_by_hand(pair_matrix):
    R_hat = np.array([[1.0, 0.5], [0.5, 1.0]])                                                  # Eigenvektoren (1, ∓1)/√2
    out = K.oracle_shrinkage(R_hat, pair_matrix)
    assert out == pytest.approx(pair_matrix, abs=1e-12)                                         # dieselben Eigenvektoren: das Orakel trifft die Wahrheit


def test_errors_by_hand(pair_matrix):
    other = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert K.frobenius_error(pair_matrix, other) == pytest.approx(math.sqrt(2) * 0.3) and K.spectral_error(pair_matrix, other) == pytest.approx(0.3)
    assert K.inverse_error(other, pair_matrix) == pytest.approx(np.linalg.norm(np.eye(2) - np.linalg.inv(pair_matrix)) / np.linalg.norm(np.linalg.inv(pair_matrix)))
    assert K.inverse_error(np.ones((2, 2)), pair_matrix) == math.inf


def test_rank_and_condition_of_a_singular_matrix():
    X = S.VRng(3).normal((5, 8))
    R = K.sample_corr(X)
    assert K.matrix_rank(R) == 4 and K.condition_number(R) == math.inf                          # Rang n − 1 = 4


def test_count_above_edge_by_hand():
    R = np.diag([5.0, 1.0, 1.0, 1.0])
    assert K.count_above_edge(R, n=16) == 1                                                     # q = 0,25: Rand 2,25
    assert K.count_above_edge(np.eye(4), n=16) == 0


def test_estimate_all_returns_the_five_matrices_and_the_weight():
    X = K.draw("kette", 40, 12, S.VRng(4))
    est = K.estimate_all(X, K.true_matrix("kette", 12))
    assert set(est) == set(C.ESTIMATORS) | {"a"} and all(est[k].shape == (12, 12) for k in C.ESTIMATORS)
    assert set(K.estimate_all(X)) == {"stichprobe", "linear", "clip", "nl", "a"}


def test_draw_and_max_threshold_are_reproducible_and_depend_on_the_stream():
    a, b = K.draw("faktor", 20, 5, S.VRng(1)), K.draw("faktor", 20, 5, S.VRng(1))
    assert (a == b).all() and not (K.draw("faktor", 20, 5, S.VRng(2)) == a).all()
    t1 = K.max_threshold(a, 49, S.VRng(7))
    assert t1 == K.max_threshold(a, 49, S.VRng(7)) and 0 < t1 < 1
