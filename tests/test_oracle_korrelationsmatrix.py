"""Orakel (ORACLE-PLAYBOOK): jedes Verfahren gegen einen anderen Rechenweg.

* Stichprobenmatrix gegen `numpy.corrcoef`, Eigenwerte gegen `scipy.linalg.eigh` und die Spurbeziehung, Schwellen gegen `scipy.stats.t`.
* Lineare Schrumpfung gegen `sklearn.covariance.ledoit_wolf` (Gewicht und Matrix).
* Marchenko–Pastur: Dichte durch numerische Integration (Masse, Mittel, zweites Moment), Ränder und Konditionszahl gegen die Simulation.
* Nichtlineare Schrumpfung gegen eine Schleifenfassung der Formel und gegen Eigenschaften (Einheitsmatrix bleibt nahe 1, Rang bei p > n); Orakel gegen die direkte Schleife uᵀRu.
* Maximum-Schwelle gegen die Simulation unabhängiger Referenzmatrizen.
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit."""

import math

import numpy as np
import pytest

import kmx_algorithm as K
import prs_scenario as S

scipy_linalg = pytest.importorskip("scipy.linalg")
scipy_stats = pytest.importorskip("scipy.stats")
sk_cov = pytest.importorskip("sklearn.covariance")


@pytest.mark.parametrize("truth", ["einheit", "kette", "faktor", "bloecke"])
def test_sample_correlation_and_spectrum_match_numpy_and_scipy(truth):
    X = K.draw(truth, 40, 12, S.VRng(3))
    R = K.sample_corr(X)
    assert R == pytest.approx(np.corrcoef(X, rowvar=False), abs=1e-12)
    w = K.spectrum(R)
    assert w == pytest.approx(scipy_linalg.eigh(R, eigvals_only=True), abs=1e-10) and w.sum() == pytest.approx(12.0, abs=1e-10)


@pytest.mark.parametrize("n", [10, 30, 100, 500])
@pytest.mark.parametrize("tests", [1, 10, 1225])
def test_thresholds_match_scipy(n, tests):
    t = scipy_stats.t.ppf(1 - 0.025 / tests, n - 2)
    assert K.r_threshold(n, tests=tests) == pytest.approx(t / math.sqrt(t * t + n - 2), abs=1e-9)


@pytest.mark.parametrize("truth", ["kette", "faktor", "einheit"])
@pytest.mark.parametrize("n,p", [(60, 20), (30, 20), (20, 30)])
def test_linear_shrinkage_matches_sklearn(truth, n, p):
    X = K.draw(truth, n, p, S.VRng(5))
    R, a = K.ledoit_wolf(X)
    Z = (X - X.mean(0)) / X.std(0)
    cov, shrink = sk_cov.ledoit_wolf(Z, assume_centered=True)
    assert a == pytest.approx(shrink, abs=1e-12) and R == pytest.approx(cov, abs=1e-12)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.9])
def test_marchenko_pastur_density_integrates_to_one_with_mean_one_and_second_moment_one_plus_q(q):
    a, b = K.mp_edges(q)
    theta = np.linspace(0.0, math.pi, 400001)
    x = a + (b - a) * np.sin(theta / 2) ** 2                                                 # x = a + (b − a) sin²(θ/2): glättet die Wurzelsingularität an den Rändern
    dx = (b - a) * np.sin(theta / 2) * np.cos(theta / 2)
    f = K.mp_density(x, q) * dx
    assert np.trapezoid(f, theta) == pytest.approx(1.0, abs=1e-5)
    assert np.trapezoid(f * x, theta) == pytest.approx(1.0, abs=1e-5) and np.trapezoid(f * x * x, theta) == pytest.approx(1.0 + q, abs=1e-5)


def test_marchenko_pastur_density_for_q_above_one_has_mass_one_over_q():
    q = 2.0
    a, b = K.mp_edges(q)
    theta = np.linspace(0.0, math.pi, 400001)
    x = a + (b - a) * np.sin(theta / 2) ** 2
    f = K.mp_density(x, q) * (b - a) * np.sin(theta / 2) * np.cos(theta / 2)
    assert np.trapezoid(f, theta) == pytest.approx(1 / q, abs=1e-5)


@pytest.mark.parametrize("q,n", [(0.1, 1000), (0.5, 200)])
def test_simulated_extreme_eigenvalues_lie_near_the_marchenko_pastur_edges(q, n):
    p = int(q * n)
    w = np.array([K.spectrum(K.sample_corr(S.VRng(i).normal((n, p)))) for i in range(30)])
    a, b = K.mp_edges(q)
    assert w[:, -1].mean() == pytest.approx(b, rel=0.06) and w[:, 0].mean() == pytest.approx(a, rel=0.2) and w[:, -1].mean() < b            # der größte liegt knapp darunter
    cond = (w[:, -1] / w[:, 0]).mean()
    assert cond == pytest.approx(K.mp_condition(q), rel=0.25)


def _nl_loop(R, n):
    """Schleifenfassung der analytischen nichtlinearen Schrumpfung (p ≤ n − 1), Eintrag für Eintrag nach der Formel."""
    p = len(R)
    n_eff = n - 1
    lam, U = np.linalg.eigh(R)
    h = n_eff ** (-1 / 3)
    s5 = math.sqrt(5)
    d = np.empty(p)
    for i in range(p):
        f = 0.0
        hil = 0.0
        for j in range(p):
            hj = h * lam[j]
            x = (lam[i] - lam[j]) / hj
            f += max(1 - x * x / 5, 0.0) / hj
            if abs(x) == s5:
                term = (-3 / 10 / math.pi) * x
            else:
                term = (-3 / 10 / math.pi) * x + (3 / 4 / s5 / math.pi) * (1 - x * x / 5) * math.log(abs((s5 - x) / (s5 + x)))
            hil += term / hj
        f *= 3 / 4 / s5 / p
        hil /= p
        q = p / n_eff
        d[i] = lam[i] / ((math.pi * q * lam[i] * f) ** 2 + (1 - q - math.pi * q * lam[i] * hil) ** 2)
    C = (U * d) @ U.T
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


@pytest.mark.parametrize("truth", ["kette", "faktor", "bloecke"])
def test_nonlinear_shrinkage_matches_the_loop_version(truth):
    n, p = 80, 20
    R = K.sample_corr(K.draw(truth, n, p, S.VRng(6)))
    assert K.nonlinear_shrinkage(R, n) == pytest.approx(_nl_loop(R, n), abs=1e-10)


def test_nonlinear_shrinkage_pulls_the_spectrum_of_a_noise_matrix_towards_one():
    n, p = 200, 50
    R = K.sample_corr(S.VRng(7).normal((n, p)))
    w_before, w_after = K.spectrum(R), K.spectrum(K.nonlinear_shrinkage(R, n))
    assert w_after.min() > w_before.min() + 0.2 and w_after.max() < w_before.max() - 0.2 and abs(w_after.mean() - 1.0) < 1e-9


def test_nonlinear_shrinkage_for_more_variables_than_weeks_gives_a_full_rank_matrix():
    n, p = 30, 80
    R = K.sample_corr(S.VRng(8).normal((n, p)))
    assert K.matrix_rank(R) == n - 1
    out = K.nonlinear_shrinkage(R, n)
    assert K.matrix_rank(out) == p and np.linalg.norm(out - np.eye(p)) < 0.3 * np.linalg.norm(R - np.eye(p))


@pytest.mark.parametrize("truth", ["kette", "faktor", "bloecke"])
def test_oracle_matches_the_direct_loop(truth):
    R_true = K.true_matrix(truth, 8)
    R_hat = K.sample_corr(K.draw(truth, 30, 8, S.VRng(9)))
    w, U = np.linalg.eigh(R_hat)
    d = np.array([sum(U[j, i] * R_true[j, k] * U[k, i] for j in range(8) for k in range(8)) for i in range(8)])
    C = (U * d) @ U.T
    s = np.sqrt(np.diag(C))
    assert K.oracle_shrinkage(R_hat, R_true) == pytest.approx(C / np.outer(s, s), abs=1e-12)


def test_the_oracle_is_never_worse_than_the_computable_estimators_on_average():
    for truth in ("kette", "faktor", "bloecke"):
        R_true = K.true_matrix(truth, 30)
        errs = {k: [] for k in ("stichprobe", "linear", "clip", "nl", "orakel")}
        for seed in range(12):
            est = K.estimate_all(K.draw(truth, 60, 30, S.VRng(100 + seed)), R_true)
            for k in errs:
                errs[k].append(K.frobenius_error(est[k], R_true))
        assert np.mean(errs["orakel"]) <= min(np.mean(errs[k]) for k in ("stichprobe", "linear", "clip", "nl")) + 1e-9, truth


def test_the_max_threshold_matches_a_simulation_of_independent_matrices():
    n, p = 40, 15
    ref = np.quantile([K.max_abs_offdiag(K.sample_corr(S.VRng(1000 + i).normal((n, p)))) for i in range(1500)], 0.95)
    X = S.VRng(5).normal((n, p))
    assert K.max_threshold(X, 499, S.VRng(6)) == pytest.approx(ref, abs=0.025)


def test_the_bonferroni_threshold_keeps_the_family_wise_error_near_five_percent():
    n, p, hits = 40, 15, 0
    thr = K.r_threshold(n, tests=p * (p - 1) // 2)
    for i in range(400):
        hits += K.max_abs_offdiag(K.sample_corr(S.VRng(3000 + i).normal((n, p)))) > thr
    assert 0.02 <= hits / 400 <= 0.09


def _nl_loop_wide(R, n):
    """Schleifenfassung der analytischen nichtlinearen Schrumpfung für p > n − 1: die n − 1 positiven Eigenwerte, die übrigen p − (n − 1) bekommen den gemeinsamen Wert d₀."""
    p = len(R)
    n_eff = n - 1
    lam, U = np.linalg.eigh(R)
    k = n_eff
    top = lam[p - k:]
    h = n_eff ** (-1 / 3)
    s5 = math.sqrt(5)
    d_top = np.empty(k)
    for i in range(k):
        f = 0.0
        hil = 0.0
        for j in range(k):
            hj = h * top[j]
            x = (top[i] - top[j]) / hj
            f += max(1 - x * x / 5, 0.0) / hj
            term = (-3 / 10 / math.pi) * x if abs(x) == s5 else (-3 / 10 / math.pi) * x + (3 / 4 / s5 / math.pi) * (1 - x * x / 5) * math.log(abs((s5 - x) / (s5 + x)))
            hil += term / hj
        f *= 3 / 4 / s5 / k
        hil /= k
        d_top[i] = top[i] / (math.pi**2 * top[i] ** 2 * (f**2 + hil**2))
    hf0 = (1 / math.pi) * (3 / 10 / h**2 + 3 / 4 / s5 / h * (1 - 1 / 5 / h**2) * math.log((1 + s5 * h) / (1 - s5 * h))) * sum(1 / t for t in top) / k
    d0 = 1 / (math.pi * (p - n_eff) / n_eff * hf0)
    d = np.concatenate([np.full(p - k, d0), d_top])
    C = (U * d) @ U.T
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


@pytest.mark.parametrize("n,p", [(30, 40), (25, 60)])
def test_nonlinear_shrinkage_for_more_variables_than_weeks_matches_the_loop_version(n, p):
    R = K.sample_corr(K.draw("kette", n, p, S.VRng(12)))
    assert K.nonlinear_shrinkage(R, n) == pytest.approx(_nl_loop_wide(R, n), abs=1e-9)


@pytest.mark.parametrize("B", [49, 99, 100])
def test_the_max_threshold_is_the_order_statistic_of_the_permuted_maxima(B):
    X = K.draw("kette", 25, 8, S.VRng(3))
    rng = S.VRng(4)
    peaks = []
    for _ in range(B):
        order = np.argsort(rng.uniform((25, 8)), axis=0)
        peaks.append(K.max_abs_offdiag(K.sample_corr(np.take_along_axis(X, order, axis=0))))
    k = min(B, math.ceil(0.95 * (B + 1)))
    assert K.max_threshold(X, B, S.VRng(4)) == pytest.approx(sorted(peaks)[k - 1], abs=1e-14)
    assert K.max_threshold(X, B, S.VRng(4), alpha=0.5) == pytest.approx(sorted(peaks)[min(B, math.ceil(0.5 * (B + 1))) - 1], abs=1e-14)


def test_the_rank_uses_a_tolerance_relative_to_the_largest_eigenvalue():
    assert K.matrix_rank(np.diag([1e6, 1e-5])) == 1 and K.matrix_rank(np.diag([1e6, 1e-3])) == 2 and K.matrix_rank(np.diag([0.5, 1e-11])) == 1
