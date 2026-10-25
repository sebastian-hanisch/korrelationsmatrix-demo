"""Das vorgerechnete Raster (data/matrix_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import math

import pytest

import kmx_constants as C
import kmx_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/matrix_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    _assert_close(probe, grid["probe"])
    assert probe["counts"] == grid["probe"]["counts"]


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["noise"], "p", "n") == sorted((p, n) for p in C.NOISE_PS for n in C.NOISE_NS)
    assert keys(grid["spec"], "q") == sorted((q,) for q in C.SPEC_QS)
    assert keys(grid["above"], "truth", "n") == sorted((t, n) for t in C.TRUTHS for n in C.ABOVE_NS)
    assert keys(grid["err"], "truth", "p", "n") == sorted([(t, C.ERR_P, n) for t in C.TRUTHS for n in C.ERR_NS] + [("kette", C.WIDE_P, n) for n in C.WIDE_NS])


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["runs"] == 4 * C.NOISE_RUNS for r in grid["noise"]) and all(r["runs"] == 4 * C.SPEC_RUNS and r["p"] == C.SPEC_P for r in grid["spec"])
    assert all(r["runs"] == 4 * C.ABOVE_RUNS for r in grid["above"]) and all(r["runs"] == 4 * C.ERR_RUNS for r in grid["err"])


def test_every_share_is_a_probability_and_a_multiple_of_the_run_count(grid):
    for r in grid["noise"]:
        for k in ("any_naive", "any_bonferroni", "any_max"):
            assert 0.0 <= r[k] <= 1.0 and r[k] * r["runs"] == pytest.approx(round(r[k] * r["runs"]), abs=1e-6)
        assert r["r_naive"] < r["r_bonferroni"] and 0 < r["mean_max"] < 1 and r["mean_max_sd"] >= 0


def test_pure_noise_always_has_a_naively_significant_pair_from_p_50_on(grid):
    for r in grid["noise"]:
        if r["p"] >= 50:
            assert r["any_naive"] == 1.0 and r["mean_max"] > r["r_naive"] + 0.05, r
        else:
            assert 0.85 < r["any_naive"] < 0.95
        assert r["naive_pairs"] == pytest.approx(0.05 * r["pairs"], rel=0.06)


def test_the_strict_thresholds_hold_the_family_wise_error_and_coincide(grid):
    for r in grid["noise"]:
        se = math.sqrt(0.05 * 0.95 / r["runs"])
        assert abs(r["any_bonferroni"] - 0.05) < 4 * se + 0.01 and abs(r["any_max"] - 0.05) < 4 * se + 0.02, r
        assert abs(r["r_bonferroni"] - r["r_max"]) < 0.03 * r["r_bonferroni"] + 0.01


def test_the_maximum_grows_with_p_and_shrinks_with_n(grid):
    cell = lambda p, n: ev.find(grid["noise"], p=p, n=n)["mean_max"]          # noqa: E731
    assert cell(10, 30) < cell(50, 30) < cell(200, 30) and cell(50, 30) > cell(50, 100) > cell(50, 500)


def test_the_spectrum_lies_near_the_marchenko_pastur_edges_and_the_condition_follows_the_formula(grid):
    for r in grid["spec"]:
        assert r["largest"] < r["edge_upper"] and r["largest"] > 0.9 * r["edge_upper"] and r["smallest"] >= r["edge_lower"] - 0.005
        if r["q"] < 1:
            assert r["rank"] == r["p"] and 0.7 * r["cond_formula"] < r["cond"] < r["cond_formula"]
        else:
            assert r["rank"] == r["n"] - 1 and r["cond"] is None


def test_the_counting_rule_hits_factor_and_blocks_but_not_the_chain(grid):
    for n in (120, 500):
        assert ev.find(grid["above"], truth="faktor", n=n)["mean_above"] == pytest.approx(1.0) and abs(ev.find(grid["above"], truth="bloecke", n=n)["mean_above"] - 5) < 0.05
        assert ev.find(grid["above"], truth="einheit", n=n)["mean_above"] < 0.15
    chain = ev.find(grid["above"], truth="kette", n=120)
    assert chain["mean_above"] > chain["true_above"] + 1 and ev.find(grid["above"], truth="kette", n=30)["mean_above"] > 1 and ev.find(grid["above"], truth="kette", n=30)["true_above"] == 0


def test_no_estimator_without_the_truth_wins_everywhere(grid):
    cells = [r for r in grid["err"] if r["p"] == C.ERR_P]
    wins = {k: sum(ev.best_computable(r) == k for r in cells) for k in ev.COMPUTABLE}
    assert len(cells) == 16 and sum(wins.values()) == 16 and sum(1 for v in wins.values() if v > 0) >= 3 and max(wins.values()) < 16 and wins["stichprobe"] == 0


def test_the_oracle_is_the_best_in_every_cell(grid):
    for r in grid["err"]:
        assert r["frob"]["orakel"] <= min(r["frob"][k] for k in ev.COMPUTABLE) + 1e-9, r


def test_linear_shrinkage_does_not_help_the_factor_and_clipping_hurts_the_chain(grid):
    for n in C.ERR_NS:
        f = ev.find(grid["err"], truth="faktor", n=n, p=50)["frob"]
        assert f["linear"] > 0.93 * f["stichprobe"] and f["clip"] < 0.6 * f["stichprobe"]
    for n in (120, 500):
        k = ev.find(grid["err"], truth="kette", n=n, p=50)["frob"]
        assert k["clip"] > k["stichprobe"]


def test_nonlinear_shrinkage_is_best_for_the_blocks_and_worse_than_linear_for_the_identity(grid):
    for n in C.ERR_NS:
        assert ev.best_computable(ev.find(grid["err"], truth="bloecke", n=n, p=50)) == "nl"
        e = ev.find(grid["err"], truth="einheit", n=n, p=50)["frob"]
        assert e["nl"] > e["linear"]


def test_the_inverse_of_the_sample_matrix_is_far_off_near_p_equal_n_and_schrumpfung_helps(grid):
    for truth in C.TRUTHS:
        r = ev.find(grid["err"], truth=truth, n=60, p=50)
        assert r["inverse"]["stichprobe"] > 15 and r["inverse"]["linear"] < 0.1 * r["inverse"]["stichprobe"] and r["inverse"]["nl"] < 0.1 * r["inverse"]["stichprobe"]
    assert ev.find(grid["err"], truth="kette", n=30, p=50)["inverse"]["stichprobe"] is None


def test_for_more_variables_than_weeks_the_shrinkages_halve_the_error(grid):
    for n in (30, 60):
        r = ev.find(grid["err"], truth="kette", n=n, p=100)
        assert r["cond"]["stichprobe"] is None and r["frob"]["linear"] < 0.6 * r["frob"]["stichprobe"] and r["frob"]["nl"] < 0.6 * r["frob"]["stichprobe"]
        assert r["cond"]["linear"] < 10 and r["cond"]["nl"] < 10


def test_verdicts_run_on_every_grid_row(grid):
    for r in grid["noise"]:
        assert ev.verdict_noise(grid["noise"], r["p"], r["n"])[0] == "info"
    assert ev.verdict_spectrum_grid(grid["spec"])[0] == "warning"
    for t in C.TRUTHS:
        assert ev.verdict_above(grid["above"], t)[0] in ("success", "warning")
        for n in C.ERR_NS:
            assert ev.verdict_errors(grid["err"], t, n)[0] in ("info", "warning")
    assert ev.verdict_winners(grid["err"])[0] == "warning" and ev.verdict_inverse(grid["err"], 60)[0] == "warning" and ev.verdict_wide(grid["err"])[0] == "warning"
