"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import kmx_algorithm as K
import kmx_constants as C
import kmx_evaluation as ev
import kmx_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings("kette", 30, 40, 99, 3))
    figs = [V.build_heatmap(a.R_hat), V.build_spectrum(a.eig, a.eig_true, 30 / 40), V.build_error_bars(a.errors), V.build_noise_lines(grid["noise"], 30), V.build_spectrum_grid(grid["spec"]),
            V.build_cond(grid["spec"]), V.build_above(grid["above"]), V.build_errors(grid["err"], "kette"), V.build_wide(grid["err"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_heatmap_shows_the_matrix_on_a_fixed_symmetric_color_range():
    R = K.true_matrix("kette", 6)
    fig = V.build_heatmap(R)
    assert np.array(fig.data[0].z).shape == (6, 6) and fig.data[0].zmin == -1 and fig.data[0].zmax == 1 and fig.layout.yaxis.autorange == "reversed"


def test_spectrum_chart_has_histogram_density_and_true_eigenvalues():
    eig = np.array([0.1, 0.5, 1.0, 1.5, 3.0])
    fig = V.build_spectrum(eig, np.ones(5), 0.5)
    assert [t.name for t in fig.data] == ["Stichprobenmatrix", "Marchenko–Pastur", "wahre Eigenwerte"] and list(fig.data[0].x) == eig.tolist() and list(fig.data[2].x) == [1.0] * 5
    a, b = K.mp_edges(0.5)
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == pytest.approx(b)
    density = np.array(fig.data[1].y)
    assert density.max() > 0 and (np.array(fig.data[1].x)[density > 0] > a).all() and (np.array(fig.data[1].x)[density > 0] < b).all()


def test_noise_chart_has_the_four_lines_for_one_n_sorted_by_p(grid):
    fig = V.build_noise_lines(grid["noise"], 100)
    assert [t.name for t in fig.data] == ["größtes |r| (Mittel)", "naive Schwelle", "Bonferroni-Schwelle", "Maximum-Schwelle (95 %)"] and list(fig.data[0].x) == list(C.NOISE_PS)
    assert list(fig.data[1].y) == pytest.approx([ev.find(grid["noise"], p=p, n=100)["r_naive"] for p in C.NOISE_PS])


def test_spectrum_grid_and_condition_charts(grid):
    fig = V.build_spectrum_grid(grid["spec"])
    assert [t.name for t in fig.data] == ["größter Eigenwert", "Rand (1 + √q)²", "kleinster positiver Eigenwert", "Rand (1 − √q)²"] and list(fig.data[0].x) == [0.1, 0.5, 0.9, 1.5]
    cond = V.build_cond(grid["spec"])
    assert list(cond.data[0].x) == ["0.1", "0.5", "0.9"] and cond.layout.yaxis.type == "log" and len(cond.data[1].y) == 3                  # q = 1,5 hat keine endliche Konditionszahl


def test_above_chart_has_a_measured_and_a_true_line_per_truth(grid):
    fig = V.build_above(grid["above"])
    assert len(fig.data) == 8 and [t.name for t in fig.data][::2] == list(C.TRUTHS.values())
    faktor = next(t for t in fig.data if t.name == C.TRUTHS["faktor"])
    assert list(faktor.x) == list(C.ABOVE_NS) and list(faktor.y) == pytest.approx([1.0] * 4)


def test_error_chart_has_the_five_estimators_and_error_bars_the_same_order(grid):
    fig = V.build_errors(grid["err"], "faktor")
    assert [t.name for t in fig.data] == list(C.ESTIMATORS.values()) and list(fig.data[0].x) == list(C.ERR_NS)
    a = ev.analyse(ev.Settings())
    assert list(V.build_error_bars(a.errors).data[0].x) == list(C.ESTIMATOR_SHORT.values())


def test_wide_chart_groups_by_n_and_uses_only_the_p_100_cells(grid):
    fig = V.build_wide(grid["err"])
    assert [t.name for t in fig.data] == list(C.ESTIMATORS.values()) and list(fig.data[0].x) == [f"n = {n}" for n in C.WIDE_NS]
    assert fig.data[0].y[0] == pytest.approx(ev.find(grid["err"], truth="kette", p=100, n=30)["frob"]["stichprobe"])
