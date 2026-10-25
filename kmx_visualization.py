"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kmx_algorithm as K
import kmx_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_heatmap(R, height: int = 340):
    """Korrelationsmatrix als Heatmap (−1 blau, 0 weiß, +1 rot), Zeile 1 oben."""
    R = np.asarray(R)
    fig = go.Figure(go.Heatmap(z=R, zmin=-1, zmax=1, colorscale="RdBu_r", reversescale=False, colorbar=dict(thickness=10, title="r"), hovertemplate="Kennzahl %{y} und %{x}: r = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", title_text="Kennzahl")
    fig.update_xaxes(title_text="Kennzahl", constrain="domain")
    return _base(fig, height)


def build_spectrum(eig, eig_true, q: float, height: int = 340):
    """Eigenwerte der Stichprobenmatrix (Histogramm als Dichte) mit der Marchenko–Pastur-Dichte, den Rändern und den wahren Eigenwerten als Striche."""
    a, b = K.mp_edges(q)
    top = max(float(np.max(eig)), b, float(np.max(eig_true)))
    xs = np.linspace(0.0, top * 1.05, 600)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=list(eig), histnorm="probability density", xbins=dict(start=0.0, end=top * 1.05, size=top * 1.05 / 40), name="Stichprobenmatrix", marker_color=C.COLOR_SAMPLE, opacity=0.75,
                               hovertemplate="Eigenwert um %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=K.mp_density(xs, q), mode="lines", name="Marchenko–Pastur", line=dict(color=C.COLOR_TRUE, width=2.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(eig_true), y=[0] * len(eig_true), mode="markers", name="wahre Eigenwerte", marker=dict(symbol="line-ns-open", size=14, color="#e45756", line=dict(width=2)),
                             hovertemplate="wahrer Eigenwert %{x:.2f}<extra></extra>"))
    fig.add_vline(x=b, line=dict(color=C.COLOR_EDGE, dash="dash", width=1.5), annotation_text="Rand", annotation_position="top")
    fig.update_xaxes(title_text="Eigenwert")
    fig.update_yaxes(title_text="Dichte")
    fig.update_layout(bargap=0.05)
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_noise_lines(rows: list[dict], n: int, height: int = 340):
    """Größtes |r| bei reinem Rauschen gegen p (für ein n) mit den drei Schwellen."""
    sub = sorted((r for r in rows if r["n"] == n), key=lambda r: r["p"])
    ps = [r["p"] for r in sub]
    fig = go.Figure()
    for key, name, color, dash in (("mean_max", "größtes |r| (Mittel)", C.COLOR_TRUE, "solid"), ("r_naive", "naive Schwelle", "#e45756", "dash"), ("r_bonferroni", "Bonferroni-Schwelle", "#54a24b", "dash"),
                                   ("r_max", "Maximum-Schwelle (95 %)", "#4c78a8", "dot")):
        fig.add_trace(go.Scatter(x=ps, y=[r[key] for r in sub], mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash), hovertemplate="p = %{x}: %{y:.3f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Zahl der Kennzahlen p", type="log", tickvals=ps, ticktext=[str(p) for p in ps])
    fig.update_yaxes(title_text="|r|", range=[0, 1])
    return _base(fig, height)


def build_spectrum_grid(rows: list[dict], height: int = 340):
    """Größter und kleinster Eigenwert gegen p/n mit den Marchenko–Pastur-Rändern (Linien)."""
    rows = sorted(rows, key=lambda r: r["q"])
    qs = [r["q"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=[r["largest"] for r in rows], mode="lines+markers", name="größter Eigenwert", line=dict(color=C.COLOR_SAMPLE, width=3), hovertemplate="p/n = %{x}: %{y:.2f}<extra>größter</extra>"))
    fig.add_trace(go.Scatter(x=qs, y=[r["edge_upper"] for r in rows], mode="lines", name="Rand (1 + √q)²", line=dict(color=C.COLOR_EDGE, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=qs, y=[r["smallest"] for r in rows], mode="lines+markers", name="kleinster positiver Eigenwert", line=dict(color="#54a24b", width=3), hovertemplate="p/n = %{x}: %{y:.3f}<extra>kleinster</extra>"))
    fig.add_trace(go.Scatter(x=qs, y=[r["edge_lower"] for r in rows], mode="lines", name="Rand (1 − √q)²", line=dict(color="#54a24b", width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="p/n (Kennzahlen je Woche)", type="category")
    fig.update_yaxes(title_text="Eigenwert (wahr: alle gleich 1)")
    return _base(fig, height)


def build_cond(rows: list[dict], height: int = 300):
    """Konditionszahl der Stichprobenmatrix gegen p/n (nur p/n < 1) gegen die Formel, logarithmisch."""
    sub = sorted((r for r in rows if r["cond"] is not None), key=lambda r: r["q"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["q"]) for r in sub], y=[r["cond"] for r in sub], name="gemessen", marker_color=C.COLOR_SAMPLE, hovertemplate="p/n = %{x}: %{y:.1f}<extra>gemessen</extra>"))
    fig.add_trace(go.Scatter(x=[str(r["q"]) for r in sub], y=[r["cond_formula"] for r in sub], mode="markers", name="Formel ((1 + √q)/(1 − √q))²", marker=dict(symbol="diamond-open", size=12, color=C.COLOR_EDGE, line=dict(width=2)),
                             hovertemplate="p/n = %{x}: %{y:.1f}<extra>Formel</extra>"))
    fig.update_xaxes(title_text="p/n", type="category")
    fig.update_yaxes(title_text="Konditionszahl", type="log")
    return _base(fig, height)


def build_above(rows: list[dict], height: int = 340):
    """Zahl der Eigenwerte über dem Rand gegen n (durchgezogen: gemessen, gestrichelt: wahr), eine Farbe je Wahrheit."""
    fig = go.Figure()
    for truth in C.TRUTHS:
        sub = sorted((r for r in rows if r["truth"] == truth), key=lambda r: r["n"])
        ns = [r["n"] for r in sub]
        fig.add_trace(go.Scatter(x=ns, y=[r["mean_above"] for r in sub], mode="lines+markers", name=C.TRUTHS[truth], line=dict(color=C.TRUTH_COLORS[truth], width=3),
                                 hovertemplate="n = %{x}: %{y:.2f}<extra>" + C.TRUTHS[truth] + "</extra>"))
        fig.add_trace(go.Scatter(x=ns, y=[r["true_above"] for r in sub], mode="lines", name="wahr: " + C.TRUTHS[truth], line=dict(color=C.TRUTH_COLORS[truth], width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"))
    ns = sorted({r["n"] for r in rows})
    fig.update_xaxes(title_text="Zahl der Wochen n (p = 50)", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Eigenwerte über dem Rand")
    return _base(fig, height)


def build_errors(rows: list[dict], truth: str, p: int = C.ERR_P, height: int = 340):
    """Frobenius-Fehler der fünf Schätzer gegen n für eine Wahrheit."""
    sub = sorted((r for r in rows if r["truth"] == truth and r["p"] == p), key=lambda r: r["n"])
    ns = [r["n"] for r in sub]
    fig = go.Figure()
    for k in C.ESTIMATORS:
        fig.add_trace(go.Scatter(x=ns, y=[r["frob"][k] for r in sub], mode="lines+markers", name=C.ESTIMATORS[k], line=dict(color=C.ESTIMATOR_COLORS[k], width=3, dash="dot" if k == "orakel" else "solid"),
                                 hovertemplate="n = %{x}: %{y:.2f}<extra>" + C.ESTIMATOR_SHORT[k] + "</extra>"))
    fig.update_xaxes(title_text=f"Zahl der Wochen n (p = {p})", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Fehler (Frobenius-Norm)", rangemode="tozero")
    return _base(fig, height)


def build_error_bars(a_errors: dict, height: int = 300):
    """Fehler der fünf Schätzer für die Matrix der Hauptansicht (Balken)."""
    keys = list(C.ESTIMATORS)
    fig = go.Figure(go.Bar(x=[C.ESTIMATOR_SHORT[k] for k in keys], y=[a_errors[k]["frob"] for k in keys], marker_color=[C.ESTIMATOR_COLORS[k] for k in keys], hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title_text="Fehler (Frobenius-Norm)", rangemode="tozero")
    return _base(fig, height)


def build_wide(rows: list[dict], height: int = 320):
    """p > n: Fehler der fünf Schätzer je n (gruppierte Balken)."""
    sub = sorted((r for r in rows if r["p"] == C.WIDE_P), key=lambda r: r["n"])
    labels = [f"n = {r['n']}" for r in sub]
    fig = go.Figure()
    for k in C.ESTIMATORS:
        fig.add_trace(go.Bar(x=labels, y=[r["frob"][k] for r in sub], name=C.ESTIMATORS[k], marker_color=C.ESTIMATOR_COLORS[k]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Fehler (Frobenius-Norm), p = {C.WIDE_P}", rangemode="tozero")
    return _base(fig, height)
