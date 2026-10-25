"""Korrelationsmatrix bei vielen Variablen: was ist noch wahr? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Korrelations-Reihe. p Kennzahlen eines Standorts über n Wochen, die wahre Korrelationsmatrix ist bekannt (Einheit, Kette, ein Faktor, Blöcke): größte Scheinkorrelation,
Spektrum gegen Marchenko–Pastur, Zahl der echten Eigenwerte und der Fehler von fünf Schätzern (Stichprobe, lineare und nichtlineare Schrumpfung, Abschneiden, Orakel).

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import kmx_constants as C
import kmx_evaluation as ev
from kmx_layout import dunkles_design, kopf_css
from kmx_pdf_export import build_pdf
from kmx_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kmx_visualization import (
    build_above,
    build_cond,
    build_error_bars,
    build_errors,
    build_heatmap,
    build_noise_lines,
    build_spectrum,
    build_spectrum_grid,
    build_wide,
)

st.set_page_config(page_title="Korrelationsmatrix – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kmx_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔲 Korrelationsmatrix bei vielen Variablen: was ist noch wahr?")
st.markdown(
    """
**Fünftes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios mit dem Spektrum einer geschätzten Korrelationsmatrix**. **p Kennzahlen** eines Standorts, gemessen über **n Wochen**; die
**wahre Korrelationsmatrix ist bekannt** (vier Formen). Die ersten Stücke betrachteten **ein** Paar. Hier sind es p(p − 1)/2 Paare zugleich, und wenn p nicht klein gegen n ist, ist die
Stichprobenmatrix kein verkleinertes Abbild der Wahrheit: ihr größter Eintrag ist Zufall, ihre Eigenwerte sind verzerrt, sie ist schlecht konditioniert oder gar nicht invertierbar.
**Wie viel davon ist echt, und welcher Schätzer verbessert die Matrix wann?**

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Eine Schrumpfung als Baustein des Prognoseabgleichs steht in `hierarchy-forecast-demo`,
Eigenwerte realer Kennzahlen in `pca-demo`, das Testen vieler Varianten in `multiple-testing-demo`, das Netz aus Teilkorrelationen im nächsten Stück.
"""
)

with st.expander("So funktionieren die Schätzer", expanded=True):
    st.markdown(
        """
Die **Stichprobenmatrix** R̂ ist die Korrelationsmatrix der n Wochen. Ihre **Eigenwerte** sind bei p/n = q auch dann verteilt, wenn alle Kennzahlen unabhängig sind (**Marchenko–Pastur**: zwischen (1 − √q)² und
(1 + √q)²). Ein Schätzer verbessert R̂, indem er die Eigenwerte (die Eigenvektoren bleiben) korrigiert:

| Schätzer | Idee | hilft bei | schadet bei |
|---|---|---|---|
| **Stichprobenmatrix** | die Korrelationen der n Wochen | n ≫ p | p ≈ n, p > n |
| **lineare Schrumpfung** | R̂ mit der Einheitsmatrix mischen, Gewicht aus den Daten | Einheit, Kette | ein Faktor (kein Effekt) |
| **Eigenwerte abschneiden** | alle Eigenwerte unter dem Marchenko–Pastur-Rand durch ihr Mittel ersetzen | ein Faktor | Kette, kleine Strukturen |
| **nichtlineare Schrumpfung** | jeden Eigenwert einzeln aus der Dichte aller Eigenwerte korrigieren | Kette, Blöcke, Faktor | Einheit (schlechter als die lineare) |
| **Orakel** | Eigenwerte uᵀRu mit der WAHREN Matrix R | – (nur zum Vergleich, nicht berechenbar) | – |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    truth = st.selectbox("Wahre Matrix", list(C.TRUTHS), format_func=C.TRUTHS.get, key="truth_select",
                         help="Wie die p Kennzahlen wirklich zusammenhängen: nicht (Einheit), mit den Nachbarn (Kette), über einen gemeinsamen Faktor oder in fünf Gruppen (Blöcke; bei p < 50 weniger Blöcke).")
    p = st.slider("Kennzahlen p", *bounds("p_slider"), key="p_slider", help="Zahl der Kennzahlen (Spalten). Es gibt p(p − 1)/2 Paare.")
    n = st.slider("Wochen n", *bounds("n_slider"), key="n_slider", help="Zahl der Beobachtungen (Zeilen). p/n ist das Verhältnis, auf das es ankommt.")
    st.markdown("**Verfahren**")
    perms = st.slider("Permutationen B", *bounds("perm_slider"), step=100, key="perm_slider",
                      help="Wie oft jede Spalte für sich umgeordnet wird, um die Maximum-Schwelle zu bestimmen. Bei großem p und n begrenzt das Budget B · n · p² die Zahl.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"truth_select": truth, "p_slider": int(p), "n_slider": int(n), "perm_slider": int(perms), "seed_input": int(seed)})

settings = ev.Settings(truth, int(p), int(n), int(perms), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
cols = st.columns(5)
cols[0].metric("Größtes |r|", f"{a.max_abs:.2f}", help="Der größte Betrag eines Eintrags über der Diagonale der Stichprobenmatrix.")
cols[1].metric(f"Paare naiv (von {a.counts['pairs']})", f"{a.counts['naiv']}", help=f"Paare mit |r| über {a.thresholds['naiv']:.3f}, der Schwelle eines einzelnen t-Tests.")
cols[2].metric("Paare (streng)", f"{a.counts['bonferroni']} / {a.counts['max']}", help=f"Bonferroni (|r| > {a.thresholds['bonferroni']:.3f}) / Maximum-Schwelle (|r| > {a.thresholds['max']:.3f}).")
cols[3].metric("Rang", f"{a.rank} von {a.settings.p}", help="Zahl der Eigenwerte über null. Bei p > n höchstens n − 1.")
cols[4].metric("Konditionszahl", "unendlich" if not math.isfinite(a.cond) else f"{a.cond:,.0f}".replace(",", " "), help="Größter durch kleinsten Eigenwert; bei einer Einheitsmatrix wäre sie 1.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_heatmap(a.R_hat), width="stretch", key="main_sample")
    st.caption("Stichprobenmatrix: rot = positive, blau = negative Korrelation. Bei der Einheitsmatrix sollte alles weiß sein.")
with right:
    st.plotly_chart(build_heatmap(a.R_true), width="stretch", key="main_true")
    st.caption("Wahre Matrix (so erzeugt, aber im echten Leben unbekannt).")
st.plotly_chart(build_spectrum(a.eig, a.eig_true, a.settings.p / a.settings.n), width="stretch", key="main_spectrum")
st.caption("Eigenwerte der Stichprobenmatrix (blau, Dichte), die Marchenko–Pastur-Verteilung (schwarz) und die wahren Eigenwerte (rote Striche). Gestrichelt: der obere Rand (1 + √(p/n))². "
           "Bei p > n liegt eine Spitze bei 0 (die fehlenden Eigenwerte).")
kind, text = ev.verdict_spectrum(a)
getattr(st, kind)(text)
st.plotly_chart(build_error_bars(a.errors), width="stretch", key="main_errors")
kind, text = ev.verdict_estimators(a)
getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="korrelationsmatrix-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wie viele Paare sind echt?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/matrix_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")

# --- E1 Rauschmaximum ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das größte |r| bei reinem Rauschen")
st.caption("Wahre Matrix = Einheit, alle Kennzahlen unabhängig. Anteil der Datensätze mit mindestens einem Paar über der Schwelle (erwartet 5 % bei den strengen Schwellen). Vorgerechnet: 4 Seeds mit je 250 Datensätzen.")
if grid is not None:
    e1c1, e1c2 = st.columns(2)
    e1_n = e1c1.selectbox("Wochen n", list(C.NOISE_NS), index=0, key="e1_n")
    e1_p = e1c2.selectbox("Kennzahlen p für das Verdikt", list(C.NOISE_PS), index=1, key="e1_p")
    st.plotly_chart(build_noise_lines(grid["noise"], e1_n), width="stretch", key="noise_chart")
    kind, text = ev.verdict_noise(grid["noise"], e1_p, e1_n)
    getattr(st, kind)(text)
    st.caption("Die naive Schwelle gilt für ein einzelnes Paar. Bei p(p − 1)/2 Paaren ist das Maximum fast immer darüber. Bonferroni und die Maximum-Schwelle (Quantil des größten |r| unter Unabhängigkeit) fallen hier zusammen.")

st.markdown("---")

# --- E2 Spektrum ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Eigenwerte und Konditionszahl der Stichprobenmatrix")
st.caption("Wahre Matrix = Einheit (alle Eigenwerte 1), p = 100, q = p/n. Mittel über 4 Seeds mit je 100 Datensätzen.")
if grid is not None:
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(build_spectrum_grid(grid["spec"]), width="stretch", key="spec_chart")
    with s2:
        st.plotly_chart(build_cond(grid["spec"]), width="stretch", key="cond_chart")
    kind, text = ev.verdict_spectrum_grid(grid["spec"])
    getattr(st, kind)(text)
    st.caption("Der größte Eigenwert liegt knapp unter dem oberen Rand, der kleinste knapp darüber. Ab q > 1 sind n − 1 Eigenwerte positiv und der Rest null; die Konditionszahl ist dann unendlich (im Diagramm links fehlt der Wert).")

st.markdown("---")

# --- E3 Wie viele Eigenwerte sind echt? ----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie viele Eigenwerte sind echt?")
st.caption("Regel: Eigenwerte über dem Marchenko–Pastur-Rand sind echt. p = 50, Mittel über 4 Seeds mit je 100 Datensätzen; gestrichelt die wahre Zahl.")
if grid is not None:
    st.plotly_chart(build_above(grid["above"]), width="stretch", key="above_chart")
    e3_truth = st.selectbox("Wahrheit für das Verdikt", list(C.TRUTHS), format_func=C.TRUTHS.get, index=1, key="e3_truth")
    kind, text = ev.verdict_above(grid["above"], e3_truth)
    getattr(st, kind)(text)
    st.caption("Die Regel trifft bei Einheit, Faktor und Blöcken. Bei der Kette ist der größte wahre Eigenwert (2,98) klein gegen den Rand, und das Rauschen bläht die Zahl auf.")

st.markdown("---")

# --- E4 Welcher Schätzer wann? -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Welcher Schätzer wann?")
st.caption("Fehler der Korrelationsmatrix gegen die Wahrheit (Frobenius-Norm), p = 50, Mittel über 4 Seeds mit je 50 Datensätzen.")
if grid is not None:
    e4c1, e4c2 = st.columns(2)
    e4_truth = e4c1.selectbox("Wahrheit", list(C.TRUTHS), format_func=C.TRUTHS.get, index=2, key="e4_truth")
    e4_n = e4c2.selectbox("Wochen n für das Verdikt", list(C.ERR_NS), index=1, key="e4_n")
    st.plotly_chart(build_errors(grid["err"], e4_truth), width="stretch", key="error_chart")
    kind, text = ev.verdict_errors(grid["err"], e4_truth, e4_n)
    getattr(st, kind)(text)
    kind, text = ev.verdict_winners(grid["err"])
    getattr(st, kind)(text)
    kind, text = ev.verdict_inverse(grid["err"], 60)
    getattr(st, kind)(text)
    st.caption("Die lineare Schrumpfung zieht zur Einheitsmatrix und hilft deshalb, wo die Wahrheit nahe an der Einheit liegt. Das Abschneiden hält die großen Eigenwerte und mittelt den Rest; "
               "es hilft bei einem Faktor und schadet bei vielen kleinen Strukturen. Die nichtlineare Schrumpfung kennt weder Ziel noch Struktur und liegt meist nahe am Orakel.")

st.markdown("---")

# --- E5 p > n ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Mehr Kennzahlen als Wochen")
st.caption("Wahrheit Kette, p = 100, n = 30, 60 und 120; Mittel über 4 Seeds mit je 50 Datensätzen.")
if grid is not None:
    st.plotly_chart(build_wide(grid["err"]), width="stretch", key="wide_chart")
    kind, text = ev.verdict_wide(grid["err"])
    getattr(st, kind)(text)
    st.caption("Bei p > n sind nur n − 1 Eigenwerte positiv, die Stichprobenmatrix ist nicht invertierbar. Beide Schrumpfungen liefern eine volle, gut konditionierte Matrix.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die vier Wahrheiten stehen für echte Kennzahlen** | es sind Beispiele; echte Matrizen sind Mischungen, und wie oft welche Form vorkommt, zeigt die Demo nicht | – |
| **Die Daten sind normalverteilt und unabhängig über die Wochen** | schwere Ränder und Autokorrelation verschieben alle Zahlen (Stücke 1 und 4) | Stücke 1 und 4 |
| **Die Marchenko–Pastur-Regel zählt echte Eigenwerte** | bei vielen kleinen Strukturen (Kette) zählt sie falsch (Experiment 3) | Schätzer mit Modell (Faktor, Block) |
| **Schrumpfung hilft immer** | die lineare hilft nicht beim Faktor, das Abschneiden schadet bei der Kette, die nichtlineare ist bei der Einheit schlechter (Experiment 4) | den Schätzer nach der vermuteten Struktur wählen |
| **Der Fehler der Matrix ist der Fehler der Anwendung** | gemessen wird die Frobenius-Norm der Matrix, nicht der Schaden einer Entscheidung (etwa einer Prognose oder eines Portfolios) | Anwendungsverlust messen |
| **Ein Paar „signifikant“ heißt echter Zusammenhang** | bei vielen Paaren ist das größte \|r\| auch ohne Zusammenhang groß (Experiment 1) | strenge Schwellen |
| **Die Orakel-Zeile ist ein Schätzer** | sie nutzt die Wahrheit und zeigt nur, was mit diesen Eigenvektoren höchstens möglich ist | – |
| **Netz und Teilkorrelation** | wer mit wem zusammenhängt (nach Abzug der übrigen), zeigt diese Demo nicht | nächstes Stück (Korrelationsnetz) |
| **Ursachen** | auch eine echte Korrelation sagt nicht, welche Kennzahl welche bewegt | Linie *Kausale Diagramme und Adjustierung* |
"""
)
st.caption("Alle Daten sind simuliert; die Wahrheiten sind Modellwahlen, kein Befund über echte Standorte. Die nichtlineare Schrumpfung ist die analytische Fassung von Ledoit und Wolf (2020), hier aus der Beschreibung nachgebaut und gegen das Orakel geprüft.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Beobachtungen von $p$ Kennzahlen; $q = p/n$.

| Symbol | Bedeutung |
|---|---|
| $R$ | wahre Korrelationsmatrix ($p\times p$, Diagonale 1) |
| $\hat R$ | Stichprobenmatrix, $\hat R = Z^\top Z/n$ mit den standardisierten Daten $Z$ |
| $\lambda_i, u_i$ | Eigenwerte und Eigenvektoren von $\hat R$ |
| $\tilde R$ | ein Schätzer von $R$ |

**Marchenko–Pastur.** Bei $R = I$ liegen die Eigenwerte von $\hat R$ für großes $p$ und $n$ bei festem $q$ in $[(1-\sqrt q)^2,\ (1+\sqrt q)^2]$ mit der Dichte

$$ f_q(x) = \frac{\sqrt{(b - x)(x - a)}}{2\pi q\,x}, \qquad a = (1-\sqrt q)^2,\ b = (1+\sqrt q)^2 $$

(für $q > 1$ hat dieser Teil die Masse $1/q$, die übrigen Eigenwerte sind null). Die **Konditionszahl** wird für $q < 1$ etwa

$$ \kappa \approx \left(\frac{1+\sqrt q}{1-\sqrt q}\right)^2 $$

**Maximum-Schwelle.** Mit dem größten Betrag $T = \max_{i<j}\lvert\hat r_{ij}\rvert$ und seinen Werten $T^*_b$ nach unabhängiger Umordnung jeder Spalte ($b = 1,\dots,B$) ist die Schwelle das $\lceil(1-\alpha)(B+1)\rceil$-te kleinste $T^*_b$. Die **Bonferroni-Schwelle** ist die des t-Tests zum Niveau $\alpha/\binom p2$:

$$ \lvert r\rvert > \frac{t_{n-2,\,1-\alpha/(2m)}}{\sqrt{t_{n-2,\,1-\alpha/(2m)}^2 + n-2}}, \qquad m = \binom p2 $$

**Lineare Schrumpfung** (Ledoit–Wolf, Ziel $\mu I$ mit $\mu=1$) mit dem aus den Daten geschätzten Gewicht $a$:

$$ \tilde R = (1-a)\,\hat R + a\,I $$

**Abschneiden der Eigenwerte.** Eigenwerte $\lambda_i \le (1+\sqrt q)^2$ werden durch ihr Mittel $\bar\lambda_{\text{Rauschen}}$ ersetzt, die übrigen bleiben; danach wird die Diagonale auf 1 gebracht:

$$ \tilde R = \sum_i d_i\, u_i u_i^\top, \qquad d_i = \begin{cases} \bar\lambda_{\text{Rauschen}} & \lambda_i \le (1+\sqrt q)^2 \\ \lambda_i & \text{sonst} \end{cases} $$

**Nichtlineare Schrumpfung** und **Orakel.** Beide behalten die Eigenvektoren $u_i$ und ersetzen die Eigenwerte durch $d_i$; das Orakel nimmt mit der WAHREN Matrix

$$ d_i^{\text{Orakel}} = u_i^\top R\, u_i $$

und ist unter allen so gebauten Schätzern der beste im Frobenius-Fehler (vor dem Normieren der Diagonale). Die analytische nichtlineare Schrumpfung schätzt $d_i$ ohne $R$ aus einer Kerndichte der Eigenwerte und ihrer Hilbert-Transformierten.

**Fehler** der Schätzer: $\lVert \tilde R - R\rVert_F$; Fehler der Inversen: $\lVert \tilde R^{-1} - R^{-1}\rVert_F / \lVert R^{-1}\rVert_F$.

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
