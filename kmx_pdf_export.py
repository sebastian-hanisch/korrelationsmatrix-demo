"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis, Spektrum und Schätzer der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

import math

from fpdf import FPDF

import kmx_constants as C
import kmx_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "φ": "phi", "σ": "sigma", "τ": "tau",
             "²": "^2", "√": "sqrt ", "…": "...", "π": "pi", "|": "|"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Korrelationsmatrix bei vielen Variablen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Wahrheit: {C.TRUTHS[s.truth]}   Kennzahlen p: {s.p}   Wochen n: {s.n}   p/n: {s.p / s.n:.2f}   Permutationen: {a.perms_used}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, ev.verdict_main(a)[1])
    _para(pdf, 5.5, ev.verdict_spectrum(a)[1])
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Schätzer der Korrelationsmatrix (Fehler gegen die Wahrheit)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, name in C.ESTIMATORS.items():
        e = a.errors[key]
        cond = "unendlich" if not math.isfinite(e["cond"]) else f"{e['cond']:.1f}"
        _para(pdf, 5.5, f"{name}: Frobenius {e['frob']:.3f}, Spektralnorm {e['spectral']:.3f}, Konditionszahl {cond}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Korrelationsmatrix. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
