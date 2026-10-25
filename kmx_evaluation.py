"""Auswertung: Einzelmatrix, Zellen des vorgerechneten Rasters, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import kmx_algorithm as K
import kmx_constants as C
import prs_scenario as S

GRID_PATH = Path(__file__).resolve().parent / "data" / "matrix_grid.json"
COMPUTABLE = ("stichprobe", "linear", "clip", "nl")        # Schätzer ohne Kenntnis der Wahrheit


# --- Einzelmatrix ----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Settings:
    truth: str = "einheit"
    p: int = C.DEFAULT_P
    n: int = C.DEFAULT_N
    perms: int = C.DEFAULT_PERM
    seed: int = C.DEFAULT_SEED

    def key(self) -> tuple:
        return (self.truth, self.p, self.n, self.perms, self.seed)

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    R_hat: np.ndarray
    R_true: np.ndarray
    eig: np.ndarray            # Eigenwerte der Stichprobenmatrix, aufsteigend
    eig_true: np.ndarray
    edge: float                # Marchenko–Pastur-Rand (1 + √(p/n))²
    max_abs: float
    thresholds: dict           # "naiv", "bonferroni", "max"
    counts: dict               # Paare über der Schwelle: "pairs", "naiv", "bonferroni", "max"
    perms_used: int
    rank: int
    cond: float
    above: int                 # Eigenwerte der Stichprobenmatrix über dem Rand
    true_above: int            # wahre Eigenwerte über dem Rand
    estimates: dict            # Schlüssel -> Matrix
    errors: dict               # Schlüssel -> {"frob", "spectral", "cond"}
    weight: float              # Gewicht der linearen Schrumpfung


def affordable_perms(perms: int, n: int, p: int, budget: int = C.PERM_BUDGET) -> int:
    """So viele Permutationen, wie das Budget B · n · p² erlaubt (mindestens 49)."""
    return max(49, min(perms, budget // max(1, n * p * p)))


def analyse(s: Settings) -> Analysis:
    data_seed, perm_seed = S.substream_seeds(s.seed, 2)
    X = K.draw(s.truth, s.n, s.p, S.VRng(data_seed))
    R_true = K.true_matrix(s.truth, s.p)
    R = K.sample_corr(X)
    B = affordable_perms(s.perms, s.n, s.p)
    r = np.abs(K.offdiag(R))
    thr = {"naiv": K.r_threshold(s.n), "bonferroni": K.r_threshold(s.n, tests=len(r)), "max": K.max_threshold(X, B, S.VRng(perm_seed))}
    counts = {"pairs": len(r), **{k: int((r > thr[k]).sum()) for k in thr}}
    est = K.estimate_all(X, R_true)
    weight = est.pop("a")
    errors = {k: {"frob": K.frobenius_error(M, R_true), "spectral": K.spectral_error(M, R_true), "cond": K.condition_number(M)} for k, M in est.items()}
    edge = K.mp_edges(s.p / s.n)[1]
    return Analysis(s, R, R_true, K.spectrum(R), K.spectrum(R_true), edge, float(r.max()), thr, counts, B, K.matrix_rank(R), K.condition_number(R),
                    int((K.spectrum(R) > edge).sum()), int((K.spectrum(R_true) > edge).sum()), est, errors, weight)


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, key: str, p: int, n: int, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, Wahrheit, p und n)."""
    kinds = ("noise", "ref", "spec", "above", "err", "probe")
    return s * 1_000_003 + n * 101 + p * 7919 + list(C.TRUTHS).index(key) * 1013 + kinds.index(kind) * 100_000_007


def _finite_mean(values: list[float]) -> float | None:
    """Mittel, None wenn ein Wert unendlich ist (z. B. Konditionszahl einer singulären Matrix)."""
    return None if any(not math.isfinite(v) for v in values) else float(np.mean(values))


def noise_cell(p: int, n: int, runs_per_seed: int = C.NOISE_RUNS, seeds=C.GRID_SEEDS, ref: int = C.NOISE_REF) -> dict:
    """E1: reines Rauschen (wahre Einheitsmatrix): größtes |r|, Zahl der Paare über der naiven Schwelle und Anteil der Datensätze mit mindestens einem Paar über der naiven, der Bonferroni-
    und der Maximum-Schwelle (95 %-Quantil des größten |r| aus `ref` unabhängigen Referenzmatrizen)."""
    naive, bonf = K.r_threshold(n), K.r_threshold(n, tests=p * (p - 1) // 2)
    rr = S.VRng(cell_seed("ref", "einheit", p, n, 999))
    r_max = float(np.quantile([K.max_abs_offdiag(K.sample_corr(rr.normal((n, p)))) for _ in range(ref)], 1 - C.ALPHA))
    per_seed = []
    for s in seeds:
        rng = S.VRng(cell_seed("noise", "einheit", p, n, s))
        peaks, counts = [], []
        for _ in range(runs_per_seed):
            r = np.abs(K.offdiag(K.sample_corr(rng.normal((n, p)))))
            peaks.append(r.max())
            counts.append(int((r > naive).sum()))
        peaks = np.array(peaks)
        per_seed.append([peaks.mean(), np.mean(counts), np.mean(np.array(counts) > 0), (peaks > bonf).mean(), (peaks > r_max).mean()])
    m = np.mean(per_seed, axis=0)
    sd = np.std(per_seed, axis=0, ddof=1) if len(per_seed) > 1 else np.zeros(5)
    return {"p": p, "n": n, "pairs": p * (p - 1) // 2, "runs": runs_per_seed * len(seeds), "mean_max": float(m[0]), "naive_pairs": float(m[1]), "any_naive": float(m[2]), "any_bonferroni": float(m[3]),
            "any_max": float(m[4]), "r_naive": naive, "r_bonferroni": bonf, "r_max": r_max, "mean_max_sd": float(sd[0])}


def spectrum_cell(q: float, p: int = C.SPEC_P, runs_per_seed: int = C.SPEC_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E2: Eigenwerte der Stichprobenmatrix bei wahrer Einheitsmatrix mit p/n = q: größter, kleinster positiver, Rang, Konditionszahl."""
    n = int(round(p / q))
    big, small, ranks, conds = [], [], [], []
    for s in seeds:
        rng = S.VRng(cell_seed("spec", "einheit", p, int(round(q * 100)), s))
        for _ in range(runs_per_seed):
            R = K.sample_corr(rng.normal((n, p)))
            w = K.spectrum(R)
            pos = w[w > K.RANK_TOL * w[-1]]
            big.append(w[-1])
            small.append(pos[0])
            ranks.append(len(pos))
            conds.append(K.condition_number(R))
    a, b = K.mp_edges(q)
    return {"q": q, "p": p, "n": n, "runs": len(big), "largest": float(np.mean(big)), "smallest": float(np.mean(small)), "rank": float(np.mean(ranks)), "cond": _finite_mean(conds),
            "edge_upper": b, "edge_lower": a, "cond_formula": None if q >= 1 else K.mp_condition(q)}


def above_cell(truth: str, n: int, p: int = C.ABOVE_P, runs_per_seed: int = C.ABOVE_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E3: Zahl der Eigenwerte der Stichprobenmatrix über dem Marchenko–Pastur-Rand, gegen die Zahl der wahren Eigenwerte über dem Rand."""
    edge = K.mp_edges(p / n)[1]
    w_true = K.spectrum(K.true_matrix(truth, p))
    counts = []
    for s in seeds:
        rng = S.VRng(cell_seed("above", truth, p, n, s))
        counts += [int((K.spectrum(K.sample_corr(K.draw(truth, n, p, rng))) > edge).sum()) for _ in range(runs_per_seed)]
    return {"truth": truth, "p": p, "n": n, "runs": len(counts), "edge": edge, "mean_above": float(np.mean(counts)), "true_above": int((w_true > edge).sum()), "largest_true": float(w_true[-1])}


def error_cell(truth: str, p: int, n: int, runs_per_seed: int = C.ERR_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E4 und E5: Fehler der fünf Schätzer (Frobenius, Mittel und Streuung über die Seeds), Konditionszahl (None bei singulären Matrizen), relative Fehler der Inversen (nur n > p) und Gewicht a."""
    R_true = K.true_matrix(truth, p)
    frob = {k: [] for k in C.ESTIMATORS}
    cond = {k: [] for k in COMPUTABLE}
    inv = {k: [] for k in ("stichprobe", "linear", "nl")}
    weights = []
    per_seed = {k: [] for k in C.ESTIMATORS}
    for s in seeds:
        rng = S.VRng(cell_seed("err", truth, p, n, s))
        block = {k: [] for k in C.ESTIMATORS}
        for _ in range(runs_per_seed):
            est = K.estimate_all(K.draw(truth, n, p, rng), R_true)
            weights.append(est.pop("a"))
            for k, M in est.items():
                block[k].append(K.frobenius_error(M, R_true))
                if k in cond:
                    cond[k].append(K.condition_number(M))
                if k in inv and n > p:
                    inv[k].append(K.inverse_error(M, R_true))
        for k in C.ESTIMATORS:
            frob[k] += block[k]
            per_seed[k].append(float(np.mean(block[k])))
    return {"truth": truth, "p": p, "n": n, "runs": len(weights), "frob": {k: float(np.mean(v)) for k, v in frob.items()}, "frob_sd": {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in per_seed.items()},
            "cond": {k: _finite_mean(v) for k, v in cond.items()}, "inverse": {k: (_finite_mean(v) if v else None) for k, v in inv.items()}, "weight": float(np.mean(weights))}


def grid_probe() -> dict:
    """Kleine feste Zelle und eine feste Einzelmatrix, die der Test nachrechnet: ändert sich Erzeuger oder Schätzer, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings("kette", 30, 40, 99, 11))
    return {"cell": error_cell("kette", 20, 30, 6, seeds=(7, 8)), "noise": noise_cell(12, 25, 20, seeds=(7,), ref=60), "frob": {k: v["frob"] for k, v in a.errors.items()}, "max_abs": a.max_abs,
            "thresholds": a.thresholds, "counts": a.counts, "weight": a.weight}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


def best_computable(cell: dict) -> str:
    """Der Schätzer ohne Kenntnis der Wahrheit mit dem kleinsten mittleren Fehler in der Zelle."""
    return min(COMPUTABLE, key=lambda k: cell["frob"][k])


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f} %"


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Wie viele Paare sind „signifikant“, wenn man sie zählt, wie die Wahrheit aussieht, und was sagt das Spektrum?"""
    s, c, t = a.settings, a.counts, a.thresholds
    base = (f"Gemessen: größtes |r| = **{a.max_abs:.2f}**; Schwellen für |r|: naiv {t['naiv']:.2f}, Bonferroni {t['bonferroni']:.2f}, Maximum {t['max']:.2f}. "
            f"Paare darüber (von {c['pairs']}): naiv **{c['naiv']}**, Bonferroni **{c['bonferroni']}**, Maximum **{c['max']}**.")
    if s.truth == "einheit":
        if c["bonferroni"] == 0 and c["max"] == 0:
            return "success", f"{base} Die Wahrheit ist die Einheitsmatrix (kein Paar hängt zusammen): jedes naiv „signifikante“ Paar ist ein Fehlalarm, die beiden strengen Schwellen melden keines."
        return "warning", f"{base} Die Wahrheit ist die Einheitsmatrix: auch eine der strengen Schwellen schlägt hier an, das passiert in etwa 5 % der Datensätze."
    return "info", f"{base} Die Wahrheit hat Struktur ({C.TRUTHS[s.truth]}); die Zahlen der Schwellen mischen echte Paare und Rauschen."


def verdict_spectrum(a: Analysis) -> tuple[str, str]:
    """Spektrum der Stichprobenmatrix: Rang, Konditionszahl und Eigenwerte über dem Rand gegen die wahre Zahl."""
    s = a.settings
    cond = "unendlich (die Matrix ist singulär)" if not math.isfinite(a.cond) else f"{a.cond:,.1f}".replace(",", " ")
    text = (f"p/n = {s.p / s.n:.2f}: größter Eigenwert **{a.eig[-1]:.2f}**, Rand (1 + √(p/n))² = {a.edge:.2f}, kleinster **{a.eig[0]:.3f}**; Rang **{a.rank}** von {s.p}, Konditionszahl **{cond}**. "
            f"Über dem Rand stehen **{a.above}** Eigenwerte, die Wahrheit hat **{a.true_above}**.")
    return ("warning" if a.rank < s.p else "info"), text


def verdict_estimators(a: Analysis) -> tuple[str, str]:
    """Der beste Schätzer ohne Kenntnis der Wahrheit auf dieser Matrix gegen die Stichprobenmatrix und das Orakel."""
    e = {k: a.errors[k]["frob"] for k in C.ESTIMATORS}
    best = min(COMPUTABLE, key=lambda k: e[k])
    text = f"Fehler (Frobenius): " + ", ".join(f"{C.ESTIMATOR_SHORT[k]} **{e[k]:.2f}**" for k in C.ESTIMATORS) + f". Am besten ohne Kenntnis der Wahrheit: **{C.ESTIMATORS[best]}**."
    if best == "stichprobe":
        return "warning", text + " Keine Schrumpfung verbessert die Stichprobenmatrix hier."
    return "success", text + f" Das sind {100 * (1 - e[best] / e['stichprobe']):.0f} % weniger Fehler als bei der Stichprobenmatrix, das Orakel erreicht {e['orakel']:.2f}."


def verdict_noise(rows: list[dict], p: int, n: int) -> tuple[str, str]:
    """E1: größtes |r| bei reinem Rauschen und die Datensätze mit mindestens einem Paar über der naiven bzw. der strengen Schwelle."""
    c = find(rows, p=p, n=n)
    return "info", (f"Bei p = {p} Kennzahlen und n = {n} Wochen liegt das größte |r| im Mittel bei **{c['mean_max']:.3f}**, die naive Schwelle bei {c['r_naive']:.3f}: "
                    f"**{_pct(c['any_naive'])}** der {c['runs']} Datensätze haben mindestens ein „signifikantes“ Paar, bei der Bonferroni-Schwelle ({c['r_bonferroni']:.3f}) {_pct(c['any_bonferroni'])}, "
                    f"bei der Maximum-Schwelle ({c['r_max']:.3f}) {_pct(c['any_max'])}.")


def verdict_spectrum_grid(rows: list[dict]) -> tuple[str, str]:
    """E2: Konditionszahl und Rang gegen p/n bei wahrer Einheitsmatrix."""
    rows = sorted(rows, key=lambda r: r["q"])
    low, high = next(r for r in rows if r["q"] < 1 and r["cond"] is not None), max((r for r in rows if r["q"] < 1), key=lambda r: r["q"])
    over = [r for r in rows if r["q"] > 1]
    text = (f"Bei wahrer Einheitsmatrix hat die Stichprobenmatrix bei p/n = {low['q']:g} die Konditionszahl **{low['cond']:.1f}** und bei p/n = {high['q']:g} **{high['cond']:,.0f}**".replace(",", " ")
            + (f"; bei p/n = {over[0]['q']:g} ({over[0]['n']} Wochen, {over[0]['p']} Kennzahlen) ist der Rang **{over[0]['rank']:.0f}** statt {over[0]['p']}." if over else "."))
    return "warning", text


def verdict_above(rows: list[dict], truth: str) -> tuple[str, str]:
    """E3: Zahl der Eigenwerte über dem Rand gegen die wahre Zahl bei kleinstem und größtem n."""
    rows = sorted((r for r in rows if r["truth"] == truth), key=lambda r: r["n"])
    first, last = rows[0], rows[-1]
    hit = all(abs(r["mean_above"] - r["true_above"]) < 0.5 for r in rows if r["n"] >= 120)
    text = (f"{C.TRUTHS[truth]}: bei n = {first['n']} stehen im Mittel **{first['mean_above']:.2f}** Eigenwerte über dem Rand ({first['edge']:.2f}), bei n = {last['n']} **{last['mean_above']:.2f}**; "
            f"wahr sind {first['true_above']} und {last['true_above']}.")
    return ("success" if hit else "warning"), text + (" Die Zählregel trifft ab n = 120." if hit else " Die Zählregel verfehlt die Wahrheit.")


def verdict_errors(rows: list[dict], truth: str, n: int) -> tuple[str, str]:
    """E4: der beste Schätzer ohne Wahrheit für eine Wahrheit und ein n, mit den Fehlern aller fünf."""
    c = find(rows, truth=truth, n=n, p=C.ERR_P)
    best = best_computable(c)
    e = c["frob"]
    text = (f"{C.TRUTHS[truth]}, p = {c['p']}, n = {n}: Fehler " + ", ".join(f"{C.ESTIMATOR_SHORT[k]} **{e[k]:.2f}**" for k in C.ESTIMATORS) + f". Am besten ohne Kenntnis der Wahrheit: **{C.ESTIMATORS[best]}**.")
    return ("warning" if best == "stichprobe" else "info"), text


def verdict_winners(rows: list[dict]) -> tuple[str, str]:
    """E4: welcher Schätzer ohne Wahrheit in wie vielen Zellen vorn liegt; kein Schätzer gewinnt überall."""
    cells = [r for r in rows if r["p"] == C.ERR_P]
    wins = {k: sum(best_computable(r) == k for r in cells) for k in COMPUTABLE}
    all_win = [k for k in COMPUTABLE if wins[k] == len(cells)]
    text = f"Von {len(cells)} Zellen (vier Wahrheiten, vier n) gewinnt " + ", ".join(f"{C.ESTIMATOR_SHORT[k]} {wins[k]}" for k in COMPUTABLE) + ". " + ("Ein Schätzer gewinnt überall." if all_win else "**Kein Schätzer gewinnt überall.**")
    return ("success" if all_win else "warning"), text


def verdict_inverse(rows: list[dict], n: int) -> tuple[str, str]:
    """E4: relative Fehler der Inversen bei einem n (nur n > p) für die Wahrheiten: Stichprobe gegen Schrumpfungen."""
    cells = [r for r in rows if r["p"] == C.ERR_P and r["n"] == n and r["inverse"]["stichprobe"] is not None]
    worst = max(cells, key=lambda r: r["inverse"]["stichprobe"])
    return "warning", (f"Bei n = {n} (p = {C.ERR_P}) ist die Inverse der Stichprobenmatrix um bis zu **{100 * worst['inverse']['stichprobe']:.0f} %** falsch ({C.TRUTHS[worst['truth']]}); "
                       f"nach linearer Schrumpfung um {100 * worst['inverse']['linear']:.0f} %, nach nichtlinearer um {100 * worst['inverse']['nl']:.0f} %.")


def verdict_wide(rows: list[dict]) -> tuple[str, str]:
    """E5: p > n: Rang der Stichprobenmatrix und Fehler gegen die beiden Schrumpfungen bei kleinstem n."""
    c = min((r for r in rows if r["p"] == C.WIDE_P), key=lambda r: r["n"])
    return "warning", (f"Bei p = {c['p']} Kennzahlen und n = {c['n']} Wochen hat die Stichprobenmatrix höchstens den Rang {c['n'] - 1} und ist nicht invertierbar; ihr Fehler ist **{c['frob']['stichprobe']:.2f}**, "
                       f"nach linearer Schrumpfung **{c['frob']['linear']:.2f}**, nach nichtlinearer **{c['frob']['nl']:.2f}** (Orakel {c['frob']['orakel']:.2f}); beide Schrumpfungen liefern eine volle Matrix.")
