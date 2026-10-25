"""Kern: wahre Korrelationsmatrizen, Stichprobenmatrix, Rauschmaximum und Schwellen, Spektrum mit Marchenko–Pastur und vier Schätzer der Korrelationsmatrix.

Pearson, Student-Verteilung und der Zufallserzeuger kommen aus den wörtlichen Kopien (prs_algorithm, prs_scenario). Datensätze haben die Form (n, p): n Wochen, p Kennzahlen.
Schätzer: lineare Schrumpfung nach Ledoit–Wolf (Ziel: Einheitsmatrix), Abschneiden der Eigenwerte bis zum Marchenko–Pastur-Rand, analytische nichtlineare Schrumpfung nach Ledoit–Wolf
(2020, aus dem Gedächtnis; Prüfung über Eigenschaften und das Orakel) und das Orakel (Eigenvektoren der Stichprobenmatrix, Eigenwerte uᵀRu mit der wahren Matrix)."""

import math

import numpy as np

import kmx_constants as C
import prs_algorithm as PA

TRUTH_KEYS = tuple(C.TRUTHS)
RANK_TOL = 1e-10


# --- Wahrheiten und Daten --------------------------------------------------------------------------------------------------------------------

def true_matrix(truth: str, p: int) -> np.ndarray:
    """Wahre Korrelationsmatrix: Einheit, Kette ρ^|i−j|, ein Faktor (alle Paare 0,36), Blöcke zu zehn Kennzahlen mit 0,5 innen (der letzte Block kann kürzer sein)."""
    if p < 2:
        raise ValueError("mindestens zwei Kennzahlen nötig")
    idx = np.arange(p)
    if truth == "einheit":
        return np.eye(p)
    if truth == "kette":
        return C.CHAIN_RHO ** np.abs(idx[:, None] - idx[None, :])
    if truth == "faktor":
        R = np.full((p, p), C.FACTOR_RHO)
        np.fill_diagonal(R, 1.0)
        return R
    if truth == "bloecke":
        block = idx // C.BLOCK_SIZE
        R = np.where(block[:, None] == block[None, :], C.BLOCK_RHO, 0.0)
        np.fill_diagonal(R, 1.0)
        return R
    raise ValueError(f"unbekannte Wahrheit: {truth}")


def draw(truth: str, n: int, p: int, rng) -> np.ndarray:
    """n Wochen von p normalverteilten Kennzahlen mit der wahren Korrelationsmatrix (Ziehung: erst alle n·p Normalwerte, dann Mischung mit dem Cholesky-Faktor)."""
    if n < 3:
        raise ValueError("mindestens drei Wochen nötig")
    L = np.linalg.cholesky(true_matrix(truth, p))
    return rng.normal((n, p)) @ L.T


def sample_corr(X) -> np.ndarray:
    """Stichproben-Korrelationsmatrix (Spalten zentriert und auf Länge 1 gebracht); konstante Spalten geben NaN."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xc = Xc / np.linalg.norm(Xc, axis=0)
    return Xc.T @ Xc


# --- Rauschmaximum und Schwellen ---------------------------------------------------------------------------------------------------------------

def offdiag(R) -> np.ndarray:
    """Die p(p − 1)/2 Einträge über der Diagonale."""
    return np.asarray(R)[np.triu_indices(len(R), 1)]


def max_abs_offdiag(R) -> float:
    return float(np.abs(offdiag(R)).max())


def r_threshold(n: int, alpha: float = C.ALPHA, tests: int = 1) -> float:
    """Betrag von r, ab dem ein t-Test mit n − 2 Freiheitsgraden bei `tests` gleichzeitigen Tests (Bonferroni: α/tests) verwirft."""
    if n < 4:
        raise ValueError("mindestens vier Wochen nötig")
    t = PA.t_critical(n - 2, alpha / tests)
    return t / math.sqrt(t * t + n - 2)


def pair_counts(R, n: int, alpha: float = C.ALPHA) -> dict:
    """Zahl der Paare mit |r| über der naiven und über der Bonferroni-Schwelle."""
    r = np.abs(offdiag(R))
    return {"pairs": len(r), "naiv": int((r > r_threshold(n, alpha)).sum()), "bonferroni": int((r > r_threshold(n, alpha, len(r))).sum())}


def max_threshold(X, B: int, rng, alpha: float = C.ALPHA) -> float:
    """Schwelle für das größte |r|: Quantil (1 − α) der Nullverteilung, aus B Datensätzen mit je Spalte für sich zufällig umgeordnet (Spaltenpermutation); gezählt wird die
    ⌈(1 − α)(B + 1)⌉-te kleinste Maximum."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    peaks = np.empty(B)
    for b in range(B):
        order = np.argsort(rng.uniform((n, p)), axis=0)
        peaks[b] = max_abs_offdiag(sample_corr(np.take_along_axis(X, order, axis=0)))
    k = min(B, math.ceil((1.0 - alpha) * (B + 1)))
    return float(np.sort(peaks)[k - 1])


# --- Spektrum --------------------------------------------------------------------------------------------------------------------------------

def spectrum(R) -> np.ndarray:
    """Eigenwerte aufsteigend."""
    return np.linalg.eigvalsh(np.asarray(R, dtype=float))


def mp_edges(q: float) -> tuple[float, float]:
    """Ränder (1 − √q)² und (1 + √q)² der Marchenko–Pastur-Verteilung (q = p/n)."""
    if q <= 0:
        raise ValueError("q muss positiv sein")
    return (1.0 - math.sqrt(q)) ** 2, (1.0 + math.sqrt(q)) ** 2


def mp_density(x, q: float) -> np.ndarray:
    """Dichte des stetigen Teils der Marchenko–Pastur-Verteilung: √((b − x)(x − a))/(2π q x) zwischen den Rändern a, b (für q > 1 hat sie die Masse 1/q, der Rest liegt bei 0)."""
    a, b = mp_edges(q)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > a) & (x < b)
    out[inside] = np.sqrt((b - x[inside]) * (x[inside] - a)) / (2.0 * math.pi * q * x[inside])
    return out


def mp_condition(q: float) -> float:
    """Konditionszahl ((1 + √q)/(1 − √q))² der Stichprobenmatrix einer Einheitsmatrix (q < 1; sonst unendlich)."""
    if q >= 1.0:
        return float("inf")
    return ((1.0 + math.sqrt(q)) / (1.0 - math.sqrt(q))) ** 2


def matrix_rank(R) -> int:
    w = spectrum(R)
    return int((w > RANK_TOL * max(w[-1], 1.0)).sum())


def condition_number(R) -> float:
    """Größter durch kleinsten Eigenwert; unendlich, wenn der kleinste (numerisch) null ist."""
    w = spectrum(R)
    if w[0] <= RANK_TOL * w[-1]:
        return float("inf")
    return float(w[-1] / w[0])


def count_above_edge(R, n: int) -> int:
    """Zahl der Eigenwerte über dem Marchenko–Pastur-Rand (1 + √(p/n))²."""
    p = len(R)
    return int((spectrum(R) > mp_edges(p / n)[1]).sum())


# --- Schätzer --------------------------------------------------------------------------------------------------------------------------------

def standardize(X) -> np.ndarray:
    """Spalten zentrieren und durch die Standardabweichung (Teiler n) teilen; eine konstante Spalte ist ein Fehler."""
    X = np.asarray(X, dtype=float)
    sd = X.std(0)
    if (sd == 0).any():
        raise ValueError("eine Kennzahl ist konstant: die Korrelation ist nicht definiert")
    return (X - X.mean(0)) / sd


def ledoit_wolf(X) -> tuple[np.ndarray, float]:
    """Lineare Schrumpfung der Korrelationsmatrix (Ledoit–Wolf 2004, Ziel μI mit μ = 1): (1 − a) S + a I, S = ZᵀZ/n der standardisierten Daten; (Matrix, Gewicht a)."""
    Z = standardize(X)
    n, p = Z.shape
    S = Z.T @ Z / n
    mu = np.trace(S) / p
    d2 = float(((S - mu * np.eye(p)) ** 2).sum())
    if d2 == 0.0:
        return S, 0.0
    sq = (Z * Z).sum(1)
    b2 = float((sq**2).sum() - 2.0 * np.einsum("ij,jk,ik->i", Z, S, Z).sum() + n * (S**2).sum()) / n**2
    a = min(b2, d2) / d2
    return (1.0 - a) * S + a * mu * np.eye(p), float(a)


def _to_correlation(C_: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(C_))
    return C_ / np.outer(d, d)


def clip_eigenvalues(R, n: int) -> np.ndarray:
    """Eigenwerte bis zum Marchenko–Pastur-Rand durch ihr Mittel ersetzen (die Spur bleibt), danach die Diagonale auf 1 normieren."""
    R = np.asarray(R, dtype=float)
    w, U = np.linalg.eigh(R)
    noise = w <= mp_edges(len(R) / n)[1]
    if noise.any():
        w = w.copy()
        w[noise] = w[noise].mean()
    return _to_correlation((U * w) @ U.T)


def nonlinear_shrinkage(R, n: int) -> np.ndarray:
    """Analytische nichtlineare Schrumpfung (Ledoit und Wolf 2020, Epanechnikov-Kern mit Bandbreite n^(−1/3), Hilbert-Transformierte). Wirksame Stichprobengröße n − 1 (Daten zentriert),
    bei p > n werden nur die n − 1 positiven Eigenwerte benutzt, die übrigen Eigenwerte bekommen einen gemeinsamen positiven Wert. Das Ergebnis hat die Diagonale 1."""
    R = np.asarray(R, dtype=float)
    p = len(R)
    n_eff = n - 1
    lam, U = np.linalg.eigh(R)
    lam = np.maximum(lam, 0.0)
    k = min(p, n_eff)
    lam_k = lam[p - k:]
    h = n_eff ** (-1.0 / 3.0)
    H = h * lam_k[None, :]                                          # H[i, j] = h λ_j
    x = (lam_k[:, None] - lam_k[None, :]) / H                       # (λ_i − λ_j)/(h λ_j)
    s5 = math.sqrt(5.0)
    f = (3.0 / 4.0 / s5) * np.mean(np.maximum(1.0 - x**2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hil = (-3.0 / 10.0 / math.pi) * x + (3.0 / 4.0 / s5 / math.pi) * (1.0 - x**2 / 5.0) * np.log(np.abs((s5 - x) / (s5 + x)))
    edge = np.abs(x) == s5
    hil[edge] = ((-3.0 / 10.0 / math.pi) * x)[edge]
    hf = np.mean(hil / H, axis=1)
    if p <= n_eff:
        d = lam_k / ((math.pi * (p / n_eff) * lam_k * f) ** 2 + (1.0 - (p / n_eff) - math.pi * (p / n_eff) * lam_k * hf) ** 2)
    else:
        if s5 * h >= 1.0:
            raise ValueError("die nichtlineare Schrumpfung bei p > n − 1 braucht mindestens 13 Wochen")
        hf0 = (1.0 / math.pi) * (3.0 / 10.0 / h**2 + 3.0 / 4.0 / s5 / h * (1.0 - 1.0 / 5.0 / h**2) * math.log((1.0 + s5 * h) / (1.0 - s5 * h))) * np.mean(1.0 / lam_k)
        d0 = 1.0 / (math.pi * (p - n_eff) / n_eff * hf0)
        d1 = lam_k / (math.pi**2 * lam_k**2 * (f**2 + hf**2))
        d = np.concatenate([np.full(p - k, d0), d1])
    return _to_correlation((U * d) @ U.T)


def oracle_shrinkage(R_hat, R_true) -> np.ndarray:
    """Orakel unter allen Schätzern mit den Eigenvektoren der Stichprobenmatrix: Eigenwerte d_i = u_iᵀ R u_i mit der WAHREN Matrix (nur mit bekannter Wahrheit berechenbar), Diagonale 1."""
    w, U = np.linalg.eigh(np.asarray(R_hat, dtype=float))
    d = np.einsum("ji,jk,ki->i", U, np.asarray(R_true, dtype=float), U)
    return _to_correlation((U * d) @ U.T)


def estimate_all(X, R_true=None) -> dict:
    """Alle fünf Schätzer auf einem Datensatz (das Orakel nur mit bekannter Wahrheit); Rückgabe Schlüssel -> Matrix, dazu „a“ = Gewicht der linearen Schrumpfung."""
    n = len(X)
    R = sample_corr(X)
    lw, a = ledoit_wolf(X)
    out = {"stichprobe": R, "linear": lw, "clip": clip_eigenvalues(R, n), "nl": nonlinear_shrinkage(R, n), "a": a}
    if R_true is not None:
        out["orakel"] = oracle_shrinkage(R, R_true)
    return out


def frobenius_error(A, B) -> float:
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B)))


def spectral_error(A, B) -> float:
    return float(np.abs(spectrum(np.asarray(A) - np.asarray(B))).max())


def inverse_error(A, B) -> float:
    """Relative Frobenius-Abweichung der Inversen ‖A⁻¹ − B⁻¹‖/‖B⁻¹‖ (B ist die Wahrheit); unendlich, wenn A nicht invertierbar ist."""
    if matrix_rank(A) < len(A):
        return float("inf")
    Bi = np.linalg.inv(B)
    return float(np.linalg.norm(np.linalg.inv(A) - Bi) / np.linalg.norm(Bi))
