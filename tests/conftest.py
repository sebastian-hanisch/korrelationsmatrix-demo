"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def pair_matrix():
    """R = [[1, 0,3], [0,3, 1]]: Eigenwerte 1 ∓ 0,3, also 0,7 und 1,3 (Eigenvektoren (1, ∓1)/√2); Spur 2, Konditionszahl 1,3/0,7 = 13/7."""
    return np.array([[1.0, 0.3], [0.3, 1.0]])


@pytest.fixture
def equi3():
    """Drei Kennzahlen, alle Paare 0,5: Eigenwerte 1 + 2·0,5 = 2 (einfach, Vektor (1, 1, 1)) und 1 − 0,5 = 0,5 (doppelt)."""
    R = np.full((3, 3), 0.5)
    np.fill_diagonal(R, 1.0)
    return R


@pytest.fixture
def four_weeks():
    """Vier Wochen, zwei Kennzahlen: x = (1, 2, 3, 4), y = (2, 1, 4, 3). Zentriert (−1,5; −0,5; 0,5; 1,5) und (−0,5; −1,5; 1,5; 0,5): Σxy = 0,75 + 0,75 + 0,75 + 0,75 = 3,
    Σx² = Σy² = 5, also r = 0,6; die Stichprobenmatrix hat die Eigenwerte 1,6 und 0,4."""
    return np.column_stack([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
