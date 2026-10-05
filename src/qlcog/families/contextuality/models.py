"""Contextuality analysis for binary (+1/-1) measurements.

Tools (no fitted parameters):
  chsh(E)                      CHSH value S = |E11 + E12 + E21 - E22| from the four correlations
                               (classical bound 2; quantum bound 2 sqrt 2).
  s_odd(values)                maximum of sum(+-v_i) over sign patterns with an odd number of minus signs.
  cyclic_contextuality(...)    criterion of Kujala, Dzhafarov and Larsson (2015) for cyclic systems of
                               rank n (Contextuality-by-Default): the system is contextual if
                               s_odd(<R_i R_{i+1}>) - (n - 2) - Delta > 0, where Delta is the sum, over
                               contents, of |<R_q^c> - <R_q^c'>| between the two contexts in which each
                               content q is measured (inconsistent connectedness).
  correlations_from_data(...)  product expectations and means from raw +-1 observations.
  qubit_chsh_correlations(...) correlations of a maximally entangled qubit pair measured at given
                               angles (the quantum prediction used by the CHSH circuit)."""
from __future__ import annotations

import itertools
import numpy as np


def s_odd(values):
    v = np.asarray(values, float); best = -np.inf
    for signs in itertools.product((1, -1), repeat=len(v)):
        if signs.count(-1) % 2 == 1:
            best = max(best, float(np.dot(signs, v)))
    return best


def chsh(E11, E12, E21, E22):
    return abs(E11 + E12 + E21 - E22)


def cyclic_contextuality(product_expectations, means_pairs):
    """product_expectations: [<R_1 R_2>_c1, <R_2 R_3>_c2, ..., <R_n R_1>_cn] (one per context);
    means_pairs: for each content q, the pair (<R_q> in its first context, <R_q> in its second context).
    Returns dict with s_odd, Delta, n, the contextuality measure (positive means contextual)."""
    n = len(product_expectations)
    so = s_odd(product_expectations)
    delta = float(sum(abs(a - b) for a, b in means_pairs))
    measure = so - (n - 2) - delta
    return {'n': n, 's_odd': so, 'delta': delta, 'measure': measure, 'contextual': measure > 0}


def correlations_from_data(pairs):
    """pairs: dict context -> array of shape (m, 2) of +-1 outcomes (first, second variable).
    Returns {context: (<X Y>, <X>, <Y>)}."""
    out = {}
    for c, arr in pairs.items():
        a = np.asarray(arr, float)
        out[c] = (float(np.mean(a[:, 0] * a[:, 1])), float(np.mean(a[:, 0])), float(np.mean(a[:, 1])))
    return out


def qubit_chsh_correlations(a0=0.0, a1=np.pi / 2, b0=np.pi / 4, b1=-np.pi / 4):
    """E(a, b) = cos(a - b) for the Bell state (|00> + |11>)/sqrt 2 measured in the x-z plane at
    angles a and b; the defaults give S = 2 sqrt 2."""
    E = lambda a, b: float(np.cos(a - b))
    return {'E11': E(a0, b0), 'E12': E(a0, b1), 'E21': E(a1, b0), 'E22': E(a1, b1)}
