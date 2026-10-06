r"""Contextuality analysis for binary (+1/-1) measurements.

These tools have no fitted parameters.

* :func:`chsh`: CHSH value :math:`S = |E_{11} + E_{12} + E_{21} - E_{22}|` (classical bound 2,
  quantum bound :math:`2\sqrt2`).
* :func:`s_odd`: maximum of :math:`\sum_i \pm v_i` over sign patterns with an odd number of minus signs.
* :func:`cyclic_contextuality`: criterion of Kujala, Dzhafarov and Larsson (2015) for cyclic systems
  of rank n (Contextuality-by-Default). The system is contextual if
  :math:`s_{odd}(\langle R_i R_{i+1}\rangle) - (n - 2) - \Delta > 0`, where :math:`\Delta` sums, over
  contents, :math:`|\langle R_q^c\rangle - \langle R_q^{c'}\rangle|` between the two contexts in
  which each content q is measured (inconsistent connectedness).
* :func:`correlations_from_data`: product expectations and means from raw +-1 observations.
* :func:`qubit_chsh_correlations`: correlations of a maximally entangled qubit pair measured at given
  angles (the quantum prediction used by the CHSH circuit).
"""
from __future__ import annotations

import itertools
import numpy as np


def s_odd(values):
    """Largest signed sum with an odd number of minus signs.

    Parameters
    ----------
    values : array_like
        Numbers :math:`x_1, \\dots, x_n`.

    Returns
    -------
    float
        :math:`\\max \\sum_i s_i x_i` over :math:`s \\in \\{\\pm1\\}^n` with an odd number of :math:`-1`.

    Examples
    --------
    >>> from quantum_mind.families.contextuality import s_odd
    >>> s_odd([1, 1, 1, -1])
    4.0
    """
    v = np.asarray(values, float)
    best = -np.inf
    for signs in itertools.product((1, -1), repeat=len(v)):
        if signs.count(-1) % 2 == 1:
            best = max(best, float(np.dot(signs, v)))
    return best


def chsh(E11, E12, E21, E22):
    """CHSH value.

    Parameters
    ----------
    E11, E12, E21, E22 : float
        Correlations :math:`\\langle A_i B_j \\rangle`.

    Returns
    -------
    float
        :math:`|E_{11} + E_{12} + E_{21} - E_{22}|`. Above 2 no local hidden-variable model fits;
        quantum mechanics allows up to :math:`2\\sqrt2`.
    """
    return abs(E11 + E12 + E21 - E22)


def cyclic_contextuality(product_expectations, means_pairs):
    """Contextuality-by-Default criterion for a cyclic system.

    Parameters
    ----------
    product_expectations : sequence of float
        ``[<R_1 R_2>, <R_2 R_3>, ..., <R_n R_1>]``, one per context.
    means_pairs : sequence of tuple
        For each content q, ``(<R_q> in its first context, <R_q> in its second context)``.

    Returns
    -------
    dict
        ``n``, ``s_odd``, ``delta``, ``measure`` (``s_odd - (n - 2) - delta``) and ``contextual``
        (``measure > 0``).
    """
    n = len(product_expectations)
    so = s_odd(product_expectations)
    delta = float(sum(abs(a - b) for a, b in means_pairs))
    measure = so - (n - 2) - delta
    return {'n': n, 's_odd': so, 'delta': delta, 'measure': measure, 'contextual': measure > 0}


def correlations_from_data(pairs):
    """Expectations from raw +-1 observations.

    Parameters
    ----------
    pairs : dict
        ``{context: array of shape (m, 2)}``, each row the outcomes of the first and second variable.

    Returns
    -------
    dict
        ``{context: (<X Y>, <X>, <Y>)}``.
    """
    out = {}
    for c, arr in pairs.items():
        a = np.asarray(arr, float)
        out[c] = (float(np.mean(a[:, 0] * a[:, 1])), float(np.mean(a[:, 0])), float(np.mean(a[:, 1])))
    return out


def qubit_chsh_correlations(a0=0.0, a1=np.pi / 2, b0=np.pi / 4, b1=-np.pi / 4):
    """Quantum correlations of a Bell pair.

    For :math:`(|00\\rangle + |11\\rangle)/\\sqrt2` measured in the x-z plane at angles ``a`` and ``b``,
    :math:`E(a, b) = \\cos(a - b)`.

    Parameters
    ----------
    a0, a1 : float
        Measurement angles of the first party.
    b0, b1 : float
        Measurement angles of the second party. The defaults give :math:`S = 2\\sqrt2`.

    Returns
    -------
    dict
        ``{'E11', 'E12', 'E21', 'E22'}``.
    """
    E = lambda a, b: float(np.cos(a - b))
    return {'E11': E(a0, b0), 'E12': E(a0, b1), 'E21': E(a1, b0), 'E22': E(a1, b1)}
