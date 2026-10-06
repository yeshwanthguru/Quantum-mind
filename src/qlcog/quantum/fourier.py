r"""Quantum Fourier transform (QFT) and a period-finding demonstration.

.. math:: \mathrm{QFT}\,|j\rangle = \frac{1}{\sqrt N}\sum_{k=0}^{N-1} e^{2\pi i jk/N}\,|k\rangle,
          \qquad N = 2^n,

built from Hadamard, controlled-phase and swap gates (:math:`n(n+1)/2` gates plus :math:`n/2` swaps).
On a state whose amplitudes are periodic with period r (r dividing N) the QFT concentrates
probability on the multiples of N / r, the step at the heart of Shor's algorithm (Shor, SIAM J.
Comput. 26, 1484, 1997; Nielsen and Chuang, chapter 5).

Examples
--------
>>> import numpy as np
>>> from qlcog.quantum import qft_circuit, qft_matrix, find_period
>>> U = qft_circuit(3).state(None, None, init=np.eye(8)[1])[0]
>>> bool(np.allclose(U, qft_matrix(3)[:, 1]))
True
>>> find_period(6, 8)[1]
8
"""
from __future__ import annotations

import numpy as np
from .statevector import Circuit

__all__ = ['qft_circuit', 'qft_matrix', 'periodic_state', 'find_period']


def qft_circuit(n, inverse=False, swaps=True):
    """Circuit for the QFT or its inverse.

    Parameters
    ----------
    n : int
        Number of qubits (Qiskit order: qubit 0 is the least significant bit).
    inverse : bool, optional
        Build the inverse QFT.
    swaps : bool, optional
        Include the final qubit-reversal swaps.

    Returns
    -------
    Circuit
        Exports to Qiskit with :meth:`Circuit.to_qiskit`.
    """
    c = Circuit(n)
    sign = -1 if inverse else 1
    ops = []
    for j in reversed(range(n)):
        ops.append(('h', j))
        for k in reversed(range(j)):
            ops.append(('cp', sign * np.pi / 2 ** (j - k), k, j))
    if inverse:
        # the inverse runs the same gates backwards, with the swaps first
        ops = ops[::-1]
        if swaps:
            for q in range(n // 2):
                c.swap(q, n - 1 - q)
    for op in ops:
        if op[0] == 'h':
            c.h(op[1])
        else:
            c.cp(op[1], op[2], op[3])
    if swaps and not inverse:
        for q in range(n // 2):
            c.swap(q, n - 1 - q)
    return c


def qft_matrix(n):
    """The QFT as a matrix.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        :math:`F_{kj} = e^{2\\pi i jk/N} / \\sqrt N`, shape ``(N, N)``.
    """
    N = 2 ** n
    j = np.arange(N)
    return np.exp(2j * np.pi * np.outer(j, j) / N) / np.sqrt(N)


def periodic_state(n, r, offset=0):
    """Uniform superposition over a periodic set of basis states.

    Parameters
    ----------
    n : int
        Number of qubits.
    r : int
        Period.
    offset : int, optional
        First basis state.

    Returns
    -------
    numpy.ndarray
        Normalised state over ``offset, offset + r, offset + 2r, ...``.
    """
    N = 2 ** n
    psi = np.zeros(N, complex)
    psi[offset::r] = 1
    return psi / np.linalg.norm(psi)


def find_period(n, r, offset=0, shots=None, rng=None):
    """Recover a period with the QFT.

    Applies the QFT to :func:`periodic_state` and reads the peaks of the output distribution.

    Parameters
    ----------
    n : int
        Number of qubits.
    r : int
        True period (used to prepare the state).
    offset : int, optional
        Offset of the periodic state; it changes only the phases, not the peaks.
    shots : int, optional
        If given, the peaks are read from this many samples instead of the exact probabilities.
    rng : numpy.random.Generator, optional
        Random number generator for the samples.

    Returns
    -------
    probabilities : numpy.ndarray
        Output distribution of the QFT.
    period : int or None
        Estimate ``N / smallest non-zero peak``.
    """
    psi = periodic_state(n, r, offset)
    P = np.abs(qft_circuit(n).state(None, None, init=psi)[0]) ** 2
    if shots:
        rng = np.random.default_rng() if rng is None else rng
        counts = np.bincount(rng.choice(len(P), size=shots, p=P / P.sum()), minlength=len(P))
        peaks = np.nonzero(counts > 0.2 * counts.max())[0]
    else:
        peaks = np.nonzero(P > 1e-9)[0]
    k = peaks[peaks > 0]
    return P, (int(round(2 ** n / k.min())) if len(k) else None)
