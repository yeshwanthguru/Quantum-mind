"""Quantum Fourier transform (QFT) and a period-finding demonstration.

QFT |j> = N^{-1/2} sum_k exp(2 pi i j k / N) |k>, N = 2^n, built from Hadamard, controlled-phase and
swap gates (n(n+1)/2 + n/2 gates). On a state whose amplitudes are periodic with period r (r dividing
N) the QFT concentrates probability on the multiples of N / r, the step at the heart of Shor's
algorithm (Shor, SIAM J. Comput. 26, 1484, 1997; Nielsen and Chuang, ch. 5)."""
from __future__ import annotations

import numpy as np
from .statevector import Circuit

__all__ = ['qft_circuit', 'qft_matrix', 'periodic_state', 'find_period']


def qft_circuit(n, inverse=False, swaps=True):
    """Circuit for the QFT (or its inverse) on n qubits, Qiskit qubit order (qubit 0 = least
    significant bit)."""
    c = Circuit(n); sign = -1 if inverse else 1
    ops = []
    for j in reversed(range(n)):
        ops.append(('h', j))
        for k in reversed(range(j)):
            ops.append(('cp', sign * np.pi / 2 ** (j - k), k, j))
    if inverse:
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
    """The QFT as a matrix: F[k, j] = exp(2 pi i j k / N) / sqrt(N)."""
    N = 2 ** n; j = np.arange(N)
    return np.exp(2j * np.pi * np.outer(j, j) / N) / np.sqrt(N)


def periodic_state(n, r, offset=0):
    """Uniform superposition over the basis states offset, offset + r, offset + 2r, ... (n qubits)."""
    N = 2 ** n; psi = np.zeros(N, complex); psi[offset::r] = 1
    return psi / np.linalg.norm(psi)


def find_period(n, r, offset=0, shots=None, rng=None):
    """Apply the QFT to periodic_state(n, r) and read the peaks. Returns (probabilities, estimated
    period from the smallest non-zero peak, N / peak). With shots, peaks are taken from samples."""
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
