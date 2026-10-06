r"""Quantum amplitude estimation for expected values (risk, pricing), with classical Monte Carlo as the
baseline.

A distribution :math:`p` over :math:`2^n` values and a payoff :math:`f` with values in [0, 1] define
the operator

.. math:: A|0\rangle|0\rangle = \sum_i \sqrt{p_i}\,|i\rangle\bigl(\sqrt{1 - f_i}\,|0\rangle + \sqrt{f_i}\,|1\rangle\bigr),

so that the probability of the ancilla reading 1 is :math:`a = \sum_i p_i f_i = E[f]` (Woerner and
Egger, npj Quantum Information 5, 15, 2019). With the Grover operator
:math:`Q = A S_0 A^\dagger S_\chi`, measuring :math:`Q^m A` gives 1 with probability
:math:`\sin^2((2m + 1)\theta)`, :math:`\sin^2\theta = a`.

Maximum-likelihood amplitude estimation (Suzuki et al., Quantum Information Processing 19, 75, 2020)
combines measurements at :math:`m = 0, 1, 2, 4, \dots` and needs no phase-estimation register. Its
error falls roughly as 1 / (number of calls to A), against 1 / sqrt(number of samples) for classical
Monte Carlo.

Here the circuits are simulated exactly (dense matrices, n + 1 <= 12 qubits) and the measurements are
sampled; :meth:`AmplitudeEstimation.to_qiskit` exports the circuit for one power m. No speed-up is
claimed for simulated runs: the comparison counts calls to A, which is what matters on hardware.

Examples
--------
>>> import numpy as np
>>> from qlcog.quantum import AmplitudeEstimation
>>> ae = AmplitudeEstimation(np.ones(8) / 8, np.linspace(0, 1, 8))
>>> round(ae.exact, 3)
0.5
>>> abs(ae.run(rng=np.random.default_rng(0)).estimate - ae.exact) < 0.01
True
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ['AmplitudeEstimation', 'AEResult', 'monte_carlo_estimate']


@dataclass
class AEResult:
    r"""Result of :meth:`AmplitudeEstimation.run`.

    Attributes
    ----------
    estimate : float
        Maximum-likelihood estimate of :math:`a = E[f]`.
    exact : float
        True value :math:`\sum_i p_i f_i`.
    oracle_calls : int
        Total applications of A (each :math:`Q^m A` costs 2m + 1).
    powers : list of int
        Grover powers m used.
    hits : list of int
        Number of measured ones per power.
    shots : int
        Shots per power.
    """
    estimate: float          # estimate of a = E[f]
    exact: float             # true value sum_i p_i f_i
    oracle_calls: int        # total applications of A (each Q^m A costs 2m + 1)
    powers: list             # the m used
    hits: list               # measured ones per power
    shots: int


class AmplitudeEstimation:
    r"""Maximum-likelihood amplitude estimation of an expected payoff.

    Parameters
    ----------
    probabilities : array_like
        Distribution over :math:`2^n` points (n <= 11); normalised on construction.
    payoff : array_like
        Payoff in [0, 1] at the same points.

    Attributes
    ----------
    exact : float
        The exact expectation (for comparison).
    theta : float
        :math:`\arcsin\sqrt{a}`.
    A : numpy.ndarray
        Dense state-preparation unitary on n + 1 qubits.

    Raises
    ------
    ValueError
        If the sizes are wrong or the payoff is outside [0, 1].
    """

    def __init__(self, probabilities, payoff):
        p = np.asarray(probabilities, float); f = np.asarray(payoff, float)
        n = int(round(np.log2(len(p))))
        if 2 ** n != len(p) or len(f) != len(p) or n > 11:
            raise ValueError('need 2^n probabilities (n <= 11) and one payoff per point')
        if np.any(p < 0) or np.any(f < 0) or np.any(f > 1):
            raise ValueError('probabilities must be >= 0 and the payoff must lie in [0, 1]')
        self.p, self.f, self.n = p / p.sum(), f, n
        self.exact = float(self.p @ self.f)
        self.theta = float(np.arcsin(np.sqrt(self.exact)))
        self.A = self._operator_A()

    def _operator_A(self):
        """Dense unitary A on n + 1 qubits (ancilla = qubit n, the most significant).

        A Householder reflection maps ``|0>`` to sqrt(p) on the n data qubits; a controlled RY then loads
        the payoff into the ancilla amplitude.
        """
        N = 2 ** self.n
        # state preparation: Householder reflection mapping |0> to sqrt(p)
        v = np.sqrt(self.p); e0 = np.zeros(N); e0[0] = 1
        w = e0 - v
        P = np.eye(N) - 2 * np.outer(w, w) / (w @ w) if np.linalg.norm(w) > 1e-12 else np.eye(N)
        A = np.zeros((2 * N, 2 * N))
        for i in range(N):                                     # controlled RY on the ancilla
            c, s = np.sqrt(1 - self.f[i]), np.sqrt(self.f[i])
            A[i, i], A[i, N + i], A[N + i, i], A[N + i, N + i] = c, -s, s, c
        return A @ np.kron(np.eye(2), P)

    def good_probability(self, m):
        r"""Probability of measuring the ancilla in :math:`|1\rangle` after :math:`Q^m A`.

        Parameters
        ----------
        m : int
            Grover power.

        Returns
        -------
        float
            Computed by applying the operators (equals :math:`\sin^2((2m + 1)\theta)`).
        """
        N = 2 ** self.n; A = self.A
        S_chi = np.diag(np.r_[np.ones(N), -np.ones(N)])       # flips the good (ancilla = 1) states
        S0 = np.eye(2 * N); S0[0, 0] = -1
        Q = -A @ S0 @ A.T @ S_chi
        psi = A[:, 0]
        for _ in range(m):
            psi = Q @ psi
        return float(np.sum(psi[N:] ** 2))

    def run(self, powers=(0, 1, 2, 4, 8, 16), shots=100, rng=None):
        """Sample measurements at several powers and return the maximum-likelihood estimate.

        Parameters
        ----------
        powers : sequence of int, optional
            Grover powers m.
        shots : int, optional
            Measurements per power.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        AEResult
        """
        rng = np.random.default_rng() if rng is None else rng
        hits = [int(rng.binomial(shots, self.good_probability(m))) for m in powers]
        th = np.linspace(1e-6, np.pi / 2 - 1e-6, 20001)
        ll = np.zeros_like(th)
        for m, h in zip(powers, hits):
            q = np.clip(np.sin((2 * m + 1) * th) ** 2, 1e-15, 1 - 1e-15)
            ll += h * np.log(q) + (shots - h) * np.log(1 - q)
        est = float(np.sin(th[np.argmax(ll)]) ** 2)
        return AEResult(est, self.exact, int(sum((2 * m + 1) * shots for m in powers)), list(powers), hits, shots)

    def to_qiskit(self, m=0, measure=True):
        """Qiskit circuit for :math:`Q^m A`.

        Parameters
        ----------
        m : int, optional
            Grover power.
        measure : bool, optional
            Measure the ancilla.

        Returns
        -------
        qiskit.QuantumCircuit
            n + 1 qubits with one dense ``UnitaryGate``; the ancilla is the last qubit.
        """
        from .._optional import require
        require('qiskit')
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        N = 2 ** self.n; A = self.A
        S_chi = np.diag(np.r_[np.ones(N), -np.ones(N)]); S0 = np.eye(2 * N); S0[0, 0] = -1
        U = np.linalg.matrix_power(-A @ S0 @ A.T @ S_chi, m) @ A
        qc = QuantumCircuit(self.n + 1, 1)
        qc.append(UnitaryGate(U, label='Q^%d A' % m), range(self.n + 1))
        if measure:
            qc.measure(self.n, 0)
        return qc


def monte_carlo_estimate(probabilities, payoff, samples, rng=None):
    """Classical Monte Carlo baseline.

    Samples points from p and averages a Bernoulli(f) draw at each, the same information per call as
    one run of A.

    Parameters
    ----------
    probabilities : array_like
        Distribution.
    payoff : array_like
        Payoff in [0, 1].
    samples : int
        Number of samples.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Estimate of :math:`E[f]`.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(probabilities, float); p = p / p.sum()
    idx = rng.choice(len(p), size=samples, p=p)
    return float(np.mean(rng.random(samples) < np.asarray(payoff)[idx]))
