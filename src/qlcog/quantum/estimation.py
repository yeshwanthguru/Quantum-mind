"""Quantum amplitude estimation for expected values (risk, pricing), with classical Monte Carlo as the
baseline.

A distribution p over 2^n values and a payoff f with values in [0, 1] define the operator
A |0>|0> = sum_i sqrt(p_i) |i> (sqrt(1 - f_i) |0> + sqrt(f_i) |1>), so that the probability of the
ancilla reading 1 is a = sum_i p_i f_i = E[f] (Woerner and Egger, npj Quantum Information 5, 15, 2019).
With the Grover operator Q = A S_0 A^dagger S_chi, measuring Q^m A gives 1 with probability
sin^2((2m + 1) theta), sin^2 theta = a. Maximum-likelihood amplitude estimation (Suzuki et al., Quantum
Information Processing 19, 75, 2020) combines measurements at m = 0, 1, 2, 4, ... and needs no
phase-estimation register; its error falls roughly as 1 / (number of A calls), against
1 / sqrt(number of samples) for classical Monte Carlo.

Here the circuits are simulated exactly (dense matrices, n + 1 <= 12 qubits) and the measurements are
sampled. `to_qiskit(m)` exports the circuit for one power m."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ['AmplitudeEstimation', 'AEResult', 'monte_carlo_estimate']


@dataclass
class AEResult:
    estimate: float          # estimate of a = E[f]
    exact: float             # true value sum_i p_i f_i
    oracle_calls: int        # total applications of A (each Q^m A costs 2m + 1)
    powers: list             # the m used
    hits: list               # measured ones per power
    shots: int


class AmplitudeEstimation:
    """probabilities: p over 2^n points (n <= 11); payoff: f in [0, 1] at the same points."""

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
        """Dense unitary on n + 1 qubits (ancilla = qubit n, the most significant)."""
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
        """Exact probability of measuring the ancilla in |1> after Q^m A (simulated with the circuit)."""
        N = 2 ** self.n; A = self.A
        S_chi = np.diag(np.r_[np.ones(N), -np.ones(N)])       # flips the good (ancilla = 1) states
        S0 = np.eye(2 * N); S0[0, 0] = -1
        Q = -A @ S0 @ A.T @ S_chi
        psi = A[:, 0]
        for _ in range(m):
            psi = Q @ psi
        return float(np.sum(psi[N:] ** 2))

    def run(self, powers=(0, 1, 2, 4, 8, 16), shots=100, rng=None):
        """Sample `shots` measurements at each power and return the maximum-likelihood estimate."""
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
        """Qiskit circuit for Q^m A on n + 1 qubits (dense UnitaryGate; ancilla = last qubit)."""
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
    """Classical baseline: sample `samples` points from p and average a Bernoulli(f) draw (the same
    information per call as one run of A)."""
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(probabilities, float); p = p / p.sum()
    idx = rng.choice(len(p), size=samples, p=p)
    return float(np.mean(rng.random(samples) < np.asarray(payoff)[idx]))
