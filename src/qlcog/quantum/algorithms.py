"""Quantum algorithms on the built-in simulator: QAOA for QUBO problems, VQE for Pauli Hamiltonians
and Grover search. Each returns a result object with the circuit, which `to_qiskit` exports for Aer,
IBM Quantum or Amazon Braket."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from .statevector import Circuit, W
from .ansatz import hardware_efficient, shifted_weights

__all__ = ['QAOA', 'QAOAResult', 'pauli_matrix', 'Hamiltonian', 'VQE', 'grover', 'GroverResult']


# ------------------------------------------------------------------------------------------- QAOA
@dataclass
class QAOAResult:
    x: np.ndarray                 # best bit string found among the most probable outcomes
    energy: float                 # its QUBO energy
    expectation: float            # <H_C> of the optimised state (QUBO units)
    probabilities: np.ndarray     # over all 2^n bit strings (x_0 = least significant bit)
    gammas: np.ndarray
    betas: np.ndarray
    circuit: Circuit = field(repr=False, default=None)
    optimum: float = None         # exact minimum (brute force) when n is small

    @property
    def p_optimal(self):
        """Probability of measuring an optimal bit string."""
        return float(self._p_opt)


class QAOA:
    """Quantum Approximate Optimisation Algorithm (Farhi, Goldstone and Gutmann, 2014) for a
    `qlcog.problems.Qubo`. The cost layer is built from RZ and RZZ gates of the equivalent Ising model,
    so the simulated and exported circuits are identical.

    Parameters: p (layers), restarts, maxiter, seed."""

    def __init__(self, qubo, p=2, restarts=6, maxiter=300, seed=0):
        self.qubo = qubo.normalised(); self.p = p
        self.restarts, self.maxiter, self.seed = restarts, maxiter, seed
        self.h, self.J, self.c = self.qubo.to_ising()
        self.scale = max(np.abs(self.h).max(initial=0), np.abs(self.J).max(initial=0), 1e-12)
        self.circuit = self._build()
        self.energies = self.qubo.all_energies()

    def _build(self):
        n = self.qubo.n; c = Circuit(n); s = 1 / self.scale
        for q in range(n):
            c.h(q)
        for layer in range(self.p):
            g, b = 2 * layer, 2 * layer + 1                  # weights: gamma_l, beta_l
            for i in range(n):
                if self.h[i]:
                    c.rz(W(g, 2 * self.h[i] * s), i)
            for i, j in zip(*np.nonzero(self.J)):
                c.rzz(W(g, 2 * self.J[i, j] * s), int(i), int(j))
            for q in range(n):
                c.rx(W(b, 2.0), q)
        return c

    def expectation(self, w):
        return float(self.circuit.probabilities(w)[0] @ self.energies)

    def _optimise(self, w0, depth):
        """Optimise the first `depth` layers; deeper layers stay at gamma = beta = 0 (identity)."""
        pad = np.zeros(2 * self.p)

        def f(v):
            pad[:2 * depth] = v; return self.expectation(pad)
        r = minimize(f, np.asarray(w0, float), method='L-BFGS-B', options={'maxiter': self.maxiter})
        r = minimize(f, r.x, method='COBYLA', options={'maxiter': self.maxiter})
        w = np.zeros(2 * self.p); w[:2 * depth] = r.x
        return float(r.fun), w

    def run(self):
        """Schedule search: (1) grid search at depth 1, then layer-by-layer interpolation of the
        optimal angles to the next depth (INTERP, Zhou et al., PRX 10, 021067, 2020); (2) random restarts
        at full depth. The best schedule over both is kept."""
        rng = np.random.default_rng(self.seed); cands = []
        G, B = np.meshgrid(np.linspace(0, np.pi, 25)[1:], np.linspace(0, np.pi / 2, 13)[1:])
        grid = [(self.expectation(np.r_[[g, b], np.zeros(2 * self.p - 2)]), g, b) for g, b in zip(G.ravel(), B.ravel())]
        _, g, b = min(grid)
        val, w = self._optimise([g, b], 1)
        for d in range(2, self.p + 1):                       # INTERP initialisation of depth d from d - 1
            gp, bp = np.r_[0, w[0:2 * (d - 1):2], 0], np.r_[0, w[1:2 * (d - 1):2], 0]
            i = np.arange(1, d + 1)
            g0 = (i - 1) / (d - 1) * gp[i - 1] + (d - i) / (d - 1) * gp[i]
            b0 = (i - 1) / (d - 1) * bp[i - 1] + (d - i) / (d - 1) * bp[i]
            val, w = self._optimise(np.column_stack([g0, b0]).ravel(), d)
        cands.append((val, w))
        for _ in range(self.restarts):
            w0 = np.column_stack([rng.uniform(0, np.pi, self.p), rng.uniform(0, np.pi / 2, self.p)]).ravel()
            cands.append(self._optimise(w0, self.p))
        w = min(cands, key=lambda c: c[0])[1]; P = self.circuit.probabilities(w)[0]
        top = np.argsort(P)[::-1][:max(8, self.qubo.n)]
        k = int(top[np.argmin(self.energies[top])])
        res = QAOAResult(x=np.array([(k >> i) & 1 for i in range(self.qubo.n)]), energy=float(self.energies[k]),
                         expectation=float(P @ self.energies), probabilities=P, gammas=w[0::2], betas=w[1::2],
                         circuit=self.circuit, optimum=float(self.energies.min()))
        res._p_opt = P[np.isclose(self.energies, self.energies.min())].sum()
        res.weights = w
        return res

    def to_qiskit(self, result, measure=True):
        return self.circuit.to_qiskit(result.weights, None, measure)


# ------------------------------------------------------------------------------------------- VQE
_PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}


def pauli_matrix(label):
    """Matrix of a Pauli string in Qiskit order (the leftmost letter acts on the highest qubit)."""
    M = np.array([[1.0 + 0j]])
    for ch in label:
        M = np.kron(M, _PAULI[ch])
    return M


class Hamiltonian:
    """Sum of weighted Pauli strings, e.g. Hamiltonian([(1.0, 'ZZ'), (0.5, 'XI')])."""

    def __init__(self, terms):
        self.terms = [(float(c), s.upper()) for c, s in terms]
        self.n = len(self.terms[0][1])
        self.matrix = sum(c * pauli_matrix(s) for c, s in self.terms)

    @classmethod
    def from_qubo(cls, qubo):
        h, J, c = qubo.to_ising(); n = qubo.n; terms = [(c, 'I' * n)]
        for i in range(n):
            if h[i]:
                terms.append((h[i], ''.join('Z' if q == i else 'I' for q in reversed(range(n)))))
        for i, j in zip(*np.nonzero(np.triu(J, 1))):
            terms.append((J[i, j], ''.join('Z' if q in (i, j) else 'I' for q in reversed(range(n)))))
        return cls(terms)

    def expectation(self, psi):
        """<psi|H|psi> for each state of a batch (B, 2^n) -> (B,)."""
        return np.real(np.einsum('bi,ij,bj->b', psi.conj(), self.matrix, psi))

    def ground_energy(self):
        return float(np.linalg.eigvalsh(self.matrix)[0])


class VQE:
    """Variational Quantum Eigensolver (Peruzzo et al., Nature Communications 5, 4213, 2014) with a
    hardware-efficient ansatz and parameter-shift gradients. rotations=('ry',) gives a real ansatz,
    enough for Hamiltonians with real ground states (e.g. transverse-field Ising models)."""

    def __init__(self, hamiltonian, layers=2, restarts=4, maxiter=400, seed=0, rotations=('ry', 'rz'),
                 entangle='linear', entangler='cx'):
        self.H = hamiltonian; self.layers = layers
        self.restarts, self.maxiter, self.seed = restarts, maxiter, seed
        self.circuit, _ = hardware_efficient(self.H.n, layers, rotations=rotations, entangle=entangle,
                                             entangler=entangler)
        for q in range(self.H.n):
            self.circuit.ry(W(self.circuit.n_weights), q)

    def energy(self, w):
        return float(self.H.expectation(self.circuit.state(w))[0])

    def _energy_and_grad(self, w):
        Ws = np.vstack([w[None, :], shifted_weights(w)])            # one batch: w and all 2K shifts
        E = self.H.expectation(self.circuit.state(Ws))
        return float(E[0]), (E[1::2] - E[2::2]) / 2

    def run(self):
        rng = np.random.default_rng(self.seed); best = None
        fg = self._energy_and_grad
        for _ in range(self.restarts):
            r = minimize(fg, rng.uniform(0, 2 * np.pi, self.circuit.n_weights), jac=True, method='L-BFGS-B',
                         options={'maxiter': self.maxiter})
            if best is None or r.fun < best.fun:
                best = r
        self.weights_ = best.x
        return {'energy': float(best.fun), 'exact': self.H.ground_energy(), 'weights': best.x,
                'state': self.circuit.state(best.x)[0]}

    def to_qiskit(self, measure=True):
        return self.circuit.to_qiskit(self.weights_, None, measure)


# ------------------------------------------------------------------------------------------- Grover
@dataclass
class GroverResult:
    probabilities: np.ndarray
    iterations: int
    marked: list
    success: float
    circuit: Circuit = field(repr=False, default=None)

    def to_qiskit(self, measure=True):
        return self.circuit.to_qiskit(None, None, measure)


def grover(n, marked, iterations=None):
    """Grover search over n qubits. marked: list of basis indices, or a predicate on the bit tuple
    (x_0, ..., x_{n-1}). The oracle and diffuser are diagonal phase layers (exported as DiagonalGate)."""
    N = 2 ** n
    if callable(marked):
        marked = [k for k in range(N) if marked(tuple((k >> i) & 1 for i in range(n)))]
    marked = sorted(set(int(k) for k in marked)); M = len(marked)
    if not 0 < M < N:
        raise ValueError('need at least one marked and one unmarked state')
    it = int(np.floor(np.pi / 4 * np.sqrt(N / M))) if iterations is None else iterations
    oracle = np.zeros(N); oracle[marked] = np.pi
    reflect = np.full(N, np.pi); reflect[0] = 0.0
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for _ in range(it):
        c.diagonal(oracle, 'oracle')
        for q in range(n):
            c.h(q)
        c.diagonal(reflect, 'reflect')
        for q in range(n):
            c.h(q)
    P = c.probabilities()[0]
    return GroverResult(P, it, marked, float(P[marked].sum()), c)
