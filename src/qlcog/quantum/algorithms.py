"""Quantum algorithms on the built-in simulator.

* :class:`QAOA`: Quantum Approximate Optimisation Algorithm for any :class:`qlcog.problems.Qubo`
  (allocation, scheduling, MaxCut, portfolios).
* :class:`VQE` with :class:`Hamiltonian`: ground states of Pauli Hamiltonians.
* :func:`grover`: Grover search with an oracle from a list or a predicate.

Each returns a result object with the circuit, which ``to_qiskit`` exports for Aer, IBM Quantum or
Amazon Braket.

Examples
--------
>>> from qlcog.quantum import grover
>>> res = grover(5, marked=[3, 17])
>>> res.iterations, round(res.success, 3)
(3, 0.961)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from .statevector import Circuit, W
from .ansatz import hardware_efficient

__all__ = ['QAOA', 'QAOAResult', 'pauli_matrix', 'Hamiltonian', 'VQE', 'grover', 'GroverResult']


# ------------------------------------------------------------------------------------------- QAOA
@dataclass
class QAOAResult:
    r"""Result of :meth:`QAOA.run`.

    Attributes
    ----------
    x : numpy.ndarray
        Best bit string found among the most probable outcomes.
    energy : float
        Its QUBO energy.
    expectation : float
        :math:`\langle H_C \rangle` of the optimised state (QUBO units).
    probabilities : numpy.ndarray
        Output distribution over all :math:`2^n` bit strings (:math:`x_0` = least significant bit).
    gammas, betas : numpy.ndarray
        Optimised angles, one per layer.
    circuit : Circuit
        The QAOA circuit.
    optimum : float
        Exact minimum over all bit strings (brute force).
    p_optimal : float
        Probability of measuring an optimal bit string.
    weights : numpy.ndarray
        ``(gamma_1, beta_1, ..., gamma_p, beta_p)``, the circuit weights.
    """
    x: np.ndarray                 # best bit string found among the most probable outcomes
    energy: float                 # its QUBO energy
    expectation: float            # <H_C> of the optimised state (QUBO units)
    probabilities: np.ndarray     # over all 2^n bit strings (x_0 = least significant bit)
    gammas: np.ndarray
    betas: np.ndarray
    circuit: Circuit = field(repr=False, default=None)
    optimum: float = None         # exact minimum over all bit strings
    p_optimal: float = 0.0        # probability of measuring an optimal bit string
    weights: np.ndarray = field(repr=False, default=None)   # (gamma_1, beta_1, ..., gamma_p, beta_p)


class QAOA:
    """Quantum Approximate Optimisation Algorithm (Farhi, Goldstone and Gutmann, 2014).

    The cost layer is built from RZ and RZZ gates of the equivalent Ising model, so the simulated and
    exported circuits are identical. Angles are optimised with exact adjoint gradients (L-BFGS-B,
    then a COBYLA polish), starting from a depth-1 grid search extended layer by layer (INTERP) and
    from random restarts.

    Parameters
    ----------
    qubo : qlcog.problems.Qubo
        Problem to minimise (normalised internally).
    p : int, optional
        Number of layers.
    restarts : int, optional
        Random restarts at full depth.
    maxiter : int, optional
        Iterations per optimisation.
    seed : int, optional
        Seed.
    """

    def __init__(self, qubo, p=2, restarts=6, maxiter=300, seed=0):
        self.qubo = qubo.normalised(); self.p = p
        self.restarts, self.maxiter, self.seed = restarts, maxiter, seed
        self.h, self.J, self.c = self.qubo.to_ising()
        self.scale = max(np.abs(self.h).max(initial=0), np.abs(self.J).max(initial=0), 1e-12)
        self.circuit = self._build()
        self.energies = self.qubo.all_energies()

    def _build(self):
        """Build the circuit: Hadamards, then p layers of cost (RZ, RZZ) and mixer (RX) gates."""
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
        """Expected QUBO energy of the QAOA state.

        Parameters
        ----------
        w : array_like
            Angles ``(gamma_1, beta_1, ..., gamma_p, beta_p)``.

        Returns
        -------
        float
        """
        return float(self.circuit.probabilities(w)[0] @ self.energies)

    def _value_and_grad(self, w):
        """Expected energy and its adjoint gradient (phi = E * psi)."""
        E = self.energies
        return self.circuit.value_and_grad(w, None, lambda psi, rows: (((np.abs(psi) ** 2) @ E).sum(), psi * E[None, :]))

    def _optimise(self, w0, depth):
        """Optimise the first `depth` layers (exact adjoint gradients, L-BFGS-B); deeper layers stay at
        gamma = beta = 0 (identity)."""
        def fg(v):
            w = np.zeros(2 * self.p); w[:2 * depth] = v
            val, g = self._value_and_grad(w)
            return float(val), g[:2 * depth]
        r = minimize(fg, np.asarray(w0, float), jac=True, method='L-BFGS-B', options={'maxiter': self.maxiter})
        val, v = float(r.fun), r.x
        # derivative-free polish: escapes some flat regions L-BFGS stops in (16/16 vs 14/16 optima found on
        # random 8-variable portfolio and MaxCut instances)
        r2 = minimize(lambda u: fg(u)[0], v, method='COBYLA', options={'maxiter': self.maxiter})
        if r2.fun < val:
            val, v = float(r2.fun), r2.x
        w = np.zeros(2 * self.p); w[:2 * depth] = v
        return val, w

    def run(self):
        """Optimise the angles and sample the result.

        The schedule search has two parts: (1) a grid search at depth 1, then layer-by-layer
        interpolation of the optimal angles to the next depth (INTERP; Zhou et al., PRX 10, 021067,
        2020); (2) random restarts at full depth. The best schedule over both is kept.

        Returns
        -------
        QAOAResult
        """
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
        opt = float(self.energies.min())
        return QAOAResult(x=np.array([(k >> i) & 1 for i in range(self.qubo.n)]), energy=float(self.energies[k]),
                          expectation=float(P @ self.energies), probabilities=P, gammas=w[0::2], betas=w[1::2],
                          circuit=self.circuit, optimum=opt, p_optimal=float(P[np.isclose(self.energies, opt)].sum()),
                          weights=w)

    def to_qiskit(self, result, measure=True):
        """Qiskit circuit with optimised angles bound.

        Parameters
        ----------
        result : QAOAResult
            Output of :meth:`run`.
        measure : bool, optional
            Add measurements.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(result.weights, None, measure)


# ------------------------------------------------------------------------------------------- VQE
_PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}


def pauli_matrix(label):
    """Matrix of a Pauli string.

    Parameters
    ----------
    label : str
        Letters from ``IXYZ`` in Qiskit order (the leftmost letter acts on the highest qubit).

    Returns
    -------
    numpy.ndarray
    """
    M = np.array([[1.0 + 0j]])
    for ch in label:
        M = np.kron(M, _PAULI[ch])
    return M


class Hamiltonian:
    """Sum of weighted Pauli strings.

    Parameters
    ----------
    terms : list of tuple
        ``(coefficient, label)`` pairs, for example ``[(1.0, 'ZZ'), (0.5, 'XI')]``.

    Attributes
    ----------
    n : int
        Number of qubits.
    matrix : numpy.ndarray
        Dense matrix of the Hamiltonian.
    """

    def __init__(self, terms):
        self.terms = [(float(c), s.upper()) for c, s in terms]
        self.n = len(self.terms[0][1])
        self.matrix = sum(c * pauli_matrix(s) for c, s in self.terms)

    @classmethod
    def from_qubo(cls, qubo):
        """Ising Hamiltonian (Z and ZZ terms) with the same energies as a QUBO.

        Parameters
        ----------
        qubo : qlcog.problems.Qubo

        Returns
        -------
        Hamiltonian
        """
        h, J, c = qubo.to_ising(); n = qubo.n; terms = [(c, 'I' * n)]
        for i in range(n):
            if h[i]:
                terms.append((h[i], ''.join('Z' if q == i else 'I' for q in reversed(range(n)))))
        for i, j in zip(*np.nonzero(np.triu(J, 1))):
            terms.append((J[i, j], ''.join('Z' if q in (i, j) else 'I' for q in reversed(range(n)))))
        return cls(terms)

    def expectation(self, psi):
        r"""Energy expectation of a batch of states.

        Parameters
        ----------
        psi : numpy.ndarray
            States, shape ``(B, 2**n)``.

        Returns
        -------
        numpy.ndarray
            :math:`\langle\psi|H|\psi\rangle`, shape ``(B,)``.
        """
        return np.real(np.einsum('bi,ij,bj->b', psi.conj(), self.matrix, psi))

    def ground_energy(self):
        """Exact ground-state energy (dense diagonalisation).

        Returns
        -------
        float
        """
        return float(np.linalg.eigvalsh(self.matrix)[0])


class VQE:
    """Variational Quantum Eigensolver (Peruzzo et al., Nature Communications 5, 4213, 2014).

    A hardware-efficient ansatz is optimised with exact adjoint gradients and L-BFGS from several random
    starts.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Operator whose ground state is sought.
    layers : int, optional
        Ansatz layers.
    restarts : int, optional
        Random starts.
    maxiter : int, optional
        Iterations per start.
    seed : int, optional
        Seed.
    rotations : tuple of str, optional
        Rotation gates; ``('ry',)`` gives a real ansatz, enough for Hamiltonians with real ground states
        (for example transverse-field Ising models).
    entangle : {'linear', 'ring'}, optional
        Entangler topology.
    entangler : {'cx', 'cz'}, optional
        Two-qubit gate.
    """

    def __init__(self, hamiltonian, layers=2, restarts=4, maxiter=400, seed=0, rotations=('ry', 'rz'),
                 entangle='linear', entangler='cx'):
        self.H = hamiltonian; self.layers = layers
        self.restarts, self.maxiter, self.seed = restarts, maxiter, seed
        self.circuit, _ = hardware_efficient(self.H.n, layers, rotations=rotations, entangle=entangle,
                                             entangler=entangler)
        for q in range(self.H.n):
            self.circuit.ry(W(self.circuit.n_weights), q)

    def energy(self, w):
        """Energy of the ansatz state.

        Parameters
        ----------
        w : array_like
            Weights.

        Returns
        -------
        float
        """
        return float(self.H.expectation(self.circuit.state(w))[0])

    def _energy_and_grad(self, w):
        """Energy and its adjoint gradient (phi = H psi)."""
        M = self.H.matrix                                           # adjoint gradient: phi = H psi
        return self.circuit.value_and_grad(w, None, lambda psi, rows: (self.H.expectation(psi).sum(), psi @ M.T))

    def run(self):
        """Optimise from several random starts.

        Returns
        -------
        dict
            ``energy`` (best found), ``exact`` (ground energy), ``weights`` and ``state``.
        """
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
        """Qiskit circuit of the optimised ansatz.

        Parameters
        ----------
        measure : bool, optional
            Add measurements.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(self.weights_, None, measure)


# ------------------------------------------------------------------------------------------- Grover
@dataclass
class GroverResult:
    """Result of :func:`grover`.

    Attributes
    ----------
    probabilities : numpy.ndarray
        Output distribution.
    iterations : int
        Number of Grover iterations.
    marked : list of int
        Marked basis states.
    success : float
        Probability of measuring a marked state.
    circuit : Circuit
        Simulated circuit (diagonal phase layers).
    n : int
        Number of qubits.
    """
    probabilities: np.ndarray
    iterations: int
    marked: list
    success: float
    circuit: Circuit = field(repr=False, default=None)
    n: int = 0

    def to_qiskit(self, measure=True, style='gates'):
        """Qiskit circuit of the search.

        Parameters
        ----------
        measure : bool, optional
            Add measurements.
        style : {'gates', 'diagonal'}, optional
            ``'gates'`` (default): textbook oracle and diffuser from X and multi-controlled Z gates, one
            multi-controlled Z per marked state; this form compiles sensibly for hardware. ``'diagonal'``:
            the phase layers as ``DiagonalGate`` (exact and compact to write, but its compiled size grows
            exponentially with the number of qubits).

        Returns
        -------
        qiskit.QuantumCircuit
        """
        if style == 'diagonal':
            return self.circuit.to_qiskit(None, None, measure)
        from .._optional import require
        require('qiskit')
        from qiskit import QuantumCircuit
        n = self.n; qc = QuantumCircuit(n)

        def mcz():
            if n == 1:
                qc.z(0); return
            qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)

        qc.h(range(n))
        for _ in range(self.iterations):
            for k in self.marked:                                   # phase flip of |k>
                zeros = [q for q in range(n) if not (k >> q) & 1]
                if zeros:
                    qc.x(zeros)
                mcz()
                if zeros:
                    qc.x(zeros)
            qc.h(range(n)); qc.x(range(n)); mcz(); qc.x(range(n)); qc.h(range(n))   # diffuser (up to phase)
        if measure:
            qc.measure_all()
        return qc


def grover(n, marked, iterations=None):
    r"""Grover search.

    Parameters
    ----------
    n : int
        Number of qubits (1 to 16).
    marked : list of int or callable
        Marked basis indices, or a predicate on the bit tuple :math:`(x_0, \dots, x_{n-1})`.
    iterations : int, optional
        Number of iterations (default :math:`\lfloor \frac{\pi}{4}\sqrt{N/M} \rfloor`).

    Returns
    -------
    GroverResult
        Simulated with diagonal phase layers; ``to_qiskit()`` exports a gate-level circuit.

    Raises
    ------
    ValueError
        If n is out of range or no (or every) state is marked.
    """
    if not 1 <= n <= 16:
        raise ValueError('grover supports 1 to 16 qubits (the marked set is enumerated), got %d' % n)
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
    return GroverResult(P, it, marked, float(P[marked].sum()), c, n)
