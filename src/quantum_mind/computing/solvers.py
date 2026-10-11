"""Solvers for robot QUBO problems: classical baselines and a thin Qiskit QAOA adapter.

The classical solvers (:func:`exhaustive`, :func:`simulated_annealing`, :func:`greedy_descent`) are the baselines a
quantum result must be reported against. :func:`qaoa` does not re-implement QAOA: it hands the problem's Ising
Hamiltonian to Qiskit's own ``QAOAAnsatz`` and simulates the circuit exactly, so the algorithm is Qiskit's and
the only thing added here is the connection to the robot problems and the bookkeeping (energy, probability of the
optimum, best sampled solution).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .._optional import require

__all__ = ['SolverResult', 'exhaustive', 'simulated_annealing', 'greedy_descent', 'random_sampling', 'qaoa']


@dataclass
class SolverResult:
    """What a solver returns.

    Attributes
    ----------
    x : numpy.ndarray
        Best bit string found.
    energy : float
        QUBO energy of ``x``.
    method : str
    info : dict
        Method-specific details (for QAOA: ``expected_energy``, ``p_optimal``, ``depth``).
    """
    x: np.ndarray
    energy: float
    method: str
    info: dict = field(default_factory=dict)


def _coupling(qubo):
    """Upper-triangular linear terms and the symmetric off-diagonal coupling used for single-bit flips."""
    U = qubo.normalised().Q
    lin = np.diag(U).copy()
    off = U - np.diag(lin)
    return lin, off + off.T


def exhaustive(qubo):
    """Exact optimum by enumeration (at most 22 variables).

    Returns
    -------
    SolverResult
    """
    x, e = qubo.brute_force()
    return SolverResult(x, e, 'exhaustive')


def simulated_annealing(qubo, sweeps=1000, restarts=10, t_start=None, t_end=None, seed=0):
    """Single-bit-flip simulated annealing with a geometric schedule (classical baseline).

    Single flips cannot move between two feasible assignments without crossing the penalty barrier of the encoding,
    so on penalty QUBOs this solver is often weaker than :func:`greedy_descent` with many restarts; report both.

    Parameters
    ----------
    qubo : Qubo
    sweeps : int, optional
        Sweeps over all variables per restart.
    restarts : int, optional
        Independent restarts; the best result is returned.
    t_start, t_end : float, optional
        Start and end temperature (defaults from the largest energy change of one flip).
    seed : int, optional

    Returns
    -------
    SolverResult
    """
    rng = np.random.default_rng(seed)
    lin, sym = _coupling(qubo)
    n = qubo.n
    if t_start is None or t_end is None:                  # temperatures from the energy changes of random flips
        X = rng.integers(0, 2, (64, n)).astype(float)
        steps = np.abs((1 - 2 * X) * (lin + X @ sym.T)).ravel()
        steps = steps[steps > 1e-12]
        typical = float(np.mean(steps)) if len(steps) else 1.0
        t_start = t_start or typical
        t_end = t_end or typical * 0.01
    t0, t1 = t_start, t_end
    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        for sweep in range(sweeps):
            T = t0 * (t1 / t0) ** (sweep / max(sweeps - 1, 1))
            for i in rng.permutation(n):
                d = 1 - 2 * x[i]
                delta = d * (lin[i] + sym[i] @ x)
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    x[i] += d
        while True:                                       # final quench: descend to the nearest local minimum
            delta = (1 - 2 * x) * (lin + sym @ x)
            i = int(np.argmin(delta))
            if delta[i] >= -1e-12:
                break
            x[i] += 1 - 2 * x[i]
        e = qubo.energy(x)
        if e < best_e:
            best_x, best_e = x.copy(), e
    return SolverResult(best_x.astype(int), float(best_e), 'simulated annealing')


def greedy_descent(qubo, restarts=20, seed=0):
    """Best-improvement single-bit descent from random starts (the simplest classical baseline).

    Returns
    -------
    SolverResult
    """
    rng = np.random.default_rng(seed)
    lin, sym = _coupling(qubo)
    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, qubo.n).astype(float)
        while True:
            delta = (1 - 2 * x) * (lin + sym @ x)
            i = int(np.argmin(delta))
            if delta[i] >= -1e-12:
                break
            x[i] += 1 - 2 * x[i]
        e = qubo.energy(x)
        if e < best_e:
            best_x, best_e = x.copy(), e
    return SolverResult(best_x.astype(int), float(best_e), 'greedy descent')


def random_sampling(qubo, shots=1000, seed=0):
    """Best of ``shots`` uniformly random bit strings: the null baseline for any sampling-based solver.

    A quantum sampler that returns the best of ``shots`` samples must beat this with the same ``shots``, otherwise
    its success only reflects how many samples it drew from a small search space.

    Returns
    -------
    SolverResult
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (shots, qubo.n))
    e = qubo.energy(X.astype(float))
    k = int(np.argmin(e))
    return SolverResult(X[k], float(e[k]), 'random sampling', dict(shots=shots))


def qaoa(qubo, depth=2, restarts=3, shots=1000, maxiter=150, seed=0):
    """QAOA through Qiskit's ``QAOAAnsatz``, simulated exactly (at most 16 qubits).

    The circuit parameters are optimised classically (COBYLA) to minimise the expected QUBO energy; the returned
    solution is the lowest-energy bit string among ``shots`` samples of the final state, as on hardware.

    Parameters
    ----------
    qubo : Qubo
    depth : int, optional
        Number of QAOA layers ``p``.
    restarts : int, optional
        Random starting angles; the best expected energy is kept.
    shots : int, optional
        Samples drawn from the optimised state.
    maxiter : int, optional
        COBYLA iterations per restart.
    seed : int, optional

    Returns
    -------
    SolverResult
        ``info`` holds ``expected_energy``, ``p_optimal`` (probability mass on the exact optimum), ``depth`` and
        ``ratio`` (expected energy over the optimum, for energies measured from the worst case).

    Raises
    ------
    ImportError
        If Qiskit is not installed (``pip install "quantum-mind[qiskit]"``).
    ValueError
        If the problem has more than 16 variables.
    """
    require('qiskit')
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize
    if qubo.n > 16:
        raise ValueError('the exact simulation supports at most 16 qubits')
    n = qubo.n
    h, J, _ = qubo.to_ising()
    terms = [('Z', [i], float(h[i])) for i in range(n) if abs(h[i]) > 1e-12]
    terms += [('ZZ', [i, j], float(J[i, j])) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-12]
    if not terms:
        terms = [('Z', [0], 0.0)]
    cost_op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    energies = qubo.all_energies()
    e_min, e_max = float(energies.min()), float(energies.max())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ansatz = QAOAAnsatz(cost_op, reps=depth).decompose(reps=4)     # basis gates: far faster to simulate
        rng = np.random.default_rng(seed)

        def probs(theta):
            return Statevector(ansatz.assign_parameters(theta)).probabilities()

        def objective(theta):
            return float(probs(theta) @ energies)

        best = None
        for _ in range(restarts):
            theta0 = rng.uniform(0, np.pi, ansatz.num_parameters)
            res = minimize(objective, theta0, method='COBYLA', options={'maxiter': maxiter})
            if best is None or res.fun < best.fun:
                best = res
        p = probs(best.x)
    k = rng.choice(len(p), size=shots, p=p / p.sum())
    k_best = int(k[np.argmin(energies[k])])
    x = np.array([(k_best >> i) & 1 for i in range(n)])
    p_opt = float(p[np.isclose(energies, e_min)].sum())
    ratio = (e_max - best.fun) / (e_max - e_min) if e_max > e_min else 1.0
    return SolverResult(x, float(energies[k_best]), 'QAOA (Qiskit)',
                        dict(expected_energy=float(best.fun), p_optimal=p_opt, depth=depth, ratio=float(ratio)))
