"""Quantum-inspired optimisers: classical algorithms that borrow a quantum idea (superposed bit
probabilities, quantum potential wells, transverse-field tunnelling). They run on ordinary hardware and
make no use of a quantum computer.

* :class:`QIEA`: quantum-inspired evolutionary algorithm for binary problems (Han and Kim, IEEE TEVC
  6(6), 2002), with their rotation lookup table by default and a simplified fixed-step rule as an option.
* :class:`QPSO`: quantum-behaved particle swarm optimisation for continuous problems (Sun, Feng and Xu,
  CEC 2004).
* :class:`SQA`: simulated quantum annealing by path-integral Monte Carlo (Martoňák, Santoro and
  Tosatti, PRB 66, 2002).
* :func:`simulated_annealing`: the classical baseline.

Examples
--------
>>> from qlcog.problems import maxcut
>>> from qlcog.inspired import QIEA, SQA, simulated_annealing
>>> q = maxcut([(0, 1), (1, 2), (2, 3), (3, 0)])
>>> QIEA(q, generations=50).run().value, SQA(q, sweeps=50).run().value, simulated_annealing(q, sweeps=50).value
(-4.0, -4.0, -4.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ['OptimResult', 'QIEA', 'QPSO', 'SQA', 'simulated_annealing']


@dataclass
class OptimResult:
    """Result of an optimiser.

    Attributes
    ----------
    x : numpy.ndarray
        Best solution found.
    value : float
        Its objective value.
    history : list of float
        Best value after each iteration.
    evaluations : int
        Number of objective evaluations.
    """
    x: np.ndarray
    value: float
    history: list = field(default_factory=list, repr=False)     # best value per iteration
    evaluations: int = 0


def _as_objective(problem):
    """Accept a qlcog.problems.Qubo (minimised) or a callable f(x) -> float (minimised)."""
    if hasattr(problem, 'energy'):
        return problem.energy, problem.n
    return problem, None


class QIEA:
    r"""Quantum-inspired evolutionary algorithm for binary minimisation.

    Each individual is a string of Q-bits; Q-bit i holds an angle :math:`\theta_i` with
    :math:`P(x_i = 1) = \sin^2\theta_i`. Observing an individual samples a bit string. A rotation gate
    moves every angle toward the best solution found so far, and migration exchanges best solutions
    within groups and globally.

    With ``rotation='lookup'`` (default) the rotation table of Han and Kim (2002, Table I) is used, with
    the comparison f(x) >= f(b) of their maximisation problem read as f(x) <= f(b) for minimisation. For
    each bit the table gives the rotation angle from (x_i, b_i, x better than b); the sign moves the
    amplitude toward the bit of the better solution. Q-bit angles stay in the first quadrant, so the sign
    column for alpha * beta > 0 applies, and an H-epsilon bound keeps P(x_i = 1) away from 0 and 1.
    ``rotation='simple'`` takes one fixed step ``delta`` toward b_i whenever x_i != b_i and x is not better.

    Parameters
    ----------
    problem : Qubo or callable
        A :class:`qlcog.problems.Qubo`, or ``f(x)`` for :math:`x \in \{0, 1\}^n` (then give ``n``).
    n : int, optional
        Number of bits (needed for a callable).
    pop : int, optional
        Population size.
    generations : int, optional
        Number of generations.
    rotation : {'lookup', 'simple'}, optional
        Rotation rule.
    delta : float, optional
        Step of the simple rule.
    migrate_every : int, optional
        Generations between global migrations (local migration every 5).
    groups : int, optional
        Number of local-migration groups.
    seed : int, optional
        Seed.
    """

    # Han and Kim (2002), Table I: (x_i, b_i, x better) -> (rotation angle, sign for alpha * beta > 0)
    LOOKUP = {(0, 0, False): (0.0, 0), (0, 0, True): (0.0, 0), (0, 1, False): (0.0, 0), (0, 1, True): (0.05 * np.pi, -1),
              (1, 0, False): (0.01 * np.pi, -1), (1, 0, True): (0.025 * np.pi, 1), (1, 1, False): (0.005 * np.pi, 1),
              (1, 1, True): (0.025 * np.pi, 1)}

    def __init__(self, problem, n=None, pop=20, generations=300, rotation='lookup', delta=0.01 * np.pi,
                 migrate_every=20, groups=4, seed=0):
        if rotation not in ('lookup', 'simple'):
            raise ValueError("rotation must be 'lookup' or 'simple'")
        self.f, n0 = _as_objective(problem); self.n = n or n0
        self.pop, self.generations, self.delta, self.rotation = pop, generations, delta, rotation
        self.migrate_every, self.groups, self.seed = migrate_every, groups, seed
        ang = np.zeros((2, 2, 2)); sgn = np.zeros((2, 2, 2))
        for (xi, bi, better), (a, sg) in self.LOOKUP.items():
            ang[xi, bi, int(better)] = a; sgn[xi, bi, int(better)] = sg
        self._step = ang * sgn                                   # signed rotation per case

    def run(self):
        """Run the evolutionary search.

        Returns
        -------
        OptimResult
        """
        rng = np.random.default_rng(self.seed); n, P = self.n, self.pop
        theta = np.full((P, n), np.pi / 4)                     # equal superposition
        observe = lambda: (rng.random((P, n)) < np.sin(theta) ** 2).astype(int)   # noqa: E731
        X = observe(); fx = np.array([self.f(x) for x in X])
        B, fB = X.copy(), fx.copy()                             # each individual's best
        g = int(np.argmin(fB)); best, fbest = B[g].copy(), float(fB[g]); hist = [fbest]; evals = P
        for gen in range(1, self.generations + 1):
            X = observe(); fx = np.array([self.f(x) for x in X]); evals += P
            if self.rotation == 'lookup':                       # Han and Kim's table, per bit
                better = np.broadcast_to((fx <= fB)[:, None], X.shape).astype(int)
                theta += self._step[X, B, better]
            else:                                               # simplified fixed step towards b
                diff = X != B
                direction = np.where(B == 1, 1.0, -1.0)
                worse = (fx >= fB)[:, None]
                theta += self.delta * direction * (diff & worse)
            theta = np.clip(theta, 0.02, np.pi / 2 - 0.02)      # H-epsilon gate: keep some randomness
            upd = fx < fB; B[upd] = X[upd]; fB[upd] = fx[upd]
            g = int(np.argmin(fB))
            if fB[g] < fbest:
                best, fbest = B[g].copy(), float(fB[g])
            if gen % self.migrate_every == 0:                   # global migration
                B[:] = best; fB[:] = fbest
            elif gen % 5 == 0:                                  # local migration within groups
                for grp in np.array_split(np.arange(P), self.groups):
                    k = grp[np.argmin(fB[grp])]; B[grp] = B[k]; fB[grp] = fB[k]
            hist.append(fbest)
        return OptimResult(best, fbest, hist, evals)


class QPSO:
    r"""Quantum-behaved particle swarm optimisation for continuous minimisation on a box.

    Each particle sits in a delta potential well centred on a local attractor p between its own best and
    the swarm best; a new position is sampled from the well's wave function,

    .. math:: x = p \pm \beta\,|m - x|\,\ln(1/u),

    where m is the mean of the personal bests and the contraction-expansion coefficient
    :math:`\beta` decreases linearly from ``beta0`` to ``beta1``.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float``.
    lower, upper : array_like
        Box bounds.
    particles : int, optional
        Swarm size.
    iterations : int, optional
        Number of iterations.
    beta0, beta1 : float, optional
        Start and end values of :math:`\beta`.
    seed : int, optional
        Seed.
    """

    def __init__(self, f, lower, upper, particles=30, iterations=300, beta0=1.0, beta1=0.5, seed=0):
        self.f = f; self.lo, self.hi = np.asarray(lower, float), np.asarray(upper, float)
        self.N, self.iterations, self.beta0, self.beta1, self.seed = particles, iterations, beta0, beta1, seed

    def run(self):
        """Run the swarm.

        Returns
        -------
        OptimResult
        """
        rng = np.random.default_rng(self.seed); d = len(self.lo)
        x = self.lo + (self.hi - self.lo) * rng.random((self.N, d))
        fx = np.array([self.f(v) for v in x]); pb, fpb = x.copy(), fx.copy()
        g = int(np.argmin(fpb)); hist = [float(fpb[g])]; evals = self.N
        for it in range(self.iterations):
            beta = self.beta0 - (self.beta0 - self.beta1) * it / max(self.iterations - 1, 1)
            mbest = pb.mean(0); phi = rng.random((self.N, d))
            p = phi * pb + (1 - phi) * pb[g]
            u = rng.random((self.N, d)); sign = np.where(rng.random((self.N, d)) < 0.5, -1.0, 1.0)
            x = np.clip(p + sign * beta * np.abs(mbest - x) * np.log(1 / u), self.lo, self.hi)
            fx = np.array([self.f(v) for v in x]); evals += self.N
            upd = fx < fpb; pb[upd] = x[upd]; fpb[upd] = fx[upd]; g = int(np.argmin(fpb))
            hist.append(float(fpb[g]))
        return OptimResult(pb[g].copy(), float(fpb[g]), hist, evals)


class SQA:
    r"""Simulated quantum annealing (path-integral Monte Carlo) for a QUBO.

    The transverse-field Ising model is mapped (Suzuki-Trotter) onto P coupled classical replicas at
    temperature T; the inter-replica coupling
    :math:`J_\perp = -\tfrac{PT}{2}\ln\tanh\bigl(\Gamma / (PT)\bigr)` grows as the transverse field
    :math:`\Gamma` is lowered, so the replicas merge into one solution. Tunnelling through tall, thin
    barriers is the quantum feature it imitates.

    Parameters
    ----------
    qubo : Qubo
        Problem.
    replicas : int, optional
        Number of Trotter replicas P.
    sweeps : int, optional
        Monte Carlo sweeps.
    T : float, optional
        Temperature.
    gamma0, gamma1 : float, optional
        Initial and final transverse field (geometric schedule).
    seed : int, optional
        Seed.
    """

    def __init__(self, qubo, replicas=16, sweeps=400, T=0.05, gamma0=3.0, gamma1=1e-3, seed=0):
        self.qubo = qubo; self.P, self.sweeps, self.T = replicas, sweeps, T
        self.g0, self.g1, self.seed = gamma0, gamma1, seed

    def run(self):
        """Run the annealing schedule.

        Returns
        -------
        OptimResult
            Best bit string found in any replica.
        """
        h, J, c = self.qubo.to_ising(); n = len(h)
        scale = max(np.abs(h).max(initial=0), np.abs(J).max(initial=0), 1e-12)
        h, Jf = h / scale, (J + J.T) / scale
        rng = np.random.default_rng(self.seed); P, T = self.P, self.T
        s = rng.choice([-1, 1], size=(P, n)).astype(float)
        energy = lambda v: float(h @ v + 0.5 * v @ Jf @ v)     # noqa: E731
        best = s[0].copy(); ebest = energy(best); hist = []
        for k in range(self.sweeps):
            G = self.g0 * (self.g1 / self.g0) ** (k / max(self.sweeps - 1, 1))
            Jp = -0.5 * P * T * np.log(np.tanh(G / (P * T)))
            for i in rng.permutation(n):
                for parity in (0, 1):                          # even, then odd replicas (neighbours fixed)
                    rows = np.arange(parity, P, 2)
                    local = h[i] + s[rows] @ Jf[:, i]
                    nb = s[(rows - 1) % P, i] + s[(rows + 1) % P, i]
                    # H_P = sum_k [E(s_k) - J_perp sum_i s_i^k s_i^{k+1}], Boltzmann weight exp(-H_P / (P T))
                    dE = -2 * s[rows, i] * local + 2 * Jp * s[rows, i] * nb
                    acc = rng.random(len(rows)) < np.exp(-np.clip(dE / (P * T), -50, 50))
                    s[rows[acc], i] *= -1
            for r in range(P):
                e = energy(s[r])
                if e < ebest:
                    ebest, best = e, s[r].copy()
            hist.append(ebest)
        x = ((1 - best) / 2).astype(int)
        return OptimResult(x, self.qubo.energy(x), [v * scale + c for v in hist], self.sweeps * P * n)


def simulated_annealing(qubo, sweeps=400, T0=2.0, T1=0.01, seed=0):
    """Classical simulated annealing baseline (single-spin Metropolis on the same Ising model).

    Parameters
    ----------
    qubo : Qubo
        Problem.
    sweeps : int, optional
        Sweeps over all spins.
    T0, T1 : float, optional
        Initial and final temperature (geometric schedule).
    seed : int, optional
        Seed.

    Returns
    -------
    OptimResult
    """
    h, J, c = qubo.to_ising(); n = len(h)
    scale = max(np.abs(h).max(initial=0), np.abs(J).max(initial=0), 1e-12)
    h, Jf = h / scale, (J + J.T) / scale
    rng = np.random.default_rng(seed); s = rng.choice([-1.0, 1.0], n)
    e = h @ s + 0.5 * s @ Jf @ s; best, ebest = s.copy(), e; hist = []
    for k in range(sweeps):
        T = T0 * (T1 / T0) ** (k / max(sweeps - 1, 1))
        for i in rng.permutation(n):
            dE = -2 * s[i] * (h[i] + Jf[i] @ s)
            if dE < 0 or rng.random() < np.exp(-dE / T):
                s[i] *= -1; e += dE
                if e < ebest:
                    ebest, best = e, s.copy()
        hist.append(ebest * scale + c)
    x = ((1 - best) / 2).astype(int)
    return OptimResult(x, qubo.energy(x), hist, sweeps * n)
