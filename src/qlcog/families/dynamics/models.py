"""Belief dynamics: how a belief or preference evolves over time and how intermediate judgements
change later ones.

1. Evidence accumulation on N ordered belief states (Busemeyer, Kvam and Pleskac, 2019):
   MarkovWalk     probability vector phi, d phi/dt = K phi; K tridiagonal with up rate alpha and down
                  rate beta (drift mu = alpha/(alpha+beta), rate gamma = alpha + beta), reflecting ends.
   QuantumWalk    amplitude vector psi, U(t) = exp(-i H t); H_jj = mu * x_j (x_j = j/(N-1)) and
                  H_{j,j+1} = H_{j+1,j} = sigma.
   OpenSystemWalk QuantumWalk plus Lindblad dephasing of rate lam in the state basis; lam = 0 is the
                  quantum walk, large lam suppresses interference.
   The N states are grouped into K response categories (equal bins). Design: dict
   {condition: ('single', t)} -> distribution over K categories, or ('joint', t1, t2) -> K x K joint
   distribution of judgements at t1 and t2 (flattened row-major). A judgement at t1 collapses the
   state onto the chosen category (projection), which is how an intermediate judgement can change a
   later one in the quantum and open-system models but not, on average, in the Markov model.

2. Belief updating over discrete events, answered as yes/no (e.g. trust in an agent after its
   successes and failures):
   MarkovBelief      P(yes) moves toward 1 after a positive event (rate up) and toward 0 after a
                     negative one (rate down); a query reads the state.
   OpenSystemBelief  qubit density matrix; rotation by +a_pos / -a_neg per event, then dephasing of
                     strength gamma; a query is a projective measurement.
   Design: dict {condition: (events tuple of 1/0, query indices tuple)} -> distribution over the
   2^k answer sequences (yes = 1), in itertools.product order with 1 first."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.linalg import expm
from ...core import Model, Param
from ...core.linalg import lindblad_superoperator


def _bins(N, K):
    edges = np.linspace(0, N, K + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(K)]


class _WalkBase(Model):
    def setup(self):
        """Build the generator matrices from the parameters."""
        N = int(self.options.get('n_states', 21)); K = int(self.options.get('n_categories', 3))
        width = float(self.options.get('initial_width', 0.15))
        x = np.linspace(0, 1, N)
        init = np.exp(-0.5 * ((x - 0.5) / width) ** 2)
        return N, K, x, init / init.sum(), _bins(N, K)

    def predict(self, design=None):
        design = design or {'t1': ('single', 1.0)}
        out = {}
        for cond, spec in design.items():
            if spec[0] == 'single':
                p = self.state_probs(spec[1]); out[cond] = np.array([p[b].sum() for b in self.bins])
            else:
                out[cond] = self.joint(spec[1], spec[2]).reshape(-1)
        return out


class MarkovWalk(_WalkBase):
    """Classical baseline: evidence accumulation as a Markov random walk over N belief states (a judgement
    does not disturb the state)."""
    PARAMS = [Param('mu', 'prob', 0.6), Param('gamma', 'positive', 5.0)]
    name = 'Markov random walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.N, self.K, self.x, self.p0, self.bins = self.setup()
        a, b = self.mu * self.gamma, (1 - self.mu) * self.gamma
        Kmat = np.zeros((self.N, self.N))
        for j in range(self.N - 1):
            Kmat[j + 1, j] += a; Kmat[j, j + 1] += b               # K[to, from]
        Kmat -= np.diag(Kmat.sum(axis=0))
        self.Kmat = Kmat

    def state_probs(self, t):
        """Distribution over belief states at time t."""
        return expm(self.Kmat * t) @ self.p0

    def joint(self, t1, t2):
        """Joint distribution of the judgements at times t1 and t2 (judgement at t1 included)."""
        p1 = expm(self.Kmat * t1) @ self.p0; T = expm(self.Kmat * (t2 - t1)); J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            q = np.zeros(self.N); q[bi] = p1[bi]
            q2 = T @ q
            for j, bj in enumerate(self.bins): J[i, j] = q2[bj].sum()
        return J


class QuantumWalk(_WalkBase):
    """Quantum walk over N belief states (Busemeyer, Kvam and Pleskac): unitary evolution, so an
    intermediate judgement changes later judgements."""
    PARAMS = [Param('mu', 'real', 2.0), Param('sigma', 'positive', 5.0)]
    name = 'Quantum walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.N, self.K, self.x, p0, self.bins = self.setup()
        self.psi0 = np.sqrt(p0).astype(complex)
        H = np.diag(self.mu * self.x)
        for j in range(self.N - 1): H[j, j + 1] = H[j + 1, j] = self.sigma
        self.H = H

    def U(self, t):
        """Unitary exp(-i H t)."""
        return expm(-1j * self.H * t)

    def state_probs(self, t):
        """Distribution over belief states at time t (Born rule)."""
        return np.abs(self.U(t) @ self.psi0) ** 2

    def joint(self, t1, t2):
        """Joint distribution of the judgements at t1 and t2, with collapse at t1 (Lueders)."""
        a1 = self.U(t1) @ self.psi0; U2 = self.U(t2 - t1); J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            v = np.zeros(self.N, complex); v[bi] = a1[bi]
            w = U2 @ v
            for j, bj in enumerate(self.bins): J[i, j] = np.sum(np.abs(w[bj]) ** 2)
        return J


class OpenSystemWalk(QuantumWalk):
    """Quantum walk with Lindblad dephasing between the quantum walk (rate 0) and Markov-like behaviour
    (large rate)."""
    PARAMS = QuantumWalk.PARAMS + [Param('lam', 'positive', 1.0)]
    name = 'Open-system walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        jumps = [np.diag(np.eye(self.N)[j]) for j in range(self.N)]
        self.L = lindblad_superoperator(self.H, jumps, [self.lam] * self.N)
        self.rho0 = np.outer(self.psi0, self.psi0.conj())

    def _evolve(self, rho, t):
        n = rho.shape[0]
        return (expm(self.L * t) @ rho.reshape(-1)).reshape(n, n)

    def state_probs(self, t):
        """Distribution over belief states at time t under Lindblad dynamics."""
        return np.real(np.diag(self._evolve(self.rho0, t)))

    def joint(self, t1, t2):
        """Joint distribution of judgements at t1 and t2 with collapse at t1."""
        r1 = self._evolve(self.rho0, t1); J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            P = np.zeros((self.N, self.N)); P[bi, bi] = 1.0
            r = self._evolve(P @ r1 @ P, t2 - t1)
            d = np.real(np.diag(r))
            for j, bj in enumerate(self.bins): J[i, j] = d[bj].sum()
        return J


# ---------------------------------------------------------------------------- discrete events
def _answer_keys(k):
    return list(itertools.product((1, 0), repeat=k))


class MarkovBelief(Model):
    """Classical baseline: a yes-probability updated by each event (successes raise, failures lower); asking
    does not change it."""
    PARAMS = [Param('p0', 'prob', 0.5), Param('up', 'prob', 0.3), Param('down', 'prob', 0.3)]
    name = 'Markov belief'

    def step(self, p, e):
        """Updated yes-probability after one event (1 = success, 0 = failure)."""
        return p + (1 - p) * self.up if e else p * (1 - self.down)

    def answer_probs(self, events, queries):
        """Probabilities of the answer sequences given at the query times, after the events."""
        out = {(): (1.0, self.p0)}
        for t, e in enumerate(events):
            out = {k: (pr, self.step(p, e)) for k, (pr, p) in out.items()}
            if t in queries:
                new = {}
                for k, (pr, p) in out.items():
                    new[k + (1,)] = (pr * p, 1.0); new[k + (0,)] = (pr * (1 - p), 0.0)
                out = new
        return {k: pr for k, (pr, _) in out.items()}

    def predict(self, design=None):
        design = design or {'final': ((1, 1, 0, 1, 0, 1), (5,))}
        out = {}
        for cond, (events, queries) in design.items():
            pr = self.answer_probs(events, set(queries))
            out[cond] = np.array([pr.get(k, 0.0) for k in _answer_keys(len(queries))])
        return out


class OpenSystemBelief(MarkovBelief):
    """Belief (e.g. trust) as a qubit: events rotate it, dephasing gamma pulls it towards the measurement
    axis, and asking collapses it (Lueders), so asking can change later answers."""
    PARAMS = [Param('phi0', 'bounded', 1.6, 0, np.pi), Param('a_pos', 'bounded', 0.8, 0, np.pi),
              Param('a_neg', 'bounded', 1.2, 0, np.pi), Param('gamma', 'prob', 0.3)]
    name = 'Open-system belief'

    @staticmethod
    def _ry(a):
        c, s = np.cos(a / 2), np.sin(a / 2); return np.array([[c, -s], [s, c]])

    def answer_probs(self, events, queries):
        """Probabilities of the answer sequences given at the query times (rotation, dephasing, collapse)."""
        v = np.array([np.cos(self.phi0 / 2), np.sin(self.phi0 / 2)])
        out = {(): (1.0, np.outer(v, v))}
        for t, e in enumerate(events):
            U = self._ry(-self.a_pos if e else self.a_neg)
            new = {}
            for k, (pr, r) in out.items():
                r = U @ r @ U.T; f = 1 - self.gamma
                new[k] = (pr, np.array([[r[0, 0], f * r[0, 1]], [f * r[1, 0], r[1, 1]]]))
            out = new
            if t in queries:
                new = {}
                for k, (pr, r) in out.items():
                    py = float(np.clip(r[0, 0], 0, 1))
                    new[k + (1,)] = (pr * py, np.diag([1.0, 0.0])); new[k + (0,)] = (pr * (1 - py), np.diag([0.0, 1.0]))
                out = new
        return {k: pr for k, (pr, _) in out.items()}


def final_yes(model, events, queries):
    """P(yes) at the last query, marginalised over earlier answers."""
    pr = model.answer_probs(events, set(queries))
    return sum(p for k, p in pr.items() if k[-1] == 1)


def question_effect(model, events, last, intermediate):
    """Change in P(yes at `last`) caused by also asking at `intermediate`."""
    return final_yes(model, events, (intermediate, last)) - final_yes(model, events, (last,))
