r"""Belief dynamics: how a belief or preference evolves over time, and how intermediate judgements
change later ones.

**1. Evidence accumulation on N ordered belief states** (Busemeyer, Kvam and Pleskac, 2019).

* :class:`MarkovWalk`: probability vector :math:`\phi`, :math:`\dot\phi = K\phi`, with :math:`K`
  tridiagonal (up rate :math:`\alpha`, down rate :math:`\beta`; drift
  :math:`\mu = \alpha / (\alpha + \beta)`, rate :math:`\gamma = \alpha + \beta`) and reflecting ends.
* :class:`QuantumWalk`: amplitude vector :math:`\psi`, :math:`U(t) = e^{-iHt}` with
  :math:`H_{jj} = \mu x_j` (:math:`x_j = j/(N-1)`) and :math:`H_{j,j+1} = H_{j+1,j} = \sigma`.
* :class:`OpenSystemWalk`: the quantum walk plus Lindblad dephasing of rate :math:`\lambda` in the
  state basis; :math:`\lambda = 0` is the quantum walk, a large :math:`\lambda` suppresses interference.

The N states are grouped into K response categories (equal bins). The design is
``{condition: ('single', t)}`` for a distribution over the K categories, or
``{condition: ('joint', t1, t2)}`` for the K x K joint distribution of judgements at t1 and t2
(flattened row-major). A judgement at t1 collapses the state onto the chosen category, which is how
an intermediate judgement can change a later one in the quantum and open-system models but not, on
average, in the Markov model.

**2. Belief updating over discrete events, queried as yes/no** (for example trust in an agent after
its successes and failures).

* :class:`MarkovBelief`: P(yes) moves toward 1 after a positive event and toward 0 after a negative
  one; a query only reads the state.
* :class:`OpenSystemBelief`: a qubit density matrix rotated by each event, then dephased; a query is
  a projective measurement.

The design is ``{condition: (events, queries)}``, with ``events`` a tuple of 1/0 and ``queries`` the
event indices after which the question is asked. The prediction is a distribution over the
:math:`2^k` answer sequences (yes = 1), in :func:`itertools.product` order with 1 first.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.linalg import expm
from ...core import Model, Param
from ...core.linalg import lindblad_superoperator


def _bins(N, K):
    """Split N ordered states into K contiguous, nearly equal response categories."""
    edges = np.linspace(0, N, K + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(K)]


class _WalkBase(Model):
    """Shared set-up and prediction of the three walk models."""

    def setup(self):
        """Build the state grid, the initial distribution and the response bins from the options.

        Returns
        -------
        tuple
            ``(N, K, x, p0, bins)``: number of states, number of categories, state positions in
            [0, 1], initial distribution (a Gaussian bump around 0.5) and the category bins.
        """
        N = int(self.options.get('n_states', 21))
        K = int(self.options.get('n_categories', 3))
        width = float(self.options.get('initial_width', 0.15))
        x = np.linspace(0, 1, N)
        init = np.exp(-0.5 * ((x - 0.5) / width) ** 2)
        return N, K, x, init / init.sum(), _bins(N, K)

    def predict(self, design=None):
        """Category distribution (``'single'``) or flattened joint distribution (``'joint'``) per condition."""
        design = design or {'t1': ('single', 1.0)}
        out = {}
        for cond, spec in design.items():
            if spec[0] == 'single':
                p = self.state_probs(spec[1])
                out[cond] = np.array([p[b].sum() for b in self.bins])
            else:
                out[cond] = self.joint(spec[1], spec[2]).reshape(-1)
        return out


class MarkovWalk(_WalkBase):
    """Classical baseline: Markov random walk over N belief states.

    A judgement reads the state without disturbing its distribution.

    Parameters
    ----------
    mu : float
        Drift, the probability of a step up, in (0, 1).
    gamma : float
        Total transition rate.
    n_states : int, optional
        Number of belief states (default 21).
    n_categories : int, optional
        Number of response categories (default 3).
    initial_width : float, optional
        Width of the initial Gaussian distribution (default 0.15).
    """
    PARAMS = [Param('mu', 'prob', 0.6), Param('gamma', 'positive', 5.0)]
    name = 'Markov random walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.N, self.K, self.x, self.p0, self.bins = self.setup()
        a, b = self.mu * self.gamma, (1 - self.mu) * self.gamma
        Kmat = np.zeros((self.N, self.N))
        for j in range(self.N - 1):
            Kmat[j + 1, j] += a                                    # K[to, from]: step up
            Kmat[j, j + 1] += b                                    # step down
        Kmat -= np.diag(Kmat.sum(axis=0))                          # columns sum to zero
        self.Kmat = Kmat

    def state_probs(self, t):
        """Distribution over belief states.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            :math:`e^{Kt}\\phi_0`.
        """
        return expm(self.Kmat * t) @ self.p0

    def joint(self, t1, t2):
        """Joint distribution of judgements at two times.

        Parameters
        ----------
        t1, t2 : float
            Times of the two judgements, ``t1 <= t2``.

        Returns
        -------
        numpy.ndarray
            K x K matrix: rows are the category at ``t1``, columns the category at ``t2``.
        """
        p1 = expm(self.Kmat * t1) @ self.p0
        T = expm(self.Kmat * (t2 - t1))
        J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            q = np.zeros(self.N)
            q[bi] = p1[bi]                       # condition on the first judgement
            q2 = T @ q
            for j, bj in enumerate(self.bins):
                J[i, j] = q2[bj].sum()
        return J


class QuantumWalk(_WalkBase):
    """Quantum walk over N belief states (Busemeyer, Kvam and Pleskac).

    Evolution is unitary, so an intermediate judgement (a collapse) changes later judgements.

    Parameters
    ----------
    mu : float
        Drift (diagonal potential strength).
    sigma : float
        Diffusion (coupling between neighbouring states).
    n_states, n_categories, initial_width
        As in :class:`MarkovWalk`.
    """
    PARAMS = [Param('mu', 'real', 2.0), Param('sigma', 'positive', 5.0)]
    name = 'Quantum walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.N, self.K, self.x, p0, self.bins = self.setup()
        self.psi0 = np.sqrt(p0).astype(complex)                    # amplitudes with the same distribution
        H = np.diag(self.mu * self.x)
        for j in range(self.N - 1):
            H[j, j + 1] = H[j + 1, j] = self.sigma
        self.H = H

    def U(self, t):
        """Unitary :math:`e^{-iHt}`.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
        """
        return expm(-1j * self.H * t)

    def state_probs(self, t):
        """Distribution over belief states at time ``t`` (Born rule).

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
        """
        return np.abs(self.U(t) @ self.psi0) ** 2

    def joint(self, t1, t2):
        """Joint distribution of judgements at ``t1`` and ``t2``, with collapse at ``t1`` (Lüders rule).

        Parameters
        ----------
        t1, t2 : float
            Times of the two judgements.

        Returns
        -------
        numpy.ndarray
            K x K joint distribution.
        """
        a1 = self.U(t1) @ self.psi0
        U2 = self.U(t2 - t1)
        J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            v = np.zeros(self.N, complex)
            v[bi] = a1[bi]                       # project onto category i (unnormalised)
            w = U2 @ v
            for j, bj in enumerate(self.bins):
                J[i, j] = np.sum(np.abs(w[bj]) ** 2)
        return J


class OpenSystemWalk(QuantumWalk):
    """Quantum walk with Lindblad dephasing.

    Interpolates between the quantum walk (``lam = 0``) and Markov-like behaviour (large ``lam``).

    Parameters
    ----------
    mu, sigma : float
        As in :class:`QuantumWalk`.
    lam : float
        Dephasing rate in the state basis.
    """
    PARAMS = QuantumWalk.PARAMS + [Param('lam', 'positive', 1.0)]
    name = 'Open-system walk'

    def __init__(self, **kw):
        super().__init__(**kw)
        jumps = [np.diag(np.eye(self.N)[j]) for j in range(self.N)]     # |j><j| for every state
        self.L = lindblad_superoperator(self.H, jumps, [self.lam] * self.N)
        self.rho0 = np.outer(self.psi0, self.psi0.conj())

    def _evolve(self, rho, t):
        """Evolve a density matrix for time t."""
        n = rho.shape[0]
        return (expm(self.L * t) @ rho.reshape(-1)).reshape(n, n)

    def state_probs(self, t):
        """Distribution over belief states at time ``t`` under Lindblad dynamics.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
        """
        return np.real(np.diag(self._evolve(self.rho0, t)))

    def joint(self, t1, t2):
        """Joint distribution of judgements at ``t1`` and ``t2``, with collapse at ``t1``.

        Parameters
        ----------
        t1, t2 : float
            Times of the two judgements.

        Returns
        -------
        numpy.ndarray
            K x K joint distribution.
        """
        r1 = self._evolve(self.rho0, t1)
        J = np.zeros((self.K, self.K))
        for i, bi in enumerate(self.bins):
            P = np.zeros((self.N, self.N))
            P[bi, bi] = 1.0
            r = self._evolve(P @ r1 @ P, t2 - t1)
            d = np.real(np.diag(r))
            for j, bj in enumerate(self.bins):
                J[i, j] = d[bj].sum()
        return J


# ---------------------------------------------------------------------------- discrete events
def _answer_keys(k):
    """All yes/no answer sequences of length k, yes (1) first."""
    return list(itertools.product((1, 0), repeat=k))


class MarkovBelief(Model):
    """Classical baseline: a yes-probability updated by each event.

    Successes raise it, failures lower it; asking does not change it.

    Parameters
    ----------
    p0 : float
        Initial P(yes).
    up : float
        Fraction of the distance to 1 covered after a success.
    down : float
        Fraction of the distance to 0 covered after a failure.
    """
    PARAMS = [Param('p0', 'prob', 0.5), Param('up', 'prob', 0.3), Param('down', 'prob', 0.3)]
    name = 'Markov belief'

    def step(self, p, e):
        """Update the yes-probability after one event.

        Parameters
        ----------
        p : float
            Current P(yes).
        e : int
            1 for a success, 0 for a failure.

        Returns
        -------
        float
        """
        return p + (1 - p) * self.up if e else p * (1 - self.down)

    def answer_probs(self, events, queries):
        """Probabilities of the answer sequences.

        Parameters
        ----------
        events : sequence of int
            Events (1 success, 0 failure).
        queries : set of int
            Event indices after which the question is asked.

        Returns
        -------
        dict
            ``{answers: probability}``.
        """
        out = {(): (1.0, self.p0)}
        for t, e in enumerate(events):
            out = {k: (pr, self.step(p, e)) for k, (pr, p) in out.items()}
            if t in queries:
                # the answer is recorded; the state becomes certain about what was answered
                new = {}
                for k, (pr, p) in out.items():
                    new[k + (1,)] = (pr * p, 1.0)
                    new[k + (0,)] = (pr * (1 - p), 0.0)
                out = new
        return {k: pr for k, (pr, _) in out.items()}

    def predict(self, design=None):
        """Distribution over answer sequences for each condition."""
        design = design or {'final': ((1, 1, 0, 1, 0, 1), (5,))}
        out = {}
        for cond, (events, queries) in design.items():
            pr = self.answer_probs(events, set(queries))
            out[cond] = np.array([pr.get(k, 0.0) for k in _answer_keys(len(queries))])
        return out


class OpenSystemBelief(MarkovBelief):
    """Belief (for example trust) as a qubit.

    Each event rotates the belief about the y axis; dephasing ``gamma`` then pulls it toward the
    measurement axis; asking collapses it (Lüders rule), so asking can change later answers.

    Parameters
    ----------
    phi0 : float
        Initial polar angle on the Bloch sphere (0 is certain "yes").
    a_pos : float
        Rotation toward "yes" after a success.
    a_neg : float
        Rotation toward "no" after a failure.
    gamma : float
        Dephasing strength per event, in (0, 1).
    """
    PARAMS = [Param('phi0', 'bounded', 1.6, 0, np.pi), Param('a_pos', 'bounded', 0.8, 0, np.pi),
              Param('a_neg', 'bounded', 1.2, 0, np.pi), Param('gamma', 'prob', 0.3)]
    name = 'Open-system belief'

    @staticmethod
    def _ry(a):
        """Real rotation RY(a)."""
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -s], [s, c]])

    def answer_probs(self, events, queries):
        """Probabilities of the answer sequences (rotation, dephasing, collapse).

        Parameters
        ----------
        events : sequence of int
            Events (1 success, 0 failure).
        queries : set of int
            Event indices after which the question is asked.

        Returns
        -------
        dict
            ``{answers: probability}``.
        """
        v = np.array([np.cos(self.phi0 / 2), np.sin(self.phi0 / 2)])
        out = {(): (1.0, np.outer(v, v))}
        for t, e in enumerate(events):
            U = self._ry(-self.a_pos if e else self.a_neg)
            new = {}
            for k, (pr, r) in out.items():
                r = U @ r @ U.T
                f = 1 - self.gamma                       # off-diagonal terms shrink by (1 - gamma)
                new[k] = (pr, np.array([[r[0, 0], f * r[0, 1]], [f * r[1, 0], r[1, 1]]]))
            out = new
            if t in queries:
                new = {}
                for k, (pr, r) in out.items():
                    py = float(np.clip(r[0, 0], 0, 1))
                    new[k + (1,)] = (pr * py, np.diag([1.0, 0.0]))
                    new[k + (0,)] = (pr * (1 - py), np.diag([0.0, 1.0]))
                out = new
        return {k: pr for k, (pr, _) in out.items()}


def final_yes(model, events, queries):
    """P(yes) at the last query, marginalised over the earlier answers.

    Parameters
    ----------
    model : MarkovBelief or OpenSystemBelief
        Belief model.
    events : sequence of int
        Events.
    queries : sequence of int
        Query positions; the last one is reported.

    Returns
    -------
    float
    """
    pr = model.answer_probs(events, set(queries))
    return sum(p for k, p in pr.items() if k[-1] == 1)


def question_effect(model, events, last, intermediate):
    """Effect of an intermediate question on a later answer.

    Parameters
    ----------
    model : MarkovBelief or OpenSystemBelief
        Belief model.
    events : sequence of int
        Events.
    last : int
        Position of the final question.
    intermediate : int
        Position of the extra, earlier question.

    Returns
    -------
    float
        P(yes at ``last`` | also asked at ``intermediate``) - P(yes at ``last`` | not asked). Zero for
        :class:`MarkovBelief` on average; generally non-zero for :class:`OpenSystemBelief`.
    """
    return final_yes(model, events, (intermediate, last)) - final_yes(model, events, (last,))
