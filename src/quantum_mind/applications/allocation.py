r"""Trust-aware task allocation: quantum and quantum-inspired optimisation driven by models of people.

Task allocation as a QUBO (Lucas, 2014; :func:`quantum_mind.problems.task_allocation`) uses fixed
costs. In a team of people working with a robot, a task that needs the robot's help (a hand-over, a
shared lift) goes wrong more often with a person who does not trust the robot, and people who carry
too many tasks slow down. :func:`trust_aware_allocation` adds both effects to the costs:

.. math::

    E(x) = \sum_{a,t} x_{a,t}\,\bigl(C_{a,t} + F\,r_t\,(1 - \tau_a)\bigr)
         + \lambda \sum_a \Bigl(\sum_t x_{a,t}\Bigr)^2
         + A \sum_t \Bigl(\sum_a x_{a,t} - 1\Bigr)^2 ,

where :math:`\tau_a` is person :math:`a`'s probability of trusting the robot, :math:`r_t \in [0, 1]`
how much task :math:`t` depends on the robot, :math:`F` the cost of a failed robot-assisted step,
:math:`\lambda` the workload weight and :math:`A` the one-agent-per-task penalty.

The trust values come from the library's people models: the trust qubit of
:class:`~quantum_mind.applications.handover.TrustAwareHandover`, an
:class:`~quantum_mind.core.online.OnlinePersonModel`, or, for a cautious allocation, the lower end of
a credible interval. The QUBO is solved with any of the package's solvers (exhaustive search,
simulated quantum annealing, QAOA).

Examples
--------
>>> import numpy as np
>>> from quantum_mind.applications.allocation import trust_aware_allocation, decode
>>> base = np.ones((2, 2))                         # two people, two tasks, equal effort
>>> q = trust_aware_allocation(base, trust=[0.9, 0.3], reliance=[1.0, 0.0], fail_cost=10, workload=1.0)
>>> x, e = q.brute_force()
>>> decode(x, 2, 2)        # the robot-assisted task 0 goes to the trusting person; workload splits the rest
{0: [0], 1: [1]}
"""
from __future__ import annotations

import itertools
import numpy as np

from ..problems import Qubo

__all__ = ['expected_costs', 'trust_aware_allocation', 'decode']


def expected_costs(base_costs, trust, reliance, fail_cost=10.0):
    """Costs of each person doing each task, including expected failures of robot-assisted steps.

    Parameters
    ----------
    base_costs : array_like
        ``base_costs[a, t]``: effort of person a on task t.
    trust : array_like
        Probability that each person trusts the robot, in [0, 1].
    reliance : array_like
        How much each task depends on the robot, in [0, 1].
    fail_cost : float, optional
        Cost of a failed robot-assisted step.

    Returns
    -------
    numpy.ndarray
        ``C[a, t] + fail_cost * reliance[t] * (1 - trust[a])``.

    Raises
    ------
    ValueError
        If the shapes do not match or a probability lies outside [0, 1].
    """
    C = np.asarray(base_costs, float)
    tau = np.asarray(trust, float)
    r = np.asarray(reliance, float)
    if C.ndim != 2 or tau.shape != (C.shape[0],) or r.shape != (C.shape[1],):
        raise ValueError('base_costs must be (people, tasks), trust one per person and reliance one per task')
    if np.any((tau < 0) | (tau > 1)) or np.any((r < 0) | (r > 1)):
        raise ValueError('trust and reliance must lie in [0, 1]')
    return C + fail_cost * np.outer(1 - tau, r)


def trust_aware_allocation(base_costs, trust, reliance, fail_cost=10.0, workload=0.0, penalty=None):
    """QUBO for assigning every task to one person, accounting for trust in the robot and workload.

    Parameters
    ----------
    base_costs : array_like
        ``base_costs[a, t]``: effort of person a on task t.
    trust : array_like
        Probability that each person trusts the robot (pass a lower credible bound for a cautious plan).
    reliance : array_like
        How much each task depends on the robot, in [0, 1].
    fail_cost : float, optional
        Cost of a failed robot-assisted step.
    workload : float, optional
        Weight :math:`\\lambda` of the quadratic workload term (0 ignores workload).
    penalty : float, optional
        One-agent-per-task penalty :math:`A` (default: large enough to dominate the costs).

    Returns
    -------
    Qubo
        Variable :math:`x_{a,t}` has index ``a * n_tasks + t``.
    """
    C = expected_costs(base_costs, trust, reliance, fail_cost)
    na, nt = C.shape
    A = penalty or 2 * (np.abs(C).max() * nt + workload * nt ** 2 + 1)
    idx = lambda a, t: a * nt + t                                    # noqa: E731
    Q = np.zeros((na * nt, na * nt))
    offset = 0.0
    for a in range(na):
        for t in range(nt):
            Q[idx(a, t), idx(a, t)] += C[a, t] + workload                # x^2 = x for the workload square
        for t1, t2 in itertools.combinations(range(nt), 2):
            Q[idx(a, t1), idx(a, t2)] += 2 * workload
    for t in range(nt):                                                  # A (sum_a x_at - 1)^2
        vs = [idx(a, t) for a in range(na)]
        for i in vs:
            Q[i, i] -= A
        for i, j in itertools.combinations(vs, 2):
            Q[i, j] += 2 * A
        offset += A
    labels = ['person%d->task%d' % (a, t) for a in range(na) for t in range(nt)]
    return Qubo(Q, offset, labels, 'Trust-aware task allocation')


def decode(x, n_people, n_tasks):
    """Tasks given to each person by a bit string.

    Parameters
    ----------
    x : array_like
        Bits in the order of :func:`trust_aware_allocation`.
    n_people, n_tasks : int

    Returns
    -------
    dict
        ``{person: [tasks]}``.
    """
    x = np.asarray(x).reshape(n_people, n_tasks)
    return {a: [int(t) for t in np.flatnonzero(x[a])] for a in range(n_people)}
