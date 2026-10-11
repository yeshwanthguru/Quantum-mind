"""Run several solvers on one robot problem and report them in the robot's terms."""
from __future__ import annotations

from .solvers import exhaustive, greedy_descent, qaoa, random_sampling, simulated_annealing

__all__ = ['compare_solvers']


def compare_solvers(problem, solvers=None, seed=0):
    """Solve a :class:`~quantum_mind.computing.RobotProblem` with several solvers.

    Parameters
    ----------
    problem : RobotProblem
    solvers : dict, optional
        ``{name: callable(qubo) -> SolverResult}``. Default: exhaustive (if at most 16 variables), simulated
        annealing, greedy descent and, when Qiskit is installed and there are at most 14 variables, QAOA with depth 2.
    seed : int, optional

    Returns
    -------
    list of dict
        One row per solver with ``solver``, ``feasible``, ``objective`` (the true objective, ``None`` if infeasible),
        ``gap`` (objective minus the exact optimum, when the optimum is known), ``energy`` and the solver's ``info``.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_mind.computing import multi_robot_allocation, compare_solvers
    >>> rows = compare_solvers(multi_robot_allocation(np.array([[1.0, 5.0], [4.0, 2.0]])))
    >>> [(r['solver'], r['gap']) for r in rows if r['solver'] == 'exhaustive']
    [('exhaustive', 0.0)]
    """
    q = problem.qubo
    if solvers is None:
        solvers = {}
        if q.n <= 16:
            solvers['exhaustive'] = exhaustive
        solvers['simulated annealing'] = lambda qb: simulated_annealing(qb, seed=seed)
        solvers['greedy descent'] = lambda qb: greedy_descent(qb, seed=seed)
        solvers['random sampling (1000)'] = lambda qb: random_sampling(qb, 1000, seed=seed)
        if q.n <= 14:
            try:
                import qiskit  # noqa: F401
                solvers['QAOA (Qiskit, p=2)'] = lambda qb: qaoa(qb, depth=2, seed=seed)
            except ImportError:
                pass
    optimum = problem.optimum()[1] if q.n <= 16 else None
    rows = []
    for name, solve in solvers.items():
        res = solve(q)
        ok = problem.feasible(res.x)
        obj = problem.objective(res.x) if ok else None
        rows.append(dict(solver=name, feasible=ok, objective=obj,
                         gap=(obj - optimum) if (ok and optimum is not None) else None,
                         energy=res.energy, info=res.info))
    return rows
