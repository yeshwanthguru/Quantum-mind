"""Assignment slot: with several motors, which motor goes to which target? (quantum computing strand)

The VLA grounds an instruction to target positions; with several joints or arms the choice of who goes where is the multi-robot
allocation QUBO of :mod:`quantum_mind.computing`, solved by any of its solvers (exact, annealing, greedy, or Qiskit QAOA).
Cost is travel time, taken as the distance to the target. Assigning every target to exactly one motor allows several targets per motor,
which is what that problem encodes.
"""
from __future__ import annotations

import numpy as np

from ..computing import exhaustive, greedy_descent, multi_robot_allocation, qaoa, simulated_annealing

SOLVERS = {'exact': exhaustive, 'annealing': simulated_annealing, 'greedy': greedy_descent, 'qaoa': qaoa}


def assign_targets(motor_positions, target_positions, solver='exact', balance=0.5, **kw):
    """Assign each target to a motor. Returns ``(mapping, total_travel, problem)``; ``mapping[target] = motor``.

    ``balance`` penalises piling several targets on one motor. ``solver`` is a key of :data:`SOLVERS`; extra keywords go to the solver.
    """
    m, t = np.asarray(motor_positions, float), np.asarray(target_positions, float)
    costs = np.abs(m[:, None] - t[None, :])
    problem = multi_robot_allocation(costs, balance=balance)
    res = SOLVERS[solver](problem.qubo, **kw)
    x = res.x if hasattr(res, 'x') else res[0]
    grid = np.asarray(x).reshape(len(m), len(t))
    mapping = {int(j): int(np.argmax(grid[:, j])) for j in range(len(t))}
    return mapping, float(sum(costs[i, j] for j, i in mapping.items())), problem
