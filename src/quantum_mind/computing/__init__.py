"""Quantum computing for robot algorithms: robot problems as QUBOs, classical baselines and a Qiskit QAOA adapter.

This is the first of the three threads of quantum robotics that Quantum Mind covers (the others are quantum sensing,
:mod:`quantum_mind.sensing`, and quantum-probability beliefs about people, :mod:`quantum_mind.belief`). It does not
re-implement quantum algorithms: Qiskit and PennyLane do that better. It supplies the robot side: realistic problems
with their true objectives, the classical baselines every quantum result has to be reported against, and a thin adapter
to Qiskit's QAOA, so a claim such as "QAOA solves multi-robot allocation" can be checked on equal terms.

* :func:`multi_robot_allocation`, :func:`grasp_selection`, :func:`slot_scheduling`: the problems.
* :func:`exhaustive`, :func:`simulated_annealing`, :func:`greedy_descent`, :func:`random_sampling`: classical baselines
  (random sampling with the same number of shots is the null baseline for any sampling-based quantum solver).
* :func:`qaoa`: QAOA through Qiskit's ``QAOAAnsatz`` on an exact simulator.
* :func:`compare_solvers`: run several solvers on one problem and report them in robot terms.
"""
from .problems import RobotProblem, grasp_selection, multi_robot_allocation, slot_scheduling
from .solvers import SolverResult, exhaustive, greedy_descent, qaoa, random_sampling, simulated_annealing
from .compare import compare_solvers

__all__ = ['RobotProblem', 'grasp_selection', 'multi_robot_allocation', 'slot_scheduling', 'SolverResult',
           'exhaustive', 'greedy_descent', 'qaoa', 'random_sampling', 'simulated_annealing', 'compare_solvers']
