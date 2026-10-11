"""Robot problems on quantum and classical solvers
=================================================

Quantum computing for robot algorithms: a multi-robot allocation problem is encoded as a QUBO and solved by exhaustive search,
greedy descent, simulated annealing, random sampling with the same number of shots, and QAOA through Qiskit on an exact
simulator. The table reports each solver in the robot's own terms (true objective and gap to the exact optimum). The data are
**simulated**; QAOA is a simulation, not hardware.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_18.png'
import warnings

import numpy as np

from quantum_mind.computing import compare_solvers, multi_robot_allocation

warnings.simplefilter('ignore')
rng = np.random.default_rng(7)
costs = rng.uniform(1, 9, (3, 3))                       # costs[robot, task], simulated
problem = multi_robot_allocation(costs, balance=0.3)
print('cost matrix (robot x task):\n', costs.round(2))
print('exact optimum:', problem.decode(problem.optimum()[0]), 'objective %.3f' % problem.optimum()[1])

# %%
# Compare the solvers
# -------------------
rows = compare_solvers(problem, seed=1)
print('%-26s %-9s %-10s %s' % ('solver', 'feasible', 'objective', 'gap to optimum'))
for r in rows:
    obj = '%.3f' % r['objective'] if r['objective'] is not None else '-'
    gap = '%.3f' % r['gap'] if r['gap'] is not None else '-'
    print('%-26s %-9s %-10s %s' % (r['solver'], r['feasible'], obj, gap))
    if 'p_optimal' in r['info']:
        print('    QAOA: P(optimal) = %.3f (uniform guess %.4f), approximation ratio %.3f'
              % (r['info']['p_optimal'], 2.0 ** -problem.n_variables, r['info']['ratio']))
