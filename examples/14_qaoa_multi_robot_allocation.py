"""Multi-robot task allocation: QAOA, SQA, QIEA, SA
================================================

Robotics: allocate tasks to robots with quantum, quantum-inspired and classical solvers.

Three robots and three tasks with travel costs (illustrative numbers). Every task must go to one
robot. The same QUBO is solved by brute force, QAOA (quantum), simulated quantum annealing and the
quantum-inspired evolutionary algorithm (quantum-inspired), and simulated annealing (classical). The
QAOA circuit is then sampled on Aer, and on Aer with the IBM FakeTorino noise model.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/robot.png'
import numpy as np
from qlcog.problems import task_allocation
from qlcog.quantum import QAOA
from qlcog.inspired import SQA, QIEA, simulated_annealing

# %%
# The allocation problem as a QUBO
# --------------------------------
# One binary variable per (robot, task); a penalty enforces one robot per task.
costs = np.array([[2.0, 7.0, 4.0],      # robot 0 -> tasks 0, 1, 2
                  [6.0, 3.0, 5.0],      # robot 1
                  [5.0, 6.0, 2.5]])     # robot 2
q = task_allocation(costs)
x, e = q.brute_force(); print('exact optimum   energy %.2f  assignment\n%s' % (e, x.reshape(3, 3)))

# %%
# Quantum, quantum-inspired and classical solvers
# -----------------------------------------------
qa = QAOA(q, p=3, restarts=3).run()
print('QAOA p=3        energy %.2f  P(optimal) = %.3f (uniform %.4f)' % (qa.energy, qa.p_optimal, 1 / 2 ** q.n))
for name, r in (('SQA', SQA(q).run()), ('QIEA', QIEA(q).run()), ('simulated annealing', simulated_annealing(q))):
    print('%-15s energy %.2f' % (name, r.value))

# %%
# Sample the QAOA circuit
# -----------------------
# The optimised circuit on ideal Aer and under the FakeTorino noise model.
try:
    from qlcog.circuits import run
    qc = QAOA(q, p=3).to_qiskit(qa)
    for be in ('aer', 'aer:FakeTorino'):
        counts = run(qc, be, 4000)
        best = min(counts, key=lambda b: q.energy([int(c) for c in b.replace(' ', '')[::-1]]))
        good = sum(k for b, k in counts.items() if np.isclose(q.energy([int(c) for c in b.replace(' ', '')[::-1]]), e)) / 4000
        print('%-15s sampled P(optimal) = %.3f; best sample energy %.2f' % (be, good, q.energy([int(c) for c in best[::-1]])))
except ImportError:
    print('install qiskit-aer and qiskit-ibm-runtime to sample the circuit')
