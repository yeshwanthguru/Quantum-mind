"""Portfolio selection with a cardinality budget
=============================================

Finance (simulated): choose 3 of 8 assets, trading expected return against risk.

Returns and covariances are simulated. The portfolio QUBO is solved exactly and by QAOA, simulated
quantum annealing, the quantum-inspired evolutionary algorithm and classical simulated annealing.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/qi.png'
import numpy as np
from quantum_mind.problems import portfolio
from quantum_mind.quantum import QAOA
from quantum_mind.inspired import SQA, QIEA, simulated_annealing

# %%
# Simulated returns and covariances
# ---------------------------------
rng = np.random.default_rng(7)
mu = rng.uniform(0.04, 0.16, 8); A = rng.normal(size=(8, 8)); cov = A @ A.T / 60
q = portfolio(mu, cov, budget=3, risk=0.5)
x, e = q.brute_force()
print('exact: assets %s  return %.3f  risk %.4f' % (np.nonzero(x)[0], mu @ x, x @ cov @ x))

# %%
# Four solvers on the same QUBO
# -----------------------------
for name, r in (('QAOA p=2', QAOA(q, p=2, restarts=3).run()), ('SQA', SQA(q).run()), ('QIEA', QIEA(q).run()),
                ('simulated annealing', simulated_annealing(q))):
    xs = r.x
    print('%-20s assets %s  energy %.4f  (optimum %.4f)' % (name, np.nonzero(xs)[0], q.energy(xs), e))
