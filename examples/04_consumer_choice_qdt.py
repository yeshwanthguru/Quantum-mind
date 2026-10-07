"""Consumer choice with quantum decision theory
============================================

Consumer and economic choice: quantum decision theory against expected utility and prospect theory.

Choice problems between a sure amount and a gamble (gains and losses). Choices are simulated from a
QDT decision maker (synthetic data), then all three models are fitted and compared.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_04.png'
import numpy as np
from quantum_mind.core import compare
from quantum_mind.families.decision import QDTModel, ExpectedUtilityModel, ProspectTheoryModel

# %%
# Choice problems
# ---------------
# Each problem is a pair of lotteries, given as (outcome, probability) pairs.
problems = {
    'gain_small': ([(30, 1.0)], [(45, 0.8), (0, 0.2)]),
    'gain_large': ([(300, 1.0)], [(450, 0.8), (0, 0.2)]),
    'gain_long':  ([(5, 1.0)], [(500, 0.01), (0, 0.99)]),
    'loss_small': ([(-30, 1.0)], [(-45, 0.8), (0, 0.2)]),
    'mixed':      ([(0, 1.0)], [(100, 0.5), (-80, 0.5)]),
}

# %%
# Simulate choices and compare models
# -----------------------------------
# Choices are simulated from quantum decision theory; QDT, expected utility and prospect theory are
# fitted to them and ranked by BIC.
truth = QDTModel(alpha=0.85, beta=0.08, q0=0.2)
data = truth.sample(problems, 150, np.random.default_rng(5))
print('Simulated choices of option 1 (synthetic, 150 per problem):', {k: int(v[0]) for k, v in data.items()})
for fr in compare([QDTModel, ExpectedUtilityModel, ProspectTheoryModel], data, problems, restarts=10):
    print('  %-22s k=%d BIC %.1f  params %s' % (type(fr.model).__name__, fr.k, fr.bic, {k: round(v, 3) for k, v in fr.model.params.items()}))
