"""Consumer and economic choice: quantum decision theory against expected utility and prospect theory.

Choice problems between a sure amount and a gamble (gains and losses). Choices are simulated from a
QDT decision maker (synthetic data), then all three models are fitted and compared."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.decision import QDTModel, ExpectedUtilityModel, ProspectTheoryModel

problems = {
    'gain_small': ([(30, 1.0)], [(45, 0.8), (0, 0.2)]),
    'gain_large': ([(300, 1.0)], [(450, 0.8), (0, 0.2)]),
    'gain_long':  ([(5, 1.0)], [(500, 0.01), (0, 0.99)]),
    'loss_small': ([(-30, 1.0)], [(-45, 0.8), (0, 0.2)]),
    'mixed':      ([(0, 1.0)], [(100, 0.5), (-80, 0.5)]),
}
truth = QDTModel(alpha=0.85, beta=0.08, q0=0.2)
data = truth.sample(problems, 150, np.random.default_rng(5))
print('Simulated choices of option 1 (synthetic, 150 per problem):', {k: int(v[0]) for k, v in data.items()})
for fr in compare([QDTModel, ExpectedUtilityModel, ProspectTheoryModel], data, problems, restarts=10):
    print('  %-22s k=%d BIC %.1f  params %s' % (type(fr.model).__name__, fr.k, fr.bic, {k: round(v, 3) for k, v in fr.model.params.items()}))
