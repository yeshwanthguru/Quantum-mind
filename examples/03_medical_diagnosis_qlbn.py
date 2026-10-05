"""Medical decision support: classical versus quantum-like inference in a Bayesian network.

A synthetic network (not clinical data): Disease -> TestA, Disease -> TestB, with a hidden Context
variable that affects TestB. The query is P(Disease | TestA positive) with the other variables
unobserved. The quantum-like network adds interference between the hidden configurations; phases
can be fitted to observed judgements (here: judgements simulated from chosen phases)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.qlbn import BayesNet, classical_marginal, quantum_like_marginal, for_network, ClassicalBNModel

net = BayesNet(
    {'Disease': ['yes', 'no'], 'Context': ['clinic', 'home'], 'TestA': ['pos', 'neg'], 'TestB': ['pos', 'neg']},
    {'TestA': ['Disease'], 'TestB': ['Disease', 'Context']},
    {'Disease': {(): {'yes': 0.1, 'no': 0.9}}, 'Context': {(): {'clinic': 0.6, 'home': 0.4}},
     'TestA': {('yes',): {'pos': 0.9, 'neg': 0.1}, ('no',): {'pos': 0.2, 'neg': 0.8}},
     'TestB': {('yes', 'clinic'): {'pos': 0.85, 'neg': 0.15}, ('yes', 'home'): {'pos': 0.6, 'neg': 0.4},
               ('no', 'clinic'): {'pos': 0.1, 'neg': 0.9}, ('no', 'home'): {'pos': 0.3, 'neg': 0.7}}})
ev = {'TestA': 'pos'}
print('Classical P(Disease | TestA+):', round(classical_marginal(net, 'Disease', ev)['yes'], 3))
for ph in ([0, 0, 0, 0], [0, 1.0, 2.0, 3.0], [0, np.pi, 0, np.pi]):
    print('Quantum-like, phases', ph, '->', round(quantum_like_marginal(net, 'Disease', ev, ph)['yes'], 3))

conds = {'A+': {'TestA': 'pos'}, 'A-': {'TestA': 'neg'}, 'B+': {'TestB': 'pos'}}
QL = for_network(net, 'Disease', conds)
truth = QL(net=net, query='Disease', conditions=conds, **{p.name: v for p, v in zip(QL.PARAMS, [1.2, 2.0, 0.4])})
data = truth.sample(None, 300, np.random.default_rng(3))
print('\nSimulated diagnostic judgements (synthetic), 300 per condition:', {k: v.tolist() for k, v in data.items()})
for fr in compare([(QL, dict(net=net, query='Disease', conditions=conds)),
                   (ClassicalBNModel, dict(net=net, query='Disease', conditions=conds))], data, restarts=10):
    print('  %-18s k=%d BIC %.1f' % (type(fr.model).__name__, fr.k, fr.bic))
