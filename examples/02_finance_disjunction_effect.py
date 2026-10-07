"""Disjunction effect in a two-stage gamble
========================================

Behavioural finance / gambling: the disjunction effect in a two-stage gamble.

Published data (Tversky and Shafir, 1992; 98 students): play again after a win 0.69, after a loss 0.59,
when the outcome is unknown 0.36. Classical probability requires 0.59 <= P(unknown) <= 0.69.
Compares the quantum-like interference model with the classical mixture, and shows the same data as a
two-node quantum-like Bayesian network.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_02.png'
import numpy as np
from quantum_mind.core import compare
from quantum_mind.data import TWO_STAGE_GAMBLE, proportions_to_counts
from quantum_mind.families.interference import InterferenceModel, ClassicalMixtureModel, total_probability_bounds
from quantum_mind.families.qlbn import BayesNet, classical_marginal, quantum_like_marginal

# %%
# Observed proportions and the classical range
# --------------------------------------------
# People play the second gamble less often when the first outcome is unknown than after either known
# outcome, which violates the law of total probability.
c = TWO_STAGE_GAMBLE['conditions']
print('Observed:', c, '| classical range for unknown:', total_probability_bounds(c['known_1'], c['known_2']))
counts = proportions_to_counts(c, TWO_STAGE_GAMBLE['n'])
for fr in compare([InterferenceModel, ClassicalMixtureModel], counts, restarts=20):
    pred = {k: round(float(v[0]), 3) for k, v in fr.model.predict().items()}
    print('  %-22s k=%d  BIC %.1f  predicted %s' % (type(fr.model).__name__, fr.k, fr.bic, pred))
print('Note: the interference model has 4 parameters for 3 proportions, so it fits exactly; the comparison')
print('shows that it can represent the effect, not that it predicts it.')

# %%
# The same effect as a quantum-like Bayesian network
# --------------------------------------------------
# The unknown first outcome is a hidden variable. Summing amplitudes instead of probabilities adds an
# interference term whose size depends on the phase.
net = BayesNet({'First': ['win', 'lose'], 'Play': ['yes', 'no']}, {'Play': ['First']},
               {'First': {(): {'win': 0.5, 'lose': 0.5}},
                'Play': {('win',): {'yes': c['known_1'], 'no': 1 - c['known_1']}, ('lose',): {'yes': c['known_2'], 'no': 1 - c['known_2']}}})
print('\nQuantum-like Bayesian network, P(play) when the first outcome is unknown:')
print('  classical', round(classical_marginal(net, 'Play')['yes'], 3))
for th in (0.0, np.pi / 2, 2.5, 3.0):
    print('  phase %.2f -> %.3f' % (th, quantum_like_marginal(net, 'Play', None, [0, th])['yes']))
