"""Question-order effects in a survey
==================================

Survey research / market research: question-order effects.

1. Fits the quantum-like, anchoring and Bayesian models to the four published Clinton-Gore rates.
2. Simulates a split-ballot survey (both orders) from the fitted quantum-like model and runs the QQ
   test and a BIC model comparison, as one would on real survey counts.

Simulated counts are labelled as such; only the four rates are human data.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ql.png'
import numpy as np
from scipy.optimize import least_squares
from quantum_mind.core import compare
from quantum_mind.data import CLINTON_GORE
from quantum_mind.families.order_effects import (QuantumOrderModel, QuantumOrderModel4D, AnchoringOrderModel, BayesOrderModel, RANK_STRUCTURES,
                                          rates, qq_test)

# %%
# Published data
# --------------
# The four "yes" rates of the Gallup poll: each question asked first and asked second.
r = CLINTON_GORE['rates']
target = np.array([r['A_first'], r['B_first'], r['A_second'], r['B_second']])
print('Observed (Moore 2002):', target)

# %%
# Fit the models to the four rates
# --------------------------------
# The 3D quantum-like model is fitted by least squares from 30 random starts for each combination of
# subspace ranks; the anchoring model is fitted directly. The Bayesian model has one joint
# distribution for both orders, so it cannot produce an order effect at all.
best = None
for st in RANK_STRUCTURES:
    for s in range(30):
        res = least_squares(lambda x: rates(QuantumOrderModel(a=x[0], b=x[1], g=x[2], **st).predict()) - target,
                            np.random.default_rng(s).uniform(0, np.pi, 3))
        if best is None or res.cost < best[0].cost: best = (res, st)
ql = QuantumOrderModel(a=best[0].x[0], b=best[0].x[1], g=best[0].x[2], **best[1])
anc = least_squares(lambda y: rates(AnchoringOrderModel(pA=y[0], pB=y[1], wA=y[2], wB=y[3]).predict()) - target,
                    [0.5, 0.6, 0.2, 0.2], bounds=([0.01, 0.01, -0.99, -0.99], [0.99, 0.99, 0.99, 0.99])).x
print('Quantum-like fit  ', rates(ql.predict()).round(3), 'ranks', best[1]['ranks'])
print('Anchoring fit     ', rates(AnchoringOrderModel(pA=anc[0], pB=anc[1], wA=anc[2], wB=anc[3]).predict()).round(3))
print('Bayes (no order effect) cannot change a rate with its position.')

# %%
# Test the models on a simulated survey
# -------------------------------------
# Counts are simulated from the fitted quantum-like model (500 people per order). The QQ test checks
# the parameter-free prediction of every projective model, and BIC ranks the candidate models.
print('\nSimulated split-ballot survey, 500 respondents per order (synthetic data):')
counts = ql.sample(None, 500, np.random.default_rng(1))
print('counts', {k: v.tolist() for k, v in counts.items()})
q, z, p = qq_test(counts); print('QQ test: q = %.3f, z = %.2f, p = %.3f' % (q, z, p))
for fr in compare([QuantumOrderModel4D, (QuantumOrderModel, {"structures": RANK_STRUCTURES}), AnchoringOrderModel, BayesOrderModel], counts, restarts=12):
    print('  %-22s BIC %.1f' % (type(fr.model).__name__, fr.bic))
