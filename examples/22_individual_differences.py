"""Individual differences: pooled versus per-person fits
=====================================================

Surveys, HRI: do people differ? Pooled versus individual-level model comparison (simulated people).

Twelve simulated people answer the robot's two clarification questions in both orders: six follow
the quantum-like model, six follow the anchoring model (illustrative parameters). Fitting the pooled
counts treats them as one population; fitting each person separately tests that assumption.
Result: the pooled fit selects the quantum-like model at every sample size, although half the people
anchor. Individual fits favour the simplest model (Bayes) with 60 or 400 answers per person and
order, because these two populations differ only slightly per answer; with 3,000 answers they assign
all 12 people to the model that generated them. Individual differences can therefore hide behind,
or be mistaken for, a quantum-like pattern in pooled data.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_22.png'
import numpy as np
from quantum_mind.core import compare, compare_individuals
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel
from quantum_mind.applications.robotics import domain_models

# %%
# Twelve simulated people at three sample sizes
# ---------------------------------------------
# Six follow the quantum-like model and six the anchoring model.
MODELS = [QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]
gen = domain_models('object_clarification')
for answers in (60, 400, 3000):
    rng = np.random.default_rng(1)
    people = {('QL-%d' % i if i < 6 else 'AN-%d' % i): (gen['QL'] if i < 6 else gen['Anchoring']).sample(None, answers, rng)
              for i in range(12)}
    pooled = {c: sum(d[c] for d in people.values()) for c in ('AB', 'BA')}
    best_pooled = type(compare(MODELS, pooled, rng=rng)[0].model).__name__
    r = compare_individuals(MODELS, people, rng=rng)
    correct = sum(1 for p, f in ((p, min(r['fits'], key=lambda m: r['fits'][m].results[p].bic)) for p in people)
                  if (p.startswith('QL') and f == 'QuantumOrderModel4D') or (p.startswith('AN') and f == 'AnchoringOrderModel'))
    print('%d answers per person and order' % answers)
    print('  pooled fit selects:            %s' % best_pooled)
    print('  best model per person:         %s' % r['best_counts'])
    print('  people assigned their own model: %d of 12' % correct)
    print('  group BIC (individual fits):   %s' % [(m, round(b, 1)) for m, b in r['group']])
