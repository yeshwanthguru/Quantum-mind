"""A robot that asks questions and tracks trust
============================================

Robotics: a robot that asks people questions and tracks their trust.

1. Questioning design: how should the robot learn people's unprimed answers?
2. Human-model ensemble: predicted answers and uncertainty for the robot's orchestrator.
3. Trust: does the robot's question about trust change trust?

All data are simulated from the illustrative domain populations.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/robot.png'
import numpy as np
from quantum_mind.applications.robotics import (domain_models, estimate_unprimed_rates, HumanModelEnsemble, TRUST_MODELS, TRUST_EVENTS)
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel
from quantum_mind.families.dynamics import question_effect

# %%
# 1. Questioning designs
# ----------------------
# How well does each design recover the rate of "yes" to B when B is asked first? Populations are
# simulated from the quantum-like and the anchoring model.
rng = np.random.default_rng(0)
models = domain_models('object_clarification')
for gen_name in ('QL', 'Anchoring'):
    gen = models[gen_name]; truth = gen.predict()['BA'][:2].sum()
    for design, model in (('fixed', None), ('split', None), ('probe', QuantumOrderModel4D)):
        est = [estimate_unprimed_rates(design, gen, 400, rng, model=model)[1] for _ in range(15)]
        print('%-9s people, %-5s design: RMSE of pi_B = %.3f' % (gen_name, design, np.sqrt(np.mean((np.array(est) - truth) ** 2))))

# %%
# 2. Human-model ensemble
# -----------------------
# The robot fits competing human models, weights them by BIC and reports a prediction with its
# uncertainty.
data = models['QL'].sample(None, 300, rng)
ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]).update(data)
print('\nEnsemble weights:', [(s['model'], round(s['weight'], 3)) for s in ens.summary()])
p, unc = ens.predict('AB'); print('Predicted answers (A first):', p.round(3), 'uncertainty:', {k: round(v, 3) for k, v in unc.items()})

# %%
# 3. Trust
# --------
# Change in final trust caused by also asking about trust halfway.
print('\nTrust question effect, open-system: %.3f' % question_effect(TRUST_MODELS['open_system'], TRUST_EVENTS, 5, 2))
