"""Fusing a detector, speech and gaze
===================================

Robotics, perception: which object does the person mean? A detector's scores, a speech recogniser's
n-best list and a gaze direction become likelihoods over the same objects, and are fused with Bayes'
rule, Dempster-Shafer and a quantum-like rule whose cues can be incompatible. The three rules are then
compared on held-out simulated choices.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_33.png'
import numpy as np
from quantum_mind.applications.fusion import (detector_likelihood, asr_likelihood, direction_likelihood, bayes_fusion,
                                              dempster_shafer_fusion, compare_fusion)
from quantum_mind.applications.intent import QuantumIntentResolver

# %%
# One request, three cues
# -----------------------
objects = ['red cup', 'blue cup', 'bowl']
cues = [detector_likelihood({'red cup': 0.62, 'blue cup': 0.55, 'bowl': 0.20}, objects, temperature=1.5),
        asr_likelihood([('the blue one', 0.7)], {'red cup': ['red'], 'blue cup': ['blue'], 'bowl': ['bowl']}, objects),
        direction_likelihood([0.9, 0.3], {'red cup': [-1, 0], 'blue cup': [1, 0.2], 'bowl': [0.2, 1]}, objects)]
print('Bayes          ', bayes_fusion(cues).round(3))
ds, ign = dempster_shafer_fusion(cues, [0.8, 0.9, 0.6])
print('Dempster-Shafer', ds.round(3), 'ignorance %.3f' % ign)

# %%
# Held-out comparison on simulated, order-dependent choices
# ---------------------------------------------------------
intents, L = ['a', 'b', 'c'], {'x': [0.7, 0.2, 0.1], 'y': [0.25, 0.6, 0.15]}
truth = QuantumIntentResolver(intents)
truth.add_cue('x', L['x'], 0.7)
truth.add_cue('y', L['y'], 0.7)
rng = np.random.default_rng(1)
orders = [['x', 'y'], ['y', 'x']]
trials = [(o, intents[rng.choice(3, p=truth.posterior(o))]) for o in [orders[i % 2] for i in range(1500)]]
for rule, r in compare_fusion(intents, L, trials[:1000], trials[1000:]).items():
    if isinstance(r, dict) and 'log_loss' in r:
        print('%-16s held-out log loss %.3f' % (rule, r['log_loss']))
print('The simulated people follow the quantum-like rule, so its advantage here is expected.')
