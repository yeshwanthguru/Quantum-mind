"""Resolving an ambiguous command
==============================

Robotics / assistants: resolving an ambiguous command from context cues.

"Bring me the cup": a word cue, a pointing gesture and the user's gaze each carry a likelihood over
three objects (illustrative values). Bayesian updating is order-free; the quantum-like resolver with
incompatible cues (theta > 0) depends on the order in which the cues arrive, and reduces to Bayes when
every theta = 0. The robot asks for clarification when the posterior is still uncertain.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/robot.png'
from quantum_mind.applications.intent import QuantumIntentResolver, BayesIntentResolver
from quantum_mind.applications.robotics import ask_or_act

# %%
# Cues and their likelihoods
# --------------------------
# Each cue has a likelihood over the three objects and an incompatibility angle (0 for the word cue).
objects = ['red cup', 'blue cup', 'bowl']
cues = {'word "cup"': ([0.50, 0.40, 0.10], 0.0), 'points left': ([0.70, 0.20, 0.10], 0.5), 'looks right': ([0.25, 0.60, 0.15], 0.5)}
for cls in (BayesIntentResolver, QuantumIntentResolver):
    r = cls(objects)
    for name, (L, th) in cues.items():
        r.add_cue(name, L, th)
    a = r.posterior(['word "cup"', 'points left', 'looks right']); b = r.posterior(['looks right', 'points left', 'word "cup"'])
    print('%-22s order 1 %s  order 2 %s  order effect %.3f' % (cls.__name__, a.round(3), b.round(3),
                                                                   r.order_effect('points left', 'looks right')))

# %%
# Ask or act?
# -----------
# After two cues the robot checks whether the most likely object is certain enough to act on.
q = QuantumIntentResolver(objects)
for name, (L, th) in cues.items():
    q.add_cue(name, L, th)
intent, p, H = q.resolve(['word "cup"', 'points left'])
print('after "cup" + pointing: %s (P = %.2f, entropy %.2f bits) -> %s' % (intent, p, H, ask_or_act(p, ask_cost=1, error_cost=4)['action']))
