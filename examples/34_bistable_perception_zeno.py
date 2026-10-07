"""Bistable perception and the quantum Zeno effect
===============================================

Perception: the quantum Zeno model predicts that the mean dwell time in one interpretation of an
ambiguous figure grows as the percept is checked more often. The example simulates dwell times at two
check intervals and compares the Zeno model with Markov switching and gamma renewal by BIC.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_34.png'
import numpy as np
from quantum_mind.core import compare
from quantum_mind.families.perception import (QuantumZenoBistableModel, MarkovSwitchingModel, GammaRenewalModel,
                                              dwell_time_design, mean_dwell_time)

zeno = QuantumZenoBistableModel(g=1.5)
for dt in (0.035, 0.07, 0.14):
    print('check interval %.3f s -> mean dwell time %.2f s' % (dt, mean_dwell_time(zeno, dt)))

# %%
# Two check intervals identify the model (simulated dwell times)
# --------------------------------------------------------------
design = dwell_time_design({'fast': 0.035, 'slow': 0.14}, max_time=40, n_bins=25)
data = zeno.sample(design, 300, np.random.default_rng(0))
for r in compare([QuantumZenoBistableModel, MarkovSwitchingModel, GammaRenewalModel], data, design):
    print('  %-26s BIC %.1f' % (type(r.model).__name__, r.bic))
