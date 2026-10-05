"""Perception and judgement over time: Markov versus quantum walks (Busemeyer, Kvam and Pleskac, 2019).

Confidence is judged at t1 and t2 or only at t2. The Markov walk predicts that the t2 distribution
does not depend on whether t1 was judged; the quantum walk predicts an interference effect.
Data are simulated from a quantum walk and both models are fitted."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.dynamics import MarkovWalk, QuantumWalk

design = {'t2_only': ('single', 1.5), 'joint': ('joint', 0.5, 1.5)}
truth = QuantumWalk(mu=3.0, sigma=3.0)
p = truth.predict(design)
print('P(t2) alone      ', p['t2_only'].round(3))
print('P(t2) after t1   ', p['joint'].reshape(3, 3).sum(0).round(3), '(differs: interference of the first judgement)')
data = truth.sample(design, 400, np.random.default_rng(2))
for fr in compare([QuantumWalk, MarkovWalk], data, design, restarts=6):
    print('  %-12s BIC %.1f params %s' % (type(fr.model).__name__, fr.bic, {k: round(v, 2) for k, v in fr.model.params.items()}))
