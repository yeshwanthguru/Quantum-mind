"""Every family as a circuit
=========================

Every family as a quantum circuit: compare circuit output with the analytic model on the ideal
Aer simulator, on Aer with the FakeTorino (IBM Heron) noise model, and on the Braket local simulator.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
import numpy as np
from quantum_mind.core import tvd
from quantum_mind.circuits import (run, order_effects_circuit, interference_circuit, qlbn_circuit, walk_circuit, belief_circuit)
from quantum_mind.families.order_effects import QuantumOrderModel
from quantum_mind.families.interference import InterferenceModel
from quantum_mind.families.qlbn import BayesNet, quantum_like_marginal
from quantum_mind.families.dynamics import QuantumWalk, OpenSystemBelief, final_yes

# %%
# Build the circuits
# ------------------
# Each builder returns a circuit and a decoder that turns counts into the quantity the model predicts.
ql = QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2))
cases = []
qc, dec = order_effects_circuit(ql, 'AB', 'deferred'); cases.append(('order effects (AB)', qc, dec, ql.predict()['AB']))
im = InterferenceModel(p1=0.69, p2=0.59, c=0.5, theta=2.2)
qc, dec = interference_circuit(im.p1, im.p2, im.c, im.theta, 'unknown'); cases.append(('interference (unknown)', qc, dec, im.predict()['unknown']))
net = BayesNet({'O': ['w', 'l'], 'P': ['y', 'n']}, {'P': ['O']}, {'O': {(): {'w': .5, 'l': .5}}, 'P': {('w',): {'y': .69, 'n': .31}, ('l',): {'y': .59, 'n': .41}}})
qc, dec = qlbn_circuit(net, 'P', None, [0, 2.2]); m = quantum_like_marginal(net, 'P', None, [0, 2.2]); cases.append(('QLBN', qc, dec, np.array([m['y'], m['n']])))
w = QuantumWalk(mu=3, sigma=3, n_states=16); qc, dec = walk_circuit(w, 1.0); cases.append(('quantum walk', qc, dec, w.predict({'a': ('single', 1.0)})['a']))

# %%
# Run on three back ends
# ----------------------
# Ideal Aer, Aer with the IBM FakeTorino noise model, and the Amazon Braket local simulator. The total
# variation distance (TVD) to the exact prediction measures the sampling and noise error.
for name, qc, dec, exact in cases:
    row = [name]
    for be in ('aer', 'aer:FakeTorino', 'braket_local'):
        try:
            row.append('%s TVD %.3f' % (be, tvd(dec(run(qc, be, 8000)), exact)))
        except ValueError as e:
            row.append('%s n/a (%s)' % (be, str(e)[:40]))
    print(' | '.join(row))

# %%
# Belief dynamics with dephasing
# ------------------------------
# Dephasing is implemented by an ancilla that triggers a Z gate with the right probability.
b = OpenSystemBelief(); qc, dec = belief_circuit(b, (1, 1, 0, 1, 0, 1), (2, 5))
print('belief dynamics: circuit %.3f, exact %.3f' % (dec(run(qc, 'aer', 20000)), final_yes(b, (1, 1, 0, 1, 0, 1), (2, 5))))
