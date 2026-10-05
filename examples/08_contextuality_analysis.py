"""Contextuality analysis for any set of binary measurements (physics, psychology, survey items).

1. The CHSH value of a maximally entangled qubit pair, computed and simulated with the Qiskit circuit.
2. A cyclic rank-4 system tested with the Contextuality-by-Default criterion, with and without
   inconsistent connectedness (toy numbers, not data)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.families.contextuality import chsh, qubit_chsh_correlations, cyclic_contextuality
from qlcog.circuits import chsh_circuit, run

E = qubit_chsh_correlations()
print('Quantum prediction: S = %.3f (classical bound 2)' % chsh(**E))
angles = {'E11': (0, np.pi / 4), 'E12': (0, -np.pi / 4), 'E21': (np.pi / 2, np.pi / 4), 'E22': (np.pi / 2, -np.pi / 4)}
sim = {}
for k, (a, b) in angles.items():
    qc, dec = chsh_circuit(a, b); sim[k] = dec(run(qc, 'aer', 20000))
print('Aer simulation:     S = %.3f' % chsh(**sim))
corr = [0.7, 0.7, 0.7, -0.7]
print('Consistent toy system:   ', cyclic_contextuality(corr, [(0, 0)] * 4))
print('Inconsistent toy system: ', cyclic_contextuality(corr, [(0.1, -0.1), (0.2, 0.0), (0.0, 0.05), (0.1, 0.1)]))
