"""VQE for a transverse-field Ising chain
======================================

Physics and materials: ground-state energy of a transverse-field Ising chain with VQE.

H = -J sum Z_i Z_{i+1} - h sum X_i on 4 spins, for several field strengths; VQE energies are compared
with exact diagonalisation.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/q.png'
from qlcog.quantum import Hamiltonian, VQE

# %%
# Ground-state energies for three field strengths
# -----------------------------------------------
# The Hamiltonian is built from Pauli strings; VQE uses a real RY ansatz with a CX chain.
n = 4
for h in (0.5, 1.0, 1.5):
    terms = [(-1.0, ''.join('Z' if q in (i, i + 1) else 'I' for q in reversed(range(n)))) for i in range(n - 1)]
    terms += [(-h, ''.join('X' if q == i else 'I' for q in reversed(range(n)))) for i in range(n)]
    r = VQE(Hamiltonian(terms), layers=3, restarts=4, rotations=("ry",)).run()      # real ansatz, CX chain
    print('h = %.1f  VQE %.5f  exact %.5f  error %.1e' % (h, r['energy'], r['exact'], r['energy'] - r['exact']))
