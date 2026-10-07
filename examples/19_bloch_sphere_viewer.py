"""Watching qubits on the Bloch sphere
===================================

Visualisation: watch qubits move on the Bloch sphere.

1. A circuit that entangles two qubits, gate by gate: the vectors leave the sphere surface and move
   inside the ball while the qubits are entangled, then return.
2. Interactive HTML (rotate, zoom, play, slider) and an animated GIF.
3. The same final state measured by tomography on Aer and on the FakeTorino noise model.
4. --live: real-time animation in a Matplotlib window (or in Jupyter: LiveBloch(2).show() then .play()).

Outputs are written to examples/output/.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_19.png'
import pathlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantum_mind.viz import circuit_trajectory, bloch_vectors, bloch_tomography

# %%
# A circuit, gate by gate
# -----------------------
# While the qubits are entangled their Bloch vectors are shorter than 1.
out = pathlib.Path(sys.argv[0]).resolve().parent / 'output'; out.mkdir(exist_ok=True)
qc = QuantumCircuit(2)
qc.h(0); qc.ry(np.pi / 3, 1); qc.cx(0, 1); qc.rz(np.pi / 2, 0); qc.cx(0, 1); qc.h(0)
traj = circuit_trajectory(qc, steps=12); traj.names = ['qubit 0', 'qubit 1']
print('%d frames; length of the Bloch vectors (1 = pure, < 1 = entangled):' % traj.frames)
for k in range(0, traj.frames, 12):
    print('  after %-5s %s' % (traj.labels[k], traj.purity()[k].round(3)))

# %%
# Interactive HTML, GIF and a live window
# ---------------------------------------
try:
    from quantum_mind.viz import animate_bloch, save_html
    save_html(animate_bloch(traj, title='Entangling two qubits'), out / 'bloch_circuit.html')
    print('interactive figure:', out / 'bloch_circuit.html')
except ImportError:
    print('install plotly for the interactive figure')
if '--gif' in sys.argv or '--live' in sys.argv:
    import matplotlib
    if '--live' not in sys.argv:
        matplotlib.use('Agg')
    from quantum_mind.viz import animate_trajectory, LiveBloch
    if '--gif' in sys.argv:
        animate_trajectory(traj, save=out / 'bloch_circuit.gif', fps=20); print('animation:', out / 'bloch_circuit.gif')
    if '--live' in sys.argv:
        LiveBloch(2, traj.names, backend='matplotlib').show().play(traj, fps=30)

# %%
# Tomography on simulators
# ------------------------
# Bloch vectors estimated from X, Y and Z measurements, ideal and under device noise.
ideal = bloch_vectors(Statevector(qc))
for be in ('aer', 'aer:FakeTorino'):
    try:
        est = bloch_tomography(qc, be, shots=8000)
        print('%-15s measured Bloch vectors %s  (ideal %s)' % (be, est.round(2).tolist(), ideal.round(2).tolist()))
    except ImportError:
        print('install qiskit-aer and qiskit-ibm-runtime for tomography')
