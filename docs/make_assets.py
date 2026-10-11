"""Regenerate the images and animations used in the README and docs (python3 docs/make_assets.py).
All figures come from the package itself; nothing is drawn by hand except the SVG logo, wordmark and banner
(docs/assets/logo.svg, logo-wordmark.svg, banner.svg) and the social card rendered from them. The
interactive HTML versions are not committed (they are large); examples/11 and 12 write them to
examples/output/."""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np                                                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'docs' / 'assets'; OUT.mkdir(parents=True, exist_ok=True)

from qiskit import QuantumCircuit                                 # noqa: E402
from quantum_mind.viz import circuit_trajectory, belief_trajectory, animate_trajectory  # noqa: E402
from quantum_mind.families.dynamics import OpenSystemBelief              # noqa: E402

BG, FG, GRID = '#0d1117', '#e6edf3', '#30363d'
PAL = ['#ff7b72', '#79c0ff', '#d2a8ff', '#7ee787', '#ffa657']


def dark(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color(GRID) for s in ax.spines.values()]
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG); ax.title.set_color(FG); ax.grid(color=GRID, lw=0.5)


# 1. Entanglement on the Bloch sphere: H, rotation, CNOT (vectors shrink into the ball), then undo
qc = QuantumCircuit(2)
qc.h(0); qc.ry(np.pi / 3, 1); qc.cx(0, 1); qc.rz(np.pi / 2, 0); qc.cx(0, 1); qc.h(0)
tr = circuit_trajectory(qc, steps=9)
tr.names = ['qubit 0', 'qubit 1']; tr.title = 'H · RY(π/3) · CNOT · RZ(π/2) · CNOT · H'
animate_trajectory(tr, save=OUT / 'entanglement.gif', fps=16, rotate=0.9, dpi=64)

# 2. Trust belief of a person watching a robot (quantum_mind OpenSystemBelief, simulated parameters)
m = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
bt = belief_trajectory(m, (1, 1, 0, 1, 0, 1), steps=6)
bt.title = 'Trust belief (simulated person)'
animate_trajectory(bt, save=OUT / 'trust_belief.gif', fps=12, rotate=0.6, dpi=64, size=4.2)

# 3. The COBRA figures (context effect, belief timeline, density matrix, benchmark) come from docs/make_cobra_figures.py
