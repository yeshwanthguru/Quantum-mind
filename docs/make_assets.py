"""Regenerate the images and animations used in the README and docs (python3 docs/make_assets.py).
All figures come from the package itself; nothing is drawn by hand except the SVG logo, wordmark and banner
(docs/assets/logo.svg, logo-wordmark.svg, banner.svg) and the social card rendered from them. The
interactive HTML versions are not committed (they are large); examples/19 and 20 write them to
examples/output/."""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'docs' / 'assets'; OUT.mkdir(parents=True, exist_ok=True)

from qiskit import QuantumCircuit                                 # noqa: E402
from quantum_mind.viz import circuit_trajectory, belief_trajectory, animate_trajectory  # noqa: E402
from quantum_mind.families.dynamics import OpenSystemBelief              # noqa: E402
from quantum_mind.problems import maxcut                                 # noqa: E402
from quantum_mind.quantum import QAOA, VariationalClassifier            # noqa: E402
from quantum_mind.inspired import QIEA, SQA, simulated_annealing, MPSClassifier  # noqa: E402

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

# 3. Three pillars on one problem: MaxCut on a 12-node random graph
rng = np.random.default_rng(4)
edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if rng.random() < 0.35]
q = maxcut(edges, 12); opt = q.brute_force()[1]
qa = QAOA(maxcut(edges[:], 12), p=3, restarts=3).run()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG)
for name, r, c in [('QIEA (quantum-inspired)', QIEA(q, generations=150).run(), PAL[2]),
                   ('SQA (quantum-inspired)', SQA(q, sweeps=150).run(), PAL[1]),
                   ('Simulated annealing (classical)', simulated_annealing(q, sweeps=150), PAL[4])]:
    ax[0].plot(r.history, color=c, lw=2, label=name)
ax[0].axhline(opt, color=PAL[3], ls='--', lw=1.2, label='exact optimum')
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('best energy'); ax[0].set_title('Optimisers on MaxCut (12 nodes)')
ax[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
E = q.all_energies(); order = np.argsort(E)
ax[1].bar(range(len(E)), qa.probabilities[order], color=PAL[0], width=1.0)
ax[1].set_xlabel('bit strings sorted by energy (best on the left)'); ax[1].set_ylabel('probability')
ax[1].set_title('QAOA p=3: P(optimal) = %.2f (uniform: %.4f)' % (qa.p_optimal, np.isclose(E, opt).mean()))
ax[1].set_xlim(-20, 600)
for a in ax:
    dark(a)
fig.tight_layout(); fig.savefig(OUT / 'optimisers.png', dpi=130, facecolor=BG)

# 4. Classifiers: decision regions of the variational quantum classifier and the MPS classifier
n = 150; th = rng.uniform(0, np.pi, n)
X = np.r_[np.c_[np.cos(th), np.sin(th)], np.c_[1 - np.cos(th), 0.5 - np.sin(th)]] + rng.normal(0, 0.12, (2 * n, 2))
y = np.r_[np.zeros(n), np.ones(n)]
models = [('Variational quantum classifier', VariationalClassifier(layers=3).fit(X, y)),
          ('Tensor-network (MPS) classifier', MPSClassifier(bond=6, local_dim=4, epochs=80).fit(X, y))]
gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 120), np.linspace(-1.0, 1.5, 90))
G = np.c_[gx.ravel(), gy.ravel()]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG)
for a, (name, mdl) in zip(ax, models):
    P = mdl.predict_proba(G)[:, 1].reshape(gx.shape)
    a.contourf(gx, gy, P, levels=20, cmap='RdBu_r', alpha=0.55)
    a.scatter(*X[y == 0].T, s=9, color=PAL[1]); a.scatter(*X[y == 1].T, s=9, color=PAL[0])
    a.set_title('%s (training accuracy %.2f)' % (name, mdl.score(X, y)), fontsize=10); dark(a)
fig.tight_layout(); fig.savefig(OUT / 'classifiers.png', dpi=130, facecolor=BG)

# 5. Entanglement along the circuit of animation 1: Bloch-vector lengths and concurrence
from quantum_mind.viz import plot_entanglement                           # noqa: E402
plot_entanglement(tr, size=(9, 2.8)).savefig(OUT / 'entanglement_timeline.png', dpi=130, facecolor=BG)
print('assets written to', OUT)
