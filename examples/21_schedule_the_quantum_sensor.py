"""Scheduling a quantum accelerometer with a QUBO
===============================================

Thread 1 (quantum computing) applied to thread 2 (quantum sensing). A slow cold-atom accelerometer can only be interrogated in a few of the
possible time slots, and only where its fringe contrast is good. Which slots? The exact answer for any choice comes from the Kalman filter's
covariance; the QUBO solvers of ``quantum_mind.computing`` search the choices through a local quadratic model of that objective. The mission,
the sensors and the contrast are **simulated** (illustrative values), and the quantum solver is a simulator, not hardware.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_21.png'
import numpy as np
import matplotlib.pyplot as plt

from quantum_mind.sensing import random_mission, schedule_interrogations

pr = random_mission(seed=0, n_slots=12, k=3)
print('%d slots, %d usable (contrast >= %.1f), choose %d; objective = weighted RMS position std over the mission (m)'
      % (len(pr.t_slots), len(pr.feasible), pr.min_contrast, pr.k))

# %%
# Baselines and the QUBO schedule
# -------------------------------
# ``uniform`` spaces the readings evenly, ``greedy`` adds the best slot one at a time on the exact objective, and the QUBO method starts from
# the uniform schedule, builds a quadratic model around it (exact within two flips) and asks a QUBO solver for the best schedule in that model.
rows = {'uniform spacing': pr.uniform(), 'random': pr.random(0), 'greedy on the exact objective': pr.greedy(),
        'local QUBO + exact solver': schedule_interrogations(pr, 'exact'),
        'local QUBO + annealing': schedule_interrogations(pr, 'annealing'),
        'exact optimum (enumeration)': pr.optimum()}
for name, s in rows.items():
    print('%-32s slots %-14s objective %.3f m' % (name, s.slots, s.objective))

# %%
# Where the readings go
# ---------------------
# The first reading matters most: the MEMS bias is unknown at the start, and the position error it causes grows with the square of time.
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.fill_between(pr.grid, pr.w / pr.w.max(), color='0.5', alpha=0.25, step='mid', label='importance of accuracy')
for y, (name, s) in zip((1.25, 1.1, 0.95), (('uniform', rows['uniform spacing']), ('QUBO', rows['local QUBO + exact solver']),
                                             ('optimum', rows['exact optimum (enumeration)']))):
    ax.plot(pr.t_slots[s.slots], [y] * len(s.slots), 'o', label='%s (%.2f m)' % (name, s.objective))
ax.plot(pr.t_slots[pr.contrast < pr.min_contrast], np.zeros((pr.contrast < pr.min_contrast).sum()), 'x', color='r', label='unusable slot')
ax.set_xlabel('time (s)'); ax.set_yticks([]); ax.legend(frameon=False, fontsize=8, ncol=2)
plt.show()
