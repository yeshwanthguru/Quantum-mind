"""Trust as a qubit
================

Robotics / HRI: a person's trust in a robot as a qubit (qlcog OpenSystemBelief, simulated
parameters). Successes rotate the belief towards ``|0>`` ('I trust the robot'), failures away from it,
and dephasing pulls it towards the z axis. The animation shows the belief over the event sequence;
the printout compares P(trust) read from the Bloch vector with the model's prediction.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/viz.png'
import pathlib
from qlcog.families.dynamics import OpenSystemBelief, final_yes
from qlcog.applications.robotics import TRUST_EVENTS
from qlcog.viz import belief_trajectory

# %%
# The belief trajectory
# ---------------------
# P(trust) read from the final Bloch vector equals the model's prediction.
m = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
traj = belief_trajectory(m, TRUST_EVENTS, steps=10)
p_sphere = (1 + traj.vectors[-1, 0, 2]) / 2
print('events %s: P(trust) from the Bloch vector %.4f, model %.4f' % (TRUST_EVENTS, p_sphere, final_yes(m, TRUST_EVENTS, (5,))))
print('Bloch-vector length (coherence lost to dephasing): start %.3f, end %.3f' % (traj.purity()[0, 0], traj.purity()[-1, 0]))

# %%
# Interactive animation
# ---------------------
try:
    from qlcog.viz import animate_bloch, save_html
    out = pathlib.Path(sys.argv[0]).resolve().parent / 'output'; out.mkdir(exist_ok=True)
    save_html(animate_bloch(traj, title='Trust belief (simulated person)'), out / 'trust_belief.html')
    print('interactive figure:', out / 'trust_belief.html')
except ImportError:
    print('install plotly for the interactive figure')
