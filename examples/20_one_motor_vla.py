"""A tiny vision-language-action model for one motor
===================================================

A one-joint motor must move to the object an instruction names on a one-dimensional table ("go to the red", "go to the
leftmost", "go to the block"). TinyVLA grounds the instruction in the image with one attention step and a small action head
turns the expected target into a motor command. When the instruction is true of two objects, a split-conformal gate makes the
robot ask, and the answer is fused with the first belief. The data are **simulated**; this is a prototype, not a real robot.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_20.png'
import numpy as np

from quantum_mind.embodied import TinyVLA, bc_plan, calibrate_gate, make_dataset, oracle_plan, run_episode, vla_plan, BCBaseline

train, calib, test = make_dataset(2000, 0), make_dataset(800, 1), make_dataset(200, 2)
vla = TinyVLA(seed=0).fit(train, steps=2000)
print('TinyVLA parameters:', vla.n_params)

# %%
# Look at one ambiguous instruction
# ---------------------------------
s = next(s for s in test if s.ambiguous)
g = vla.ground(s.image, s.text)
print('instruction:', ' '.join(s.words), '| objects at cells', s.cells, '| intended cell', s.target)
print('belief over cells (top 3):', {int(i): round(float(g[i]), 2) for i in np.argsort(g)[::-1][:3]})

# %%
# Closed loop: act blindly, ask through a conformal gate, or use the end-to-end baseline
# ---------------------------------------------------------------------------------------
gate = calibrate_gate(vla, calib, alpha=0.02)
bc = BCBaseline(hidden=128, seed=0).fit(train, steps=4000, lr=0.003)
print('%-22s %-9s %-12s %s' % ('policy', 'success', 'questions', '(ambiguous scenes only: success)'))
for name, plan in (('oracle', oracle_plan), ('BC baseline', bc_plan(bc)), ('TinyVLA, act', vla_plan(vla)),
                   ('TinyVLA, ask gate', vla_plan(vla, gate))):
    rng = np.random.default_rng(3)
    rows = [run_episode(sc, rng, plan) for sc in test]
    amb = [r['success'] for r, sc in zip(rows, test) if sc.ambiguous]
    print('%-22s %-9.3f %-12.2f %.3f' % (name, np.mean([r['success'] for r in rows]), np.mean([r['asked'] for r in rows]), np.mean(amb)))
