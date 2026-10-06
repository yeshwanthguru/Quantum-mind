"""When should a robot ask for help?
=================================

Robotics: when should a robot ask for help? (simulated people)

A robot must fetch "the red cup on the left". Before acting it can ask two clarification questions;
the answers of a simulated population (quantum-like order effects, illustrative parameters) arrive
in batches. After each batch the robot refits its HumanModelEnsemble, predicts how likely a person
is to confirm its hypothesis, and uses ask_or_act() to decide whether asking is worth its cost.
While the human models still disagree the robot asks whatever the costs; once they agree, the
decision is driven by the cost of a wrong action: with an expensive error (3) it keeps asking, with
a cheap error (1.5) it acts.

This is a decision layer only: no robot middleware is needed. In a ROS 2 system, the same calls
would sit in a service or behaviour-tree node (not included in the package).
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/robot.png'
import numpy as np
from qlcog.applications.robotics import domain_models, HumanModelEnsemble, ask_or_act
from qlcog.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel

# %%
# Answers arrive in batches
# -------------------------
# After each batch the ensemble is refitted and the robot decides, for two error costs, whether to
# ask or to act.
rng = np.random.default_rng(3)
people = domain_models('object_clarification')['QL']                # simulated population
counts = {'AB': np.zeros(4, int), 'BA': np.zeros(4, int)}
print('batch  people  P(confirm)  disagreement(bits)  error cost 3.0   error cost 1.5')
for batch in range(1, 7):
    new = people.sample(None, 25, rng)
    counts = {c: counts[c] + new[c] for c in counts}
    ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]).update(counts, restarts=4, rng=rng)
    p, unc = ens.predict('AB')
    p_confirm = p[0] + p[1]                                            # 'yes' to the first question
    hi, lo = (ask_or_act(p_confirm, unc, ask_cost=1.0, error_cost=c, max_disagreement_bits=0.01)['action'] for c in (3.0, 1.5))
    print('%5d  %6d  %10.3f  %18.4f  %-15s %s' % (batch, 50 * batch, p_confirm, unc['model_disagreement_bits'], hi, lo))
print('ensemble weights:', [(s['model'], round(s['weight'], 2)) for s in ens.summary()])
