"""Trust-aware hand-over
=====================

Robotics, human-robot interaction: before each hand-over the robot chooses to hand over, hand over
slowly, ask about trust or wait, by expected cost under a trust model. The order-aware policy (trust
as a qubit that asking can change) is compared with a Markov-based policy and with always handing over,
on people simulated from the open-system trust model (which favours the order-aware policy by
construction).
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_36.png'
import numpy as np
from quantum_mind.applications.handover import TrustAwareHandover, simulate_handover_session
from quantum_mind.families.dynamics import OpenSystemBelief, MarkovBelief

person = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)


class AlwaysHandOver(TrustAwareHandover):
    def decide(self):
        return {'action': 'handover'}


for name, make in (('order-aware', lambda: TrustAwareHandover(person)),
                   ('Markov-based', lambda: TrustAwareHandover(MarkovBelief(p0=0.5, up=0.3, down=0.3))),
                   ('always hand over', lambda: AlwaysHandOver(person))):
    costs = [simulate_handover_session(make(), TrustAwareHandover(person), np.random.default_rng(i), 20)['cost'] for i in range(40)]
    print('%-17s mean session cost %.1f' % (name, np.mean(costs)))
