"""Human-computer interaction: does asking about trust change trust?

An automated assistant succeeds or fails over six interactions; the user is asked 'Do you trust it?'
either only at the end or also halfway. Markov dynamics predict no effect of the halfway question;
open-system dynamics predict one. Data are simulated."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.core import compare
from qlcog.families.dynamics import MarkovBelief, OpenSystemBelief, question_effect

events = (1, 1, 0, 1, 0, 1)
design = {'end_only': (events, (5,)), 'also_halfway': (events, (2, 5))}
os_true = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
print('Question effect, open-system: %.3f; Markov: %.3f' % (question_effect(os_true, events, 5, 2),
      question_effect(MarkovBelief(p0=0.5, up=0.3, down=0.3), events, 5, 2)))
for n in (100, 1600):
    data = os_true.sample(design, n, np.random.default_rng(n))
    res = compare([OpenSystemBelief, MarkovBelief], data, design, restarts=6)
    print('n = %4d per condition: selected %s (BIC %s)' % (n, type(res[0].model).__name__, [round(r.bic, 1) for r in res]))
