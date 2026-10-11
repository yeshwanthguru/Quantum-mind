"""A robot's belief that depends on context (COBRA)
=================================================

A person reaches toward a robot. The same hand cue means "hand over" after a look at the robot and
"retrieve" after a look away. A classical Bayes filter adds the same evidence either way; COBRA lets the
previous frame rotate the basis in which the next frame is read.

The data are **simulated** and built to contain this context effect. The script prints the Bayes filter and
COBRA side by side, the shift each gives to the same hand cue after opposite looks, and the decision cost.
"""
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_21.png'
import numpy as np

from quantum_mind.belief import COBRA, GaussianBayesFilter, min_cost_action

MU_A, MU_F, KAPPA = np.array([0.7, -0.7, 0.0]), np.array([0.0, 0.0, 0.9]), np.array([1.0, -1.0, 0.0])


def simulate(n, rng, frames=4):
    z = rng.integers(0, 3, n)
    a = MU_A[z][:, None] + rng.standard_normal((n, frames))
    f = MU_F[z][:, None] + rng.standard_normal((n, frames))
    prev = np.concatenate([np.zeros((n, 1)), a[:, :-1]], axis=1)
    b = KAPPA[z][:, None] * np.sign(prev) + rng.standard_normal((n, frames))
    return np.stack([a, b, f], axis=2), z


rng = np.random.default_rng(11)
x_train, z_train = simulate(500, rng)
x_test, z_test = simulate(2000, rng)

bayes = GaussianBayesFilter(3).fit(x_train, z_train)
cobra = COBRA(3, 3, steps=60, batch=100, restarts=4, seed=0).fit(x_train, z_train)

cost = np.array([[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]])          # hand over, wait, ask
for name, model in (('Bayes filter', bayes), ('COBRA', cobra)):
    p = model.predict_proba(x_test)
    act = min_cost_action(p, cost)
    print('%-13s accuracy %.3f   expected cost %.3f   ask rate %.2f'
          % (name, (p.argmax(1) == z_test).mean(), cost[act, z_test].mean(), (act == 2).mean()))


def shift(model, sign):
    """Log-odds shift (handover vs retrieve) caused by one hand cue b = +1.5 after a gaze cue of +-1.5."""
    before = np.zeros((1, 1, 3)); before[0, 0, 0] = sign * 1.5
    after = np.zeros((1, 2, 3)); after[0, 0, 0] = sign * 1.5; after[0, 1, 1] = 1.5
    odds = lambda p: float(np.log(p[0, 0] / p[0, 1]))
    return odds(model.predict_proba(after)) - odds(model.predict_proba(before))


for name, model in (('Bayes filter', bayes), ('COBRA', cobra)):
    print('%-13s hand-cue shift after a look at the robot %+.2f, after a look away %+.2f'
          % (name, shift(model, +1), shift(model, -1)))

belief = cobra.start()                                   # online, one frame at a time, as on a robot
for frame in x_test[np.flatnonzero(z_test == 0)[0]]:
    p = belief.update(frame)
print('online belief after one hand-over episode:', dict(zip(['handover', 'retrieve', 'withdraw'], p.round(2))))
