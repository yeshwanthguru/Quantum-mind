# Trust on a qubit

A person's trust in a robot rises after good hand-overs and falls after bad ones. Two things make it
harder than a simple counter: trust **drifts** between interactions, and **asking** "do you trust me?"
can itself change it. The open-system model represents trust as a qubit whose state rotates after
each outcome, loses coherence over time (dephasing, Lindblad dynamics), and is projected when the
person is asked. This tutorial shows the trust qubit on the Bloch sphere, measures the effect of
asking, compares the model with a Markov baseline, and uses it in a hand-over policy.

All people here are **simulated**; the parameters are illustrative.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style, belief_trajectory
from quantum_mind.viz.mpl import BlochSphere
from quantum_mind.families.dynamics import OpenSystemBelief, MarkovBelief, question_effect, final_yes
theme = use_mpl_style('dark')

person = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)   # rotations per outcome, dephasing rate
events = (1, 1, 0, 1, 0, 1)                                              # 1 = good hand-over, 0 = bad
print('P(trust) after the six hand-overs: %.3f' % final_yes(person, events, (5,)))
```

## 1. Watch trust move on the Bloch sphere

The north pole is "trusts", the south pole "does not trust". Good outcomes rotate the state one way,
bad outcomes the other; dephasing pulls it towards the vertical axis, inside the sphere.

```python
traj = belief_trajectory(person, events, steps=10)
path = traj.vectors[:, 0, :]
fig = plt.figure(figsize=(5.2, 5.2))
b = BlochSphere(ax=fig.add_subplot(projection='3d'), title='Trust belief over six hand-overs', theme='dark')
b.add_trajectory(path, color=theme['palette'][1], lw=2.4)
b.add_vector(path[0], color=theme['palette'][3], label='start')
b.add_vector(path[-1], color=theme['palette'][0], label='end')
plt.show()
print('Bloch-vector length (1 = fully coherent): start %.2f, end %.2f' % (np.linalg.norm(path[0]), np.linalg.norm(path[-1])))
```

## 2. Does asking about trust change trust?

Ask once at the end, or also half-way through. In the open-system model the half-way question
projects the state and changes the final answer; in the Markov model it does not.

```python
markov = MarkovBelief(p0=0.5, up=0.3, down=0.3)
print('Question effect (change in final P(trust) caused by asking half-way):')
print('  open-system: %+.3f' % question_effect(person, events, 5, 2))
print('  Markov     : %+.3f' % question_effect(markov, events, 5, 2))

steps = range(len(events))
fig, ax = plt.subplots(figsize=(6.8, 3.0))
ax.plot(steps, [final_yes(person, events[:k + 1], (k,)) for k in steps], 'o-', label='open-system (quantum-like)')
ax.plot(steps, [final_yes(markov, events[:k + 1], (k,)) for k in steps], 's--', label='Markov baseline')
ax.set_xticks(list(steps), ['good' if e else 'bad' for e in events]); ax.set_ylim(0, 1)
ax.set_ylabel('P(trust)'); ax.set_title('Trust after each hand-over'); ax.legend()
plt.show()
```

## 3. Which model do the data support?

Simulate people from the open-system model under two questioning designs and compare both models by
BIC. With few people the simpler Markov model can win; with more, the order effect becomes visible.

```python
from quantum_mind.core import compare
design = {'end_only': (events, (5,)), 'also_halfway': (events, (2, 5))}
for n in (100, 1600):
    data = person.sample(design, n, np.random.default_rng(n))
    res = compare([OpenSystemBelief, MarkovBelief], data, design, restarts=4)
    print('n = %4d per design: selected %-17s BIC %s' % (n, type(res[0].model).__name__, [round(float(r.bic), 1) for r in res]))
```

## 4. Use it: a trust-aware hand-over policy

The robot keeps its own copy of the trust model. Before each hand-over it chooses: hand over, hand
over slowly, ask, or wait, by expected cost. Below, the order-aware policy is run against simulated
people next to a policy that uses the Markov model and one that always hands over.

```python
from quantum_mind.applications.handover import TrustAwareHandover, simulate_handover_session

def run(make_policy, sessions=40):
    costs = []
    for i in range(sessions):
        out = simulate_handover_session(make_policy(), TrustAwareHandover(person), np.random.default_rng(i), n_steps=20)
        costs.append(out['cost'])
    return np.array(costs)

class Always(TrustAwareHandover):
    def decide(self):
        return {'action': 'handover'}

results = {'order-aware': run(lambda: TrustAwareHandover(person)),
           'Markov-based': run(lambda: TrustAwareHandover(markov)),
           'always hand over': run(lambda: Always(person))}
for k, v in results.items():
    print('%-17s mean session cost %.1f' % (k, v.mean()))
```

```python
fig, ax = plt.subplots(figsize=(6.4, 3.0))
parts = ax.violinplot(list(results.values()), showmeans=True)
for body, c in zip(parts['bodies'], theme['palette']):
    body.set_facecolor(c); body.set_alpha(0.6)
ax.set_xticks([1, 2, 3], list(results)); ax.set_ylabel('session cost (lower is better)')
ax.set_title('Hand-over policies on simulated people')
plt.show()
```

The statements below are checked by this cell every time the documentation is built:

```python
assert abs(question_effect(markov, events, 5, 2)) < 1e-12 < abs(question_effect(person, events, 5, 2))
assert results['order-aware'].mean() < results['Markov-based'].mean() < results['always hand over'].mean()
print('checked')
```

**Caveat.** The simulated people follow the open-system model, which favours the order-aware policy
by construction. Whether real people behave this way is an empirical question; tutorial 2 shows how
to test it.

References: Busemeyer & Bruza (2024), *Quantum Models of Cognition and Decision*, ch. 8-9; Hancock et
al. (2011), *Human Factors* 53, 517-527.
