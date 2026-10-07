# End to end: a robot that fetches and hands over objects

The earlier tutorials show each component on its own. This one puts them together in one loop, the
way they would run on a robot:

```{mermaid}
flowchart LR
    accTitle: The end-to-end robot loop
    accDescr: Camera, speech and gaze are turned into likelihoods, the detector is recalibrated, the cues are fused, the robot decides whether to ask clarifying questions, picks the object, then chooses how to hand it over based on its trust model, and updates trust from the outcome.
    cam["camera → detector"]:::in --> cal["temperature scaling"]:::op
    mic["speech → n-best"]:::in --> fus
    gaze["gaze direction"]:::in --> fus
    cal --> fus["fuse cues"]:::op --> dec{"confident?"}:::op
    dec -->|no| ask["question planner<br/>(order-aware answers)"]:::hum --> pick
    dec -->|yes| pick["pick the object"]:::op --> ho["trust-aware hand-over"]:::op --> out["outcome"]:::out
    out -->|update trust| ho
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

Three objects lie on a table: a red cup, a blue cup and a red bowl. A **simulated** person asks for one
of them. The robot sees the scene with an over-confident detector, hears a noisy speech recogniser and
follows the person's gaze. It can ask yes/no questions whose answers depend on what was asked before,
then hands the object over to a person whose trust follows the trust qubit. The full pipeline is
compared with a naive robot that trusts its detector, never asks and always hands over at normal speed.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.envs.hri import default_domain
from quantum_mind.applications.calibration import TemperatureScaling
from quantum_mind.applications.fusion import detector_likelihood, asr_likelihood, direction_likelihood, bayes_fusion
from quantum_mind.applications.questioning import QuestionPlanner
from quantum_mind.applications.handover import TrustAwareHandover
from quantum_mind.families.dynamics import OpenSystemBelief
theme = use_mpl_style('dark')

objects = ['red cup', 'blue cup', 'red bowl']
keywords = {'red cup': ['red', 'cup'], 'blue cup': ['blue', 'cup'], 'red bowl': ['red', 'bowl']}
positions = {'red cup': [-1.0, 0.3], 'blue cup': [0.0, 1.0], 'red bowl': [1.0, 0.3]}
answers = default_domain()            # how the simulated person answers 'red?', 'cup?', 'left?'
trust_model = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
COSTS = dict(ask=0.3, wrong_object=5.0)
print(answers.hypotheses, answers.questions)
```

## 1. Simulated sensors

```python
def detector_logits(target, rng):
    """Logits of a detector that is right more often than not, before it is made over-confident."""
    z = rng.normal(0, 1.0, 3)
    z[target] += 1.2
    return z

def over_confident(z):
    p = np.exp(2.5 * z)                     # temperature 0.4: far too sharp
    return p / p.sum()

def speech(target, rng):
    word = rng.choice(keywords[objects[target]]) if rng.random() < 0.8 else rng.choice(['red', 'blue', 'cup', 'bowl'])
    return [('the %s one' % word, 0.9)]

def gaze(target, rng):
    return np.array(positions[objects[target]]) + rng.normal(0, 0.5, 2)

# calibrate the detector on 500 labelled frames recorded beforehand
rng = np.random.default_rng(0)
cal_t = rng.integers(0, 3, 500)
cal_p = np.array([over_confident(detector_logits(t, rng)) for t in cal_t])
ts = TemperatureScaling().fit(cal_p, cal_t)
print('fitted temperature %.2f (the detector is over-confident when T > 1)' % ts.T)
```

## 2. One episode

```python
def episode(rng, full=True, policy=None, person=None):
    target = int(rng.integers(3))
    raw = over_confident(detector_logits(target, rng))
    cost = 0.0
    if full:
        cues = [detector_likelihood(ts.transform(raw[None])[0], objects),
                asr_likelihood(speech(target, rng), keywords, objects),
                direction_likelihood(gaze(target, rng), positions, objects, kappa=3.0)]
        belief = bayes_fusion(cues)
        planner = QuestionPlanner(answers, prior=belief, ask_cost=COSTS['ask'], error_cost=COSTS['wrong_object'])
        out = planner.simulate(answers, objects[target], rng)          # asks only while it is worth it
        choice, cost = objects.index(out['choice']), out['cost']
    else:
        choice = int(raw.argmax())                                     # trust the raw detector
    if choice != target:
        return {'cost': cost + COSTS['wrong_object'], 'success': False, 'asked': cost > 0}
    # hand-over: the full robot uses its trust model, the naive robot always hands over normally
    d = policy.decide()['action'] if full else 'handover'
    if d == 'ask':
        yes = rng.random() < person.p_trust
        person.answer(yes); policy.answer(yes)
        cost += policy.ask_cost
        d = policy.decide()['action']
        d = d if d in ('handover', 'slow_handover') else 'slow_handover'
    p_fail = (1 - person.p_trust) * (policy.slow_factor if d == 'slow_handover' else 1.0)
    cost += policy.slow_cost if d == 'slow_handover' else 0.0
    ok = rng.random() >= p_fail
    person.observe(int(ok))
    if full:
        policy.observe(int(ok))
    return {'cost': cost + (0.0 if ok else policy.fail_cost), 'success': bool(ok), 'asked': cost > 0}
```

## 3. Run both robots

Each robot serves 20 people, 15 requests each; trust carries over between requests of the same person.

```python
def run(full, people=20, requests=15):
    rows = []
    for i in range(people):
        rng = np.random.default_rng(1000 + i)
        policy, person = TrustAwareHandover(trust_model), TrustAwareHandover(trust_model)
        rows += [episode(rng, full, policy, person) for _ in range(requests)]
    return rows

full, naive = run(True), run(False)
for name, rows in (('full pipeline', full), ('naive robot', naive)):
    c = np.array([r['cost'] for r in rows]); s = np.mean([r['success'] for r in rows])
    print('%-14s success %.2f   mean cost per request %.2f   asked in %.0f%% of requests'
          % (name, s, c.mean(), 100 * np.mean([r['asked'] for r in rows])))
```

```python
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for name, rows, c in (('full pipeline', full, theme['palette'][3]), ('naive robot', naive, theme['palette'][0])):
    ax.plot(np.cumsum([r['cost'] for r in rows]), color=c, lw=2.2, label=name)
ax.set_xlabel('requests served'); ax.set_ylabel('cumulative cost'); ax.legend()
ax.set_title('Simulated people: full pipeline against a naive robot')
plt.show()
```

The statements below are checked by this cell every time the documentation is built:

```python
cost = {n: np.mean([r['cost'] for r in rows]) for n, rows in (('full', full), ('naive', naive))}
succ = {n: np.mean([r['success'] for r in rows]) for n, rows in (('full', full), ('naive', naive))}
assert cost['full'] < cost['naive'] and succ['full'] > succ['naive']
print('checked')
```

**Result.** On simulated people the full pipeline costs less per request and succeeds more often than
the naive robot: recalibration and fusion pick the right object more often, questions resolve the
remaining doubt, and the trust model avoids risky fast hand-overs. The comparison is not a fair test
of the quantum-like parts on their own, because the simulated people follow the same models the robot
uses; tutorials 3 and 7 isolate those. On a robot, each simulated sensor is replaced by the real one,
and the loop stays the same (tutorial 18 shows the ROS 2 side).

References: the component tutorials (3, 5, 7, 8) and their references.
