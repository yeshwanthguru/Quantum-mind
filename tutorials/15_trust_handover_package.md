# A robot package: trust-aware hand-over

This tutorial builds one robotics use case end to end, as a separate installable package:
`tutorials/trust_handover_package` (`pip install -e tutorials/trust_handover_package`). A robot arm hands
objects to a person and decides, before each hand-over, whether to **hand over**, **hand over slowly**,
**ask "Are you ready?"** or **wait**. Its model of the person is Quantum Mind's quantum-like trust model.
The package adds populations of simulated people, a benchmark, a PyBullet simulation recorded as a GIF,
a ROS 2 node and its own tests. All people are simulated.

The trust-aware hand-over was chosen among the robotics models because it is unique to Quantum Mind,
quantum-like (no quantum computer is needed), a physical decision a robot takes many times a shift, and
it has a classical counterpart to compare with: the same decision rule on a Markov trust model.

```{mermaid}
flowchart LR
    accTitle: Structure of the quantum-handover package
    accDescr: Simulated people and the robot's trust model feed the decision rule; the decision drives the PyBullet arm or a real robot through ROS 2; outcomes and answers update the trust model.
    T["trust model<br/>quantum-like or Markov"]:::hum --> D["decision<br/>hand over · slow · ask · wait"]:::op
    D --> S["PyBullet arm<br/>(simulation)"]:::out
    D --> R["ROS 2 node<br/>/handover/decision"]:::out
    S & R -->|"taken · dropped · yes · no"| T
    P["simulated people<br/>two populations"]:::in --> S
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

```python
import sys, json
sys.path.insert(0, 'tutorials/trust_handover_package/src')     # or: pip install -e tutorials/trust_handover_package
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_handover import POLICIES, make_policy, run_benchmark, to_markdown
from quantum_handover.ros2_node import HandoverController
theme = use_mpl_style('dark')
```

## 1. One decision

With the population-average trust model, a new person's P(trust) is about 0.56. Handing over normally
would risk a drop, so the policy compares the expected cost of every action.

```python
d = make_policy('quantum-like').decide()
print('P(trust) = %.2f, best action: %s' % (d['p_trust'], d['action']))
for action, cost in sorted(d['costs'].items(), key=lambda kv: kv[1]):
    print('  %-14s expected cost %.2f' % (action, cost))
```

## 2. The benchmark

Every policy meets 300 simulated people from each of two populations: people whose trust follows the
quantum-like model and people whose trust follows the Markov model. The robot only knows population
averages, never the person in front of it.

```python
result = run_benchmark(n_people=300, n_steps=20, seed=0)
print(to_markdown(result))
```

```python
fig, ax = plt.subplots(figsize=(8, 3.6))
pops = list(result)
width = 0.2
for k, name in enumerate(POLICIES):
    means = [result[p][name]['cost'] for p in pops]
    errs = [result[p][name]['se'] for p in pops]
    ax.bar(np.arange(len(pops)) + (k - 1.5) * width, means, width, yerr=errs, label=name, color=theme['palette'][k])
ax.set_xticks(range(len(pops)), pops)
ax.set_ylabel('mean session cost (lower is better)')
ax.set_title('Hand-over policies against two kinds of simulated people')
ax.legend(fontsize=8, ncol=2)
plt.show()
```

The statements below are checked by this cell every time the documentation is built:

```python
ql, mk = result['quantum-like people'], result['Markov people']
fixed = ('always hand over', 'always slow')
for rows in (ql, mk):                                      # model-based beats fixed, for both kinds of people
    assert max(rows['quantum-like']['cost'], rows['markov']['cost']) < min(rows[f]['cost'] for f in fixed)
assert ql['quantum-like']['cost'] < ql['markov']['cost']    # each model wins on its own kind of people
assert mk['markov']['cost'] < mk['quantum-like']['cost']
regret_ql = mk['quantum-like']['cost'] - mk['markov']['cost']
regret_mk = ql['markov']['cost'] - ql['quantum-like']['cost']
assert regret_ql < regret_mk                               # the quantum-like policy loses less when wrong
print('loss when the model is wrong: quantum-like %.1f, Markov %.1f' % (regret_ql, regret_mk))
```

**Result.** Both model-based policies beat both fixed policies. Each model wins on the people who behave
like it, and the quantum-like policy loses less when it is wrong, so it is the safer choice when the
robot does not know which kind of person it faces. Which kind real people are is what a user study has
to establish.

## 3. The simulation

`quantum-handover animate` runs one session in PyBullet: a KUKA iiwa arm picks a cube from a stand and
hands it to the person; when the person does not take it, the cube falls under gravity. Left, the
quantum-like policy; right, always handing over; same simulated person (initial P(trust) 0.34) and the
same random draws. The quantum-like robot asks when trust is low and hands over slowly after a "no".
In this session it dropped 2 cubes (cost 27.0) against 5 (cost 50.0).

![Quantum-like policy (left) and always handing over (right) in PyBullet](../_static/tutorials/handover.gif)

The GIF is made by the package (it needs `pybullet` and `pillow`, which the documentation build does not
install):

```bash
pip install -e "tutorials/trust_handover_package[sim]"
quantum-handover animate --out handover.gif
```

## 4. On a robot (ROS 2)

The ROS 2 node is a thin layer over `HandoverController`, which turns events into the next decision. A
motion controller publishes what happened on `/handover/events`; the node answers on
`/handover/decision`. Here the controller runs without ROS:

```python
ctl = HandoverController('quantum-like')
print('start      ', ctl.decision())
for event in ({'answer': 'yes'}, {'outcome': 'taken'}, {'outcome': 'dropped'}, {'answer': 'no'}):
    print('%-11s' % list(event.values())[0], ctl.handle(json.dumps(event)))
```

On a ROS 2 machine: `python -m quantum_handover.ros2_node --policy quantum-like`.

**A limitation of the trust model, visible above.** After the "yes", trust is certain (1.0). The
successful hand-over then *lowers* it to about 0.85, and the drop that follows *raises* it slightly. In
this model every success and failure turns the trust state by a fixed angle, so right after an answer
has made trust certain, any event moves it away from certainty, and a failure can turn it back towards
it. The benchmark results include this behaviour; whether people's trust behaves this way is one of the
things a study with real people should test.

**Summary.** The package is a plain Python package with an optional simulation and an optional ROS 2
node, its requirements are in `tutorials/trust_handover_package/requirements.txt`, and its six tests
run with `pytest tutorials/trust_handover_package/tests`. The results are for simulated people only.
