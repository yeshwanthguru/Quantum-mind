# Getting started

## Install

```bash
pip install quantum-mind              # core: numpy, scipy
pip install "quantum-mind[all]"       # everything (Qiskit, cloud back ends, viewer)
pip install "quantum-mind[all] @ git+https://github.com/yeshwanthguru/Quantum-mind"   # latest development version
```

| Extra | Adds | For |
|---|---|---|
| (core) | numpy, scipy | quantum-like families, COBRA belief filter, robot decision layer, state-vector simulator |
| `[qiskit]` | qiskit, qiskit-aer, qiskit-ibm-runtime | circuits, Aer, IBM fake-backend noise, `circuit_trajectory` |
| `[cloud]` | qiskit-ibm-runtime, amazon-braket-sdk | IBM Quantum and Amazon Braket hardware |
| `[viz]` | matplotlib, plotly, ipywidgets, anywidget, seaborn | Bloch-sphere viewer, animations, live widget |
| `[all]` | all of the above | |
| `[dev]` | pytest, pytest-cov, ruff, nbclient, build, twine | tests, coverage, linting, notebooks, packaging |
| `[docs]` | Sphinx, pydata-sphinx-theme, sphinx-gallery, MyST-NB | this documentation |

From a clone: `pip install -e ".[all,dev]"`. Python 3.10 or later.

## First steps

::::{tab-set}

:::{tab-item} Quantum-like
Is there a question-order effect, and which model explains it?

```python
import numpy as np
from quantum_mind import compare
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel, qq_test

counts = {'AB': np.array([212, 48, 61, 179]),   # A asked first: yes-yes, yes-no, no-yes, no-no
          'BA': np.array([240, 33, 52, 175])}   # B asked first
print(qq_test(counts))                           # parameter-free test of the quantum-like prediction
for r in compare([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], counts):
    print(type(r.model).__name__, round(r.bic, 1))
```
:::

:::{tab-item} COBRA belief filter
A robot's belief about a person, where the previous sensor frame changes what the next one means.

```python
import numpy as np
from quantum_mind.belief import COBRA, GaussianBayesFilter, min_cost_action

rng = np.random.default_rng(0)
frames = rng.normal(size=(300, 4, 3))                    # (sequences, frames, sensor features)
intent = (frames[:, :, 0].sum(1) > 0).astype(int) + (frames[:, 0, 2] > 1)
model = COBRA(n_hyp=3, n_features=3, steps=40, restarts=2).fit(frames, np.minimum(intent, 2))
baseline = GaussianBayesFilter(3).fit(frames, np.minimum(intent, 2))

belief = model.start()                                   # online, one frame at a time
for frame in frames[0]:
    p = belief.update(frame)
cost = np.array([[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]])   # hand over, wait, ask
print(p.round(2), min_cost_action(p, cost))
```
:::

:::{tab-item} Robotics
A human model with uncertainty, for a planner or a ROS 2 system.

```python
from quantum_mind.applications.robotics import HumanModelService

svc = HumanModelService()
svc.add_answer({'order': 'AB', 'answers': [1, 0]})          # one person's two answers (1 = yes)
svc.query({'order': 'AB', 'ask_cost': 1.0, 'error_cost': 3.0})
# -> prediction, uncertainty, ensemble weights, and 'ask' or 'act'
```
:::

:::{tab-item} Viewer
Watch a circuit on the Bloch sphere and measure the same state on a noisy simulator.

```python
from qiskit import QuantumCircuit
from quantum_mind.viz import circuit_trajectory, animate_bloch, save_html, LiveBloch, bloch_tomography

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.ry(0.6, 1)
traj = circuit_trajectory(qc, steps=15)
save_html(animate_bloch(traj), 'bell.html')        # rotate, zoom, play, slider
LiveBloch(2).show().play(traj)                     # real time in Jupyter or a Matplotlib window
bloch_tomography(qc, 'aer:FakeTorino')             # Bloch vectors under IBM device noise
```
:::
::::

## Choosing and testing a model

1. **Check the phenomenon first.** Look for an order effect, a violation of total probability, or
   asymmetric similarity. If there is none, a classical model is enough.
2. **Fit quantum-like models and classical baselines together** with {func}`~quantum_mind.core.fit.compare`
   and report all of them.
3. **Use each family's distinctive test**: the QQ equality, total-probability bounds,
   Contextuality-by-Default, or the effect of an intermediate judgement.
4. **Run a recovery study** ({func}`~quantum_mind.core.fit.recovery`) at the planned sample size before
   collecting data.
5. **Always print the classical baseline and the ablation next to a quantum-like result**, as every
   example and the COBRA benchmark do, and look for a control task where the new idea should not help.

## Running the tests

```bash
python3 -m pytest -q --cov=quantum_mind
```

The tests check the simulator against Qiskit gate by gate, gradients against finite differences,
exported circuits against simulation, circuit builders against the analytic models, the IBM and
Braket submission code against local stand-ins, and Bloch trajectories against Qiskit's partial
trace. The docstring examples run as doctests.
