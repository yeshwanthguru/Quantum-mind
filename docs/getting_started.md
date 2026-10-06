# Getting started

## Install

```bash
pip install quantum-mind              # core: numpy, scipy
pip install "quantum-mind[all]"       # everything (Qiskit, cloud back ends, viewer)
pip install "quantum-mind[all] @ git+https://github.com/yeshwanthguru/quantum-mind"   # latest development version
```

| Extra | Adds | For |
|---|---|---|
| (core) | numpy, scipy | quantum-like families, quantum simulator, quantum-inspired solvers, problems |
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

:::{tab-item} Quantum
Classify with a circuit, then run it under IBM device noise.

```python
from quantum_mind.quantum import VariationalClassifier
from quantum_mind.circuits import run

clf = VariationalClassifier(layers=3).fit(X_train, y_train)
print(clf.score(X_test, y_test))
counts = run(clf.to_qiskit(X_test[0]), 'aer:FakeTorino', shots=4000)
```
:::

:::{tab-item} Quantum-inspired
One robot task-allocation problem, three kinds of solver.

```python
from quantum_mind.problems import task_allocation
from quantum_mind.quantum import QAOA
from quantum_mind.inspired import SQA, QIEA, simulated_annealing

q = task_allocation(costs)                       # costs[robot, task]
print(q.brute_force()[1],                        # exact
      QAOA(q, p=3).run().energy,                 # quantum
      SQA(q).run().value, QIEA(q).run().value,   # quantum-inspired
      simulated_annealing(q).value)              # classical baseline
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
5. **For quantum and quantum-inspired solvers, always print the exact or classical baseline**, as
   every example does.

## Running the tests

```bash
python3 -m pytest -q --cov=quantum_mind
```

The tests check the simulator against Qiskit gate by gate, gradients against finite differences,
exported circuits against simulation, circuit builders against the analytic models, the IBM and
Braket submission code against local stand-ins, and Bloch trajectories against Qiskit's partial
trace. The docstring examples run as doctests.
