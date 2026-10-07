# Running the human model on a robot (ROS 2)

The decision functions are plain Python, so they run anywhere. For a robot, `HumanModelService` wraps
the human-model ensemble and the ask-or-act rule behind JSON messages, and the ROS 2 package
`quantum_mind_ros` publishes it as a node. This tutorial drives the service exactly as the node does,
then runs the node's own callbacks with stand-in ROS modules, so it works without a ROS installation.

```{mermaid}
flowchart LR
    accTitle: Message flow of the ROS 2 human-model node
    accDescr: Planner and dialogue nodes publish answers and queries as JSON strings; the human model node updates its ensemble and publishes a prediction with an ask-or-act decision.
    dlg["dialogue node<br/>(speech, buttons)"]:::hum -->|"/human_model/answers<br/>{order, answers}"| node["human_model_node<br/>HumanModelService"]:::op
    plan["planner / behaviour tree"]:::in -->|"/human_model/query<br/>{order, ask_cost, error_cost}"| node
    node -->|"/human_model/decision<br/>{prediction, uncertainty, action}"| plan
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

```python
import json, sys, types, pathlib
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.applications.robotics import HumanModelService, domain_models
theme = use_mpl_style('dark')
```

## 1. The service and its messages

Answers arrive one person at a time: the order the two clarification questions were asked in, and the
two answers (1 = yes). Queries ask for a prediction and a decision.

```python
svc = HumanModelService(refit_every=25, min_answers=40)
print(json.dumps(svc.query({'order': 'AB'})))             # nothing learnt yet: ask

people = domain_models('object_clarification')['QL']       # simulated population
rng = np.random.default_rng(0)
trace = []
for step in range(8):
    batch = people.sample(None, 15, rng)                    # 15 people per order arrive
    for order, counts in batch.items():
        for cell, n in enumerate(counts):
            for _ in range(n):
                svc.add_answer({'order': order, 'answers': [[1, 1], [1, 0], [0, 1], [0, 0]][cell]})
    reply = svc.query({'order': 'AB', 'ask_cost': 1.0, 'error_cost': 1.5})
    trace.append((svc.n_answers, reply.get('p_first_yes'), reply.get('uncertainty', {}).get('model_disagreement_bits'), reply['action']))
    print('%3d answers -> %s (%s)' % (svc.n_answers, reply['action'], reply['reason']))
print(json.dumps(reply, indent=1)[:600])
```

```python
n, p, dis, act = zip(*[t for t in trace if t[1] is not None])
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.plot(n, dis, 'o-', color=theme['palette'][0], label='model disagreement (bits)')
ax.axhline(0.05, color=theme['palette'][0], ls=':', lw=1, label='threshold')
ax.set_xlabel('answers received'); ax.set_ylabel('bits'); ax.legend(); ax.set_title('What the node learns as answers arrive')
plt.show()
```

## 2. The node's callbacks, without ROS

The node subscribes to `~/answers` and `~/query` and publishes to `~/decision`. Here `rclpy` and
`std_msgs` are replaced by minimal stand-ins (the same ones the package's tests use), and the node's
real callbacks are called with JSON strings.

```python
published = []

class Node:                                  # stand-in for rclpy.node.Node
    def __init__(self, name): self.params = {}
    def declare_parameter(self, name, default): self.params[name] = default
    def get_parameter(self, name): return types.SimpleNamespace(value=self.params[name])
    def create_publisher(self, typ, topic, depth): return types.SimpleNamespace(publish=published.append)
    def create_subscription(self, typ, topic, cb, depth): pass
    def get_logger(self): return types.SimpleNamespace(info=print, warning=print)

class String:                                # stand-in for std_msgs.msg.String
    def __init__(self, data=''): self.data = data

for name, attrs in (('rclpy', {}), ('rclpy.node', {'Node': Node}), ('std_msgs', {}), ('std_msgs.msg', {'String': String})):
    mod = types.ModuleType(name); mod.__dict__.update(attrs); sys.modules[name] = mod
sys.path.insert(0, str(pathlib.Path('integrations/ros2/quantum_mind_ros').resolve()))
from quantum_mind_ros.human_model_node import HumanModelNode

node = HumanModelNode()
for _ in range(60):
    order = 'AB' if rng.random() < 0.5 else 'BA'
    node.on_answer(String(json.dumps({'order': order, 'answers': [int(rng.random() < 0.6), int(rng.random() < 0.5)]})))
node.on_answer(String('not json'))           # bad messages are logged, not raised
node.on_query(String(json.dumps({'order': 'AB', 'error_cost': 3.0})))
decision = json.loads(published[-1].data)
print({k: decision[k] for k in ('action', 'reason', 'p_first_yes')})
```

## 3. On a real robot

```bash
cp -r integrations/ros2/quantum_mind_ros ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select quantum_mind_ros && source install/setup.bash
ros2 launch quantum_mind_ros human_model.launch.py
ros2 topic pub --once /human_model/answers std_msgs/String "{data: '{\"order\": \"AB\", \"answers\": [1, 0]}'}"
ros2 topic pub --once /human_model/query std_msgs/String "{data: '{\"order\": \"AB\"}'}"
ros2 topic echo /human_model/decision
```

A behaviour tree can call the same service through a condition node: ask the clarifying question when
`action == 'ask'`, otherwise continue to the grasp.

The statements below are checked by this cell every time the documentation is built:

```python
assert trace[0][3] == 'ask' and svc.n_answers == 8 * 30
assert decision['action'] in ('ask', 'act') and 0 <= decision['p_first_yes'] <= 1
json.dumps(reply)                            # every reply is plain JSON
print('checked')
```

**Summary.** The service turns a stream of answers into predictions, an uncertainty and an ask-or-act
decision, all as JSON, and the ROS 2 node is a thin wrapper around it. The node has been tested with
stand-in ROS modules, as here; it has not yet been run inside a ROS 2 installation in CI (see
[validation status](../help/validation.md)).

References: Macenski et al. (2022), *Science Robotics* 7, eabm6074; Colledanchise & Ögren (2018),
*Behavior Trees in Robotics and AI*.
