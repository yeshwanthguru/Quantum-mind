"""HumanModelService, and the ROS 2 node's callbacks with stand-in rclpy / std_msgs modules (ROS 2
itself is not installed in CI)."""
import json
import pathlib
import sys
import types

import numpy as np
import pytest
from quantum_mind.applications.robotics import HumanModelService, domain_models


def _answers(n, rng):
    model = domain_models('object_clarification')['QL']
    for order in ('AB', 'BA'):
        for k, c in enumerate(model.sample(None, n, rng)[order]):
            for _ in range(c):
                yield {'order': order, 'answers': [[1, 1], [1, 0], [0, 1], [0, 0]][k]}


def test_human_model_service():
    svc = HumanModelService(refit_every=50, min_answers=40)
    assert svc.query()['action'] == 'ask' and 'not enough data' in svc.query()['reason']
    refits = sum(svc.add_answer(m)['refitted'] for m in _answers(60, np.random.default_rng(0)))
    assert svc.n_answers == 120 and refits >= 2
    r = svc.query({'order': 'BA', 'ask_cost': 1.0, 'error_cost': 3.0})
    assert r['action'] in ('ask', 'act') and np.isclose(sum(r['prediction']), 1) and abs(sum(r['weights'].values()) - 1) < 1e-9
    json.dumps(r)                                                         # JSON-serialisable
    for bad in ({'order': 'XY', 'answers': [1, 0]}, {'order': 'AB', 'answers': [1]}, {'order': 'AB', 'answers': [2, 0]}):
        with pytest.raises(ValueError):
            svc.add_answer(bad)


def _fake_ros(monkeypatch):
    published = []

    class Node:
        def __init__(self, name):
            self.params = {}; self.subs = {}; self.logs = []

        def declare_parameter(self, name, default):
            self.params[name] = default

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self.params[name])

        def create_publisher(self, typ, topic, depth):
            return types.SimpleNamespace(publish=published.append)

        def create_subscription(self, typ, topic, cb, depth):
            self.subs[topic] = cb

        def get_logger(self):
            return types.SimpleNamespace(info=self.logs.append, warning=self.logs.append)

    class String:
        def __init__(self, data=''):
            self.data = data

    rclpy = types.ModuleType('rclpy'); rclpy_node = types.ModuleType('rclpy.node'); rclpy_node.Node = Node
    std_msgs = types.ModuleType('std_msgs'); std_msgs_msg = types.ModuleType('std_msgs.msg'); std_msgs_msg.String = String
    for name, mod in (('rclpy', rclpy), ('rclpy.node', rclpy_node), ('std_msgs', std_msgs), ('std_msgs.msg', std_msgs_msg)):
        monkeypatch.setitem(sys.modules, name, mod)
    return String, published


def test_ros2_node_callbacks(monkeypatch):
    String, published = _fake_ros(monkeypatch)
    pkg = pathlib.Path(__file__).resolve().parents[1] / 'integrations' / 'ros2' / 'quantum_mind_ros'
    monkeypatch.syspath_prepend(str(pkg))
    sys.modules.pop('quantum_mind_ros.human_model_node', None)
    from quantum_mind_ros.human_model_node import HumanModelNode
    node = HumanModelNode()
    node.on_query(String(''))
    assert json.loads(published[-1].data)['action'] == 'ask'
    for m in _answers(30, np.random.default_rng(1)):
        node.on_answer(String(json.dumps(m)))
    node.on_answer(String('not json'))                                   # logged, not raised
    node.on_query(String(json.dumps({'order': 'AB', 'error_cost': 1.5})))
    reply = json.loads(published[-1].data)
    assert reply['n_answers'] == 60 and reply['action'] in ('ask', 'act') and 'prediction' in reply
    assert any('ignored answer' in str(x) for x in node.logs) and any('refitted' in str(x) for x in node.logs)
