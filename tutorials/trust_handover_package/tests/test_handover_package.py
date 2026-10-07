"""Tests of the quantum_handover package (simulated people only)."""
import importlib.util
import json
import sys
import types

import numpy as np
import pytest

from quantum_handover import POLICIES, make_policy, person_state, quantum_like_people, markov_people, run_benchmark, to_markdown
from quantum_handover.ros2_node import HandoverController
from quantum_mind.applications.handover import simulate_handover_session
from quantum_mind.families.dynamics import OpenSystemBelief


def test_policies_decide_and_reject_unknown_names():
    for name in POLICIES:
        assert make_policy(name).decide()['action'] in ('handover', 'slow_handover', 'ask', 'wait')
    with pytest.raises(ValueError):
        make_policy('psychic')


def test_people_are_reproducible_and_varied():
    a = quantum_like_people(5, np.random.default_rng(0))
    b = quantum_like_people(5, np.random.default_rng(0))
    assert [p.phi0 for p in a] == [p.phi0 for p in b] and len({p.phi0 for p in a}) == 5
    assert len({p.p0 for p in markov_people(5, np.random.default_rng(1))}) == 5


def test_benchmark_is_deterministic_and_model_based_policies_beat_fixed_ones():
    r1 = run_benchmark(n_people=60, n_steps=20, seed=0)
    assert r1 == run_benchmark(n_people=60, n_steps=20, seed=0)
    for rows in r1.values():
        best_fixed = min(rows['always hand over']['cost'], rows['always slow']['cost'])
        assert rows['quantum-like']['cost'] < best_fixed and rows['markov']['cost'] < best_fixed
    assert rows['always hand over']['asks'] == 0
    table = to_markdown(r1)
    assert table.count('**') == 4                       # one best policy per population


def test_controller_follows_events_like_the_session_simulator():
    ctl = HandoverController('quantum-like')
    first = json.loads(ctl.decision())
    assert set(first) == {'action', 'p_trust', 'costs'}
    after_yes = json.loads(ctl.handle(json.dumps({'answer': 'yes'})))
    assert after_yes['p_trust'] == 1.0 and after_yes['action'] == 'handover'
    after_drop = json.loads(ctl.handle(json.dumps({'outcome': 'dropped'})))
    assert after_drop['p_trust'] < 1.0
    assert json.loads(ctl.handle(json.dumps({'event': 'reset'})))['p_trust'] == first['p_trust']
    with pytest.raises(ValueError):
        ctl.handle(json.dumps({'outcome': 'teleported'}))


def test_ros2_node_with_stand_in_rclpy(monkeypatch):
    published, subs = [], {}

    class Node:
        def __init__(self, name):
            self.name = name

        def create_publisher(self, typ, topic, depth):
            return types.SimpleNamespace(publish=lambda m: published.append((topic, m.data)))

        def create_subscription(self, typ, topic, cb, depth):
            subs[topic] = cb

        def get_logger(self):
            return types.SimpleNamespace(warning=lambda s: published.append(('warning', s)))

        def destroy_node(self):
            pass

    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: subs['/handover/events'](types.SimpleNamespace(data=json.dumps({'outcome': 'taken'})))
    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = Node
    std = types.ModuleType('std_msgs')
    msg = types.ModuleType('std_msgs.msg')
    msg.String = type('String', (), {'data': ''})
    for name, mod in {'rclpy': rclpy, 'rclpy.node': node_mod, 'std_msgs': std, 'std_msgs.msg': msg}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    from quantum_handover.ros2_node import main
    main([])
    assert [t for t, _ in published] == ['/handover/decision', '/handover/decision']
    assert json.loads(published[1][1])['p_trust'] > 0


@pytest.mark.skipif(importlib.util.find_spec('pybullet') is None or importlib.util.find_spec('PIL') is None,
                    reason='simulation extra not installed')
def test_simulation_matches_the_session_simulator(tmp_path):
    from quantum_handover.sim import run_session, side_by_side, save_gif
    person = dict(phi0=1.9, a_pos=0.8, a_neg=1.2, gamma=0.3)
    frames, log = run_session(make_policy('quantum-like'), person_state(OpenSystemBelief(**person)),
                              np.random.default_rng(0), n_steps=6, width=64, height=48)
    ref = simulate_handover_session(make_policy('quantum-like'), person_state(OpenSystemBelief(**person)),
                                    np.random.default_rng(0), 6)
    assert [e['action'] for e in log] == ref['actions']
    assert log[-1]['cost'] == pytest.approx(ref['cost'])
    assert frames[0].shape == (48, 64, 3)
    out = save_gif(side_by_side(frames, frames[:3]), tmp_path / 'h.gif')
    assert out.stat().st_size > 1000
