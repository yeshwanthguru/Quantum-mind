"""One-motor VLA prototype: dynamics, scene generator, hand-written gradients, closed-loop behaviour."""
import numpy as np
import pytest

from quantum_mind.embodied import (BCBaseline, OneMotor, TinyVLA, bc_plan, calibrate_gate, clarification, expert_command,
                                   fuse, make_dataset, oracle_plan, run_episode, sample_scene, vla_plan)
from quantum_mind.embodied.vla import _batch


def test_motor_reaches_target_with_the_expert_and_respects_limits():
    m = OneMotor()
    rng = np.random.default_rng(0)
    th, om = m.reset(-0.8, rng)
    for _ in range(80):
        th, om = m.step(expert_command(0.6, th, om))
    assert abs(m.theta - 0.6) < 0.02
    th, om = m.reset(0.0, rng)
    m.step(50.0)                                                  # command is clipped to [-1, 1]
    assert abs(m.omega) <= m.g * 1.0 * m.dt + 1e-9


def test_scene_ground_truth_is_consistent():
    rng = np.random.default_rng(1)
    for _ in range(300):
        s = sample_scene(rng)
        assert s.target in s.valid and set(s.valid) <= set(s.cells)
        text2, word = clarification(s)
        assert word in ('leftmost', 'rightmost', 'red', 'green', 'blue')
    assert any(s.ambiguous for s in make_dataset(100, 0)) and any(not s.ambiguous for s in make_dataset(100, 0))


def test_hand_written_gradients_match_finite_differences():
    m = TinyVLA(seed=1)
    b = _batch(make_dataset(30, 0), np.arange(8), np.random.default_rng(0), 16)
    _, g = m.loss_and_grads(*b)
    for k, v in m.p.items():
        for i in list(np.ndindex(*v.shape))[:4]:
            o = v[i]
            v[i] = o + 1e-6; lp = m.loss_and_grads(*b)[0]
            v[i] = o - 1e-6; lm = m.loss_and_grads(*b)[0]
            v[i] = o
            assert (lp - lm) / 2e-6 == pytest.approx(g[k][i], rel=1e-4, abs=1e-8)


@pytest.fixture(scope='module')
def trained():
    tr = make_dataset(1500, 0)
    return TinyVLA(seed=0).fit(tr, steps=1500), tr


def test_grounding_is_a_distribution_and_ambiguity_spreads_it(trained):
    vla, _ = trained
    test = make_dataset(200, 5)
    g = vla.ground(np.stack([s.image for s in test]), np.stack([s.text for s in test]))
    assert np.allclose(g.sum(1), 1)
    amb = np.array([s.ambiguous for s in test])
    ent = -(g * np.log(g + 1e-12)).sum(1)
    assert ent[amb].mean() > ent[~amb].mean() + 0.2               # it knows when it does not know


def test_closed_loop_beats_acting_blind_and_asking_helps(trained):
    vla, _ = trained
    test = make_dataset(150, 7)
    gate = calibrate_gate(vla, make_dataset(500, 8), alpha=0.02)

    def success(plan):
        rng = np.random.default_rng(3)
        rows = [run_episode(s, rng, plan) for s in test]
        return np.mean([r['success'] for r in rows]), np.mean([r['asked'] for r in rows])
    s_act, q_act = success(vla_plan(vla))
    s_ask, q_ask = success(vla_plan(vla, gate))
    s_orc, _ = success(oracle_plan)
    assert s_orc == 1.0 and q_act == 0 and 0 < q_ask < 1
    assert s_ask > s_act + 0.1


def test_fuse_and_baseline_interface(trained):
    g1, g2 = np.array([.5, .5, 0]), np.array([1., 0, 0])
    assert np.allclose(fuse(g1, g2), [1, 0, 0])
    assert np.allclose(fuse(np.array([1., 0, 0]), np.array([0, 1., 0])), [0, 1, 0])   # contradictory evidence: trust the answer
    bc = BCBaseline(seed=0).fit(make_dataset(100, 0), steps=20)
    s = make_dataset(1, 0)[0]
    assert -1 <= bc.command(s.image, s.text, 0.0, 0.0) <= 1
    assert callable(bc_plan(bc))
