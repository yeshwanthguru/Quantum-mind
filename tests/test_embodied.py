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


def test_belief_slots_limits_and_fit():
    from quantum_mind.embodied import BayesSlot, QuantumSlot, TemperedSlot, sequence_batch
    vla = TinyVLA(seed=0).fit(make_dataset(800, 0), steps=800)
    amb = [s for s in make_dataset(600, 3) if s.ambiguous]
    b = sequence_batch(vla, amb, np.random.default_rng(0))
    args = (b['g0'], b['L'], b['ctx'], b['mask'])
    bayes = BayesSlot().posterior(*args)
    assert np.allclose(bayes.sum(1), 1) and (bayes[b['mask'] == 0] == 0).all()
    q = QuantumSlot()
    q.x = np.zeros(q.n_params)                                    # no rotation: exactly the Bayes slot
    assert np.allclose(q.posterior(*args), bayes, atol=1e-9)
    t = TemperedSlot()
    t.x = np.ones(t.n_params)
    assert np.allclose(t.posterior(*args), bayes, atol=1e-9)
    nll = lambda p: -np.mean(np.log(p[np.arange(len(b['y'])), b['y']] + 1e-9))
    assert nll(TemperedSlot().fit(b).posterior(*args)) <= nll(bayes) + 1e-6   # a fitted slot cannot be worse in-sample


def test_qiskit_circuit_reproduces_the_numpy_quantum_slot():
    pytest.importorskip('qiskit')
    from quantum_mind.embodied import QuantumSlot, sequence_batch
    from quantum_mind.embodied.qiskit_slot import qiskit_posterior
    vla = TinyVLA(seed=0).fit(make_dataset(600, 0), steps=600)
    amb = [s for s in make_dataset(300, 4) if s.ambiguous][:12]
    b = sequence_batch(vla, amb, np.random.default_rng(1))
    q = QuantumSlot()
    q.x = 0.5 * np.random.default_rng(2).standard_normal(q.n_params)
    ref = q.posterior(b['g0'], b['L'], b['ctx'], b['mask'])
    for i in range(len(amb)):
        sl = {k: v[i:i + 1] for k, v in b.items()}
        p, accept = qiskit_posterior(q, sl['g0'], sl['L'], sl['ctx'], sl['mask'])
        assert np.allclose(p, ref[i], atol=1e-8) and 0 < accept <= 1


def test_state_estimators_track_and_the_accelerometer_helps_under_an_unmodelled_push():
    from quantum_mind.embodied import MODES, JointEstimator, SensedMotor

    def err(mode, outage, dist, n=40):
        out = []
        for seed in range(n):
            rng = np.random.default_rng(seed)
            sm, est = SensedMotor(outage_at=outage, disturbance=dist), JointEstimator(mode)
            tgt = rng.uniform(-0.9, 0.9)
            z0 = sm.reset(rng.uniform(-1, 1), rng)
            est.reset(z0)
            th, om = z0, 0.0
            for _ in range(60):
                u = float(expert_command(tgt, th, om))
                th, om = est.step(sm.step(u), u)
            out.append(abs(sm.m.theta - tgt))
        return np.mean(out)
    assert all(err(m, None, 0.0) < 0.05 for m in MODES if m != 'kalman + MEMS')          # encoder present: all fine
    assert err('kalman + motor model', 1.0, 0.0) < err('kalman + MEMS + quantum', 1.0, 0.0)   # known plant: the model wins
    assert err('kalman + MEMS + quantum', 1.0, 3.0) < err('kalman + motor model', 1.0, 3.0)   # unmodelled push: the sensors win


def test_assignment_matches_the_exact_optimum_and_the_system_facade_runs():
    from quantum_mind.embodied import QuantumRobotVLA, QuantumSlot, assign_targets
    m, t = [-0.8, 0.1, 0.7], [0.65, -0.75, 0.15]
    exact, cost, _ = assign_targets(m, t, 'exact')
    assert exact == {0: 2, 1: 0, 2: 1} and cost == pytest.approx(0.15)
    assert assign_targets(m, t, 'annealing')[0] == exact
    sysm = QuantumRobotVLA.fit(make_dataset(600, 0), make_dataset(300, 1), slot=QuantumSlot(), steps=600)
    sysm.slot.x = np.zeros(sysm.slot.n_params)
    r = sysm.run(make_dataset(5, 2)[0], np.random.default_rng(0))
    assert 'success' in r and sysm.assign(m, t)[0] == exact
