"""The equations on the documentation's mathematics pages (docs/math) checked against the code.

Each test recomputes one equation by hand and compares it with the implementation, so the
documentation fails CI if the code changes without the mathematics being updated.
"""
import numpy as np
import pytest

from quantum_mind.core import projector
from quantum_mind.core.linalg import luders, sequence_probabilities, angles_to_unit


# --------------------------------------------------------------------------------- core (math/core.md)
def test_sequence_probability_is_norm_of_projector_product():
    psi = angles_to_unit([0.4, 1.1])
    PA, PB = projector([1, 0, 0]), projector([0.6, 0.8, 0.0])
    seq = sequence_probabilities(psi, {'A': PA, 'B': PB}, ['A', 'B'])
    I = np.eye(3)
    for (a, b), p in seq.items():
        Pa = PA if a else I - PA
        Pb = PB if b else I - PB
        assert p == pytest.approx(np.linalg.norm(Pb @ Pa @ psi) ** 2)
    p, post = luders(psi, PA)
    assert p == pytest.approx(np.linalg.norm(PA @ psi) ** 2) and np.allclose(post, PA @ psi / np.linalg.norm(PA @ psi))


def test_angles_to_unit_formula():
    th = np.array([0.3, 0.9, 1.4])
    v = angles_to_unit(th)
    expect = [np.cos(th[0]), np.sin(th[0]) * np.cos(th[1]), np.sin(th[0]) * np.sin(th[1]) * np.cos(th[2]),
              np.sin(th[0]) * np.sin(th[1]) * np.sin(th[2])]
    assert np.allclose(v, expect)


def test_bic_and_aic_formulas():
    from quantum_mind.core import fit
    from quantum_mind.families.order_effects import BayesOrderModel
    data = BayesOrderModel(pA=0.6, pB=0.4, rho=0.2).sample(None, 300, np.random.default_rng(0))
    r = fit(BayesOrderModel, data, restarts=3)
    assert r.bic == pytest.approx(r.k * np.log(r.n) - 2 * r.loglik)
    assert r.aic == pytest.approx(2 * r.k - 2 * r.loglik)


# ------------------------------------------------------------------- quantum-like (math/quantum_like.md)
def test_order_model_projections_and_qq_equality():
    from quantum_mind.families.order_effects import QuantumOrderModel, qq_statistic
    a, b, g = 0.9, 1.7, 0.6
    m = QuantumOrderModel(a=a, b=b, g=g, ranks=(1, 1))
    uA, uB = np.array([np.cos(a), np.sin(a), 0]), np.array([np.cos(b), np.sin(b) * np.cos(g), np.sin(b) * np.sin(g)])
    PA, PB, psi, I = np.outer(uA, uA), np.outer(uB, uB), np.array([1.0, 0, 0]), np.eye(3)
    ab = [np.linalg.norm(Q2 @ Q1 @ psi) ** 2 for Q1, Q2 in ((PA, PB), (PA, I - PB), (I - PA, PB), (I - PA, I - PB))]
    p = m.predict()
    assert np.allclose(p['AB'], ab)
    assert qq_statistic(p['AB'], p['BA']) == pytest.approx(0, abs=1e-12)


def test_4d_model_with_zero_angle_is_order_free():
    from quantum_mind.families.order_effects import QuantumOrderModel4D
    p = QuantumOrderModel4D(t1=0.7, t2=1.1, t3=0.4, phi=0.0).predict()
    assert np.allclose(p['AB'], p['BA'][[0, 2, 1, 3]])          # same joint table, cells relabelled


def test_interference_law():
    from quantum_mind.families.interference import InterferenceModel
    p1, p2, c, th = 0.9, 0.8, 0.4, 2.2
    A = c * p1 + (1 - c) * p2 + 2 * np.cos(th) * np.sqrt(c * p1 * (1 - c) * p2)
    B = c * (1 - p1) + (1 - c) * (1 - p2) + 2 * np.cos(th) * np.sqrt(c * (1 - p1) * (1 - c) * (1 - p2))
    assert InterferenceModel(p1=p1, p2=p2, c=c, theta=th).predict(None)['unknown'][0] == pytest.approx(A / (A + B))
    classical = InterferenceModel(p1=p1, p2=p2, c=c, theta=np.pi / 2).predict(None)['unknown'][0]
    assert classical == pytest.approx(c * p1 + (1 - c) * p2)


def test_open_system_belief_step_and_query():
    from quantum_mind.families.dynamics import OpenSystemBelief, MarkovBelief, final_yes
    phi0, ap, an, gam = 1.3, 0.7, 1.1, 0.25
    m = OpenSystemBelief(phi0=phi0, a_pos=ap, a_neg=an, gamma=gam)

    def ry(a):
        return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])
    v = np.array([np.cos(phi0 / 2), np.sin(phi0 / 2)])
    rho = np.outer(v, v)
    for e in (1, 0):
        R = ry(-ap if e else an)
        rho = R @ rho @ R.T
        rho[0, 1] *= 1 - gam
        rho[1, 0] *= 1 - gam
    assert final_yes(m, (1, 0), (1,)) == pytest.approx(rho[0, 0])
    mk = MarkovBelief(p0=0.4, up=0.3, down=0.2)
    assert final_yes(mk, (1, 0), (1,)) == pytest.approx((0.4 + 0.6 * 0.3) * (1 - 0.2))


def test_fock_space_concept_formula():
    from quantum_mind.families.concepts import FockSpaceConceptModel
    mA, mB, m2, k = 0.4, 0.6, 0.3, 0.5
    expect = m2 * mA * mB + (1 - m2) * ((mA + mB) / 2 + k * min(np.sqrt(mA * mB), np.sqrt((1 - mA) * (1 - mB))))
    assert FockSpaceConceptModel(m2=m2, kappa=k).predict({'x': (mA, mB)})['x'][0] == pytest.approx(expect)


def test_zeno_mean_dwell_time():
    from quantum_mind.families.perception import QuantumZenoBistableModel, mean_dwell_time
    g, dt = 1.3, 0.08
    assert mean_dwell_time(QuantumZenoBistableModel(g=g), dt) == pytest.approx(dt / np.sin(g * dt) ** 2, rel=1e-3)


# ---------------------------------------------------------------------- decision layer (math/robotics.md)
def test_ask_or_act_break_even():
    from quantum_mind.applications.robotics import ask_or_act
    Ca, Ce = 1.0, 4.0
    p_star = 1 - Ca / Ce
    assert ask_or_act(p_star + 0.01, ask_cost=Ca, error_cost=Ce)['action'] == 'act'
    assert ask_or_act(p_star - 0.01, ask_cost=Ca, error_cost=Ce)['action'] == 'ask'


def test_ensemble_uncertainty_decomposition():
    from quantum_mind.applications.robotics import HumanModelEnsemble
    from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel
    data = QuantumOrderModel4D().sample(None, 200, np.random.default_rng(2))
    ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel]).update(data, restarts=2)
    mix, unc = ens.predict('AB')

    def H(p):
        p = np.asarray(p)
        return float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1)), 0)))
    preds = [f.model.predict(None)['AB'] for f in ens.fits]
    assert unc['entropy_bits'] == pytest.approx(H(mix))
    assert unc['model_disagreement_bits'] == pytest.approx(H(mix) - sum(w * H(p) for w, p in zip(ens.weights, preds)))
    b = np.array([f.bic for f in ens.fits])
    w = np.exp(-0.5 * (b - b.min()))
    assert np.allclose(ens.weights, w / w.sum())


def test_calibration_measures_and_recalibrators():
    from quantum_mind.applications.calibration import (expected_calibration_error, brier_score, PlattScaling,
                                                       TemperatureScaling, SplitConformalClassifier)
    c = np.array([0.15, 0.25, 0.85, 0.95])
    y = np.array([0, 1, 1, 1])
    assert expected_calibration_error(c, y, bins=2) == pytest.approx(0.5 * abs(0.5 - 0.2) + 0.5 * abs(1.0 - 0.9))
    assert brier_score(c, y) == pytest.approx(np.mean((c - y) ** 2))
    rng = np.random.default_rng(0)
    cc = rng.uniform(0.05, 0.95, 400)
    pl = PlattScaling().fit(cc, rng.random(400) < cc)
    assert pl.transform([0.7])[0] == pytest.approx(1 / (1 + np.exp(-(pl.a * np.log(0.7 / 0.3) + pl.b))))
    P = np.array([[0.7, 0.2, 0.1]])
    ts = TemperatureScaling()
    ts.T = 2.0
    z = np.log(P) / 2.0
    assert np.allclose(ts.transform(P), np.exp(z) / np.exp(z).sum())
    Pc = rng.dirichlet(np.ones(3), 200)
    yc = rng.integers(0, 3, 200)
    cp = SplitConformalClassifier(alpha=0.1).fit(Pc, yc)
    s = np.sort(1 - Pc[np.arange(200), yc])
    assert cp.qhat == pytest.approx(s[int(np.ceil(201 * 0.9)) - 1])


def test_meta_calibrated_gate_posterior_offset():
    from quantum_mind.applications.orchestration import MetaCalibratedGate, Module
    mu, n0 = np.array([0.1, 0.3]), np.array([4.0, 4.0])
    g = MetaCalibratedGate([Module('a', 1.0), Module('b', 2.0)], mu, n0, lam=0.0, explore=0.0)
    obs = [(0.9, 1), (0.8, 0), (0.95, 1)]
    for c, y in obs:
        g.update(0, c, y)
    expect = (n0[0] * mu[0] + sum(c - y for c, y in obs)) / (n0[0] + len(obs))
    assert g.offsets[0] == pytest.approx(expect)


def test_value_of_information_formula():
    from quantum_mind.applications.questioning import IndependentAnswerModel, QuestionPlanner
    m = IndependentAnswerModel(['h1', 'h2'], ['q'], [[0.9], [0.2]])
    pl = QuestionPlanner(m, prior=[0.6, 0.4], ask_cost=0.1, error_cost=5.0)
    prior = np.array([0.6, 0.4])
    py = prior @ np.array([0.9, 0.2])
    post_y, post_n = prior * [0.9, 0.2] / py, prior * [0.1, 0.8] / (1 - py)
    voi = 5 * (1 - prior.max()) - (py * 5 * (1 - post_y.max()) + (1 - py) * 5 * (1 - post_n.max()))
    assert pl.value_of_information('q', []) == pytest.approx(voi)


def test_handover_expected_costs():
    from quantum_mind.applications.handover import TrustAwareHandover
    from quantum_mind.families.dynamics import MarkovBelief
    pol = TrustAwareHandover(MarkovBelief(p0=0.7, up=0.3, down=0.3), fail_cost=10, slow_cost=1, ask_cost=0.5,
                             wait_cost=0.3, slow_factor=0.5)
    p, Cf, Cs, s = 0.7, 10, 1, 0.5
    c = pol.decide()['costs']
    assert c['handover'] == pytest.approx(Cf * (1 - p))
    assert c['slow_handover'] == pytest.approx(Cs + s * Cf * (1 - p))
    assert c['ask'] == pytest.approx(0.5 + (1 - p) * min(Cf, Cs + s * Cf))
    assert c['wait'] == pytest.approx(0.3 + min(Cf * (1 - p), Cs + s * Cf * (1 - p)))


# ----------------------------------------------------------------------- perception (math/perception.md)
def test_adapters():
    from quantum_mind.applications.fusion import detector_likelihood, direction_likelihood
    s, T, eps = np.array([0.6, 0.3, 0.1]), 2.0, 1e-3
    v = s ** (1 / T) + eps
    assert np.allclose(detector_likelihood(s, ['a', 'b', 'c'], temperature=T, floor=eps), v / v.sum())
    d, targets, k = np.array([1.0, 0.2]), {'a': [1, 0], 'b': [0, 1]}, 5.0
    cos = [d @ np.array(t) / np.linalg.norm(d) / np.linalg.norm(t) for t in targets.values()]
    v = np.exp(k * (np.array(cos) - 1)) + eps
    assert np.allclose(direction_likelihood(d, targets, ['a', 'b'], kappa=k, floor=eps), v / v.sum())


def test_dempster_shafer_rule():
    from quantum_mind.applications.fusion import dempster_shafer_fusion
    L, r = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]), np.array([0.8, 0.6])
    m, omega = np.zeros(3), 1.0
    for li, ri in zip(L, r):
        mi, wi = ri * li / li.sum(), 1 - ri
        new = m * mi + m * wi + omega * mi
        z = new.sum() + omega * wi
        m, omega = new / z, omega * wi / z
    out, ign = dempster_shafer_fusion(L, r)
    assert np.allclose(out, m + omega / 3) and ign == pytest.approx(omega)


def test_quantum_fusion_reduces_to_bayes_at_zero_angle():
    from quantum_mind.applications.intent import QuantumIntentResolver
    L1, L2 = np.array([0.7, 0.2, 0.1]), np.array([0.25, 0.6, 0.15])
    q = QuantumIntentResolver(['a', 'b', 'c'])
    q.add_cue('x', L1, 0.0)
    q.add_cue('y', L2, 0.0)
    bayes = L1 * L2 / (L1 * L2).sum()
    assert np.allclose(q.posterior(['x', 'y']), bayes) and np.allclose(q.posterior(['y', 'x']), bayes)


# --------------------------------------------------------------------------- learning (math/learning.md)
def test_tensor_train_storage_formula():
    from quantum_mind.inspired.tensor_layers import TTMatrix
    W = np.random.default_rng(0).normal(size=(64, 256))
    tt = TTMatrix.from_dense(W, (4, 4, 4), (8, 8, 4), max_rank=5)
    r = [1] + tt.ranks + [1]
    assert tt.n_params == sum(r[k] * m * n * r[k + 1] for k, (m, n) in enumerate(zip((4, 4, 4), (8, 8, 4))))


def test_policy_probabilities_are_readout_marginals():
    from quantum_mind.quantum.policy import VariationalPolicy
    pi = VariationalPolicy(n_features=3, n_actions=3, layers=1, seed=2)
    s = np.array([[0.2, -0.5, 0.9]])
    P = pi.circuit.probabilities(pi.weights, s * pi.scale)[0]
    idx = np.arange(len(P)) & 3                       # two read-out qubits
    M = np.array([P[idx == a].sum() for a in range(3)])
    assert np.allclose(pi.probabilities(s)[0], M / M.sum())


def test_amplitude_exploration_rotation():
    from quantum_mind.inspired.exploration import AmplitudeExploration
    ex = AmplitudeExploration(1, 3, k=1.0, max_step=0.3, floor=0.0)
    q = np.array([1.0, 0.0, 0.0])
    psi = ex.amp[0].copy()
    ex.update(0, 0, 0.0, q)
    adv = q[0] - (psi ** 2) @ q
    phi = np.arcsin(psi[0]) + np.clip(adv, -0.3, 0.3)
    assert ex.amp[0][0] == pytest.approx(np.sin(phi))


# ------------------------------------------------------------------- quantum and inspired (math/*.md)
def test_grover_success_probability():
    from quantum_mind.quantum import grover
    n, marked = 5, [3, 17]
    res = grover(n, marked, iterations=2)
    theta = np.arcsin(np.sqrt(len(marked) / 2 ** n))
    assert res.success == pytest.approx(np.sin(5 * theta) ** 2, abs=1e-9)


def test_task_allocation_energy():
    from quantum_mind.problems import task_allocation
    C = np.array([[2.0, 7.0], [6.0, 3.0]])
    q = task_allocation(C, penalty=20.0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 2, 4)
        X = x.reshape(2, 2)
        expect = (C * X).sum() + 20.0 * ((X.sum(0) - 1) ** 2).sum()
        assert q.energy(x) == pytest.approx(expect)


def test_sqa_coupling_formula():
    P, T, G = 16, 0.05, 1.5
    Jp = -0.5 * P * T * np.log(np.tanh(G / (P * T)))
    assert Jp > 0 and -0.5 * P * T * np.log(np.tanh(0.1 / (P * T))) > Jp      # grows as the field falls
