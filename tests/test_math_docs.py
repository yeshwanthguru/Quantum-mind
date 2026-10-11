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


def test_extension_formulas():
    """Online reweighting, ESS, information-gain bound and model posterior as on docs/math/core.md."""
    from quantum_mind.core import OnlinePersonModel, information_gain, model_posterior
    from quantum_mind.families.order_effects import BayesOrderModel, QuantumOrderModel4D
    person = OnlinePersonModel(BayesOrderModel, n_particles=50, resample_below=0.0, rng=np.random.default_rng(0))
    p = np.array([pr['AB'][2] for pr in person._preds])
    person.update('AB', 2)
    assert np.allclose(person.weights, p / p.sum())
    assert np.isclose(person.ess, 1 / np.sum((p / p.sum()) ** 2))
    models = [BayesOrderModel(pA=0.9, pB=0.1), BayesOrderModel(pA=0.1, pB=0.9), QuantumOrderModel4D()]
    assert 0 <= information_gain(models, 'AB') <= np.log2(3)
    y = {'AB': [3, 1, 0, 2]}
    ll = np.array([np.dot(y['AB'], np.log(m.predict(None)['AB'])) for m in models])
    w = np.array([0.2, 0.3, 0.5])
    expected = w * np.exp(ll - ll.max())
    assert np.allclose(model_posterior(models, y, prior=w), expected / expected.sum())


# --------------------------------------------------------------------------- belief (math/belief.md)
def _random_cobra(seed=0, d=3, m=3):
    from quantum_mind.belief import COBRA
    model = COBRA(d, m, restarts=1, steps=1, seed=seed)
    model.theta = model.theta + 0.3 * np.random.default_rng(seed + 5).standard_normal(model.theta.size)
    return model


def test_cobra_keeps_a_valid_density_matrix():
    model = _random_cobra()
    x = np.random.default_rng(1).normal(size=(20, 5, 3))
    rho = model.trace(x).rho
    assert np.allclose(np.einsum('ntii->nt', rho).real, 1.0)
    assert np.allclose(rho, rho.conj().transpose(0, 1, 3, 2), atol=1e-10)
    assert np.linalg.eigvalsh(rho).min() > -1e-9


def test_cobra_without_rotation_is_the_bayes_filter():
    model = _random_cobra(seed=2)
    p = model._unpack(model.theta)
    p['gens'][...] = 0.0                                       # H_k = 0
    x = np.random.default_rng(3).normal(size=(15, 4, 3))
    prior = np.exp(p['prior'] - p['prior'].max())
    logpost = np.log(prior / prior.sum()) + (x @ p['W'].T + p['b']).sum(axis=1)
    bayes = np.exp(logpost - logpost.max(axis=1, keepdims=True))
    bayes /= bayes.sum(axis=1, keepdims=True)
    assert np.allclose(model.predict_proba(x), bayes, atol=1e-8)
    assert np.allclose(model.predict_proba(x[:, ::-1]), bayes, atol=1e-8)       # order cannot matter


def test_cobra_full_dephasing_gives_a_diagonal_belief():
    model = _random_cobra(seed=4)
    model._unpack(model.theta)['gamma'][...] = 40.0              # g -> 1
    x = np.random.default_rng(5).normal(size=(10, 4, 3))
    assert model.coherence(x).max() < 1e-9


def test_cobra_is_order_sensitive_with_rotation():
    model = _random_cobra(seed=6)
    x = np.random.default_rng(7).normal(size=(10, 4, 3))
    assert np.abs(model.predict_proba(x) - model.predict_proba(x[:, ::-1])).max() > 1e-3


def test_cobra_online_filter_matches_the_batch_filter():
    model = _random_cobra(seed=8)
    x = np.random.default_rng(9).normal(size=(1, 6, 3))
    belief = model.start()
    for t in range(6):
        p = belief.update(x[0, t])
    assert np.allclose(p, model.predict_proba(x)[0], atol=1e-10)
    assert belief.frames == 6


def test_min_cost_action_hedges_when_the_belief_is_spread_out():
    from quantum_mind.belief import min_cost_action
    cost = np.array([[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]])
    assert min_cost_action([0.97, 0.02, 0.01], cost)[0] == 0
    assert min_cost_action([0.7, 0.2, 0.1], cost)[0] == 2


def test_trust_aware_allocation_energy():
    from quantum_mind.applications.allocation import trust_aware_allocation
    C = np.array([[2.0, 7.0], [6.0, 3.0]])
    tau, r, F, lam, A = np.array([0.9, 0.3]), np.array([1.0, 0.2]), 8.0, 0.5, 20.0
    q = trust_aware_allocation(C, trust=tau, reliance=r, fail_cost=F, workload=lam, penalty=A)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 2, 4)
        X = x.reshape(2, 2)
        expect = ((C + F * r[None, :] * (1 - tau)[:, None]) * X).sum() + lam * (X.sum(1) ** 2).sum() + A * ((X.sum(0) - 1) ** 2).sum()
        assert q.energy(x) == pytest.approx(expect)
