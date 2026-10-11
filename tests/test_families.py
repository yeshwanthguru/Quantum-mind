import numpy as np
from quantum_mind.data import CLINTON_GORE


def test_order_effects_qq_and_nesting():
    from quantum_mind.families.order_effects import (QuantumOrderModel, QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel,
                                              qq_statistic, rates)
    rng = np.random.default_rng(1)
    for _ in range(20):
        for m in (QuantumOrderModel(a=rng.uniform(0, 3), b=rng.uniform(0, 3), g=rng.uniform(0, 3), ranks=(1, 2)),
                  QuantumOrderModel4D(t1=rng.uniform(0, 3), t2=rng.uniform(0, 3), t3=rng.uniform(0, 3), phi=rng.uniform(0, 3))):
            p = m.predict()
            assert np.isclose(p['AB'].sum(), 1) and abs(qq_statistic(p['AB'], p['BA'])) < 1e-10
    b = rates(BayesOrderModel(pA=0.3, pB=0.6, rho=0.4).predict())          # Bayes: no order effect
    assert np.isclose(b[0], b[2]) and np.isclose(b[1], b[3])
    q = QuantumOrderModel4D(phi=0.0)
    # phi = 0 has no order effect
    pq = q.predict(); r = rates(pq)
    assert np.isclose(r[0], r[2]) and np.isclose(r[1], r[3])
    a = AnchoringOrderModel(pA=0.5, pB=0.68, wA=0.2, wB=0.5).predict()
    assert abs(qq_statistic(a['AB'], a['BA'])) > 1e-3


def test_order_effects_fit_clinton_gore_rates():
    from scipy.optimize import least_squares
    from quantum_mind.families.order_effects import QuantumOrderModel4D, rates
    r = CLINTON_GORE['rates']; t = np.array([r['A_first'], r['B_first'], r['A_second'], r['B_second']])
    best = min((least_squares(lambda x: rates(QuantumOrderModel4D(t1=x[0], t2=x[1], t3=x[2], phi=x[3]).predict()) - t,
                              np.random.default_rng(s).uniform(0, 3, 4)) for s in range(30)), key=lambda z: z.cost)
    assert best.cost < 1e-8


def test_qlbn_reduces_to_classical_and_normalises():
    from quantum_mind.families.qlbn import BayesNet, classical_marginal, quantum_like_marginal
    net = BayesNet({'O': ['w', 'l'], 'P': ['y', 'n']}, {'P': ['O']},
                   {'O': {(): {'w': .5, 'l': .5}}, 'P': {('w',): {'y': .69, 'n': .31}, ('l',): {'y': .59, 'n': .41}}})
    assert np.isclose(quantum_like_marginal(net, 'P', None, [0, np.pi / 2])['y'], classical_marginal(net, 'P')['y'])
    m = quantum_like_marginal(net, 'P', None, [0, 2.0]); assert np.isclose(sum(m.values()), 1)
    assert np.isclose(quantum_like_marginal(net, 'P', {'O': 'w'})['y'], 0.69)


def test_dynamics_markov_obeys_total_probability_quantum_does_not():
    from quantum_mind.families.dynamics import MarkovWalk, QuantumWalk, MarkovBelief, OpenSystemBelief, question_effect
    d = {'s': ('single', 1.5), 'j': ('joint', 0.5, 1.5)}
    mk = MarkovWalk(mu=0.7, gamma=5).predict(d); qw = QuantumWalk(mu=3, sigma=3).predict(d)
    assert np.allclose(mk['j'].reshape(3, 3).sum(0), mk['s'])
    assert np.abs(qw['j'].reshape(3, 3).sum(0) - qw['s']).sum() > 0.01
    ev = (1, 1, 0, 1, 0, 1)
    assert abs(question_effect(MarkovBelief(), ev, 5, 2)) < 1e-12
    assert abs(question_effect(OpenSystemBelief(gamma=0.3), ev, 5, 2)) > 0.01
    assert abs(question_effect(OpenSystemBelief(gamma=1.0), ev, 5, 2)) < 1e-12


def test_open_system_walk_probabilities():
    from quantum_mind.families.dynamics import OpenSystemWalk
    p = OpenSystemWalk(mu=2, sigma=2, lam=1.0, n_states=11).predict({'s': ('single', 1.0), 'j': ('joint', 0.3, 1.0)})
    assert np.isclose(p['s'].sum(), 1) and np.isclose(p['j'].sum(), 1)


def test_robotics_application():
    from quantum_mind.applications.robotics import domain_models, estimate_unprimed_rates, HumanModelEnsemble, HRI_DOMAINS
    from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, rates
    for d in HRI_DOMAINS:
        m = domain_models(d)
        r_ql, r_an = rates(m['QL'].predict()), rates(m['Anchoring'].predict())
        assert np.allclose(r_ql[:2], r_an[:2], atol=0.01)
    rng = np.random.default_rng(0); gen = domain_models('object_clarification')['QL']
    a, b = estimate_unprimed_rates('split', gen, 4000, rng)
    assert abs(b - gen.predict()['BA'][:2].sum()) < 0.05
    ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel]).update(gen.sample(None, 500, rng))
    p, unc = ens.predict('AB')
    assert np.isclose(p.sum(), 1) and np.isclose(sum(s['weight'] for s in ens.summary()), 1) and unc['total_bits'] > 0
