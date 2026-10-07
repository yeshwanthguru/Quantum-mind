"""Adaptive conformal sets, online personalisation, trust-aware allocation and people populations."""
import itertools

import numpy as np
import pytest

from quantum_mind import fit, bootstrap, OnlinePersonModel
from quantum_mind.applications.allocation import trust_aware_allocation, expected_costs, decode
from quantum_mind.applications.calibration import AdaptiveConformalSets
from quantum_mind.applications.personalisation import PopulationPrior, PersonalisedHumanModel
from quantum_mind.envs import ClarificationEnv, TrustHandoverEnv, sample_people
from quantum_mind.envs.hri import default_domain
from quantum_mind.families.dynamics import OpenSystemBelief
from quantum_mind.families.order_effects import BayesOrderModel


# ------------------------------------------------------------------ adaptive conformal sets
def test_adaptive_conformal_meets_its_bound_when_the_person_changes():
    rng = np.random.default_rng(0)
    aci = AdaptiveConformalSets(alpha=0.1, gamma=0.02, calibration_scores=rng.uniform(0, 1, 20))
    model = np.array([0.6, 0.3, 0.1])                       # the robot's model never adapts
    for t in range(2000):
        truth = model if t < 700 else np.array([0.1, 0.3, 0.6])
        aci.update(model, rng.choice(3, p=truth))
    assert abs(aci.miscoverage - 0.1) <= aci.bound()


def test_adaptive_conformal_starts_cautious_and_asks():
    aci = AdaptiveConformalSets(alpha=0.1)
    assert aci.predict_set([0.9, 0.1]).all() and aci.should_ask([0.9, 0.1])     # no scores yet: every answer
    aci = AdaptiveConformalSets(alpha=0.5, calibration_scores=[0.05, 0.1, 0.2, 0.3])
    assert not aci.should_ask([0.95, 0.05])
    with pytest.raises(ValueError):
        AdaptiveConformalSets(alpha=1.5)


# ------------------------------------------------------------------ online personalisation
def test_personalisation_online_posterior_follows_the_person():
    prior = PopulationPrior(BayesOrderModel, mean=[0.0, 0.0, 0.0], cov=np.eye(3) * 0.5)
    svc = PersonalisedHumanModel(BayesOrderModel, prior, method='online', n_particles=400)
    truth = BayesOrderModel(pA=0.85, pB=0.3, rho=0.0).predict(None)
    rng = np.random.default_rng(2)
    for i in range(200):
        c = 'AB' if i % 2 else 'BA'
        svc.add('bob', c, int(rng.choice(4, p=truth[c])))
    lo, hi = svc.credible_interval('bob', 'pA', 0.95)
    assert lo <= 0.85 <= hi and hi - lo < 0.3
    assert abs(svc.predict('bob')['AB'][0] - truth['AB'][0]) < 0.08
    with pytest.raises(ValueError):
        svc.credible_interval('nobody', 'pA')
    with pytest.raises(ValueError):
        PersonalisedHumanModel(BayesOrderModel, prior, method='other')


def test_population_prior_online_uses_its_covariance():
    prior = PopulationPrior(BayesOrderModel, mean=[1.0, -1.0, 0.0], cov=np.diag([0.01, 0.01, 0.01]))
    post = prior.online(n_particles=500, rng=np.random.default_rng(0))
    assert np.allclose(post.particles.mean(axis=0), [1.0, -1.0, 0.0], atol=0.03)
    assert np.allclose(post.particles.std(axis=0), 0.1, atol=0.02)


# ------------------------------------------------------------------ trust-aware allocation
def _valid_assignments(na, nt):
    for owners in itertools.product(range(na), repeat=nt):
        x = np.zeros((na, nt), int)
        x[list(owners), range(nt)] = 1
        yield x


def test_allocation_energy_is_expected_cost_plus_workload():
    rng = np.random.default_rng(1)
    C, tau, r = rng.uniform(1, 3, (3, 3)), [0.9, 0.5, 0.2], [1.0, 0.5, 0.0]
    lam = 0.7
    q = trust_aware_allocation(C, tau, r, fail_cost=8, workload=lam)
    Ct = expected_costs(C, tau, r, 8)
    for x in _valid_assignments(3, 3):
        expected = float((Ct * x).sum() + lam * (x.sum(axis=1) ** 2).sum())
        assert q.energy(x.ravel()) == pytest.approx(expected)
    best = min(_valid_assignments(3, 3), key=lambda x: q.energy(x.ravel()))
    xb, eb = q.brute_force()
    assert eb == pytest.approx(q.energy(best.ravel()))
    assert all(len(v) for v in [decode(xb, 3, 3)]) and np.asarray(xb).reshape(3, 3).sum(axis=0).tolist() == [1, 1, 1]


def test_allocation_moves_robot_assisted_work_to_trusting_people():
    C = np.ones((2, 2))
    blind, _ = trust_aware_allocation(C, [0.5, 0.5], [1.0, 0.0], workload=1.0).brute_force()
    aware, _ = trust_aware_allocation(C, [0.95, 0.2], [1.0, 0.0], workload=1.0).brute_force()
    assert decode(aware, 2, 2)[0] == [0]
    assert np.asarray(blind).reshape(2, 2).sum() == 2
    with pytest.raises(ValueError):
        expected_costs(C, [1.2, 0.5], [1, 0])
    with pytest.raises(ValueError):
        expected_costs(C, [0.5], [1, 0])


# ------------------------------------------------------------------ people populations
def test_sample_people_from_every_source():
    rng = np.random.default_rng(0)
    data = {'AB': [40, 10, 20, 30], 'BA': [35, 15, 15, 35]}
    res = fit(BayesOrderModel, data, restarts=2)
    boot = bootstrap(res, data, n_boot=5)
    online = OnlinePersonModel(BayesOrderModel, prior=res, n_particles=50)
    prior = PopulationPrior(BayesOrderModel, mean=[0.0, 0.0, 0.0], cov=np.eye(3) * 0.2)
    for src in (boot, online, prior, res.model):
        people = sample_people(src, 6, rng)
        assert len(people) == 6 and all(isinstance(p, BayesOrderModel) for p in people)
    assert len({p.pA for p in sample_people(prior, 6, rng)}) == 6


def test_environments_draw_a_new_person_every_episode():
    people = [OpenSystemBelief(phi0=0.3), OpenSystemBelief(phi0=2.8)]
    env = TrustHandoverEnv(people=people)
    start = set()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        start.add(round(info['p_trust'], 6))
    assert len(start) == 2
    env = TrustHandoverEnv(people=lambda rng: OpenSystemBelief(phi0=float(rng.uniform(0, np.pi))))
    assert len({round(env.reset(seed=s)[1]['p_trust'], 6) for s in range(5)}) == 5
    cl = ClarificationEnv(people=[default_domain(1), default_domain(2)])
    models = set()
    for seed in range(20):
        cl.reset(seed=seed)
        models.add(id(cl.model))
    assert len(models) == 2
