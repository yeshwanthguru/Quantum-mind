"""Robot decision layer: calibration, orchestration, adaptive questioning, personalisation, hand-over."""
import numpy as np
import pytest

from qlcog.applications.calibration import (expected_calibration_error, reliability_curve, PlattScaling,
                                            TemperatureScaling, IsotonicCalibration, SplitConformalClassifier,
                                            clopper_pearson, CalibrationMonitor, brier_score)
from qlcog.applications.orchestration import (Module, ConfidenceGate, MetaCalibratedGate, RoutingTask,
                                              simulate_routing, meta_fit_offsets)
from qlcog.applications.questioning import ProjectiveAnswerModel, IndependentAnswerModel, QuestionPlanner
from qlcog.applications.personalisation import PopulationPrior, fit_map, PersonalisedHumanModel
from qlcog.applications.handover import TrustAwareHandover, simulate_handover_session
from qlcog.applications.robotics import risk_aware_ask_or_act
from qlcog.core import fit, projector
from qlcog.families.dynamics import OpenSystemBelief, MarkovBelief
from qlcog.families.order_effects import BayesOrderModel


def test_calibration_measures_and_recalibrators():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.95, 6000)
    y = rng.random(6000) < p
    assert expected_calibration_error(p, y) < 0.03                       # calibrated by construction
    over = np.clip(p + 0.15, 0, 1)
    assert expected_calibration_error(over, y) > 0.1
    for cal in (PlattScaling(), IsotonicCalibration()):
        assert expected_calibration_error(cal.fit(over, y).transform(over), y) < expected_calibration_error(over, y)
    r = reliability_curve(p, y, bins=5)
    assert r['count'].sum() == 6000 and len(r['accuracy']) == 5
    assert brier_score(p, y) < brier_score(over, y)
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [1])


def test_temperature_scaling_and_conformal_coverage():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4000, 4)) * 2
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=q) for q in P])
    sharp = P ** 2 / (P ** 2).sum(1, keepdims=True)                        # over-confident version
    assert abs(TemperatureScaling().fit(sharp[:2000], y[:2000]).T - 2.0) < 0.3
    cp = SplitConformalClassifier(alpha=0.1).fit(P[:2000], y[:2000])
    sets = cp.predict_sets(P[2000:])
    assert sets[np.arange(2000), y[2000:]].mean() >= 0.87                 # at least about 90% coverage


def test_clopper_pearson_and_monitor():
    lo, hi = clopper_pearson(8, 10)
    assert lo < 0.8 < hi and clopper_pearson(0, 0) == (0.0, 1.0)
    m = CalibrationMonitor(min_count=5)
    for _ in range(50):
        m.update(0.95, 0)                                                  # claims 0.95, always wrong
    est, lower = m.estimate(0.95)
    assert est == 0.0 and lower == 0.0 and m.ece() > 0.9


def test_confidence_gate_cost_calibration_and_budget():
    mods = [Module('cheap', 1.0), Module('expensive', 5.0)]
    assert ConfidenceGate(mods, lam=0.1).select([0.7, 0.9]) == 0          # 0.2 better is not worth 0.4
    assert ConfidenceGate(mods, lam=0.01).select([0.7, 0.9]) == 1

    class Halve:
        def transform(self, c):
            return np.asarray(c) * 0.5
    assert ConfidenceGate(mods, lam=0.0, calibrators=[None, Halve()]).select([0.7, 0.9]) == 0
    g = ConfidenceGate(mods, lam=0.0, budget=1.5, window=10)
    choices = [g.select([0.1, 0.9]) for _ in range(30)]
    assert np.mean([mods[k].cost for k in choices]) < 2.5                   # the budget keeps the average down
    with pytest.raises(ValueError):
        ConfidenceGate(mods, calibrators=[None])


def test_meta_calibrated_gate_learns_faster_than_from_scratch():
    rng = np.random.default_rng(0)
    logs = []
    for _ in range(30):
        task = RoutingTask.sample(rng)
        log = []
        for _ in range(300):
            p, c = task.step(rng)
            k = int(rng.integers(4))
            log.append((k, c[k], rng.random() < p[k]))
        logs.append(log)
    mean, n0 = meta_fit_offsets(logs)
    assert np.allclose(mean, RoutingTask.MEAN_OFFSETS, atol=0.06)
    early = {}
    for name, pm, pn in (('scratch', np.zeros(4), 1.0), ('meta', mean, n0)):
        runs = []
        for i in range(120):
            task = RoutingTask.sample(np.random.default_rng(1000 + i))
            gate = MetaCalibratedGate(task.modules(), pm, pn, seed=i)
            runs.append(simulate_routing(gate, task, 40, np.random.default_rng(2000 + i))['success'][:25].mean())
        early[name] = np.mean(runs)
    assert early['meta'] > early['scratch']


def _toy_projective():
    e = np.eye(3)
    states = {'h1': np.array([1.0, 0.2, 0.0]), 'h2': np.array([0.1, 1.0, 0.3]), 'h3': np.array([0.2, 0.1, 1.0])}
    projs = {'q1': projector(e[0]), 'q2': projector([0.6, 0.8, 0.0]), 'q3': projector([0.0, 0.5, 0.8])}
    return ProjectiveAnswerModel(states, projs)


def test_answer_models_and_planner():
    m = _toy_projective()
    # order dependence: the likelihood of the same two answers depends on the order asked
    assert abs(m.likelihood('h2', [('q1', 1), ('q2', 1)]) - m.likelihood('h2', [('q2', 1), ('q1', 1)])) > 1e-3
    blind = m.order_free()
    assert abs(blind.likelihood('h2', [('q1', 1), ('q2', 1)]) - blind.likelihood('h2', [('q2', 1), ('q1', 1)])) < 1e-12
    pl = QuestionPlanner(m, ask_cost=0.1, error_cost=5.0)
    assert all(pl.value_of_information(q, []) >= -1e-12 for q in m.questions)
    assert pl.posterior([]).sum() == pytest.approx(1.0)
    out = pl.simulate(m, 'h1', np.random.default_rng(0))
    assert out['choice'] in m.hypotheses and out['total_cost'] >= out['cost']
    certain = QuestionPlanner(IndependentAnswerModel(['a', 'b'], ['q'], [[0.99], [0.01]]), prior=[0.999, 0.001])
    assert certain.decide([])[0] == 'act'
    with pytest.raises(ValueError):
        IndependentAnswerModel(['a'], ['q'], [[0.5, 0.5]])


def test_partial_pooling_shrinks_with_little_data():
    prior = PopulationPrior(BayesOrderModel, mean=[0.0, 0.0, 0.0], cov=np.eye(3) * 0.2)
    few = {'AB': np.array([4, 0, 0, 0]), 'BA': np.array([4, 0, 0, 0])}
    many = {k: v * 200 for k, v in few.items()}
    p_few = fit_map(BayesOrderModel, few, prior).model.pA
    p_many = fit_map(BayesOrderModel, many, prior).model.pA
    p_mle = fit(BayesOrderModel, many).model.pA
    assert 0.5 < p_few < p_many and abs(p_many - p_mle) < 0.05
    svc = PersonalisedHumanModel(BayesOrderModel, prior)
    assert np.allclose(svc.predict('new')['AB'].sum(), 1.0)
    svc.add('ann', 'AB', [1, 1])
    assert svc.predict('ann')['AB'][0] > svc.predict('new')['AB'][0]
    with pytest.raises(ValueError):
        PopulationPrior(BayesOrderModel, [0.0], np.eye(1))


def test_trust_aware_handover_and_risk():
    m = OpenSystemBelief(phi0=0.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
    pol = TrustAwareHandover(m)
    assert pol.decide()['action'] == 'handover'
    pol.observe(0)
    pol.observe(0)
    assert pol.decide()['action'] in ('slow_handover', 'ask')
    pol.answer(1)
    assert pol.p_trust == 1.0
    out = simulate_handover_session(TrustAwareHandover(m), TrustAwareHandover(m), np.random.default_rng(0), 15)
    assert len(out['actions']) == 15 and out['cost'] >= 0
    mk = TrustAwareHandover(MarkovBelief(p0=0.4, up=0.3, down=0.3))
    mk.observe(1)
    assert mk.p_trust > 0.4
    assert risk_aware_ask_or_act(0.95, 'low')['action'] == 'act'
    assert risk_aware_ask_or_act(0.95, 'high')['action'] == 'ask'
    assert risk_aware_ask_or_act(0.95, 'low', p_lower=0.3)['action'] == 'ask'
    assert risk_aware_ask_or_act(0.97, 1.0, max_risk=0.01)['action'] == 'ask'
