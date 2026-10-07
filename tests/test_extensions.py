"""Bootstrap uncertainty, online per-person models and model-discriminating design."""
import numpy as np
import pytest

from quantum_mind import fit
from quantum_mind.applications.robotics import HumanModelEnsemble
from quantum_mind.core import bootstrap, OnlinePersonModel, information_gain, rank_conditions, model_posterior, tvd
from quantum_mind.families.conjunction import QuantumConjunctionModel
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel


def test_bootstrap_interval_covers_the_generating_parameters():
    truth = BayesOrderModel(pA=0.6, pB=0.7, rho=0.4)
    data = truth.sample(None, 400, np.random.default_rng(3))
    res = fit(BayesOrderModel, data, restarts=2)
    boot = bootstrap(res, data, n_boot=60, rng=np.random.default_rng(4))
    for name, (lo, hi) in boot.ci(0.95).items():
        assert lo <= truth.params[name] <= hi
    assert all(v > 0 for v in boot.se().values())
    lo, hi = boot.predictive_interval()['AB']
    fitted = res.model.predict(None)['AB']
    assert np.all(lo <= fitted + 1e-9) and np.all(fitted <= hi + 1e-9)


def test_bootstrap_least_squares_model():
    m = QuantumConjunctionModel()
    pred = m.predict(None)
    data = {c: np.asarray(v, float) + 0.02 for c, v in pred.items()}
    res = fit(QuantumConjunctionModel, data, restarts=2)
    boot = bootstrap(res, data, n_boot=10)
    assert boot.n_boot == 10 and set(boot.predictions) == set(data)


def test_fit_uses_the_starting_point():
    data = {'AB': [40, 10, 20, 30], 'BA': [35, 15, 15, 35]}
    best = fit(BayesOrderModel, data, restarts=4)
    warm = fit(BayesOrderModel, data, restarts=1, x0=best.model.to_vector())
    assert warm.loss <= best.loss + 1e-6


def test_online_model_learns_one_person():
    truth = BayesOrderModel(pA=0.8, pB=0.3, rho=0.5)
    pt = truth.predict(None)
    person = OnlinePersonModel(BayesOrderModel, n_particles=400, rng=np.random.default_rng(0))
    before = tvd(person.predict()['AB'], pt['AB'])
    rng = np.random.default_rng(1)
    for i in range(300):
        c = 'AB' if i % 2 else 'BA'
        person.update(c, rng.choice(4, p=pt[c]))
    after = tvd(person.predict()['AB'], pt['AB'])
    assert after < before and after < 0.06
    lo, hi = person.credible_interval('pA', 0.95)
    assert lo <= 0.8 <= hi
    assert abs(sum(person.predict()['BA']) - 1) < 1e-9


def test_online_model_batch_and_prior_from_fit():
    data = {'AB': [40, 10, 20, 30], 'BA': [35, 15, 15, 35]}
    res = fit(BayesOrderModel, data, restarts=2)
    person = OnlinePersonModel(BayesOrderModel, prior=res, prior_scale=0.3, n_particles=200)
    person.observe({'AB': [3, 0, 1, 0]})
    assert person.n_observed == 4 and 1 <= person.ess <= 200


def test_online_model_rejects_least_squares_models():
    with pytest.raises(ValueError):
        OnlinePersonModel(QuantumConjunctionModel)


def test_information_gain_bounds_and_identity():
    q = QuantumOrderModel4D(phi=0.7)
    assert information_gain([q, q], 'AB') == pytest.approx(0, abs=1e-12)
    g = information_gain([q, BayesOrderModel(), AnchoringOrderModel(wA=1.5, wB=-1.0)], 'AB')
    assert 0 <= g <= np.log2(3)
    with pytest.raises(ValueError):
        information_gain([q, q], 'AB', weights=[1, -1])


def test_information_gain_equals_ensemble_disagreement():
    data = {'AB': [40, 10, 20, 30], 'BA': [30, 20, 10, 40]}
    ens = HumanModelEnsemble().update(data, restarts=2)
    models = [f.model for f in ens.fits]
    _, unc = ens.predict('AB')
    assert information_gain(models, 'AB', weights=ens.weights) == pytest.approx(unc['model_disagreement_bits'], abs=1e-9)


def test_rank_conditions_and_model_posterior_find_the_generating_model():
    a = AnchoringOrderModel(pA=0.5, pB=0.5, wA=0.0, wB=0.9)     # anchoring only when B is asked second
    b = BayesOrderModel(pA=0.5, pB=0.5, rho=0.0)                # independent answers in both orders
    ranking = rank_conditions([a, b], ['AB', 'BA'])
    assert ranking[0][0] == 'AB' and ranking[0][1] > 0.1
    assert ranking[1][1] == pytest.approx(0, abs=1e-12)          # the models agree when A is second
    data = a.sample(None, 200, np.random.default_rng(0))
    post = model_posterior([a, b], data)
    assert post[0] > 0.99


def _cases():
    """(model class, design, structural options, LOSS) for every family labelled as extended in the atlas."""
    from quantum_mind.families.interference import InterferenceModel, ClassicalMixtureModel
    from quantum_mind.families.qlbn import BayesNet, ClassicalBNModel, for_network
    from quantum_mind.families.dynamics import QuantumWalk, MarkovWalk, OpenSystemBelief, MarkovBelief
    from quantum_mind.families.decision import QDTModel, ExpectedUtilityModel
    from quantum_mind.families.perception import QuantumZenoBistableModel, MarkovSwitchingModel, GammaRenewalModel
    from quantum_mind.families.memory import QuantumEpisodicModel
    from quantum_mind.families.similarity import QuantumSimilarityModel
    from quantum_mind.families.concepts import FockSpaceConceptModel
    net = BayesNet({'O': ['w', 'l'], 'P': ['y', 'n']}, {'P': ['O']},
                   {'O': {(): {'w': .5, 'l': .5}}, 'P': {('w',): {'y': .69, 'n': .31}, ('l',): {'y': .59, 'n': .41}}})
    bn = dict(net=net, query='P', conditions={'unknown': None, 'won': {'O': 'w'}})
    gamble = {'p': ([(30, 1.0)], [(45, 0.8), (0, 0.2)])}
    return [(QuantumOrderModel4D, None, {}), (BayesOrderModel, None, {}), (AnchoringOrderModel, None, {}),
            (InterferenceModel, None, {}), (ClassicalMixtureModel, None, {}),
            (for_network(net, 'P', bn['conditions']), None, bn), (ClassicalBNModel, None, bn),
            (QuantumWalk, None, {}), (MarkovWalk, None, {}), (OpenSystemBelief, None, {}), (MarkovBelief, None, {}),
            (QDTModel, gamble, {}), (ExpectedUtilityModel, gamble, {}),
            (QuantumZenoBistableModel, None, {}), (MarkovSwitchingModel, None, {}), (GammaRenewalModel, None, {}),
            (QuantumConjunctionModel, None, {}), (QuantumEpisodicModel, None, {}),
            (QuantumSimilarityModel, [('K', 'C'), ('C', 'K')], {'concepts': ['K', 'C'], 'ranks': {'C': 2}}),
            (FockSpaceConceptModel, {'x': (0.4, 0.6), 'y': (0.2, 0.7)}, {})]


@pytest.mark.parametrize('cls, design, opts', _cases(), ids=lambda v: getattr(v, '__name__', ''))
def test_extensions_run_on_every_extended_family(cls, design, opts):
    rng = np.random.default_rng(0)
    m = cls(**opts)
    pred = m.predict(design)
    if cls.LOSS == 'sse':
        data = {c: np.asarray(v, float) + 0.01 for c, v in pred.items()}
    else:
        data = m.sample(design, 50, rng)
    res = fit(cls, data, design, restarts=1, **opts)
    boot = bootstrap(res, data, design, n_boot=2, restarts=1)
    assert boot.n_boot == 2
    if cls.LOSS == 'sse':
        with pytest.raises(ValueError):
            OnlinePersonModel(cls, **opts)
        return
    person = OnlinePersonModel(cls, prior=res, n_particles=20, design=design, rng=rng)
    c = next(iter(data))
    person.update(c, int(np.argmax(data[c])))
    assert abs(float(np.sum(person.predict()[c])) - 1) < 1e-6
    assert information_gain([res.model, person], c, design) >= 0
