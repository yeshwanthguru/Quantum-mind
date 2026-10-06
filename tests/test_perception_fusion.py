"""Bistable perception family and multimodal fusion."""
import numpy as np
import pytest

from quantum_mind.core import compare
from quantum_mind.families.perception import (QuantumZenoBistableModel, MarkovSwitchingModel, GammaRenewalModel,
                                       dwell_time_design, dwell_counts, mean_dwell_time)
from quantum_mind.applications.fusion import (detector_likelihood, asr_likelihood, direction_likelihood, bayes_fusion,
                                       dempster_shafer_fusion, fit_incompatibility, compare_fusion)
from quantum_mind.applications.intent import QuantumIntentResolver


def test_zeno_dwell_time_scales_inversely_with_the_observation_interval():
    z = QuantumZenoBistableModel(g=1.0)
    assert mean_dwell_time(z, 0.05) == pytest.approx(2 * mean_dwell_time(z, 0.1), rel=0.01)
    assert MarkovSwitchingModel(rate=0.5).mean_dwell(0.05) == MarkovSwitchingModel(rate=0.5).mean_dwell(0.1) == 2.0
    assert GammaRenewalModel(shape=3, scale=1).mean_dwell() == 3
    design = dwell_time_design({'a': 0.05, 'b': 0.1}, max_time=30, n_bins=15)
    for m in (z, MarkovSwitchingModel(), GammaRenewalModel()):
        for v in m.predict(design).values():
            assert v.sum() == pytest.approx(1.0) and np.all(v >= 0)


def test_two_intervals_identify_the_zeno_model():
    design = dwell_time_design({'fast': 0.035, 'slow': 0.14}, max_time=40, n_bins=25)
    data = QuantumZenoBistableModel(g=1.5).sample(design, 300, np.random.default_rng(0))
    best = compare([QuantumZenoBistableModel, MarkovSwitchingModel, GammaRenewalModel], data, design)[0]
    assert type(best.model) is QuantumZenoBistableModel and abs(best.model.g - 1.5) < 0.15


def test_dwell_counts():
    c = dwell_counts([0.5, 1.5, 1.6, 9.0], np.array([0, 1, 2, 3]))
    assert c.tolist() == [1, 2, 0, 1]


def test_adapters_and_fusion_rules():
    objs = ['red cup', 'blue cup', 'bowl']
    v = detector_likelihood({'red cup': 0.9, 'bowl': 0.1}, objs)
    assert v.argmax() == 0 and v.sum() == pytest.approx(1)
    assert detector_likelihood([3.0, 0.0, -1.0], objs).argmax() == 0                      # logits
    a = asr_likelihood([('the blue one', 0.8)], {'red cup': ['red'], 'blue cup': ['blue'], 'bowl': ['bowl']}, objs)
    assert a.argmax() == 1
    g = direction_likelihood([0, 1], {'red cup': [1, 0], 'blue cup': [0, 1], 'bowl': [-1, 0]}, objs)
    assert g.argmax() == 1
    post = bayes_fusion([v, g])
    assert post.sum() == pytest.approx(1)
    ds, ign = dempster_shafer_fusion([v, g], [0.9, 0.9])
    assert ds.sum() == pytest.approx(1) and 0 <= ign < 0.1
    ds_unrel, ign_unrel = dempster_shafer_fusion([v, g], [0.1, 0.1])
    assert ign_unrel > ign                                                               # unreliable cues: more ignorance


def test_fit_incompatibility_and_compare():
    intents = ['a', 'b', 'c']
    cues = {'x': [0.7, 0.2, 0.1], 'y': [0.25, 0.6, 0.15]}
    true = QuantumIntentResolver(intents)
    true.add_cue('x', cues['x'], 0.7)
    true.add_cue('y', cues['y'], 0.7)
    rng = np.random.default_rng(1)
    orders = [['x', 'y'], ['y', 'x']]
    trials = [(o, intents[rng.choice(3, p=true.posterior(o))]) for o in [orders[i % 2] for i in range(1500)]]
    _, angles = fit_incompatibility(QuantumIntentResolver, intents, cues, trials[:1000])
    assert all(0 <= a <= np.pi / 2 for a in angles.values())
    res = compare_fusion(intents, cues, trials[:1000], trials[1000:])
    assert res['quantum_like']['log_loss'] <= res['bayes']['log_loss'] + 0.01
