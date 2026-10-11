"""Families and models added in 1.4: games, memory, concepts, intent, QFT, amplitude estimation,
quantum walks, regression, kernel anomaly detection and clustering, QRL, quantum language model."""
import numpy as np
import pytest


def test_memory_and_concept_models():
    from quantum_mind.core import compare
    from quantum_mind.families.memory import QuantumEpisodicModel, AdditiveMemoryModel, overdistribution
    from quantum_mind.families.concepts import FockSpaceConceptModel, ProductConceptModel, MinConceptModel, interference_bound
    gen = QuantumEpisodicModel(c=0.9, a_target=0.6, b_target=0.9)
    pred = gen.predict()
    assert overdistribution(pred, 'target') > 0.1
    assert abs(overdistribution(AdditiveMemoryModel().predict(), 'target')) < 1e-12
    assert type(compare([QuantumEpisodicModel, AdditiveMemoryModel], pred, restarts=4)[0].model).__name__ == 'QuantumEpisodicModel'
    items = {'a': (0.3, 0.25), 'b': (0.05, 0.9), 'c': (0.75, 0.9), 'd': (0.1, 0.8)}
    truth = FockSpaceConceptModel(m2=0.3, kappa=0.6).predict(items)
    best = compare([FockSpaceConceptModel, ProductConceptModel, MinConceptModel], truth, items, restarts=4)[0]
    assert type(best.model).__name__ == 'FockSpaceConceptModel' and abs(best.model.kappa - 0.6) < 0.05
    assert interference_bound(1.0, 0.5) == 0 and interference_bound(0.5, 0.5) == 0.5
    with pytest.raises(ValueError):
        FockSpaceConceptModel().predict(None)


def test_intent_resolver_nests_bayes_and_shows_order_effects():
    from quantum_mind.applications.intent import QuantumIntentResolver, BayesIntentResolver
    q, b = QuantumIntentResolver(['a', 'b', 'c']), BayesIntentResolver(['a', 'b', 'c'])
    for r in (q, b):
        r.add_cue('x', [0.6, 0.3, 0.1]); r.add_cue('y', [0.2, 0.5, 0.3])
    assert np.allclose(q.posterior(['x', 'y']), b.posterior(['x', 'y'])) and b.order_effect('x', 'y') == 0
    q.add_cue('z', [0.7, 0.2, 0.1], theta=0.5)
    assert q.order_effect('x', 'z') > 0.005 and np.isclose(q.posterior(['x', 'z']).sum(), 1)
    with pytest.raises(ValueError):
        q.add_cue('bad', [1, -1, 0])
