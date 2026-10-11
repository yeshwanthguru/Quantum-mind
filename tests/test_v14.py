"""Intent resolution: the quantum-like resolver nests the Bayes resolver and shows order effects."""
import numpy as np
import pytest


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
