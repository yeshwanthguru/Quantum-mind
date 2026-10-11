"""Edge cases, input validation, limits and exactness of gradients."""
import numpy as np
import pytest
from quantum_mind.quantum import Circuit, W, X
from quantum_mind.quantum.statevector import MAX_QUBITS
from quantum_mind.problems import Qubo
from quantum_mind.applications.robotics import ask_or_act


def test_adjoint_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    c = Circuit(3)
    c.h(0).ry(W(0), 1).cx(0, 2).rz(X(0, 2.0), 2).cz(1, 2).rx(W(1, 0.7), 0).p(W(2), 2).rzz(W(1), 2, 0).ry(W(3), 0)
    w = rng.normal(size=4); x = rng.normal(size=(5, 1)); T = rng.normal(size=8)
    L = lambda v: float(np.sum((np.abs(c.state(v, x)) ** 2) @ T))                          # noqa: E731
    val, g = c.value_and_grad(w, x, lambda psi, rows: ((np.abs(psi) ** 2 @ T).sum(), psi * T), chunk=2)
    fd = np.array([(L(w + e) - L(w - e)) / 2e-6 for e in np.eye(4) * 1e-6])
    assert np.isclose(val, L(w)) and np.abs(g - fd).max() < 1e-6


def test_input_validation():
    with pytest.raises(ValueError):
        Circuit(MAX_QUBITS + 1)
    with pytest.raises(ValueError):
        Qubo(np.zeros((23, 23))).brute_force()
    with pytest.raises(ValueError):
        Circuit(2).ry(W(1), 0).state(np.zeros(1))                          # too few weights
    with pytest.raises(ValueError):
        ask_or_act(1.5)


def test_ask_or_act():
    assert ask_or_act(0.9, ask_cost=1, error_cost=5)['action'] == 'act'
    assert ask_or_act(0.5, ask_cost=1, error_cost=5)['action'] == 'ask'
    assert ask_or_act(0.99, {'model_disagreement_bits': 0.2})['action'] == 'ask'


def test_missing_optional_dependency_message(monkeypatch):
    import builtins
    from quantum_mind._optional import require
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == 'plotly':
            raise ImportError('no plotly')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda n: fake(n))
    with pytest.raises(ImportError, match=r'quantum-mind\[viz\]'):
        require('plotly')
