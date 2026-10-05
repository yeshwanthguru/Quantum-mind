"""Edge cases, input validation, limits and exactness of gradients."""
import numpy as np
import pytest
from qlcog.quantum import Circuit, W, X, VariationalClassifier, QuantumKernelClassifier, QAOA, grover
from qlcog.quantum.statevector import MAX_QUBITS
from qlcog.problems import Qubo, maxcut
from qlcog.inspired import MPSClassifier
from qlcog.applications.robotics import ask_or_act


def test_adjoint_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    c = Circuit(3)
    c.h(0).ry(W(0), 1).cx(0, 2).rz(X(0, 2.0), 2).cz(1, 2).rx(W(1, 0.7), 0).p(W(2), 2).rzz(W(1), 2, 0).ry(W(3), 0)
    w = rng.normal(size=4); x = rng.normal(size=(5, 1)); T = rng.normal(size=8)
    L = lambda v: float(np.sum((np.abs(c.state(v, x)) ** 2) @ T))                          # noqa: E731
    val, g = c.value_and_grad(w, x, lambda psi, rows: ((np.abs(psi) ** 2 @ T).sum(), psi * T), chunk=2)
    fd = np.array([(L(w + e) - L(w - e)) / 2e-6 for e in np.eye(4) * 1e-6])
    assert np.isclose(val, L(w)) and np.abs(g - fd).max() < 1e-6


def test_classifier_gradient_is_exact_and_training_scales():
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(40, 3)); y = rng.integers(0, 3, 40)
    m = VariationalClassifier(layers=2, maxiter=1).fit(X_, y)
    Xs = m.scaler.transform(X_); Y = (y[:, None] == m.classes_[None]).astype(float)
    L = lambda w: -np.mean(np.sum(Y * np.log(np.clip(m._probs(w, Xs), 1e-12, 1)), 1))   # noqa: E731
    w = rng.normal(size=m.circuit.n_weights)
    val, g = m._objective(w, Xs, Y)
    fd = np.array([(L(w + e) - L(w - e)) / 2e-6 for e in np.eye(len(w)) * 1e-6])
    assert np.isclose(val, L(w)) and np.abs(g - fd).max() < 1e-6
    # 8 features (256 amplitudes), 300 samples: the pre-1.2 batched parameter-shift gradient did not
    # finish three steps in five minutes; the adjoint gradient takes about 0.2 s per step
    Xb = rng.normal(size=(300, 8)); yb = (Xb[:, 0] > 0).astype(int)
    big = VariationalClassifier(layers=1, maxiter=15).fit(Xb, yb)
    assert len(big.history_) >= 2 and np.isfinite(big.loss_)


def test_input_validation():
    with pytest.raises(ValueError):
        VariationalClassifier().fit(np.zeros((5, 2)), np.zeros(5))          # one class
    with pytest.raises(ValueError):
        VariationalClassifier().fit(np.zeros((5, 2)), np.arange(4) % 2)     # length mismatch
    with pytest.raises(ValueError):
        QuantumKernelClassifier().fit(np.array([[np.nan, 1.0], [0, 1]]), [0, 1])
    with pytest.raises(ValueError):
        VariationalClassifier().fit(np.zeros(5), np.arange(5) % 2)          # 1-D X
    with pytest.raises(ValueError):
        Circuit(MAX_QUBITS + 1)
    with pytest.raises(ValueError):
        Qubo(np.zeros((23, 23))).brute_force()
    with pytest.raises(ValueError):
        grover(3, [])
    with pytest.raises(ValueError):
        grover(17, [1])
    with pytest.raises(ValueError):
        Circuit(2).ry(W(1), 0).state(np.zeros(1))                          # too few weights
    with pytest.raises(ValueError):
        ask_or_act(1.5)


def test_qaoa_result_fields_and_mps_on_constant_feature():
    r = QAOA(maxcut([(0, 1), (1, 2)]), p=1, restarts=1).run()
    assert r.weights.shape == (2,) and 0 < r.p_optimal <= 1 and r.optimum == -2
    X_ = np.c_[np.ones(30), np.linspace(0, 1, 30)]; y = (X_[:, 1] > 0.5).astype(int)   # constant column
    assert MPSClassifier(bond=2, epochs=30).fit(X_, y).score(X_, y) > 0.8


def test_grover_gate_export_matches_simulation():
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    for n, marked in ((1, [1]), (4, [5, 11])):
        g = grover(n, marked)
        for style in ('gates', 'diagonal'):
            assert np.allclose(Statevector(g.to_qiskit(measure=False, style=style)).probabilities(), g.probabilities)


def test_ask_or_act():
    assert ask_or_act(0.9, ask_cost=1, error_cost=5)['action'] == 'act'
    assert ask_or_act(0.5, ask_cost=1, error_cost=5)['action'] == 'ask'
    assert ask_or_act(0.99, {'model_disagreement_bits': 0.2})['action'] == 'ask'


def test_missing_optional_dependency_message(monkeypatch):
    import builtins
    from qlcog._optional import require
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == 'plotly':
            raise ImportError('no plotly')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda n: fake(n))
    with pytest.raises(ImportError, match=r'qlcog\[viz\]'):
        require('plotly')
