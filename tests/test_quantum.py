import numpy as np
import pytest
from qlcog.quantum import (Circuit, W, X, XX, VariationalClassifier, QuantumKernelClassifier, QuantumKernel, QAOA,
                           Hamiltonian, VQE, grover, parameter_shift)
from qlcog.quantum.ansatz import reuploading_classifier_circuit, shifted_weights
from qlcog.problems import maxcut


def _random_circuit(rng):
    c = Circuit(3)
    c.h(0).ry(W(0), 1).cx(0, 2).rz(X(0, 2.0), 2).cz(1, 2).rx(W(1), 0).rzz(XX(0, 1), 0, 1).swap(0, 2).s(1)
    c.p(W(2), 2).cx(2, 1).y(0).rzz(W(1), 2, 0)
    U = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    c.unitary(U, [2, 0]); c.diagonal(rng.uniform(0, 6, 8))
    return c


def test_simulator_matches_qiskit():
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(1); c = _random_circuit(rng)
    w = rng.normal(size=3); x = rng.normal(size=(3, 2)); S = c.state(w, x)
    for b in range(3):
        v = Statevector(c.to_qiskit(w, x[b], measure=False)).data
        assert abs(abs(np.vdot(v, S[b])) - 1) < 1e-10
    Ws = rng.normal(size=(4, 3)); S2 = c.state(Ws, np.tile(x[:1], (4, 1)))   # one weight vector per sample
    for b in range(4):
        assert np.allclose(S2[b], c.state(Ws[b], x[:1])[0])


def test_parameter_shift_is_exact():
    rng = np.random.default_rng(0)
    c = reuploading_classifier_circuit(2, 2, 2); w = rng.normal(size=c.n_weights); Xs = rng.random((3, 2))
    f = lambda v: c.probabilities(v, Xs)                                       # noqa: E731
    g = parameter_shift(f, w)
    for k in (0, 5):
        e = np.zeros_like(w); e[k] = 1e-6
        assert np.abs(g[k] - (f(w + e) - f(w - e)) / 2e-6).max() < 1e-6
    assert shifted_weights(w).shape == (2 * len(w), len(w))


def _moons(rng, n=120):
    th = rng.uniform(0, np.pi, n)
    X_ = np.r_[np.c_[np.cos(th), np.sin(th)], np.c_[1 - np.cos(th), 0.5 - np.sin(th)]] + rng.normal(0, 0.1, (2 * n, 2))
    return X_, np.r_[np.zeros(n), np.ones(n)]


def test_classifiers_learn():
    X_, y = _moons(np.random.default_rng(0))
    vqc = VariationalClassifier(layers=3, maxiter=100).fit(X_, y)
    assert vqc.score(X_, y) > 0.85
    P = vqc.predict_proba(X_[:5]); assert np.allclose(P.sum(1), 1)
    assert QuantumKernelClassifier(reps=1, scale=0.5).fit(X_, y).score(X_, y) > 0.85
    K = QuantumKernel(reps=1).fit(X_)(X_[:6]); assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)


def test_vqc_circuit_on_aer():
    pytest.importorskip('qiskit_aer')
    from qlcog.circuits import run
    X_, y = _moons(np.random.default_rng(1), 40)
    vqc = VariationalClassifier(layers=1, maxiter=20).fit(X_, y)
    counts = run(vqc.to_qiskit(X_[0]), 'aer', 20000)
    p1 = sum(k for b, k in counts.items() if b[-1] == '1') / 20000
    assert abs(p1 - vqc.predict_proba(X_[:1])[0, 1]) < 0.02


def test_qaoa_vqe_grover():
    q = maxcut([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    r = QAOA(q, p=2, restarts=3).run()
    assert r.energy == r.optimum and r.p_optimal > 0.5
    v = VQE(Hamiltonian([(1, 'ZZ'), (0.5, 'XI'), (0.5, 'IX')]), restarts=2).run()
    assert abs(v['energy'] - v['exact']) < 1e-6
    g = grover(4, [5, 11]); assert g.iterations == 2 and g.success > 0.9


def test_qaoa_export_matches_simulation():
    from qiskit.quantum_info import Statevector
    q = maxcut([(0, 1), (1, 2), (2, 0)]); qa = QAOA(q, p=1, restarts=1); r = qa.run()
    P = Statevector(qa.to_qiskit(r, measure=False)).probabilities()
    assert np.allclose(P, r.probabilities, atol=1e-9)
