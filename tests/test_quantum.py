import numpy as np
from quantum_mind.quantum import Circuit, W, X, XX


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


def _moons(rng, n=120):
    th = rng.uniform(0, np.pi, n)
    X_ = np.r_[np.c_[np.cos(th), np.sin(th)], np.c_[1 - np.cos(th), 0.5 - np.sin(th)]] + rng.normal(0, 0.1, (2 * n, 2))
    return X_, np.r_[np.zeros(n), np.ones(n)]
