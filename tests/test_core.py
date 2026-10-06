import numpy as np
from quantum_mind.core import projector, sequence_probabilities, luders, lindblad_superoperator, evolve_density, fit, Param, Model


def test_projector_and_luders():
    P = projector([[1, 1, 0]])
    assert np.allclose(P @ P, P) and np.isclose(np.trace(P), 1)
    p, v = luders(np.array([1.0, 0, 0]), P)
    assert np.isclose(p, 0.5) and np.isclose(np.linalg.norm(v), 1)


def test_sequence_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4); psi /= np.linalg.norm(psi)
    Ps = {'A': projector(rng.normal(size=(2, 4))), 'B': projector(rng.normal(size=(1, 4)))}
    pr = sequence_probabilities(psi, Ps, ('A', 'B', 'A'))
    assert np.isclose(sum(pr.values()), 1)


def test_lindblad_trace_preserving():
    H = np.diag([0.0, 1.0]) + 0.3 * np.array([[0, 1], [1, 0]])
    L = lindblad_superoperator(H, [np.diag([1.0, 0]), np.diag([0, 1.0])], [0.5, 0.5])
    rho = evolve_density(np.array([[0.5, 0.5], [0.5, 0.5]], complex), L, 2.0)
    assert np.isclose(np.trace(rho).real, 1) and np.allclose(rho, rho.conj().T)


class Coin(Model):
    PARAMS = [Param('p', 'prob', 0.5)]

    def predict(self, design=None):
        return {'flip': np.array([self.p, 1 - self.p])}


def test_fit_recovers_parameter():
    f = fit(Coin, {'flip': np.array([700, 300])}, restarts=3)
    assert abs(f.model.p - 0.7) < 1e-4 and f.k == 1 and f.n == 1000
