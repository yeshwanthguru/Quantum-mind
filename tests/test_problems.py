import numpy as np
from quantum_mind.problems import Qubo


def test_qubo_energies_and_ising_round_trip():
    rng = np.random.default_rng(0)
    q = Qubo(rng.normal(size=(5, 5)), 1.5)
    E = q.all_energies()
    X = ((np.arange(32)[:, None] >> np.arange(5)) & 1)
    assert np.allclose(E, [q.energy(x) for x in X])
    h, J, c = q.to_ising(); S = 1 - 2 * X
    assert np.allclose(E, S @ h + np.einsum('bi,ij,bj->b', S, np.triu(J, 1), S) + c)
