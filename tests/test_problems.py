import numpy as np
from qlcog.problems import Qubo, maxcut, knapsack, task_allocation, portfolio, from_ising


def test_qubo_energies_and_ising_round_trip():
    rng = np.random.default_rng(0)
    q = Qubo(rng.normal(size=(5, 5)), 1.5)
    E = q.all_energies()
    X = ((np.arange(32)[:, None] >> np.arange(5)) & 1)
    assert np.allclose(E, [q.energy(x) for x in X])
    h, J, c = q.to_ising(); S = 1 - 2 * X
    assert np.allclose(E, S @ h + np.einsum('bi,ij,bj->b', S, np.triu(J, 1), S) + c)
    assert np.allclose(from_ising(h, J, c).all_energies(), E)


def test_problem_builders_have_the_right_optima():
    assert maxcut([(0, 1), (1, 2), (2, 3), (3, 0)], 4).brute_force()[1] == -4
    x, e = knapsack([6, 5, 8, 9], [2, 3, 4, 5], 7).brute_force()
    assert list(x[:4]) == [1, 0, 0, 1] and e == -15
    x, _ = task_allocation(np.array([[1, 9, 9], [9, 1, 9], [9, 9, 1]])).brute_force()
    assert (x.reshape(3, 3) == np.eye(3)).all()
    x, _ = portfolio([0.3, 0.1, 0.2, 0.05], np.eye(4) * 0.01, budget=2).brute_force()
    assert list(x) == [1, 0, 1, 0]
