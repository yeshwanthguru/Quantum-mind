"""Robot problems as QUBOs: encodings, true objectives, classical solvers and the Qiskit QAOA adapter."""
import numpy as np
import pytest

from quantum_mind.computing import (compare_solvers, exhaustive, grasp_selection, greedy_descent, multi_robot_allocation,
                                    qaoa, simulated_annealing, slot_scheduling)


def _problems(seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 0.6, (7, 7))
    a = np.triu(rng.random((4, 4)) < 0.6, 1)
    return [multi_robot_allocation(rng.uniform(1, 9, (3, 3)), balance=0.3),
            grasp_selection(rng.uniform(0, 1, 7), (c + c.T) / 2, 3),
            slot_scheduling((a | a.T).astype(int), 3)]


def test_qubo_optimum_is_the_true_feasible_optimum():
    for p in _problems():
        x, _ = p.qubo.brute_force()
        assert p.feasible(x)
        assert p.objective(x) == pytest.approx(p.optimum()[1])


def test_penalty_makes_every_infeasible_string_worse_than_the_optimum():
    for p in _problems(1):
        energies = p.qubo.all_energies()
        n = p.qubo.n
        best_feasible = min(p.qubo.energy(((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)[k])
                            for k in range(2 ** n) if p.feasible(((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)[k]))
        assert energies.min() == pytest.approx(best_feasible)


def test_energy_equals_objective_on_feasible_strings():
    rng = np.random.default_rng(2)
    p = multi_robot_allocation(rng.uniform(1, 9, (2, 3)), balance=0.5)
    for _ in range(20):
        x = np.zeros((2, 3), int)
        x[rng.integers(0, 2, 3), np.arange(3)] = 1                    # one robot per task
        assert p.qubo.energy(x.ravel()) == pytest.approx(p.objective(x.ravel()))
    g = grasp_selection([0.9, 0.5, 0.7, 0.2], np.array([[0, .3, 0, 0], [.3, 0, .2, 0], [0, .2, 0, .4], [0, 0, .4, 0]]), 2)
    for pair in ((0, 1), (1, 2), (0, 3), (2, 3)):
        x = np.zeros(4, int)
        x[list(pair)] = 1
        assert g.qubo.energy(x) == pytest.approx(g.objective(x))


def test_decoding_and_validation():
    p = multi_robot_allocation(np.array([[1.0, 5.0], [4.0, 2.0]]))
    x, value = p.optimum()
    assert p.decode(x) == {0: [0], 1: [1]} and value == pytest.approx(3.0)
    with pytest.raises(ValueError):
        multi_robot_allocation(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        grasp_selection([1, 2], np.array([[0, 1], [2, 0]]), 1)         # not symmetric
    with pytest.raises(ValueError):
        slot_scheduling(np.array([[0, 1], [0, 0]]), 2)                 # not symmetric
    s = slot_scheduling(np.array([[0, 1], [1, 0]]), 1)                 # one slot, two conflicting robots
    assert s.optimum()[1] == 1.0


def test_classical_solvers_find_feasible_good_solutions():
    for p in _problems(3):
        opt = p.optimum()[1]
        r = greedy_descent(p.qubo, restarts=30, seed=0)
        assert p.feasible(r.x) and p.objective(r.x) == pytest.approx(opt)
        assert exhaustive(p.qubo).energy == pytest.approx(p.qubo.brute_force()[1])
        sa = simulated_annealing(p.qubo, sweeps=300, restarts=5, seed=0)
        assert sa.energy == pytest.approx(p.qubo.energy(sa.x))


def test_compare_solvers_reports_robot_terms():
    rows = compare_solvers(_problems(4)[0], solvers={'exhaustive': exhaustive,
                                                     'greedy': lambda q: greedy_descent(q, restarts=30)})
    assert [r['solver'] for r in rows] == ['exhaustive', 'greedy']
    assert rows[0]['gap'] == pytest.approx(0.0) and rows[1]['feasible']


def test_qaoa_adapter_returns_a_valid_distribution_summary():
    pytest.importorskip('qiskit')
    p = _problems(5)[2]                                                   # 12 qubits is too slow for a unit test
    p = slot_scheduling(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), 2)   # 6 qubits
    r = qaoa(p.qubo, depth=2, restarts=2, maxiter=60, seed=0)
    assert r.x.shape == (p.qubo.n,) and r.energy == pytest.approx(p.qubo.energy(r.x))
    assert 0 <= r.info['p_optimal'] <= 1 and r.info['expected_energy'] >= p.qubo.all_energies().min() - 1e-9
    assert r.info['p_optimal'] > 1 / 2 ** p.qubo.n                       # better than a uniform guess
    with pytest.raises(ValueError):
        qaoa(multi_robot_allocation(np.ones((5, 4))).qubo)                # 20 variables
