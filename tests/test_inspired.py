import numpy as np
from quantum_mind.problems import maxcut, knapsack
from quantum_mind.inspired import QIEA, QPSO, SQA, simulated_annealing, MPSClassifier


def test_binary_optimisers_reach_the_optimum():
    rng = np.random.default_rng(3)
    edges = [(i, j) for i in range(10) for j in range(i + 1, 10) if rng.random() < 0.4]
    q = maxcut(edges, 10); opt = q.brute_force()[1]
    assert SQA(q, sweeps=200).run().value == opt
    assert QIEA(q, generations=200).run().value == opt
    assert simulated_annealing(q).value == opt
    k = knapsack([6, 5, 8, 9, 7, 3], [2, 3, 4, 5, 4, 1], 9)
    assert QIEA(k).run().value == k.brute_force()[1]


def test_qpso_minimises_a_smooth_function():
    r = QPSO(lambda v: np.sum((v - 0.3) ** 2), [-2] * 3, [2] * 3, iterations=150).run()
    assert r.value < 1e-6 and np.allclose(r.x, 0.3, atol=1e-3)
    assert r.history[-1] <= r.history[0]


def test_mps_classifier_learns_and_gradients_are_exact():
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(200, 4)); y = (X_[:, 0] + X_[:, 1] * X_[:, 2] > 0).astype(int)
    m = MPSClassifier(bond=4, local_dim=3, epochs=40).fit(X_, y)
    assert m.score(X_, y) > 0.85 and np.allclose(m.predict_proba(X_[:3]).sum(1), 1)
    Phi = m._phi(m._scale(X_[:8])); Y = np.eye(2)[y[:8]]
    _, g, _ = m._grads(Phi, Y)
    j, idx = 1, (0, 1, 2); c0 = m.cores[j][idx]; eps = 1e-6
    m.cores[j][idx] = c0 + eps; lp = m._grads(Phi, Y)[0]
    m.cores[j][idx] = c0 - eps; lm = m._grads(Phi, Y)[0]; m.cores[j][idx] = c0
    assert abs(g[j][idx] - (lp - lm) / (2 * eps)) < 1e-6
