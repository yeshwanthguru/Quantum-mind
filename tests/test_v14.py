"""Families and models added in 1.4: games, memory, concepts, intent, QFT, amplitude estimation,
quantum walks, regression, kernel anomaly detection and clustering, QRL, quantum language model."""
import numpy as np
import pytest


def test_ewl_game_reproduces_published_results():
    from qlcog.families.game_theory import EWLGame, PRISONERS_DILEMMA, C, D, Q, classical_nash_equilibria
    G0, G = EWLGame(PRISONERS_DILEMMA, 0.0), EWLGame(PRISONERS_DILEMMA, np.pi / 2)
    for g in (G0, G):                                    # classical strategies give the classical game
        assert np.allclose(g.payoff(C, D), (0, 5)) and np.allclose(g.payoff(D, D), (1, 1))
    assert np.allclose(G.payoff(Q, Q), (3, 3)) and np.allclose(G.payoff(D, Q), (0, 5))
    assert G.is_nash(Q, Q, grid=21) and not G.is_nash(D, D, grid=21) and G0.is_nash(D, D, grid=21)
    assert not G.is_nash(Q, Q, grid=13, three_parameter=True)          # Benjamin and Hayden (2001)
    assert classical_nash_equilibria(PRISONERS_DILEMMA) == [(0.0, 0.0, 1.0, 1.0)]
    assert np.isclose(G.outcome_probabilities(Q, D).sum(), 1)


def test_memory_and_concept_models():
    from qlcog.core import compare
    from qlcog.families.memory import QuantumEpisodicModel, AdditiveMemoryModel, overdistribution
    from qlcog.families.concepts import FockSpaceConceptModel, ProductConceptModel, MinConceptModel, interference_bound
    gen = QuantumEpisodicModel(c=0.9, a_target=0.6, b_target=0.9)
    pred = gen.predict()
    assert overdistribution(pred, 'target') > 0.1
    assert abs(overdistribution(AdditiveMemoryModel().predict(), 'target')) < 1e-12
    assert type(compare([QuantumEpisodicModel, AdditiveMemoryModel], pred, restarts=4)[0].model).__name__ == 'QuantumEpisodicModel'
    items = {'a': (0.3, 0.25), 'b': (0.05, 0.9), 'c': (0.75, 0.9), 'd': (0.1, 0.8)}
    truth = FockSpaceConceptModel(m2=0.3, kappa=0.6).predict(items)
    best = compare([FockSpaceConceptModel, ProductConceptModel, MinConceptModel], truth, items, restarts=4)[0]
    assert type(best.model).__name__ == 'FockSpaceConceptModel' and abs(best.model.kappa - 0.6) < 0.05
    assert interference_bound(1.0, 0.5) == 0 and interference_bound(0.5, 0.5) == 0.5
    with pytest.raises(ValueError):
        FockSpaceConceptModel().predict(None)


def test_intent_resolver_nests_bayes_and_shows_order_effects():
    from qlcog.applications.intent import QuantumIntentResolver, BayesIntentResolver
    q, b = QuantumIntentResolver(['a', 'b', 'c']), BayesIntentResolver(['a', 'b', 'c'])
    for r in (q, b):
        r.add_cue('x', [0.6, 0.3, 0.1]); r.add_cue('y', [0.2, 0.5, 0.3])
    assert np.allclose(q.posterior(['x', 'y']), b.posterior(['x', 'y'])) and b.order_effect('x', 'y') == 0
    q.add_cue('z', [0.7, 0.2, 0.1], theta=0.5)
    assert q.order_effect('x', 'z') > 0.005 and np.isclose(q.posterior(['x', 'z']).sum(), 1)
    with pytest.raises(ValueError):
        q.add_cue('bad', [1, -1, 0])


def test_qft_and_period_finding():
    from qlcog.quantum import qft_circuit, qft_matrix, find_period
    for n in (1, 2, 4):
        c, ci = qft_circuit(n), qft_circuit(n, inverse=True)
        U = np.column_stack([c.state(None, None, init=np.eye(2 ** n)[j])[0] for j in range(2 ** n)])
        Ui = np.column_stack([ci.state(None, None, init=np.eye(2 ** n)[j])[0] for j in range(2 ** n)])
        assert np.allclose(U, qft_matrix(n)) and np.allclose(Ui @ U, np.eye(2 ** n))
    assert find_period(6, 8)[1] == 8 and find_period(6, 4, offset=3, shots=300, rng=np.random.default_rng(1))[1] == 4


def test_qft_export_and_cp_gradient():
    from qlcog.quantum import qft_circuit, qft_matrix, Circuit, W
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator
    assert np.allclose(Operator(qft_circuit(3).to_qiskit(measure=False)).data, qft_matrix(3))
    c = Circuit(2); c.h(0).h(1).cp(W(0), 0, 1).ry(W(1), 0)
    T = np.array([0.3, -1.0, 2.0, 0.5]); w = np.array([0.7, 0.4])
    L = lambda v: float((np.abs(c.state(v)[0]) ** 2) @ T)               # noqa: E731
    val, g = c.value_and_grad(w, None, lambda psi, rows: ((np.abs(psi) ** 2 @ T).sum(), psi * T))
    fd = np.array([(L(w + e) - L(w - e)) / 2e-6 for e in np.eye(2) * 1e-6])
    assert np.isclose(val, L(w)) and np.abs(g - fd).max() < 1e-6


def test_amplitude_estimation():
    from qlcog.quantum import AmplitudeEstimation, monte_carlo_estimate
    x = np.linspace(-3, 3, 8); p = np.exp(-x ** 2 / 2); f = np.clip((x + 3) / 6, 0, 1)
    ae = AmplitudeEstimation(p, f)
    assert np.allclose(ae.A @ ae.A.T, np.eye(16))
    assert all(abs(ae.good_probability(m) - np.sin((2 * m + 1) * ae.theta) ** 2) < 1e-12 for m in range(6))
    rng = np.random.default_rng(0)
    errs = [abs(ae.run(rng=rng).estimate - ae.exact) for _ in range(20)]
    mc = [abs(monte_carlo_estimate(p, f, 6800, rng) - ae.exact) for _ in range(20)]
    assert np.sqrt(np.mean(np.square(errs))) < np.sqrt(np.mean(np.square(mc)))
    with pytest.raises(ValueError):
        AmplitudeEstimation([0.5, 0.5], [0.2, 1.5])


def test_quantum_walks():
    from qlcog.quantum import adjacency, ctqw_probabilities, quantum_walk_centrality, pagerank
    A = adjacency([(0, 1), (0, 2), (0, 3), (3, 4)], 5)
    assert np.isclose(ctqw_probabilities(A, 0.7, start=0).sum(), 1)
    T = np.linspace(0, 300, 3001)
    assert np.abs(np.mean([ctqw_probabilities(A, t) for t in T], 0) - quantum_walk_centrality(A)).max() < 2e-3
    pr = pagerank(A); assert np.isclose(pr.sum(), 1) and np.argmax(pr) == 0


def test_regressor_anomaly_clustering():
    from qlcog.quantum import VariationalRegressor, QuantumKernelAnomalyDetector, QuantumKernelClustering
    rng = np.random.default_rng(0)
    s = np.sin(np.arange(160) * 0.3); X = np.array([s[i:i + 3] for i in range(150)]); y = s[3:153]
    assert VariationalRegressor(layers=2, maxiter=80).fit(X[:110], y[:110]).score(X[110:], y[110:]) > 0.9
    normal = rng.normal(0, 1, (120, 2)); test = np.r_[rng.normal(0, 1, (60, 2)), rng.normal([2.5, 2.5], 0.3, (10, 2))]
    lab = np.r_[np.zeros(60), np.ones(10)]
    for k in ('quantum', 'rbf'):
        sc = QuantumKernelAnomalyDetector(kernel=k).fit(normal).score_samples(test)
        assert np.mean([sc[i] > sc[j] for i in np.flatnonzero(lab) for j in np.flatnonzero(1 - lab)]) > 0.85
    Xc = np.r_[rng.normal([0, 0], 0.3, (40, 2)), rng.normal([2.5, 2.5], 0.3, (40, 2))]; yc = np.r_[np.zeros(40), np.ones(40)]
    for k in ('quantum', 'rbf'):
        lab_c = QuantumKernelClustering(2, kernel=k).fit_predict(Xc)
        assert max(np.mean(lab_c == yc), np.mean(lab_c != yc)) > 0.9


def test_quantum_inspired_rl_learns_the_shortest_path():
    from qlcog.inspired import GridWorld, QuantumInspiredQLearning, QLearning, train
    env = GridWorld()
    q = QuantumInspiredQLearning(env.n_states, env.n_actions, seed=0); steps = train(q, GridWorld(), 300)
    assert steps[-30:].mean() <= env.shortest_path() + 1 and np.allclose(np.linalg.norm(q.amp, axis=1), 1)
    assert train(QLearning(env.n_states, env.n_actions, seed=0), GridWorld(), 300)[-30:].mean() < 15


def test_quantum_language_model():
    from qlcog.inspired import QuantumLanguageModel, QueryLikelihoodModel
    docs = ['the robot picks up the red cup from the table', 'a red apple and a green cup',
            'the cup is on the left', 'robot arms and conveyor belts']
    for M in (QuantumLanguageModel(window=2), QueryLikelihoodModel()):
        M.fit(docs); assert M.rank('robot table')[0] == 0
    m = QuantumLanguageModel().fit(docs)
    assert all(np.isclose(np.trace(r), 1) and np.linalg.eigvalsh(r).min() > 0 for r in m.rhos)
