"""Han and Kim QIEA table, MPS sweep training, entanglement views, individual-level fitting."""
import numpy as np
import pytest
from quantum_mind.problems import maxcut
from quantum_mind.inspired import QIEA, MPSClassifier


def test_qiea_lookup_table_and_simple_rule():
    rng = np.random.default_rng(5)
    q = maxcut([(i, j) for i in range(10) for j in range(i + 1, 10) if rng.random() < 0.4], 10)
    opt = q.brute_force()[1]
    assert QIEA(q, rotation='lookup', generations=200).run().value == opt
    assert QIEA(q, rotation='simple', generations=200).run().value == opt
    t = QIEA(q)._step
    assert t[0, 1, 1] < 0 and t[1, 0, 1] > 0 and t[1, 1, 0] > 0 and t[0, 0, 0] == 0     # towards the better bit
    with pytest.raises(ValueError):
        QIEA(q, rotation='other')


def test_mps_sweep_training_adapts_bonds_and_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 4)); y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    m = MPSClassifier(bond=4, local_dim=3, method='sweep', sweeps=3, steps=25).fit(X, y)
    assert m.score(X, y) > 0.85 and np.allclose(m.predict_proba(X[:4]).sum(1), 1)
    assert len(m.bond_dimensions) == 3 and max(m.bond_dimensions) <= 4 and len(m.history_) == 3
    with pytest.raises(ValueError):
        MPSClassifier(method='sweep').fit(X[:, :1], y)


def test_concurrence_and_entanglement_views():
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace, concurrence as qiskit_concurrence
    from quantum_mind.viz import reduced_density_pair, concurrence, entanglement_summary, circuit_trajectory
    qc = QuantumCircuit(3); qc.h(0); qc.ry(0.7, 1); qc.cx(0, 2); qc.rx(1.1, 2); qc.t(1); qc.cx(1, 0)
    sv = Statevector(qc)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        r = partial_trace(sv, [q for q in range(3) if q not in (a, b)])
        assert np.allclose(reduced_density_pair(sv.data, a, b, 3), r.data)
        assert np.isclose(concurrence(r.data), qiskit_concurrence(r))
    bell = QuantumCircuit(2); bell.h(0); bell.cx(0, 1)
    s = entanglement_summary(Statevector(bell).data)
    assert np.isclose(s['concurrence'][0, 1], 1) and np.allclose(s['bloch_length'], 0)
    tr = circuit_trajectory(bell, steps=4)
    c = tr.concurrence(0, 1)
    assert c[0] == 0 and np.isclose(c[-1], 1) and len(c) == tr.frames


def test_entanglement_and_qsphere_plots():
    pytest.importorskip('matplotlib'); pytest.importorskip('seaborn'); pytest.importorskip('qiskit')
    import matplotlib
    matplotlib.use('Agg')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from quantum_mind.viz import circuit_trajectory, plot_entanglement, plot_qsphere
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
    assert plot_entanglement(circuit_trajectory(qc, steps=3)) is not None
    assert plot_qsphere(Statevector(qc)) is not None


def test_individual_level_fitting():
    from quantum_mind.core import fit_individuals, compare_individuals
    from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel
    from quantum_mind.applications.robotics import domain_models
    g = domain_models('object_clarification'); rng = np.random.default_rng(1)
    people = {('QL-%d' % i if i < 3 else 'AN-%d' % i): (g['QL'] if i < 3 else g['Anchoring']).sample(None, 3000, rng)
              for i in range(6)}
    r = compare_individuals([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], people, rng=rng)
    assert r['best_counts'] == {'QuantumOrderModel4D': 3, 'BayesOrderModel': 0, 'AnchoringOrderModel': 3}
    f = fit_individuals(BayesOrderModel, people)
    assert f.k == 3 * 6 and np.isclose(f.bic, sum(x.bic for x in f.results.values())) and len(f.parameters()['pA']) == 6
    with pytest.raises(ValueError):
        fit_individuals(BayesOrderModel, {})
