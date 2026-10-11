"""Entanglement views and individual-level fitting."""
import numpy as np
import pytest


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
