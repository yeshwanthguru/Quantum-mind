import numpy as np
import pytest
from quantum_mind.viz import (bloch_vector, bloch_vectors, reduced_density, circuit_trajectory, belief_trajectory,
                       rotation_trajectory, Trajectory)


def test_bloch_vectors_match_qiskit_partial_trace():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
    qc = QuantumCircuit(3); qc.h(0); qc.ry(0.7, 1); qc.cx(0, 2); qc.rx(1.1, 2); qc.t(1); qc.cx(1, 0)
    sv = Statevector(qc)
    for q in range(3):
        rq = partial_trace(sv, [i for i in range(3) if i != q])
        assert np.allclose(bloch_vector(rq), bloch_vectors(sv)[q])
        assert np.allclose(reduced_density(DensityMatrix(qc).data, q, 3), rq.data)
    assert np.allclose(bloch_vector([1, 1j] / np.sqrt(2)), [0, 1, 0])


def test_circuit_trajectory_ends_at_the_final_state():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.ry(0.4, 1)
    tr = circuit_trajectory(qc, steps=5)
    assert tr.vectors.shape == (16, 2, 3)
    assert np.allclose(tr.vectors[-1], bloch_vectors(Statevector(qc)))
    assert tr.purity()[10].max() < 1e-9                       # Bell state: both reduced states maximally mixed
    q2 = QuantumCircuit(2); q2.prepare_state([0.6, 0, 0, 0.8j], [0, 1]); q2.measure_all()
    assert np.allclose(circuit_trajectory(q2).vectors[-1], bloch_vectors(np.array([0.6, 0, 0, 0.8j])))


def test_belief_trajectory_matches_the_trust_model():
    from quantum_mind.families.dynamics import OpenSystemBelief, final_yes
    m = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3); ev = (1, 1, 0, 1, 0, 1)
    tr = belief_trajectory(m, ev, steps=6)
    assert np.isclose((1 + tr.vectors[-1, 0, 2]) / 2, final_yes(m, ev, (5,)))
    r = rotation_trajectory([1, 0, 0], np.pi / 2, steps=3)
    assert np.allclose(r.vectors[-1, 0], [0, -1, 0])


def test_plotting_back_ends(tmp_path):
    pytest.importorskip('matplotlib'); pytest.importorskip('plotly')
    import matplotlib
    matplotlib.use('Agg')
    from quantum_mind.viz import plot_bloch, animate_trajectory, animate_bloch, bloch_figure, save_html, LiveBloch
    tr = Trajectory(rotation_trajectory([0, 1, 0], np.pi, steps=6).vectors)
    fig, _ = plot_bloch(np.array([[0, 0, 1], [1, 0, 0]]))
    animate_trajectory(tr, save=tmp_path / 'a.gif', fps=5)
    assert (tmp_path / 'a.gif').stat().st_size > 0
    f = animate_bloch(tr); assert len(f.frames) == 6
    save_html(f, tmp_path / 'a.html'); assert (tmp_path / 'a.html').stat().st_size > 0
    assert len(bloch_figure(np.array([[0, 0, 1]])).data) > 3
    LiveBloch(1, backend='matplotlib').play(tr, fps=1000)


def test_tomography_on_aer():
    pytest.importorskip('qiskit_aer')
    from qiskit import QuantumCircuit
    from quantum_mind.viz import bloch_tomography
    qc = QuantumCircuit(1); qc.ry(1.0, 0); qc.rz(0.5, 0)
    r = bloch_tomography(qc, 'aer', 20000)[0]
    assert np.allclose(r, [np.sin(1) * np.cos(0.5), np.sin(1) * np.sin(0.5), np.cos(1)], atol=0.03)
