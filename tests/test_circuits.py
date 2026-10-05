import numpy as np
import pytest
pytest.importorskip('qiskit_aer')
from qiskit.quantum_info import Statevector
from qlcog.core import tvd
from qlcog.circuits import (run, order_effects_circuit, interference_circuit, qlbn_circuit, walk_circuit, belief_circuit,
                            chsh_circuit, conjunction_circuit, similarity_circuit)


def test_order_circuits_match_model():
    from qlcog.families.order_effects import QuantumOrderModel, QuantumOrderModel4D
    for m in (QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2)), QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)):
        for form in ('dynamic', 'deferred'):
            qc, dec = order_effects_circuit(m, 'BA', form)
            assert tvd(dec(run(qc, 'aer', 20000)), m.predict()['BA']) < 0.02


def test_interference_and_qlbn_exact():
    from qlcog.families.interference import InterferenceModel
    from qlcog.families.qlbn import BayesNet, quantum_like_marginal
    im = InterferenceModel(p1=0.97, p2=0.84, c=0.5, theta=2.5)
    qc, _ = interference_circuit(im.p1, im.p2, im.c, im.theta, 'unknown'); qc.remove_final_measurements()
    p = Statevector(qc).probabilities_dict()
    assert np.isclose(p['00'] / (p['00'] + p['10']), im.predict()['unknown'][0])
    net = BayesNet({'O': ['w', 'l'], 'P': ['y', 'n']}, {'P': ['O']},
                   {'O': {(): {'w': .5, 'l': .5}}, 'P': {('w',): {'y': .69, 'n': .31}, ('l',): {'y': .59, 'n': .41}}})
    qc, _ = qlbn_circuit(net, 'P', None, [0, 2.5]); qc.remove_final_measurements()
    p = Statevector(qc).probabilities_dict()
    assert np.isclose(p['00'] / (p['00'] + p['01']), quantum_like_marginal(net, 'P', None, [0, 2.5])['y'])


def test_walk_belief_chsh_conjunction_similarity():
    from qlcog.families.dynamics import QuantumWalk, OpenSystemBelief, final_yes
    from qlcog.families.conjunction import QuantumConjunctionModel
    from qlcog.families.similarity import QuantumSimilarityModel
    w = QuantumWalk(mu=3, sigma=3, n_states=16); qc, dec = walk_circuit(w, 1.0)
    assert tvd(dec(run(qc, 'aer', 20000)), w.predict({'a': ('single', 1.0)})['a']) < 0.02
    b = OpenSystemBelief(); qc, dec = belief_circuit(b, (1, 1, 0, 1, 0, 1), (2, 5))
    assert abs(dec(run(qc, 'aer', 20000)) - final_yes(b, (1, 1, 0, 1, 0, 1), (2, 5))) < 0.02
    qc, dec = chsh_circuit(0, np.pi / 4); assert abs(dec(run(qc, 'aer', 20000)) - np.cos(np.pi / 4)) < 0.03
    cm = QuantumConjunctionModel(a=1.910885, b=0.80936, g=0.12292, ranks=(1, 2)); qc, dec = conjunction_circuit(cm)
    assert abs(dec(run(qc, 'aer', 20000))['A&B'] - cm.judgements()['A&B']) < 0.02
    sm = QuantumSimilarityModel(concepts=['K', 'C'], ranks={'C': 2}); qc, dec = similarity_circuit(sm, 'K', 'C')
    assert abs(dec(run(qc, 'aer', 20000)) - sm.predict([('K', 'C')])[('K', 'C')][0]) < 0.02


def test_braket_local_if_available():
    pytest.importorskip('braket')
    from qlcog.families.order_effects import QuantumOrderModel
    m = QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2))
    qc, dec = order_effects_circuit(m, 'AB', 'deferred')
    assert tvd(dec(run(qc, 'braket_local', 5000)), m.predict()['AB']) < 0.03


def test_ibm_submission_path_with_fake_backend():
    """The IBM hardware code path (ISA transpilation + SamplerV2 + count parsing) on a fake device."""
    pytest.importorskip('qiskit_ibm_runtime')
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qlcog.circuits.backends import run_on_ibm_backend
    qc, dec = chsh_circuit(0.0, np.pi / 4)
    E = dec(run_on_ibm_backend(qc, FakeTorino(), 4000))
    assert abs(E - np.cos(np.pi / 4)) < 0.1


def test_braket_submission_path_with_local_device():
    """The Amazon Braket device code path (OpenQASM 3 program to device.run) on the local simulator."""
    pytest.importorskip('braket')
    from braket.devices import LocalSimulator
    from qlcog.circuits.backends import run_on_braket_device
    from qlcog.families.order_effects import QuantumOrderModel4D
    m = QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)
    qc, dec = order_effects_circuit(m, 'AB', 'deferred')
    assert tvd(dec(run_on_braket_device(qc, LocalSimulator(), 20000)), m.predict()['AB']) < 0.02
