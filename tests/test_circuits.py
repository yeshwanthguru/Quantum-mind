import numpy as np
import pytest
pytest.importorskip('qiskit_aer')
from qiskit.quantum_info import Statevector
from quantum_mind.core import tvd
from quantum_mind.circuits import run, order_effects_circuit, qlbn_circuit, walk_circuit, belief_circuit


def test_order_circuits_match_model():
    from quantum_mind.families.order_effects import QuantumOrderModel, QuantumOrderModel4D
    for m in (QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2)), QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)):
        for form in ('dynamic', 'deferred'):
            qc, dec = order_effects_circuit(m, 'BA', form)
            assert tvd(dec(run(qc, 'aer', 20000)), m.predict()['BA']) < 0.02


def test_qlbn_circuit_exact():
    from quantum_mind.families.qlbn import BayesNet, quantum_like_marginal
    net = BayesNet({'O': ['w', 'l'], 'P': ['y', 'n']}, {'P': ['O']},
                   {'O': {(): {'w': .5, 'l': .5}}, 'P': {('w',): {'y': .69, 'n': .31}, ('l',): {'y': .59, 'n': .41}}})
    qc, _ = qlbn_circuit(net, 'P', None, [0, 2.5]); qc.remove_final_measurements()
    p = Statevector(qc).probabilities_dict()
    assert np.isclose(p['00'] / (p['00'] + p['01']), quantum_like_marginal(net, 'P', None, [0, 2.5])['y'])


def test_walk_and_belief_circuits():
    from quantum_mind.families.dynamics import QuantumWalk, OpenSystemBelief, final_yes
    w = QuantumWalk(mu=3, sigma=3, n_states=16); qc, dec = walk_circuit(w, 1.0)
    assert tvd(dec(run(qc, 'aer', 20000)), w.predict({'a': ('single', 1.0)})['a']) < 0.02
    b = OpenSystemBelief(); qc, dec = belief_circuit(b, (1, 1, 0, 1, 0, 1), (2, 5))
    assert abs(dec(run(qc, 'aer', 20000)) - final_yes(b, (1, 1, 0, 1, 0, 1), (2, 5))) < 0.02


def test_braket_local_if_available():
    pytest.importorskip('braket')
    from quantum_mind.families.order_effects import QuantumOrderModel
    m = QuantumOrderModel(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2))
    qc, dec = order_effects_circuit(m, 'AB', 'deferred')
    assert tvd(dec(run(qc, 'braket_local', 5000)), m.predict()['AB']) < 0.03


def test_ibm_submission_path_with_fake_backend():
    """The IBM hardware code path (ISA transpilation + SamplerV2 + count parsing) on a fake device."""
    pytest.importorskip('qiskit_ibm_runtime')
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from quantum_mind.circuits.backends import run_on_ibm_backend
    from quantum_mind.families.order_effects import QuantumOrderModel4D
    m = QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)
    qc, dec = order_effects_circuit(m, 'AB', 'dynamic')
    assert tvd(dec(run_on_ibm_backend(qc, FakeTorino(), 8000)), m.predict()['AB']) < 0.1


def test_braket_submission_path_with_local_device():
    """The Amazon Braket device code path (OpenQASM 3 program to device.run) on the local simulator."""
    pytest.importorskip('braket')
    from braket.devices import LocalSimulator
    from quantum_mind.circuits.backends import run_on_braket_device
    from quantum_mind.families.order_effects import QuantumOrderModel4D
    m = QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)
    qc, dec = order_effects_circuit(m, 'AB', 'deferred')
    assert tvd(dec(run_on_braket_device(qc, LocalSimulator(), 20000)), m.predict()['AB']) < 0.02
