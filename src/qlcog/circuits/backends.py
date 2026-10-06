"""Running circuits: one function, many backends.

.. code-block:: python

   run(circuit, backend='aer', shots=10000)

Backend strings:

=========================  =================================================================
``'aer'``                  Qiskit Aer, ideal.
``'aer:<FakeBackend>'``    Aer with the noise model of an IBM fake backend, for example
                           ``'aer:FakeTorino'`` (see ``qiskit_ibm_runtime.fake_provider``).
``'braket_local'``         Amazon Braket local state-vector simulator (circuit sent as
                           OpenQASM 3; deferred-measurement circuits only).
``'braket_dm:<p>'``        Braket local density-matrix simulator with depolarising noise p
                           after every gate.
``'ibm:<backend name>'``   IBM Quantum hardware through the Qiskit Runtime Sampler; needs a
                           saved account.
``'ibm:least_busy'``       The least busy operational IBM device.
``'braket:<device ARN>'``  Amazon Braket managed simulator or QPU; needs AWS credentials;
                           billed.
=========================  =================================================================

:func:`run` returns a dict of bit strings to counts in Qiskit's convention (highest classical bit
first), so the decode functions of :mod:`qlcog.circuits.builders` work unchanged for every backend.
"""
from __future__ import annotations


from .._optional import require

__all__ = ['run', 'to_qasm3', 'run_on_ibm_backend', 'run_on_braket_device']


def run_on_ibm_backend(qc, device, shots):
    """Submit a circuit to an IBM device and return counts.

    Transpiles to the device's instruction set and runs with the Qiskit Runtime Sampler primitive.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
    device : qiskit backend
        A real backend from ``QiskitRuntimeService``, or a fake backend (local testing mode), so the same
        code path is exercised in the tests.
    shots : int

    Returns
    -------
    dict
        Counts.
    """
    try:                                   # client-side Sampler (qiskit-ibm-runtime >= 0.50)
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:                    # older releases: SamplerV2 (deprecated from 0.50)
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    isa = generate_preset_pass_manager(backend=device, optimization_level=3).run(qc)
    res = Sampler(mode=device).run([isa], shots=shots).result()[0]
    creg = qc.cregs[0].name if len(qc.cregs) == 1 else None
    data = getattr(res.data, creg) if creg else res.join_data()
    return dict(data.get_counts())


def run_on_braket_device(qc, device, shots):
    """Submit a circuit to an Amazon Braket device and return counts.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Deferred-measurement form (no reset).
    device : braket device
        An ``AwsDevice`` (billed) or a ``LocalSimulator``, which the tests use as a stand-in.
    shots : int

    Returns
    -------
    dict
        Counts in Qiskit bit order.
    """
    from braket.ir.openqasm import Program
    src, t = to_qasm3(qc)
    res = device.run(Program(source=src), shots=shots).result()
    return _braket_counts_to_qiskit(dict(res.measurement_counts), t)


def to_qasm3(qc):
    """Export a circuit as OpenQASM 3 in Braket's gate names.

    The circuit is transpiled to {rz, sx, x, cx, measure}; ``sx`` becomes ``v`` and ``cx`` becomes
    ``cnot``, and the stdgates include is removed.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit

    Returns
    -------
    source : str
        OpenQASM 3 program.
    transpiled : qiskit.QuantumCircuit

    Raises
    ------
    ValueError
        If the circuit contains a reset (use the deferred-measurement form).
    """
    import re
    require('qiskit')
    from qiskit import transpile, qasm3
    if any(inst.operation.name == 'reset' for inst in qc.data):
        raise ValueError('Braket backends need the deferred-measurement form (no reset).')
    t = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=7)
    src = qasm3.dumps(t)
    src = src.replace('include "stdgates.inc";\n', '')
    src = re.sub(r'\bcx\b', 'cnot', src)
    src = re.sub(r'\bsx\b', 'v', src)
    return src, t


def _braket_counts_to_qiskit(counts, qc, all_qubits=False):
    """Map Braket bit strings (measured or all qubits, in qubit order) to the circuit's classical bits,
    as Qiskit-style strings (highest classical bit first).
    """
    meas = [(qc.find_bit(i.qubits[0]).index, qc.find_bit(i.clbits[0]).index) for i in qc.data if i.operation.name == 'measure']
    measured_qubits = list(range(qc.num_qubits)) if all_qubits else sorted({q for q, _ in meas})
    pos = {q: k for k, q in enumerate(measured_qubits)}
    out = {}
    for bits, k in counts.items():
        cl = ['0'] * qc.num_clbits
        for q, cb in meas:
            cl[cb] = bits[pos[q]]
        key = ''.join(reversed(cl)); out[key] = out.get(key, 0) + k
    return out


def run(qc, backend='aer', shots=10000, seed=7):
    """Run a circuit and return counts.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
    backend : str, optional
        Backend string (see the module description).
    shots : int, optional
        Number of shots.
    seed : int, optional
        Simulator and transpiler seed.

    Returns
    -------
    dict
        ``{bit string: count}`` in Qiskit bit order.

    Raises
    ------
    ValueError
        For an unknown backend string.
    """
    if backend == 'aer' or backend.startswith('aer:'):
        require('qiskit_aer', 'qiskit')
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        if backend == 'aer':
            sim = AerSimulator(seed_simulator=seed); t = transpile(qc, sim, optimization_level=1, seed_transpiler=seed)
        else:
            from qiskit_ibm_runtime import fake_provider
            fb = getattr(fake_provider, backend.split(':', 1)[1])()
            sim = AerSimulator.from_backend(fb, seed_simulator=seed); t = transpile(qc, fb, optimization_level=3, seed_transpiler=seed)
        return dict(sim.run(t, shots=shots).result().get_counts())
    if backend == 'braket_local' or backend.startswith('braket_dm:'):
        require('braket')
        from braket.devices import LocalSimulator
        from braket.circuits import Circuit, Noise
        from braket.ir.openqasm import Program
        src, t = to_qasm3(qc)
        if backend == 'braket_local':
            res = LocalSimulator().run(Program(source=src), shots=shots).result()
            return _braket_counts_to_qiskit(dict(res.measurement_counts), t)
        # noise must be inserted before measurement: drop the measurements, add noise after every
        # gate, and let the simulator measure all qubits at the end
        p = float(backend.split(':', 1)[1])
        body = '\n'.join(l for l in src.splitlines() if '= measure' not in l)
        circ = Circuit.from_ir(body)
        circ.apply_gate_noise(Noise.Depolarizing(probability=p))
        res = LocalSimulator('braket_dm').run(circ, shots=shots).result()
        return _braket_counts_to_qiskit(dict(res.measurement_counts), t, all_qubits=True)
    if backend.startswith('ibm:'):
        require('qiskit_ibm_runtime')
        from qiskit_ibm_runtime import QiskitRuntimeService
        name = backend.split(':', 1)[1]; service = QiskitRuntimeService()
        dev = service.least_busy(operational=True, simulator=False) if name == 'least_busy' else service.backend(name)
        return run_on_ibm_backend(qc, dev, shots)
    if backend.startswith('braket:'):
        require('braket')
        from braket.aws import AwsDevice
        return run_on_braket_device(qc, AwsDevice(backend.split(':', 1)[1]), shots)
    raise ValueError('unknown backend %r' % backend)
