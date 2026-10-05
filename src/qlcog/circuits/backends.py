"""Running circuits: one function, many backends.

    run(circuit, backend='aer', shots=10000)

backend strings
  'aer'                 Qiskit Aer, ideal
  'aer:<FakeBackend>'   Aer with the noise model of an IBM fake backend, e.g. 'aer:FakeTorino'
                        (list: qiskit_ibm_runtime.fake_provider)
  'braket_local'        Amazon Braket local state-vector simulator (circuit sent as OpenQASM 3;
                        deferred-measurement circuits only)
  'braket_dm:<p>'       Braket local density-matrix simulator with depolarizing noise p after every gate
  'ibm:<backend name>'  IBM Quantum hardware through Qiskit Runtime (SamplerV2); needs a saved account
  'ibm:least_busy'      the least busy operational IBM device
  'braket:<device ARN>' Amazon Braket managed simulator or QPU; needs AWS credentials; billed

Returns a dict of bit strings to counts in Qiskit's convention (highest classical bit first), so the
decode functions of builders.py work unchanged for every backend."""
from __future__ import annotations


__all__ = ['run', 'to_qasm3']


def to_qasm3(qc):
    """Transpile to {rz, sx, x, cx, measure} and export OpenQASM 3 in Braket's gate names
    (sx -> v, cx -> cnot; no stdgates include). Used for every Braket backend."""
    import re
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
    """Braket returns bit strings over the measured qubits (or all qubits) in qubit order; map them to
    the circuit's classical bits and return Qiskit-style strings (highest classical bit first)."""
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
    if backend == 'aer' or backend.startswith('aer:'):
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
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        name = backend.split(':', 1)[1]; service = QiskitRuntimeService()
        dev = service.least_busy(operational=True, simulator=False) if name == 'least_busy' else service.backend(name)
        isa = generate_preset_pass_manager(backend=dev, optimization_level=3).run(qc)
        res = SamplerV2(mode=dev).run([isa], shots=shots).result()[0]
        creg = qc.cregs[0].name if len(qc.cregs) == 1 else None
        data = getattr(res.data, creg) if creg else res.join_data()
        return dict(data.get_counts())
    if backend.startswith('braket:'):
        from braket.aws import AwsDevice
        from braket.ir.openqasm import Program
        src, t = to_qasm3(qc)
        res = AwsDevice(backend.split(':', 1)[1]).run(Program(source=src), shots=shots).result()
        return _braket_counts_to_qiskit(dict(res.measurement_counts), t)
    raise ValueError('unknown backend %r' % backend)
