"""Bloch-vector geometry and state trajectories (NumPy only).

A single-qubit density matrix rho = (I + r . sigma) / 2 is represented by its Bloch vector r with
|r| <= 1: pure states lie on the sphere, mixed states inside, and the reduced state of an entangled
qubit shrinks towards the centre.

Trajectories are arrays of shape (frames, qubits, 3) with a list of frame labels; every plotting
function in qlcog.viz accepts them."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ['bloch_vector', 'state_from_bloch', 'reduced_density', 'bloch_vectors', 'Trajectory',
           'circuit_trajectory', 'belief_trajectory', 'rotation_trajectory', 'bloch_tomography',
           'tomography_circuits']

_SX = np.array([[0, 1], [1, 0]], complex)
_SY = np.array([[0, -1j], [1j, 0]], complex)
_SZ = np.array([[1, 0], [0, -1]], complex)


def _to_array(state):
    """Accept a NumPy vector or density matrix, or a Qiskit Statevector / DensityMatrix."""
    if hasattr(state, 'data') and not isinstance(state, np.ndarray):
        state = state.data
    return np.asarray(state, complex)


def bloch_vector(state):
    """Bloch vector (x, y, z) of a qubit given as a 2-vector or a 2 x 2 density matrix."""
    s = _to_array(state)
    rho = np.outer(s, s.conj()) / np.vdot(s, s).real if s.ndim == 1 else s
    return np.real([np.trace(rho @ _SX), np.trace(rho @ _SY), np.trace(rho @ _SZ)])


def state_from_bloch(r):
    """Density matrix of a Bloch vector."""
    x, y, z = r
    return 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])


def reduced_density(state, q, n):
    """Reduced density matrix of qubit q of an n-qubit state vector or density matrix (Qiskit order:
    qubit q = bit q of the basis index)."""
    s = _to_array(state)
    ax = n - 1 - q
    if s.ndim == 1:
        psi = np.moveaxis(s.reshape((2,) * n), ax, 0).reshape(2, -1)
        return psi @ psi.conj().T
    rho = s.reshape((2,) * (2 * n))
    rho = np.moveaxis(rho, [ax, n + ax], [0, n]).reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.einsum('aibi->ab', rho)


def bloch_vectors(state, n=None):
    """Bloch vectors of every qubit, shape (n, 3)."""
    s = _to_array(state); n = n or int(round(np.log2(s.shape[0])))
    return np.array([bloch_vector(reduced_density(s, q, n)) for q in range(n)])


@dataclass
class Trajectory:
    """Bloch vectors over time: vectors (frames, qubits, 3) and one label per frame."""
    vectors: np.ndarray
    labels: list = field(default_factory=list)
    names: list = field(default_factory=list)          # qubit names
    title: str = ''

    def __post_init__(self):
        self.vectors = np.asarray(self.vectors, float)
        if self.vectors.ndim == 2:
            self.vectors = self.vectors[:, None, :]
        if not self.labels:
            self.labels = ['%d' % k for k in range(len(self.vectors))]
        if not self.names:
            self.names = ['q%d' % q for q in range(self.vectors.shape[1])]

    @property
    def frames(self):
        return len(self.vectors)

    @property
    def n_qubits(self):
        return self.vectors.shape[1]

    def purity(self):
        """|r| per frame and qubit (1 = pure, 0 = maximally mixed or maximally entangled)."""
        return np.linalg.norm(self.vectors, axis=-1)


def _fractional(U, t):
    """U^t on the principal branch (unitary U)."""
    from scipy.linalg import schur
    T, Z = schur(U, output='complex')
    ev = np.diag(T)
    return Z @ np.diag(np.exp(1j * t * np.angle(ev))) @ Z.conj().T


def _unitary_of(op):
    from qiskit.quantum_info import Operator
    return Operator(op).data


def circuit_trajectory(qc, steps=12, initial=None):
    """Smooth Bloch-sphere trajectory of every qubit of a Qiskit circuit, gate by gate.

    Each unitary gate U is applied in `steps` fractional steps U^(k/steps), so rotations are drawn as
    arcs and entangling gates as vectors moving into the ball. Measurements, resets and barriers are
    skipped (the trajectory shows the coherent evolution); gates without a matrix (e.g. state
    preparation) are decomposed first."""
    from qiskit import QuantumCircuit, transpile
    from ..quantum.statevector import apply_matrix
    n = qc.num_qubits
    psi = np.zeros((1, 2 ** n), complex); psi[0, 0] = 1
    if initial is not None:
        psi[0] = _to_array(initial)
    vecs = [bloch_vectors(psi[0], n)]; labels = ['start']

    def instructions(circ):
        for inst in circ.data:
            name = inst.operation.name
            if name in ('measure', 'barrier', 'reset', 'delay'):
                continue
            qs = [circ.find_bit(b).index for b in inst.qubits]
            try:
                U = _unitary_of(inst.operation)
                yield name, qs, U
            except Exception:                         # decompose into a basis with matrices
                sub = QuantumCircuit(len(qs)); sub.append(inst.operation, range(len(qs)))
                sub = transpile(sub, basis_gates=['u', 'cx'], optimization_level=0)
                for nm, sq, U in instructions(sub):
                    yield nm, [qs[i] for i in sq], U

    for name, qs, U in instructions(qc):
        Uk = _fractional(U, 1.0 / steps)
        for k in range(1, steps + 1):
            psi = apply_matrix(psi, Uk, qs, n)
            vecs.append(bloch_vectors(psi[0], n)); labels.append(name)
    return Trajectory(np.array(vecs), labels, title='circuit')


def rotation_trajectory(axis, angle, start=(0, 0, 1), steps=60):
    """Rotation of a Bloch vector about `axis` by `angle` (useful for demonstrations)."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a); v = np.asarray(start, float); out = []
    for t in np.linspace(0, angle, steps):
        out.append(v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t)))
    return Trajectory(np.array(out), ['%.2f' % t for t in np.linspace(0, angle, steps)], title='rotation')


def belief_trajectory(model, events, steps=12):
    """Trajectory of the belief qubit of qlcog's OpenSystemBelief (trust) model over a sequence of
    events (1 = success, 0 = failure): each event rotates the belief about the y axis and dephasing of
    strength gamma pulls the vector towards the z axis. |0> (north pole) = 'yes, I trust the robot'."""
    v = np.array([np.cos(model.phi0 / 2), np.sin(model.phi0 / 2)])
    rho = np.outer(v, v); vecs = [bloch_vector(rho)]; labels = ['prior']
    for t, e in enumerate(events):
        a = -model.a_pos if e else model.a_neg
        c, s = np.cos(a / (2 * steps)), np.sin(a / (2 * steps))
        R = np.array([[c, -s], [s, c]])
        f = (1 - model.gamma) ** (1.0 / steps) if model.gamma < 1 else 0.0
        tag = 'event %d: %s' % (t + 1, 'success' if e else 'failure')
        for k in range(steps):                                      # the event rotates the belief ...
            rho = R @ rho @ R.T
            vecs.append(bloch_vector(rho)); labels.append(tag)
        for k in range(steps):                                      # ... then dephasing (as in the model)
            rho = np.array([[rho[0, 0], f * rho[0, 1]], [f * rho[1, 0], rho[1, 1]]])
            vecs.append(bloch_vector(rho)); labels.append(tag + ' (dephasing)')
    return Trajectory(np.array(vecs), labels, ['belief'], 'trust belief')


def tomography_circuits(qc):
    """Three copies of a (measurement-free) circuit measured in the Z, X and Y bases on every qubit."""
    out = {}
    for basis in 'zxy':
        c = qc.remove_final_measurements(inplace=False)
        for q in range(c.num_qubits):
            if basis == 'x':
                c.h(q)
            elif basis == 'y':
                c.sdg(q); c.h(q)
        c.measure_all(); out[basis] = c
    return out


def bloch_tomography(qc, backend='aer', shots=4000, seed=7):
    """Bloch vector of every qubit estimated from measurements on a simulator or quantum hardware
    (single-qubit state tomography). backend: any string accepted by qlcog.circuits.run."""
    from ..circuits import run
    n = qc.num_qubits; r = np.zeros((n, 3))
    for basis, c in tomography_circuits(qc).items():
        counts = run(c, backend, shots, seed); tot = sum(counts.values())
        for q in range(n):
            e = sum(k * (1 - 2 * int(b.replace(' ', '')[::-1][q])) for b, k in counts.items()) / tot
            r[q, 'xyz'.index(basis)] = e
    return r
