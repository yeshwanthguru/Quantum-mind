r"""Bloch-vector geometry and state trajectories (NumPy only).

A single-qubit density matrix :math:`\rho = (I + \vec r\cdot\vec\sigma)/2` is represented by its
Bloch vector :math:`\vec r` with :math:`|\vec r| \le 1`: pure states lie on the sphere, mixed states
inside, and the reduced state of an entangled qubit shrinks toward the centre.

Trajectories (:class:`Trajectory`) hold arrays of shape ``(frames, qubits, 3)`` with one label per
frame; every plotting function in :mod:`quantum_mind.viz` accepts them.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.viz import bloch_vector, bloch_vectors, concurrence
>>> bloch_vector([1, 1] / np.sqrt(2))                 # |+> points along +x
array([1., 0., 0.])
>>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
>>> np.round(bloch_vectors(bell), 6)                 # both reduced states are maximally mixed
array([[0., 0., 0.],
       [0., 0., 0.]])
>>> round(concurrence(bell), 6)
1.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ['bloch_vector', 'state_from_bloch', 'reduced_density', 'reduced_density_pair', 'concurrence',
           'entanglement_summary', 'bloch_vectors', 'Trajectory',
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
    r"""Bloch vector of one qubit.

    Parameters
    ----------
    state : array_like or qiskit.quantum_info.Statevector or DensityMatrix
        A 2-vector or a 2 x 2 density matrix.

    Returns
    -------
    numpy.ndarray
        ``(x, y, z)`` with :math:`r_k = \mathrm{tr}(\rho\,\sigma_k)`.
    """
    s = _to_array(state)
    rho = np.outer(s, s.conj()) / np.vdot(s, s).real if s.ndim == 1 else s
    return np.real([np.trace(rho @ _SX), np.trace(rho @ _SY), np.trace(rho @ _SZ)])


def state_from_bloch(r):
    r"""Density matrix of a Bloch vector.

    Parameters
    ----------
    r : sequence of float
        ``(x, y, z)``.

    Returns
    -------
    numpy.ndarray
        :math:`(I + \vec r\cdot\vec\sigma) / 2`.
    """
    x, y, z = r
    return 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])


def reduced_density(state, q, n):
    """Reduced density matrix of one qubit.

    Parameters
    ----------
    state : array_like
        n-qubit state vector or density matrix (Qiskit order: qubit q = bit q of the basis index).
    q : int
        Qubit to keep.
    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        2 x 2 density matrix.
    """
    s = _to_array(state)
    ax = n - 1 - q
    if s.ndim == 1:
        psi = np.moveaxis(s.reshape((2,) * n), ax, 0).reshape(2, -1)
        return psi @ psi.conj().T
    rho = s.reshape((2,) * (2 * n))
    rho = np.moveaxis(rho, [ax, n + ax], [0, n]).reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.einsum('aibi->ab', rho)


def reduced_density_pair(state, a, b, n):
    """Reduced density matrix of two qubits.

    Parameters
    ----------
    state : array_like
        n-qubit state vector.
    a, b : int
        Qubits to keep; index bit 0 of the result is qubit a.
    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        4 x 4 density matrix.
    """
    psi = _to_array(state)
    if psi.ndim != 1:
        raise ValueError('reduced_density_pair expects a state vector')
    T = psi.reshape((2,) * n)
    T = np.moveaxis(T, [n - 1 - b, n - 1 - a], [0, 1]).reshape(4, -1)      # row index = 2 * bit_b + bit_a
    return T @ T.conj().T


def concurrence(rho):
    """Wootters concurrence of a two-qubit state.

    Parameters
    ----------
    rho : array_like
        4 x 4 density matrix or 4-vector.

    Returns
    -------
    float
        0 for separable states, 1 for maximally entangled ones (checked against Qiskit).
    """
    rho = _to_array(rho)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    yy = np.kron(_SY, _SY)
    R = rho @ yy @ rho.conj() @ yy
    lam = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(R)))[::-1], 0, None))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def entanglement_summary(state, n=None):
    """Entanglement overview of a pure multi-qubit state.

    Parameters
    ----------
    state : array_like
        State vector.
    n : int, optional
        Number of qubits (inferred from the length).

    Returns
    -------
    dict
        ``bloch_length``: per-qubit Bloch-vector length (1 = not entangled with the rest);
        ``concurrence``: n x n matrix of pairwise concurrences.
    """
    psi = _to_array(state); n = n or int(round(np.log2(psi.shape[0])))
    C = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            C[a, b] = C[b, a] = concurrence(reduced_density_pair(psi, a, b, n))
    return {'bloch_length': np.linalg.norm(bloch_vectors(psi, n), axis=1), 'concurrence': C}


def bloch_vectors(state, n=None):
    """Bloch vectors of every qubit.

    Parameters
    ----------
    state : array_like
        n-qubit state vector or density matrix.
    n : int, optional
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 3)``.
    """
    s = _to_array(state); n = n or int(round(np.log2(s.shape[0])))
    return np.array([bloch_vector(reduced_density(s, q, n)) for q in range(n)])


@dataclass
class Trajectory:
    """Bloch vectors over time.

    Parameters
    ----------
    vectors : array_like
        Shape ``(frames, qubits, 3)`` (or ``(frames, 3)`` for one qubit).
    labels : list of str, optional
        One label per frame (for example the gate being applied).
    names : list of str, optional
        Qubit names.
    title : str, optional
        Title for plots.
    states : list of numpy.ndarray, optional
        Full state per frame (circuit trajectories), needed for :meth:`concurrence`.
    """
    vectors: np.ndarray
    labels: list = field(default_factory=list)
    names: list = field(default_factory=list)          # qubit names
    title: str = ''
    states: list = field(default_factory=list, repr=False)   # full state per frame (circuit trajectories)

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
        """int: number of frames."""
        return len(self.vectors)

    @property
    def n_qubits(self):
        """int: number of qubits."""
        return self.vectors.shape[1]

    def concurrence(self, a=0, b=1):
        """Concurrence of two qubits in every frame.

        Parameters
        ----------
        a, b : int, optional
            Qubits.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If the trajectory has no stored states (only circuit trajectories do).
        """
        if not self.states:
            raise ValueError('this trajectory has no stored states')
        n = self.n_qubits
        return np.array([concurrence(reduced_density_pair(psi, a, b, n)) for psi in self.states])

    def purity(self):
        """Bloch-vector length per frame and qubit.

        Returns
        -------
        numpy.ndarray
            1 for a pure (unentangled) qubit, 0 for a maximally mixed or maximally entangled one.
        """
        return np.linalg.norm(self.vectors, axis=-1)


def _fractional(U, t):
    """U^t on the principal branch, for a unitary U (via the Schur decomposition)."""
    from scipy.linalg import schur
    T, Z = schur(U, output='complex')
    ev = np.diag(T)
    return Z @ np.diag(np.exp(1j * t * np.angle(ev))) @ Z.conj().T


def _unitary_of(op):
    """Matrix of a Qiskit instruction."""
    from qiskit.quantum_info import Operator
    return Operator(op).data


def circuit_trajectory(qc, steps=12, initial=None):
    r"""Smooth Bloch-sphere trajectory of every qubit of a Qiskit circuit, gate by gate.

    Each unitary gate U is applied in ``steps`` fractional steps :math:`U^{k/\text{steps}}`, so rotations
    are drawn as arcs and entangling gates as vectors moving into the ball. Measurements, resets and
    barriers are skipped (the trajectory shows the coherent evolution); gates without a matrix (for
    example state preparation) are decomposed first.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
    steps : int, optional
        Frames per gate.
    initial : array_like, optional
        Initial state (default all zeros).

    Returns
    -------
    Trajectory
        With the full state stored in every frame.
    """
    from qiskit import QuantumCircuit, transpile
    from ..quantum.statevector import apply_matrix
    n = qc.num_qubits
    psi = np.zeros((1, 2 ** n), complex); psi[0, 0] = 1
    if initial is not None:
        psi[0] = _to_array(initial)
    vecs = [bloch_vectors(psi[0], n)]; labels = ['start']; states = [psi[0].copy()]

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
            vecs.append(bloch_vectors(psi[0], n)); labels.append(name); states.append(psi[0].copy())
    return Trajectory(np.array(vecs), labels, title='circuit', states=states)


def rotation_trajectory(axis, angle, start=(0, 0, 1), steps=60):
    """Rotation of a Bloch vector about an axis (for demonstrations).

    Parameters
    ----------
    axis : array_like
        Rotation axis.
    angle : float
        Total angle.
    start : array_like, optional
        Initial Bloch vector.
    steps : int, optional
        Number of frames.

    Returns
    -------
    Trajectory
    """
    a = np.asarray(axis, float); a /= np.linalg.norm(a); v = np.asarray(start, float); out = []
    for t in np.linspace(0, angle, steps):
        out.append(v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t)))
    return Trajectory(np.array(out), ['%.2f' % t for t in np.linspace(0, angle, steps)], title='rotation')


def belief_trajectory(model, events, steps=12):
    r"""Trajectory of the belief qubit of the trust model over a sequence of events.

    Each event (1 = success, 0 = failure) rotates the belief about the y axis of an
    :class:`~quantum_mind.families.dynamics.OpenSystemBelief`; dephasing of strength gamma then pulls the vector
    toward the z axis. The north pole :math:`|0\rangle` means "yes, I trust the robot".

    Parameters
    ----------
    model : OpenSystemBelief
    events : sequence of int
    steps : int, optional
        Frames per rotation and per dephasing.

    Returns
    -------
    Trajectory
    """
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
    """Circuits for single-qubit state tomography.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        Circuit (final measurements are removed).

    Returns
    -------
    dict
        ``{'z': ..., 'x': ..., 'y': ...}``: copies measured in the Z, X and Y bases on every qubit.
    """
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
    """Bloch vectors estimated from measurements on a simulator or quantum hardware.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
    backend : str, optional
        Any backend string accepted by :func:`quantum_mind.circuits.run`.
    shots : int, optional
        Shots per basis.
    seed : int, optional
        Simulator seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 3)``.
    """
    from ..circuits import run
    n = qc.num_qubits; r = np.zeros((n, 3))
    for basis, c in tomography_circuits(qc).items():
        counts = run(c, backend, shots, seed); tot = sum(counts.values())
        for q in range(n):
            e = sum(k * (1 - 2 * int(b.replace(' ', '')[::-1][q])) for b, k in counts.items()) / tot
            r[q, 'xyz'.index(basis)] = e
    return r
