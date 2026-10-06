"""A small batched state-vector simulator for parameterised circuits, in pure NumPy.

Qubit q is bit q of the basis-state index (Qiskit ordering). Gate angles are *expressions*: a
number, :class:`W` (trainable weight k), :class:`X` (feature j of each sample, scaled and shifted),
:class:`XX` (a product feature) or a constant.

Circuits are simulated for a whole batch of samples at once, which makes training variational models
fast without a quantum SDK. Gradients are exact and cheap (:meth:`Circuit.value_and_grad`, adjoint
method). :meth:`Circuit.to_qiskit` produces the same circuit for Aer, IBM Quantum or Amazon Braket
(via :func:`qlcog.circuits.run`).

Examples
--------
>>> import numpy as np
>>> from qlcog.quantum import Circuit, W
>>> c = Circuit(2).h(0).cx(0, 1)                   # Bell state
>>> c.probabilities().round(3)
array([[0.5, 0. , 0. , 0.5]])
>>> c = Circuit(1).ry(W(0), 0)                     # one trainable rotation
>>> c.probabilities([np.pi]).round(3)
array([[0., 1.]])
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ['W', 'X', 'XX', 'Circuit', 'apply_matrix', 'basis_probs', 'expectation_z', 'MAX_QUBITS']

#: Largest number of qubits: beyond this a state vector does not fit in typical memory (2^22 x 16 bytes per sample).
MAX_QUBITS = 22
_CHUNK_AMPLITUDES = 2 ** 21   # samples are simulated in chunks of about this many amplitudes (32 MB)


@dataclass(frozen=True)
class W:
    """Trainable weight as a gate angle.

    Parameters
    ----------
    k : int
        Index of the weight in the weight vector.
    scale : float, optional
        Constant factor: the angle is ``scale * w[k]``.
    """
    k: int
    scale: float = 1.0


@dataclass(frozen=True)
class X:
    """Input feature as a gate angle.

    Parameters
    ----------
    j : int
        Feature index.
    scale : float, optional
        Factor.
    shift : float, optional
        Offset: the angle is ``scale * x[j] + shift``.
    """
    j: int
    scale: float = 1.0
    shift: float = 0.0


@dataclass(frozen=True)
class XX:
    """Product feature used by ZZ feature maps.

    Parameters
    ----------
    i, j : int
        Feature indices.
    a : float, optional
        Offset; the angle is ``scale * (a - x[i]) * (a - x[j])`` (Havlíček et al., 2019).
    scale : float, optional
        Factor.
    """
    i: int
    j: int
    a: float = np.pi
    scale: float = 1.0


def _value(p, w, x, B):
    """Angle per sample, shape (B,). w: (K,) shared weights or (B, K) one weight vector per
    sample (used to evaluate many weight settings, e.g. parameter shifts, in one batch)."""
    if isinstance(p, W):
        return p.scale * w[:, p.k] if w.ndim == 2 else np.full(B, p.scale * w[p.k])
    if isinstance(p, X):
        return p.scale * x[:, p.j] + p.shift
    if isinstance(p, XX):
        return p.scale * (p.a - x[:, p.i]) * (p.a - x[:, p.j])
    return np.full(B, float(p))


def _axes(n, q):
    """Axis of qubit q in a (B, 2, ..., 2) array reshaped from the big-endian index."""
    return n - 1 - q


def _apply_1q(state, M, q, n):
    """Fast path for one qubit: view the batch as (B, high, 2, low) with low = 2^q."""
    B = state.shape[0]
    psi = state.reshape(B, 2 ** (n - 1 - q), 2, 2 ** q)
    a0, a1 = psi[:, :, 0, :], psi[:, :, 1, :]              # amplitudes with qubit q = 0 and = 1
    if M.ndim == 3:                                        # one matrix per sample
        m = M[:, :, :, None, None]
        out0 = m[:, 0, 0] * a0 + m[:, 0, 1] * a1
        out1 = m[:, 1, 0] * a0 + m[:, 1, 1] * a1
    else:
        out0 = M[0, 0] * a0 + M[0, 1] * a1
        out1 = M[1, 0] * a0 + M[1, 1] * a1
    return np.stack([out0, out1], 2).reshape(B, 2 ** n)


def _diag_phases(D, qubits, n):
    """Expand the diagonal of a k-qubit diagonal gate to all 2^n basis states (shape (..., 2^n))."""
    idx = np.arange(2 ** n)
    sub = np.zeros(2 ** n, int)
    for i, q in enumerate(qubits):
        sub |= ((idx >> q) & 1) << i
    return D[..., sub]


def apply_matrix(state, M, qubits, n):
    """Apply a k-qubit matrix to a batch of states.

    Parameters
    ----------
    state : numpy.ndarray
        Batch of states, shape ``(B, 2**n)``.
    M : numpy.ndarray
        ``(2**k, 2**k)`` matrix shared by the batch, or ``(B, 2**k, 2**k)`` one per sample. The
        first qubit in ``qubits`` is the least significant bit of M's index.
    qubits : sequence of int
        Qubits acted on.
    n : int
        Total number of qubits.

    Returns
    -------
    numpy.ndarray
        New batch of states, shape ``(B, 2**n)``. Diagonal two-qubit gates use a fast phase path.
    """
    B = state.shape[0]
    k = len(qubits)
    if k == 1:
        return _apply_1q(state, M, qubits[0], n)
    if k == 2 and np.count_nonzero(M - np.einsum('...ii->...i', M)[..., None] * np.eye(4)) == 0:
        return state * _diag_phases(np.einsum('...ii->...i', M), qubits, n)
    # general case: move the target axes to the front, multiply, move them back
    psi = state.reshape((B,) + (2,) * n)
    ax = [1 + _axes(n, q) for q in reversed(qubits)]          # big-endian order of M's index
    psi = np.moveaxis(psi, ax, list(range(1, k + 1))).reshape(B, 2 ** k, -1)
    psi = (M @ psi) if M.ndim == 3 else np.einsum('ij,bjr->bir', M, psi)
    psi = psi.reshape((B,) + (2,) * n)
    psi = np.moveaxis(psi, list(range(1, k + 1)), ax)
    return psi.reshape(B, 2 ** n)


def _rot(name, th):
    """Matrices of a parametric gate for a batch of angles, shape (B, 2, 2) or (B, 4, 4)."""
    c, s = np.cos(th / 2), np.sin(th / 2)
    B = len(th)
    M = np.zeros((B, 2, 2), complex)
    if name == 'ry':
        M[:, 0, 0] = c; M[:, 0, 1] = -s; M[:, 1, 0] = s; M[:, 1, 1] = c
    elif name == 'rx':
        M[:, 0, 0] = c; M[:, 0, 1] = -1j * s; M[:, 1, 0] = -1j * s; M[:, 1, 1] = c
    elif name == 'rz':
        M[:, 0, 0] = np.exp(-0.5j * th); M[:, 1, 1] = np.exp(0.5j * th)
    elif name == 'p':
        M[:, 0, 0] = 1; M[:, 1, 1] = np.exp(1j * th)
    elif name == 'cp':                                         # controlled phase diag(1, 1, 1, e^{i t})
        M = np.zeros((B, 4, 4), complex)
        M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = 1
        M[:, 3, 3] = np.exp(1j * th)
    elif name == 'rzz':                                        # 4 x 4 diagonal
        M = np.zeros((B, 4, 4), complex)
        for b, z in enumerate([1, -1, -1, 1]):
            M[:, b, b] = np.exp(-0.5j * th * z)
    return M


_FIXED = {
    'h': np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    'x': np.array([[0, 1], [1, 0]], complex),
    'y': np.array([[0, -1j], [1j, 0]], complex),
    'z': np.diag([1, -1]).astype(complex),
    's': np.diag([1, 1j]),
    'sdg': np.diag([1, -1j]),
    't': np.diag([1, np.exp(1j * np.pi / 4)]),
    'cx': np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], complex),   # control = first qubit
    'cz': np.diag([1, 1, 1, -1]).astype(complex),
    'swap': np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex),
}
#: Gates whose angle can be a number, :class:`W`, :class:`X` or :class:`XX`.
PARAMETRIC = ('rx', 'ry', 'rz', 'p', 'rzz', 'cp')
# Generators G of the parametric gates, used by the adjoint gradient.
_GENERATOR = {'rx': _FIXED['x'], 'ry': _FIXED['y'], 'rz': _FIXED['z'], 'rzz': np.diag([1, -1, -1, 1]).astype(complex),
              'p': np.diag([0, 1]).astype(complex), 'cp': np.diag([0, 0, 0, 1]).astype(complex)}


def _dagger(M):
    """Conjugate transpose of the last two axes."""
    return M.conj().swapaxes(-1, -2)


class Circuit:
    """Parameterised quantum circuit with a batched simulator.

    Build the circuit with the gate methods (each returns ``self``, so calls chain) and simulate it
    with :meth:`state` or :meth:`probabilities`.

    Parameters
    ----------
    n : int
        Number of qubits, at most :data:`MAX_QUBITS`.

    Attributes
    ----------
    ops : list of tuple
        Gates as ``(name, qubits, parameter)``.

    Raises
    ------
    ValueError
        If ``n`` exceeds :data:`MAX_QUBITS`.
    """

    def __init__(self, n):
        if n > MAX_QUBITS:
            raise ValueError('%d qubits exceed the simulator limit of %d (memory grows as 2^n)' % (n, MAX_QUBITS))
        self.n = n
        self.ops = []

    def _add(self, name, qubits, param=None):
        """Append a gate and return self."""
        self.ops.append((name, tuple(qubits), param))
        return self

    # ----------------------------------------------------------------------------------- fixed gates
    def h(self, q):
        """Hadamard on qubit ``q``."""
        return self._add('h', [q])

    def x(self, q):
        """Pauli X on qubit ``q``."""
        return self._add('x', [q])

    def y(self, q):
        """Pauli Y on qubit ``q``."""
        return self._add('y', [q])

    def z(self, q):
        """Pauli Z on qubit ``q``."""
        return self._add('z', [q])

    def s(self, q):
        """S gate (phase pi/2) on qubit ``q``."""
        return self._add('s', [q])

    def sdg(self, q):
        """S-dagger gate on qubit ``q``."""
        return self._add('sdg', [q])

    def cx(self, c, t):
        """CNOT with control ``c`` and target ``t``."""
        return self._add('cx', [c, t])

    def cz(self, a, b):
        """Controlled Z on qubits ``a`` and ``b``."""
        return self._add('cz', [a, b])

    def swap(self, a, b):
        """Swap qubits ``a`` and ``b``."""
        return self._add('swap', [a, b])

    # ----------------------------------------------------------------------------------- parametric gates
    def rx(self, p, q):
        """RX rotation on qubit ``q``; ``p`` is a number, :class:`W`, :class:`X` or :class:`XX`."""
        return self._add('rx', [q], p)

    def ry(self, p, q):
        """RY rotation on qubit ``q``; ``p`` is a number or an angle expression."""
        return self._add('ry', [q], p)

    def rz(self, p, q):
        """RZ rotation on qubit ``q``; ``p`` is a number or an angle expression."""
        return self._add('rz', [q], p)

    def p(self, p, q):
        """Phase gate diag(1, e^{ip}) on qubit ``q``."""
        return self._add('p', [q], p)

    def cp(self, p, c, t):
        """Controlled phase diag(1, 1, 1, e^{ip}) on qubits ``c`` and ``t`` (symmetric)."""
        return self._add('cp', [c, t], p)

    def rzz(self, p, a, b):
        """RZZ(p) = exp(-i p Z_a Z_b / 2) on qubits ``a`` and ``b``."""
        return self._add('rzz', [a, b], p)

    def unitary(self, M, qubits, label='U'):
        """Fixed unitary on some qubits.

        Parameters
        ----------
        M : array_like
            ``(2**k, 2**k)`` unitary.
        qubits : sequence of int
            Qubits acted on (the first is the least significant bit of M's index).
        label : str, optional
            Label in the Qiskit export.

        Returns
        -------
        Circuit
            ``self``.
        """
        self.ops.append(('unitary', tuple(qubits), (np.asarray(M, complex), label)))
        return self

    def diagonal(self, phases, label='D'):
        """Diagonal unitary diag(exp(i phases)) on all qubits (for example a cost or oracle layer).

        Parameters
        ----------
        phases : array_like
            ``2**n`` phases.
        label : str, optional
            Label.

        Returns
        -------
        Circuit
            ``self``.
        """
        self.ops.append(('diag', tuple(range(self.n)), (np.asarray(phases, float), label)))
        return self

    @property
    def n_weights(self):
        """int: number of trainable weights (largest :class:`W` index + 1)."""
        ks = [p.k for _, _, p in self.ops if isinstance(p, W)]
        return 1 + max(ks) if ks else 0

    def compose(self, other):
        """Append the gates of another circuit.

        Parameters
        ----------
        other : Circuit
            Circuit on the same number of qubits.

        Returns
        -------
        Circuit
            ``self``.
        """
        self.ops += other.ops
        return self

    # ----------------------------------------------------------------------------------- simulation
    def state(self, w=None, X_=None, init=None):
        """Simulate the circuit.

        Parameters
        ----------
        w : array_like, optional
            Weights, shape ``(K,)`` shared by the batch or ``(B, K)`` one per sample. Default zeros.
        X_ : array_like, optional
            Input samples, shape ``(B, n_features)``.
        init : array_like, optional
            Initial state (default :math:`|0\\dots0\\rangle`).

        Returns
        -------
        numpy.ndarray
            Final states, shape ``(B, 2**n)``.

        Raises
        ------
        ValueError
            If fewer weights than :attr:`n_weights` are given.
        """
        w = np.zeros(self.n_weights) if w is None else np.asarray(w, float)
        Xa = None if X_ is None else np.atleast_2d(np.asarray(X_, float))
        B = Xa.shape[0] if Xa is not None else (w.shape[0] if w.ndim == 2 else 1)
        if w.ndim == 1 and len(w) < self.n_weights:
            raise ValueError('expected %d weights, got %d' % (self.n_weights, len(w)))
        if init is None:
            psi = np.zeros((B, 2 ** self.n), complex)
            psi[:, 0] = 1
        else:
            psi = np.tile(np.asarray(init, complex), (B, 1))
        for op in self.ops:
            psi = self._apply(op, psi, w, Xa, B)
        return psi

    def _matrix(self, op, w, Xa, B):
        """Matrix (or batch of matrices) of one gate."""
        name, qs, p = op
        if name in PARAMETRIC:
            return _rot(name, _value(p, w, Xa, B))
        if name == 'unitary':
            return p[0]
        return _FIXED[name]

    def _apply(self, op, psi, w, Xa, B, inverse=False):
        """Apply one gate (or its inverse) to a batch of states."""
        name, qs, p = op
        if name == 'diag':
            return psi * np.exp((-1j if inverse else 1j) * p[0])[None, :]
        M = self._matrix(op, w, Xa, B)
        return apply_matrix(psi, _dagger(M) if inverse else M, list(qs), self.n)

    def value_and_grad(self, w, X_, loss, chunk=None):
        """Loss and its exact gradient with respect to the weights (adjoint method).

        The circuit is run forward once, then backwards once with the inverse gates, so the cost is
        about three simulations whatever the number of weights. Samples are processed in chunks
        (memory ``O(chunk * 2**n)``). Weights shared by several gates (for example QAOA angles) are
        handled.

        Parameters
        ----------
        w : array_like
            Weights, shape ``(K,)``.
        X_ : array_like or None
            Input samples, shape ``(B, n_features)``.
        loss : callable
            ``loss(psi, rows) -> (L, phi)``: ``L`` is the loss summed over the samples ``rows`` of the
            chunk and ``phi`` = dL/d(conj psi), with the shape of ``psi``.
        chunk : int, optional
            Samples per chunk (default: about 2**21 amplitudes).

        Returns
        -------
        loss : float
            Total loss.
        grad : numpy.ndarray
            Gradient, shape ``(K,)``.
        """
        w = np.asarray(w, float)
        Xa = None if X_ is None else np.atleast_2d(np.asarray(X_, float))
        B = 1 if Xa is None else Xa.shape[0]
        chunk = chunk or max(1, _CHUNK_AMPLITUDES // 2 ** self.n)
        total, grad = 0.0, np.zeros(self.n_weights)
        for start in range(0, B, chunk):
            rows = np.arange(start, min(B, start + chunk))
            Xc = None if Xa is None else Xa[rows]
            b = len(rows)
            psi = self.state(w, Xc)
            L, phi = loss(psi, rows)
            total += float(L)
            # walk the circuit backwards, undoing each gate on psi and on the adjoint state phi
            for op in reversed(self.ops):
                name, qs, p = op
                if name in PARAMETRIC and isinstance(p, W):
                    Gpsi = apply_matrix(psi, _GENERATOR[name], list(qs), self.n)
                    x = np.einsum('bi,bi->', phi.conj(), Gpsi)
                    # rotation exp(-i t G / 2): dL/dt = Im<phi|G|psi>; phase gate P(t): dL/dt = -2 Im<phi|1><1|psi>
                    grad[p.k] += p.scale * (-2 * x.imag if name in ('p', 'cp') else x.imag)
                psi = self._apply(op, psi, w, Xc, b, inverse=True)
                phi = self._apply(op, phi, w, Xc, b, inverse=True)
        return total, grad

    def probabilities(self, w=None, X_=None):
        """Born-rule probabilities of the final states.

        Parameters
        ----------
        w : array_like, optional
            Weights.
        X_ : array_like, optional
            Input samples.

        Returns
        -------
        numpy.ndarray
            Shape ``(B, 2**n)``.
        """
        return np.abs(self.state(w, X_)) ** 2

    # ----------------------------------------------------------------------------------- export
    def to_qiskit(self, w=None, x=None, measure=True):
        """Export to Qiskit with every angle bound.

        Parameters
        ----------
        w : array_like, optional
            Weights.
        x : array_like, optional
            One input sample.
        measure : bool, optional
            Add a final measurement of every qubit.

        Returns
        -------
        qiskit.QuantumCircuit
            Ready for :func:`qlcog.circuits.run` on Aer, IBM Quantum or Amazon Braket.
        """
        from .._optional import require
        require('qiskit', 'qiskit')
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate, DiagonalGate
        w = np.zeros(self.n_weights) if w is None else np.asarray(w, float)
        xa = None if x is None else np.atleast_2d(np.asarray(x, float))
        qc = QuantumCircuit(self.n)
        for name, qs, p in self.ops:
            if name in PARAMETRIC:
                getattr(qc, name)(float(_value(p, w, xa, 1)[0]), *qs)
            elif name == 'unitary':
                qc.append(UnitaryGate(p[0], label=p[1]), list(qs))
            elif name == 'diag':
                qc.append(DiagonalGate(list(np.exp(1j * p[0]))), list(qs))
            else:
                getattr(qc, name)(*qs)
        if measure:
            qc.measure_all()
        return qc

    def __repr__(self):
        return 'Circuit(%d qubits, %d gates, %d weights)' % (self.n, len(self.ops), self.n_weights)


def basis_probs(psi):
    """Born-rule probabilities.

    Parameters
    ----------
    psi : numpy.ndarray
        State or batch of states.

    Returns
    -------
    numpy.ndarray
        :math:`|\\psi|^2`.
    """
    return np.abs(psi) ** 2


def expectation_z(psi, q, n):
    """Expectation of Pauli Z on one qubit.

    Parameters
    ----------
    psi : numpy.ndarray
        Batch of states, shape ``(B, 2**n)``.
    q : int
        Qubit.
    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        :math:`\\langle Z_q \\rangle` for each state, shape ``(B,)``.
    """
    idx = np.arange(2 ** n)
    sign = 1 - 2 * ((idx >> q) & 1)                  # +1 if qubit q is 0, -1 if it is 1
    return (np.abs(psi) ** 2) @ sign
