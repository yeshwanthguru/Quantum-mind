"""A small batched state-vector simulator for parameterised circuits (pure NumPy).

Qubit q is bit q of the basis-state index (Qiskit ordering). Gate angles are expressions: a number,
W(k) (trainable weight k), X(j, scale) (feature j of each sample times scale) or a product with a
constant. Circuits are simulated for a whole batch of samples at once, which makes training of
variational models fast without a quantum SDK; `to_qiskit` produces the same circuit for Aer, IBM
Quantum or Amazon Braket (via `qlcog.circuits.run`)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ['W', 'X', 'XX', 'Circuit', 'apply_matrix', 'basis_probs', 'expectation_z', 'MAX_QUBITS']

MAX_QUBITS = 22          # state vectors beyond this size do not fit in typical memory (2^22 x 16 bytes per sample)
_CHUNK_AMPLITUDES = 2 ** 21   # samples are simulated in chunks of about this many amplitudes (32 MB)


@dataclass(frozen=True)
class W:
    """Trainable weight k (times a constant scale)."""
    k: int
    scale: float = 1.0


@dataclass(frozen=True)
class X:
    """Feature j of the input sample (times scale, plus shift)."""
    j: int
    scale: float = 1.0
    shift: float = 0.0


@dataclass(frozen=True)
class XX:
    """Product feature  scale * (a - x_i)(a - x_j)  used by ZZ feature maps."""
    i: int
    j: int
    a: float = np.pi
    scale: float = 1.0


def _value(p, w, x, B):
    """Angle per sample: array of shape (B,). w: (K,) shared weights or (B, K) one weight vector per
    sample (used to evaluate many weight settings, e.g. parameter shifts, in one batch)."""
    if isinstance(p, W):
        return p.scale * w[:, p.k] if w.ndim == 2 else np.full(B, p.scale * w[p.k])
    if isinstance(p, X):
        return p.scale * x[:, p.j] + p.shift
    if isinstance(p, XX):
        return p.scale * (p.a - x[:, p.i]) * (p.a - x[:, p.j])
    return np.full(B, float(p))


def _axes(n, q):
    return n - 1 - q          # axis of qubit q in a (B, 2, ..., 2) array reshaped from big-endian index


def _apply_1q(state, M, q, n):
    """Fast path for one qubit: view the batch as (B, high, 2, low) with low = 2^q."""
    B = state.shape[0]; psi = state.reshape(B, 2 ** (n - 1 - q), 2, 2 ** q)
    a0, a1 = psi[:, :, 0, :], psi[:, :, 1, :]
    if M.ndim == 3:
        m = M[:, :, :, None, None]
        out0 = m[:, 0, 0] * a0 + m[:, 0, 1] * a1; out1 = m[:, 1, 0] * a0 + m[:, 1, 1] * a1
    else:
        out0 = M[0, 0] * a0 + M[0, 1] * a1; out1 = M[1, 0] * a0 + M[1, 1] * a1
    return np.stack([out0, out1], 2).reshape(B, 2 ** n)


def _diag_phases(D, qubits, n):
    """Expand the diagonal of a k-qubit diagonal gate to all 2^n basis states (shape (..., 2^n))."""
    idx = np.arange(2 ** n); sub = np.zeros(2 ** n, int)
    for i, q in enumerate(qubits):
        sub |= ((idx >> q) & 1) << i
    return D[..., sub]


def apply_matrix(state, M, qubits, n):
    """Apply a 2^k x 2^k matrix (same for the batch, or shape (B, 2^k, 2^k)) to `qubits` (first qubit
    = least significant bit of M's index)."""
    B = state.shape[0]; k = len(qubits)
    if k == 1:
        return _apply_1q(state, M, qubits[0], n)
    if k == 2 and np.count_nonzero(M - np.einsum('...ii->...i', M)[..., None] * np.eye(4)) == 0:
        return state * _diag_phases(np.einsum('...ii->...i', M), qubits, n)
    psi = state.reshape((B,) + (2,) * n)
    ax = [1 + _axes(n, q) for q in reversed(qubits)]          # big-endian order of M's index
    psi = np.moveaxis(psi, ax, list(range(1, k + 1))).reshape(B, 2 ** k, -1)
    psi = (M @ psi) if M.ndim == 3 else np.einsum('ij,bjr->bir', M, psi)
    psi = psi.reshape((B,) + (2,) * n)
    psi = np.moveaxis(psi, list(range(1, k + 1)), ax)
    return psi.reshape(B, 2 ** n)


def _rot(name, th):
    c, s = np.cos(th / 2), np.sin(th / 2); B = len(th)
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
        M = np.zeros((B, 4, 4), complex); M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = 1; M[:, 3, 3] = np.exp(1j * th)
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
PARAMETRIC = ('rx', 'ry', 'rz', 'p', 'rzz', 'cp')
_GENERATOR = {'rx': _FIXED['x'], 'ry': _FIXED['y'], 'rz': _FIXED['z'], 'rzz': np.diag([1, -1, -1, 1]).astype(complex),
              'p': np.diag([0, 1]).astype(complex), 'cp': np.diag([0, 0, 0, 1]).astype(complex)}


def _dagger(M):
    return M.conj().swapaxes(-1, -2)


class Circuit:
    """Parameterised circuit. Build with gate methods, simulate with `state(w, X)`."""

    def __init__(self, n):
        if n > MAX_QUBITS:
            raise ValueError('%d qubits exceed the simulator limit of %d (memory grows as 2^n)' % (n, MAX_QUBITS))
        self.n = n; self.ops = []

    def _add(self, name, qubits, param=None):
        self.ops.append((name, tuple(qubits), param)); return self

    def h(self, q):

        """Hadamard on qubit q."""

        return self._add('h', [q])
    def x(self, q):
        """Pauli X on qubit q."""
        return self._add('x', [q])
    def y(self, q):
        """Pauli Y on qubit q."""
        return self._add('y', [q])
    def z(self, q):
        """Pauli Z on qubit q."""
        return self._add('z', [q])
    def s(self, q):
        """S (phase pi/2) on qubit q."""
        return self._add('s', [q])
    def sdg(self, q):
        """S dagger on qubit q."""
        return self._add('sdg', [q])
    def cx(self, c, t):
        """CNOT with control c and target t."""
        return self._add('cx', [c, t])
    def cz(self, a, b):
        """Controlled Z on qubits a and b."""
        return self._add('cz', [a, b])
    def swap(self, a, b):
        """Swap qubits a and b."""
        return self._add('swap', [a, b])
    def rx(self, p, q):
        """RX(p) on qubit q; p is a number, W(k), X(j) or XX(i, j)."""
        return self._add('rx', [q], p)
    def ry(self, p, q):
        """RY(p) on qubit q."""
        return self._add('ry', [q], p)
    def rz(self, p, q):
        """RZ(p) on qubit q."""
        return self._add('rz', [q], p)
    def p(self, p, q):
        """Phase gate P(p) on qubit q."""
        return self._add('p', [q], p)
    def cp(self, p, c, t):
        """Controlled phase diag(1, 1, 1, e^{i p}) on qubits c and t (symmetric)."""
        return self._add('cp', [c, t], p)

    def rzz(self, p, a, b):
        """RZZ(p) = exp(-i p Z_a Z_b / 2) on qubits a and b."""
        return self._add('rzz', [a, b], p)

    def unitary(self, M, qubits, label='U'):
        """Fixed unitary on `qubits`."""
        self.ops.append(('unitary', tuple(qubits), (np.asarray(M, complex), label))); return self

    def diagonal(self, phases, label='D'):
        """Diagonal unitary diag(exp(i phases)) on all qubits (e.g. a cost or oracle layer)."""
        self.ops.append(('diag', tuple(range(self.n)), (np.asarray(phases, float), label))); return self

    @property
    def n_weights(self):
        """Number of trainable weights (largest W index + 1)."""
        ks = [p.k for _, _, p in self.ops if isinstance(p, W)]
        return 1 + max(ks) if ks else 0

    def compose(self, other):
        """Append the gates of another circuit; returns self."""
        self.ops += other.ops; return self

    # ----------------------------------------------------------------------------------- simulation
    def state(self, w=None, X_=None, init=None):
        """Final states, shape (B, 2^n). X_: (B, n_features) or None; w: (K,) or (B, K)."""
        w = np.zeros(self.n_weights) if w is None else np.asarray(w, float)
        Xa = None if X_ is None else np.atleast_2d(np.asarray(X_, float))
        B = Xa.shape[0] if Xa is not None else (w.shape[0] if w.ndim == 2 else 1)
        if w.ndim == 1 and len(w) < self.n_weights:
            raise ValueError('expected %d weights, got %d' % (self.n_weights, len(w)))
        if init is None:
            psi = np.zeros((B, 2 ** self.n), complex); psi[:, 0] = 1
        else:
            psi = np.tile(np.asarray(init, complex), (B, 1))
        for op in self.ops:
            psi = self._apply(op, psi, w, Xa, B)
        return psi

    def _matrix(self, op, w, Xa, B):
        name, qs, p = op
        if name in PARAMETRIC:
            return _rot(name, _value(p, w, Xa, B))
        if name == 'unitary':
            return p[0]
        return _FIXED[name]

    def _apply(self, op, psi, w, Xa, B, inverse=False):
        name, qs, p = op
        if name == 'diag':
            return psi * np.exp((-1j if inverse else 1j) * p[0])[None, :]
        M = self._matrix(op, w, Xa, B)
        return apply_matrix(psi, _dagger(M) if inverse else M, list(qs), self.n)

    def value_and_grad(self, w, X_, loss, chunk=None):
        """Loss and its exact gradient with respect to the trainable weights by the adjoint method.

        loss(psi, rows) -> (L, phi): L is the loss summed over the samples `rows` of the chunk and
        phi = dL/d(conj psi) has the shape of psi. The circuit is run forward once, then backwards
        once with the inverse gates, so the cost is about three simulations whatever the number of
        weights; samples are processed in chunks (memory O(chunk x 2^n)). Weights shared by several
        gates (e.g. QAOA angles) are handled."""
        w = np.asarray(w, float); Xa = None if X_ is None else np.atleast_2d(np.asarray(X_, float))
        B = 1 if Xa is None else Xa.shape[0]
        chunk = chunk or max(1, _CHUNK_AMPLITUDES // 2 ** self.n)
        total, grad = 0.0, np.zeros(self.n_weights)
        for start in range(0, B, chunk):
            rows = np.arange(start, min(B, start + chunk))
            Xc = None if Xa is None else Xa[rows]; b = len(rows)
            psi = self.state(w, Xc)
            L, phi = loss(psi, rows)
            total += float(L)
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
        """Born-rule probabilities of the final states, shape (B, 2^n)."""
        return np.abs(self.state(w, X_)) ** 2

    # ----------------------------------------------------------------------------------- export
    def to_qiskit(self, w=None, x=None, measure=True):
        """Qiskit QuantumCircuit with all angles bound (x: one sample)."""
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
    """Born-rule probabilities |psi|^2."""
    return np.abs(psi) ** 2


def expectation_z(psi, q, n):
    """<Z_q> for each state in the batch."""
    idx = np.arange(2 ** n); sign = 1 - 2 * ((idx >> q) & 1)
    return (np.abs(psi) ** 2) @ sign
