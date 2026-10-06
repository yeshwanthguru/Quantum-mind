"""Feature maps, ansatz circuits and the parameter-shift gradient.

These functions build :class:`~quantum_mind.quantum.Circuit` objects: data enter through feature-dependent
angles (:class:`~quantum_mind.quantum.X`, :class:`~quantum_mind.quantum.XX`) and trainable weights through
:class:`~quantum_mind.quantum.W`.

Examples
--------
>>> from quantum_mind.quantum import reuploading_classifier_circuit
>>> reuploading_classifier_circuit(n_features=4, n_qubits=2, layers=3)
Circuit(2 qubits, 29 gates, 14 weights)
"""
from __future__ import annotations

import numpy as np
from .statevector import Circuit, W, X, XX

__all__ = ['angle_encoding', 'zz_feature_map', 'hardware_efficient', 'reuploading_classifier_circuit',
           'parameter_shift', 'shifted_weights']


def angle_encoding(n_features, n_qubits=None, scale=1.0, circ=None):
    """Angle encoding: RY(scale * x_j) on qubit j.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_qubits : int, optional
        Number of qubits (default ``n_features``); features cycle over qubits if there are more.
    scale : float, optional
        Angle per unit of feature value.
    circ : Circuit, optional
        Circuit to append to (a new one by default).

    Returns
    -------
    Circuit
    """
    n_qubits = n_qubits or n_features
    circ = circ or Circuit(n_qubits)
    for j in range(n_features):
        circ.ry(X(j, scale), j % n_qubits)
    return circ


def zz_feature_map(n_features, reps=2, circ=None):
    """Second-order Pauli-Z feature map (Havlíček et al., Nature 567, 209-212, 2019).

    Each repetition applies H and RZ(2 x_i) on every qubit, then RZZ(2 (pi - x_i)(pi - x_j)) on every
    pair of qubits.

    Parameters
    ----------
    n_features : int
        Number of features (one qubit each).
    reps : int, optional
        Number of repetitions.
    circ : Circuit, optional
        Circuit to append to.

    Returns
    -------
    Circuit
    """
    circ = circ or Circuit(n_features)
    for _ in range(reps):
        for i in range(n_features):
            circ.h(i)
            circ.rz(X(i, 2.0), i)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                circ.rzz(XX(i, j, np.pi, 2.0), i, j)
    return circ


def hardware_efficient(n_qubits, layers=2, start=0, circ=None, entangle='ring', rotations=('ry', 'rz'),
                       entangler='cz'):
    """Hardware-efficient ansatz.

    Each layer applies single-qubit rotations (one weight each) on every qubit, then two-qubit
    entanglers along a chain.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    layers : int, optional
        Number of layers.
    start : int, optional
        Index of the first weight.
    circ : Circuit, optional
        Circuit to append to.
    entangle : {'ring', 'linear'}, optional
        Chain topology.
    rotations : tuple of str, optional
        Rotation gates per qubit and layer.
    entangler : {'cz', 'cx'}, optional
        Two-qubit gate.

    Returns
    -------
    circuit : Circuit
    next_weight : int
        Next free weight index.
    """
    circ = circ or Circuit(n_qubits)
    k = start
    for _ in range(layers):
        for q in range(n_qubits):
            for r in rotations:
                getattr(circ, r)(W(k), q)
                k += 1
        if n_qubits > 1:
            pairs = [(q, q + 1) for q in range(n_qubits - 1)]
            if entangle == "ring" and n_qubits > 2:
                pairs.append((n_qubits - 1, 0))
            for a, b in pairs:
                getattr(circ, entangler)(a, b)
    return circ, k


def reuploading_classifier_circuit(n_features, n_qubits, layers, reupload=True):
    """Data re-uploading classifier circuit (Pérez-Salinas et al., Quantum 4, 226, 2020).

    Encoding and trainable layers alternate, so the model is a trainable Fourier series of the
    inputs. A final RY layer prepares the read-out.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_qubits : int
        Number of qubits.
    layers : int
        Number of encoding-plus-trainable layers.
    reupload : bool, optional
        Encode the data in every layer (True) or only in the first.

    Returns
    -------
    Circuit
    """
    circ = Circuit(n_qubits)
    k = 0
    for layer in range(layers):
        if reupload or layer == 0:
            angle_encoding(n_features, n_qubits, 1.0, circ)
        circ, k = hardware_efficient(n_qubits, 1, k, circ)
    for q in range(n_qubits):
        circ.ry(W(k), q)
        k += 1
    return circ


def parameter_shift(f, w, shift=np.pi / 2):
    """Exact gradient by the parameter-shift rule.

    Valid when every weight sets one rotation gate :math:`e^{-iwG/2}` with :math:`G^2 = I`:

    .. math:: \\frac{\\partial f}{\\partial w_k} = \\frac{f(w + s e_k) - f(w - s e_k)}{2}, \\quad s = \\pi/2.

    This is how gradients are measured on quantum hardware; the simulator itself uses the cheaper
    adjoint method (:meth:`Circuit.value_and_grad`).

    Parameters
    ----------
    f : callable
        Function of the weights (any array output).
    w : array_like
        Weights, shape ``(K,)``.
    shift : float, optional
        Shift ``s``.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,) + shape of f(w)``.
    """
    w = np.asarray(w, float)
    grads = []
    for k in range(len(w)):
        e = np.zeros_like(w)
        e[k] = shift
        grads.append((np.asarray(f(w + e)) - np.asarray(f(w - e))) / 2)
    return np.array(grads)


def shifted_weights(w, shift=np.pi / 2):
    """All weight vectors needed by the parameter-shift rule, for one batched simulation.

    Parameters
    ----------
    w : array_like
        Weights, shape ``(K,)``.
    shift : float, optional
        Shift.

    Returns
    -------
    numpy.ndarray
        Shape ``(2K, K)``: rows ``2k`` and ``2k + 1`` are ``w + s e_k`` and ``w - s e_k``. Combine with
        ``grad_k = [f(row 2k) - f(row 2k + 1)] / 2``.
    """
    w = np.asarray(w, float)
    K = len(w)
    E = np.repeat(np.eye(K), 2, axis=0) * np.tile([shift, -shift], K)[:, None]
    return w[None, :] + E
