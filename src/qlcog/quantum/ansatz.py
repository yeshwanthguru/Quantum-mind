"""Feature maps, ansatz circuits and the parameter-shift gradient."""
from __future__ import annotations

import numpy as np
from .statevector import Circuit, W, X, XX

__all__ = ['angle_encoding', 'zz_feature_map', 'hardware_efficient', 'reuploading_classifier_circuit',
           'parameter_shift', 'shifted_weights']


def angle_encoding(n_features, n_qubits=None, scale=1.0, circ=None):
    """RY(scale * x_j) on qubit j (features cycle over qubits if there are more features)."""
    n_qubits = n_qubits or n_features; circ = circ or Circuit(n_qubits)
    for j in range(n_features):
        circ.ry(X(j, scale), j % n_qubits)
    return circ


def zz_feature_map(n_features, reps=2, circ=None):
    """Second-order Pauli-Z feature map of Havlicek et al. (Nature 567, 209-212, 2019):
    H, RZ(2 x_i) and RZZ(2 (pi - x_i)(pi - x_j)) on every pair, repeated."""
    circ = circ or Circuit(n_features)
    for _ in range(reps):
        for i in range(n_features):
            circ.h(i); circ.rz(X(i, 2.0), i)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                circ.rzz(XX(i, j, np.pi, 2.0), i, j)
    return circ


def hardware_efficient(n_qubits, layers=2, start=0, circ=None, entangle='ring', rotations=('ry', 'rz'),
                       entangler='cz'):
    """Layers of single-qubit rotations (one weight each) and two-qubit entanglers ('cz' or 'cx') on a
    'ring' or 'linear' chain. Weights are numbered from `start`; returns (circuit, next free weight index)."""
    circ = circ or Circuit(n_qubits); k = start
    for _ in range(layers):
        for q in range(n_qubits):
            for r in rotations:
                getattr(circ, r)(W(k), q); k += 1
        if n_qubits > 1:
            pairs = [(q, q + 1) for q in range(n_qubits - 1)]
            if entangle == "ring" and n_qubits > 2:
                pairs.append((n_qubits - 1, 0))
            for a, b in pairs:
                getattr(circ, entangler)(a, b)
    return circ, k


def reuploading_classifier_circuit(n_features, n_qubits, layers, reupload=True):
    """Data re-uploading classifier (Perez-Salinas et al., Quantum 4, 226, 2020): encoding and
    trainable layers alternate, so the model is a trainable Fourier series of the inputs."""
    circ = Circuit(n_qubits); k = 0
    for layer in range(layers):
        if reupload or layer == 0:
            angle_encoding(n_features, n_qubits, 1.0, circ)
        circ, k = hardware_efficient(n_qubits, 1, k, circ)
    for q in range(n_qubits):
        circ.ry(W(k), q); k += 1
    return circ


def parameter_shift(f, w, shift=np.pi / 2):
    """Exact gradient of f(w) (any array output) for circuits in which every weight sets one rotation
    gate exp(-i w G / 2) with G^2 = I:  df/dw_k = [f(w + s e_k) - f(w - s e_k)] / 2 with s = pi/2.
    Returns an array of shape (len(w),) + shape of f(w)."""
    w = np.asarray(w, float); grads = []
    for k in range(len(w)):
        e = np.zeros_like(w); e[k] = shift
        grads.append((np.asarray(f(w + e)) - np.asarray(f(w - e))) / 2)
    return np.array(grads)


def shifted_weights(w, shift=np.pi / 2):
    """The 2K weight vectors needed by the parameter-shift rule, stacked as (2K, K): rows 2k and 2k+1
    are w + s e_k and w - s e_k. Evaluate them in one batched simulation and combine with
    grad_k = [f(row 2k) - f(row 2k+1)] / 2."""
    w = np.asarray(w, float); K = len(w)
    E = np.repeat(np.eye(K), 2, axis=0) * np.tile([shift, -shift], K)[:, None]
    return w[None, :] + E
