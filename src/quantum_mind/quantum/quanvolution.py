r"""Quanvolutional filters: a fixed random quantum circuit applied to image patches.

Henderson et al. (Quantum Machine Intelligence, 2020) proposed a convolution-like layer in which every
:math:`k \times k` patch of an image is angle-encoded into :math:`k^2` qubits (:math:`R_y(\pi x)` per
pixel, with pixels scaled to [0, 1]), passed through a random circuit, and read out as the Pauli-Z
expectation of each qubit. The result is a feature map with :math:`k^2` channels that a classical model
(here any scikit-learn classifier) then uses. The circuit is not trained.

* :class:`QuanvolutionFilter`: the quantum filter (simulated; the whole batch of patches in one call).
* :class:`RandomConvFilter`: the classical baseline with the same shape, a random linear filter bank
  followed by :math:`\tanh`.

On the 8 x 8 digits data set (5-fold cross-validation, logistic regression on the 64 features) the
quanvolution features reached 0.75 to 0.81 accuracy over three random circuits, the random classical
filter 0.90 to 0.92, and the raw pixels 0.93. The quantum filter is therefore worse than both baselines
here. It is included because it is a common reference point in quantum image processing and its
circuits export to Qiskit, not because it is expected to help a robot's vision pipeline.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.quantum.quanvolution import QuanvolutionFilter
>>> f = QuanvolutionFilter(seed=0)
>>> f.transform(np.zeros((3, 8, 8))).shape
(3, 4, 4, 4)
"""
from __future__ import annotations

import numpy as np
from .statevector import Circuit, X, expectation_z

__all__ = ['QuanvolutionFilter', 'RandomConvFilter', 'extract_patches']

_GATES = ('rx', 'ry', 'rz')


def extract_patches(images, size=2, stride=2):
    """Cut images into square patches.

    Parameters
    ----------
    images : array_like
        Shape ``(n, height, width)``.
    size : int, optional
        Patch side.
    stride : int, optional
        Step between patches.

    Returns
    -------
    patches : numpy.ndarray
        Shape ``(n, rows, cols, size * size)``.
    """
    images = np.asarray(images, float)
    if images.ndim == 2:
        images = images[None]
    n, H, Wd = images.shape
    rows = (H - size) // stride + 1
    cols = (Wd - size) // stride + 1
    out = np.empty((n, rows, cols, size * size))
    for i in range(rows):
        for j in range(cols):
            out[:, i, j] = images[:, i * stride:i * stride + size, j * stride:j * stride + size].reshape(n, -1)
    return out


class QuanvolutionFilter:
    """Fixed random quantum circuit applied to every image patch.

    Parameters
    ----------
    size : int, optional
        Patch side; the circuit has ``size**2`` qubits.
    stride : int, optional
        Step between patches.
    layers : int, optional
        Random layers, each a rotation on every qubit with a random axis and angle followed by a ring of
        CNOTs.
    seed : int, optional
        Seed of the random circuit.

    Attributes
    ----------
    circuit : Circuit
        The encoding plus random circuit; feature ``j`` is pixel ``j`` of the patch.
    """

    def __init__(self, size=2, stride=2, layers=1, seed=0):
        self.size, self.stride = size, stride
        nq = size * size
        rng = np.random.default_rng(seed)
        c = Circuit(nq)
        for q in range(nq):
            c.ry(X(q, scale=np.pi), q)                 # pixel in [0, 1] -> angle in [0, pi]
        for _ in range(layers):
            for q in range(nq):
                getattr(c, _GATES[rng.integers(3)])(float(rng.uniform(0, 2 * np.pi)), q)
            if nq > 1:
                for q in range(nq):
                    c.cx(q, (q + 1) % nq)
        self.circuit = c
        self.n_qubits = nq

    def transform(self, images):
        """Apply the filter.

        Parameters
        ----------
        images : array_like
            Shape ``(n, height, width)`` with pixel values in [0, 1].

        Returns
        -------
        numpy.ndarray
            Shape ``(n, rows, cols, size**2)``: the Pauli-Z expectation of each qubit for each patch.
        """
        P = extract_patches(images, self.size, self.stride)
        n, r, cc, k = P.shape
        psi = self.circuit.state(None, P.reshape(-1, k))
        Z = np.stack([expectation_z(psi, q, self.n_qubits) for q in range(self.n_qubits)], 1)
        return Z.reshape(n, r, cc, self.n_qubits)

    def to_qiskit(self, patch, measure=True):
        """Qiskit circuit for one patch.

        Parameters
        ----------
        patch : array_like
            ``size**2`` pixel values in [0, 1].
        measure : bool, optional

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(None, np.asarray(patch, float).ravel(), measure)


class RandomConvFilter:
    """Classical baseline: random linear filters on the same patches, followed by :math:`\\tanh`.

    Parameters
    ----------
    size : int, optional
        Patch side.
    stride : int, optional
        Step between patches.
    channels : int, optional
        Number of filters (default ``size**2``, the same as the quantum filter).
    seed : int, optional
    """

    def __init__(self, size=2, stride=2, channels=None, seed=0):
        self.size, self.stride = size, stride
        k = size * size
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 1, (k, channels or k))
        self.bias = rng.uniform(-1, 1, channels or k)

    def transform(self, images):
        """Apply the filters.

        Parameters
        ----------
        images : array_like
            Shape ``(n, height, width)`` with pixel values in [0, 1].

        Returns
        -------
        numpy.ndarray
            Shape ``(n, rows, cols, channels)``.
        """
        return np.tanh(extract_patches(images, self.size, self.stride) @ self.weights + self.bias)
