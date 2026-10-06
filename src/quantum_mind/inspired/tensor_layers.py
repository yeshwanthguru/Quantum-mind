r"""Tensor-train (matrix product operator) compression of neural-network layers.

A dense weight matrix :math:`W \in \mathbb{R}^{M \times N}` with :math:`M = \prod_k m_k` and
:math:`N = \prod_k n_k` can be reshaped into a tensor with one (output, input) index pair per core and
factorised by a sequence of truncated SVDs (TT-SVD, Oseledets 2011) into a *tensor train*

.. math:: W[(i_1..i_d),(j_1..j_d)] = G_1[i_1, j_1]\,G_2[i_2, j_2] \cdots G_d[i_d, j_d],

where each :math:`G_k[i_k, j_k]` is an :math:`r_{k-1} \times r_k` matrix. The bond dimensions (TT
ranks) :math:`r_k` control the trade-off: small ranks give large compression with some loss of
accuracy (Novikov et al., NeurIPS 2015, "Tensorizing neural networks"). This is the same matrix
product structure used for quantum many-body states, hence "quantum-inspired"; it runs on any CPU and
is aimed at fitting vision or policy networks on small onboard computers.

The saving is in memory, not necessarily in time. On a desktop CPU a 64 x 256 layer at rank 4 is stored
in about 290 times fewer numbers, but :meth:`TTMatrix.matvec` took about twice as long as the dense
product (0.75 ms against 0.38 ms for a batch of 256), because NumPy's dense matrix product is highly
optimised and the TT contraction is a chain of small products. Speed-ups appear only for large layers
at low rank; measure on the target hardware.

* :class:`TTMatrix`: a matrix in tensor-train format, with :meth:`~TTMatrix.from_dense` (TT-SVD),
  :meth:`~TTMatrix.matvec` (applied to a batch without forming the dense matrix), :meth:`~TTMatrix.to_dense`
  and :attr:`~TTMatrix.n_params`.
* :func:`factorise`: balanced factorisation of a layer size into ``d`` modes.
* :func:`compress_layers`: compress every weight matrix of a multilayer perceptron given as a list of
  ``(W, b)`` pairs.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.inspired.tensor_layers import TTMatrix
>>> rng = np.random.default_rng(0)
>>> W = rng.normal(size=(64, 256))
>>> tt = TTMatrix.from_dense(W, out_modes=(4, 4, 4), in_modes=(8, 8, 4), max_rank=64)
>>> bool(np.allclose(tt.to_dense(), W))                  # full rank: exact
True
>>> small = TTMatrix.from_dense(W, (4, 4, 4), (8, 8, 4), max_rank=4)
>>> small.n_params < W.size / 10
True
"""
from __future__ import annotations

import numpy as np

__all__ = ['TTMatrix', 'factorise', 'compress_layers', 'apply_mlp']


def factorise(n, d):
    """Split an integer into ``d`` factors that are as equal as possible.

    Parameters
    ----------
    n : int
        Layer size.
    d : int
        Number of factors.

    Returns
    -------
    tuple of int
        Factors whose product is ``n`` (ones are used when ``n`` has too few prime factors).
    """
    primes, m, p = [], n, 2
    while m > 1 and p * p <= m:
        while m % p == 0:
            primes.append(p)
            m //= p
        p += 1
    if m > 1:
        primes.append(m)
    out = [1] * d
    for q in sorted(primes, reverse=True):       # greedy: give the next prime to the smallest factor
        out[int(np.argmin(out))] *= q
    return tuple(sorted(out, reverse=True))


class TTMatrix:
    """A matrix in tensor-train (MPO) format.

    Parameters
    ----------
    cores : list of numpy.ndarray
        Core k has shape ``(r_{k-1}, m_k, n_k, r_k)`` with ``r_0 = r_d = 1``.

    Attributes
    ----------
    out_modes, in_modes : tuple of int
        Factorisations of the output and input sizes.
    """

    def __init__(self, cores):
        self.cores = [np.asarray(c, float) for c in cores]
        self.out_modes = tuple(c.shape[1] for c in self.cores)
        self.in_modes = tuple(c.shape[2] for c in self.cores)

    @classmethod
    def from_dense(cls, W, out_modes, in_modes, max_rank=8, tol=0.0):
        """Factorise a dense matrix by TT-SVD.

        Parameters
        ----------
        W : array_like
            Matrix of shape ``(prod(out_modes), prod(in_modes))``.
        out_modes, in_modes : sequence of int
            Factorisations of the two sizes (same length d).
        max_rank : int, optional
            Largest bond dimension.
        tol : float, optional
            Singular values below ``tol`` times the largest are also discarded.

        Returns
        -------
        TTMatrix

        Raises
        ------
        ValueError
            If the modes do not match the matrix shape.
        """
        W = np.asarray(W, float)
        out_modes, in_modes = tuple(out_modes), tuple(in_modes)
        d = len(out_modes)
        if len(in_modes) != d or W.shape != (int(np.prod(out_modes)), int(np.prod(in_modes))):
            raise ValueError('modes %s x %s do not match the shape %s' % (out_modes, in_modes, W.shape))
        # reorder to (m1, n1, m2, n2, ..., md, nd)
        T = W.reshape(out_modes + in_modes)
        perm = [ax for k in range(d) for ax in (k, d + k)]
        T = T.transpose(perm)
        cores, r = [], 1
        for k in range(d - 1):
            mk, nk = out_modes[k], in_modes[k]
            T = T.reshape(r * mk * nk, -1)
            U, S, Vt = np.linalg.svd(T, full_matrices=False)
            keep = int(min(max_rank, np.sum(S > tol * S[0]) if tol > 0 else len(S)))
            keep = max(keep, 1)
            cores.append(U[:, :keep].reshape(r, mk, nk, keep))
            T = S[:keep, None] * Vt[:keep]
            r = keep
        cores.append(T.reshape(r, out_modes[-1], in_modes[-1], 1))
        return cls(cores)

    @property
    def shape(self):
        """tuple: ``(rows, columns)`` of the represented matrix."""
        return int(np.prod(self.out_modes)), int(np.prod(self.in_modes))

    @property
    def ranks(self):
        """list of int: the bond dimensions."""
        return [c.shape[-1] for c in self.cores[:-1]]

    @property
    def n_params(self):
        """int: number of stored numbers."""
        return int(sum(c.size for c in self.cores))

    def to_dense(self):
        """Reconstruct the dense matrix.

        Returns
        -------
        numpy.ndarray
        """
        T = self.cores[0]                                     # (1, m1, n1, r1)
        for c in self.cores[1:]:
            T = np.tensordot(T, c, axes=([-1], [0]))          # (..., r) x (r, m, n, r')
        T = T.reshape([s for c in self.cores for s in c.shape[1:3]])
        d = len(self.cores)
        perm = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
        return T.transpose(perm).reshape(self.shape)

    def matvec(self, X):
        """Multiply a batch of vectors without forming the dense matrix.

        Parameters
        ----------
        X : array_like
            Shape ``(batch, columns)`` (or one vector of length ``columns``).

        Returns
        -------
        numpy.ndarray
            ``X @ W.T``, shape ``(batch, rows)``.
        """
        X = np.atleast_2d(np.asarray(X, float))
        B = X.shape[0]
        # T holds (batch, produced output modes..., remaining input modes..., bond)
        T = X.reshape((B,) + self.in_modes)[..., None]        # bond r0 = 1
        for k, c in enumerate(self.cores):
            # contract input mode k (now axis 1 + k) and the bond (last axis) with the core
            T = np.tensordot(T, c, axes=([1 + k, T.ndim - 1], [2, 0]))   # -> (..., m_k, r_k)
            T = np.moveaxis(T, -2, 1 + k)                      # put the new output mode in place
        return T.reshape(B, -1)

    def relative_error(self, W):
        """Frobenius relative error against a dense matrix.

        Parameters
        ----------
        W : array_like

        Returns
        -------
        float
        """
        W = np.asarray(W, float)
        return float(np.linalg.norm(self.to_dense() - W) / np.linalg.norm(W))


def compress_layers(layers, max_rank=8, d=3, min_size=256):
    """Compress the weight matrices of a multilayer perceptron.

    Parameters
    ----------
    layers : sequence of tuple
        ``[(W, b), ...]`` with ``W`` of shape ``(out, in)``.
    max_rank : int, optional
        Largest bond dimension.
    d : int, optional
        Number of cores per matrix.
    min_size : int, optional
        Matrices with fewer entries are kept dense.

    Returns
    -------
    compressed : list
        ``[(TTMatrix or numpy.ndarray, b), ...]``.
    report : dict
        ``dense_params``, ``compressed_params``, ``ratio`` and per-layer ``errors``.
    """
    out, dense, comp, errs = [], 0, 0, []
    for W, b in layers:
        W = np.asarray(W, float)
        dense += W.size + np.size(b)
        if W.size < min_size:
            out.append((W, b))
            comp += W.size + np.size(b)
            errs.append(0.0)
            continue
        tt = TTMatrix.from_dense(W, factorise(W.shape[0], d), factorise(W.shape[1], d), max_rank)
        out.append((tt, b))
        comp += tt.n_params + np.size(b)
        errs.append(tt.relative_error(W))
    return out, {'dense_params': int(dense), 'compressed_params': int(comp), 'ratio': dense / comp, 'errors': errs}


def apply_mlp(layers, X, activation=np.tanh):
    """Forward pass of a (possibly compressed) multilayer perceptron.

    Parameters
    ----------
    layers : sequence of tuple
        ``[(W or TTMatrix, b), ...]``.
    X : array_like
        Inputs, shape ``(batch, in)``.
    activation : callable, optional
        Hidden-layer activation (the last layer is linear).

    Returns
    -------
    numpy.ndarray
        Outputs (logits).
    """
    h = np.atleast_2d(np.asarray(X, float))
    for i, (W, b) in enumerate(layers):
        h = (W.matvec(h) if isinstance(W, TTMatrix) else h @ np.asarray(W).T) + b
        if i < len(layers) - 1:
            h = activation(h)
    return h

