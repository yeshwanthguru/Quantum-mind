"""Linear-algebra building blocks of quantum-like models.

States are unit vectors (real or complex) or density matrices; a 'yes' answer to a question is an
orthogonal projector; answering applies the Lueders rule. Nothing here assumes a particular domain."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ['normalize', 'projector', 'complement', 'luders', 'sequence_probabilities', 'angles_to_unit',
           'givens_frame', 'density', 'dephase', 'lindblad_superoperator', 'evolve_density', 'unitary',
           'is_projector']


def normalize(v):
    """Return v / ||v|| (raises on the zero vector)."""
    v = np.asarray(v, dtype=complex if np.iscomplexobj(v) else float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError('cannot normalise the zero vector')
    return v / n


def projector(basis):
    """Orthogonal projector onto the span of the given vectors (rows or a single vector).
    The vectors are orthonormalised first, so any spanning set may be passed."""
    B = np.atleast_2d(np.asarray(basis))
    Q, _ = np.linalg.qr(B.T)
    rank = np.linalg.matrix_rank(B)
    Q = Q[:, :rank]
    return Q @ Q.conj().T


def complement(P):
    """I - P."""
    return np.eye(P.shape[0]) - P


def is_projector(P, tol=1e-9):
    """True if P is Hermitian and idempotent within tol."""
    return np.allclose(P @ P, P, atol=tol) and np.allclose(P, P.conj().T, atol=tol)


def luders(psi, P):
    """Lueders update: returns (probability, post-measurement state) for projector P."""
    w = P @ psi
    p = float(np.real(np.vdot(w, w)))
    return p, (w / np.sqrt(p) if p > 1e-15 else w)


def sequence_probabilities(psi, projectors, order):
    """Probabilities of every yes/no answer sequence to the questions in `order`.

    projectors: dict name -> projector of the 'yes' answer.
    Returns dict {(a1, a2, ...): probability} with a = 1 for yes and 0 for no."""
    out = {(): (1.0, np.asarray(psi))}
    for q in order:
        P = projectors[q]; Q = complement(P)
        new = {}
        for seq, (p, v) in out.items():
            for ans, M in ((1, P), (0, Q)):
                pr, w = luders(v, M)
                new[seq + (ans,)] = (p * pr, w)
        out = new
    return {k: p for k, (p, _) in out.items()}


def angles_to_unit(angles):
    """Unit vector in R^(len(angles)+1) from hyperspherical angles."""
    angles = np.asarray(angles, float)
    d = len(angles) + 1
    v = np.ones(d)
    for i, a in enumerate(angles):
        v[i] *= np.cos(a)
        v[i + 1:] *= np.sin(a)
    return v


def givens_frame(angles, d):
    """Orthonormal frame (d x d orthogonal matrix) from d(d-1)/2 Givens rotation angles."""
    R = np.eye(d); k = 0
    for i in range(d - 1):
        for j in range(i + 1, d):
            G = np.eye(d); c, s = np.cos(angles[k]), np.sin(angles[k])
            G[i, i] = G[j, j] = c; G[i, j] = -s; G[j, i] = s
            R = R @ G; k += 1
    return R


def density(psi):
    """Density matrix |psi><psi| of a (normalised) state vector."""
    psi = np.asarray(psi)
    return np.outer(psi, psi.conj())


def dephase(rho, strength, basis_projectors=None):
    """Partial dephasing: rho -> (1 - s) rho + s * sum_k P_k rho P_k (in the computational basis
    if no projectors are given)."""
    if basis_projectors is None:
        D = np.diag(np.diag(rho))
    else:
        D = sum(P @ rho @ P for P in basis_projectors)
    return (1 - strength) * rho + strength * D


def unitary(H, t):
    """exp(-i H t)."""
    return expm(-1j * np.asarray(H) * t)


def lindblad_superoperator(H, jumps, rates):
    """Superoperator L (acting on row-stacked vec(rho)) of
    d rho/dt = -i[H, rho] + sum_k g_k (L_k rho L_k^+ - 1/2 {L_k^+ L_k, rho})."""
    H = np.asarray(H, complex); n = H.shape[0]; I = np.eye(n)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for Lk, g in zip(jumps, rates):
        Lk = np.asarray(Lk, complex); LdL = Lk.conj().T @ Lk
        L += g * (np.kron(Lk, Lk.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T))
    return L


def evolve_density(rho, superop, t):
    """Evolve a density matrix for time t under a Lindblad superoperator (matrix exponential)."""
    n = rho.shape[0]
    v = expm(superop * t) @ rho.reshape(-1)
    return v.reshape(n, n)
