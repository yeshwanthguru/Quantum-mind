r"""Linear-algebra building blocks of quantum-like models.

States are unit vectors (real or complex) or density matrices. A "yes" answer to a question is an
orthogonal projector :math:`P`, and answering applies the Lüders rule

.. math:: p(\text{yes}) = \lVert P\psi \rVert^2, \qquad \psi \mapsto P\psi / \lVert P\psi \rVert .

Nothing here assumes a particular domain: the model families, the robotics layer and the circuit
builders are all written on top of these functions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ['normalize', 'projector', 'complement', 'luders', 'sequence_probabilities', 'angles_to_unit',
           'givens_frame', 'density', 'dephase', 'lindblad_superoperator', 'evolve_density', 'unitary',
           'is_projector']


def normalize(v):
    """Scale a vector to unit length.

    Parameters
    ----------
    v : array_like
        Real or complex vector.

    Returns
    -------
    numpy.ndarray
        ``v / ||v||``, real if ``v`` is real and complex otherwise.

    Raises
    ------
    ValueError
        If ``v`` is the zero vector.
    """
    v = np.asarray(v, dtype=complex if np.iscomplexobj(v) else float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError('cannot normalise the zero vector')
    return v / n


def projector(basis):
    """Orthogonal projector onto the span of a set of vectors.

    The vectors are orthonormalised first (QR decomposition), so any spanning set may be passed.

    Parameters
    ----------
    basis : array_like
        A single vector, or a 2D array whose rows span the subspace.

    Returns
    -------
    numpy.ndarray
        Hermitian, idempotent matrix :math:`QQ^\\dagger` of shape ``(d, d)``.

    Examples
    --------
    >>> from qlcog.core import projector
    >>> projector([1, 1]).round(2)
    array([[0.5, 0.5],
           [0.5, 0.5]])
    """
    B = np.atleast_2d(np.asarray(basis))
    Q, _ = np.linalg.qr(B.T)
    rank = np.linalg.matrix_rank(B)
    Q = Q[:, :rank]                      # keep only the columns that span the subspace
    return Q @ Q.conj().T


def complement(P):
    """Projector onto the orthogonal complement, :math:`I - P`.

    Parameters
    ----------
    P : numpy.ndarray
        Orthogonal projector of shape ``(d, d)``.

    Returns
    -------
    numpy.ndarray
        The "no" projector belonging to the "yes" projector ``P``.
    """
    return np.eye(P.shape[0]) - P


def is_projector(P, tol=1e-9):
    """Test whether a matrix is an orthogonal projector.

    Parameters
    ----------
    P : numpy.ndarray
        Square matrix.
    tol : float, optional
        Absolute tolerance for :math:`P^2 = P` and :math:`P = P^\\dagger`.

    Returns
    -------
    bool
        True if ``P`` is Hermitian and idempotent within ``tol``.
    """
    return np.allclose(P @ P, P, atol=tol) and np.allclose(P, P.conj().T, atol=tol)


def luders(psi, P):
    """Apply the Lüders rule for one projective measurement outcome.

    Parameters
    ----------
    psi : numpy.ndarray
        Normalised state vector.
    P : numpy.ndarray
        Projector of the outcome.

    Returns
    -------
    probability : float
        :math:`\\lVert P\\psi \\rVert^2`.
    state : numpy.ndarray
        Post-measurement state :math:`P\\psi / \\lVert P\\psi \\rVert`; the unnormalised vector is
        returned when the probability is numerically zero.
    """
    w = P @ psi
    p = float(np.real(np.vdot(w, w)))
    return p, (w / np.sqrt(p) if p > 1e-15 else w)


def sequence_probabilities(psi, projectors, order):
    """Probabilities of every yes/no answer sequence to a sequence of questions.

    Each question is answered in turn with the Lüders rule, so the result depends on the order
    whenever the projectors do not commute (the source of question-order effects).

    Parameters
    ----------
    psi : numpy.ndarray
        Initial normalised state.
    projectors : dict
        Maps a question name to the projector of its "yes" answer.
    order : sequence of str
        Question names in the order they are asked.

    Returns
    -------
    dict
        ``{(a1, a2, ...): probability}`` with ``a = 1`` for yes and ``0`` for no. The values sum to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from qlcog.core import projector, sequence_probabilities
    >>> P = {'A': projector([1, 0]), 'B': projector([1, 1])}
    >>> p = sequence_probabilities(np.array([1.0, 0.0]), P, ['A', 'B'])
    >>> round(p[(1, 1)], 3)
    0.5
    """
    # Each branch keeps (probability so far, post-measurement state).
    out = {(): (1.0, np.asarray(psi))}
    for q in order:
        P = projectors[q]
        Q = complement(P)
        new = {}
        for seq, (p, v) in out.items():
            for ans, M in ((1, P), (0, Q)):
                pr, w = luders(v, M)
                new[seq + (ans,)] = (p * pr, w)
        out = new
    return {k: p for k, (p, _) in out.items()}


def angles_to_unit(angles):
    """Unit vector from hyperspherical angles.

    Parameters
    ----------
    angles : array_like
        ``d - 1`` angles :math:`\\theta_1, \\dots, \\theta_{d-1}`.

    Returns
    -------
    numpy.ndarray
        Real unit vector in :math:`\\mathbb{R}^d` with
        :math:`v_i = \\cos\\theta_i \\prod_{j<i} \\sin\\theta_j` (the last entry has no cosine).
    """
    angles = np.asarray(angles, float)
    d = len(angles) + 1
    v = np.ones(d)
    for i, a in enumerate(angles):
        v[i] *= np.cos(a)
        v[i + 1:] *= np.sin(a)
    return v


def givens_frame(angles, d):
    """Orthonormal frame built from Givens rotations.

    Parameters
    ----------
    angles : array_like
        ``d (d - 1) / 2`` rotation angles, one per coordinate plane ``(i, j)`` with ``i < j``.
    d : int
        Dimension.

    Returns
    -------
    numpy.ndarray
        Orthogonal ``(d, d)`` matrix whose columns form an orthonormal basis. Every rotation of
        :math:`\\mathbb{R}^d` with determinant 1 can be written this way.
    """
    R = np.eye(d)
    k = 0
    for i in range(d - 1):
        for j in range(i + 1, d):
            # rotation by angles[k] in the (i, j) plane
            G = np.eye(d)
            c, s = np.cos(angles[k]), np.sin(angles[k])
            G[i, i] = G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            R = R @ G
            k += 1
    return R


def density(psi):
    """Density matrix of a pure state.

    Parameters
    ----------
    psi : array_like
        Normalised state vector.

    Returns
    -------
    numpy.ndarray
        :math:`\\rho = |\\psi\\rangle\\langle\\psi|`.
    """
    psi = np.asarray(psi)
    return np.outer(psi, psi.conj())


def dephase(rho, strength, basis_projectors=None):
    """Partial dephasing channel.

    .. math:: \\rho \\mapsto (1 - s)\\,\\rho + s \\sum_k P_k \\rho P_k

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix.
    strength : float
        Dephasing strength ``s`` in [0, 1]; ``s = 1`` removes all coherences between the subspaces.
    basis_projectors : sequence of numpy.ndarray, optional
        Projectors :math:`P_k` that sum to the identity. Defaults to the computational basis.

    Returns
    -------
    numpy.ndarray
        The dephased density matrix.
    """
    if basis_projectors is None:
        D = np.diag(np.diag(rho))
    else:
        D = sum(P @ rho @ P for P in basis_projectors)
    return (1 - strength) * rho + strength * D


def unitary(H, t):
    """Time-evolution operator :math:`e^{-iHt}`.

    Parameters
    ----------
    H : array_like
        Hermitian Hamiltonian.
    t : float
        Time.

    Returns
    -------
    numpy.ndarray
        Unitary matrix.
    """
    return expm(-1j * np.asarray(H) * t)


def lindblad_superoperator(H, jumps, rates):
    """Lindblad generator as a matrix acting on the row-stacked density matrix.

    .. math:: \\dot\\rho = -i[H, \\rho] + \\sum_k \\gamma_k \\left(L_k \\rho L_k^\\dagger
              - \\tfrac12 \\{L_k^\\dagger L_k, \\rho\\}\\right)

    Parameters
    ----------
    H : array_like
        Hamiltonian, shape ``(n, n)``.
    jumps : sequence of array_like
        Jump operators :math:`L_k`.
    rates : sequence of float
        Rates :math:`\\gamma_k \\ge 0`, one per jump operator.

    Returns
    -------
    numpy.ndarray
        Superoperator of shape ``(n**2, n**2)``; use it with :func:`evolve_density`.
    """
    H = np.asarray(H, complex)
    n = H.shape[0]
    I = np.eye(n)
    # vec(A rho B) = (A kron B^T) vec(rho) for the row-stacked vec
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for Lk, g in zip(jumps, rates):
        Lk = np.asarray(Lk, complex)
        LdL = Lk.conj().T @ Lk
        L += g * (np.kron(Lk, Lk.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T))
    return L


def evolve_density(rho, superop, t):
    """Evolve a density matrix under a Lindblad generator.

    Parameters
    ----------
    rho : numpy.ndarray
        Initial density matrix, shape ``(n, n)``.
    superop : numpy.ndarray
        Generator from :func:`lindblad_superoperator`.
    t : float
        Evolution time.

    Returns
    -------
    numpy.ndarray
        :math:`\\rho(t)`, computed with a matrix exponential.
    """
    n = rho.shape[0]
    v = expm(superop * t) @ rho.reshape(-1)
    return v.reshape(n, n)
