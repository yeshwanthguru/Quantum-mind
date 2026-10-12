"""The quantum belief slot as a Qiskit circuit (cross-check of :class:`~quantum_mind.embodied.belief.QuantumSlot`).

Each answer is a non-unitary measurement ``M = U diag(sqrt(L)) U^dagger`` with ``||M|| <= 1``. A circuit can only apply
unitaries, so ``M`` is embedded in the unitary Halmos dilation ``W = [[M, sqrt(I - M M^dagger)], [sqrt(I - M^dagger M), -M^dagger]]``
acting on the objects (two qubits hold three objects and a spare level) plus one ancilla qubit per answer; the answer is accepted
when every ancilla reads 0. This reproduces the NumPy slot exactly on a simulator; it adds no speed or accuracy, and it is not
run on hardware. It needs the optional ``qiskit`` package.
"""
from __future__ import annotations

import numpy as np

from .belief import _ctx_matrices, _floor


def _sqrtm_psd(a):
    w, v = np.linalg.eigh((a + a.conj().T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T


def qiskit_posterior(slot, g0, L, ctx, mask):
    """Posterior over the objects of one sequence from a Qiskit circuit; ``g0, L, ctx, mask`` as for ``slot.posterior`` with one row."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector

    U = _ctx_matrices(slot.x)
    Lf = _floor(L, mask)[0]
    n_ans = L.shape[1]
    psi0 = np.zeros(4, complex)
    psi0[:3] = np.sqrt(g0[0])
    qc = QuantumCircuit(2 + n_ans)
    qc.initialize(psi0, [0, 1])
    for j in range(n_ans):
        Uj = U[ctx[0, j]]
        M3 = Uj @ np.diag(np.sqrt(Lf[j])) @ Uj.conj().T
        M = np.eye(4, dtype=complex)
        M[:3, :3] = M3
        D1, D2 = _sqrtm_psd(np.eye(4) - M @ M.conj().T), _sqrtm_psd(np.eye(4) - M.conj().T @ M)
        W = np.block([[M, D1], [D2, -M.conj().T]])
        qc.append(UnitaryGate(W), [0, 1, 2 + j])
    amp = Statevector.from_instruction(qc).data
    sys_amp = amp[:4]                                             # all ancillas read 0: the highest qubits are 0
    p = np.abs(sys_amp[:3]) ** 2 * mask[0]
    return p / p.sum(), float(np.sum(np.abs(sys_amp) ** 2))
