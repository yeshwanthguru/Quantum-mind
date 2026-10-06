"""Qiskit circuits for the quantum-like model families.

Every builder returns ``(circuit, decode)``, where ``decode(counts)`` turns measurement counts into the
same probability vectors that the analytic model predicts, so circuit and model can be compared
directly.

**Forms.** ``'dynamic'`` circuits use mid-circuit measurement and reset (IBM hardware, Aer).
``'deferred'`` circuits copy each measurement to a fresh ancilla and read everything at the end (the
deferred-measurement principle); they run on hardware without mid-circuit measurement (for example ion
traps on Amazon Braket) and on OpenQASM 3 simulators.

**Binary projective measurement.** To ask "is the state in the subspace of projector P (rank r)?", a
unitary V maps that subspace onto the first r computational basis states, an ancilla is flipped for
each of those basis states (multi-controlled X), only the ancilla is measured, and V is undone. The
system is never measured directly, so the post-measurement state follows the Lüders rule.

Examples
--------
>>> from qlcog.families.order_effects import QuantumOrderModel
>>> from qlcog.circuits import order_effects_circuit, run
>>> model = QuantumOrderModel(a=0.8, b=1.2, g=0.3)
>>> qc, decode = order_effects_circuit(model, 'AB')            # doctest: +SKIP
>>> decode(run(qc, 'aer', shots=20000)).round(2)               # doctest: +SKIP
"""
from __future__ import annotations

import itertools
import numpy as np
from .._optional import require
require('qiskit')
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate, StatePreparation, MCXGate

__all__ = ['subspace_unitary', 'projective_sequence_circuit', 'order_effects_circuit', 'conjunction_circuit',
           'similarity_circuit', 'interference_circuit', 'qlbn_circuit', 'walk_circuit', 'belief_circuit',
           'chsh_circuit', 'n_qubits']


def n_qubits(d):
    r"""Number of qubits needed for a dimension.

    Parameters
    ----------
    d : int
        Hilbert-space dimension.

    Returns
    -------
    int
        :math:`\lceil \log_2 d \rceil`, at least 1.
    """
    return max(1, int(np.ceil(np.log2(d))))


def _pad(v, n):
    """Pad a state vector with zeros to length 2^n."""
    out = np.zeros(2 ** n, complex); out[:len(v)] = v; return out


def subspace_unitary(P, n):
    """Unitary that maps the range of a projector onto the first basis states.

    Parameters
    ----------
    P : numpy.ndarray
        ``d x d`` projector with ``d <= 2**n``.
    n : int
        Number of qubits; padding dimensions are completed to a full unitary.

    Returns
    -------
    V : numpy.ndarray
        ``2**n x 2**n`` unitary whose first r rows are an orthonormal basis (conjugated) of range(P), so
        it maps range(P) onto span(``|0>``, ..., ``|r-1>``).
    r : int
        Rank of P.
    """
    d = P.shape[0]; D = 2 ** n
    w, V = np.linalg.eigh((P + P.conj().T) / 2)
    B = V[:, w > 0.5]                        # columns: basis of the subspace (d x r)
    r = B.shape[1]
    M = np.zeros((D, D), complex); M[:d, :r] = B
    rng = np.random.default_rng(12345)
    rest = rng.normal(size=(D, D - r)) + 1j * rng.normal(size=(D, D - r))
    Q, _ = np.linalg.qr(np.hstack([M[:, :r], rest]))
    Q[:, :r] = M[:, :r]                      # keep the exact basis (QR may change phases)
    # re-orthonormalise the completion against the exact basis
    for k in range(r, D):
        v = Q[:, k] - Q[:, :k] @ (Q[:, :k].conj().T @ Q[:, k]); Q[:, k] = v / np.linalg.norm(v)
    return Q.conj().T, r


def _flag(qc, sysq, anc, r):
    """Flip the ancilla for each of the first r basis states of the system register."""
    for k in range(r):
        qc.append(MCXGate(len(sysq), ctrl_state=k), list(sysq) + [anc])


def projective_sequence_circuit(psi, projectors, order, form='dynamic'):
    """Circuit for a sequence of yes/no questions.

    Parameters
    ----------
    psi : array_like
        State of length d (normalised here).
    projectors : dict
        ``{name: d x d "yes" projector}``.
    order : sequence of str
        Questions in the order asked.
    form : {'dynamic', 'deferred'}, optional
        Mid-circuit measurement, or deferred measurement on fresh ancillas.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> ``{answer tuple: probability}`` with 1 = yes, in the order asked.
    """
    psi = np.asarray(psi, complex); psi = psi / np.linalg.norm(psi)
    n = n_qubits(len(psi))
    s = QuantumRegister(n, 'sys'); m = len(order)
    a = QuantumRegister(1 if form == 'dynamic' else m, 'anc'); c = ClassicalRegister(m, 'ans')
    qc = QuantumCircuit(s, a, c)
    qc.append(StatePreparation(_pad(psi, n)), s)
    for i, q in enumerate(order):
        V, r = subspace_unitary(projectors[q], n)
        anc = a[0] if form == 'dynamic' else a[i]
        qc.append(UnitaryGate(V, label='V_%s' % q), s)
        _flag(qc, s, anc, r)
        qc.append(UnitaryGate(V.conj().T, label='V_%s^+' % q), s)
        if form == 'dynamic':
            qc.measure(anc, c[i]); qc.reset(anc)
    if form == 'deferred':
        for i in range(m): qc.measure(a[i], c[i])

    def decode(counts):
        tot = sum(counts.values()); out = {k: 0.0 for k in itertools.product((1, 0), repeat=m)}
        for bits, k in counts.items():
            b = bits.replace(' ', '')[::-1]        # Qiskit prints the highest classical bit first
            out[tuple(int(b[i]) for i in range(m))] += k / tot
        return out
    return qc, decode


def _order_cells(probs):
    """Answer-sequence probabilities as [yy, yn, ny, nn]."""
    return np.array([probs[(1, 1)], probs[(1, 0)], probs[(0, 1)], probs[(0, 0)]])


def order_effects_circuit(model, order='AB', form='dynamic'):
    """Circuit for a question-order model.

    Parameters
    ----------
    model : QuantumOrderModel, QuantumOrderModel4D or ProjectiveQuestionModel
        The model.
    order : {'AB', 'BA'}, optional
        Question order.
    form : {'dynamic', 'deferred'}, optional
        Circuit form.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> ``[yy, yn, ny, nn]``.
    """
    pm = model.as_projective() if hasattr(model, 'as_projective') else model
    qc, dec = projective_sequence_circuit(pm.psi, pm.projectors, tuple(order), form)
    return qc, (lambda counts: _order_cells(dec(counts)))


def conjunction_circuit(model, form='dynamic'):
    """Circuit for a :class:`~qlcog.families.conjunction.QuantumConjunctionModel`.

    The two events are asked in the model's order (more likely first).

    Parameters
    ----------
    model : QuantumConjunctionModel
    form : {'dynamic', 'deferred'}, optional

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> ``{'A&B': ..., 'A|B': ...}``.
    """
    psi = np.array([1.0, 0, 0]); P = model.projectors(); j = model.judgements()
    order = ('A', 'B') if j['A'] >= j['B'] else ('B', 'A')
    qc, dec = projective_sequence_circuit(psi, P, order, form)
    return qc, (lambda counts: (lambda p: {'A&B': p[(1, 1)], 'A|B': 1 - p[(0, 0)]})(dec(counts)))


def similarity_circuit(model, a, b, form='dynamic'):
    """Circuit for Sim(a, b) of a :class:`~qlcog.families.similarity.QuantumSimilarityModel`.

    Parameters
    ----------
    model : QuantumSimilarityModel
    a, b : str
        Concepts.
    form : {'dynamic', 'deferred'}, optional

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> the similarity.
    """
    qc, dec = projective_sequence_circuit(np.array([1.0, 0, 0]), model.projectors(), (a, b), form)
    return qc, (lambda counts: dec(counts)[(1, 1)])


def interference_circuit(p1, p2, c, theta, condition='unknown'):
    """Two-path interference circuit (normalised form of the interference model).

    Qubit 0 is the path (event) and qubit 1 the action (``|0>`` = act). Known conditions prepare one path;
    ``'unknown'`` superposes both paths with relative phase theta, erases the path with a Hadamard gate
    and post-selects path = 0.

    Parameters
    ----------
    p1, p2 : float
        P(act) on each path.
    c : float
        Weight of path 1.
    theta : float
        Relative phase.
    condition : {'known_1', 'known_2', 'unknown'}, optional

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> ``[P(act), P(not act)]``.
    """
    qc = QuantumCircuit(2, 2)
    if condition == 'known_1':
        pass
    elif condition == 'known_2':
        qc.x(0)
    else:
        qc.ry(2 * np.arccos(np.sqrt(c)), 0); qc.p(theta, 0)
    t1, t2 = 2 * np.arccos(np.sqrt(p1)), 2 * np.arccos(np.sqrt(p2))
    qc.x(0); qc.cry(t1, 0, 1); qc.x(0); qc.cry(t2, 0, 1)
    if condition == 'unknown':
        qc.h(0)
    qc.measure([0, 1], [0, 1])

    def decode(counts):
        act = nact = 0
        for bits, k in counts.items():
            b = bits.replace(' ', '')                 # bit string: action (clbit 1), path (clbit 0)
            path, action = int(b[-1]), int(b[-2])
            if condition == 'unknown' and path != 0:
                continue
            if action == 0: act += k
            else: nact += k
        tot = act + nact
        return np.array([act / tot, nact / tot]) if tot else np.array([np.nan, np.nan])
    return qc, decode


def qlbn_circuit(net, query, evidence=None, phases=None):
    r"""Quantum-like Bayesian network inference as a circuit.

    The amplitudes :math:`\sqrt{P(v, e, h)}\,e^{i\theta_h}` are loaded with ``StatePreparation``; the
    hidden register goes through Hadamard gates and is post-selected on 0, which sums the amplitudes
    over the hidden configurations.

    Parameters
    ----------
    net : BayesNet
    query : str
    evidence : dict, optional
    phases : array_like, optional

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> distribution of the query.
    """
    from ..families.qlbn.models import _hidden
    evidence = evidence or {}
    vals = net.variables[query]; hid = _hidden(net, query, evidence)
    configs = list(net.configurations(hid)); H = len(configs)
    phases = np.zeros(H) if phases is None else np.asarray(phases, float)
    nq, nh = n_qubits(len(vals)), (n_qubits(H) if H > 1 else 0)
    amps = np.zeros(2 ** (nq + nh), complex)
    for iv, v in enumerate(vals):
        for ih, (h, th) in enumerate(zip(configs, phases)):
            amps[iv + (ih << nq)] = np.sqrt(net.joint({**evidence, **h, query: v})) * np.exp(1j * th)
    amps /= np.linalg.norm(amps)
    qc = QuantumCircuit(nq + nh, nq + nh)
    qc.append(StatePreparation(amps), range(nq + nh))
    for k in range(nh): qc.h(nq + k)
    qc.measure(range(nq + nh), range(nq + nh))

    def decode(counts):
        acc = np.zeros(len(vals))
        for bits, k in counts.items():
            x = int(bits.replace(' ', ''), 2)
            if x >> nq == 0 and (x & (2 ** nq - 1)) < len(vals):
                acc[x & (2 ** nq - 1)] += k
        return acc / acc.sum() if acc.sum() else acc
    return qc, decode


def walk_circuit(model, t):
    """Quantum-walk circuit at time t.

    State preparation, :math:`U(t) = e^{-iHt}` as a unitary gate on the padded space, and measurement of
    every qubit.

    Parameters
    ----------
    model : QuantumWalk
    t : float

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> distribution over the model's response categories.
    """
    from scipy.linalg import expm
    N = model.N; n = n_qubits(N); D = 2 ** n
    Hp = np.zeros((D, D), complex); Hp[:N, :N] = model.H
    qc = QuantumCircuit(n, n)
    qc.append(StatePreparation(_pad(model.psi0 / np.linalg.norm(model.psi0), n)), range(n))
    qc.append(UnitaryGate(expm(-1j * Hp * t), label='U(t)'), range(n))
    qc.measure(range(n), range(n))

    def decode(counts):
        p = np.zeros(N); tot = sum(counts.values())
        for bits, k in counts.items():
            x = int(bits.replace(' ', ''), 2)
            if x < N: p[x] += k / tot
        return np.array([p[b].sum() for b in model.bins])
    return qc, decode


def belief_circuit(model, events, queries, form='dynamic'):
    """Circuit for an :class:`~qlcog.families.dynamics.OpenSystemBelief` over a sequence of events.

    Dephasing of strength gamma is applied by an ancilla that triggers Z with probability gamma / 2.

    Parameters
    ----------
    model : OpenSystemBelief
    events : sequence of int
    queries : sequence of int
    form : {'dynamic', 'deferred'}, optional

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> P(yes) at the last query.
    """
    queries = list(queries); nq = len(queries)
    s = QuantumRegister(1, 'sys')
    if form == 'dynamic':
        anc = QuantumRegister(1, 'deph'); qa = None
    else:
        anc = QuantumRegister(len(events), 'deph'); qa = QuantumRegister(nq, 'query')
    c = ClassicalRegister(nq, 'ans')
    qc = QuantumCircuit(s, anc, c) if qa is None else QuantumCircuit(s, anc, qa, c)
    qc.ry(model.phi0, s[0])
    theta = 2 * np.arcsin(np.sqrt(model.gamma / 2)); k = 0
    for t, e in enumerate(events):
        qc.ry(-model.a_pos if e else model.a_neg, s[0])
        a = anc[0] if form == 'dynamic' else anc[t]
        if model.gamma > 0:
            qc.ry(theta, a); qc.cz(a, s[0])
            if form == 'dynamic': qc.reset(a)
        if t in queries:
            if form == 'dynamic': qc.measure(s[0], c[k])
            else: qc.cx(s[0], qa[k])
            k += 1
    if form == 'deferred':
        for j in range(nq): qc.measure(qa[j], c[j])

    def decode(counts):
        tot = sum(counts.values())
        return sum(v for b, v in counts.items() if b.replace(' ', '')[0] == '0') / tot   # |0> = yes
    return qc, decode


def chsh_circuit(a, b):
    """Bell pair measured in the x-z plane.

    Parameters
    ----------
    a : float
        Measurement angle of qubit 0.
    b : float
        Measurement angle of qubit 1.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    decode : callable
        ``decode(counts)`` -> the correlation E(a, b).
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.ry(-a, 0); qc.ry(-b, 1); qc.measure([0, 1], [0, 1])

    def decode(counts):
        tot = sum(counts.values())
        return sum((1 if b_.count('1') % 2 == 0 else -1) * k for b_, k in counts.items()) / tot
    return qc, decode
