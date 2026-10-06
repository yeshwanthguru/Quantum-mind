r"""Episodic memory: overdistribution in recognition judgements.

People answer three questions about the same test item: "was it studied?" (V, verbatim), "is it
related to a studied item?" (G, gist) and "was it studied or related?" (V or G). The first two
probabilities often add up to more than the third (overdistribution), which classical probability
forbids when the two events are exclusive (Brainerd and Reyna's conjoint recognition).

**Models** (one test-probe type per condition, for example ``'target'``, ``'related'``,
``'unrelated'``).

* :class:`QuantumEpisodicModel` (after Brainerd, Wang and Reyna, Topics in Cognitive Science 5,
  773-799, 2013): a memory state :math:`\psi_k` for probe type k in a real 3D space; V and G are rank-1
  subspaces at angle c (shared); "V or G" is the plane they span (quantum disjunction), so
  :math:`P(V) + P(G) - P(V \vee G) > 0` unless V and G are orthogonal.
* :class:`AdditiveMemoryModel`: classical baseline with exclusive events,
  :math:`P(V \vee G) = P(V) + P(G)`.

**Design.** A tuple of ``(probe, question)`` with question in ``{'V', 'G', 'V|G'}``. Data are observed
proportions ``{(probe, question): value}``, fitted by least squares.
"""
from __future__ import annotations

import numpy as np
from ...core import Model, Param

__all__ = ['QuestionsVG', 'QuantumEpisodicModel', 'AdditiveMemoryModel', 'overdistribution']
#: The three recognition questions.
QuestionsVG = ('V', 'G', 'V|G')


def _design(probes):
    """Every (probe, question) pair."""
    return tuple((p, q) for p in probes for q in QuestionsVG)


class QuantumEpisodicModel(Model):
    """Quantum episodic memory model.

    Parameters
    ----------
    c : float
        Angle between the verbatim and gist directions, in (0, pi/2).
    a_<probe>, b_<probe> : float
        Two angles of the memory state for each probe type, in (0, pi/2).
    """
    LOSS = 'sse'
    name = 'Quantum episodic memory'
    PROBES = ('target', 'related', 'unrelated')
    PARAMS = [Param('c', 'bounded', 1.0, 0.0, np.pi / 2)] + [
        Param('%s_%s' % (k, p), 'bounded', d, 0.0, np.pi / 2) for p in PROBES for k, d in (('a', 0.8), ('b', 0.7))]

    def probe_probabilities(self, probe):
        """Answer probabilities for one probe type, from the projections of the memory state.

        Parameters
        ----------
        probe : str
            Probe type.

        Returns
        -------
        dict
            ``{'V': P(V), 'G': P(G), 'V|G': P(V or G)}``.
        """
        a, b = getattr(self, 'a_' + probe), getattr(self, 'b_' + probe)
        psi = np.array([np.cos(a), np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)])
        uV = np.array([1.0, 0, 0])
        uG = np.array([np.cos(self.c), np.sin(self.c), 0.0])
        pV, pG = (psi @ uV) ** 2, (psi @ uG) ** 2
        pVG = psi[0] ** 2 + psi[1] ** 2                       # projection onto span(uV, uG) = e1-e2 plane
        return {'V': pV, 'G': pG, 'V|G': pVG}

    def predict(self, design=None):
        """Predicted proportion for each ``(probe, question)``."""
        design = design or _design(self.PROBES)
        return {(p, q): np.array([self.probe_probabilities(p)[q]]) for p, q in design}


class AdditiveMemoryModel(Model):
    """Classical baseline: V and G are exclusive, so P(V or G) = P(V) + P(G).

    Parameters
    ----------
    v_<probe>, g_<probe> : float
        Logits of "verbatim" and "gist" against "neither" for each probe type.
    """
    LOSS = 'sse'
    name = 'Additive (classical)'
    PROBES = QuantumEpisodicModel.PROBES
    PARAMS = [Param('%s_%s' % (k, p), 'real', 0.0) for p in PROBES for k in ('v', 'g')]

    def probe_probabilities(self, probe):
        """Answer probabilities for one probe type (softmax over verbatim, gist, neither).

        Parameters
        ----------
        probe : str
            Probe type.

        Returns
        -------
        dict
            ``{'V': P(V), 'G': P(G), 'V|G': P(V) + P(G)}``.
        """
        z = np.array([getattr(self, 'v_' + probe), getattr(self, 'g_' + probe), 0.0])
        e = np.exp(z - z.max())
        pr = e / e.sum()                                       # softmax over V, G, neither: sums to 1
        return {'V': pr[0], 'G': pr[1], 'V|G': pr[0] + pr[1]}

    def predict(self, design=None):
        """Predicted proportion for each ``(probe, question)``."""
        design = design or _design(self.PROBES)
        return {(p, q): np.array([self.probe_probabilities(p)[q]]) for p, q in design}


def overdistribution(values, probe):
    """Overdistribution for one probe type.

    Parameters
    ----------
    values : dict
        ``{(probe, question): value}``, data or predictions.
    probe : str
        Probe type.

    Returns
    -------
    float
        :math:`P(V) + P(G) - P(V \\vee G)`; positive values are impossible for exclusive events.
    """
    g = lambda q: float(np.ravel(values[(probe, q)])[0])     # noqa: E731
    return g('V') + g('G') - g('V|G')
