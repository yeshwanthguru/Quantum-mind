r"""Concept combination: membership of an item in a conjunction of concepts ("is a guppy a pet fish?").

People often rate an item's membership in "A and B" above its membership in A or in B
(overextension, the guppy effect; Hampton, JEP:LMC 14, 12-32, 1988), which fuzzy-set and
probabilistic conjunction rules cannot produce.

**Design.** ``{item: (mu_A, mu_B)}``, the observed single-concept memberships. The prediction is the
membership in "A and B".

**Models.**

* :class:`FockSpaceConceptModel`: Aerts' two-sector Fock-space model (Aerts, J. Math. Psych. 53,
  314-348, 2009),

  .. math:: \mu(A \wedge B) = m^2 \mu_A\mu_B + (1 - m^2)\Bigl[\frac{\mu_A + \mu_B}{2}
            + \kappa\,\min\bigl(\sqrt{\mu_A\mu_B}, \sqrt{(1-\mu_A)(1-\mu_B)}\bigr)\Bigr],

  where the interference term :math:`\mathrm{Re}\langle A|M|B\rangle` is written as :math:`\kappa`
  (in [-1, 1], shared by all items) times its largest possible size. Two parameters.
* :class:`ProductConceptModel`: classical baseline, the independent product (no parameter).
* :class:`MinConceptModel`: fuzzy-set baseline, the minimum (no parameter).
* :class:`WeightedAverageModel`: a weighted average with one weight; overextension is possible only
  up to the larger single membership.
"""
from __future__ import annotations

import numpy as np
from ...core import Model, Param

__all__ = ['FockSpaceConceptModel', 'ProductConceptModel', 'MinConceptModel', 'WeightedAverageModel',
           'interference_bound', 'overextension']


def interference_bound(mA, mB):
    """Largest possible size of the interference term.

    Parameters
    ----------
    mA, mB : float or numpy.ndarray
        Memberships in A and in B.

    Returns
    -------
    float or numpy.ndarray
        :math:`\\min(\\sqrt{m_A m_B}, \\sqrt{(1 - m_A)(1 - m_B)})`, the largest
        :math:`|\\mathrm{Re}\\langle A|M|B\\rangle|` compatible with orthogonal concept states.
    """
    return np.minimum(np.sqrt(mA * mB), np.sqrt((1 - mA) * (1 - mB)))


class _ConceptBase(Model):
    """Shared prediction of the concept-combination models; subclasses define the rule."""
    LOSS = 'sse'

    def _rule(self, mA, mB):
        """Membership in "A and B" from the two single memberships."""
        raise NotImplementedError

    def predict(self, design):
        """Predicted membership in "A and B" for each item.

        Parameters
        ----------
        design : dict
            ``{item: (mu_A, mu_B)}``.

        Returns
        -------
        dict
            ``{item: [membership]}``, clipped to [0, 1].

        Raises
        ------
        ValueError
            If the design is empty.
        """
        if not design:
            raise ValueError('design must map each item to its single-concept memberships (mu_A, mu_B)')
        return {k: np.array([float(np.clip(self._rule(*map(float, v)), 0, 1))]) for k, v in design.items()}


class FockSpaceConceptModel(_ConceptBase):
    """Aerts' Fock-space model of concept combination.

    Parameters
    ----------
    m2 : float
        Weight of the first (logical) sector, in (0, 1).
    kappa : float
        Interference as a fraction of its largest possible size, in (-1, 1).
    """
    PARAMS = [Param('m2', 'prob', 0.5), Param('kappa', 'bounded', 0.0, -1.0, 1.0)]
    name = 'Fock-space (Aerts)'

    def _rule(self, mA, mB):
        # sector 1: product; sector 2: average plus interference
        return self.m2 * mA * mB + (1 - self.m2) * ((mA + mB) / 2 + self.kappa * interference_bound(mA, mB))


class ProductConceptModel(_ConceptBase):
    """Classical baseline: independent product :math:`\\mu_A \\mu_B` (no parameter)."""
    PARAMS = []
    name = 'Product (classical)'

    def _rule(self, mA, mB):
        return mA * mB


class MinConceptModel(_ConceptBase):
    """Fuzzy-set baseline: :math:`\\min(\\mu_A, \\mu_B)` (no parameter)."""
    PARAMS = []
    name = 'Minimum (fuzzy set)'

    def _rule(self, mA, mB):
        return min(mA, mB)


class WeightedAverageModel(_ConceptBase):
    """Baseline: :math:`w\\,\\mu_A + (1 - w)\\,\\mu_B`.

    Parameters
    ----------
    w : float
        Weight of concept A, in (0, 1).
    """
    PARAMS = [Param('w', 'prob', 0.5)]
    name = 'Weighted average'

    def _rule(self, mA, mB):
        return self.w * mA + (1 - self.w) * mB


def overextension(design, values):
    """Overextension of each item.

    Parameters
    ----------
    design : dict
        ``{item: (mu_A, mu_B)}``.
    values : dict
        ``{item: membership in "A and B"}``, data or predictions.

    Returns
    -------
    dict
        ``{item: membership in "A and B" - min(mu_A, mu_B)}``; positive values are overextension.
    """
    return {k: float(np.ravel(values[k])[0]) - min(design[k]) for k in design}
