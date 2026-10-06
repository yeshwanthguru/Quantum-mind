"""Concept combination: membership of an item in a conjunction of concepts ("is a guppy a pet fish?").

People often rate an item's membership in "A and B" above its membership in A or in B
(overextension, the guppy effect; Hampton, JEP:LMC 14, 12-32, 1988), which fuzzy-set and
probabilistic conjunction rules cannot produce.

Models (design: {item: (mu_A, mu_B)} with the observed single-concept memberships; prediction:
membership in "A and B")
  FockSpaceConceptModel  Aerts' two-sector Fock-space model (Aerts, J. Math. Psych. 53, 314-348, 2009):
                         mu(A and B) = m^2 mu_A mu_B + (1 - m^2) [ (mu_A + mu_B)/2 + kappa * bound ],
                         where the interference term Re<A|M|B> is written as kappa (in [-1, 1], shared
                         by all items) times its largest possible size
                         bound = min(sqrt(mu_A mu_B), sqrt((1 - mu_A)(1 - mu_B))). Two parameters.
  ProductConceptModel    classical baseline: independent product mu_A mu_B (no parameter).
  MinConceptModel        fuzzy-set baseline: min(mu_A, mu_B) (no parameter).
  WeightedAverageModel   baseline: w mu_A + (1 - w) mu_B, one weight (overextension is possible only up
                         to the larger single membership)."""
from __future__ import annotations

import numpy as np
from ...core import Model, Param

__all__ = ['FockSpaceConceptModel', 'ProductConceptModel', 'MinConceptModel', 'WeightedAverageModel',
           'interference_bound', 'overextension']


def interference_bound(mA, mB):
    """Largest |Re<A|M|B>| compatible with orthogonal concept states and memberships mA, mB."""
    return np.minimum(np.sqrt(mA * mB), np.sqrt((1 - mA) * (1 - mB)))


class _ConceptBase(Model):
    LOSS = 'sse'

    def _rule(self, mA, mB):
        raise NotImplementedError

    def predict(self, design):
        """Predicted membership in "A and B" for each item of the design {item: (mu_A, mu_B)}."""
        if not design:
            raise ValueError('design must map each item to its single-concept memberships (mu_A, mu_B)')
        return {k: np.array([float(np.clip(self._rule(*map(float, v)), 0, 1))]) for k, v in design.items()}


class FockSpaceConceptModel(_ConceptBase):
    PARAMS = [Param('m2', 'prob', 0.5), Param('kappa', 'bounded', 0.0, -1.0, 1.0)]
    name = 'Fock-space (Aerts)'

    def _rule(self, mA, mB):
        return self.m2 * mA * mB + (1 - self.m2) * ((mA + mB) / 2 + self.kappa * interference_bound(mA, mB))


class ProductConceptModel(_ConceptBase):
    PARAMS = []
    name = 'Product (classical)'

    def _rule(self, mA, mB):
        return mA * mB


class MinConceptModel(_ConceptBase):
    PARAMS = []
    name = 'Minimum (fuzzy set)'

    def _rule(self, mA, mB):
        return min(mA, mB)


class WeightedAverageModel(_ConceptBase):
    PARAMS = [Param('w', 'prob', 0.5)]
    name = 'Weighted average'

    def _rule(self, mA, mB):
        return self.w * mA + (1 - self.w) * mB


def overextension(design, values):
    """For each item: membership in "A and B" minus the smaller single membership (> 0 = overextension)."""
    return {k: float(np.ravel(values[k])[0]) - min(design[k]) for k in design}
