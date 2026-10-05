"""Base class shared by every model in qlcog.

A model has named parameters with declared kinds (used to map them to an unconstrained vector for
fitting), and a `predict(design)` method returning, for every condition of the design, a probability
vector over the outcomes of that condition (or, for judgement models, a vector of predicted values).
Data are dictionaries with the same condition keys: outcome counts (multinomial models) or observed
values (judgement models, fitted by least squares)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit

__all__ = ['Param', 'Model']


@dataclass(frozen=True)
class Param:
    """A model parameter. kind: 'prob' (0..1), 'angle' (free real, periodic), 'real', 'positive',
    'bounded' (lo..hi) or 'fixed' (not fitted)."""
    name: str
    kind: str = 'real'
    default: float = 0.0
    lo: float = 0.0
    hi: float = 1.0

    def to_free(self, v):
        if self.kind == 'prob':
            return float(logit(np.clip(v, 1e-9, 1 - 1e-9)))
        if self.kind == 'positive':
            return float(np.log(max(v, 1e-12)))
        if self.kind == 'bounded':
            return float(logit(np.clip((v - self.lo) / (self.hi - self.lo), 1e-9, 1 - 1e-9)))
        return float(v)

    def from_free(self, x):
        if self.kind == 'prob':
            return float(expit(x))
        if self.kind == 'positive':
            return float(np.exp(np.clip(x, -50, 50)))
        if self.kind == 'bounded':
            return float(self.lo + (self.hi - self.lo) * expit(x))
        return float(x)

    def random_free(self, rng):
        if self.kind == 'angle':
            return rng.uniform(0, np.pi)
        return rng.normal(0, 1.0)


class Model:
    """Base class. Subclasses define PARAMS (list of Param), LOSS ('multinomial' or 'sse') and
    predict(design)."""
    PARAMS: list = []
    LOSS = 'multinomial'
    name = 'Model'

    def __init__(self, **params):
        for p in self.PARAMS:
            setattr(self, p.name, float(params.pop(p.name, p.default)))
        self.options = params        # structural (non-fitted) options, e.g. projector ranks

    # ------------------------------------------------------------------ parameters
    @classmethod
    def free_params(cls):
        return [p for p in cls.PARAMS if p.kind != 'fixed']

    @property
    def params(self):
        return {p.name: getattr(self, p.name) for p in self.PARAMS}

    def to_vector(self):
        return np.array([p.to_free(getattr(self, p.name)) for p in self.free_params()])

    @classmethod
    def from_vector(cls, x, **options):
        kw = {p.name: p.from_free(v) for p, v in zip(cls.free_params(), x)}
        kw.update(options)
        return cls(**kw)

    @property
    def n_params(self):
        return len(self.free_params())

    # ------------------------------------------------------------------ predictions and data
    def predict(self, design):
        raise NotImplementedError

    def loglik(self, data, design):
        """Multinomial log-likelihood of outcome counts (constant terms omitted)."""
        pred = self.predict(design)
        ll = 0.0
        for cond, counts in data.items():
            p = np.clip(np.asarray(pred[cond], float), 1e-12, 1.0)
            ll += float(np.dot(np.asarray(counts, float), np.log(p)))
        return ll

    def sse(self, data, design):
        pred = self.predict(design)
        return float(sum(np.sum((np.asarray(pred[c], float) - np.asarray(v, float)) ** 2) for c, v in data.items()))

    def sample(self, design, n, rng=None):
        """Simulated outcome counts: n observations per condition."""
        rng = np.random.default_rng() if rng is None else rng
        out = {}
        for cond, p in self.predict(design).items():
            p = np.clip(np.asarray(p, float), 0, None); p = p / p.sum()
            out[cond] = rng.multinomial(n, p)
        return out

    def __repr__(self):
        ps = ', '.join('%s=%.4g' % (k, v) for k, v in self.params.items())
        return '%s(%s)' % (type(self).__name__, ps)
