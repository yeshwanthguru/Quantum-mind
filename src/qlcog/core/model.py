"""Base class shared by every model in qlcog.

A model has named parameters with declared kinds, which map them to an unconstrained vector for
fitting, and a :meth:`Model.predict` method. For every condition of a *design*, ``predict`` returns a
probability vector over the outcomes of that condition, or, for judgement models, a vector of
predicted values.

Data are dictionaries with the same condition keys as the design: outcome counts for multinomial
models (fitted by maximum likelihood) or observed values for judgement models (fitted by least
squares).

Examples
--------
A minimal model with one probability parameter:

>>> import numpy as np
>>> from qlcog.core import Model, Param
>>> class Coin(Model):
...     PARAMS = [Param('p', 'prob', 0.5)]
...     def predict(self, design):
...         return {c: np.array([self.p, 1 - self.p]) for c in design}
>>> Coin(p=0.3).predict(['toss'])['toss']
array([0.3, 0.7])
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit

__all__ = ['Param', 'Model']


@dataclass(frozen=True)
class Param:
    """Declaration of one model parameter.

    Parameters
    ----------
    name : str
        Attribute name on the model.
    kind : {'prob', 'angle', 'real', 'positive', 'bounded', 'fixed'}
        Constraint of the parameter. ``'prob'`` lies in (0, 1) (logit transform), ``'angle'`` is a
        free periodic real, ``'real'`` is unconstrained, ``'positive'`` is > 0 (log transform),
        ``'bounded'`` lies in (``lo``, ``hi``) and ``'fixed'`` is never fitted.
    default : float
        Value used when the model is constructed without this parameter.
    lo, hi : float
        Bounds of a ``'bounded'`` parameter.
    """
    name: str
    kind: str = 'real'
    default: float = 0.0
    lo: float = 0.0
    hi: float = 1.0

    def to_free(self, v):
        """Map a parameter value to the unconstrained real line used by the optimiser.

        Parameters
        ----------
        v : float
            Parameter value.

        Returns
        -------
        float
            Unconstrained value (logit, log or identity, depending on ``kind``).
        """
        if self.kind == 'prob':
            return float(logit(np.clip(v, 1e-9, 1 - 1e-9)))
        if self.kind == 'positive':
            return float(np.log(max(v, 1e-12)))
        if self.kind == 'bounded':
            return float(logit(np.clip((v - self.lo) / (self.hi - self.lo), 1e-9, 1 - 1e-9)))
        return float(v)

    def from_free(self, x):
        """Inverse of :meth:`to_free`.

        Parameters
        ----------
        x : float
            Unconstrained value.

        Returns
        -------
        float
            Parameter value inside its constraint set.
        """
        if self.kind == 'prob':
            return float(expit(x))
        if self.kind == 'positive':
            return float(np.exp(np.clip(x, -50, 50)))    # clip to avoid overflow
        if self.kind == 'bounded':
            return float(self.lo + (self.hi - self.lo) * expit(x))
        return float(x)

    def random_free(self, rng):
        """Random unconstrained starting value for multi-start fitting.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Uniform on [0, pi) for angles, standard normal otherwise.
        """
        if self.kind == 'angle':
            return rng.uniform(0, np.pi)
        return rng.normal(0, 1.0)


class Model:
    """Base class of every quantum-like and classical model.

    Subclasses define three class attributes and one method:

    * ``PARAMS``: list of :class:`Param`;
    * ``LOSS``: ``'multinomial'`` (outcome counts) or ``'sse'`` (observed values);
    * ``name``: label used in comparison tables;
    * :meth:`predict`: ``design -> {condition: vector}``.

    Parameters
    ----------
    **params
        Parameter values by name; missing parameters take their default. Any other keyword is kept
        in :attr:`options` as a structural (non-fitted) option, for example a projector rank.

    Attributes
    ----------
    options : dict
        Structural options passed to the constructor.
    """
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
        """Parameters that are fitted.

        Returns
        -------
        list of Param
            Every parameter whose kind is not ``'fixed'``.
        """
        return [p for p in cls.PARAMS if p.kind != 'fixed']

    @property
    def params(self):
        """dict: current parameter values by name."""
        return {p.name: getattr(self, p.name) for p in self.PARAMS}

    def to_vector(self):
        """Unconstrained vector of the fitted parameters.

        Returns
        -------
        numpy.ndarray
            One entry per :meth:`free_params` element; the inverse of :meth:`from_vector`.
        """
        return np.array([p.to_free(getattr(self, p.name)) for p in self.free_params()])

    @classmethod
    def from_vector(cls, x, **options):
        """Build a model from an unconstrained parameter vector.

        Parameters
        ----------
        x : array_like
            Unconstrained values, in the order of :meth:`free_params`.
        **options
            Structural options passed to the constructor.

        Returns
        -------
        Model
            New instance of the class.
        """
        kw = {p.name: p.from_free(v) for p, v in zip(cls.free_params(), x)}
        kw.update(options)
        return cls(**kw)

    @property
    def n_params(self):
        """int: number of fitted parameters (used by BIC and AIC)."""
        return len(self.free_params())

    # ------------------------------------------------------------------ predictions and data
    def predict(self, design):
        """Predictions for every condition of a design.

        Parameters
        ----------
        design : iterable
            Condition keys (their meaning depends on the model family).

        Returns
        -------
        dict
            ``{condition: numpy.ndarray}``: a probability vector over outcomes, or predicted values.
        """
        raise NotImplementedError

    def loglik(self, data, design):
        """Multinomial log-likelihood of outcome counts.

        Parameters
        ----------
        data : dict
            ``{condition: counts}`` with the outcome order of :meth:`predict`.
        design : iterable
            Conditions passed to :meth:`predict`.

        Returns
        -------
        float
            :math:`\\sum_c \\sum_k n_{ck} \\log p_{ck}` (constant terms omitted; probabilities are
            clipped at 1e-12).
        """
        pred = self.predict(design)
        ll = 0.0
        for cond, counts in data.items():
            p = np.clip(np.asarray(pred[cond], float), 1e-12, 1.0)
            ll += float(np.dot(np.asarray(counts, float), np.log(p)))
        return ll

    def sse(self, data, design):
        """Sum of squared errors between predicted and observed values.

        Parameters
        ----------
        data : dict
            ``{condition: observed values}``.
        design : iterable
            Conditions passed to :meth:`predict`.

        Returns
        -------
        float
            Sum over conditions of the squared differences.
        """
        pred = self.predict(design)
        return float(sum(np.sum((np.asarray(pred[c], float) - np.asarray(v, float)) ** 2) for c, v in data.items()))

    def sample(self, design, n, rng=None):
        """Simulate outcome counts from the model.

        Parameters
        ----------
        design : iterable
            Conditions.
        n : int
            Number of observations per condition.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict
            ``{condition: counts}``, multinomial draws from the predicted probabilities.
        """
        rng = np.random.default_rng() if rng is None else rng
        out = {}
        for cond, p in self.predict(design).items():
            p = np.clip(np.asarray(p, float), 0, None)
            p = p / p.sum()                                  # guard against rounding error
            out[cond] = rng.multinomial(n, p)
        return out

    def __repr__(self):
        ps = ', '.join('%s=%.4g' % (k, v) for k, v in self.params.items())
        return '%s(%s)' % (type(self).__name__, ps)
