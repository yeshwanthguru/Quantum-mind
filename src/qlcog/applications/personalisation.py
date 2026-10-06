r"""Personalised human models: start from the population, adapt to each person.

Fitting one model to everyone's answers assumes that people are alike; fitting each person separately
needs thousands of answers (see example 22). Partial pooling sits in between: the population's spread
of parameters becomes a prior, and each person's parameters are the maximum a posteriori (MAP) estimate

.. math:: \hat x_i = \arg\min_x\; -\log L_i(x) + \tfrac12 (x - \mu)^T \Sigma^{-1} (x - \mu),

in the unconstrained parameter space of the model (see :meth:`qlcog.core.Param.to_free`). With few
answers the estimate stays near the population; with many it follows the person. A robot can therefore
personalise its human model from the first few interactions without over-fitting.

* :class:`PopulationPrior`: mean and covariance of the free parameters, estimated from per-person fits
  (empirical Bayes) or set by hand.
* :func:`fit_map`: MAP fit of one model class to one person's data.
* :class:`PersonalisedHumanModel`: keeps each person's answer counts and their personalised model.

Examples
--------
>>> import numpy as np
>>> from qlcog.families.order_effects import BayesOrderModel
>>> from qlcog.applications.personalisation import PopulationPrior, fit_map
>>> prior = PopulationPrior(BayesOrderModel, mean=[0.0, 0.0, 0.0], cov=np.eye(3) * 0.5)
>>> few = {'AB': np.array([3, 0, 0, 1]), 'BA': np.array([2, 1, 0, 1])}
>>> r = fit_map(BayesOrderModel, few, prior)
>>> 0.5 < r.model.pA < 0.9                     # pulled toward the population mean 0.5
True
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from ..core.fit import FitResult, fit, _n_obs

__all__ = ['PopulationPrior', 'fit_map', 'PersonalisedHumanModel']


class PopulationPrior:
    """Gaussian prior on a model's free (unconstrained) parameters.

    Parameters
    ----------
    model_cls : type
        Model class whose ``free_params()`` the prior describes.
    mean : array_like
        Prior mean in the free space.
    cov : array_like
        Prior covariance (positive definite).
    """

    def __init__(self, model_cls, mean, cov):
        self.model_cls = model_cls
        self.mean = np.asarray(mean, float)
        self.cov = np.atleast_2d(np.asarray(cov, float))
        k = len(model_cls.free_params())
        if self.mean.shape != (k,) or self.cov.shape != (k, k):
            raise ValueError('mean must have %d entries and cov shape (%d, %d)' % (k, k, k))
        self.prec = np.linalg.inv(self.cov)

    @classmethod
    def from_fits(cls, fits, shrink=1e-2):
        """Empirical-Bayes prior from per-person fits.

        Parameters
        ----------
        fits : sequence of FitResult
            Maximum-likelihood fits of the same model class to different people (for example
            ``fit_individuals(...).results.values()``).
        shrink : float, optional
            Added to the covariance diagonal so it stays invertible with few people.

        Returns
        -------
        PopulationPrior
        """
        fits = list(fits)
        if not fits:
            raise ValueError('need at least one fit')
        model_cls = type(fits[0].model)
        X = np.array([f.model.to_vector() for f in fits])
        cov = np.cov(X.T) if len(X) > 1 else np.zeros((X.shape[1], X.shape[1]))
        return cls(model_cls, X.mean(0), np.atleast_2d(cov) + shrink * np.eye(X.shape[1]))

    def neg_log_prior(self, x):
        """Negative log prior density, up to a constant.

        Parameters
        ----------
        x : array_like
            Free parameter vector.

        Returns
        -------
        float
        """
        d = np.asarray(x, float) - self.mean
        return 0.5 * float(d @ self.prec @ d)


def fit_map(model_cls, data, prior, design=None, restarts=4, rng=None, **options):
    """Maximum a posteriori fit of one person's data under a population prior.

    Parameters
    ----------
    model_cls : type
        Model class (multinomial loss).
    data : dict
        The person's answer counts.
    prior : PopulationPrior
        Prior on the free parameters.
    design : iterable, optional
        Conditions passed to ``predict``.
    restarts : int, optional
        Random restarts (the first start is the prior mean).
    rng : numpy.random.Generator, optional
    **options
        Structural options of the model.

    Returns
    -------
    FitResult
        ``loss`` is the negative log posterior; ``loglik`` the log-likelihood at the MAP estimate.
    """
    if model_cls.LOSS != 'multinomial':
        raise ValueError('fit_map supports multinomial models')
    rng = np.random.default_rng(0) if rng is None else rng

    def objective(x):
        m = model_cls.from_vector(x, **options)
        return -m.loglik(data, design) + prior.neg_log_prior(x)

    starts = [prior.mean] + [prior.mean + rng.multivariate_normal(np.zeros(len(prior.mean)), prior.cov)
                             for _ in range(restarts - 1)]
    best = min((minimize(objective, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
                for x0 in starts), key=lambda r: r.fun)
    model = model_cls.from_vector(best.x, **options)
    return FitResult(model=model, loss=float(best.fun), loglik=float(model.loglik(data, design)),
                     k=len(model_cls.free_params()), n=_n_obs(model_cls, data), options=options)


class PersonalisedHumanModel:
    """Per-person human models with partial pooling.

    Parameters
    ----------
    model_cls : type
        Model class (for example :class:`~qlcog.families.order_effects.QuantumOrderModel4D`).
    prior : PopulationPrior
        Population prior; build it with :meth:`PopulationPrior.from_fits` from earlier people.
    design : iterable, optional
        Conditions.
    outcomes : int, optional
        Number of outcome cells per condition (4 for two yes/no questions).

    Examples
    --------
    >>> svc = PersonalisedHumanModel(BayesOrderModel, prior)      # doctest: +SKIP
    >>> svc.add('alice', 'AB', [1, 0])                            # doctest: +SKIP
    >>> svc.predict('alice')['AB']                                # doctest: +SKIP
    """

    CELLS = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}

    def __init__(self, model_cls, prior, design=None, outcomes=4):
        self.model_cls, self.prior, self.design, self.outcomes = model_cls, prior, design, outcomes
        self.counts = {}
        self.models = {}

    def add(self, person, condition, answers):
        """Record one answer pair and refit that person.

        Parameters
        ----------
        person : hashable
            Person identifier.
        condition : hashable
            Condition (for example the question order ``'AB'``).
        answers : sequence of int or int
            The answers (two yes/no answers for order designs) or an outcome index.

        Returns
        -------
        Model
            The person's updated model.
        """
        c = self.counts.setdefault(person, {})
        cell = self.CELLS[tuple(int(a) for a in answers)] if np.ndim(answers) else int(answers)
        c.setdefault(condition, np.zeros(self.outcomes, int))[cell] += 1
        self.models[person] = fit_map(self.model_cls, c, self.prior, self.design, restarts=2).model
        return self.models[person]

    def model(self, person):
        """The person's model (the population mean model for a new person).

        Parameters
        ----------
        person : hashable

        Returns
        -------
        Model
        """
        if person in self.models:
            return self.models[person]
        return self.model_cls.from_vector(self.prior.mean)

    def predict(self, person, design=None):
        """Predicted answer distribution for one person.

        Parameters
        ----------
        person : hashable
        design : iterable, optional

        Returns
        -------
        dict
        """
        return self.model(person).predict(design if design is not None else self.design)


def population_prior(model_cls, data_by_person, design=None, restarts=4, rng=None):
    """Convenience: fit every person by maximum likelihood and build the empirical-Bayes prior.

    Parameters
    ----------
    model_cls : type
    data_by_person : dict
        ``{person: data}``.
    design : iterable, optional
    restarts : int, optional
    rng : numpy.random.Generator, optional

    Returns
    -------
    PopulationPrior
    """
    fits = [fit(model_cls, d, design, restarts=restarts, rng=rng) for d in data_by_person.values()]
    return PopulationPrior.from_fits(fits)


__all__.append('population_prior')
