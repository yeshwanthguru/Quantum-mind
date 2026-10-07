"""Per-person models that update after every answer (sequential Monte Carlo over parameters).

Published models of judgement are fitted once to a group's data. A robot meets one person at a time
and has to adapt while the interaction is running. :class:`OnlinePersonModel` turns any multinomial
model in the library into a per-person model: it keeps a weighted sample of parameter vectors
(particles) drawn around a population estimate, reweights them by the likelihood of each new answer,
and resamples with a small kernel jitter when the weights degenerate (Liu and West, 2001; Chopin,
2002). Its predictions are posterior predictive distributions, so they carry the uncertainty about
this person's parameters.

The particles live on the unconstrained scale of :meth:`~quantum_mind.core.Model.to_vector`, so every
parameter constraint of the model is respected.

Examples
--------
>>> from quantum_mind.core.online import OnlinePersonModel
>>> from quantum_mind.families.order_effects import BayesOrderModel
>>> person = OnlinePersonModel(BayesOrderModel, n_particles=200)
>>> person.update('AB', 0)             # this person answered yes, yes when A was asked first
>>> p = person.predict()['AB']
>>> round(float(p.sum()), 6)
1.0
"""
from __future__ import annotations

import numpy as np

__all__ = ['OnlinePersonModel']


class OnlinePersonModel:
    """Posterior over one person's parameters, updated one observed outcome at a time.

    Parameters
    ----------
    model_cls : type
        Multinomial :class:`~quantum_mind.core.Model` subclass.
    prior : Model or FitResult, optional
        Population estimate at the centre of the prior (default: the model's default parameters).
    prior_scale : float, optional
        Standard deviation of the Gaussian prior around the centre, on the unconstrained scale.
    n_particles : int, optional
        Number of parameter samples.
    design : iterable, optional
        Design passed to ``predict``.
    resample_below : float, optional
        Resample when the effective sample size falls below this fraction of ``n_particles``.
    jitter : float, optional
        Kernel shrinkage ``h`` of the Liu-West move after resampling (0 disables it).
    rng : numpy.random.Generator, optional
        Random number generator (default seed 0).
    **options
        Structural options passed to the model constructor.

    Raises
    ------
    ValueError
        If the model is not multinomial (``LOSS == 'sse'``): single answers have no likelihood there.
    """

    def __init__(self, model_cls, prior=None, prior_scale=1.0, n_particles=500, design=None,
                 resample_below=0.5, jitter=0.1, rng=None, **options):
        if model_cls.LOSS != 'multinomial':
            raise ValueError('%s is fitted by least squares; online updating needs a likelihood per answer' % model_cls.__name__)
        prior = getattr(prior, 'model', prior)               # accept a FitResult
        if prior is not None:
            options = {**getattr(prior, 'options', {}), **options}
        self.model_cls, self.design, self.options = model_cls, design, options
        self.rng = np.random.default_rng(0) if rng is None else rng
        centre = (prior if prior is not None else model_cls(**options)).to_vector()
        k = len(centre)
        self.particles = centre + prior_scale * self.rng.normal(size=(n_particles, k))
        self.logw = np.zeros(n_particles)
        self.resample_below, self.jitter = resample_below, jitter
        self.n_observed = 0
        self._refresh()

    # ------------------------------------------------------------------ internals
    def _refresh(self):
        """Cache every particle's predictions (recomputed only after a resampling move)."""
        self._preds = [self.model_cls.from_vector(x, **self.options).predict(self.design) for x in self.particles]

    @property
    def weights(self):
        """numpy.ndarray: normalised particle weights."""
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    @property
    def ess(self):
        """float: effective sample size :math:`1 / \\sum_i w_i^2`."""
        return float(1.0 / np.sum(self.weights ** 2))

    def _resample(self):
        """Systematic resampling followed by a Liu-West kernel move."""
        n = len(self.particles)
        w = self.weights
        idx = np.searchsorted(np.cumsum(w), (self.rng.random() + np.arange(n)) / n)
        x = self.particles[np.minimum(idx, n - 1)]
        if self.jitter > 0:
            h = self.jitter
            a = np.sqrt(1 - h ** 2)
            mean = x.mean(axis=0)
            cov = np.atleast_2d(np.cov(x.T)) + 1e-9 * np.eye(x.shape[1])
            x = a * x + (1 - a) * mean + self.rng.multivariate_normal(np.zeros(x.shape[1]), h ** 2 * cov, size=n)
        self.particles, self.logw = x, np.zeros(n)
        self._refresh()

    # ------------------------------------------------------------------ public API
    def update(self, condition, outcome, count=1):
        """Condition on an observed outcome.

        Parameters
        ----------
        condition : hashable
            Condition of the design, for example ``'AB'``.
        outcome : int
            Index of the observed outcome in the model's prediction vector.
        count : int, optional
            Number of times the outcome was observed.
        """
        p = np.array([pr[condition][outcome] for pr in self._preds], float)
        self.logw += count * np.log(np.clip(p, 1e-12, 1.0))
        self.n_observed += count
        if self.ess < self.resample_below * len(self.particles):
            self._resample()

    def observe(self, data):
        """Condition on a batch of outcome counts.

        Parameters
        ----------
        data : dict
            ``{condition: counts}`` as used by :func:`~quantum_mind.core.fit.fit`.
        """
        for c, counts in data.items():
            for k, n in enumerate(counts):
                if n:
                    self.update(c, k, int(n))

    def predict(self, design=None):
        """Posterior predictive distribution for every condition.

        Parameters
        ----------
        design : iterable, optional
            Ignored (the design given at construction is used); accepted so the object can stand in
            for a model.

        Returns
        -------
        dict
            ``{condition: probability vector}`` averaged over the posterior.
        """
        w = self.weights
        return {c: sum(wi * np.asarray(pr[c], float) for wi, pr in zip(w, self._preds)) for c in self._preds[0]}

    def posterior_mean(self):
        """Posterior mean of the parameters on their natural scale.

        Returns
        -------
        dict
            ``{parameter name: weighted mean}``.
        """
        w = self.weights
        vals = [self.model_cls.from_vector(x, **self.options).params for x in self.particles]
        return {k: float(np.dot(w, [v[k] for v in vals])) for k in vals[0]}

    def credible_interval(self, name, level=0.9):
        """Weighted quantile interval of one parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        level : float, optional
            Posterior probability of the interval.

        Returns
        -------
        tuple of float
            ``(lower, upper)``.
        """
        w = self.weights
        v = np.array([self.model_cls.from_vector(x, **self.options).params[name] for x in self.particles])
        order = np.argsort(v)
        cw = np.cumsum(w[order])
        a = (1 - level) / 2
        return float(v[order][np.searchsorted(cw, a)]), float(v[order][min(np.searchsorted(cw, 1 - a), len(v) - 1)])
