"""Fitting, model comparison and model-recovery studies, common to every family.

* :func:`fit` fits one model class by multi-start optimisation (maximum likelihood or least squares).
* :func:`compare` fits several classes and ranks them by BIC or AIC.
* :func:`recovery` checks that the comparison selects the model that generated simulated data.
* :func:`fit_individuals` and :func:`compare_individuals` fit every person separately, which tests
  whether one set of parameters describes everyone.

Examples
--------
>>> from quantum_mind import fit, compare
>>> from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel
>>> data = QuantumOrderModel4D().sample(None, 500)          # doctest: +SKIP
>>> ranking = compare([QuantumOrderModel4D, BayesOrderModel], data)   # doctest: +SKIP
>>> ranking[0].summary()['model']                                     # doctest: +SKIP
'QuantumOrderModel4D'
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

__all__ = ['FitResult', 'fit', 'compare', 'recovery', 'kl', 'tvd', 'IndividualFits', 'fit_individuals', 'compare_individuals']


@dataclass
class FitResult:
    """Result of :func:`fit`.

    Attributes
    ----------
    model : Model
        Fitted model instance.
    loss : float
        Negative log-likelihood (multinomial models) or sum of squared errors (``'sse'`` models).
    loglik : float
        Log-likelihood. For ``'sse'`` models this is the Gaussian log-likelihood with the error
        variance profiled out, so that AIC and BIC are comparable across models.
    k : int
        Number of fitted parameters.
    n : int
        Number of observations.
    options : dict
        Structural options of the selected fit.
    """
    model: object
    loss: float          # negative log-likelihood (multinomial) or sum of squared errors (sse)
    loglik: float        # log-likelihood (for sse: Gaussian with the error variance profiled out)
    k: int               # number of fitted parameters
    n: int               # number of observations
    options: dict = field(default_factory=dict)

    @property
    def aic(self):
        """float: Akaike information criterion :math:`2k - 2\\log L`."""
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self):
        """float: Bayesian information criterion :math:`k\\log n - 2\\log L`."""
        return self.k * np.log(max(self.n, 1)) - 2 * self.loglik

    def summary(self):
        """Summary of the fit.

        Returns
        -------
        dict
            Keys ``model``, ``params``, ``options``, ``loglik``, ``k``, ``n``, ``aic`` and ``bic``.
        """
        return {'model': type(self.model).__name__, 'params': self.model.params, 'options': self.options,
                'loglik': self.loglik, 'k': self.k, 'n': self.n, 'aic': self.aic, 'bic': self.bic}


def _n_obs(model_cls, data):
    """Number of observations: data points for 'sse' models, total counts for multinomial ones."""
    if model_cls.LOSS == 'sse':
        return int(sum(np.size(v) for v in data.values()))
    return int(sum(np.sum(v) for v in data.values()))


def fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options):
    """Fit a model class to data by multi-start optimisation.

    Parameters
    ----------
    model_cls : type
        Subclass of :class:`~quantum_mind.core.Model`.
    data : dict
        ``{condition: counts}`` (multinomial models) or ``{condition: values}`` (``'sse'`` models).
    design : iterable, optional
        Conditions passed to ``predict``; most families accept ``None`` for their default design.
    restarts : int, optional
        Number of random starting points per structure.
    rng : numpy.random.Generator, optional
        Random number generator for the starting points (default: seed 0, so fits are reproducible).
    structures : list of dict, optional
        Discrete structural choices (for example projector ranks). Every structure is fitted and
        the best one kept.
    method : str, optional
        :func:`scipy.optimize.minimize` method.
    **options
        Fixed options passed to the model constructor.

    Returns
    -------
    FitResult
        Best fit over all structures and restarts.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    free = model_cls.free_params()
    best = None
    for st in (structures or [{}]):
        opts = dict(options, **st)

        def objective(x):
            m = model_cls.from_vector(x, **opts)
            return m.sse(data, design) if model_cls.LOSS == 'sse' else -m.loglik(data, design)

        for _ in range(restarts):
            x0 = np.array([p.random_free(rng) for p in free])
            if len(x0) == 0:                     # nothing to fit: evaluate once
                r_fun, r_x = objective(x0), x0
            else:
                r = minimize(objective, x0, method=method, options=dict(maxiter=6000, xatol=1e-8, fatol=1e-10))
                r_fun, r_x = r.fun, r.x
            if best is None or r_fun < best[0]:
                best = (r_fun, r_x, opts)
    loss, x, opts = best
    model = model_cls.from_vector(x, **opts)
    n = _n_obs(model_cls, data)
    if model_cls.LOSS == 'sse':
        # Gaussian log-likelihood with the maximum-likelihood variance sse / n plugged in
        ll = -0.5 * n * (np.log(2 * np.pi * max(loss, 1e-300) / n) + 1)
    else:
        ll = -loss
    return FitResult(model=model, loss=float(loss), loglik=float(ll), k=len(free), n=n, options=opts)


def compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic'):
    """Fit several model classes to the same data and rank them.

    Parameters
    ----------
    candidates : list
        Model classes, or ``(class, fit_kwargs)`` tuples.
    data : dict
        Data passed to :func:`fit`.
    design : iterable, optional
        Conditions passed to ``predict``.
    restarts : int, optional
        Restarts per fit.
    rng : numpy.random.Generator, optional
        Random number generator.
    criterion : {'bic', 'aic'}, optional
        Ranking criterion (lower is better).

    Returns
    -------
    list of FitResult
        Sorted from best to worst.
    """
    res = []
    for c in candidates:
        cls, kw = (c if isinstance(c, tuple) else (c, {}))
        res.append(fit(cls, data, design, restarts=restarts, rng=rng, **kw))
    return sorted(res, key=lambda r: getattr(r, criterion))


def recovery(generators, candidates, design, n, reps=20, rng=None, criterion='bic', restarts=8):
    """Model-recovery study.

    Data are simulated from each generating model, every candidate is fitted, and the criterion's
    choice is counted. A good design selects the generating model most of the time.

    Parameters
    ----------
    generators : dict
        ``{name: model instance}`` used to simulate data.
    candidates : list
        Model classes (or ``(class, fit_kwargs)``) to compare.
    design : iterable
        Conditions to simulate.
    n : int
        Observations per condition.
    reps : int, optional
        Simulated data sets per generator.
    rng : numpy.random.Generator, optional
        Random number generator.
    criterion : {'bic', 'aic'}, optional
        Selection criterion.
    restarts : int, optional
        Restarts per fit.

    Returns
    -------
    dict
        ``{generator: {candidate class name: times selected}}``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    names = [c[0].__name__ if isinstance(c, tuple) else c.__name__ for c in candidates]
    out = {g: {nm: 0 for nm in names} for g in generators}
    for g, gen in generators.items():
        for _ in range(reps):
            data = gen.sample(design, n, rng)
            best = compare(candidates, data, design, restarts=restarts, rng=rng, criterion=criterion)[0]
            out[g][type(best.model).__name__] += 1
    return out


def kl(p, q):
    """Kullback-Leibler divergence.

    Parameters
    ----------
    p, q : array_like
        Probability vectors (clipped at 1e-12).

    Returns
    -------
    float
        :math:`\\mathrm{KL}(p\\,\\Vert\\,q) = \\sum_i p_i \\log(p_i / q_i)` in nats.
    """
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    q = np.clip(np.asarray(q, float), 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def tvd(p, q):
    """Total variation distance.

    Parameters
    ----------
    p, q : array_like
        Probability vectors.

    Returns
    -------
    float
        :math:`\\tfrac12 \\sum_i |p_i - q_i|`, between 0 and 1.
    """
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


@dataclass
class IndividualFits:
    """Per-person fits of one model class, returned by :func:`fit_individuals`.

    Every person has their own parameters, so group criteria are sums over people.

    Attributes
    ----------
    model_name : str
        Name of the fitted class.
    results : dict
        ``{person: FitResult}``.
    """
    model_name: str
    results: dict

    @property
    def loglik(self):
        """float: sum of the per-person log-likelihoods."""
        return float(sum(r.loglik for r in self.results.values()))

    @property
    def k(self):
        """int: total number of fitted parameters (per person times number of people)."""
        return int(sum(r.k for r in self.results.values()))

    @property
    def n(self):
        """int: total number of observations."""
        return int(sum(r.n for r in self.results.values()))

    @property
    def bic(self):
        """float: group BIC, the sum of the per-person BICs."""
        return float(sum(r.bic for r in self.results.values()))

    @property
    def aic(self):
        """float: group AIC, the sum of the per-person AICs."""
        return float(sum(r.aic for r in self.results.values()))

    def parameters(self):
        """Fitted parameter values across people.

        Returns
        -------
        dict
            ``{parameter name: numpy.ndarray over people}``.
        """
        names = list(next(iter(self.results.values())).model.params)
        return {p: np.array([r.model.params[p] for r in self.results.values()]) for p in names}


def fit_individuals(model_cls, data_by_person, design=None, restarts=4, rng=None, **fit_kwargs):
    """Fit a model class separately to each person's data.

    Aggregate fitting assumes that everyone shares one set of parameters; per-person fits test that
    assumption.

    Parameters
    ----------
    model_cls : type
        Model class.
    data_by_person : dict
        ``{person: data}``.
    design : iterable, optional
        Conditions passed to ``predict``.
    restarts : int, optional
        Restarts per fit.
    rng : numpy.random.Generator, optional
        Random number generator.
    **fit_kwargs
        Passed to :func:`fit`.

    Returns
    -------
    IndividualFits

    Raises
    ------
    ValueError
        If ``data_by_person`` is empty.
    """
    if not data_by_person:
        raise ValueError('data_by_person is empty')
    rng = np.random.default_rng(0) if rng is None else rng
    return IndividualFits(model_cls.__name__, {p: fit(model_cls, d, design, restarts=restarts, rng=rng, **fit_kwargs)
                                               for p, d in data_by_person.items()})


def compare_individuals(candidates, data_by_person, design=None, restarts=4, rng=None, criterion='bic'):
    """Fit every candidate to every person and compare them.

    Parameters
    ----------
    candidates : list
        Model classes, or ``(class, fit_kwargs)`` tuples.
    data_by_person : dict
        ``{person: data}``.
    design : iterable, optional
        Conditions passed to ``predict``.
    restarts : int, optional
        Restarts per fit.
    rng : numpy.random.Generator, optional
        Random number generator.
    criterion : {'bic', 'aic'}, optional
        Comparison criterion (lower is better).

    Returns
    -------
    dict
        ``'group'``: ``[(model name, group criterion)]`` sorted from best;
        ``'best_counts'``: ``{model name: number of people it fits best}``;
        ``'fits'``: ``{model name: IndividualFits}``.
    """
    fits = {}
    for c in candidates:
        cls, kw = (c if isinstance(c, tuple) else (c, {}))
        fits[cls.__name__] = fit_individuals(cls, data_by_person, design, restarts, rng, **kw)
    counts = {m: 0 for m in fits}
    for p in data_by_person:
        best = min(fits, key=lambda m: getattr(fits[m].results[p], criterion))
        counts[best] += 1
    group = sorted(((m, getattr(f, criterion)) for m, f in fits.items()), key=lambda t: t[1])
    return {'group': group, 'best_counts': counts, 'fits': fits}
