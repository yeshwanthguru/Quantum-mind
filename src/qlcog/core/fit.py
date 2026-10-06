"""Fitting, model comparison and model-recovery benchmarks, common to every family."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

__all__ = ['FitResult', 'fit', 'compare', 'recovery', 'kl', 'tvd', 'IndividualFits', 'fit_individuals', 'compare_individuals']


@dataclass
class FitResult:
    """Result of `fit`: the fitted model, loss, log-likelihood, number of parameters k, observations n and
    options; properties aic and bic."""
    model: object
    loss: float          # negative log-likelihood (multinomial) or sum of squared errors (sse)
    loglik: float        # log-likelihood (for sse: Gaussian with the error variance profiled out)
    k: int               # number of fitted parameters
    n: int               # number of observations
    options: dict = field(default_factory=dict)

    @property
    def aic(self):
        """Akaike information criterion 2k - 2 log L."""
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self):
        """Bayesian information criterion k log n - 2 log L."""
        return self.k * np.log(max(self.n, 1)) - 2 * self.loglik

    def summary(self):
        """Dictionary with model name, parameters, options, log-likelihood, k, n, AIC and BIC."""
        return {'model': type(self.model).__name__, 'params': self.model.params, 'options': self.options,
                'loglik': self.loglik, 'k': self.k, 'n': self.n, 'aic': self.aic, 'bic': self.bic}


def _n_obs(model_cls, data):
    if model_cls.LOSS == 'sse':
        return int(sum(np.size(v) for v in data.values()))
    return int(sum(np.sum(v) for v in data.values()))


def fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options):
    """Fit a model class to data by multi-start optimisation.

    structures: optional list of option dicts (discrete structural choices, e.g. projector ranks);
                every structure is tried and the best fit kept.
    options:    fixed options passed to the model constructor."""
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
            if len(x0) == 0:
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
        ll = -0.5 * n * (np.log(2 * np.pi * max(loss, 1e-300) / n) + 1)
    else:
        ll = -loss
    return FitResult(model=model, loss=float(loss), loglik=float(ll), k=len(free), n=n, options=opts)


def compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic'):
    """Fit several models and rank them. candidates: list of model classes or (class, fit kwargs).
    Returns a list of FitResult sorted by the criterion (lower is better)."""
    res = []
    for c in candidates:
        cls, kw = (c if isinstance(c, tuple) else (c, {}))
        res.append(fit(cls, data, design, restarts=restarts, rng=rng, **kw))
    return sorted(res, key=lambda r: getattr(r, criterion))


def recovery(generators, candidates, design, n, reps=20, rng=None, criterion='bic', restarts=8):
    """Model-recovery study: simulate data from each generator (dict name -> model instance), fit all
    candidates and count which one the criterion selects. Returns {generator: {candidate: count}}."""
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
    """Kullback-Leibler divergence KL(p || q) in nats (probabilities clipped at 1e-12)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1); q = np.clip(np.asarray(q, float), 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def tvd(p, q):
    """Total variation distance between two probability vectors."""
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


@dataclass
class IndividualFits:
    """Per-person fits of one model class: results {person: FitResult}; summed log-likelihood, total
    parameters and group AIC/BIC (each person has their own parameters)."""
    model_name: str
    results: dict

    @property
    def loglik(self):
        """Sum of the per-person log-likelihoods."""
        return float(sum(r.loglik for r in self.results.values()))

    @property
    def k(self):
        """Total number of fitted parameters (per person times number of people)."""
        return int(sum(r.k for r in self.results.values()))

    @property
    def n(self):
        """Total number of observations."""
        return int(sum(r.n for r in self.results.values()))

    @property
    def bic(self):
        """Group BIC: sum of the per-person BICs."""
        return float(sum(r.bic for r in self.results.values()))

    @property
    def aic(self):
        """Group AIC: sum of the per-person AICs."""
        return float(sum(r.aic for r in self.results.values()))

    def parameters(self):
        """{parameter: array over people} of the fitted parameter values."""
        names = list(next(iter(self.results.values())).model.params)
        return {p: np.array([r.model.params[p] for r in self.results.values()]) for p in names}


def fit_individuals(model_cls, data_by_person, design=None, restarts=4, rng=None, **fit_kwargs):
    """Fit `model_cls` separately to each person's data {person: data}. Aggregate fitting assumes that
    everyone follows the same parameters; per-person fits test that assumption."""
    if not data_by_person:
        raise ValueError('data_by_person is empty')
    rng = np.random.default_rng(0) if rng is None else rng
    return IndividualFits(model_cls.__name__, {p: fit(model_cls, d, design, restarts=restarts, rng=rng, **fit_kwargs)
                                               for p, d in data_by_person.items()})


def compare_individuals(candidates, data_by_person, design=None, restarts=4, rng=None, criterion='bic'):
    """Fit every candidate to every person. Returns {'group': [(model, group criterion)] sorted (lower is
    better), 'best_counts': {model: number of people it fits best}, 'fits': {model: IndividualFits}}."""
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
