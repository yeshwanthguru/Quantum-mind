"""Parametric bootstrap for any fitted model: parameter intervals and predictive intervals.

Published quantum-like models are usually reported with point estimates only. :func:`bootstrap`
gives every model in the library, quantum-like or classical, the same uncertainty treatment: data
are simulated from the fitted model with the observed sample sizes, the model is refitted to each
simulated data set, and percentile intervals are read off the refits (Efron and Tibshirani, 1993).

For multinomial models the simulated data are multinomial draws with each condition's observed
total. For ``'sse'`` models they are the fitted predictions plus Gaussian noise with the
maximum-likelihood standard deviation :math:`\\sqrt{\\mathrm{SSE}/n}`.

Parameter intervals of models with symmetric parameterisations (for example an angle and its
reflection give the same predictions) can be wide or bimodal; predictive intervals are not affected,
so a robot should use :meth:`BootstrapResult.predictive_interval`.

Examples
--------
>>> from quantum_mind import fit
>>> from quantum_mind.core.uncertainty import bootstrap
>>> from quantum_mind.families.order_effects import BayesOrderModel
>>> data = {'AB': [40, 10, 20, 30], 'BA': [35, 15, 15, 35]}
>>> res = fit(BayesOrderModel, data, restarts=2)
>>> boot = bootstrap(res, data, n_boot=20)
>>> lo, hi = boot.predictive_interval()['AB']
>>> bool((lo <= hi).all())
True
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fit import FitResult, fit

__all__ = ['BootstrapResult', 'bootstrap']


@dataclass
class BootstrapResult:
    """Refits of a model to data simulated from its own fit, returned by :func:`bootstrap`.

    Attributes
    ----------
    fit : FitResult
        Original fit.
    samples : dict
        ``{parameter name: numpy.ndarray of length n_boot}``: refitted parameter values.
    predictions : dict
        ``{condition: numpy.ndarray of shape (n_boot, n_outcomes)}``: predictions of the refits.
    """
    fit: FitResult
    samples: dict
    predictions: dict

    @property
    def n_boot(self):
        """int: number of bootstrap refits."""
        return len(next(iter(self.predictions.values()))) if self.predictions else 0

    def se(self):
        """Bootstrap standard errors of the parameters.

        Returns
        -------
        dict
            ``{parameter name: standard deviation of the refits}``.
        """
        return {k: float(np.std(v, ddof=1)) for k, v in self.samples.items()}

    def ci(self, level=0.95):
        """Percentile intervals of the parameters.

        Parameters
        ----------
        level : float, optional
            Coverage of the interval.

        Returns
        -------
        dict
            ``{parameter name: (lower, upper)}``.
        """
        a = 100 * (1 - level) / 2
        return {k: (float(np.percentile(v, a)), float(np.percentile(v, 100 - a))) for k, v in self.samples.items()}

    def predictive_interval(self, level=0.95):
        """Percentile intervals of the predicted probabilities (or values) per condition.

        Parameters
        ----------
        level : float, optional
            Coverage of the interval.

        Returns
        -------
        dict
            ``{condition: (lower vector, upper vector)}``.
        """
        a = 100 * (1 - level) / 2
        return {c: (np.percentile(P, a, axis=0), np.percentile(P, 100 - a, axis=0)) for c, P in self.predictions.items()}


def _simulate(model, data, design, loss, n, rng):
    """One simulated data set with the structure and sample sizes of ``data``."""
    pred = model.predict(design)
    if type(model).LOSS == 'sse':
        sigma = np.sqrt(max(loss, 0.0) / max(n, 1))
        return {c: np.asarray(pred[c], float) + rng.normal(0, sigma, np.shape(v)) for c, v in data.items()}
    out = {}
    for c, counts in data.items():
        p = np.clip(np.asarray(pred[c], float), 0, None)
        out[c] = rng.multinomial(int(np.sum(counts)), p / p.sum())
    return out


def bootstrap(result, data, design=None, n_boot=200, restarts=2, rng=None):
    """Parametric bootstrap of a fitted model.

    Parameters
    ----------
    result : FitResult
        Fit returned by :func:`~quantum_mind.core.fit.fit` or :func:`~quantum_mind.core.fit.compare`.
    data : dict
        The data that were fitted (only the conditions and sample sizes are used).
    design : iterable, optional
        Design passed to ``predict``.
    n_boot : int, optional
        Number of simulated data sets.
    restarts : int, optional
        Restarts per refit; the first starts at the original estimate.
    rng : numpy.random.Generator, optional
        Random number generator (default seed 0).

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(0) if rng is None else rng
    model = result.model
    cls = type(model)
    names = [p.name for p in cls.free_params()]
    samples = {k: [] for k in names}
    preds = {c: [] for c in data}
    x0 = model.to_vector()
    for _ in range(n_boot):
        sim = _simulate(model, data, design, result.loss, result.n, rng)
        r = fit(cls, sim, design, restarts=restarts, rng=rng, x0=x0, **result.options)
        for k in names:
            samples[k].append(r.model.params[k])
        p = r.model.predict(design)
        for c in data:
            preds[c].append(np.asarray(p[c], float))
    return BootstrapResult(result, {k: np.array(v) for k, v in samples.items()},
                           {c: np.array(v) for c, v in preds.items()})
