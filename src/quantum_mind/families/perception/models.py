r"""Bistable perception: how an ambiguous stimulus flips between two interpretations.

A Necker cube, a rotating silhouette or an ambiguous object seen by a robot's camera is perceived as
one of two interpretations, which alternate. The time spent in one interpretation (the *dwell time*)
is the standard measurement.

**Models.**

* :class:`QuantumZenoBistableModel` (Atmanspacher, Filk and Römer, Biological Cybernetics 90, 33-40,
  2004): the percept is a qubit whose two basis states are the two interpretations. Between two
  "observations" (internal sampling of the percept, every :math:`\Delta t`) the state rotates freely,
  :math:`U = e^{-i g \Delta t\, \sigma_x}`; each observation collapses it. The probability of a switch at
  one observation is :math:`q = \sin^2(g\,\Delta t)`, so dwell times are geometric with mean
  :math:`\Delta t / \sin^2(g \Delta t) \approx 1 / (g^2 \Delta t)`: *more frequent observation slows the
  switching* (the quantum Zeno effect).
* :class:`MarkovSwitchingModel`: classical baseline; switches occur at a constant rate :math:`r`, so dwell
  times are exponential and do not depend on the observation interval.
* :class:`GammaRenewalModel`: the standard empirical description of dwell times (Levelt), a gamma
  distribution with shape :math:`k` and scale :math:`\theta`.

**What distinguishes them.** With one observation interval the Zeno and Markov models predict the same
(geometric or exponential) shape; the gamma model fits the typical unimodal histograms better. The
distinctive prediction of the Zeno model is the dependence on :math:`\Delta t`: halving the observation
interval should double the mean dwell time. Design conditions with different intervals (for example
different presentation or attention-sampling rates) to test it.

**Design.** ``{condition: (dt, bin_edges)}`` with the observation interval and the histogram bins (in
seconds); built with :func:`dwell_time_design`. Outcomes: the probability of a dwell time falling in each
bin (multinomial, with a final bin for longer dwell times). Data: dwell-time counts per bin, built from
raw dwell times with :func:`dwell_counts`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist
from ...core import Model, Param

__all__ = ['QuantumZenoBistableModel', 'MarkovSwitchingModel', 'GammaRenewalModel', 'dwell_time_design',
           'dwell_counts', 'mean_dwell_time']


def dwell_time_design(dts, max_time=20.0, n_bins=20):
    """Design with equal-width dwell-time bins for several observation intervals.

    Parameters
    ----------
    dts : dict or sequence
        ``{condition: dt}`` or a sequence of intervals (conditions named ``'dt=<value>'``).
    max_time : float, optional
        Upper edge of the last finite bin, in seconds.
    n_bins : int, optional
        Number of finite bins (one more bin collects longer dwell times).

    Returns
    -------
    dict
        ``{condition: (dt, bin_edges)}``.
    """
    if not isinstance(dts, dict):
        dts = {'dt=%g' % d: d for d in dts}
    edges = np.linspace(0, max_time, n_bins + 1)
    return {c: (float(d), edges) for c, d in dts.items()}


def dwell_counts(dwell_times, edges):
    """Histogram of dwell times, with a final bin for times beyond the last edge.

    Parameters
    ----------
    dwell_times : array_like
        Observed dwell times in seconds.
    edges : array_like
        Bin edges.

    Returns
    -------
    numpy.ndarray
        Counts, length ``len(edges)``.
    """
    t = np.asarray(dwell_times, float)
    c, _ = np.histogram(np.clip(t, edges[0], None), bins=edges)
    return np.r_[c, np.sum(t >= edges[-1])]


def _cdf_bins(cdf, edges):
    """Bin probabilities from a CDF, with the tail probability as the last entry."""
    F = np.asarray([cdf(e) for e in edges], float)
    return np.clip(np.r_[np.diff(F), 1 - F[-1]], 0, None)


def mean_dwell_time(model, dt):
    """Mean dwell time predicted by a bistable-perception model.

    Parameters
    ----------
    model : QuantumZenoBistableModel, MarkovSwitchingModel or GammaRenewalModel
    dt : float
        Observation interval (ignored by the Markov and gamma models).

    Returns
    -------
    float
        Seconds.
    """
    return model.mean_dwell(dt)


class QuantumZenoBistableModel(Model):
    """Quantum Zeno model of bistable perception.

    Parameters
    ----------
    g : float
        Free switching frequency (rad/s) of the unobserved percept.
    """
    PARAMS = [Param('g', 'positive', 1.0)]
    name = 'Quantum Zeno'

    def switch_probability(self, dt):
        """Probability of a switch at one observation.

        Parameters
        ----------
        dt : float
            Observation interval.

        Returns
        -------
        float
            :math:`\\sin^2(g\\,\\Delta t)`.
        """
        return float(np.clip(np.sin(self.g * dt) ** 2, 1e-12, 1.0))

    def mean_dwell(self, dt):
        """Mean dwell time :math:`\\Delta t / \\sin^2(g \\Delta t)`.

        Parameters
        ----------
        dt : float

        Returns
        -------
        float
        """
        return dt / self.switch_probability(dt)

    def predict(self, design=None):
        """Dwell-time bin probabilities for each condition (geometric in units of ``dt``)."""
        design = design or dwell_time_design([0.07])
        out = {}
        for c, (dt, edges) in design.items():
            q = self.switch_probability(dt)
            # P(dwell <= t) = 1 - (1 - q)^floor(t / dt)
            out[c] = _cdf_bins(lambda t: 1 - (1 - q) ** np.floor(t / dt + 1e-9), edges)
        return out


class MarkovSwitchingModel(Model):
    """Classical baseline: switches at a constant rate (exponential dwell times, no Zeno effect).

    Parameters
    ----------
    rate : float
        Switching rate (1/s); the mean dwell time is ``1 / rate``.
    """
    PARAMS = [Param('rate', 'positive', 0.3)]
    name = 'Markov switching'

    def mean_dwell(self, dt=None):
        """Mean dwell time ``1 / rate`` (independent of the observation interval).

        Parameters
        ----------
        dt : float, optional
            Ignored.

        Returns
        -------
        float
        """
        return 1 / self.rate

    def predict(self, design=None):
        """Dwell-time bin probabilities (exponential distribution) for each condition."""
        design = design or dwell_time_design([0.07])
        return {c: _cdf_bins(lambda t: 1 - np.exp(-self.rate * t), edges) for c, (dt, edges) in design.items()}


class GammaRenewalModel(Model):
    """Classical baseline: gamma-distributed dwell times (the usual empirical description).

    Parameters
    ----------
    shape : float
        Gamma shape :math:`k` (k = 1 is the exponential).
    scale : float
        Gamma scale :math:`\\theta` (seconds); the mean is :math:`k\\theta`.
    """
    PARAMS = [Param('shape', 'positive', 3.0), Param('scale', 'positive', 1.0)]
    name = 'Gamma renewal'

    def mean_dwell(self, dt=None):
        """Mean dwell time :math:`k\\theta` (independent of the observation interval).

        Parameters
        ----------
        dt : float, optional
            Ignored.

        Returns
        -------
        float
        """
        return self.shape * self.scale

    def predict(self, design=None):
        """Dwell-time bin probabilities (gamma distribution) for each condition."""
        design = design or dwell_time_design([0.07])
        dist = gamma_dist(self.shape, scale=self.scale)
        return {c: _cdf_bins(dist.cdf, edges) for c, (dt, edges) in design.items()}
