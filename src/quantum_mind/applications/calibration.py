r"""Calibration of confidence signals: measure it, correct it, and bound it.

A robot that routes between modules (vision, language, a human model, a planner) or decides whether to
ask a person before acting relies on the confidence each module reports. A confidence is *calibrated*
when, among all the times a module reports 0.8, it is right about 80% of the time. This module provides
the standard tools to check and repair that property.

* Measures: :func:`reliability_curve`, :func:`expected_calibration_error`,
  :func:`max_calibration_error`, :func:`brier_score`.
* Recalibration: :class:`PlattScaling` (binary), :class:`TemperatureScaling` (multi-class),
  :class:`IsotonicCalibration` (monotone, non-parametric).
* Guarantees: :class:`SplitConformalClassifier` (prediction sets with a coverage guarantee) and
  :func:`clopper_pearson` (exact binomial bounds), combined in :class:`CalibrationMonitor`, which tracks
  a module's outcomes online and returns a lower confidence bound on its success probability.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.applications.calibration import expected_calibration_error, IsotonicCalibration
>>> rng = np.random.default_rng(0)
>>> p = rng.uniform(0.3, 0.9, 5000)                 # true success probabilities
>>> y = rng.random(5000) < p                         # outcomes
>>> over = np.clip(p + 0.15, 0, 1)                   # an over-confident module
>>> expected_calibration_error(over, y) > 0.1          # reports about 0.15 too much
True
>>> fixed = IsotonicCalibration().fit(over, y).transform(over)
>>> expected_calibration_error(fixed, y) < 0.03
True
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import beta

__all__ = ['reliability_curve', 'expected_calibration_error', 'max_calibration_error', 'brier_score',
           'PlattScaling', 'TemperatureScaling', 'IsotonicCalibration', 'SplitConformalClassifier',
           'clopper_pearson', 'CalibrationMonitor', 'AdaptiveConformalSets']


def _binary_inputs(conf, outcome):
    """Validate a vector of confidences in [0, 1] and matching 0/1 outcomes."""
    c = np.asarray(conf, float).ravel()
    y = np.asarray(outcome, float).ravel()
    if c.shape != y.shape or c.size == 0:
        raise ValueError('confidences and outcomes must be non-empty and of the same length')
    if np.any((c < 0) | (c > 1)):
        raise ValueError('confidences must lie in [0, 1]')
    return c, y


def reliability_curve(conf, outcome, bins=10):
    """Data of a reliability diagram.

    Parameters
    ----------
    conf : array_like
        Reported confidences in [0, 1].
    outcome : array_like
        1 if the prediction was right (or the action succeeded), 0 otherwise.
    bins : int, optional
        Number of equal-width confidence bins.

    Returns
    -------
    dict
        ``confidence`` (mean confidence per bin), ``accuracy`` (success rate per bin), ``count`` and
        ``edges``. Empty bins hold ``nan``. A calibrated module lies on the diagonal.
    """
    c, y = _binary_inputs(conf, outcome)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1]), 0, bins - 1)
    cnt = np.bincount(idx, minlength=bins).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mc = np.bincount(idx, weights=c, minlength=bins) / cnt
        acc = np.bincount(idx, weights=y, minlength=bins) / cnt
    return {'confidence': mc, 'accuracy': acc, 'count': cnt.astype(int), 'edges': edges}


def expected_calibration_error(conf, outcome, bins=10):
    """Expected calibration error (ECE).

    Parameters
    ----------
    conf : array_like
        Confidences in [0, 1].
    outcome : array_like
        0/1 outcomes.
    bins : int, optional
        Number of bins.

    Returns
    -------
    float
        :math:`\\sum_b \\frac{n_b}{N} |\\mathrm{acc}_b - \\mathrm{conf}_b|`; 0 for a perfectly calibrated module.
    """
    r = reliability_curve(conf, outcome, bins)
    ok = r['count'] > 0
    return float(np.sum(r['count'][ok] * np.abs(r['accuracy'][ok] - r['confidence'][ok])) / r['count'].sum())


def max_calibration_error(conf, outcome, bins=10):
    """Largest calibration gap over the non-empty bins.

    Parameters
    ----------
    conf : array_like
        Confidences.
    outcome : array_like
        0/1 outcomes.
    bins : int, optional
        Number of bins.

    Returns
    -------
    float
    """
    r = reliability_curve(conf, outcome, bins)
    ok = r['count'] > 0
    return float(np.max(np.abs(r['accuracy'][ok] - r['confidence'][ok])))


def brier_score(prob, outcome):
    """Brier score (mean squared error of probabilities).

    Parameters
    ----------
    prob : array_like
        Binary confidences, shape ``(n,)``, or class probabilities, shape ``(n, k)``.
    outcome : array_like
        0/1 outcomes, shape ``(n,)``, or class labels ``0..k-1`` for the multi-class case.

    Returns
    -------
    float
        Lower is better; it rewards both calibration and sharpness.
    """
    p = np.asarray(prob, float)
    if p.ndim == 1:
        c, y = _binary_inputs(p, outcome)
        return float(np.mean((c - y) ** 2))
    Y = np.eye(p.shape[1])[np.asarray(outcome, int)]
    return float(np.mean(np.sum((p - Y) ** 2, axis=1)))


@dataclass
class PlattScaling:
    """Logistic recalibration of a binary confidence: :math:`\\sigma(a\\,\\mathrm{logit}(c) + b)`.

    Attributes
    ----------
    a, b : float
        Slope and intercept, fitted by maximum likelihood (``a = 1, b = 0`` is the identity).
    """
    a: float = 1.0
    b: float = 0.0

    def fit(self, conf, outcome):
        """Fit the slope and intercept.

        Parameters
        ----------
        conf : array_like
            Confidences in [0, 1].
        outcome : array_like
            0/1 outcomes.

        Returns
        -------
        PlattScaling
            ``self``.
        """
        c, y = _binary_inputs(conf, outcome)
        z = logit(np.clip(c, 1e-6, 1 - 1e-6))

        def nll(w):
            s = w[0] * z + w[1]
            return float(np.sum(np.logaddexp(0, s) - y * s))
        self.a, self.b = (float(v) for v in minimize(nll, [1.0, 0.0], method='BFGS').x)
        return self

    def transform(self, conf):
        """Recalibrated confidences.

        Parameters
        ----------
        conf : array_like
            Confidences in [0, 1].

        Returns
        -------
        numpy.ndarray
        """
        z = logit(np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6))
        return expit(self.a * z + self.b)


@dataclass
class TemperatureScaling:
    """Multi-class recalibration by one temperature: :math:`\\mathrm{softmax}(\\log p / T)`.

    Attributes
    ----------
    T : float
        Temperature (> 1 softens over-confident predictions, < 1 sharpens under-confident ones).
    """
    T: float = 1.0

    @staticmethod
    def _scaled(prob, T):
        """Softmax of log-probabilities divided by T."""
        z = np.log(np.clip(np.asarray(prob, float), 1e-12, None)) / T
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, prob, labels):
        """Fit the temperature by minimising the negative log-likelihood.

        Parameters
        ----------
        prob : array_like
            Class probabilities, shape ``(n, k)``.
        labels : array_like
            True classes ``0..k-1``.

        Returns
        -------
        TemperatureScaling
            ``self``.
        """
        p = np.asarray(prob, float)
        y = np.asarray(labels, int)

        def nll(logT):
            q = self._scaled(p, np.exp(logT))
            return -float(np.sum(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None))))
        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x))
        return self

    def transform(self, prob):
        """Recalibrated class probabilities.

        Parameters
        ----------
        prob : array_like
            Shape ``(n, k)``.

        Returns
        -------
        numpy.ndarray
        """
        return self._scaled(prob, self.T)


@dataclass
class IsotonicCalibration:
    """Monotone, non-parametric recalibration by the pool-adjacent-violators algorithm.

    Attributes
    ----------
    x_, y_ : numpy.ndarray
        Knots of the fitted non-decreasing step function (interpolated linearly between knots).
    """
    x_: np.ndarray = field(default=None, repr=False)
    y_: np.ndarray = field(default=None, repr=False)

    def fit(self, conf, outcome):
        """Fit the monotone map from confidence to success rate.

        Parameters
        ----------
        conf : array_like
            Confidences in [0, 1].
        outcome : array_like
            0/1 outcomes.

        Returns
        -------
        IsotonicCalibration
            ``self``.
        """
        c, y = _binary_inputs(conf, outcome)
        order = np.argsort(c)
        c, y = c[order], y[order]
        # pool adjacent violators: blocks of (sum, count, x-mean)
        sums, cnts, xs = [], [], []
        for ci, yi in zip(c, y):
            sums.append(yi); cnts.append(1.0); xs.append(ci)
            while len(sums) > 1 and sums[-2] / cnts[-2] > sums[-1] / cnts[-1]:
                s, n, x = sums.pop(), cnts.pop(), xs.pop()
                xs[-1] = (xs[-1] * cnts[-1] + x * n) / (cnts[-1] + n)
                sums[-1] += s
                cnts[-1] += n
        self.x_ = np.array(xs)
        self.y_ = np.array(sums) / np.array(cnts)
        return self

    def transform(self, conf):
        """Recalibrated confidences.

        Parameters
        ----------
        conf : array_like

        Returns
        -------
        numpy.ndarray
        """
        return np.interp(np.asarray(conf, float), self.x_, self.y_)


@dataclass
class SplitConformalClassifier:
    """Split conformal prediction sets for any probabilistic classifier.

    After :meth:`fit` on held-out calibration data, the prediction set of a new input contains the
    true class with probability at least :math:`1 - \\alpha` (marginally, for exchangeable data),
    whatever the classifier. A robot can act when the set has one element and ask when it has several.

    Attributes
    ----------
    alpha : float
        Allowed miscoverage, for example 0.1 for 90% coverage.
    qhat : float
        Fitted threshold on the score :math:`1 - p_{\\text{true}}`.
    """
    alpha: float = 0.1
    qhat: float = None

    def fit(self, prob, labels):
        """Compute the conformal threshold from calibration data.

        Parameters
        ----------
        prob : array_like
            Class probabilities on the calibration set, shape ``(n, k)``.
        labels : array_like
            True classes.

        Returns
        -------
        SplitConformalClassifier
            ``self``.
        """
        p = np.asarray(prob, float)
        y = np.asarray(labels, int)
        n = len(y)
        scores = 1 - p[np.arange(n), y]
        k = int(np.ceil((n + 1) * (1 - self.alpha)))          # rank of the conformal quantile
        self.qhat = float(np.sort(scores)[k - 1]) if k <= n else 1.0   # too few points: every label
        return self

    def predict_sets(self, prob):
        """Prediction sets.

        Parameters
        ----------
        prob : array_like
            Class probabilities, shape ``(n, k)``.

        Returns
        -------
        numpy.ndarray
            Boolean mask ``(n, k)``: class j is in the set of sample i when ``mask[i, j]``.
        """
        return 1 - np.asarray(prob, float) <= self.qhat


def clopper_pearson(successes, trials, alpha=0.05):
    """Exact (Clopper-Pearson) confidence interval for a success probability.

    Parameters
    ----------
    successes : int
        Number of successes.
    trials : int
        Number of trials.
    alpha : float, optional
        Two-sided error level.

    Returns
    -------
    tuple of float
        ``(lower, upper)``; ``(0, 1)`` when there are no trials.
    """
    if trials <= 0:
        return 0.0, 1.0
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lo, hi


class CalibrationMonitor:
    """Online calibration check of one module, with a guaranteed lower bound on its success rate.

    Outcomes are binned by reported confidence. For a new confidence the monitor returns the
    recalibrated success estimate of its bin and a Clopper-Pearson lower bound, which a risk-aware
    decision rule (:func:`quantum_mind.applications.robotics.risk_aware_ask_or_act`) can use instead of the
    raw confidence.

    Parameters
    ----------
    bins : int, optional
        Number of confidence bins.
    alpha : float, optional
        Error level of the bounds.
    min_count : int, optional
        Below this many outcomes in a bin, the raw confidence is returned as the estimate.

    Examples
    --------
    >>> from quantum_mind.applications.calibration import CalibrationMonitor
    >>> m = CalibrationMonitor()
    >>> for c, ok in [(0.9, 1), (0.9, 0), (0.9, 1), (0.9, 0)] * 10:
    ...     m.update(c, ok)
    >>> est, lower = m.estimate(0.9)
    >>> round(est, 2), lower < 0.5
    (0.5, True)
    """

    def __init__(self, bins=10, alpha=0.05, min_count=10):
        self.bins, self.alpha, self.min_count = bins, alpha, min_count
        self.n = np.zeros(bins, int)
        self.k = np.zeros(bins, int)

    def _bin(self, conf):
        """Bin index of a confidence."""
        return int(min(self.bins - 1, max(0, np.floor(float(conf) * self.bins))))

    def update(self, conf, success):
        """Record one outcome.

        Parameters
        ----------
        conf : float
            Confidence the module reported.
        success : bool or int
            Whether the module turned out to be right.
        """
        b = self._bin(conf)
        self.n[b] += 1
        self.k[b] += int(bool(success))

    def estimate(self, conf):
        """Recalibrated success estimate and lower bound for a reported confidence.

        Parameters
        ----------
        conf : float

        Returns
        -------
        estimate : float
            Observed success rate of the bin (the raw confidence while the bin has too few outcomes).
        lower : float
            Clopper-Pearson lower bound (0 while the bin is empty).
        """
        b = self._bin(conf)
        lo, _ = clopper_pearson(self.k[b], self.n[b], self.alpha)
        est = self.k[b] / self.n[b] if self.n[b] >= self.min_count else float(conf)
        return float(est), float(lo)

    def ece(self):
        """Expected calibration error of the outcomes recorded so far (bin centres as confidences).

        Returns
        -------
        float
        """
        ok = self.n > 0
        if not ok.any():
            return 0.0
        centres = (np.arange(self.bins) + 0.5) / self.bins
        return float(np.sum(self.n[ok] * np.abs(self.k[ok] / self.n[ok] - centres[ok])) / self.n.sum())


class AdaptiveConformalSets:
    """Prediction sets for a person's next answer that keep their coverage while the person changes.

    Split conformal prediction (:class:`SplitConformalClassifier`) assumes the calibration answers and
    the new answer are exchangeable. Within one interaction they are not: trust, attention and the
    person's view of the task drift. Adaptive conformal inference (Gibbs and Candès, 2021) keeps a
    running miscoverage level :math:`\\alpha_t` and corrects it after every answer,

    .. math:: \\alpha_{t+1} = \\alpha_t + \\gamma\\,(\\alpha - \\mathrm{err}_t),

    where :math:`\\mathrm{err}_t = 1` if the answer fell outside the set. Whatever the sequence of
    answers, the long-run miscoverage satisfies

    .. math:: \\Bigl|\\frac1T \\sum_{t=1}^T \\mathrm{err}_t - \\alpha\\Bigr| \\le
              \\frac{\\max(\\alpha_1, 1 - \\alpha_1) + \\gamma}{\\gamma T}.

    The probabilities can come from any model of the person, for example the posterior predictive of
    :class:`~quantum_mind.core.online.OnlinePersonModel`. A robot acts when the set holds one answer
    and asks when it holds several.

    Parameters
    ----------
    alpha : float, optional
        Target miscoverage (0.1 for 90% coverage).
    gamma : float, optional
        Step size of the correction.
    calibration_scores : array_like, optional
        Scores :math:`1 - p_{\\text{true}}` from earlier people, used before this person has answered.

    Attributes
    ----------
    alpha_t : float
        Current working miscoverage level.
    errors : list of int
        1 for every answer outside its set.

    Examples
    --------
    >>> import numpy as np
    >>> aci = AdaptiveConformalSets(alpha=0.2, gamma=0.05, calibration_scores=[0.1, 0.3, 0.5, 0.7])
    >>> s = aci.predict_set([0.7, 0.2, 0.1])
    >>> bool(s[0])                   # the most likely answer is in the set
    True
    >>> aci.update([0.7, 0.2, 0.1], 1)
    """

    def __init__(self, alpha=0.1, gamma=0.01, calibration_scores=None):
        if not 0 < alpha < 1 or gamma <= 0:
            raise ValueError('alpha must be in (0, 1) and gamma positive')
        self.alpha, self.gamma = alpha, gamma
        self.alpha_t = alpha
        self.scores = [float(s) for s in (calibration_scores if calibration_scores is not None else [])]
        self.errors = []

    @property
    def threshold(self):
        """float: current threshold on the score (inf: every answer, -inf: none)."""
        if self.alpha_t <= 0 or not self.scores:
            return np.inf
        if self.alpha_t >= 1:
            return -np.inf
        n = len(self.scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha_t)))
        return float(np.sort(self.scores)[k - 1]) if k <= n else np.inf

    def predict_set(self, prob):
        """Answers in the prediction set.

        Parameters
        ----------
        prob : array_like
            Predicted probability of each answer.

        Returns
        -------
        numpy.ndarray
            Boolean mask over answers.
        """
        return 1 - np.asarray(prob, float) <= self.threshold

    def should_ask(self, prob):
        """bool: True when more than one answer is in the set."""
        return int(np.sum(self.predict_set(prob))) > 1

    def update(self, prob, answer):
        """Record the observed answer and correct the working level.

        Parameters
        ----------
        prob : array_like
            The probabilities used for this answer's prediction set.
        answer : int
            Index of the observed answer.
        """
        p = np.asarray(prob, float)
        err = int(not self.predict_set(p)[int(answer)])
        self.errors.append(err)
        self.alpha_t += self.gamma * (self.alpha - err)
        self.scores.append(float(1 - p[int(answer)]))

    @property
    def miscoverage(self):
        """float: fraction of answers that fell outside their sets so far."""
        return float(np.mean(self.errors)) if self.errors else 0.0

    def bound(self):
        """float: the guaranteed bound on ``|miscoverage - alpha|`` after the answers so far."""
        T = max(len(self.errors), 1)
        return (max(self.alpha, 1 - self.alpha) + self.gamma) / (self.gamma * T)

