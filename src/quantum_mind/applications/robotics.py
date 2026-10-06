"""Human-robot interaction: order-aware human models for robots that ask questions and track trust.

* :data:`HRI_DOMAINS`: three question pairs a robot asks people, with illustrative population
  parameters (object clarification, trust and hand-over, preference elicitation).
* :func:`domain_models`: quantum-like, anchoring and Bayesian populations for a domain.
* :func:`estimate_unprimed_rates`: questioning designs for learning what people think before the
  robot's own questions influence them (fixed order, reversed-order probe, split order).
* :data:`TRUST_PROTOCOL`: hand-over outcomes and query designs for testing whether asking about trust
  changes trust.
* :class:`HumanModelEnsemble`: competing human models weighted by evidence (BIC weights); gives the
  robot a predicted answer distribution and its uncertainty, which can be passed to an orchestrator
  as a confidence signal.
* :func:`ask_or_act`: decision rule for asking a person before acting.
* :class:`HumanModelService`: middleware-independent service wrapping the ensemble (used by the ROS 2
  node in ``integrations/ros2``).

The parameter values are illustrative (calibrated to answer rates of the size seen in survey data),
not estimates from human-robot interaction studies. The simulation study that uses them is in the
companion paper 'Order-Aware Human Models for Robot Questioning and Trust'."""
from __future__ import annotations

import numpy as np
from ..core import fit, compare
from ..families.order_effects import (QuantumOrderModel, QuantumOrderModel4D, AnchoringOrderModel, BayesOrderModel,
                                      RANK_STRUCTURES)
from ..families.dynamics import MarkovBelief, OpenSystemBelief

#: Question pairs and illustrative population parameters for three human-robot interaction domains.
HRI_DOMAINS = {
    'object_clarification': dict(A='Is it the red cup?', B='Is it the cup on the left?',
                                 ql=dict(a=2.3543, b=0.9676, g=0.5846, ranks=(1, 2)),
                                 anchoring=dict(pA=0.50, pB=0.68, wA=0.3889, wB=0.4444)),
    'trust_handover': dict(A='Do you trust the robot to hand you the tool?', B='Was the last hand-over safe?',
                           ql=dict(a=0.6602, b=2.0304, g=0.6955, ranks=(1, 2)),
                           anchoring=dict(pA=0.62, pB=0.80, wA=0.3333, wB=0.2778)),
    'preference_elicitation': dict(A='Is trajectory 1 acceptable?', B='Is trajectory 2 acceptable?',
                                   ql=dict(a=0.7371, b=0.9047, g=2.3389, ranks=(1, 2)),
                                   anchoring=dict(pA=0.55, pB=0.62, wA=-0.3115, wB=0.0)),
}


def domain_models(domain):
    """Simulated populations for one domain.

    Parameters
    ----------
    domain : str
        Key of :data:`HRI_DOMAINS`.

    Returns
    -------
    dict
        ``{'QL': QuantumOrderModel, 'Anchoring': AnchoringOrderModel, 'Bayes': BayesOrderModel}``. The
        Bayesian population reproduces the quantum-like model's A-first joint distribution without an
        order effect.
    """
    d = HRI_DOMAINS[domain]
    ql = QuantumOrderModel(**d['ql'])
    ab = ql.predict()['AB']
    pA, pB = ab[0] + ab[1], ab[0] + ab[2]
    # express the A-first covariance as a correlation between the Frechet bounds
    lo, hi = max(0, pA + pB - 1) - pA * pB, min(pA, pB) - pA * pB
    cov = ab[0] - pA * pB
    rho = cov / hi if cov > 0 else (-cov / lo if lo else 0.0)
    return {'QL': ql, 'Anchoring': AnchoringOrderModel(**d['anchoring']), 'Bayes': BayesOrderModel(pA=pA, pB=pB, rho=rho)}


def estimate_unprimed_rates(design, generator, n, rng=None, model=None, probe=0.1):
    """Estimate the unprimed "yes" rates from people who answer both questions.

    The unprimed rate of a question is its "yes" rate when it is asked first, before the other
    question can influence the answer.

    Parameters
    ----------
    design : {'fixed', 'probe', 'split'}
        ``'fixed'``: everyone answers A then B, and the B rate is read from the second answers
        (ignores order effects). ``'probe'``: a fraction ``probe`` answers B first, and ``model``
        fitted to all answers gives both rates. ``'split'``: half answer each order and each rate
        comes from first answers only (model-free; the robust default).
    generator : Model
        Population that produces the answers (for example from :func:`domain_models`).
    n : int
        Number of people.
    rng : numpy.random.Generator, optional
        Random number generator.
    model : type, optional
        Model class fitted in the ``'probe'`` design.
    probe : float, optional
        Fraction asked in the reversed order in the ``'probe'`` design.

    Returns
    -------
    tuple of float
        Estimated ``(pi_A, pi_B)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if design == 'fixed':
        c = rng.multinomial(n, generator.predict()['AB'])
        return (c[0] + c[1]) / n, (c[0] + c[2]) / n
    if design == 'split':
        h = n // 2
        ab, ba = rng.multinomial(h, generator.predict()['AB']), rng.multinomial(n - h, generator.predict()['BA'])
        return (ab[0] + ab[1]) / h, (ba[0] + ba[1]) / (n - h)
    nb = max(1, int(round(probe * n)))
    data = {'AB': rng.multinomial(n - nb, generator.predict()['AB']), 'BA': rng.multinomial(nb, generator.predict()['BA'])}
    kw = {'structures': RANK_STRUCTURES} if model is QuantumOrderModel else {}
    m = fit(model, data, restarts=8, rng=rng, **kw).model.predict()
    return float(m['AB'][:2].sum()), float(m['BA'][:2].sum())


#: Hand-over outcomes of the trust protocol: success, success, failure, success, failure, success.
TRUST_EVENTS = (1, 1, 0, 1, 0, 1)                      # success, success, failure, success, failure, success
#: Query designs: ask about trust only at the end, or also midway.
TRUST_PROTOCOL = {'end_only': (TRUST_EVENTS, (5,)), 'also_midway': (TRUST_EVENTS, (2, 5))}
#: Illustrative trust models: a quantum-like open-system belief and the Markov baseline class.
TRUST_MODELS = {'open_system': OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3), 'markov': MarkovBelief}


class HumanModelEnsemble:
    """Competing human models weighted by evidence.

    The uncertainty is the entropy of the mixture prediction (bits) plus the disagreement between
    models (Jensen-Shannon divergence weighted by the model weights). A robot can use it to decide
    whether to ask, which question to ask first, and how much to trust an answer, and can pass it to
    an orchestrator as a confidence signal.

    Parameters
    ----------
    candidates : list of type, optional
        Model classes; default quantum-like (4D), Bayesian and anchoring.
    design : iterable, optional
        Design passed to the models.

    Attributes
    ----------
    fits : list of FitResult
        Fits from the last :meth:`update`, best first.
    weights : numpy.ndarray
        BIC weights of the fits.

    Examples
    --------
    >>> ens = HumanModelEnsemble()                      # doctest: +SKIP
    >>> ens.update({'AB': [40, 10, 20, 30], 'BA': [30, 20, 10, 40]})   # doctest: +SKIP
    >>> p, unc = ens.predict('AB')                      # doctest: +SKIP
    """

    def __init__(self, candidates=None, design=None):
        self.candidates = candidates or [QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]
        self.design = design
        self.fits = []
        self.weights = None

    def update(self, data, restarts=8, rng=None):
        """Fit every candidate model and compute BIC weights.

        Parameters
        ----------
        data : dict
            Answer counts ``{'AB': [yy, yn, ny, nn], 'BA': [...]}``.
        restarts : int, optional
            Restarts per fit.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        HumanModelEnsemble
            ``self``.
        """
        self.fits = compare(self.candidates, data, self.design, restarts=restarts, rng=rng)
        b = np.array([f.bic for f in self.fits])
        w = np.exp(-0.5 * (b - b.min()))                # BIC (Schwarz) weights
        self.weights = w / w.sum()
        return self

    def predict(self, condition):
        """Mixture prediction and its uncertainty.

        Parameters
        ----------
        condition : str
            Condition, for example ``'AB'``.

        Returns
        -------
        prediction : numpy.ndarray
            Weighted mixture of the models' predictions.
        uncertainty : dict
            ``entropy_bits``, ``model_disagreement_bits`` (weighted Jensen-Shannon divergence) and
            ``total_bits``.
        """
        preds = np.array([np.asarray(f.model.predict(self.design)[condition], float) for f in self.fits])
        mix = self.weights @ preds
        H = lambda p: float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1)), 0)))
        js = H(mix) - float(sum(w * H(p) for w, p in zip(self.weights, preds)))     # Jensen-Shannon
        return mix, {'entropy_bits': H(mix), 'model_disagreement_bits': js, 'total_bits': H(mix) + js}

    def summary(self):
        """Ensemble weights.

        Returns
        -------
        list of dict
            One dict per model with ``model``, ``weight`` and ``bic``.
        """
        return [{'model': type(f.model).__name__, 'weight': float(w), 'bic': f.bic} for f, w in zip(self.fits, self.weights)]


def ask_or_act(p_success, uncertainty=None, ask_cost=1.0, error_cost=5.0, max_disagreement_bits=0.05):
    """Decision rule for a robot that may ask a person before acting.

    Acting costs ``error_cost * (1 - p_success)`` in expectation; asking costs ``ask_cost`` and is
    assumed to resolve the doubt. The robot also asks when the human models disagree by more than
    ``max_disagreement_bits``, because then ``p_success`` itself is unreliable. The rule is
    framework-agnostic: call it from a ROS 2 node, a behaviour tree or a planner.

    Parameters
    ----------
    p_success : float
        Probability that acting now is right (for example the ensemble's predicted "yes" rate for the
        robot's current hypothesis).
    uncertainty : dict, optional
        Output of :meth:`HumanModelEnsemble.predict`; its ``model_disagreement_bits`` is used.
    ask_cost : float, optional
        Cost of asking.
    error_cost : float, optional
        Cost of acting wrongly.
    max_disagreement_bits : float, optional
        Model disagreement above which the robot always asks.

    Returns
    -------
    dict
        ``action`` (``'ask'`` or ``'act'``), ``expected_cost_act`` and a human-readable ``reason``.

    Raises
    ------
    ValueError
        If ``p_success`` is not in [0, 1].

    Examples
    --------
    >>> from quantum_mind.applications.robotics import ask_or_act
    >>> ask_or_act(0.9, ask_cost=1.0, error_cost=5.0)['action']
    'act'
    >>> ask_or_act(0.6, ask_cost=1.0, error_cost=5.0)['action']
    'ask'
    """
    if not 0.0 <= p_success <= 1.0:
        raise ValueError('p_success must be a probability')
    cost_act = float(error_cost * (1.0 - p_success))
    disagreement = (uncertainty or {}).get('model_disagreement_bits', 0.0)
    if disagreement > max_disagreement_bits:
        return {'action': 'ask', 'expected_cost_act': cost_act,
                'reason': 'human models disagree (%.3f bits)' % disagreement}
    action = 'ask' if cost_act > ask_cost else 'act'
    reason = 'expected error cost %.2f %s asking cost %.2f' % (cost_act, '>' if action == 'ask' else '<=', ask_cost)
    return {'action': action, 'expected_cost_act': cost_act, 'reason': reason}


#: Default cost of a wrong action by hazard level (in units of the cost of asking once).
HAZARD_COSTS = {'low': 2.0, 'medium': 10.0, 'high': 100.0, 'critical': 1000.0}


def risk_aware_ask_or_act(p_success, hazard='medium', ask_cost=1.0, p_lower=None, uncertainty=None,
                          max_disagreement_bits=0.05, max_risk=None):
    """Ask-or-act rule that scales with the hazard of the action and uses a lower confidence bound.

    Handing over a cup and handing over a knife need different thresholds. The cost of a wrong action is
    taken from the hazard level, and when a lower bound on the success probability is available (for
    example from :class:`~quantum_mind.applications.calibration.CalibrationMonitor` or a conformal predictor)
    the decision uses it instead of the point estimate, so an unreliable confidence makes the robot ask.

    Parameters
    ----------
    p_success : float
        Estimated probability that acting now is right.
    hazard : str or float, optional
        A key of :data:`HAZARD_COSTS` (``'low'``, ``'medium'``, ``'high'``, ``'critical'``) or a numeric
        cost of a wrong action.
    ask_cost : float, optional
        Cost of asking.
    p_lower : float, optional
        Lower confidence bound on the success probability; used in place of ``p_success`` when given.
    uncertainty : dict, optional
        Output of :meth:`HumanModelEnsemble.predict`; disagreement between human models forces asking.
    max_disagreement_bits : float, optional
        Disagreement threshold.
    max_risk : float, optional
        Hard limit on the failure probability: above it the robot asks whatever the costs.

    Returns
    -------
    dict
        The fields of :func:`ask_or_act`, plus ``hazard_cost`` and ``p_used``.

    Examples
    --------
    >>> from quantum_mind.applications.robotics import risk_aware_ask_or_act
    >>> risk_aware_ask_or_act(0.95, 'low')['action']          # cup: act
    'act'
    >>> risk_aware_ask_or_act(0.95, 'high')['action']         # knife: ask
    'ask'
    """
    cost = HAZARD_COSTS[hazard] if isinstance(hazard, str) else float(hazard)
    p = float(p_success if p_lower is None else min(p_success, p_lower))
    d = ask_or_act(p, uncertainty, ask_cost, cost, max_disagreement_bits)
    if max_risk is not None and 1 - p > max_risk and d['action'] == 'act':
        d = {'action': 'ask', 'expected_cost_act': d['expected_cost_act'],
             'reason': 'failure risk %.3f above the limit %.3f' % (1 - p, max_risk)}
    d.update(hazard_cost=cost, p_used=p)
    return d


class HumanModelService:
    """Middleware-independent human-model service a robot can run next to its planner.

    It accumulates people's answers to two questions asked in either order, refits a
    :class:`HumanModelEnsemble` every ``refit_every`` answers, and returns a prediction with its
    uncertainty and an ask-or-act decision. Messages are plain, JSON-friendly dicts. The ROS 2 node in
    ``integrations/ros2`` wraps this class.

    Parameters
    ----------
    candidates : list of type, optional
        Model classes of the ensemble.
    refit_every : int, optional
        Number of new answers between refits.
    min_answers : int, optional
        Answers needed before the first fit.
    restarts : int, optional
        Restarts per fit.
    seed : int, optional
        Seed of the fitting random number generator.

    Examples
    --------
    >>> from quantum_mind.applications.robotics import HumanModelService
    >>> svc = HumanModelService()
    >>> svc.add_answer({'order': 'AB', 'answers': [1, 0]})     # first answer, second answer (1 = yes)
    {'n_answers': 1, 'refitted': False}
    >>> svc.query({'order': 'AB'})['action']                   # too little data yet
    'ask'
    """

    ORDERS = ('AB', 'BA')

    def __init__(self, candidates=None, refit_every=20, min_answers=20, restarts=4, seed=0):
        self.candidates = candidates or [QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]
        self.refit_every, self.min_answers, self.restarts = refit_every, min_answers, restarts
        self.counts = {o: np.zeros(4, int) for o in self.ORDERS}
        self.ensemble = None
        self._since = 0                                  # answers since the last refit
        self._rng = np.random.default_rng(seed)

    @property
    def n_answers(self):
        """int: number of answer pairs received."""
        return int(sum(c.sum() for c in self.counts.values()))

    def add_answer(self, msg):
        """Record one person's answers.

        Parameters
        ----------
        msg : dict
            ``{'order': 'AB' or 'BA', 'answers': [first, second]}`` with 1 = yes and 0 = no, in the
            order asked.

        Returns
        -------
        dict
            ``{'n_answers': int, 'refitted': bool}``.

        Raises
        ------
        ValueError
            If the message is malformed.
        """
        order = msg.get('order')
        ans = msg.get('answers')
        if order not in self.ORDERS or not isinstance(ans, (list, tuple)) or len(ans) != 2 or any(a not in (0, 1) for a in ans):
            raise ValueError("expected {'order': 'AB' or 'BA', 'answers': [0 or 1, 0 or 1]}")
        self.counts[order][{(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}[tuple(int(a) for a in ans)]] += 1
        self._since += 1
        refit = False
        if self.n_answers >= self.min_answers and (self.ensemble is None or self._since >= self.refit_every):
            self.ensemble = HumanModelEnsemble(self.candidates).update(self.counts, restarts=self.restarts, rng=self._rng)
            self._since = 0
            refit = True
        return {'n_answers': self.n_answers, 'refitted': refit}

    def query(self, msg=None):
        """Predict answers and decide whether to ask.

        Parameters
        ----------
        msg : dict, optional
            ``{'order': 'AB', 'ask_cost': 1.0, 'error_cost': 5.0, 'max_disagreement_bits': 0.05}``
            (all keys optional).

        Returns
        -------
        dict
            ``prediction`` ``[yy, yn, ny, nn]``, ``p_first_yes``, ``uncertainty``, ``weights`` and the
            :func:`ask_or_act` fields. Before enough answers have arrived, ``action`` is ``'ask'``
            with the reason "not enough data".

        Raises
        ------
        ValueError
            If the order is not ``'AB'`` or ``'BA'``.
        """
        msg = msg or {}
        order = msg.get('order', 'AB')
        if order not in self.ORDERS:
            raise ValueError("order must be 'AB' or 'BA'")
        if self.ensemble is None:
            return {'n_answers': self.n_answers, 'action': 'ask', 'reason': 'not enough data (%d of %d answers)'
                    % (self.n_answers, self.min_answers)}
        p, unc = self.ensemble.predict(order)
        d = ask_or_act(float(p[0] + p[1]), unc, msg.get('ask_cost', 1.0), msg.get('error_cost', 5.0),
                       msg.get('max_disagreement_bits', 0.05))
        return {'n_answers': self.n_answers, 'order': order, 'prediction': [float(v) for v in p],
                'p_first_yes': float(p[0] + p[1]), 'uncertainty': {k: float(v) for k, v in unc.items()},
                'weights': {s['model']: s['weight'] for s in self.ensemble.summary()}, **d}
