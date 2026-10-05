"""Human-robot interaction: order-aware human models for robots that ask questions and track trust.

Contents
  HRI_DOMAINS              three question pairs a robot asks people, with illustrative population
                           parameters (object clarification, trust and hand-over, preference elicitation)
  domain_models(domain)    quantum-like, anchoring and Bayesian populations for a domain
  estimate_unprimed_rates  questioning designs for learning what people think before the robot's own
                           questions influence them (fixed order, reversed-order probe, split order)
  TRUST_PROTOCOL           hand-over outcomes and query designs for testing whether asking about trust
                           changes trust
  HumanModelEnsemble       competing human models weighted by evidence (BIC weights); gives the robot
                           a predicted answer distribution and its uncertainty, which can be passed to
                           an orchestrator as a confidence signal

The parameter values are illustrative (calibrated to answer rates of the size seen in survey data),
not estimates from human-robot interaction studies. The simulation study that uses them is in the
companion paper 'Order-Aware Human Models for Robot Questioning and Trust'."""
from __future__ import annotations

import numpy as np
from ..core import fit, compare
from ..families.order_effects import (QuantumOrderModel, QuantumOrderModel4D, AnchoringOrderModel, BayesOrderModel,
                                      RANK_STRUCTURES)
from ..families.dynamics import MarkovBelief, OpenSystemBelief

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
    """{'QL': ..., 'Anchoring': ..., 'Bayes': ...} for a domain. Bayes reproduces the QL model's A-first
    joint distribution without an order effect."""
    d = HRI_DOMAINS[domain]
    ql = QuantumOrderModel(**d['ql'])
    ab = ql.predict()['AB']; pA, pB = ab[0] + ab[1], ab[0] + ab[2]
    lo, hi = max(0, pA + pB - 1) - pA * pB, min(pA, pB) - pA * pB
    cov = ab[0] - pA * pB
    rho = cov / hi if cov > 0 else (-cov / lo if lo else 0.0)
    return {'QL': ql, 'Anchoring': AnchoringOrderModel(**d['anchoring']), 'Bayes': BayesOrderModel(pA=pA, pB=pB, rho=rho)}


def estimate_unprimed_rates(design, generator, n, rng=None, model=None, probe=0.1):
    """Estimate the first-position 'yes' rates (pi_A, pi_B) from n people who answer both questions.

    design 'fixed'  : everyone answers A then B; pi_B read from the second answers (ignores order effects)
           'probe'  : a fraction `probe` answers B first; `model` (class) fitted to all answers gives both
           'split'  : half each order; each rate from first answers only (model-free; robust default)"""
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


TRUST_EVENTS = (1, 1, 0, 1, 0, 1)                      # success, success, failure, success, failure, success
TRUST_PROTOCOL = {'end_only': (TRUST_EVENTS, (5,)), 'also_midway': (TRUST_EVENTS, (2, 5))}
TRUST_MODELS = {'open_system': OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3), 'markov': MarkovBelief}


class HumanModelEnsemble:
    """Competing human models weighted by evidence.

        ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel])
        ens.update(counts)                         # fit all models, compute BIC weights
        p, unc = ens.predict('AB')                 # mixture prediction and its uncertainty

    The uncertainty is the entropy of the mixture prediction (bits) plus the disagreement between models
    (Jensen-Shannon divergence weighted by the model weights). A robot can use it to decide whether to
    ask, which question to ask first, and how much to trust an answer, and can pass it to an
    orchestrator as a confidence signal."""

    def __init__(self, candidates=None, design=None):
        self.candidates = candidates or [QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]
        self.design = design; self.fits = []; self.weights = None

    def update(self, data, restarts=8, rng=None):
        """Fit every candidate model to `data` and set BIC weights; returns self."""
        self.fits = compare(self.candidates, data, self.design, restarts=restarts, rng=rng)
        b = np.array([f.bic for f in self.fits])
        w = np.exp(-0.5 * (b - b.min())); self.weights = w / w.sum()
        return self

    def predict(self, condition):
        """Mixture prediction for `condition` and its uncertainty {entropy_bits, model_disagreement_bits,
        total_bits}."""
        preds = np.array([np.asarray(f.model.predict(self.design)[condition], float) for f in self.fits])
        mix = self.weights @ preds
        H = lambda p: float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1)), 0)))
        js = H(mix) - float(sum(w * H(p) for w, p in zip(self.weights, preds)))
        return mix, {'entropy_bits': H(mix), 'model_disagreement_bits': js, 'total_bits': H(mix) + js}

    def summary(self):
        """One dict per model: name, ensemble weight and BIC."""
        return [{'model': type(f.model).__name__, 'weight': float(w), 'bic': f.bic} for f, w in zip(self.fits, self.weights)]


def ask_or_act(p_success, uncertainty=None, ask_cost=1.0, error_cost=5.0, max_disagreement_bits=0.05):
    """Decision rule for a robot that may ask a person before acting.

    p_success: probability that acting now is right (e.g. the ensemble's predicted 'yes' rate for the
    robot's current hypothesis). Acting costs error_cost * (1 - p_success) in expectation; asking costs
    ask_cost and is assumed to resolve the doubt. The robot also asks when the human models disagree
    by more than `max_disagreement_bits` (model uncertainty, from HumanModelEnsemble.predict), because
    then p_success itself is unreliable. Returns {'action': 'ask' | 'act', 'expected_cost_act': ...,
    'reason': ...}. Framework-agnostic: call it from a ROS 2 node, a behaviour tree or a planner."""
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
