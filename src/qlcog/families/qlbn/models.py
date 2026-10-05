"""Quantum-like Bayesian networks (Moreira and Wichert, 2014, 2016).

A discrete Bayesian network gives every full configuration x a classical probability P(x). The
quantum-like network gives it an amplitude sqrt(P(x)) e^{i theta(x)}. When unobserved (hidden)
variables are marginalised, amplitudes rather than probabilities are summed:

    P_q(query = v | evidence) = alpha * | sum_h sqrt(P(v, evidence, h)) e^{i theta(h)} |^2

with alpha normalising over the values v of the query. Interference terms are proportional to the
cosine of the phase differences. With two hidden configurations a phase difference of pi/2 removes
them and gives classical inference; with more configurations no choice of phases removes all of them
in general, so the classical network is kept as a separate baseline.

Classes
  BayesNet               discrete network: variables with values, parents and conditional tables.
  classical_marginal     exact classical inference by enumeration.
  quantum_like_marginal  quantum-like inference with one phase per hidden configuration.
  QLBNModel              fit-able model: phases are parameters, the network is fixed (options).
  ClassicalBNModel       baseline with no free parameters (the network as given)."""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param


class BayesNet:
    """variables: {name: [values]}; parents: {name: [parent names]};
    cpt: {name: function(value, parent_values_dict) -> probability} or nested dicts
    {name: {tuple(parent values): {value: prob}}}."""
    def __init__(self, variables, parents, cpt):
        self.variables, self.parents, self.cpt = variables, parents, cpt
        self.order = self._topological()

    def _topological(self):
        done, order = set(), []
        while len(order) < len(self.variables):
            for v in self.variables:
                if v not in done and all(p in done for p in self.parents.get(v, [])):
                    order.append(v); done.add(v)
        return order

    def prob(self, var, value, assignment):
        """Conditional probability of var = value given the parents in `assignment`."""
        t = self.cpt[var]
        pv = tuple(assignment[p] for p in self.parents.get(var, []))
        return t(value, dict(zip(self.parents.get(var, []), pv))) if callable(t) else t[pv][value]

    def joint(self, assignment):
        """Joint probability of a full assignment (product of the conditional tables)."""
        p = 1.0
        for v in self.order:
            p *= self.prob(v, assignment[v], assignment)
        return p

    def configurations(self, names):
        """All assignments of the variables in `names`."""
        for vals in itertools.product(*(self.variables[n] for n in names)):
            yield dict(zip(names, vals))


def _hidden(net, query, evidence):
    return [v for v in net.order if v != query and v not in evidence]


def classical_marginal(net, query, evidence=None):
    """Classical Bayesian-network marginal of `query` given `evidence` (sum over hidden configurations)."""
    evidence = evidence or {}
    hid = _hidden(net, query, evidence); out = {}
    for val in net.variables[query]:
        out[val] = sum(net.joint({**evidence, **h, query: val}) for h in net.configurations(hid))
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def quantum_like_marginal(net, query, evidence=None, phases=None):
    """phases: sequence with one phase per hidden configuration (in itertools.product order of the
    hidden variables' values); default all zero."""
    evidence = evidence or {}
    hid = _hidden(net, query, evidence)
    configs = list(net.configurations(hid))
    phases = np.zeros(len(configs)) if phases is None else np.asarray(phases, float)
    out = {}
    for val in net.variables[query]:
        amp = sum(np.sqrt(net.joint({**evidence, **h, query: val})) * np.exp(1j * th) for h, th in zip(configs, phases))
        out[val] = abs(amp) ** 2
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def n_hidden_configurations(net, query, evidence=None):
    """Number of configurations of the hidden (non-query, non-evidence) variables, i.e. the number of
    interference phases."""
    return int(np.prod([len(net.variables[v]) for v in _hidden(net, query, evidence or {})]))


class QLBNModel(Model):
    """Options: net (BayesNet), query (variable), conditions {name: evidence dict}.
    Parameters: phases theta_1 .. theta_{H-1} of the hidden configurations (theta_0 = 0); H <= 8.
    Design: condition names. Prediction: distribution of the query in each condition."""
    PARAMS = [Param('theta%d' % i, 'angle', 0.0) for i in range(1, 8)]
    name = 'Quantum-like Bayesian network'

    def predict(self, design=None):
        net, q, conds = self.options['net'], self.options['query'], self.options['conditions']
        out = {}
        for name in (design or list(conds)):
            ev = conds[name]
            H = n_hidden_configurations(net, q, ev)
            ph = np.r_[0.0, [getattr(self, 'theta%d' % i) for i in range(1, H)]] if H > 1 else np.zeros(1)
            m = quantum_like_marginal(net, q, ev, ph)
            out[name] = np.array([m[v] for v in net.variables[q]])
        return out


class ClassicalBNModel(Model):
    """Classical baseline for QLBNModel: the same network without interference."""
    PARAMS = []
    name = 'Classical Bayesian network'

    def predict(self, design=None):
        net, q, conds = self.options['net'], self.options['query'], self.options['conditions']
        return {n: np.array([classical_marginal(net, q, conds[n])[v] for v in net.variables[q]]) for n in (design or list(conds))}


def for_network(net, query, conditions):
    """QLBNModel subclass with exactly one free phase per extra hidden configuration (the largest
    number over the conditions), so that information criteria count only the phases in use."""
    H = max(n_hidden_configurations(net, query, ev) for ev in conditions.values())
    if H > 8:
        raise ValueError('at most 8 hidden configurations are supported')
    return type('QLBNModel', (QLBNModel,), {'PARAMS': QLBNModel.PARAMS[:max(H - 1, 0)]})
