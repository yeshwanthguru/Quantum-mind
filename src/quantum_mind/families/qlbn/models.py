r"""Quantum-like Bayesian networks (Moreira and Wichert, 2014, 2016).

A discrete Bayesian network gives every full configuration :math:`x` a classical probability
:math:`P(x)`. The quantum-like network gives it an amplitude :math:`\sqrt{P(x)}\,e^{i\theta(x)}`.
When unobserved (hidden) variables are marginalised, amplitudes rather than probabilities are summed:

.. math:: P_q(\text{query} = v \mid e) = \alpha \,\Bigl|\sum_h \sqrt{P(v, e, h)}\,e^{i\theta_h}\Bigr|^2,

with :math:`\alpha` normalising over the values :math:`v`. Interference terms are proportional to the
cosine of the phase differences. With two hidden configurations a phase difference of :math:`\pi/2`
removes them and gives classical inference; with more configurations no choice of phases removes all
of them in general, so the classical network is kept as a separate baseline.

* :class:`BayesNet`: discrete network (variables, parents, conditional tables).
* :func:`classical_marginal`: exact classical inference by enumeration.
* :func:`quantum_like_marginal`: quantum-like inference with one phase per hidden configuration.
* :class:`QLBNModel`: fittable model whose parameters are the phases.
* :class:`ClassicalBNModel`: baseline with no free parameters.
"""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param


class BayesNet:
    """Discrete Bayesian network.

    Parameters
    ----------
    variables : dict
        ``{name: [values]}``.
    parents : dict
        ``{name: [parent names]}`` (variables without parents may be omitted).
    cpt : dict
        Conditional tables, per variable either a function ``f(value, parent_values_dict) -> probability``
        or nested dicts ``{tuple(parent values): {value: probability}}``.

    Attributes
    ----------
    order : list of str
        Variables in topological order (parents first).
    """
    def __init__(self, variables, parents, cpt):
        self.variables, self.parents, self.cpt = variables, parents, cpt
        self.order = self._topological()

    def _topological(self):
        """Order the variables so that every parent comes before its children."""
        done, order = set(), []
        while len(order) < len(self.variables):
            for v in self.variables:
                if v not in done and all(p in done for p in self.parents.get(v, [])):
                    order.append(v)
                    done.add(v)
        return order

    def prob(self, var, value, assignment):
        """Conditional probability of one variable.

        Parameters
        ----------
        var : str
            Variable name.
        value : object
            Value of ``var``.
        assignment : dict
            Values of (at least) the parents of ``var``.

        Returns
        -------
        float
            :math:`P(\\text{var} = \\text{value} \\mid \\text{parents})`.
        """
        t = self.cpt[var]
        pv = tuple(assignment[p] for p in self.parents.get(var, []))
        return t(value, dict(zip(self.parents.get(var, []), pv))) if callable(t) else t[pv][value]

    def joint(self, assignment):
        """Joint probability of a full assignment.

        Parameters
        ----------
        assignment : dict
            A value for every variable.

        Returns
        -------
        float
            Product of the conditional tables.
        """
        p = 1.0
        for v in self.order:
            p *= self.prob(v, assignment[v], assignment)
        return p

    def configurations(self, names):
        """Iterate over all assignments of some variables.

        Parameters
        ----------
        names : sequence of str
            Variables to enumerate.

        Yields
        ------
        dict
            ``{name: value}``, in :func:`itertools.product` order.
        """
        for vals in itertools.product(*(self.variables[n] for n in names)):
            yield dict(zip(names, vals))


def _hidden(net, query, evidence):
    """Variables that are neither the query nor observed."""
    return [v for v in net.order if v != query and v not in evidence]


def classical_marginal(net, query, evidence=None):
    """Classical posterior of one variable by enumeration.

    Parameters
    ----------
    net : BayesNet
        The network.
    query : str
        Query variable.
    evidence : dict, optional
        Observed values ``{variable: value}``.

    Returns
    -------
    dict
        ``{value: probability}`` of the query.
    """
    evidence = evidence or {}
    hid = _hidden(net, query, evidence)
    out = {}
    for val in net.variables[query]:
        out[val] = sum(net.joint({**evidence, **h, query: val}) for h in net.configurations(hid))
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def quantum_like_marginal(net, query, evidence=None, phases=None):
    """Quantum-like posterior of one variable (amplitudes summed over hidden configurations).

    Parameters
    ----------
    net : BayesNet
        The network.
    query : str
        Query variable.
    evidence : dict, optional
        Observed values.
    phases : array_like, optional
        One phase per hidden configuration, in :func:`itertools.product` order of the hidden
        variables' values. Default all zero.

    Returns
    -------
    dict
        ``{value: probability}`` of the query.
    """
    evidence = evidence or {}
    hid = _hidden(net, query, evidence)
    configs = list(net.configurations(hid))
    phases = np.zeros(len(configs)) if phases is None else np.asarray(phases, float)
    out = {}
    for val in net.variables[query]:
        # sum of amplitudes, not of probabilities: this is where interference enters
        amp = sum(np.sqrt(net.joint({**evidence, **h, query: val})) * np.exp(1j * th) for h, th in zip(configs, phases))
        out[val] = abs(amp) ** 2
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def n_hidden_configurations(net, query, evidence=None):
    """Number of hidden configurations, i.e. of interference phases.

    Parameters
    ----------
    net : BayesNet
        The network.
    query : str
        Query variable.
    evidence : dict, optional
        Observed values.

    Returns
    -------
    int
    """
    return int(np.prod([len(net.variables[v]) for v in _hidden(net, query, evidence or {})]))


class QLBNModel(Model):
    """Fittable quantum-like Bayesian network.

    The network is fixed; the parameters are the phases of the hidden configurations.

    Parameters
    ----------
    theta1, ..., theta7 : float
        Phases :math:`\\theta_1, \\dots, \\theta_{H-1}` (:math:`\\theta_0 = 0`); at most 8 hidden
        configurations. Use :func:`for_network` to fit only the phases in use.
    net : BayesNet
        The network (option).
    query : str
        Query variable (option).
    conditions : dict
        ``{condition name: evidence dict}`` (option). The design is a list of condition names.
    """
    PARAMS = [Param('theta%d' % i, 'angle', 0.0) for i in range(1, 8)]
    name = 'Quantum-like Bayesian network'

    def predict(self, design=None):
        """Distribution of the query variable in each condition."""
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
    """Classical baseline for :class:`QLBNModel`: the same network without interference.

    Takes the same options (``net``, ``query``, ``conditions``) and has no free parameters.
    """
    PARAMS = []
    name = 'Classical Bayesian network'

    def predict(self, design=None):
        """Distribution of the query variable in each condition."""
        net, q, conds = self.options['net'], self.options['query'], self.options['conditions']
        return {n: np.array([classical_marginal(net, q, conds[n])[v] for v in net.variables[q]]) for n in (design or list(conds))}


def for_network(net, query, conditions):
    """:class:`QLBNModel` subclass with exactly the phases a network needs.

    Information criteria then count only the phases in use (the largest number of hidden
    configurations over the conditions, minus one).

    Parameters
    ----------
    net : BayesNet
        The network.
    query : str
        Query variable.
    conditions : dict
        ``{name: evidence}``.

    Returns
    -------
    type
        Subclass of :class:`QLBNModel`.

    Raises
    ------
    ValueError
        If more than 8 hidden configurations are needed.
    """
    H = max(n_hidden_configurations(net, query, ev) for ev in conditions.values())
    if H > 8:
        raise ValueError('at most 8 hidden configurations are supported')
    return type('QLBNModel', (QLBNModel,), {'PARAMS': QLBNModel.PARAMS[:max(H - 1, 0)]})
