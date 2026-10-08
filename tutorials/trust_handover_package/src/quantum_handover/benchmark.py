"""Benchmark: every policy against two populations of simulated people.

Each robot policy meets ``n_people`` simulated people from each population for one session of
``n_steps`` hand-overs. Its model is the population average, never the person's own parameters. The
report gives the mean session cost (lower is better) with its standard error, failures and questions
per session. All people are simulated; no real participants are involved.
"""
from __future__ import annotations

import numpy as np
from quantum_mind.applications.handover import simulate_handover_session

from .policies import POLICIES, make_policy, quantum_like_people, markov_people, person_state

__all__ = ['run_benchmark', 'to_markdown']

POPULATIONS = {'quantum-like people': quantum_like_people, 'Markov people': markov_people}


def run_benchmark(n_people=300, n_steps=20, seed=0, policies=POLICIES):
    """Run every policy against both populations.

    Parameters
    ----------
    n_people : int, optional
        Simulated people per population.
    n_steps : int, optional
        Hand-overs per session.
    seed : int, optional
        Seed of the populations and of the session outcomes.
    policies : sequence of str, optional

    Returns
    -------
    dict
        ``{population: {policy: {'cost', 'se', 'failures', 'asks'}}}``.
    """
    out = {}
    for k, (pop_name, sampler) in enumerate(POPULATIONS.items()):
        people = sampler(n_people, np.random.default_rng(seed + 1 + k))
        out[pop_name] = {}
        for name in policies:
            runs = [simulate_handover_session(make_policy(name), person_state(p), np.random.default_rng(seed * 100_003 + i),
                                              n_steps) for i, p in enumerate(people)]
            cost = np.array([r['cost'] for r in runs])
            out[pop_name][name] = {'cost': float(cost.mean()), 'se': float(cost.std(ddof=1) / np.sqrt(len(cost))),
                                   'failures': float(np.mean([r['failures'] for r in runs])),
                                   'asks': float(np.mean([r['asks'] for r in runs]))}
    return out


def to_markdown(result):
    """Markdown table of :func:`run_benchmark` output, best policy in bold for each population."""
    lines = ['| Simulated people | Policy | Mean session cost | Failures | Questions |', '|---|---|---|---|---|']
    for pop, rows in result.items():
        best = min(rows, key=lambda k: rows[k]['cost'])
        for name, r in rows.items():
            cell = '%.1f ± %.1f' % (r['cost'], r['se'])
            if name == best:
                cell = '**%s**' % cell
            lines.append('| %s | %s | %s | %.2f | %.1f |' % (pop, name, cell, r['failures'], r['asks']))
    return '\n'.join(lines)
