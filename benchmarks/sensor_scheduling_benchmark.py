"""Scheduling a quantum accelerometer with QUBO solvers (thread 1 applied to thread 2), against strong baselines (simulation).

Each mission is a random draw: slots spread over a 120 s window, some unusable (low fringe contrast), and two phases where
position accuracy
matters most. The chosen slots are scored by the exact weighted RMS position standard deviation of the Kalman filter that fuses
the MEMS and
the quantum accelerometer (lower is better), which a Monte Carlo of the real filter reproduces to within about 1%.

Setting A (14 slots, choose 4): small enough to enumerate every schedule, so the exact optimum and each method's regret are known.
Setting B (24 slots, choose 6): too big to enumerate; methods are compared with the greedy schedule on the exact objective.

Methods: uniform spacing; random; greedy on the exact objective; a swap local search from the uniform schedule (a classical local
search that
uses no QUBO); the plain inclusion-exclusion QUBO from the empty schedule; and the local-quadratic QUBO iteration started from the uniform
schedule and solved by exact enumeration, simulated annealing, greedy descent, random sampling with the same number of shots (the null
baseline for any sampling solver) and Qiskit QAOA (A only, exact simulator). Paired sign-flip tests on the per-mission objective.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from quantum_mind.sensing.scheduling import random_mission, schedule_interrogations

warnings.simplefilter('ignore')


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def run_setting(name, n_slots, k, missions, with_optimum, with_qaoa, with_exact_qubo):
    rows = {}
    t0 = time.time()
    for seed in missions:
        pr = random_mission(seed, n_slots=n_slots, k=k)
        res = {'uniform spacing': pr.uniform().objective, 'random (mean of 5)': float(np.mean([pr.random(s).objective for s in
        range(5)])),
               'greedy on the exact objective': pr.greedy().objective, 'swap local search from uniform': _swap(pr)}
        if with_optimum:
            res['exact optimum'] = pr.optimum().objective
        empty_solver = 'exact' if with_exact_qubo else 'annealing'
        res['QUBO from the empty schedule (1 round)'] = schedule_interrogations(pr, empty_solver, start='empty', rounds=1).objective
        if with_exact_qubo:
            res['local QUBO + exact solver'] = schedule_interrogations(pr, 'exact').objective
        res['local QUBO + annealing'] = schedule_interrogations(pr, 'annealing').objective
        res['local QUBO + greedy descent'] = schedule_interrogations(pr, 'greedy descent').objective
        res['local QUBO + random sampling (1000 shots)'] = schedule_interrogations(pr, 'random sampling', shots=1000).objective
        if with_qaoa:
            res['local QUBO + QAOA (depth 2)'] = schedule_interrogations(pr, 'qaoa', rounds=2, depth=2, restarts=2, maxiter=80).objective
        for m, v in res.items():
            rows.setdefault(m, {})[seed] = float(v)
        print(name, 'mission', seed, 'done %.0fs' % (time.time() - t0), flush=True)
    return rows


def _swap(pr):
    from quantum_mind.sensing.scheduling import _swap_search
    return pr.objective(_swap_search(pr, pr.uniform().slots)[0])


def table(title, rows, ref_name, note):
    methods = list(rows)
    seeds = sorted(set.intersection(*[set(r) for r in rows.values()]))
    ref = np.array([rows[ref_name][s] for s in seeds])
    uni = np.array([rows['uniform spacing'][s] for s in seeds])
    L = ['## ' + title, '', note, '',
         '| Method | Mean objective (m) | Mean regret vs %s | Missions where it matches or beats it | p vs uniform spacing |' % ref_name,
         '|---|---|---|---|---|']
    for m in methods:
        v = np.array([rows[m][s] for s in seeds])
        regret = np.mean(v / ref - 1) * 100
        wins = int(np.sum(v <= ref * (1 + 1e-6)))
        p = '–' if m == 'uniform spacing' else '%.4f' % sign_flip_p(v - uni)
        L.append('| %s | %.3f | %+.1f%% | %d / %d | %s |' % (m, v.mean(), regret, wins, len(seeds), p))
    return L + ['']


def run(quick=False):
    a = run_setting('A', 14, 4, range(4 if quick else 20), True, True, True)
    b = run_setting('B', 24, 6, range(3 if quick else 10), False, False, False)
    return a, b


def to_markdown(a, b):
    L = ['# Scheduling a quantum accelerometer with QUBO solvers', '',
         'Simulation. Objective: weighted RMS position standard deviation over a 120 s mission (m), from the exact Kalman covariance; '
         'lower is better. Sensor values are the library\'s illustrative models.', '']
    L += table('A. 14 slots, choose 4 (the exact optimum is known)', a, 'exact optimum',
               'Regret is the mean relative excess over the exact optimum found by enumerating every schedule.')
    L += table('B. 24 slots, choose 6 (too big to enumerate)', b, 'greedy on the exact objective',
               'Regret is relative to greedy selection on the exact objective, the best reference available at this size.')
    L += ['The local QUBO is a second-order model of the true objective, exact within two flips of its expansion point, so a solver',
          'that finds its exact optimum can still be beaten by a method that uses the exact objective. Solvers of the QUBO, quantum or',
          'classical, only trade accuracy of the QUBO solution for time.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/sensor_scheduling')
    args = ap.parse_args()
    a, b = run(args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_results.md').write_text(to_markdown(a, b))
    (out / 'benchmark_raw.json').write_text(json.dumps({'A': a, 'B': b}, indent=1))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
