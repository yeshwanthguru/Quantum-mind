"""Robot problems on quantum and classical solvers: does QAOA beat the classical baselines? A simulation study.

For three robot problems (multi-robot allocation 3x3 = 9 variables, grasp selection 10 choose 3, slot scheduling of 4 robots
in 3 slots = 12 variables) 10 random instances each are solved by exhaustive search (the optimum), greedy descent with
restarts, simulated annealing, random sampling with the same 1000 shots (the null baseline for a sampling solver) and QAOA
through Qiskit's `QAOAAnsatz` (exact simulation) at depths 1 to 3. Reported: the share
of instances solved to the true optimum, the mean gap to the optimum in the robot's own objective, and for QAOA the probability
mass on the optimum and the approximation ratio of its expected energy. QAOA sees exactly the same QUBO as the classical
solvers. Problems of this size are exactly solvable, so the study measures solution quality, not speed.

Usage: python benchmarks/robot_quantum_computing_benchmark.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from quantum_mind.computing import (grasp_selection, greedy_descent, multi_robot_allocation, qaoa, random_sampling,
                                    simulated_annealing, slot_scheduling)


def instances(kind, n, rng):
    out = []
    for _ in range(n):
        if kind == 'allocation (3 robots x 3 tasks)':
            out.append(multi_robot_allocation(rng.uniform(1, 9, (3, 3)), balance=0.3))
        elif kind == 'grasp selection (3 of 10)':
            c = rng.uniform(0, 0.6, (10, 10))
            out.append(grasp_selection(rng.uniform(0, 1, 10), (c + c.T) / 2, 3))
        else:
            a = np.triu(rng.random((4, 4)) < 0.7, 1)
            out.append(slot_scheduling((a | a.T).astype(int), 3))
    return out


KINDS = ['allocation (3 robots x 3 tasks)', 'grasp selection (3 of 10)', 'slot scheduling (4 robots, 3 slots)']


def run(quick=False):
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(2026)
    n_inst = 3 if quick else 10
    rows = []
    for kind in KINDS:
        for p in instances(kind, n_inst, rng):
            opt = p.optimum()[1]
            solvers = {'greedy descent': lambda q: greedy_descent(q, restarts=20, seed=1),
                       'simulated annealing': lambda q: simulated_annealing(q, seed=1),
                       'random sampling (1000 shots)': lambda q: random_sampling(q, 1000, seed=1)}
            for d in (1, 2, 3):
                solvers['QAOA p=%d' % d] = (lambda q, d=d: qaoa(q, depth=d, restarts=3, maxiter=150, seed=1))
            for name, solve in solvers.items():
                t0 = time.time()
                r = solve(p.qubo)
                ok = p.feasible(r.x)
                gap = (p.objective(r.x) - opt) if ok else np.nan
                rows.append(dict(kind=kind, solver=name, optimal=bool(ok and abs(gap) < 1e-9), feasible=bool(ok), gap=gap,
                                 span=max(abs(opt), 1.0), seconds=time.time() - t0,
                                 p_optimal=r.info.get('p_optimal', np.nan), ratio=r.info.get('ratio', np.nan)))
    return rows, n_inst


def to_markdown(rows, n_inst):
    L = ['# Robot problems on quantum and classical solvers\n',
         '%d random instances per problem; QAOA through Qiskit `QAOAAnsatz`, exact simulation, 3 restarts, '
         'best of 1000 samples (as is random sampling). '
         '"Optimal" = the returned solution reaches the exact optimum of the robot\'s own objective.\n' % n_inst,
         '| Problem | Solver | Optimal | Feasible | Mean gap | QAOA P(optimal) | QAOA approximation ratio | Seconds / instance |',
         '|---|---|---|---|---|---|---|---|']
    for kind in KINDS:
        for solver in ['greedy descent', 'simulated annealing', 'random sampling (1000 shots)', 'QAOA p=1', 'QAOA p=2', 'QAOA p=3']:
            sel = [r for r in rows if r['kind'] == kind and r['solver'] == solver]
            gaps = [r['gap'] for r in sel if not np.isnan(r['gap'])]
            popt = [r['p_optimal'] for r in sel if not np.isnan(r['p_optimal'])]
            ratio = [r['ratio'] for r in sel if not np.isnan(r['ratio'])]
            L.append('| %s | %s | %d/%d | %d/%d | %s | %s | %s | %.2f |' % (
                kind, solver, sum(r['optimal'] for r in sel), len(sel), sum(r['feasible'] for r in sel), len(sel),
                '%.3f' % np.mean(gaps) if gaps else '–', '%.3f' % np.mean(popt) if popt else '–',
                '%.3f' % np.mean(ratio) if ratio else '–', np.mean([r['seconds'] for r in sel])))
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=str(Path(__file__).parent / 'results' / 'robot_quantum_computing'))
    args = ap.parse_args()
    t0 = time.time()
    rows, n = run(args.quick)
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'benchmark_results.md').write_text(to_markdown(rows, n))
    (path / 'benchmark_raw.json').write_text(json.dumps(rows, indent=1, default=float))
    print(to_markdown(rows, n))
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
