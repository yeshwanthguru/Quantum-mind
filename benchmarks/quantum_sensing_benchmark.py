"""Does a slow, almost drift-free quantum accelerometer help a robot's dead reckoning? A simulation study.

One-axis robot, unknown acceleration (Ornstein-Uhlenbeck, std 0.5 m/s^2, correlation time 2 s), 300 s, 10 trajectories.
Sensors: a MEMS accelerometer (100 Hz, drifting bias) and an atom-interferometer accelerometer whose noise comes from the
standard quantum limit (1e6 atoms, T = 0.1 s, 1 s cycle, bias walk assumed). A multi-rate Kalman filter with the true noise
parameters fuses whichever sensors a suite contains.

Tables
  A  three suites (MEMS only, quantum only, fused) without and with a position fix every 10 s;
  B  the fused suite when the quantum sensor's bias (walk and initial bias) grows from 1e-4 of the MEMS bias to equal it;
  C  the fused suite when the quantum sensor's cycle time grows (a slower sensor).
Each fused row is compared with MEMS-only on the same trajectories with a paired sign-flip test.

The sensor numbers are illustrative assumptions (see quantum_mind.sensing.sensors); the study shows how the benefit depends on
them. Usage: python benchmarks/quantum_sensing_benchmark.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_mind.sensing import (atom_interferometer_accelerometer, compare_sensor_suites, mems_accelerometer)


def sign_flip_p(diff, n_perm=4000, seed=0):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    obs = abs(np.mean(diff))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def fmt(m):
    return '%.3f ± %.3f' % m


def run(quick=False):
    seeds = range(3 if quick else 10)
    dur = 60.0 if quick else 300.0
    mems = mems_accelerometer()
    q = atom_interferometer_accelerometer()
    out = {'a': [], 'b': [], 'c': []}
    suites = {'MEMS only': [mems.name], 'quantum only': [q.name], 'fused (MEMS + quantum)': [mems.name, q.name]}
    for fix in (None, 10.0):
        rows = compare_sensor_suites([mems, q], suites, duration=dur, seeds=seeds, fix_period=fix)
        base = next(r for r in rows if r['suite'] == 'MEMS only')['rmse_per_seed']
        for r in rows:
            d = np.array(r['rmse_per_seed']) - np.array(base)
            out['a'].append(dict(fix=fix, suite=r['suite'], rmse=r['rmse'], final=r['final_error'],
                                 p_vs_mems=None if r['suite'] == 'MEMS only' else sign_flip_p(d)))
    for frac in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        walk, bias0 = frac * mems.bias_walk, frac * mems.bias_std0
        qq = atom_interferometer_accelerometer(bias_walk=walk, bias_std0=bias0)
        rows = compare_sensor_suites([mems, qq], {'MEMS only': [mems.name], 'fused': [mems.name, qq.name]}, duration=dur,
                                     seeds=seeds)
        d = np.array(rows[1]['rmse_per_seed']) - np.array(rows[0]['rmse_per_seed'])
        out['b'].append(dict(bias_fraction=frac, bias_walk=walk, bias_std0=bias0, mems=rows[0]['rmse'],
                             fused=rows[1]['rmse'], p=sign_flip_p(d)))
    for cycle in (0.5, 1.0, 2.0, 5.0, 10.0):
        qq = atom_interferometer_accelerometer(cycle_time=cycle)
        rows = compare_sensor_suites([mems, qq], {'MEMS only': [mems.name], 'fused': [mems.name, qq.name]}, duration=dur,
                                     seeds=seeds)
        d = np.array(rows[1]['rmse_per_seed']) - np.array(rows[0]['rmse_per_seed'])
        out['c'].append(dict(cycle=cycle, mems=rows[0]['rmse'], fused=rows[1]['rmse'], p=sign_flip_p(d)))
    return out, mems, q, len(list(seeds)), dur


def to_markdown(out, mems, q, n_seeds, dur):
    L = ['# Quantum accelerometer dead-reckoning study\n',
         'Simulation, %d trajectories of %.0f s; position error in metres, mean ± std over trajectories. Sensor values are '
         'illustrative assumptions (see `quantum_mind.sensing.sensors`).\n' % (n_seeds, dur),
         '| Sensor | Rate (Hz) | White noise per sample (m/s²) | Bias walk ((m/s²)/√s) |', '|---|---|---|---|',
         '| %s | %g | %.3g | %.3g |' % (mems.name, mems.rate_hz, mems.noise_std, mems.bias_walk),
         '| %s | %g | %.3g | %.3g |' % (q.name, q.rate_hz, q.noise_std, q.bias_walk), '',
         '## A. Sensor suites\n',
         '| Position fixes | Suite | RMSE (m) | Final error (m) | Paired p vs MEMS only |', '|---|---|---|---|---|']
    for r in out['a']:
        L.append('| %s | %s | %s | %s | %s |' % ('every 10 s' if r['fix'] else 'none', r['suite'], fmt(r['rmse']), fmt(r['final']),
                                                '–' if r['p_vs_mems'] is None else '%.4f' % r['p_vs_mems']))
    L += ['', '## B. Fused suite when the quantum sensor is less stable (no fixes)\n',
          'The quantum sensor\'s bias walk and initial bias are set to a fraction of the MEMS values (its white noise stays '
          'at the quantum limit).\n',
          '| Quantum bias as a fraction of the MEMS bias | MEMS only RMSE (m) | Fused RMSE (m) | Paired p |', '|---|---|---|---|']
    for r in out['b']:
        L.append('| %g | %s | %s | %.4f |' % (r['bias_fraction'], fmt(r['mems']), fmt(r['fused']), r['p']))
    L += ['', '## C. Fused suite when the quantum sensor is slower (no fixes)\n',
          '| Quantum cycle time (s) | MEMS only RMSE (m) | Fused RMSE (m) | Paired p |', '|---|---|---|---|']
    for r in out['c']:
        L.append('| %g | %s | %s | %.4f |' % (r['cycle'], fmt(r['mems']), fmt(r['fused']), r['p']))
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=str(Path(__file__).parent / 'results' / 'quantum_sensing'))
    args = ap.parse_args()
    t0 = time.time()
    out, mems, q, n, dur = run(args.quick)
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'benchmark_results.md').write_text(to_markdown(out, mems, q, n, dur))
    (path / 'benchmark_raw.json').write_text(json.dumps(out, indent=1, default=float))
    print(to_markdown(out, mems, q, n, dur))
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
