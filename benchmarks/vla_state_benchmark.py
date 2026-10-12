"""State slot of the one-motor VLA: does the quantum accelerometer help the joint when the encoder drops out?

Simulation. The VLA's controller (TinyVLA acting on unambiguous instructions) is fed a state estimate instead of the encoder reading.
The encoder drops out 1 s into a 3 s move, and an unmodelled push (a constant external acceleration drawn uniformly up to the stated
size, in workspace units/s^2; the joint's own maximum acceleration is 8) acts on the joint. Estimators: raw encoder held at dropout;
Kalman filters using the encoder only, the nominal motor model plus the known command (the classical rival), the MEMS accelerometer,
and the MEMS accelerometer plus the slow quantum accelerometer. Sensor values are the library's illustrative models, scaled to a
0.2 m half-stroke linear stage. Paired sign-flip tests on per-episode final error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_mind.embodied import MODES, TinyVLA, make_dataset, run_sensed_episode, vla_plan


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def run(quick=False):
    seeds, n = range(2 if quick else 5), (60 if quick else 200)
    res = {d: {m: {'err': [], 'succ': []} for m in MODES} for d in (0.0, 2.0, 4.0)}
    for seed in seeds:
        vla = TinyVLA(seed=seed).fit(make_dataset(3000, seed), steps=3000, seed=seed)
        test = [s for s in make_dataset(4 * n, 500 + seed) if not s.ambiguous][:n]
        plan = vla_plan(vla)
        for d in res:
            for m in MODES:
                rng = np.random.default_rng(seed)
                rows = [run_sensed_episode(s, rng, plan, m, disturbance=d) for s in test]
                res[d][m]['err'].extend(r['error'] for r in rows)
                res[d][m]['succ'].extend(float(r['success']) for r in rows)
        print('seed', seed, flush=True)
    return dict(res=res, seeds=len(list(seeds)), n=n)


def to_markdown(o):
    L = ['# State slot of the one-motor VLA: quantum accelerometer under encoder dropout', '',
         'Simulation. %d seeds x %d unambiguous scenes. Encoder drops out at 1 s of a 3 s move. Success = final position within a quarter cell. '
         'p: paired sign-flip test on final error against "Kalman + MEMS + quantum".' % (o['seeds'], o['n']), '']
    for d, rows in o['res'].items():
        ref = np.array(rows['kalman + MEMS + quantum']['err'])
        L += ['## Unmodelled push up to %g units/s^2' % d, '', '| Estimator | Success | Mean final error | p vs quantum fusion |', '|---|---|---|---|']
        for m, v in rows.items():
            e = np.array(v['err'])
            p = '–' if m == 'kalman + MEMS + quantum' else '%.4f' % sign_flip_p(e - ref)
            L.append('| %s | %.3f | %.3f | %s |' % (m, np.mean(v['succ']), e.mean(), p))
        L.append('')
    L += ['The model-based Kalman filter uses the known command and the nominal plant; it is excellent when the plant is as modelled and fails '
          'when something unmodelled pushes the joint, which is where an accelerometer can help. The quantum accelerometer\'s edge over the MEMS one '
          'comes from its near-zero bias; with the encoder present, the encoder calibrates the MEMS bias and the gap is small.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/vla_state')
    a = ap.parse_args()
    o = run(a.quick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_results.md').write_text(to_markdown(o))
    (out / 'benchmark_raw.json').write_text(json.dumps({str(d): {m: {'succ': float(np.mean(v['succ'])), 'err': float(np.mean(v['err']))} for m, v in r.items()} for d, r in o['res'].items()}, indent=1))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
