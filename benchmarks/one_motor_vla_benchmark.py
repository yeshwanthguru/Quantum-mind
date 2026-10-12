"""One-motor VLA prototype: closed-loop study against an end-to-end behaviour-cloning baseline.

Everything is simulated (a DC-motor joint with a noisy encoder and a one-dimensional tabletop), so the numbers show what the
prototype does in this toy, not what a real robot would do. For each training seed, models are trained on scenes drawn from
the same generator and tested closed loop on fresh scenes.

Policies: oracle (PD controller that is told the intended cell); BC baseline (image + instruction + motor state to command, no
grounding); TinyVLA acting on its expected target; TinyVLA with a split-conformal ask gate; TinyVLA that always asks.
An instruction is *ambiguous* when it is true of more than one object ("go to the block"); the intended cell is then drawn
uniformly, so no policy can know it without asking. Success = final position within a quarter of a cell width of the intended cell.
The cost of a question is a free assumption: the table reports success and questions separately, plus success minus 0.1 per question.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_mind.embodied import (BCBaseline, TinyVLA, bc_plan, calibrate_gate, make_dataset, oracle_plan, run_episode,
                                   vla_plan)

ALPHAS = (0.1, 0.05, 0.02, 0.01)
HEAD_ALPHA = 0.02
Q_COST = 0.1


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def evaluate(plan, scenes, seed):
    rng = np.random.default_rng(seed)
    return [run_episode(s, rng, plan) for s in scenes]


def split(rows, scenes):
    out = {}
    for lab, f in (('all', lambda a: True), ('unambiguous', lambda a: not a), ('ambiguous', lambda a: a)):
        sel = [r for r, s in zip(rows, scenes) if f(s.ambiguous)]
        if not sel:
            continue
        out[lab] = dict(success=float(np.mean([r['success'] for r in sel])), right_object=float(np.mean([r['right_object'] for r in sel])),
                        error=float(np.mean([r['error'] for r in sel])), asks=float(np.mean([r['asked'] for r in sel])), n=len(sel))
    return out


def run(quick=False):
    seeds = range(2 if quick else 5)
    sizes = (250, 1000) if quick else (250, 1000, 4000, 16000)
    n_test, big = (150, 1000) if quick else (400, 4000)
    t0 = time.time()
    eff = {n: {'vla': [], 'bc': []} for n in sizes}            # success on unambiguous scenes vs training-set size
    head = {}                                                  # policy -> list of per-scene success per seed
    split_rows = {}
    param = {}
    for seed in seeds:
        test = make_dataset(n_test, 1000 + seed)
        cal = make_dataset(1000, 2000 + seed)
        for n in sizes:
            tr = make_dataset(n, seed)
            vla = TinyVLA(seed=seed).fit(tr, steps=3000, seed=seed)
            bc = BCBaseline(hidden=128, seed=seed).fit(tr, steps=8000, lr=0.003, seed=seed)
            param = dict(vla=vla.n_params, bc=bc.n_params)
            un = [s for s in test if not s.ambiguous]
            eff[n]['vla'].append(float(np.mean([r['success'] for r in evaluate(vla_plan(vla), un, seed)])))
            eff[n]['bc'].append(float(np.mean([r['success'] for r in evaluate(bc_plan(bc), un, seed)])))
            if n == big:
                plans = {'oracle': oracle_plan, 'BC baseline': bc_plan(bc), 'TinyVLA, act': vla_plan(vla),
                         'TinyVLA, always ask': vla_plan(vla, always_ask=True)}
                for a in ALPHAS:
                    plans['TinyVLA, ask gate a=%g' % a] = vla_plan(vla, calibrate_gate(vla, cal, a))
                for name, pl in plans.items():
                    rows = evaluate(pl, test, seed)
                    head.setdefault(name, []).extend(float(r['success']) - Q_COST * r['asked'] for r in rows)
                    split_rows.setdefault(name, []).append(split(rows, test))
        print('seed', seed, 'done %.0fs' % (time.time() - t0), flush=True)
    return dict(seeds=len(list(seeds)), sizes=list(sizes), big=big, n_test=n_test, eff=eff, split=split_rows,
                scored=head, params=param, seconds=time.time() - t0)


def mean_split(rows, lab, key):
    v = [r[lab][key] for r in rows]
    return np.mean(v), np.std(v)


def to_markdown(o):
    L = ['# One-motor VLA prototype: closed-loop study', '',
         'Simulation. %d training seeds, %d fresh test scenes per seed, mean ± std over seeds. Training scenes for the main table: %d. '
         'TinyVLA has %d parameters, the BC baseline %d.' % (o['seeds'], o['n_test'], o['big'], o['params']['vla'], o['params']['bc']), '',
         '## A. Policies', '',
         '"Ambiguous" = the instruction is true of more than one object, so the intended cell is a coin flip unless the robot asks. '
         'Success = final position within a quarter cell of the intended cell.', '',
         '| Policy | Success, all | Success, unambiguous | Success, ambiguous | Questions, all | Success − 0.1·questions | Paired p vs act |',
         '|---|---|---|---|---|---|---|']
    base = np.array(o['scored']['TinyVLA, act'])
    for name, rows in o['split'].items():
        a = mean_split(rows, 'all', 'success'); u = mean_split(rows, 'unambiguous', 'success'); b = mean_split(rows, 'ambiguous', 'success')
        q = mean_split(rows, 'all', 'asks')
        sc = np.array(o['scored'][name])
        p = '–' if name == 'TinyVLA, act' or name in ('oracle', 'BC baseline') else '%.4f' % sign_flip_p(sc - base)
        L.append('| %s | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.2f | %.3f | %s |' % (name, *a, *u, *b, q[0], sc.mean(), p))
    L += ['', 'Paired p: sign-flip test on per-scene (success − 0.1·questions) against "TinyVLA, act", scenes pooled over seeds. '
          'The gate level a=%g is the headline; the other levels are shown in full because the ask rate depends on it, not hidden.' % HEAD_ALPHA, '',
          '## B. Data efficiency (unambiguous instructions only)', '', '| Training scenes | TinyVLA success | BC baseline success |', '|---|---|---|']
    for n in o['sizes']:
        v = o["eff"][n]
        L.append('| %d | %.3f ± %.3f | %.3f ± %.3f |' % (n, np.mean(v['vla']), np.std(v['vla']), np.mean(v['bc']), np.std(v['bc'])))
    L += ['', 'The BC baseline was given a wider layer and more optimiser steps than TinyVLA and tuned lightly on one seed; a different '
          'architecture or far more data may close the gap, so read B as "this prototype is more sample-efficient than this baseline", not as a general result.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/one_motor_vla')
    args = ap.parse_args()
    o = run(args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_raw.json').write_text(json.dumps(o, indent=1, default=float))
    o = json.loads((out / 'benchmark_raw.json').read_text())
    o['eff'] = {int(k): v for k, v in o['eff'].items()}
    (out / 'benchmark_results.md').write_text(to_markdown(o))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
