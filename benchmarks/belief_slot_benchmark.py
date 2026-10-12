"""Belief slot of the one-motor VLA: Bayes, classical context weights and the quantum-probability slot.

Everything is simulated. A TinyVLA is trained once per seed; for ambiguous instructions the robot asks two clarifying questions and a
simulated person answers. Three persons:

* control: every answer is equally reliable, so the product rule (Bayes) is the right model and nothing else should help;
* classical context noise: the second answer is less reliable after a change of topic (position -> colour or back). This is not a quantum
  model; it is the honest classical competitor's home ground;
* interference: the intended object is drawn from a hidden quantum posterior with random rotations (a person whose answers do not
  commute). This is built from the quantum slot's own form, so it only checks that the slot can *represent* what the others cannot; it
  says nothing about whether real people behave this way.

Slots are fitted on held-out sequences and scored on fresh ones (log-loss and accuracy; paired sign-flip test on per-sequence log-loss).
Part B then runs the closed loop with the classical-context-noise person.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_mind.embodied import (BayesSlot, QuantumSlot, TemperedSlot, TinyVLA, calibrate_gate, make_dataset, oracle_plan,
                                   person_answers, run_episode, sequence_batch, vla_plan)
from quantum_mind.embodied.belief import K


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def nll_rows(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-9)


PERSONS = {'control (equal reliability)': dict(e_first=0.1, e_same=0.1, e_switch=0.1),
           'classical context noise': dict(e_first=0.05, e_same=0.10, e_switch=0.35),
           'interference (hidden quantum)': dict(e_first=0.02, e_same=0.02, e_switch=0.02)}


def make_batches(vla, name, seed, n_tr, n_te):
    rng = np.random.default_rng(seed)
    amb = lambda n, sd: [s for s in make_dataset(n, sd) if s.ambiguous]
    tr, te = amb(int(n_tr * 3), 100 + seed)[:n_tr], amb(int(n_te * 3), 200 + seed)[:n_te]
    kw = PERSONS[name]
    btr, bte = sequence_batch(vla, tr, rng, **kw), sequence_batch(vla, te, rng, **kw)
    if name.startswith('interference'):
        hidden = QuantumSlot()
        hidden.x = 0.9 * np.random.default_rng(900 + seed).standard_normal(hidden.n_params)
        for b in (btr, bte):
            p = hidden.posterior(b['g0'], b['L'], b['ctx'], b['mask'])
            b['y'] = np.array([rng.choice(K, p=q / q.sum()) for q in p])
    return btr, bte


def run(quick=False):
    seeds = range(2 if quick else 5)
    n_tr, n_te = (400, 400) if quick else (1200, 1200)
    t0 = time.time()
    part_a = {n: {'Bayes': [], 'Bayes, one fitted weight': [], 'Context-weighted (classical)': [], 'Quantum slot': []} for n in PERSONS}
    rows = {n: {k: [] for k in part_a[n]} for n in PERSONS}
    acc = {n: {k: [] for k in part_a[n]} for n in PERSONS}
    loop = {}
    for seed in seeds:
        vla = TinyVLA(seed=seed).fit(make_dataset(3000, seed), steps=3000, seed=seed)
        for name in PERSONS:
            btr, bte = make_batches(vla, name, seed, n_tr, n_te)
            for slot_name, slot in (('Bayes', BayesSlot()), ('Bayes, one fitted weight', TemperedSlot(shared=True)), ('Context-weighted (classical)', TemperedSlot()),
                                    ('Quantum slot', QuantumSlot())):
                slot.fit(btr)
                p = slot.posterior(bte['g0'], bte['L'], bte['ctx'], bte['mask'])
                r = nll_rows(p, bte['y'])
                rows[name][slot_name].extend(r.tolist())
                part_a[name][slot_name].append(float(r.mean()))
                acc[name][slot_name].append(float(np.mean(p.argmax(1) == bte['y'])))
            print('seed', seed, name, 'done %.0fs' % (time.time() - t0), flush=True)
        # closed loop, classical context noise
        btr, _ = make_batches(vla, 'classical context noise', seed, n_tr, 10)
        slots = {'Bayes': BayesSlot(), 'Bayes, one fitted weight': TemperedSlot(shared=True).fit(btr), 'Context-weighted (classical)': TemperedSlot().fit(btr), 'Quantum slot': QuantumSlot().fit(btr)}
        gate = calibrate_gate(vla, make_dataset(1000, 300 + seed), 0.02)
        test = make_dataset(300 if quick else 600, 400 + seed)
        plans = {'act on instruction': vla_plan(vla), 'gate, one idealised noise-free answer': vla_plan(vla, gate),
                 'gate, one noisy answer, Bayes': vla_plan(vla, gate, slot=slots['Bayes'], n_questions=1)}
        plans.update({'gate, two noisy answers, ' + k: vla_plan(vla, gate, slot=v) for k, v in slots.items()})
        plans['oracle'] = oracle_plan
        for name, pl in plans.items():
            rng = np.random.default_rng(seed)
            res = [run_episode(s, rng, pl) for s in test]
            amb = [r for r, s in zip(res, test) if s.ambiguous]
            d = loop.setdefault(name, dict(amb=[], all=[], asks=[], scored=[]))
            d['amb'].append(float(np.mean([r['success'] for r in amb])))
            d['all'].append(float(np.mean([r['success'] for r in res])))
            d['asks'].append(float(np.mean([r['asked'] for r in res])))
            d['scored'].extend([float(r['success']) - 0.05 * r['asked'] for r in res])
    return dict(seeds=len(list(seeds)), n_tr=n_tr, n_te=n_te, nll=part_a, acc=acc, rows=rows, loop=loop, seconds=time.time() - t0)


def to_markdown(o):
    L = ['# Belief slots of the one-motor VLA', '',
         'Simulation. %d seeds; %d sequences to fit each slot, %d fresh sequences to score it, ambiguous instructions only; mean ± std over seeds.'
         % (o['seeds'], o['n_tr'], o['n_te']), '',
         '## A. Which slot predicts the intended object best?', '',
         'Log-loss in nats (lower is better) and accuracy. p: paired sign-flip test on per-sequence log-loss, pooled over seeds.', '',
         '| Person | Slot | Log-loss | Accuracy | p vs Bayes | p vs classical context weights (quantum row only) |', '|---|---|---|---|---|---|']
    for person, d in o['nll'].items():
        for slot, v in d.items():
            rb = np.array(o['rows'][person]['Bayes']); rc = np.array(o['rows'][person]['Context-weighted (classical)'])
            r = np.array(o['rows'][person][slot])
            pb = '–' if slot == 'Bayes' else '%.4f' % sign_flip_p(r - rb)
            pc = '%.4f' % sign_flip_p(r - rc) if slot == 'Quantum slot' else '–'
            L.append('| %s | %s | %.3f ± %.3f | %.3f ± %.3f | %s | %s |' % (person, slot, np.mean(v), np.std(v),
                                                                       np.mean(o['acc'][person][slot]), np.std(o['acc'][person][slot]), pb, pc))
    L += ['', 'The "interference" person is generated by the quantum slot\'s own form, so a win there only shows the slot can represent a '
              'non-commuting person, not that people are one.', '',
          '## B. Closed loop (classical-context-noise person)', '',
          'Success within a quarter cell of the intended cell. Each question costs 0.05 in the last column (an assumption); the idealised row asks one question whose answer is always correct and informative, a reference, not a competitor. '
          'p: sign-flip on per-scene score against "gate, one noisy answer, Bayes".', '',
          '| Policy | Success, all | Success, ambiguous | Questions per episode | Success − 0.05·questions | p |', '|---|---|---|---|---|---|']
    base = np.array(o['loop']['gate, one noisy answer, Bayes']['scored'])
    for name, d in o['loop'].items():
        sc = np.array(d['scored'])
        p = '–' if name in ('oracle', 'act on instruction', 'gate, one noisy answer, Bayes') else '%.4f' % sign_flip_p(sc - base)
        L.append('| %s | %.3f ± %.3f | %.3f ± %.3f | %.2f | %.3f | %s |' % (name, np.mean(d['all']), np.std(d['all']), np.mean(d['amb']),
                                                                      np.std(d['amb']), np.mean(d['asks']), sc.mean(), p))
    L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/belief_slot')
    args = ap.parse_args()
    o = run(args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_results.md').write_text(to_markdown(o))
    del o['rows']
    (out / 'benchmark_raw.json').write_text(json.dumps(o, default=float))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
