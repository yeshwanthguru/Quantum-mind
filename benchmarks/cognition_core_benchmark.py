"""Quantum-cognition core against classical models of context and order, on data from five known generators (simulation).

Each task draws evidence sequences ``(x_t, c_t)`` (3 steps, 3 hypotheses, 3 features, 2 contexts chosen at random per step) and a
label ``y`` from a *known* conditional distribution ``P(y | x, c)``, so the best achievable log-loss (the oracle) is known and every
model is scored by its gap to it. The generators:

* ``control``: a product of likelihoods, context ignored; Bayes is exactly right and nothing else should help.
* ``context weights``: each context scales how much its evidence counts; the classical context-weighted model is exactly right.
* ``markov context``: a context-dependent stochastic transition between hypotheses; the Markov model is exactly right.
* ``quantum order``: a hidden quantum-cognition core with random rotations; only it is exactly right. It is built from the model's own form,
  so a win here shows representational capacity, not that people are quantum.
* ``random GRU``: a random recurrent network, which none of the models is built to match (a misspecification test).

Every model is fitted on the same training sequences and scored on the same fresh ones. Paired sign-flip tests on per-sequence log-loss.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_mind.cognition import (BayesCore, ContextWeightedCore, GRUCore, MarkovContextCore, QuantumCognitionCore)

D, F, K, T = 3, 3, 2, 3
MODELS = {'Bayes': BayesCore, 'Context weights': ContextWeightedCore, 'Markov context': MarkovContextCore, 'GRU': GRUCore,
          'Quantum core': QuantumCognitionCore}


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def inputs(n, rng):
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    C = np.eye(K, dtype=np.float32)[rng.integers(0, K, (n, T))]
    return X, C


def make_oracle(kind, seed):
    """Return a function (X, C) -> true P(y | X, C) for one generator."""
    rng = np.random.default_rng(seed)
    if kind == 'control':
        m = BayesCore(D, F, K)
    elif kind == 'context weights':
        m = ContextWeightedCore(D, F, K)
    elif kind == 'markov context':
        m = MarkovContextCore(D, F, K)
    elif kind == 'quantum order':
        m = QuantumCognitionCore(D, F, K)
    else:
        m = GRUCore(D, F, K, hidden=6)
    import jax
    p = m.init(jax.random.PRNGKey(10_000 + seed))
    if kind == 'control':
        p['W'] = p['W'] * 4
    elif kind == 'context weights':
        p['W'] = p['W'] * 4
        p['lam'] = np.array([1.6, 0.3], np.float32)
    elif kind == 'markov context':
        p['W'] = p['W'] * 4
        p['T'] = (3.0 * np.eye(D)[None].repeat(K, 0) + np.stack([rng.normal(0, 2.5, (D, D)) for _ in range(K)])).astype(np.float32)
    elif kind == 'quantum order':
        p['W'] = p['W'] * 4
        p['H'] = (rng.normal(0, 1.2, (K, D, D))).astype(np.float32)
        p['phi'] = rng.uniform(0, 2 * np.pi, D).astype(np.float32)
        p['g'] = np.float32(-3.0)
    else:
        for k in p:
            if k.startswith('W'):
                p[k] = p[k] * 3
    return lambda X, C: np.asarray(m.probs(p, *_to(X, C)))


def _to(X, C):
    import jax.numpy as jnp
    return jnp.asarray(X), jnp.asarray(C), jnp.ones(X.shape[:2], bool)


def draw(oracle, n, rng):
    X, C = inputs(n, rng)
    P = oracle(X, C)
    y = np.array([rng.choice(D, p=p / p.sum()) for p in P])
    return X, C, y, P


def nll(P, y):
    return -np.log(P[np.arange(len(y)), y] + 1e-7)


def run(quick=False):
    seeds = range(2 if quick else 5)
    n_tr, n_te = (600, 1500) if quick else (2500, 4000)
    kw = dict(steps=250 if quick else 400, restarts=2)
    out = {k: {m: [] for m in MODELS} for k in ('control', 'context weights', 'markov context', 'quantum order', 'random GRU')}
    gaps = {k: {m: [] for m in MODELS} for k in out}
    orac = {k: [] for k in out}
    rows = {k: {m: [] for m in MODELS} for k in out}
    info = {}
    t0 = time.time()
    for seed in seeds:
        for kind in out:
            rng = np.random.default_rng(100 * seed + 7)
            oracle = make_oracle(kind, seed)
            Xa, Ca, ya, _ = draw(oracle, n_tr, rng)
            Xb, Cb, yb, Pb = draw(oracle, n_te, rng)
            o = nll(Pb, yb)
            orac[kind].append(float(o.mean()))
            for name, cls in MODELS.items():
                args = (D, F, K)
                m = (cls(*args, hidden=6, seed=seed, **kw) if name == 'GRU' else cls(*args, seed=seed, **kw)).fit(Xa, Ca, ya)
                r = nll(m.predict_proba(Xb, Cb), yb)
                out[kind][name].append(float(r.mean()))
                gaps[kind][name].append(float((r - o).mean()))
                rows[kind][name].extend((r - o).tolist())
                info[name] = m.n_params
                if name == 'Quantum core':
                    info.setdefault('rotation_rms', {}).setdefault(kind, []).append(m.rotation_size)
            print('seed', seed, kind, 'done %.0fs' % (time.time() - t0), flush=True)
    return dict(seeds=len(list(seeds)), n_tr=n_tr, n_te=n_te, nll=out, gap=gaps, oracle=orac, rows=rows, info=info)


def to_markdown(o):
    L = ['# Quantum-cognition core against classical models of context and order', '',
         'Simulation. %d seeds; %d training and %d test sequences per seed; 3 steps, 3 hypotheses. Gap = test log-loss minus the oracle '
         'log-loss (0 is perfect; lower is better); mean ± std over seeds. p: paired sign-flip test on per-sequence log-loss, '
         'each model against the quantum core.' % (o['seeds'], o['n_tr'], o['n_te']), '',
         'Parameters: ' + ', '.join('%s %d' % (m, o['info'][m]) for m in MODELS) + '.', '']
    for kind, models in o['gap'].items():
        L += ['## Generator: %s' % kind, '', 'Oracle log-loss %.3f.' % np.mean(o['oracle'][kind]), '',
              '| Model | Gap to oracle | p vs quantum core |', '|---|---|---|']
        quantum = np.array(o['rows'][kind]['Quantum core'])
        for m, v in models.items():
            p = '–' if m == 'Quantum core' else '%.4f' % sign_flip_p(quantum - np.array(o['rows'][kind][m]))
            L.append('| %s | %.4f ± %.4f | %s |' % (m, np.mean(v), np.std(v), p))
        others = {m: np.mean(v) for m, v in models.items() if m != 'Quantum core'}
        L += ['', 'Best other model: %s. Quantum core rotation size (rms): %.3f.' % (min(others, key=others.get),
                                                                                    np.mean(o['info']['rotation_rms'][kind])), '']
    L += ['The "quantum order" generator is built from the quantum core\'s own form, so winning there shows the model can represent '
          'non-commuting context, not that people behave this way. A rotation size near 0 on the other generators means the penalty '
          'kept the model close to Bayes.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/cognition_core')
    a = ap.parse_args()
    o = run(a.quick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_results.md').write_text(to_markdown(o))
    raw = {k: {m: dict(gap=float(np.mean(v))) for m, v in g.items()} for k, g in o['gap'].items()}
    (out / 'benchmark_raw.json').write_text(json.dumps(raw, indent=1))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
