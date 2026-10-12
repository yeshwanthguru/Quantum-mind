"""Monitoring and self-healing agent for the motor: closed-loop fault study (simulation).

A two-leg move (go to A, then at 1.5 s to B; the score is taken at B after 4 s) with one fault injected at 1.5 s: encoder dropout,
encoder offset (+0.25), actuator gain loss (x0.25), friction increase (Coulomb x6), or an external push (+4 workspace units/s^2; the joint's
own maximum acceleration is 8). The controller is the trained TinyVLA action head. Policies: raw encoder; Kalman state estimate (no agent);
Kalman with the nominal motor model (no agent); the agent in monitor-only mode; the agent without the quantum accelerometer; the full agent.
Sensor values are the library's illustrative models. The agent's thresholds were set on separate random seeds, but with the same fault sizes, so they are tuned to this family of faults. Paired sign-flip tests on final error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_mind.embodied import TinyVLA, make_dataset
from quantum_mind.embodied.agent import run_fault_episode

T_FAULT = 1.5
FAULTS = {'none': [], 'encoder dropout': [dict(kind='encoder_dropout', t=T_FAULT, mag=0)],
          'encoder offset': [dict(kind='encoder_offset', t=T_FAULT, mag=0.25)],
          'actuator gain loss': [dict(kind='gain_loss', t=T_FAULT, mag=0.25)],
          'friction increase': [dict(kind='friction', t=T_FAULT, mag=6.0)],
          'external push': [dict(kind='push', t=T_FAULT, mag=4.0)]}
TRUTH = {'encoder dropout': 'encoder_lost', 'encoder offset': 'encoder_jump', 'actuator gain loss': 'actuator_degraded',
         'friction increase': 'friction_increase', 'external push': 'external_push'}
POLICIES = ('encoder', 'kalman', 'model', 'monitor', 'agent no quantum', 'agent')
LABEL = {'encoder': 'raw encoder', 'kalman': 'Kalman (no agent)', 'model': 'Kalman + motor model (no agent)',
         'monitor': 'agent, monitor only', 'agent no quantum': 'agent, no quantum accelerometer', 'agent': 'agent (monitor + self-heal)'}


def sign_flip_p(diff, n_perm=4000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    return float((np.sum(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15) + 1) / (n_perm + 1))


def run(quick=False):
    seeds, n = range(2 if quick else 5), (30 if quick else 80)
    res = {f: {p: dict(err=[], succ=[], first=[], final=[], lat=[], stop=[]) for p in POLICIES} for f in FAULTS}
    for seed in seeds:
        act = TinyVLA(seed=seed).fit(make_dataset(3000, seed), steps=3000, seed=seed).act
        for f, spec in FAULTS.items():
            for p in POLICIES:
                for i in range(n):
                    r = run_fault_episode(act, np.random.default_rng(10000 * seed + i), spec, p)
                    d = res[f][p]
                    d['err'].append(float(r['error']))
                    d['succ'].append(float(r['success']))
                    d['stop'].append(float(r['safe_stop']))
                    if p == 'agent':
                        ev = r['events']
                        d['first'].append(ev[0][1] if ev else None)
                        d['final'].append(r['final'])
                        d['lat'].append(ev[0][0] - T_FAULT if ev else None)
        print('seed', seed, flush=True)
    return dict(res=res, seeds=len(list(seeds)), n=n)


def to_markdown(o):
    L = ['# Monitoring and self-healing agent for the motor', '',
         'Simulation. %d seeds x %d episodes per cell. One fault is injected at %.1f s of a 4 s two-leg move; success = final error under a quarter '
         'cell. p: paired sign-flip test on final error against the full agent.' % (o['seeds'], o['n'], T_FAULT), '']
    for f, rows in o['res'].items():
        ref = np.array(rows['agent']['err'])
        L += ['## Fault: %s' % f, '', '| Policy | Success | Mean final error | Safe stops | p vs agent |', '|---|---|---|---|---|']
        for p, v in rows.items():
            e = np.array(v['err'])
            L.append('| %s | %.3f | %.3f | %.3f | %s |' % (LABEL[p], np.mean(v['succ']), e.mean(), np.mean(v['stop']),
                                                       '–' if p == 'agent' else '%.4f' % sign_flip_p(e - ref)))
        d = rows['agent']
        if f == 'none':
            L += ['', 'False alarms (any event in a healthy episode): %.3f of episodes.' % np.mean([x is not None for x in d['first']])]
        else:
            ok = np.mean([x == TRUTH[f] for x in d['first']])
            lat = [x for x in d['lat'] if x is not None and x >= 0]
            fin = np.mean([x == TRUTH[f] for x in d['final']])
            L += ['', 'Agent diagnosis: first event is the injected fault in %.3f of episodes; the final diagnosis is in %.3f; any event in %.3f; '
                  'median latency %.2f s.' % (ok, fin, np.mean([x is not None for x in d['first']]), np.median(lat) if lat else float('nan'))]
        L.append('')
    L += ['Limits: the agent knows the nominal plant and handles only these five fault kinds; faults are single and step-like; all data are simulated, '
          'with illustrative sensor values.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='benchmarks/results/motor_agent')
    a = ap.parse_args()
    o = run(a.quick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark_results.md').write_text(to_markdown(o))
    raw = {f: {p: dict(success=float(np.mean(v['succ'])), error=float(np.mean(v['err']))) for p, v in r.items()} for f, r in o['res'].items()}
    (out / 'benchmark_raw.json').write_text(json.dumps(raw, indent=1))
    print((out / 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
