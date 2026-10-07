"""Command line: ``python -m quantum_handover benchmark`` and ``python -m quantum_handover animate``."""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog='quantum-handover', description='Trust-aware robot hand-over (simulated people).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('benchmark', help='every policy against both populations of simulated people')
    b.add_argument('--people', type=int, default=300)
    b.add_argument('--steps', type=int, default=20)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--out', default=None, help='write the table to this Markdown file')
    a = sub.add_parser('animate', help='PyBullet session: quantum-like policy next to "always hand over"')
    a.add_argument('--steps', type=int, default=10)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--phi0', type=float, default=1.9, help="person's initial trust angle (P(trust) = cos^2(phi0/2))")
    a.add_argument('--out', default='handover.gif')
    args = ap.parse_args(argv)

    if args.cmd == 'benchmark':
        from .benchmark import run_benchmark, to_markdown
        table = to_markdown(run_benchmark(args.people, args.steps, args.seed))
        print(table)
        if args.out:
            pathlib.Path(args.out).write_text(table + '\n')
        return
    from quantum_mind.families.dynamics import OpenSystemBelief
    from .policies import make_policy, person_state
    from .sim import run_session, side_by_side, save_gif
    person = dict(phi0=args.phi0, a_pos=0.8, a_neg=1.2, gamma=0.3)
    runs = {}
    for name in ('quantum-like', 'always hand over'):
        runs[name] = run_session(make_policy(name), person_state(OpenSystemBelief(**person)),
                                 np.random.default_rng(args.seed), n_steps=args.steps,
                                 title='%s policy (same person, same draws)' % name)
    save_gif(side_by_side(runs['quantum-like'][0], runs['always hand over'][0]), args.out)
    for name, (_, log) in runs.items():
        print('%-17s cost %.1f, dropped %d, questions %d' % (name, log[-1]['cost'],
              sum(e['outcome'] == 'dropped' for e in log), sum(e['action'] == 'ask' for e in log)))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
