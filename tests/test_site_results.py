"""The 2.0 rows of the website's results table match what the code produces (site/build.py RESULTS)."""
import pathlib
import runpy

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = dict(runpy.run_path(str(ROOT / 'site' / 'build.py'), run_name='site_build')['RESULTS'])


def _row(prefix):
    return next(v for k, v in RESULTS.items() if k.startswith(prefix))


def test_handover_row():
    from quantum_mind.applications.handover import TrustAwareHandover, simulate_handover_session
    from quantum_mind.families.dynamics import OpenSystemBelief, MarkovBelief
    person = OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)

    class Always(TrustAwareHandover):
        def decide(self):
            return {'action': 'handover'}
    row = _row('Trust-aware hand-over')
    for make in (lambda: TrustAwareHandover(person), lambda: TrustAwareHandover(MarkovBelief(p0=0.5, up=0.3, down=0.3)),
                 lambda: Always(person)):
        cost = np.mean([simulate_handover_session(make(), TrustAwareHandover(person), np.random.default_rng(i), 20)['cost']
                        for i in range(40)])
        assert '%.1f' % cost in row


def test_fusion_row():
    from quantum_mind.applications.fusion import compare_fusion
    from quantum_mind.applications.intent import QuantumIntentResolver
    intents, L = ['a', 'b', 'c'], {'x': [0.7, 0.2, 0.1], 'y': [0.25, 0.6, 0.15]}
    truth = QuantumIntentResolver(intents)
    truth.add_cue('x', L['x'], 0.7)
    truth.add_cue('y', L['y'], 0.7)
    rng = np.random.default_rng(1)
    orders = [['x', 'y'], ['y', 'x']]
    trials = [(o, intents[rng.choice(3, p=truth.posterior(o))]) for o in [orders[i % 2] for i in range(1500)]]
    res = compare_fusion(intents, L, trials[:1000], trials[1000:])
    row = _row('Multimodal fusion')
    for k in ('quantum_like', 'bayes', 'dempster_shafer'):
        assert '%.3f' % res[k]['log_loss'] in row


def _run_example(name, capsys):
    import sys
    argv = sys.argv
    sys.argv = [str(ROOT / 'examples' / name)]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = argv
    return capsys.readouterr().out


def test_calibration_row(capsys):
    out = _run_example('32_calibration_and_conformal.py', capsys)
    row = _row('Detector calibration')
    for line in out.splitlines():
        if 'ECE' in line:
            assert line.split('ECE')[1].split()[0] in row
    assert out.split('coverage ')[1].split()[0] in row


def test_zeno_row(capsys):
    out = _run_example('34_bistable_perception_zeno.py', capsys)
    row = _row('Bistable perception')
    for line in out.splitlines():
        if 'BIC' in line:
            assert line.split('BIC')[1].strip() in row


def test_allocation_row(capsys):
    out = _run_example('37_robot_adapts_to_each_person.py', capsys)
    row = _row('Trust-aware task allocation')
    costs = [line.split('expected cost')[1].strip() for line in out.splitlines() if 'expected cost' in line]
    assert len(costs) == 2 and all(c in row for c in costs)
    assert out.split('miss rate ')[1].split()[0] in row
    returns = out.split('mean return on new people:')[1]
    assert all(v.strip(',') in row for v in returns.split() if v.lstrip('-').replace('.', '').replace(',', '').isdigit())
