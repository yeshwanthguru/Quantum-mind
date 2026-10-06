"""The website's in-browser playground (site/static/playground.js) must agree with the package's
simulator, and the site must build. Needs Node.js for the first test (skipped without it)."""
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ref(seq):
    from qlcog.quantum import Circuit
    c = Circuit(2)
    for g, q, _ in seq:
        if g == 'CX01':
            c.cx(0, 1)
        elif g == 'CX10':
            c.cx(1, 0)
        elif g == 'Sdg':
            c.sdg(q)
        elif g == 'T':
            c.p(np.pi / 4, q)
        elif g in ('H', 'X', 'Y', 'Z', 'S'):
            getattr(c, g.lower())(q)
        else:
            getattr(c, g.lower())(np.pi / 4, q)
    return c.state()[0]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js not installed')
def test_playground_matches_the_simulator():
    from qlcog.viz import bloch_vectors, concurrence, reduced_density_pair
    rng = np.random.default_rng(0)
    gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'Rx', 'Ry', 'Rz', 'Sdg', 'CX01', 'CX10']
    seqs = [[(str(rng.choice(gates)), int(rng.integers(0, 2)), 1.0) for _ in range(rng.integers(3, 12))] for _ in range(60)]
    out = json.loads(subprocess.check_output(['node', str(ROOT / 'site' / 'check_playground.js'),
                                              str(ROOT / 'site' / 'static' / 'playground.js'), json.dumps(seqs)]))
    for seq, r in zip(seqs, out):
        js = np.array([a + 1j * b for a, b in r['psi']]); py = _ref(seq); B = bloch_vectors(py, 2)
        assert abs(abs(np.vdot(js, py)) - 1) < 1e-9
        assert np.allclose(r['b0'], B[0], atol=1e-9) and np.allclose(r['b1'], B[1], atol=1e-9)
        assert abs(r['c'] - concurrence(reduced_density_pair(py, 0, 1, 2))) < 1e-6
    half = json.loads(subprocess.check_output(['node', str(ROOT / 'site' / 'check_playground.js'),
                                               str(ROOT / 'site' / 'static' / 'playground.js'),
                                               json.dumps([[('H', 0, 1.0), ('CX01', 0, 0.5)]])]))[0]
    assert abs(half['c'] - np.sqrt(0.5)) < 1e-9                         # CNOT^0.5 after H: concurrence 1/sqrt 2


def test_site_link_rewriting():
    import importlib.util
    spec = importlib.util.spec_from_file_location('site_build', ROOT / 'site' / 'build.py')
    sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
    md = sb.rewrite_links('[api](API.md) [ex](../examples/13_quantum_classifiers_triage.py) [web](https://x.org) [a](#h)',
                          'docs/concepts.md')
    assert '(api.html)' in md and sb.REPO + '/blob/main/examples/13_quantum_classifiers_triage.py' in md
    assert '(https://x.org)' in md and '(#h)' in md
