"""Build the project website into _site/ (python3 site/build.py).

The landing page (with the in-browser playground) comes from site/templates_index.html and the
interactive animations from qlcog.viz; the documentation under _site/docs/ is the Sphinx site in
docs/ (API reference, user guide, example gallery, executed notebooks). GitHub Actions runs this on
every push to main and deploys _site/ to GitHub Pages (.github/workflows/pages.yml).

Options: --fast builds the documentation without running the examples and notebooks;
--skip-docs builds the landing page only.
"""
import datetime
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
OUT = ROOT / '_site'
REPO = 'https://github.com/yeshwanthguru/quantum-cognition-robotics'
URL = 'https://yeshwanthguru.github.io/quantum-cognition-robotics/'
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')

RESULTS = [
    ('Triage classification, test accuracy, mean ± sd over 10 seeds (simulated)',
     '<strong>classical logistic regression on quadratic features 0.926 ± 0.016</strong> · tensor network (MPS) 0.915 ± 0.015 · '
     'variational quantum classifier 0.870 ± 0.033 · linear logistic regression 0.814 ± 0.041 · quantum kernel 0.781 ± 0.040'),
    ('Multi-robot task allocation (3 × 3)', 'QAOA, simulated quantum annealing, QIEA and simulated annealing all reach the optimum; '
     'QAOA P(optimal) 0.04 ideal and 0.03 under IBM device noise, against 0.002 uniform'),
    ('Portfolio, 3 of 8 assets (simulated)', 'QAOA (p = 2), SQA, QIEA and simulated annealing all optimal'),
    ('VQE, 4-spin transverse-field Ising chain', 'error below 1e-8 against exact diagonalisation'),
    ('Grover, 9 of 32 schedules valid', 'P(valid) 0.99 after one iteration (random 0.28)'),
    ('Individual differences (example 22)', 'a pooled fit selects the quantum-like model for a population that is half anchoring; '
     'per-person fits recover all 12 people at about 3,000 answers each'),
    ('Amplitude estimation of an expected payoff (6,800 oracle calls)', 'RMSE 0.0014 against 0.0049 for Monte Carlo '
     'with the same number of samples (ideal simulator)'),
    ('Robot navigation, 6 × 6 grid (shortest path 10)', 'quantum-inspired QRL 10.0 steps per episode after training, '
     'Q-learning 10.6; QRL is slower in the first episodes'),
    ('Anomaly detection (simulated)', '<strong>RBF kernel AUC 1.000</strong> · quantum kernel AUC 0.905 (angle encoding is '
     'periodic, so far-out points wrap around)'),
    ('Concept combination, "pet and fish" (simulated)', 'Fock-space model SSE 0.001 · weighted average 0.054 · '
     'product 0.388'),
    ('Trust as a qubit', "P(trust) read from the Bloch vector equals the model's prediction (0.7247)"),
]


def version():
    return re.search(r"__version__ = '([^']+)'", (ROOT / 'src/qlcog/__init__.py').read_text()).group(1)


def fill(text, **kw):
    for k, v in kw.items():
        text = text.replace('{{%s}}' % k, str(v))
    return text


CDN = re.compile(r'https://cdn\.plot\.ly/plotly-\d+(?:\.\d+)*(?:\.min)?(\.js)?')


def local_plotly(path, rel):
    """Point Plotly script tags (and require.js paths) of a generated page at the self-hosted copy."""
    text = path.read_text()
    path.write_text(CDN.sub(lambda m: rel + 'static/plotly.min' + (m.group(1) or ''), text))


def build_demos(out):
    import numpy as np
    from qiskit import QuantumCircuit
    from qlcog.viz import circuit_trajectory, belief_trajectory, animate_bloch, save_html
    from qlcog.families.dynamics import OpenSystemBelief
    out.mkdir(parents=True, exist_ok=True)
    qc = QuantumCircuit(2)
    qc.h(0); qc.ry(np.pi / 3, 1); qc.cx(0, 1); qc.rz(np.pi / 2, 0); qc.cx(0, 1); qc.h(0)
    tr = circuit_trajectory(qc, steps=12); tr.names = ['qubit 0', 'qubit 1']
    save_html(animate_bloch(tr, title='H · RY(π/3) · CNOT · RZ(π/2) · CNOT · H', height=480), out / 'entanglement.html')
    bt = belief_trajectory(OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3), (1, 1, 0, 1, 0, 1), steps=10)
    save_html(animate_bloch(bt, title='Trust belief over success, success, failure, success, failure, success (simulated)',
                            height=480), out / 'trust.html')
    ghz = QuantumCircuit(3); ghz.h(0); ghz.cx(0, 1); ghz.cx(1, 2)
    g = circuit_trajectory(ghz, steps=12); g.names = ['qubit 0', 'qubit 1', 'qubit 2']
    save_html(animate_bloch(g, title='GHZ state: all three vectors shrink to the centre', height=480), out / 'ghz.html')
    for f in out.glob('*.html'):
        local_plotly(f, '../')


def build_docs(out, fast=False):
    """Build the Sphinx documentation (docs/) into out."""
    import subprocess
    env = dict(os.environ, QLCOG_DOCS_FAST='1' if fast else '0')
    subprocess.run([sys.executable, '-m', 'sphinx', '-b', 'html', '-q', str(ROOT / 'docs'), str(out)], check=True, env=env)
    shutil.rmtree(out.parent / 'jupyter_execute', ignore_errors=True)       # MyST-NB's scratch folder


def main():
    """Build the landing page, the animations and the documentation into _site/."""
    ver = version()
    date = datetime.date.today().isoformat()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'static').mkdir(parents=True)
    for f in (SITE / 'static').iterdir():
        shutil.copy(f, OUT / 'static' / f.name)
    shutil.copytree(ROOT / 'docs' / 'assets', OUT / 'assets')
    shutil.copy(SITE / 'static' / 'favicon.svg', OUT / 'assets' / 'favicon.svg')
    import plotly                                          # self-host plotly.js: one version, no third-party CDN
    shutil.copy(pathlib.Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js', OUT / 'static' / 'plotly.min.js')

    n_tests = sum(len(re.findall(r'^def test_', p.read_text(), re.M)) for p in (ROOT / 'tests').glob('test_*.py'))
    n_examples = len(list((ROOT / 'examples').glob('[0-9][0-9]_*.py')))
    rows = '\n    '.join('<tr><td>%s</td><td>%s</td></tr>' % r for r in RESULTS)
    (OUT / 'index.html').write_text(fill((SITE / 'templates_index.html').read_text(), VERSION=ver, REPO=REPO, SITE=URL,
                                         TESTS=n_tests, N_EXAMPLES=n_examples, RESULTS=rows, BUILD_DATE=date))
    build_demos(OUT / 'demos')
    if '--skip-docs' not in sys.argv:
        build_docs(OUT / 'docs', fast='--fast' in sys.argv)

    (OUT / '.nojekyll').write_text('')
    (OUT / 'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: %ssitemap.xml\n' % URL)
    pages = [''] + sorted(p.relative_to(OUT).as_posix() for p in (OUT / 'docs').rglob('*.html')
                          if not p.name.startswith(('genindex', 'search', 'py-modindex', '_')) and '_static' not in p.parts
                          and '_modules' not in p.parts) if (OUT / 'docs').exists() else ['']
    (OUT / 'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                     + ''.join('  <url><loc>%s%s</loc></url>\n' % (URL, p) for p in pages) + '</urlset>\n')
    (OUT / '404.html').write_text(fill((SITE / 'templates_page.html').read_text(), TITLE='Page not found', ROOT=URL, REPO=REPO,
                                       VERSION=ver, BUILD_DATE=date,
                                       BODY='<h1>Page not found</h1><p><a href="%s">Back to the home page</a> or the '
                                            '<a href="%sdocs/">documentation</a>.</p>' % (URL, URL)))
    print('site built in', OUT, '-', sum(1 for _ in OUT.rglob('*') if _.is_file()), 'files')


if __name__ == '__main__':
    main()
