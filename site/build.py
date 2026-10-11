"""Build the project website into _site/ (python3 site/build.py).

The landing page (with the in-browser playground) comes from site/templates_index.html and the
interactive animations from quantum_mind.viz; the documentation under _site/docs/ is the Sphinx site in
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
REPO = 'https://github.com/yeshwanthguru/Quantum-mind'
URL = 'https://yeshwanthguru.github.io/Quantum-mind/'
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')

RESULTS = [
    ('Robot allocation, grasp selection and scheduling, QAOA against classical solvers (simulated, 10 instances each)', 'instances solved to the optimum, out of 30: <strong>greedy descent 29</strong> · QAOA depth 2 and 3: 26 · QAOA depth 1: 24 · random sampling with the same 1000 shots 24 · simulated annealing 16 (QAOA is within noise of random sampling; exact simulation, no hardware)'),
    ('Quantum accelerometer fused with a MEMS accelerometer, position RMSE over 300 s (simulated, illustrative sensor values)', 'no position fixes: <strong>fused 10.0 m</strong> · MEMS only 339.7 m · quantum only 229.1 m; with a fix every 10 s: fused 0.144 m · MEMS only 0.173 m; the benefit vanishes when the quantum sensor is as unstable as the MEMS sensor (297 m)'),
    ('COBRA vs Bayes filter on a context-dependent intent task (simulated, 5 seeds)', 'context: cost 0.221 (COBRA) · 0.239 (Bayes filter) · 0.233 (MLP) · oracle 0.147 at 1000 training sequences; paired test significant in 2 of 5 seeds; with 60 sequences COBRA 0.347 against Bayes 0.280; under 1.5x test noise COBRA is best calibrated (cost 0.481 against 0.608)'),
    ('Control task with no context effect (simulated)', 'expected cost 0.204 (COBRA) · 0.189 (Bayes filter) at 1000 training sequences: COBRA is slightly worse where context does not matter'),
    ('Individual differences (example 11)', 'a pooled fit selects the quantum-like model for a population that is half anchoring; '
     'per-person fits recover all 12 people at about 3,000 answers each'),
    ('Trust as a qubit', "P(trust) read from the Bloch vector equals the model's prediction (0.7247)"),
    ('Trust-aware hand-over, 40 simulated sessions of 20 hand-overs (example 16)', 'mean session cost: order-aware 42.1 · '
     'Markov-based 60.9 · always hand over 96.2 (the simulated people follow the open-system model, which favours the '
     'order-aware policy by construction)'),
    ('Detector calibration, 5 classes (simulated, example 13)', 'ECE raw 0.192 · Platt 0.020 · temperature 0.017 · '
     '<strong>isotonic 0.009</strong>; conformal sets reach 0.911 coverage at a 0.90 target'),
    ('Multimodal fusion, held-out log loss (simulated, example 14)', 'quantum-like 0.759 · Bayes 0.793 · '
     'Dempster-Shafer 0.818 (the simulated people follow the quantum-like rule)'),
    ('Bistable perception, dwell times at two check intervals (simulated, example 15)', 'BIC: quantum Zeno 2796.6 · '
     'gamma renewal 3031.8 · Markov switching 3073.8'),]


def version():
    return re.search(r"__version__ = '([^']+)'", (ROOT / 'src/quantum_mind/__init__.py').read_text()).group(1)


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
    from quantum_mind.viz import circuit_trajectory, belief_trajectory, animate_bloch, save_html
    from quantum_mind.families.dynamics import OpenSystemBelief
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
    env = dict(os.environ, QUANTUM_MIND_DOCS_FAST='1' if fast else '0')
    subprocess.run([sys.executable, '-m', 'sphinx', '-b', 'html', '-q', str(ROOT / 'docs'), str(out)], check=True, env=env)
    shutil.rmtree(out.parent / 'jupyter_execute', ignore_errors=True)       # MyST-NB's scratch folder


def share_plotly(docs):
    """Point every documentation page and animation at the site's single copy of plotly.js (saves 4.8 MB)."""
    copy = docs / '_static' / 'demos' / 'plotly.min.js'
    if not copy.exists():
        return
    for f in docs.rglob('*.html'):
        text = f.read_text()
        if 'plotly.min.js' not in text:
            continue
        shared = '../' * (len(f.relative_to(OUT).parts) - 1) + 'static/plotly.min.js'
        new = re.sub(r'src="(?:[./]*_static/demos/)?plotly\.min\.js"', f'src="{shared}"', text)
        if new != text:
            f.write_text(new)
    copy.unlink()


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
        share_plotly(OUT / 'docs')

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
