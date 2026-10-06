"""Build the project website into _site/ (python3 site/build.py).

Everything is generated from the repository: the landing page from site/templates_index.html, the
interactive animations from qlcog.viz, the documentation pages from the Markdown files, and the
notebooks by executing them and converting them to HTML. GitHub Actions runs this on every push to
main and deploys _site/ to GitHub Pages (.github/workflows/pages.yml).

Options: --skip-notebooks (faster local preview)."""
import datetime
import html
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

# Markdown files rendered as pages: source -> (output path, title)
PAGES = {
    'docs/concepts.md': ('docs/concepts.html', 'Quantum-like, quantum and quantum-inspired'),
    'docs/API.md': ('docs/api.html', 'API reference'),
    'CHANGELOG.md': ('docs/changelog.html', 'Changelog'),
    'examples/README.md': ('docs/examples.html', 'Examples'),
    'src/qlcog/quantum/README.md': ('docs/quantum.html', 'Quantum models'),
    'src/qlcog/inspired/README.md': ('docs/inspired.html', 'Quantum-inspired models'),
    'src/qlcog/viz/README.md': ('docs/viz.html', 'Bloch-sphere viewer'),
    'src/qlcog/problems/README.md': ('docs/problems.html', 'Optimisation problems'),
    'src/qlcog/circuits/README.md': ('docs/circuits.html', 'Circuits and backends'),
    'src/qlcog/applications/README.md': ('docs/robotics.html', 'Robotics application'),
    'integrations/ros2/README.md': ('docs/ros2.html', 'ROS 2 integration'),
    'CONTRIBUTING.md': ('docs/contributing.html', 'Contributing'),
    'SECURITY.md': ('docs/security.html', 'Security policy'),
}
FAMILIES = ['order_effects', 'conjunction', 'interference', 'qlbn', 'dynamics', 'decision', 'contextuality', 'similarity',
            'game_theory', 'memory', 'concepts']
for fam in FAMILIES:
    PAGES['src/qlcog/families/%s/README.md' % fam] = ('docs/families/%s.html' % fam, 'Family: ' + fam.replace('_', ' '))

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


def rewrite_links(md, src):
    """Relative links: to a page built here -> that page; anything else in the repository -> GitHub."""
    src_dir = (ROOT / src).parent
    rendered = {str((ROOT / s).resolve()): o for s, (o, _) in PAGES.items()}
    out_dir = pathlib.PurePosixPath(PAGES[src][0]).parent

    def repl(m):
        text, target = m.group(1), m.group(2)
        if re.match(r'^(https?:|mailto:|#)', target):
            return m.group(0)
        path, _, anchor = target.partition('#')
        resolved = (src_dir / path).resolve()
        if str(resolved) in rendered:
            rel = os.path.relpath(rendered[str(resolved)], out_dir)
            return '[%s](%s%s)' % (text, rel, '#' + anchor if anchor else '')
        try:
            rel_repo = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return m.group(0)
        kind = 'tree' if resolved.is_dir() else 'blob'
        return '[%s](%s/%s/main/%s%s)' % (text, REPO, kind, rel_repo, '#' + anchor if anchor else '')
    md = re.sub(r'\[([^\]]*)\]\(([^)\s]+)\)', repl, md)
    md = re.sub(r'(<img[^>]+src=")(?!https?:)([^"]+)"',
                lambda m: '%s%s"' % (m.group(1), '%s/raw/main/%s' % (REPO, (src_dir / m.group(2)).resolve().relative_to(ROOT).as_posix())), md)
    return md


def render_markdown(md):
    import markdown
    md = re.sub(r'```mermaid\n(.*?)```', lambda m: '<pre class="mermaid">%s</pre>' % html.escape(m.group(1)), md, flags=re.S)
    return markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc', 'sane_lists', 'md_in_html'])


def page(title, body, out_rel, ver, date):
    depth = len(pathlib.PurePosixPath(out_rel).parts) - 1
    root = '../' * depth
    mermaid = ('<script type="module">import m from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
               'm.initialize({startOnLoad:true, theme: matchMedia("(prefers-color-scheme: light)").matches ? "default" : "dark"});</script>'
               if 'class="mermaid"' in body else '')
    return fill((SITE / 'templates_page.html').read_text(), TITLE=html.escape(title), BODY=body + mermaid, ROOT=root,
                REPO=REPO, VERSION=ver, BUILD_DATE=date)


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


def build_notebooks(out):
    import nbformat
    import nbclient
    from nbconvert import HTMLExporter
    os.environ['PLOTLY_RENDERER'] = 'notebook_connected'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    exporter = HTMLExporter(template_name='lab')
    for f in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = nbformat.read(f, 4)
        nbclient.NotebookClient(nb, timeout=900, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
        nb.metadata.pop('widgets', None)                   # live widgets cannot run on a static page
        for c in nb.cells:
            if c.cell_type != 'code':
                continue
            kept = []
            for o in c.get('outputs', []):
                data = o.get('data', {})
                if 'application/vnd.jupyter.widget-view+json' in data:
                    kept.append(nbformat.v4.new_output('display_data', data={'text/markdown':
                        '*Live widget: it updates in place when the notebook runs in Jupyter. '
                        'Try the [playground on the home page](../index.html#playground-section) instead.*'}))
                else:
                    kept.append(o)
            c.outputs = kept
        body, _ = exporter.from_notebook_node(nb)
        (out / (f.stem + '.html')).write_text(body)
        local_plotly(out / (f.stem + '.html'), '../')
        title = next((c.source.splitlines()[0].lstrip('# ') for c in nb.cells if c.cell_type == 'markdown'), f.stem)
        items.append((f.stem, title))
    return items


def main():
    ver = version(); date = datetime.date.today().isoformat()
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

    for src, (out_rel, title) in PAGES.items():
        md = rewrite_links((ROOT / src).read_text(), src)
        target = OUT / out_rel; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page(title, render_markdown(md), out_rel, ver, date))

    fam_links = ''.join('<li><a href="families/%s.html">%s</a></li>' % (f, f.replace('_', ' ')) for f in FAMILIES)
    docs_index = ('<h1>Documentation</h1><div class="grid2">'
                  '<div class="card"><h3>Start here</h3><ul><li><a href="concepts.html">Quantum-like, quantum and quantum-inspired</a></li>'
                  '<li><a href="api.html">API reference</a> (generated from the docstrings)</li><li><a href="examples.html">Examples</a></li>'
                  '<li><a href="../notebooks/">Notebooks</a></li><li><a href="changelog.html">Changelog</a></li></ul></div>'
                  '<div class="card"><h3>Modules</h3><ul><li><a href="quantum.html">Quantum models</a></li><li><a href="inspired.html">Quantum-inspired models</a></li>'
                  '<li><a href="problems.html">Optimisation problems</a></li><li><a href="circuits.html">Circuits and backends</a></li>'
                  '<li><a href="viz.html">Bloch-sphere viewer</a></li><li><a href="robotics.html">Robotics application</a></li>'
                  '<li><a href="ros2.html">ROS 2 integration</a></li></ul></div>'
                  '<div class="card"><h3>Quantum-like families</h3><ul>%s</ul></div>'
                  '<div class="card"><h3>Project</h3><ul><li><a href="contributing.html">Contributing</a></li><li><a href="security.html">Security policy</a></li>'
                  '<li><a href="%s/blob/main/CITATION.cff">Citation</a></li><li><a href="%s/blob/main/LICENSE">Licence (Apache-2.0)</a></li></ul></div></div>'
                  % (fam_links, REPO, REPO))
    (OUT / 'docs' / 'index.html').write_text(page('Documentation', docs_index, 'docs/index.html', ver, date))

    if '--skip-notebooks' not in sys.argv:
        items = build_notebooks(OUT / 'notebooks')
        lst = ''.join('<li><a href="%s.html">%s</a> · <a href="%s/blob/main/notebooks/%s.ipynb">source</a></li>'
                      % (s, html.escape(t), REPO, s) for s, t in items)
        (OUT / 'notebooks' / 'index.html').write_text(page('Notebooks', '<h1>Notebooks</h1><p>Executed when the site is built; '
                                                     'interactive figures work in the browser.</p><ul>%s</ul>' % lst,
                                                     'notebooks/index.html', ver, date))
    (OUT / '.nojekyll').write_text('')
    (OUT / 'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: %ssitemap.xml\n' % URL)
    pages = ['', 'docs/', 'notebooks/'] + [o for o, _ in PAGES.values()]
    (OUT / 'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                     + ''.join('  <url><loc>%s%s</loc></url>\n' % (URL, p) for p in pages) + '</urlset>\n')
    (OUT / '404.html').write_text(page('Not found', '<h1>Page not found</h1><p><a href="index.html">Back to the home page</a></p>',
                                       '404.html', ver, date).replace('href="index.html"', 'href="%s"' % URL))
    print('site built in', OUT, '-', sum(1 for _ in OUT.rglob('*') if _.is_file()), 'files')


if __name__ == '__main__':
    main()
