"""Generate documentation sources from the repository before Sphinx reads them.

Called from ``docs/conf.py``. Generated files are ignored by git (see .gitignore):

* the READMEs of the subpackages, with their links rewritten for the documentation site;
* one API page per module (summary table, then every public member);
* the notebooks, executed, with live-widget outputs replaced by a note;
* the interactive Bloch-sphere demos embedded in the landing page.
"""
import os
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
GEN = DOCS / '_generated'          # static files for the build (demos, assets)
REPO = 'https://github.com/yeshwanthguru/quantum-cognition-robotics'

# README -> documentation page (path under docs/, without the suffix)
READMES = {
    'docs/concepts.md': 'user_guide/concepts',
    'src/qlcog/quantum/README.md': 'user_guide/quantum',
    'src/qlcog/inspired/README.md': 'user_guide/inspired',
    'src/qlcog/problems/README.md': 'user_guide/problems',
    'src/qlcog/circuits/README.md': 'user_guide/circuits',
    'src/qlcog/viz/README.md': 'user_guide/viz',
    'src/qlcog/applications/README.md': 'user_guide/robotics',
    'integrations/ros2/README.md': 'user_guide/ros2',
    'CHANGELOG.md': 'about/changelog',
    'CONTRIBUTING.md': 'about/contributing',
    'SECURITY.md': 'about/security',
}
FAMILIES = ['order_effects', 'conjunction', 'interference', 'qlbn', 'dynamics', 'decision', 'contextuality',
            'similarity', 'game_theory', 'memory', 'concepts']
for _f in FAMILIES:
    READMES['src/qlcog/families/%s/README.md' % _f] = 'user_guide/families/' + _f

# API pages: (module, title)
API = [
    ('qlcog.core.model', 'Model base class'), ('qlcog.core.fit', 'Fitting and comparison'),
    ('qlcog.core.linalg', 'Linear algebra'),
] + [('qlcog.families.%s.models' % f, f.replace('_', ' ').capitalize()) for f in FAMILIES] + [
    ('qlcog.applications.robotics', 'Robotics'), ('qlcog.applications.intent', 'Intent resolution'),
    ('qlcog.quantum.statevector', 'State-vector simulator'), ('qlcog.quantum.ansatz', 'Feature maps and ansatz'),
    ('qlcog.quantum.classifiers', 'Machine-learning models'), ('qlcog.quantum.algorithms', 'QAOA, VQE and Grover'),
    ('qlcog.quantum.fourier', 'Quantum Fourier transform'), ('qlcog.quantum.estimation', 'Amplitude estimation'),
    ('qlcog.quantum.walks', 'Quantum walks'),
    ('qlcog.inspired.optimisers', 'Quantum-inspired optimisers'), ('qlcog.inspired.tensor', 'Tensor-network classifier'),
    ('qlcog.inspired.rl', 'Quantum-inspired reinforcement learning'), ('qlcog.inspired.text', 'Quantum language model'),
    ('qlcog.problems', 'QUBO problems'),
    ('qlcog.circuits.builders', 'Circuit builders'), ('qlcog.circuits.backends', 'Running circuits'),
    ('qlcog.viz.states', 'Bloch vectors and trajectories'), ('qlcog.viz.mpl', 'Matplotlib viewer'),
    ('qlcog.viz.interactive', 'Interactive viewer'), ('qlcog.viz.themes', 'Themes'),
    ('qlcog.data', 'Published data'),
]
API_GROUPS = [('Core', 'qlcog.core.'), ('Quantum-like families', 'qlcog.families.'),
              ('Applications', 'qlcog.applications.'), ('Quantum', 'qlcog.quantum.'),
              ('Quantum-inspired', 'qlcog.inspired.'), ('Problems', 'qlcog.problems'),
              ('Circuits', 'qlcog.circuits.'), ('Bloch-sphere viewer', 'qlcog.viz.'), ('Data', 'qlcog.data')]


def _rewrite(md, src):
    """Point the links of one README at the documentation pages, the static assets or GitHub."""
    src_dir = (ROOT / src).parent
    pages = {str((ROOT / s).resolve()): n for s, n in READMES.items()}
    here = pathlib.PurePosixPath(READMES[src]).parent        # directory of this page under docs/
    up = '../' * len(here.parts)                              # from this page back to the docs root

    def rel(target):
        return os.path.relpath(target, here.as_posix() or '.')

    def link(m):
        text, target = m.group(1), m.group(2)
        if re.match(r'^(https?:|mailto:|#)', target):
            return m.group(0)
        path, _, anchor = target.partition('#')
        resolved = (src_dir / path).resolve()
        if str(resolved) in pages:
            return '[%s](%s.md%s)' % (text, rel(pages[str(resolved)]), '#' + anchor if anchor else '')
        try:
            rel_repo = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return m.group(0)
        if rel_repo.startswith('docs/assets/') and m.group(0).startswith('!'):
            return '![%s](%s)' % (text, rel(rel_repo[len('docs/'):]))
        if rel_repo == 'docs/API.md':
            return '[%s](%s)' % (text, rel('api/index.rst'))
        if rel_repo.startswith('examples/') and rel_repo.endswith('.py'):
            return '[%s](%s)' % (text, rel('auto_examples/%s.rst' % pathlib.Path(rel_repo).stem))
        kind = 'tree' if resolved.is_dir() else 'blob'
        return '[%s](%s/%s/main/%s%s)' % (text, REPO, kind, rel_repo, '#' + anchor if anchor else '')

    md = re.sub(r'!?\[([^\]]*)\]\(([^)\s]+)\)', link, md)

    def img(m):                                  # raw <img src="..."> tags: use the copied assets
        resolved = (src_dir / m.group(2)).resolve()
        try:
            asset = resolved.relative_to(ROOT / 'docs' / 'assets').as_posix()
            return '%s%s_static/assets/%s"' % (m.group(1), up, asset)
        except ValueError:
            return '%s%s/raw/main/%s"' % (m.group(1), REPO, resolved.relative_to(ROOT).as_posix())
    md = re.sub(r'(<img[^>]+src=")(?!https?:)([^"]+)"', img, md)
    md = re.sub(r'```mermaid\n', '```{mermaid}\n', md)
    return md


def readmes():
    """Write the rewritten READMEs as documentation pages."""
    for src, name in READMES.items():
        target = DOCS / (name + '.md')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_rewrite((ROOT / src).read_text(), src))


def api_pages():
    """One page per module: an autosummary table of its public members, then their full documentation."""
    import importlib
    out = DOCS / 'api' / 'modules'
    out.mkdir(parents=True, exist_ok=True)
    for mod, title in API:
        m = importlib.import_module(mod)
        names = getattr(m, '__all__', None) or [n for n, v in vars(m).items()
                                                  if not n.startswith('_') and getattr(v, '__module__', None) == mod]
        classes = [n for n in names if isinstance(getattr(m, n, None), type)]
        funcs = [n for n in names if callable(getattr(m, n, None)) and n not in classes]
        data = [n for n in names if n not in classes and n not in funcs]
        lines = [title, '=' * len(title), '', '.. automodule:: %s' % mod, '   :no-members:', '']
        for head, group in (('Classes', classes), ('Functions', funcs), ('Data', data)):
            if not group:
                continue
            lines += ['.. rubric:: %s' % head, '', '.. autosummary::', '   :nosignatures:', '']
            lines += ['   %s' % n for n in group] + ['']
        for n in classes:
            lines += ['.. autoclass:: %s.%s' % (mod, n), '   :members:', '   :show-inheritance:', '']
        for n in funcs:
            lines += ['.. autofunction:: %s.%s' % (mod, n), '']
        for n in data:
            lines += ['.. autodata:: %s.%s' % (mod, n), '   :no-value:', '']
        (out / (mod + '.rst')).write_text('\n'.join(lines))
    idx = ['API reference', '=============', '',
           'Every public class and function, generated from the docstrings (NumPy style). Each page starts',
           'with a summary table; the most used names are also importable from the subpackages, for example',
           '``from qlcog.quantum import VariationalClassifier``.', '']
    for head, prefix in API_GROUPS:
        mods = [(mod, title) for mod, title in API if mod.startswith(prefix)]
        idx += [head, '-' * len(head), '', '.. toctree::', '   :maxdepth: 1', '']
        idx += ['   %s <modules/%s>' % (title, mod) for mod, title in mods] + ['']
    (DOCS / 'api' / 'index.rst').write_text('\n'.join(idx))


def notebooks(execute=True):
    """Copy the notebooks to docs/notebooks, executed, with widget outputs replaced by a note."""
    import nbformat
    out = DOCS / 'notebooks'
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = nbformat.read(f, 4)
        if execute:
            import nbclient
            os.environ['PLOTLY_RENDERER'] = 'notebook_connected'
            nbclient.NotebookClient(nb, timeout=900, kernel_name='python3',
                                    resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
        nb.metadata.pop('widgets', None)
        for c in nb.cells:
            if c.cell_type != 'code':
                continue
            kept = []
            for o in c.get('outputs', []):
                if 'application/vnd.jupyter.widget-view+json' in o.get('data', {}):
                    kept.append(nbformat.v4.new_output('display_data', data={'text/markdown':
                        '*Live widget: it updates in place when the notebook runs in Jupyter.*'}))
                else:
                    kept.append(o)
            c.outputs = kept
        nbformat.write(nb, out / f.name)


def demos(static):
    """Interactive Bloch-sphere animations for the landing page (needs qiskit and plotly)."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qlcog.viz import circuit_trajectory, belief_trajectory, animate_bloch, save_html
    from qlcog.families.dynamics import OpenSystemBelief
    out = static / 'demos'
    out.mkdir(parents=True, exist_ok=True)
    qc = QuantumCircuit(2)
    qc.h(0); qc.ry(np.pi / 3, 1); qc.cx(0, 1); qc.rz(np.pi / 2, 0); qc.cx(0, 1); qc.h(0)
    tr = circuit_trajectory(qc, steps=12)
    tr.names = ['qubit 0', 'qubit 1']
    save_html(animate_bloch(tr, title='H · RY(π/3) · CNOT · RZ(π/2) · CNOT · H', height=460), out / 'entanglement.html')
    bt = belief_trajectory(OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3), (1, 1, 0, 1, 0, 1), steps=10)
    save_html(animate_bloch(bt, title='Trust belief over six interactions (simulated)', height=460), out / 'trust.html')
    # self-host plotly.js (one version, no third-party CDN) and give the page the figure's background
    import plotly
    shutil.copy(pathlib.Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js', out / 'plotly.min.js')
    for f in out.glob('*.html'):
        text = re.sub(r'src="https://cdn\.plot\.ly/plotly-[0-9.]+(?:\.min)?\.js"( integrity="[^"]*")?( crossorigin="[^"]*")?',
                      'src="plotly.min.js"', f.read_text())
        f.write_text(text.replace('<head>', '<head><style>html,body{margin:0;background:#0d1117}</style>', 1))


def copy_assets(static):
    """Make docs/assets available as _static/assets for raw HTML images."""
    dst = static / 'assets'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(DOCS / 'assets', dst)


def run(execute_notebooks=True, with_demos=True):
    """Generate everything; called once per Sphinx build."""
    if GEN.exists():
        shutil.rmtree(GEN)
    GEN.mkdir()
    static = GEN / 'static'
    static.mkdir()
    readmes()
    api_pages()
    copy_assets(static)
    notebooks(execute_notebooks)
    if with_demos:
        try:
            demos(static)
        except ImportError:
            pass
