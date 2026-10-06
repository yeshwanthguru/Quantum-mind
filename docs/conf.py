"""Sphinx configuration for the Quantum Mind documentation.

Build locally with ``python -m sphinx -b html docs docs/_build/html`` (after ``pip install -e ".[all,docs]"``).
Set ``QUANTUM_MIND_DOCS_FAST=1`` to skip running the examples and notebooks (a quick preview of the text).
"""
import os
import pathlib
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(DOCS / '_ext'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import quantum_mind      # noqa: E402
import generate   # noqa: E402

FAST = os.environ.get('QUANTUM_MIND_DOCS_FAST') == '1'
generate.run(execute_notebooks=not FAST, with_demos=True)

# -- project ----------------------------------------------------------------------------------------
project = 'Quantum Mind'
author = 'Yeshwanth Guru'
copyright = '2026, Yeshwanth Guru'
version = release = quantum_mind.__version__
REPO = 'https://github.com/yeshwanthguru/quantum-mind'
SITE = 'https://yeshwanthguru.github.io/quantum-mind/'

# -- extensions -------------------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'myst_nb',
    'sphinxcontrib.mermaid',
]
source_suffix = {'.rst': 'restructuredtext', '.md': 'myst-nb', '.ipynb': 'myst-nb'}
exclude_patterns = ['_build', 'concepts.md', '_generated/static/demos/*.html', 'API.md', 'assets', 'auto_examples/*.ipynb',
                    'auto_examples/*.py', 'auto_examples/*.zip', 'auto_examples/*.json', '**.ipynb_checkpoints']
templates_path = ['_templates']

# autodoc and napoleon: NumPy-style docstrings, members in source order, type hints in the description
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {'exclude-members': '__init__, __repr__, __post_init__'}
autosummary_generate = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = True
add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'qiskit': ('https://quantum.cloud.ibm.com/docs/api/qiskit/', None),
}
intersphinx_timeout = 10

# MyST and notebooks: the notebooks are executed by generate.py, so they are rendered as they are
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath', 'deflist', 'html_image', 'attrs_inline', 'substitution']
myst_heading_anchors = 3
nb_execution_mode = 'off'
nb_render_markdown_format = 'myst'
myst_substitutions = {'version': version}
mermaid_version = '11.4.1'

# -- example gallery --------------------------------------------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': str(ROOT / 'examples'),
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'^((?!11_cloud_run).)*$' if not FAST else r'^$',   # 11 needs command-line arguments
    'ignore_pattern': r'__init__\.py',
    'within_subsection_order': 'FileNameSortKey',
    'default_thumb_file': str(DOCS / '_static' / 'thumbs' / 'q.png'),
    'thumbnail_size': (400, 280),
    'remove_config_comments': True,
    'image_scrapers': ('matplotlib',),
    'capture_repr': (),
    'show_memory': False,
    'show_signature': False,
    'abort_on_example_error': True,
    'download_all_examples': False,
    'reference_url': {'quantum_mind': None},
    'promote_jupyter_magic': False,
}

# -- HTML output ------------------------------------------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = 'Quantum Mind %s' % version
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.svg'
html_static_path = ['_static', '_generated/static']
html_css_files = ['custom.css']
html_baseurl = SITE + 'docs/'
html_show_sourcelink = False
html_last_updated_fmt = '%d %B %Y'
html_context = {
    'github_user': 'yeshwanthguru',
    'github_repo': 'quantum-mind',
    'github_version': 'main',
    'doc_path': 'docs',
    'default_mode': 'dark',
}
html_theme_options = {
    'logo': {'text': 'Quantum Mind', 'alt_text': 'Quantum Mind documentation'},
    'icon_links': [
        {'name': 'GitHub', 'url': REPO, 'icon': 'fa-brands fa-github'},
        {'name': 'Playground', 'url': SITE + '#playground-section', 'icon': 'fa-solid fa-globe'},
    ],
    'navbar_align': 'left',
    'header_links_before_dropdown': 6,
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page'],
    'use_edit_page_button': True,
    'show_toc_level': 2,
    'show_prev_next': True,
    'navigation_with_keys': False,
    'pygments_light_style': 'tango',
    'pygments_dark_style': 'github-dark',
    'announcement': ('Quantum Mind %s: robot decision layer, perception and learning modules. '
                     '<a href="%sdocs/about/changelog.html">What changed</a>' % (version, SITE)),
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
}
html_sidebars = {'index': [], 'getting_started': [], 'auto_examples/index': []}


def _version_placeholder(app, docname, source):
    """Replace __VERSION__ in page sources (raw HTML and code blocks are not substituted by MyST)."""
    source[0] = source[0].replace('__VERSION__', version)


def _local_plotly(app, exception):
    """Point Plotly figures in the built pages at the self-hosted plotly.js (no third-party CDN)."""
    import re
    if exception is not None or app.builder.format != 'html':
        return
    out = pathlib.Path(app.outdir)
    for f in out.rglob('*.html'):
        text = f.read_text(encoding='utf-8')
        if 'cdn.plot.ly' not in text:
            continue
        rel = '../' * (len(f.relative_to(out).parts) - 1) + '_static/demos/plotly.min'
        text = re.sub(r'(src=")https://cdn\.plot\.ly/plotly-[0-9.]+\.min\.js"( integrity="[^"]*")?( crossorigin="[^"]*")?',
                      r'\1%s.js"' % rel, text)
        text = re.sub(r'https://cdn\.plot\.ly/plotly-[0-9.]+\.min', rel, text)
        f.write_text(text, encoding='utf-8')


def setup(app):
    app.connect('source-read', _version_placeholder)
    app.connect('build-finished', _local_plotly)
