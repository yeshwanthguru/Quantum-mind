"""Training data for a from-scratch model: reading your files, and building a corpus and question-answer pairs from this
library's own docstrings.

:func:`read_corpus` collects text from files and directories you point it at. :func:`library_corpus` introspects the
installed ``quantum_mind``
package and returns (a) every docstring as a text document and (b) question-answer pairs about each public class and function
("What does X do?"
answered with the first paragraph of its docstring), with some names held out entirely so the model can be tested on things
it never saw. It is a
small, closed-domain corpus: enough to show a model learning, not enough to know anything beyond this library.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import numpy as np

__all__ = ['read_corpus', 'read_qa', 'library_corpus', 'QUESTION_TEMPLATES', 'HELD_OUT_TEMPLATE']

TEXT_SUFFIXES = ('.txt', '.md', '.rst', '.py', '.tex', '.csv')
QUESTION_TEMPLATES = ('What does {name} do?', 'What is {name}?', 'Explain {name}.')
HELD_OUT_TEMPLATE = 'Tell me about {name}.'


def read_corpus(paths, suffixes=TEXT_SUFFIXES, max_chars=None):
    """The text of every file under ``paths`` (files or directories) with a text-like suffix, as a list of strings."""
    out = []
    for p in map(Path, paths if isinstance(paths, (list, tuple)) else [paths]):
        files = [p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file() and f.suffix in suffixes)
        for f in files:
            try:
                t = f.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            if t.strip():
                out.append(t[:max_chars] if max_chars else t)
    return out


def read_qa(path):
    """Question-answer pairs from a JSON-lines file: ``{"question": ..., "answer": ...}`` or ``{"messages": [...]}`` per line."""
    items = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        items.append(d['messages'] if 'messages' in d else (d['question'], d['answer']))
    return items


def _first_paragraph(doc, limit=400):
    para = ' '.join(inspect.cleandoc(doc).split('\n\n')[0].split())
    return para if len(para) <= limit else para[:limit].rsplit(' ', 1)[0] + ' ...'


def library_corpus(package='quantum_mind', held_out_fraction=0.2, seed=0):
    """Documents and question-answer pairs built from the docstrings of ``package``.

    Returns
    -------
    texts : list of str
        One document per module, class and function docstring.
    train_pairs : list of (question, answer)
        Three phrasings per name, for names the model may see.
    held_pairs : dict
        ``{'seen_names_new_phrasing': [...], 'unseen_names': [...]}``: pairs the model is tested on. The first uses a name that
        appears in training
        with a phrasing that does not; the second uses names that never appear in the question-answer pairs (their docstrings may
        still be in
        the text documents).
    """
    entries = {}
    mod = importlib.import_module(package)
    for info in pkgutil.walk_packages(mod.__path__, package + '.'):
        if info.name.endswith('__main__'):                                       # importing it would run a program
            continue
        try:
            m = importlib.import_module(info.name)
        except Exception:                                                       # an optional dependency that is not installed
            continue
        if m.__doc__:
            entries.setdefault(info.name, ('module', m.__doc__))
        for name, obj in vars(m).items():
            if name.startswith('_') or not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if getattr(obj, '__module__', '') != info.name or not inspect.getdoc(obj):
                continue
            entries.setdefault('%s.%s' % (info.name, name), (type(obj).__name__, inspect.getdoc(obj)))
    names = sorted(entries)
    texts = ['%s\n\n%s' % (n, inspect.cleandoc(entries[n][1])) for n in names]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_held = int(len(names) * held_out_fraction)
    held, seen = [names[i] for i in order[:n_held]], [names[i] for i in order[n_held:]]
    short = lambda n: n.split('.', 2)[-1] if n.count('.') >= 2 else n                      # noqa: E731
    seen_short = {short(n) for n in seen}
    # a short name shared with a seen entry would leak
    held = [n for n in held if short(n) not in seen_short]
    train_pairs = [(t.format(name=short(n)), _first_paragraph(entries[n][1])) for n in seen for t in QUESTION_TEMPLATES]
    return texts, train_pairs, {
        'seen_names_new_phrasing': [(HELD_OUT_TEMPLATE.format(name=short(n)), _first_paragraph(entries[n][1])) for n in seen],
        'unseen_names': [(QUESTION_TEMPLATES[0].format(name=short(n)), _first_paragraph(entries[n][1])) for n in held]}
