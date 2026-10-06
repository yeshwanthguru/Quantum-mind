"""Generate docs/API.md, the API reference, from the package's docstrings and signatures
(python3 docs/make_api.py). CI checks that the committed file is up to date."""
import importlib
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

MODULES = ['qlcog.core', 'qlcog.families.order_effects', 'qlcog.families.conjunction', 'qlcog.families.interference',
           'qlcog.families.qlbn', 'qlcog.families.dynamics', 'qlcog.families.decision', 'qlcog.families.contextuality',
           'qlcog.families.similarity', 'qlcog.families.game_theory', 'qlcog.families.memory',
           'qlcog.families.concepts', 'qlcog.applications.robotics', 'qlcog.applications.intent',
           'qlcog.quantum', 'qlcog.inspired', 'qlcog.problems',
           'qlcog.circuits', 'qlcog.viz', 'qlcog.data']


def first_paragraph(obj):
    doc = inspect.getdoc(obj) or ''
    return doc.split('\n\n')[0].replace('\n', ' ').strip()


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ''


def public(mod):
    names = getattr(mod, '__all__', None) or [n for n in dir(mod) if not n.startswith('_')]
    for n in sorted(names, key=str.lower):
        try:
            obj = getattr(mod, n)
        except AttributeError:
            continue
        if inspect.ismodule(obj):
            continue
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, '__module__', '').startswith('qlcog'):
            yield n, obj
        elif n.isupper() and not callable(obj):
            yield n, obj


lines = ['# API reference', '',
         'Generated from the docstrings by `docs/make_api.py`; each entry shows the signature and the first',
         'paragraph of the docstring. Full docstrings: `help(qlcog.<module>.<name>)`.', '']
for name in MODULES:
    mod = importlib.import_module(name)
    lines += ['## `%s`' % name, '', first_paragraph(mod), '']
    for n, obj in public(mod):
        if inspect.isclass(obj):
            lines.append('- **class `%s%s`**: %s' % (n, signature(obj), first_paragraph(obj)))
            for m, f in inspect.getmembers(obj, inspect.isfunction):
                if not m.startswith('_') and f.__qualname__.startswith(obj.__name__ + '.'):
                    lines.append('  - `%s%s`: %s' % (m, signature(f), first_paragraph(f)))
        elif inspect.isfunction(obj):
            lines.append('- `%s%s`: %s' % (n, signature(obj), first_paragraph(obj)))
        else:
            lines.append('- `%s` (constant)' % n)
    lines.append('')
out = ROOT / 'docs' / 'API.md'
text = '\n'.join(l.rstrip() for l in lines).rstrip() + '\n'
if '--check' in sys.argv:
    sys.exit(0 if out.exists() and out.read_text() == text else 'docs/API.md is out of date: run python3 docs/make_api.py')
out.write_text(text)
print('wrote', out, len(lines), 'lines')
