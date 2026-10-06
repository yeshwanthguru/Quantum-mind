"""Generate docs/API.md, the API reference, from the package's docstrings and signatures
(python3 docs/make_api.py). CI checks that the committed file is up to date."""
import importlib
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

MODULES = [
    'quantum_mind.core',
    'quantum_mind.families.order_effects',
    'quantum_mind.families.conjunction',
    'quantum_mind.families.interference',
    'quantum_mind.families.qlbn',
    'quantum_mind.families.dynamics',
    'quantum_mind.families.decision',
    'quantum_mind.families.contextuality',
    'quantum_mind.families.similarity',
    'quantum_mind.families.game_theory',
    'quantum_mind.families.memory',
    'quantum_mind.families.concepts',
    'quantum_mind.families.perception',
    'quantum_mind.applications.robotics',
    'quantum_mind.applications.intent',
    'quantum_mind.applications.calibration',
    'quantum_mind.applications.orchestration',
    'quantum_mind.applications.questioning',
    'quantum_mind.applications.personalisation',
    'quantum_mind.applications.handover',
    'quantum_mind.applications.fusion',
    'quantum_mind.quantum',
    'quantum_mind.inspired',
    'quantum_mind.inspired.exploration',
    'quantum_mind.inspired.tensor_layers',
    'quantum_mind.envs',
    'quantum_mind.problems',
    'quantum_mind.circuits',
    'quantum_mind.viz',
    'quantum_mind.data',
]


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
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, '__module__', '').startswith('quantum_mind'):
            yield n, obj
        elif n.isupper() and not callable(obj):
            yield n, obj


lines = ['# API reference', '',
         'Generated from the docstrings by `docs/make_api.py`; each entry shows the signature and the first',
         'paragraph of the docstring. Full docstrings: `help(quantum_mind.<module>.<name>)`.', '']
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
