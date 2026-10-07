"""The documentation keeps up with the package: every public model has an atlas diagram, every atlas
API path exists, and every tutorial is listed and has its claims checked."""
import importlib
import inspect
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docs' / '_ext'))
import atlas  # noqa: E402

MODEL_MODULES = (['quantum_mind.families.%s' % f for f in ('order_effects', 'conjunction', 'interference', 'qlbn',
                                                           'dynamics', 'decision', 'contextuality', 'similarity',
                                                           'game_theory', 'memory', 'concepts', 'perception')]
                 + ['quantum_mind.applications.%s' % a for a in ('robotics', 'intent', 'calibration', 'orchestration',
                                                                 'questioning', 'personalisation', 'handover', 'fusion')]
                 + ['quantum_mind.envs', 'quantum_mind.quantum', 'quantum_mind.inspired', 'quantum_mind.problems'])

#: Public classes that are results, settings or helpers rather than models.
NOT_MODELS = {'Module', 'RoutingTask', 'X', 'W', 'XX', 'QAOAResult', 'GroverResult', 'AEResult', 'OptimResult',
              'StateIndexer', 'BayesNet', 'ProjectiveQuestionModel', 'Hamiltonian'}


def test_every_public_model_has_an_atlas_entry():
    text = json.dumps(atlas.PAGES)
    missing = []
    for m in MODEL_MODULES:
        mod = importlib.import_module(m)
        for name in getattr(mod, '__all__', []):
            if inspect.isclass(getattr(mod, name, None)) and name not in NOT_MODELS and not re.search(r'\b%s\b' % name, text):
                missing.append('%s.%s' % (m, name))
    assert not missing, 'add these models to docs/_ext/atlas.py: %s' % missing


def test_atlas_api_paths_exist_and_pages_have_mathematics():
    for key, (_, _, models) in atlas.PAGES.items():
        assert (ROOT / 'docs' / 'math' / (atlas.MATH[key] + '.md')).exists()
        for m in models:
            for path in m['api']:
                mod, _, name = path.rpartition('.')
                assert hasattr(importlib.import_module(mod), name), path
            if m['tutorial']:
                assert (ROOT / 'tutorials' / (m['tutorial'] + '.md')).exists(), m['tutorial']


def test_every_tutorial_is_listed_and_checks_its_claims():
    index = (ROOT / 'docs' / 'tutorials' / 'index.md').read_text()
    for src in sorted((ROOT / 'tutorials').glob('[0-9][0-9]_*.md')):
        assert src.stem in index, '%s is not in docs/tutorials/index.md' % src.stem
        body = src.read_text()
        if re.search(r'\*\*(Result|Summary)\.\*\*', body):
            assert 'assert ' in body, '%s states a result but has no assertion checking it' % src.name


def test_family_and_example_counts_in_the_docs():
    families = [p for p in (ROOT / 'src' / 'quantum_mind' / 'families').iterdir()
                if p.is_dir() and (p / 'models.py').exists()]
    words = {10: 'Ten', 11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen'}
    home = (ROOT / 'docs' / 'index.md').read_text()
    assert '%s families' % words[len(families)] in home
    n_examples = len(list((ROOT / 'examples').glob('[0-9][0-9]_*.py')))
    assert '%d runnable scripts' % n_examples in home


def test_foundations_examples_run():
    """Every code example on the Foundations pages runs (and its asserts hold)."""
    import contextlib
    import io
    for page in sorted((ROOT / 'docs' / 'learn').glob('*.md')):
        ns = {}
        for block in re.findall(r"```python\n(.*?)```", page.read_text(), re.S):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(block, str(page), 'exec'), ns)
