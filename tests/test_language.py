"""The robot-instruction language model: correct labels, word order, an LLM bridge that cannot be fooled, dialogue fusion."""
import numpy as np
import pytest

pytest.importorskip('jax')
pytest.importorskip('optax')

from quantum_mind.cognition import (DialogueBelief, LLMBridge, RobotLanguageModel, belief_from_parse, corpus as C,  # noqa: E402
                                    language_plan, llm_parse, make_dialogues, normalise)
from quantum_mind.embodied import expert_command, run_episode  # noqa: E402


def _truth(u):
    """Re-derive the valid cells of a sentence from its words alone (independent of the generator)."""
    words = u.text.split()
    start = max([i for i, w in enumerate(words) if w in ('no', 'wait', 'mean')] + [-1]) + 1
    segment = words[start:]
    desc = [w for w in segment if w in C.COLORS or w in C.LEFT_WORDS or w in C.RIGHT_WORDS]
    ok = set(range(len(u.cells)))
    for w in desc:
        ok &= set(C._referents(w, u.cells, u.colors))
    return sorted(u.cells[i] for i in ok)


def test_the_corpus_labels_every_sentence_correctly_and_holds_out_combinations():
    us = C.make_corpus(1500, 0)
    assert all(sorted(u.valid) == _truth(u) and u.target in u.valid for u in us)
    assert {u.kind for u in us} == {'simple', 'polite', 'conjunction', 'correction', 'followup', 'ambiguous'}
    assert not any(u.combo in C.HELD_OUT for u in us if u.combo)                         # never in training
    held = C.make_corpus(100, 1, 'heldout')
    assert all(u.combo in C.HELD_OUT and sorted(u.valid) == _truth(u) for u in held)
    assert all(w in C.VOCAB for u in us[:50] for w in u.text.split())


def test_followups_keep_the_scene_and_can_misspeak():
    rng = np.random.default_rng(0)
    u = next(x for x in C.make_corpus(200, 2) if x.kind == 'simple' and not x.ambiguous)
    right, wrong = C.make_followup(u, rng), C.make_followup(u, rng, wrong=True)
    assert right.target == wrong.target == u.target and right.cells == u.cells
    assert u.target in right.valid and u.target not in wrong.valid or len(u.cells) == 1


@pytest.fixture(scope='module')
def trained():
    train = C.make_corpus(5000, 0)
    lm = RobotLanguageModel(steps=700, seed=0).fit(train)
    bag = RobotLanguageModel(encoder='bag', lm_weight=0.0, steps=1500, seed=0).fit(train)   # more steps than the transformer
    return lm, bag


def _acc(m, us):
    us = [u for u in us if not u.ambiguous]
    return np.mean([np.argmax(b) in u.valid for u, b in zip(us, m.ground(us))])


def test_word_order_matters_to_the_transformer_not_to_the_bag_of_words(trained):
    lm, bag = trained
    test = C.make_corpus(1200, 5)
    corr = [u for u in test if u.kind == 'correction']
    assert _acc(lm, corr) > 0.9 and _acc(bag, corr) < 0.75                  # "red, no, blue" and "blue, no, red" look alike to a bag
    assert _acc(lm, [u for u in test if u.kind == 'simple']) > 0.95
    assert _acc(bag, [u for u in test if u.kind == 'simple']) > 0.9                       # the bag is fine when order does not matter
    assert lm.n_params == bag.n_params
    ambiguous = [u for u in test if u.ambiguous][:100]
    mass = np.mean([lm.ground(u)[u.valid].sum() for u in ambiguous])
    assert mass > 0.8                                                         # an ambiguous sentence puts its belief on its referents


def test_it_is_a_language_model(trained):
    lm, bag = trained
    test = C.make_corpus(800, 6)
    assert lm.perplexity(test) < 8 < len(C.WORDS)                             # far below guessing among the words
    out = lm.complete('please go to the')
    assert out.startswith('please go to the') and all(w in C.VOCAB for w in out.split())
    with pytest.raises(ValueError):
        bag.perplexity(test)


def test_the_bridge_checks_everything_the_llm_says(trained):
    lm, _ = trained
    u = next(x for x in C.make_corpus(300, 7) if x.kind == 'simple' and not x.ambiguous and len(x.cells) == 3)
    word = next(w for w in u.text.split() if w in C.COLORS or w in C.LEFT_WORDS or w in C.RIGHT_WORDS)
    ref = {'left': 'leftmost', 'first': 'leftmost', 'right': 'rightmost', 'last': 'rightmost'}.get(word, word)
    free = 'could you possibly nip across to the %s thing there' % word                # not in the robot vocabulary

    def llm(reply_sentence, reference):
        return lambda prompt: reply_sentence if 'Rewrite' in prompt else '{"reference": "%s"}' % reference

    good = LLMBridge(lm, llm(u.text, ref)).ground(free, u.image, u.cells, u.colors)
    assert not good.ask and np.argmax(good.belief) in u.valid and good.sentence == u.text
    assert good.belief[u.valid].sum() >= good.lm_belief[u.valid].sum() - 1e-6        # agreement does not hurt
    other = next(r for r in ('red', 'green', 'blue', 'leftmost', 'rightmost')
                 if (belief_from_parse(r, u.cells, u.colors) is not None) and belief_from_parse(r, u.cells, u.colors)[u.target] == 0)
    assert LLMBridge(lm, llm(u.text, other)).ground(free, u.image, u.cells, u.colors).ask      # an LLM that contradicts it: ask
    outside = LLMBridge(lm, llm('please go to the quantum zebra', ref))                  # words outside the vocabulary
    assert outside.ground(free, u.image, u.cells, u.colors).ask
    assert LLMBridge(lm, lambda p: 'I cannot help with that').ground(free, u.image, u.cells, u.colors).ask
    assert not LLMBridge(lm, None).ground(u.text, u.image, u.cells, u.colors).ask       # no LLM: the small model alone
    bridge = LLMBridge(lm, lambda p: (_ for _ in ()).throw(RuntimeError('server down')))
    assert bridge.ground(free, u.image, u.cells, u.colors).ask                           # an LLM that fails never crashes the robot


def test_parsers_reject_anything_unexpected():
    assert llm_parse('go to the red one', lambda p: '{"reference": "red"}') == 'red'
    assert llm_parse('x', lambda p: '{"reference": "purple"}') is None and llm_parse('x', lambda p: 'no json') is None
    assert normalise('x', lambda p: 'go to the red one') == 'go to the red one'
    assert normalise('x', lambda p: 'go to the crimson one') is None
    assert belief_from_parse('none', [1, 5], [0, 1]) is None
    assert belief_from_parse('leftmost', [1, 5], [0, 1])[1] == 1.0


def test_dialogues_fuse_and_the_plan_drives_the_motor(trained):
    lm, _ = trained
    train, test = make_dialogues(300, 0), make_dialogues(200, 1)
    for fusion in ('bayes', 'context weights', 'quantum core'):
        db = DialogueBelief(lm, fusion, steps=60, restarts=1).fit(train)
        p = db.predict_proba(test)
        assert p.shape == (200, 3) and np.allclose(p.sum(1), 1, atol=1e-4)
        assert all(p[i, len(d[0].cells):].sum() < 1e-6 for i, d in enumerate(test))
        assert db.accuracy(test) > 0.6 and db.log_loss(test) < 1.0
    plan = language_plan(lm, lambda t, th, om: expert_command(t, th, om))
    u = next(x for x in C.make_corpus(300, 9) if x.kind == 'correction' and not x.ambiguous)
    r = run_episode(u, np.random.default_rng(0), plan)
    assert set(r) >= {'error', 'success'} and r['asked'] == 0
