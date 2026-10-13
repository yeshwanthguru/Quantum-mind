# A small robot language model, and a local LLM on a short leash

A robot that takes spoken instructions needs more than a bag of words: "go to the red one, **no**, the blue one" means the blue one, and
"…the blue one, no, the red one" means the red one, though both contain the same words. This tutorial trains a **small language model from
scratch** (about 29,000 parameters, on a CPU) on rule-generated robot instructions, shows what word order buys over a bag-of-words encoder,
and then connects a general local LLM (such as Llama or Qwen served by Ollama) through a **bridge that checks everything it says**. The
sentences come from a small grammar of 32 words, written by the library's author, so this shows what a model can learn from such rules, not how
it would read people. No language model is run in this tutorial: the LLM is replaced by scripted replies.

You need the optional extra: `pip install "quantum-mind[cognition]"` (JAX and Optax).

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.cognition import (RobotLanguageModel, LLMBridge, DialogueBelief, make_dialogues, language_plan, belief_from_parse,
                                    corpus as C)
from quantum_mind.embodied import expert_command, run_episode
theme = use_mpl_style('dark')
```

## 1. The data: sentences with a scene and a ground truth

Each sentence comes with a camera strip (2 or 3 coloured objects on 16 cells), the cells the sentence is true of, and the cell the speaker
means. Corrections, synonyms ("leftmost", "left", "first"), politeness, conjunctions ("the red leftmost block") and ambiguity ("go to the block")
are all in the grammar.

```python
train = C.make_corpus(5000, seed=0)
for kind in ('simple', 'conjunction', 'correction', 'followup', 'ambiguous'):
    u = next(x for x in train if x.kind == kind)
    print('%-12s "%s"   objects at %s -> valid cells %s' % (kind, u.text, list(zip(u.cells, [C.COLORS[c] for c in u.colors])), u.valid))
print('vocabulary: %d words, longest sentence %d tokens' % (len(C.WORDS), max(len(u.tokens) for u in train)))
```

## 2. A bag of words against a small transformer

Both encoders turn a sentence into the query of the same attention step over the camera strip. They have the same number of parameters. The
transformer is also trained to predict the next word (a language model); the bag of words just averages the word embeddings. The bag is given
more than twice the training steps so that its weakness below is about word order, not about under-training.

```python
bag = RobotLanguageModel(encoder='bag', lm_weight=0.0, steps=1500, seed=0).fit(train)          # the bag gets twice the training steps
lm = RobotLanguageModel(steps=700, seed=0).fit(train)
print('parameters: bag %d, transformer %d' % (bag.n_params, lm.n_params))

test = C.make_corpus(1500, seed=5)
def accuracy(model, kind):
    us = [u for u in test if u.kind == kind and not u.ambiguous]
    return np.mean([np.argmax(b) in u.valid for u, b in zip(us, model.ground(us))])
kinds = ('simple', 'polite', 'conjunction', 'followup', 'correction')
acc = {name: [accuracy(m, k) for k in kinds] for name, m in (('bag of words', bag), ('transformer + LM', lm))}
fig, ax = plt.subplots(figsize=(6.6, 3.0))
w = 0.38
for j, (name, a) in enumerate(acc.items()):
    ax.bar(np.arange(len(kinds)) + (j - 0.5) * w, a, w, label=name, color=theme['palette'][1 + 2 * j])
ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds); ax.set_ylabel('grounding accuracy'); ax.legend(frameon=False)
ax.set_title('Same head, same size: only word order differs')
plt.show()
for name, a in acc.items():
    print('%-18s' % name, '  '.join('%s %.2f' % (k, v) for k, v in zip(kinds, a)))
```

**Result.** Without word order, corrections are close to a coin flip between the two things mentioned; the transformer reads the correction.

```python
assert accuracy(lm, 'correction') > 0.9 and accuracy(bag, 'correction') < 0.75
assert accuracy(lm, 'simple') > 0.95 and accuracy(bag, 'simple') > 0.9            # the bag is fine when order does not matter
print('corrections: bag %.2f, transformer %.2f' % (accuracy(bag, 'correction'), accuracy(lm, 'correction')))
```

## 3. It is a language model too

The same transformer scores and completes sentences. Perplexity is the effective number of words it is choosing between at each step;
guessing among the vocabulary would be 32.

```python
print('perplexity %.2f (uniform guessing: %d)' % (lm.perplexity(C.make_corpus(800, seed=6)), len(C.WORDS)))
for prefix in ('please go to the', 'move over to the red one no'):
    print('"%s" ->  "%s"' % (prefix, lm.complete(prefix)))
assert lm.perplexity(C.make_corpus(800, seed=6)) < 8
```

## 4. Closed loop: the one-motor VLA follows the belief

The model's belief over cells picks the target of the one-motor VLA's action head. Here the head is a plain PD controller, to isolate the
language part; any action head with the same signature works.

```python
act = lambda target, theta, omega: expert_command(target, theta, omega)
corrections = [u for u in C.make_corpus(400, seed=8) if u.kind == 'correction' and not u.ambiguous][:60]
def success(model):
    rng = np.random.default_rng(0)
    return np.mean([run_episode(u, rng, language_plan(model, act))['success'] for u in corrections])
print('correction sentences, closed-loop success: bag %.2f, transformer %.2f' % (success(bag), success(lm)))
assert success(lm) > success(bag) + 0.2
```

## 5. A local LLM on a short leash

A general LLM understands free language the small model never saw ("could you possibly nip across to the left thing there"), but it can also
hallucinate. `LLMBridge` uses it twice and checks both uses: the LLM rewrites the request into the robot's own 32 words (accepted only if every word is
in the vocabulary), and it gives its own reading as a small JSON object (accepted only if the value is one of six). If the two readings disagree, or the
reply is unusable, the robot asks the person. `generate` is any `str -> str` function; for a model on your machine use
`quantum_mind.embodied.ollama_generate('llama3.2')`. Here the replies are **scripted**.

```python
u = next(x for x in C.make_corpus(300, seed=7) if x.kind == 'simple' and not x.ambiguous and len(x.cells) == 3)
word = next(w for w in u.text.split() if w in C.COLORS or w in C.LEFT_WORDS or w in C.RIGHT_WORDS)
reference = {'left': 'leftmost', 'first': 'leftmost', 'right': 'rightmost', 'last': 'rightmost'}.get(word, word)
free = 'could you possibly nip across to the %s thing there' % word

def scripted(rewrite, reference):
    return lambda prompt: rewrite if 'Rewrite' in prompt else '{"reference": "%s"}' % reference

contradicting = next(r for r in ('red', 'green', 'blue', 'leftmost', 'rightmost')
                     if belief_from_parse(r, u.cells, u.colors) is not None and belief_from_parse(r, u.cells, u.colors)[u.target] == 0)
cases = {'honest LLM': scripted(u.text, reference),
         'LLM that contradicts the small model': scripted(u.text, contradicting),
         'LLM that invents words': scripted('please go to the quantum zebra', reference),
         'LLM that rambles': lambda prompt: 'I am not able to help with that.'}
for name, generate in cases.items():
    r = LLMBridge(lm, generate).ground(free, u.image, u.cells, u.colors)
    print('%-38s ask the person: %-5s | %s' % (name, r.ask, r.reason))
honest = LLMBridge(lm, cases['honest LLM']).ground(free, u.image, u.cells, u.colors)
assert not honest.ask and np.argmax(honest.belief) in u.valid
assert LLMBridge(lm, cases['LLM that contradicts the small model']).ground(free, u.image, u.cells, u.colors).ask
assert LLMBridge(lm, cases['LLM that invents words']).ground(free, u.image, u.cells, u.colors).ask
assert LLMBridge(lm, cases['LLM that rambles']).ground(free, u.image, u.cells, u.colors).ask
```

## 6. Two turns of a conversation, fused by the quantum-cognition core

An instruction ("go to the block") is often followed by a clarification ("no, the left one") that sometimes misspeaks. The language model reads
each turn into a belief over the objects; `DialogueBelief` feeds those beliefs to the quantum-cognition core of tutorial 17 (with the kind of turn
as its context) or to its classical competitors. In this simulated conversation the follow-up misspeaks more often after a clear instruction, a
classical context effect, so no quantum advantage is expected.

```python
dlg_train, dlg_test = make_dialogues(600, seed=0), make_dialogues(400, seed=1)
for fusion in ('bayes', 'context weights', 'quantum core'):
    db = DialogueBelief(lm, fusion, steps=200, restarts=2).fit(dlg_train)
    print('%-16s held-out log-loss %.3f, accuracy %.3f' % (fusion, db.log_loss(dlg_test), db.accuracy(dlg_test)))
```

**Summary.** A 29k-parameter transformer trained from scratch reads corrections that a bag of words cannot, and it is a (tiny) language model too.
A general LLM can widen what the robot understands, but only through a bridge that rejects unusable replies and turns a disagreement into a question.
What this is not: a model of real human language, a test with a real LLM, or a claim that quantum probability helps here. The next steps are
real sentences from real people, and running the bridge against a model on your own machine.
