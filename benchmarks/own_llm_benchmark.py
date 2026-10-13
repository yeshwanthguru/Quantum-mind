"""A language model trained from scratch on this library's own docstrings: what it learns and what it does not (CPU run).

The corpus is the library's docstrings (about 370 documents) plus question-answer pairs about its classes and functions, with some names held out
entirely. The model is the ``tiny`` preset (under 1M parameters) trained from random weights; nothing pretrained is used anywhere. Reported:

* held-out perplexity on documents it never saw, against a bigram model fitted on the same training tokens (a model that only knows which token
  tends to follow which);
* question answering scored by word-overlap F1 against the true docstring sentence, for (a) questions it trained on, (b) a name it trained on in a
  phrasing it did not, (c) names whose question-answer pairs it never saw.

A model this small trained on this little text is expected to memorise (a) and to fail (c); the table is there to show exactly that.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_mind.chat.session import agreement
from quantum_mind.llm import BPETokenizer, ChatData, LanguageModel, MixedData, TextData, library_corpus, train


def bigram_loss(train_tokens, val_tokens, vocab, k=0.05):
    """Mean next-token cross-entropy of an add-k bigram model."""
    counts = np.full((vocab, vocab), k)
    np.add.at(counts, (train_tokens[:-1], train_tokens[1:]), 1.0)
    p = counts / counts.sum(1, keepdims=True)
    return float(-np.mean(np.log(p[val_tokens[:-1], val_tokens[1:]])))


def answer_scores(model, pairs, n, max_tokens=70):
    scores = []
    for q, a in pairs[:n]:
        out = model.chat([{'role': 'user', 'content': q}], temperature=0, max_new_tokens=max_tokens)
        scores.append(agreement(out, a))
    return scores


def run(steps, n_eval, out_dir):
    texts, pairs, held = library_corpus()
    tok = BPETokenizer.train(texts + [q + '\n' + a for q, a in pairs], 1024)
    model = LanguageModel.new(tok, 'tiny', seed=0)
    text, chat = TextData(texts, tok, model.cfg.block_size), ChatData(pairs, tok, model.cfg.block_size)
    t0 = time.time()
    hist = train(model, MixedData([(text, 0.5), (chat, 0.5)]), steps=steps, batch_size=16, lr=2e-3, warmup=100, eval_every=250, log_every=250)
    train_time = time.time() - t0
    rng = np.random.default_rng(0)
    probe = [pairs[i] for i in rng.permutation(len(pairs))[:n_eval]]
    res = dict(steps=steps, params=model.n_params, train_seconds=train_time, final_loss=hist[-1]['loss'], val_loss=[h['val_loss'] for h in hist if 'val_loss' in h],
               bigram_val_loss=bigram_loss(text.train, text.val, tok.vocab_size) if text.val is not None else None,
               qa={'questions seen in training': answer_scores(model, probe, n_eval),
                   'seen name, new phrasing': answer_scores(model, held['seen_names_new_phrasing'], n_eval),
                   'names never seen': answer_scores(model, held['unseen_names'], n_eval)})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(out_dir) / 'model')
    ex = [(q, model.chat([{'role': 'user', 'content': q}], temperature=0, max_new_tokens=70), a) for q, a in probe[:2] + held['unseen_names'][:2]]
    return res, ex


def to_markdown(res, ex):
    L = ['# A language model trained from scratch on this library\'s docstrings (CPU)', '',
         'A %d-parameter transformer from random weights, %d steps (%.0f minutes on a CPU). No pretrained model of any kind.' %
         (res['params'], res['steps'], res['train_seconds'] / 60), '',
         '| Measure | Value |', '|---|---|',
         '| held-out document loss, the model | %.3f (perplexity %.1f) |' % (res['val_loss'][-1], np.exp(res['val_loss'][-1]))]
    if res['bigram_val_loss'] is not None:
        L.append('| held-out document loss, a bigram model | %.3f (perplexity %.1f) |' % (res['bigram_val_loss'], np.exp(res['bigram_val_loss'])))
    L += ['', '## Question answering (word-overlap F1 with the true docstring sentence; 1.0 is a perfect copy)', '',
          '| Questions | Mean F1 | Share with F1 at least 0.8 |', '|---|---|---|']
    for k, v in res['qa'].items():
        L.append('| %s (%d) | %.2f | %.0f%% |' % (k, len(v), np.mean(v), 100 * np.mean(np.array(v) >= 0.8)))
    L += ['', '## Examples (greedy answers)', '']
    for q, got, want in ex:
        L += ['* **%s**\n  * model: %s\n  * truth: %s' % (q, got.strip() or '(empty)', want)]
    L += ['', 'Reading: the model memorises answers it trained on and cannot answer about names it never saw; a model this small with this little text '
          'learns the vocabulary and style of the library\'s documentation, not the knowledge in it.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--n-eval', type=int, default=100)
    ap.add_argument('--out', default='benchmarks/results/own_llm')
    a = ap.parse_args()
    res, ex = run(a.steps, a.n_eval, a.out)
    Path(a.out, 'benchmark_results.md').write_text(to_markdown(res, ex))
    Path(a.out, 'benchmark_raw.json').write_text(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))
    print(Path(a.out, 'benchmark_results.md').read_text())


if __name__ == '__main__':
    main()
