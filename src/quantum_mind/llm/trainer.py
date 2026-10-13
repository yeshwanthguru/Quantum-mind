"""Training a :class:`~quantum_mind.llm.model.LanguageModel` from scratch: data, optimiser, evaluation, checkpoints.

* :class:`TextData` turns documents into a token stream and samples random windows (ordinary language-model training).
* :class:`ChatData` turns ``(question, answer)`` pairs or whole conversations into examples in the chat format and trains
**only on the answer
  tokens**, which is how a model learns to answer rather than to continue.
* :class:`MixedData` mixes sources, for example web-style text with question-answer pairs.
* :func:`train` runs AdamW with warmup and cosine decay, gradient clipping, gradient accumulation (a large effective batch on a small GPU),
  held-out evaluation, throughput logging, and checkpoints it can resume from.

The same code runs on a CPU or on a GPU (JAX picks the device); ``compute_dtype='bfloat16'`` in the model config uses the
GPU's fast matrix units.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .._optional import require
from .model import format_chat, loss_fn

jax = require('jax', extra='cognition')
optax = require('optax', extra='cognition')
import jax.numpy as jnp  # noqa: E402

__all__ = ['TextData', 'ChatData', 'MixedData', 'train']


class TextData:
    """Random windows of ``block_size + 1`` tokens from a corpus of documents, split into training and held-out documents."""

    def __init__(self, texts, tokenizer, block_size, val_fraction=0.05, seed=0):
        docs = [np.array(tokenizer.encode(t, bos=True, eos=True), np.int32) for t in texts if t.strip()]
        if not docs:
            raise ValueError('no text to train on')
        order = np.random.default_rng(seed).permutation(len(docs))
        n_val = max(int(len(docs) * val_fraction), 1) if len(docs) > 1 else 0
        self.val = np.concatenate([docs[i] for i in order[:n_val]]) if n_val else None
        self.train = np.concatenate([docs[i] for i in order[n_val:]])
        self.block, self.pad = block_size, tokenizer.pad_id
        if len(self.train) <= block_size + 1:
            raise ValueError('the corpus (%d tokens) must be longer than block_size + 1 = %d' % (len(self.train), block_size + 1))

    @property
    def n_tokens(self):
        return len(self.train)

    def _windows(self, stream, starts):
        w = np.stack([stream[s:s + self.block + 1] for s in starts])
        return w, np.ones_like(w)

    def sample(self, batch_size, rng):
        starts = rng.integers(0, len(self.train) - self.block - 1, batch_size)
        return self._windows(self.train, starts)

    def validation(self, n_batches, batch_size):
        if self.val is None or len(self.val) <= self.block + 1:
            return []
        rng = np.random.default_rng(12345)
        return [self._windows(self.val, rng.integers(0, len(self.val) - self.block - 1, batch_size)) for _ in range(n_batches)]


class ChatData:
    """Question-answer examples in the chat format, trained on the answer only.

    Parameters
    ----------
    conversations : list
        Each item is a ``(question, answer)`` pair or a list of ``{"role", "content"}`` messages ending with an assistant message.
    tokenizer, block_size : as for the model
    system : str or None
        Optional system text put in front of every first user message.
    """

    def __init__(self, conversations, tokenizer, block_size, system=None, val_fraction=0.05, seed=0):
        self.block, self.pad = block_size, tokenizer.pad_id
        ex = []
        for c in conversations:
            msgs = [{'role': 'user', 'content': c[0]}, {'role': 'assistant', 'content': c[1]}] if isinstance(c, tuple) else list(c)
            if system:
                msgs = [{'role': 'system', 'content': system}] + msgs
            ids, mask = [], []
            prompt = format_chat(tokenizer, msgs[:-1])
            answer = tokenizer.encode(msgs[-1]['content']) + [tokenizer.eos_id]
            ids, mask = prompt + answer, [0] * len(prompt) + [1] * len(answer)
            if len(ids) <= block_size + 1:
                ex.append((ids, mask))
        if not ex:
            raise ValueError('no example fits in block_size + 1 = %d tokens' % (block_size + 1))
        order = np.random.default_rng(seed).permutation(len(ex))
        n_val = max(int(len(ex) * val_fraction), 1) if len(ex) > 1 else 0
        self.val = [ex[i] for i in order[:n_val]]
        self.train = [ex[i] for i in order[n_val:]]

    @property
    def n_tokens(self):
        return sum(len(i) for i, _ in self.train)

    def _batch(self, items):
        tok = np.full((len(items), self.block + 1), self.pad, np.int32)
        msk = np.zeros_like(tok)
        for r, (ids, mask) in enumerate(items):
            tok[r, :len(ids)] = ids
            msk[r, :len(ids)] = mask
        return tok, msk

    def sample(self, batch_size, rng):
        return self._batch([self.train[i] for i in rng.integers(0, len(self.train), batch_size)])

    def validation(self, n_batches, batch_size):
        if not self.val:
            return []
        rng = np.random.default_rng(12345)
        return [self._batch([self.val[i] for i in rng.integers(0, len(self.val), batch_size)]) for _ in range(n_batches)]


class MixedData:
    """Draw each training example from one of several sources with the given weights (for example ``[(text, 0.7), (chat, 0.3)]``)."""

    def __init__(self, sources):
        self.sources = [s for s, _ in sources]
        w = np.array([w for _, w in sources], float)
        self.p = w / w.sum()

    @property
    def n_tokens(self):
        return sum(s.n_tokens for s in self.sources)

    def sample(self, batch_size, rng):
        pick = rng.choice(len(self.sources), batch_size, p=self.p)
        parts = [self.sources[k].sample(int((pick == k).sum()), rng) for k in range(len(self.sources)) if (pick == k).any()]
        return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])

    def validation(self, n_batches, batch_size):
        return [b for s in self.sources for b in s.validation(n_batches, batch_size)]


def _decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)


def train(model, data, steps=1000, batch_size=32, lr=3e-4, warmup=100, weight_decay=0.1, grad_clip=1.0, accum=1, eval_every=200,
          eval_batches=8, log_every=50, seed=0, checkpoint_dir=None, checkpoint_every=0, resume=False, callback=None, quiet=False):
    """Train ``model`` in place and return the history (a list of dicts with ``step``, ``loss``, ``val_loss``, ``tokens_per_s``).

    Parameters
    ----------
    model : LanguageModel
    data : TextData, ChatData or MixedData
    steps : int
        Optimiser steps; each step sees ``batch_size * accum`` sequences.
    batch_size : int
        Sequences per micro-batch (what fits in memory). ``accum`` micro-batches are averaged into one step.
    lr, warmup, weight_decay, grad_clip : optional
        Peak learning rate (cosine decay to 10% after ``warmup`` linear steps), AdamW weight decay on matrices, global-norm clipping.
    eval_every, eval_batches : optional
        Held-out loss every ``eval_every`` steps over ``eval_batches`` batches.
    checkpoint_dir, checkpoint_every, resume : optional
        Save the model and optimiser state every ``checkpoint_every`` steps; ``resume=True`` continues from the last checkpoint in
        the directory.
    callback : callable, optional
        Called as ``callback(record)`` at every log.
    """
    cfg = model.cfg
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=weight_decay,
    mask=_decay_mask))
    params = model.params
    state = opt.init(params)
    start = 0
    if resume and checkpoint_dir and (Path(checkpoint_dir) / 'opt_state.npz').exists():
        saved = np.load(Path(checkpoint_dir) / 'opt_state.npz')
        leaves, treedef = jax.tree_util.tree_flatten(state)
        state = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(saved['l%d' % i]) for i in range(len(leaves))])
        start = int(json.loads((Path(checkpoint_dir) / 'train_state.json').read_text())['step'])
        from .model import LanguageModel
        params = LanguageModel.load(checkpoint_dir).params
    dropout_on = cfg.dropout > 0

    def micro_loss(p, tokens, mask, key):
        return loss_fn(cfg, p, tokens, mask, key if dropout_on else None)

    @jax.jit
    def step_fn(p, s, tokens, mask, key):
        def one(carry, xs):
            total_l, total_g = carry
            t, m, k = xs
            l, g = jax.value_and_grad(micro_loss)(p, t, m, k)
            return (total_l + l, jax.tree_util.tree_map(jnp.add, total_g, g)), None
        zero = jax.tree_util.tree_map(jnp.zeros_like, p)
        keys = jax.random.split(key, tokens.shape[0])
        (l, g), _ = jax.lax.scan(one, (0.0, zero), (tokens, mask, keys))
        g = jax.tree_util.tree_map(lambda x: x / tokens.shape[0], g)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l / tokens.shape[0]

    val_fn = jax.jit(lambda p, t, m: loss_fn(cfg, p, t, m))
    val = data.validation(eval_batches, batch_size)
    rng = np.random.default_rng(seed + start)
    key = jax.random.PRNGKey(seed)
    history, window, t0, seen = [], [], time.time(), 0
    if not quiet:
        print('device: %s | %d parameters | %d training tokens | %d steps x %d sequences' %
              (jax.devices()[0], model.n_params, data.n_tokens, steps, batch_size * accum))
    for step in range(start + 1, steps + 1):
        batches = [data.sample(batch_size, rng) for _ in range(accum)]
        toks = jnp.asarray(np.stack([b[0] for b in batches]))
        masks = jnp.asarray(np.stack([b[1] for b in batches]))
        key, sub = jax.random.split(key)
        params, state, loss = step_fn(params, state, toks, masks, sub)
        window.append(float(loss))
        seen += int(masks.sum())
        if step % log_every == 0 or step == steps:
            rec = dict(step=step, loss=float(np.mean(window)), tokens_per_s=seen / max(time.time() - t0, 1e-9), lr=float(sched(step)))
            window, seen, t0 = [], 0, time.time()
            if val and (step % eval_every == 0 or step == steps):
                rec['val_loss'] = float(np.mean([float(val_fn(params, jnp.asarray(t), jnp.asarray(m))) for t, m in val]))
            history.append(rec)
            if not quiet:
                print('step %5d  loss %.3f%s  %.0f tok/s' % (step, rec['loss'], '  val %.3f' % rec['val_loss'] if 'val_loss' in rec else '',
                                                           rec['tokens_per_s']))
            if callback:
                callback(rec)
        if checkpoint_dir and checkpoint_every and step % checkpoint_every == 0:
            model.params, model.step = params, step
            model.save(checkpoint_dir)
            leaves, _ = jax.tree_util.tree_flatten(state)
            np.savez(Path(checkpoint_dir) / 'opt_state.npz', **{'l%d' % i: np.asarray(x) for i, x in enumerate(leaves)})
            (Path(checkpoint_dir) / 'train_state.json').write_text(json.dumps({'step': step}))
    model.params, model.step = params, steps
    model._fwd = None
    return history
