"""A GPT-style language model written from scratch in JAX: no pretrained weights, no model library.

:class:`LanguageModel` bundles a config, a :class:`~quantum_mind.llm.tokenizer.BPETokenizer` and the parameters. The
architecture is a decoder-only
transformer (token embedding tied to the output, learned positions, pre-layer-norm blocks, causal multi-head attention, GELU
MLP). It runs on
whatever device JAX finds: CPU by default, a GPU if you install the CUDA build of JAX (``pip install "jax[cuda12]"``).
``compute_dtype='bfloat16'``
keeps the parameters in float32 and does the matrix multiplications in bfloat16, which is how a GPU gets its speed.

Size presets (parameters excluding the embedding, for a vocabulary of 8,192; add ``vocab_size * d_model`` for the embedding):

======== ======= ======= ======= ========== =============================
preset   d_model layers  heads   block      what it is for
======== ======= ======= ======= ========== =============================
tiny     128     4       4       128        CPU, a few minutes, tests
small    384     6       6       256        ~11M, one GPU, hours
base     768     12      12      512        ~85M, one GPU, days
======== ======= ======= ======= ========== =============================

What a model learns is what its data contains. A tiny corpus gives a tiny model that echoes its corpus; only far more text
(billions of tokens)
and GPU-days give a model that knows a lot of the world. Needs the optional ``cognition`` extra (JAX, Optax).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .._optional import require
from .tokenizer import BPETokenizer

jax = require('jax', extra='cognition')
import jax.numpy as jnp  # noqa: E402

__all__ = ['GPTConfig', 'LanguageModel', 'PRESETS', 'format_chat']

PRESETS = {'tiny': dict(d_model=128, n_layers=4, n_heads=4, block_size=128),
           'small': dict(d_model=384, n_layers=6, n_heads=6, block_size=256),
           'base': dict(d_model=768, n_layers=12, n_heads=12, block_size=512)}


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    block_size: int = 128
    dropout: float = 0.0
    compute_dtype: str = 'float32'

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')


def format_chat(tokenizer, messages, add_generation_prompt=True):
    """Token ids for a chat: ``<bos>`` then, per message, ``<user>`` or ``<assistant>`` and the text; answers end with ``<eos>``.

    A ``system`` message is placed in front of the first user message as plain text. With ``add_generation_prompt`` the ids end with
    ``<assistant>``, ready for the model to write the answer.
    """
    ids = [tokenizer.bos_id]
    system = ''
    for m in messages:
        role, text = m['role'], m['content']
        if role == 'system':
            system = text.strip() + '\n\n'
        elif role == 'user':
            ids += [tokenizer.user_id] + tokenizer.encode(system + text)
            system = ''
        else:
            ids += [tokenizer.assistant_id] + tokenizer.encode(text) + [tokenizer.eos_id]
    if add_generation_prompt and messages and messages[-1]['role'] != 'assistant':
        ids.append(tokenizer.assistant_id)
    return ids


def _ln(x, g, b):
    x32 = x.astype(jnp.float32)
    m = x32.mean(-1, keepdims=True)
    v = ((x32 - m) ** 2).mean(-1, keepdims=True)
    return ((x32 - m) / jnp.sqrt(v + 1e-5) * g + b).astype(x.dtype)


def init_params(cfg, key):
    """Parameters of a freshly initialised model (normal 0.02; residual projections scaled by 1/sqrt(2 layers))."""
    d, V, L = cfg.d_model, cfg.vocab_size, cfg.block_size
    ks = iter(jax.random.split(key, 2 + 4 * cfg.n_layers))
    nrm = lambda shape, s=0.02: s * jax.random.normal(next(ks), shape)               # noqa: E731
    res = 0.02 / np.sqrt(2 * cfg.n_layers)
    p = {'wte': nrm((V, d)), 'wpe': nrm((L, d)), 'lnf_g': jnp.ones(d), 'lnf_b': jnp.zeros(d), 'layers': []}
    for _ in range(cfg.n_layers):
        p['layers'].append({'ln1_g': jnp.ones(d), 'ln1_b': jnp.zeros(d), 'Wqkv': nrm((d, 3 * d)), 'bqkv': jnp.zeros(3 * d),
                            'Wo': nrm((d, d), res), 'bo': jnp.zeros(d), 'ln2_g': jnp.ones(d), 'ln2_b': jnp.zeros(d),
                            'W1': nrm((d, 4 * d)), 'b1': jnp.zeros(4 * d), 'W2': nrm((4 * d, d), res), 'b2': jnp.zeros(d)})
    return p


def forward(cfg, params, tokens, key=None):
    """Logits ``(B, T, vocab)`` for ``tokens`` ``(B, T)`` with ``T <= block_size``. ``key`` turns dropout on (training)."""
    B, T = tokens.shape
    cdt = jnp.dtype(cfg.compute_dtype)

    def drop(x, k):
        if key is None or cfg.dropout == 0:
            return x
        return jnp.where(jax.random.bernoulli(k, 1 - cfg.dropout, x.shape), x / (1 - cfg.dropout), 0)

    keys = iter(jax.random.split(key, 1 + 2 * cfg.n_layers)) if key is not None else iter([None] * (1 + 2 * cfg.n_layers))
    x = (params['wte'][tokens] + params['wpe'][None, :T]).astype(cdt)
    x = drop(x, next(keys))
    H, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
    mask = jnp.tril(jnp.ones((T, T), bool))
    for lyr in params['layers']:
        h = _ln(x, lyr['ln1_g'], lyr['ln1_b'])
        qkv = h @ lyr['Wqkv'].astype(cdt) + lyr['bqkv'].astype(cdt)
        q, k, v = (a.reshape(B, T, H, dh).transpose(0, 2, 1, 3) for a in jnp.split(qkv, 3, axis=-1))
        att = (q @ k.transpose(0, 1, 3, 2)).astype(jnp.float32) / np.sqrt(dh)
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), axis=-1).astype(cdt)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model) @ lyr['Wo'].astype(cdt) + lyr['bo'].astype(cdt)
        x = x + drop(y, next(keys))
        h = _ln(x, lyr['ln2_g'], lyr['ln2_b'])
        y = jax.nn.gelu(h @ lyr['W1'].astype(cdt) + lyr['b1'].astype(cdt)) @ lyr['W2'].astype(cdt) + lyr['b2'].astype(cdt)
        x = x + drop(y, next(keys))
    x = _ln(x, params['lnf_g'], params['lnf_b'])
    return (x @ params['wte'].astype(cdt).T).astype(jnp.float32)


def loss_fn(cfg, params, tokens, mask, key=None):
    """Mean next-token cross-entropy over the positions where ``mask`` is 1 (``tokens`` ``(B, T+1)``: inputs are ``[:, :-1]``)."""
    logits = forward(cfg, params, tokens[:, :-1], key)
    lp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(lp, tokens[:, 1:, None], axis=-1)[..., 0]
    m = mask[:, 1:].astype(jnp.float32)
    return (nll * m).sum() / jnp.maximum(m.sum(), 1.0)


def _flatten(params):
    out = {}

    def rec(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                rec('%s.%s' % (prefix, k) if prefix else k, v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec('%s.%d' % (prefix, i), v)
        else:
            out[prefix] = np.asarray(node)
    rec('', params)
    return out


def _unflatten(flat, template):
    def rec(prefix, node):
        if isinstance(node, dict):
            return {k: rec('%s.%s' % (prefix, k) if prefix else k, v) for k, v in node.items()}
        if isinstance(node, list):
            return [rec('%s.%d' % (prefix, i), v) for i, v in enumerate(node)]
        return jnp.asarray(flat[prefix])
    return rec('', template)


class LanguageModel:
    """A tokenizer, a config and the parameters of a GPT-style model.

    Create one with :meth:`new` (random weights, ready to train with :func:`quantum_mind.llm.train`), write it with :meth:`save`
    and read it back with
    :meth:`load`. :meth:`generate`, :meth:`stream` and :meth:`chat` produce text; :meth:`perplexity` scores it.
    """

    def __init__(self, cfg, tokenizer, params=None, seed=0):
        if cfg.vocab_size < tokenizer.vocab_size:
            raise ValueError('vocab_size is smaller than the tokenizer vocabulary')
        self.cfg, self.tokenizer = cfg, tokenizer
        self.params = params if params is not None else init_params(cfg, jax.random.PRNGKey(seed))
        self.step = 0
        self._fwd = None

    @classmethod
    def new(cls, tokenizer, preset='tiny', seed=0, **overrides):
        """A randomly initialised model sized by a preset (``'tiny'``, ``'small'``, ``'base'``) with optional overrides such as
        ``dropout=0.1``."""
        cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **{**PRESETS[preset], **overrides})
        return cls(cfg, tokenizer, seed=seed)

    @property
    def n_params(self):
        return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(self.params)))

    # ---- scoring and generation ----------------------------------------------------------------------------------------
    def logits(self, tokens):
        """Logits for a ``(B, T)`` batch of token ids."""
        if self._fwd is None:
            self._fwd = jax.jit(lambda p, t: forward(self.cfg, p, t))
        return self._fwd(self.params, jnp.asarray(tokens))

    def perplexity(self, text):
        """Per-token perplexity of ``text`` (split into windows of ``block_size``)."""
        ids = self.tokenizer.encode(text, bos=True, eos=True)
        total, n = 0.0, 0
        B = self.cfg.block_size
        for s in range(0, len(ids) - 1, B):
            w = ids[s:s + B + 1]
            if len(w) < 2:
                break
            lg = np.asarray(self.logits(np.array([w[:-1]])))[0]
            lg = lg - lg.max(-1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
            total += -lp[np.arange(len(w) - 1), w[1:]].sum()
            n += len(w) - 1
        return float(np.exp(total / n))

    def _next_logits(self, ids):
        B = self.cfg.block_size
        ids = ids[-B:]
        buf = np.zeros((1, B), np.int32)
        buf[0, :len(ids)] = ids
        return np.asarray(self.logits(buf))[0, len(ids) - 1]

    def stream_ids(self, ids, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.95, seed=0, stop_ids=None):
        """Yield new token ids one at a time after the prompt ``ids``. ``temperature=0`` is greedy decoding."""
        rng = np.random.default_rng(seed)
        ids = list(ids)
        stop = set(stop_ids if stop_ids is not None else [self.tokenizer.eos_id])
        for _ in range(max_new_tokens):
            lg = self._next_logits(ids).astype(np.float64)
            # never write pad, bos, user or assistant; eos only if it may stop
            for special in range(self.tokenizer.n_special):
                if special != self.tokenizer.eos_id or self.tokenizer.eos_id not in stop:
                    lg[special] = -1e9
            lg[self.tokenizer.vocab_size:] = -1e9
            if temperature <= 0:
                nxt = int(np.argmax(lg))
            else:
                lg = lg / temperature
                if top_k and top_k < len(lg):
                    lg[lg < np.sort(lg)[-top_k]] = -1e9
                p = np.exp(lg - lg.max())
                p /= p.sum()
                if top_p and top_p < 1.0:
                    order = np.argsort(-p)
                    cum = np.cumsum(p[order])
                    cut = order[np.searchsorted(cum, top_p) + 1:]
                    p[cut] = 0
                    p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            if nxt in stop:
                return
            ids.append(nxt)
            yield nxt

    def stream(self, prompt, **kw):
        """Yield the text of a continuation of ``prompt`` piece by piece (multi-byte characters are held back until they are complete)."""
        out, sent = [], ''
        for t in self.stream_ids(self.tokenizer.encode(prompt, bos=True), **kw):
            out.append(t)
            text = self.tokenizer.decode(out)
            if not text.endswith('�'):
                yield text[len(sent):]
                sent = text

    def generate(self, prompt, **kw):
        """A continuation of ``prompt`` as one string."""
        return ''.join(self.stream(prompt, **kw))

    def chat(self, messages, **kw):
        """The model's answer to a list of ``{"role", "content"}`` messages, using the chat format it was trained on."""
        return ''.join(self.stream_chat(messages, **kw))

    def stream_chat(self, messages, **kw):
        out, sent = [], ''
        for t in self.stream_ids(format_chat(self.tokenizer, messages), **kw):
            out.append(t)
            text = self.tokenizer.decode(out)
            if not text.endswith('�'):
                yield text[len(sent):]
                sent = text

    # ---- files ---------------------------------------------------------------------------------------------------------
    def save(self, directory):
        """Write ``config.json``, ``tokenizer.json`` and ``weights.npz`` to a directory."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / 'config.json').write_text(json.dumps({**asdict(self.cfg), 'step': self.step}))
        self.tokenizer.save(d / 'tokenizer.json')
        np.savez(d / 'weights.npz', **_flatten(self.params))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        raw = json.loads((d / 'config.json').read_text())
        step = raw.pop('step', 0)
        cfg = GPTConfig(**raw)
        tok = BPETokenizer.load(d / 'tokenizer.json')
        flat = dict(np.load(d / 'weights.npz'))
        m = cls(cfg, tok, params=_unflatten(flat, init_params(cfg, jax.random.PRNGKey(0))))
        m.step = step
        return m
