"""A byte-level BPE tokenizer written from scratch (no pretrained vocabulary, no tokenizer library).

Training starts from the 256 byte values and repeatedly merges the most frequent adjacent pair of symbols inside words, until
the vocabulary has the
requested size. Because every string is a sequence of bytes, any text can be encoded and decoded exactly (``decode(encode(s))
== s``), including
other languages, emoji and code. Words are found by a small regular expression (a word with its leading space, a run of
digits, a run of symbols,
white space), so merges never cross word boundaries and training is fast in pure Python.

Special tokens (``<pad>``, ``<bos>``, ``<eos>``, ``<user>``, ``<assistant>``) take the first ids. No part of this depends on JAX.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

__all__ = ['BPETokenizer', 'SPECIAL_TOKENS']

SPECIAL_TOKENS = ('<pad>', '<bos>', '<eos>', '<user>', '<assistant>')
WORD = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+|.", re.UNICODE | re.DOTALL)


class BPETokenizer:
    """Byte-pair encoding over bytes.

    Parameters
    ----------
    merges : list of (int, int), optional
        Learned merges in order; token ``n_special + 256 + i`` is the merge ``i``.
    """

    def __init__(self, merges=()):
        self.n_special = len(SPECIAL_TOKENS)
        self.merges = [tuple(m) for m in merges]
        self.special = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.pad_id, self.bos_id, self.eos_id, self.user_id, self.assistant_id = (self.special[t] for t in SPECIAL_TOKENS)
        self._rank = {m: i for i, m in enumerate(self.merges)}
        self._cache = {}
        self._build_vocab()

    # ---- vocabulary ----------------------------------------------------------------------------------------------------
    def _build_vocab(self):
        base = self.n_special
        self.bytes_of = {i: SPECIAL_TOKENS[i].encode() for i in range(base)}
        for b in range(256):
            self.bytes_of[base + b] = bytes([b])
        for i, (a, b) in enumerate(self.merges):
            self.bytes_of[base + 256 + i] = self.bytes_of[a] + self.bytes_of[b]

    @property
    def vocab_size(self):
        return self.n_special + 256 + len(self.merges)

    # ---- training ------------------------------------------------------------------------------------------------------
    @classmethod
    def train(cls, texts, vocab_size=1024, min_count=2):
        """Learn merges from an iterable of strings until the vocabulary has ``vocab_size`` entries (or no pair occurs ``min_count``
        times)."""
        words = Counter()
        for t in texts:
            words.update(m.group(0) for m in WORD.finditer(t))
        base = len(SPECIAL_TOKENS)
        seqs = {w: [base + b for b in w.encode('utf-8')] for w in words}
        merges = []
        while base + 256 + len(merges) < vocab_size:
            pairs = Counter()
            for w, seq in seqs.items():
                n = words[w]
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += n
            if not pairs:
                break
            (a, b), count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < min_count:
                break
            new = base + 256 + len(merges)
            merges.append((a, b))
            for w, seq in seqs.items():
                if a in seq:
                    out, i = [], 0
                    while i < len(seq):
                        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                            out.append(new)
                            i += 2
                        else:
                            out.append(seq[i])
                            i += 1
                    seqs[w] = out
        return cls(merges)

    # ---- encoding and decoding -----------------------------------------------------------------------------------------
    def _word(self, w):
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        seq = [self.n_special + b for b in w.encode('utf-8')]
        while len(seq) > 1:
            best, best_rank = None, None
            for i, pair in enumerate(zip(seq, seq[1:])):
                r = self._rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            seq[best:best + 2] = [self.n_special + 256 + best_rank]
        self._cache[w] = seq
        return seq

    def encode(self, text, bos=False, eos=False):
        """Token ids for ``text`` (optionally wrapped in ``<bos>`` / ``<eos>``)."""
        ids = [self.bos_id] if bos else []
        for m in WORD.finditer(text):
            ids.extend(self._word(m.group(0)))
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special=True):
        """The text for ``ids`` (invalid UTF-8 produced by a model is replaced, never an error)."""
        out = b''.join(self.bytes_of[i] for i in ids if not (skip_special and i < self.n_special))
        return out.decode('utf-8', errors='replace')

    # ---- files ---------------------------------------------------------------------------------------------------------
    def save(self, path):
        Path(path).write_text(json.dumps({'type': 'bytelevel-bpe', 'special': list(SPECIAL_TOKENS), 'merges': self.merges}))

    @classmethod
    def load(cls, path):
        return cls([tuple(m) for m in json.loads(Path(path).read_text())['merges']])
