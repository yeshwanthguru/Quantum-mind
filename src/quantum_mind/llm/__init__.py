"""Train your own language model from scratch: no pretrained weights, no other model.

* :class:`BPETokenizer`: a byte-level BPE tokenizer you train on your own text (pure Python, no JAX).
* :class:`LanguageModel`, :class:`GPTConfig`: a GPT-style transformer in JAX, with ``tiny``, ``small`` and ``base`` presets;
runs on CPU or on a
  GPU (``pip install "jax[cuda12]"``), with optional bfloat16 compute.
* :func:`train` with :class:`TextData`, :class:`ChatData`, :class:`MixedData`: AdamW, warmup and cosine decay, gradient accumulation,
  held-out evaluation, resumable checkpoints; ChatData trains only on the answer tokens.
* :func:`library_corpus`, :func:`read_corpus`: data from this library's docstrings, or from your files.
* ``quantum-mind-llm`` (train, sample, info) and ``quantum-mind-chat --backend scratch --checkpoint DIR``.

What a model learns is what its data contains. The model code needs the optional ``cognition`` extra (JAX, Optax); the tokenizer does not.
"""
from .corpus import library_corpus, read_corpus, read_qa
from .tokenizer import SPECIAL_TOKENS, BPETokenizer

_LAZY = {'LanguageModel': 'model', 'GPTConfig': 'model', 'PRESETS': 'model', 'format_chat': 'model', 'train': 'trainer',
'TextData': 'trainer',
         'ChatData': 'trainer', 'MixedData': 'trainer'}

__all__ = ['BPETokenizer', 'SPECIAL_TOKENS', 'library_corpus', 'read_corpus', 'read_qa', *_LAZY]


def __getattr__(name):                                      # the JAX parts load only when used, so the tokenizer works without JAX
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module('.' + _LAZY[name], __name__), name)
    raise AttributeError(name)
