"""Backends for the chat program: a local LLM served by Ollama, a local GGUF file through llama.cpp, or a scripted stand-in.

Every backend has the same two methods:

* ``stream(messages, **options)`` yields the answer piece by piece, for a list of ``{"role": ..., "content": ...}`` messages;
* ``generate(prompt, **options)`` returns the whole answer to one prompt (this is what :class:`quantum_mind.embodied.LocalLLMBrain` and
  :class:`quantum_mind.cognition.LLMBridge` take as their ``generate`` function).

Everything stays on your machine: the Ollama backend refuses a host that is not local unless you say ``allow_remote=True``. The
library ships no model. Install one yourself (for example ``ollama pull llama3.2``), or give ``LlamaCppBackend`` a GGUF file."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

__all__ = ['ChatError', 'OllamaBackend', 'LlamaCppBackend', 'ScratchBackend', 'ScriptedBackend', 'list_ollama_models']

LOCAL_HOSTS = ('localhost', '127.0.0.1', '::1', '[::1]')


class ChatError(RuntimeError):
    """A problem the user can fix (no server, no model, no package), with the fix in the message."""


class _Backend:
    def generate(self, prompt, **options):
        """The whole answer to a single prompt."""
        return ''.join(self.stream([{'role': 'user', 'content': prompt}], **options))


def _check_local(host, allow_remote):
    name = urllib.parse.urlparse(host).hostname or ''
    if name not in LOCAL_HOSTS and not allow_remote:
        raise ChatError('%r is not a local address; this program keeps your questions on your machine. Pass allow_remote=True '
                        '(--allow-remote) if you really want to send them to another host.' % host)


def list_ollama_models(host='http://localhost:11434', timeout=5.0, allow_remote=False):
    """Names of the models an Ollama server has pulled."""
    _check_local(host, allow_remote)
    try:
        with urllib.request.urlopen(host.rstrip('/') + '/api/tags', timeout=timeout) as r:      # noqa: S310 - local server
            return [m['name'] for m in json.loads(r.read()).get('models', [])]
    except (urllib.error.URLError, OSError) as e:
        raise ChatError('cannot reach Ollama at %s (%s). Install it from https://ollama.com, start it, and pull a model, for example: '
                        'ollama pull llama3.2' % (host, getattr(e, 'reason', e))) from e


class OllamaBackend(_Backend):
    """A model served by Ollama on this machine (``/api/chat`` with streaming).

    Parameters
    ----------
    model : str
        Name of a pulled model, for example ``'llama3.2'`` or ``'qwen2.5'``.
    host : str
        Where Ollama listens (default ``http://localhost:11434``).
    timeout : float
        Seconds to wait for the server to answer (a big model can take a while to load the first time).
    allow_remote : bool
        Allow a host that is not this machine.
    """

    def __init__(self, model='llama3.2', host='http://localhost:11434', timeout=300.0, allow_remote=False):
        _check_local(host, allow_remote)
        self.model, self.host, self.timeout = model, host.rstrip('/'), timeout

    def stream(self, messages, temperature=None, max_tokens=None, **_):
        options = {}
        if temperature is not None:
            options['temperature'] = temperature
        if max_tokens is not None:
            options['num_predict'] = max_tokens
        body = json.dumps({'model': self.model, 'messages': list(messages), 'stream': True, 'options': options}).encode()
        req = urllib.request.Request(self.host + '/api/chat', body, {'Content-Type': 'application/json'})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)                              # noqa: S310 - local server
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors='replace')[:300]
            if e.code == 404:
                raise ChatError('Ollama has no model %r. Pull it first: ollama pull %s (%s)' % (self.model, self.model, detail)) from e
            raise ChatError('Ollama returned HTTP %d: %s' % (e.code, detail)) from e
        except (urllib.error.URLError, OSError) as e:
            raise ChatError('cannot reach Ollama at %s (%s). Start it (ollama serve) or install it from https://ollama.com'
                            % (self.host, getattr(e, 'reason', e))) from e
        with resp:
            for line in resp:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if 'error' in msg:
                    raise ChatError('Ollama: %s' % msg['error'])
                piece = msg.get('message', {}).get('content', '')
                if piece:
                    yield piece
                if msg.get('done'):
                    return


class LlamaCppBackend(_Backend):
    """A local GGUF model file run in-process by llama.cpp (``pip install llama-cpp-python``).

    Parameters
    ----------
    model_path : str
        Path to a ``.gguf`` file you have downloaded.
    n_ctx : int
        Context length.
    """

    def __init__(self, model_path, n_ctx=4096, **kw):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ChatError('llama-cpp-python is not installed: pip install llama-cpp-python') from e
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kw)

    def stream(self, messages, temperature=None, max_tokens=None, **_):
        kw = {}
        if temperature is not None:
            kw['temperature'] = temperature
        if max_tokens is not None:
            kw['max_tokens'] = max_tokens
        for chunk in self.llm.create_chat_completion(messages=list(messages), stream=True, **kw):
            piece = chunk['choices'][0].get('delta', {}).get('content')
            if piece:
                yield piece


class ScratchBackend(_Backend):
    """A model you trained yourself with :mod:`quantum_mind.llm` (``quantum-mind-llm train``), loaded from its checkpoint directory.

    Parameters
    ----------
    checkpoint : str
        Directory written by ``LanguageModel.save`` (``config.json``, ``tokenizer.json``, ``weights.npz``).
    temperature, top_k, top_p : float, optional
        Sampling settings (``temperature=0`` is greedy).
    """

    def __init__(self, checkpoint, temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=200):
        try:
            from ..llm import LanguageModel
        except ImportError as e:
            raise ChatError('your own model needs JAX: pip install "quantum-mind[cognition]"') from e
        if not (__import__('pathlib').Path(checkpoint) / 'weights.npz').exists():
            raise ChatError('no checkpoint in %r; train one with: quantum-mind-llm train --data <your text> --out %s' % (checkpoint,
            checkpoint))
        self.lm, self.opts = LanguageModel.load(checkpoint), dict(temperature=temperature, top_k=top_k, top_p=top_p,
        max_new_tokens=max_new_tokens)
        self.model = checkpoint

    def stream(self, messages, temperature=None, max_tokens=None, **_):
        opts = dict(self.opts)
        if temperature is not None:
            opts['temperature'] = temperature
        if max_tokens is not None:
            opts['max_new_tokens'] = max_tokens
        yield from self.lm.stream_chat(messages, **opts)


class ScriptedBackend(_Backend):
    """Canned replies, for tests and for trying the program with no model: ``ScriptedBackend(['first reply', 'second reply'])``.

    A reply can also be a function of the message list. After the script runs out it repeats the last reply. ``calls`` records every
    request.
    """

    def __init__(self, replies=('(demo backend) I am a stand-in, not a language model. Install Ollama and a model to get real answers.',)):
        self.replies, self.calls, self._i = list(replies), [], 0

    def stream(self, messages, temperature=None, max_tokens=None, **_):
        self.calls.append(dict(messages=[dict(m) for m in messages], temperature=temperature, max_tokens=max_tokens))
        r = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        text = r(messages) if callable(r) else r
        for word in text.split(' '):
            yield word + ' '
