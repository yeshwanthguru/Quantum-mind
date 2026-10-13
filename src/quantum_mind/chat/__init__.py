"""Chat with a local LLM: type a question, press Enter, read the streamed answer; in a terminal or in a browser tab on this machine.

The library does not contain or train a general language model. This package is the interface to one that you run yourself (Ollama or a GGUF
file through llama.cpp), plus a conversation memory, an optional self-consistency check that says when the model is unsure, and the same
backend as a ``generate`` function for the library's robot supervisor and LLM bridge. Start it with ``quantum-mind-chat`` (see
:mod:`quantum_mind.chat.cli`).
"""
from .backends import ChatError, LlamaCppBackend, OllamaBackend, ScriptedBackend, list_ollama_models
from .cli import main
from .session import ChatSession, ConsistencyCheck, agreement
from .web import make_server, serve

__all__ = ['ChatSession', 'ConsistencyCheck', 'agreement', 'OllamaBackend', 'LlamaCppBackend', 'ScriptedBackend', 'ChatError',
           'list_ollama_models', 'main', 'make_server', 'serve']
