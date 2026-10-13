"""The terminal chat: type a question, press Enter, read the answer as it streams.

::

    quantum-mind-chat                                  # Ollama on this machine, model llama3.2
    quantum-mind-chat --model qwen2.5 --check          # another model, with the "am I sure?" check
    quantum-mind-chat --backend llamacpp --model-path ./model.gguf
    quantum-mind-chat --web                            # the same chat in a browser tab on this machine
    quantum-mind-chat --backend demo                   # no model: a stand-in, to see how it works

Commands inside the chat: ``/help``, ``/reset``, ``/model NAME``, ``/system TEXT``, ``/check on|off``, ``/save FILE``, ``/quit``.
Ctrl-C stops an answer that is being written; Ctrl-D or ``/quit`` leaves.
"""
from __future__ import annotations

import argparse
import sys

from .backends import ChatError, LlamaCppBackend, OllamaBackend, ScriptedBackend, list_ollama_models
from .session import DEFAULT_SYSTEM, ChatSession, ConsistencyCheck

HELP = '''commands: /help  /reset  /model NAME  /system TEXT  /check on|off  /save FILE  /quit
Type a question and press Enter. Ctrl-C stops an answer, Ctrl-D leaves.'''


def build_backend(args):
    if args.backend == 'demo':
        return ScriptedBackend()
    if args.backend == 'llamacpp':
        if not args.model_path:
            raise ChatError('--backend llamacpp needs --model-path FILE.gguf')
        return LlamaCppBackend(args.model_path)
    return OllamaBackend(args.model, args.host, allow_remote=args.allow_remote)


def parser():
    p = argparse.ArgumentParser(prog='quantum-mind-chat', description='Chat with a local LLM: type a question and press Enter.')
    p.add_argument('--backend', choices=('ollama', 'llamacpp', 'demo'), default='ollama')
    p.add_argument('--model', default='llama3.2', help='Ollama model name (default llama3.2)')
    p.add_argument('--host', default='http://localhost:11434', help='Ollama address (default http://localhost:11434)')
    p.add_argument('--model-path', help='GGUF file for --backend llamacpp')
    p.add_argument('--allow-remote', action='store_true', help='allow an Ollama host that is not this machine')
    p.add_argument('--system', default=DEFAULT_SYSTEM, help='system prompt')
    p.add_argument('--check', action='store_true', help='ask each question 3 times and warn when the answers disagree (slower)')
    p.add_argument('--max-tokens', type=int, default=None, help='longest answer')
    p.add_argument('--web', action='store_true', help='serve a chat page on 127.0.0.1 instead of the terminal')
    p.add_argument('--port', type=int, default=8765, help='port for --web (default 8765)')
    p.add_argument('--list-models', action='store_true', help='list the models Ollama has and exit')
    return p


def repl(session, stdin=None, stdout=None, prompt='you> '):
    """Read questions until the user leaves. ``stdin``/``stdout`` default to the terminal."""
    out = stdout or sys.stdout
    read = (lambda: (stdin.readline() or None)) if stdin is not None else None
    while True:
        try:
            if read:
                out.write(prompt)
                out.flush()
                line = read()
                if line is None:
                    raise EOFError
            else:
                line = input(prompt)
        except EOFError:
            out.write('\n')
            return
        except KeyboardInterrupt:
            out.write('\n(Ctrl-D or /quit to leave)\n')
            continue
        text = line.strip()
        if not text:
            continue
        if text.startswith('/'):
            if not command(session, text, out):
                return
            continue
        out.write('bot> ')
        out.flush()
        try:
            for piece in session.ask(text):
                out.write(piece)
                out.flush()
        except KeyboardInterrupt:
            out.write('\n(stopped)')
        except ChatError as e:
            out.write('\n[error] %s' % e)
        out.write('\n')


def command(session, text, out):
    """Handle a ``/command``; returns False to leave the chat."""
    word, _, rest = text.partition(' ')
    rest = rest.strip()
    if word in ('/quit', '/exit', '/q'):
        return False
    if word == '/help':
        out.write(HELP + '\n')
    elif word == '/reset':
        session.reset()
        out.write('(conversation cleared)\n')
    elif word == '/model':
        if hasattr(session.backend, 'model') and rest:
            session.backend.model = rest
            out.write('(model: %s)\n' % rest)
        else:
            out.write('(model: %s)\n' % getattr(session.backend, 'model', 'not applicable for this backend'))
    elif word == '/system':
        if rest:
            session.system = rest
            session.reset()
        out.write('(system prompt: %s)\n' % session.system)
    elif word == '/check':
        session.check = ConsistencyCheck() if rest == 'on' else None
        out.write('(self-consistency check %s)\n' % ('on' if session.check else 'off'))
    elif word == '/save':
        path = rest or 'chat.txt'
        with open(path, 'w') as f:
            for m in session.history:
                f.write('%s: %s\n\n' % ('you' if m['role'] == 'user' else 'bot', m['content']))
        out.write('(saved to %s)\n' % path)
    else:
        out.write('unknown command; /help lists them\n')
    return True


def main(argv=None, stdin=None, stdout=None):
    args = parser().parse_args(argv)
    out = stdout or sys.stdout
    try:
        if args.list_models:
            for name in list_ollama_models(args.host, allow_remote=args.allow_remote):
                out.write(name + '\n')
            return 0
        backend = build_backend(args)
        session = ChatSession(backend, args.system, check=ConsistencyCheck() if args.check else None, max_tokens=args.max_tokens)
        if args.web:
            from .web import serve
            serve(session, port=args.port)
            return 0
        out.write('quantum-mind-chat  [%s]  type a question and press Enter. /help for commands, /quit to leave.\n'
                  % (getattr(backend, 'model', args.backend)))
        if args.backend == 'demo':
            out.write('(demo backend: no model is running; answers are canned)\n')
        repl(session, stdin, out)
        return 0
    except ChatError as e:
        out.write('error: %s\n' % e)
        return 2


if __name__ == '__main__':
    sys.exit(main())
