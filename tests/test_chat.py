"""The chat program: a fake Ollama server stands in for the model, so the real HTTP and streaming code is exercised without any weights."""
import io
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantum_mind.chat import (ChatError, ChatSession, ConsistencyCheck, OllamaBackend, ScriptedBackend, agreement, list_ollama_models,
                               main, make_server)
from quantum_mind.chat.backends import LlamaCppBackend


@pytest.fixture()
def ollama():
    """A fake Ollama on a free local port speaking /api/tags and the streaming /api/chat protocol."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({'models': [{'name': 'llama3.2:latest'}, {'name': 'qwen2.5:latest'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(req)
            if req['model'] == 'missing':
                body = json.dumps({'error': "model 'missing' not found"}).encode()
                self.send_response(404)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            question = req['messages'][-1]['content']
            reply = 'The capital of France is Paris.' if 'France' in question else 'You said: ' + question
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.end_headers()
            for w in reply.split(' '):
                self.wfile.write((json.dumps({'message': {'role': 'assistant', 'content': w + ' '}, 'done': False}) + '\n').encode())
            self.wfile.write((json.dumps({'done': True}) + '\n').encode())

    srv = ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield 'http://127.0.0.1:%d' % srv.server_address[1], seen
    srv.shutdown()
    srv.server_close()


def test_ollama_backend_streams_and_sends_the_right_request(ollama):
    host, seen = ollama
    b = OllamaBackend('llama3.2', host)
    pieces = list(b.stream([{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'What is the capital of France?'}],
                           temperature=0.3, max_tokens=50))
    assert len(pieces) > 3 and ''.join(pieces).strip() == 'The capital of France is Paris.'        # arrives in pieces
    req = seen[-1]
    assert req['model'] == 'llama3.2' and req['stream'] is True and req['options'] == {'temperature': 0.3, 'num_predict': 50}
    assert req['messages'][0]['role'] == 'system'
    assert b.generate('hello').strip() == 'You said: hello'
    assert list_ollama_models(host) == ['llama3.2:latest', 'qwen2.5:latest']


def test_errors_tell_the_user_what_to_do(ollama):
    host, _ = ollama
    with pytest.raises(ChatError, match='ollama pull'):
        list(OllamaBackend('missing', host).stream([{'role': 'user', 'content': 'x'}]))
    with pytest.raises(ChatError, match='cannot reach Ollama'):
        list(OllamaBackend('llama3.2', 'http://127.0.0.1:9').stream([{'role': 'user', 'content': 'x'}]))
    with pytest.raises(ChatError, match='not a local address'):
        OllamaBackend('llama3.2', 'http://example.com:11434')
    OllamaBackend('llama3.2', 'http://example.com:11434', allow_remote=True)                      # explicit opt-in
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        with pytest.raises(ChatError, match='pip install llama-cpp-python'):
            LlamaCppBackend('x.gguf')


def test_the_session_remembers_trims_and_keeps_interrupted_answers():
    be = ScriptedBackend(['one two three four', 'five six seven', 'eight nine'])
    s = ChatSession(be, system='SYS', max_turns=2)
    assert s.answer('q1') == 'one two three four'
    assert s.answer('q2') == 'five six seven'
    assert be.calls[1]['messages'] == [{'role': 'system', 'content': 'SYS'}, {'role': 'user', 'content': 'q1'},
                                       {'role': 'assistant', 'content': 'one two three four'}, {'role': 'user', 'content': 'q2'}]
    s.answer('q3')
    assert len(s.history) == 4 and s.history[0]['content'] == 'q2'                            # only the last two turns are kept
    gen = s.ask('q4')
    next(gen)
    gen.close()                                                                               # Ctrl-C part-way: what was written stays
    assert s.history[-2]['content'] == 'q4' and s.history[-1]['content'] != ''
    s.reset()
    assert s.history == [] and s.generate('anything').strip() == 'eight nine'                  # generate() has no memory


def test_agreement_separates_rewordings_from_disagreements():
    assert agreement('Paris', 'The capital is Paris.') > 0.6
    assert agreement('The answer is 42', 'The answer is 17') < 0.5
    assert agreement('The answer is 42.', 'It is 42') > 0.5
    assert agreement('Paris', 'London') == 0.0 and agreement('x y z', 'x y z') == 1.0


def test_the_consistency_check_says_when_the_answers_vary():
    steady = ScriptedBackend(['The capital of France is Paris.'])
    s = ChatSession(steady, check=ConsistencyCheck(n=3))
    assert 'not sure' not in s.answer('capital?') and s.last_agreement > 0.9
    assert len(steady.calls) == 3 and steady.calls[1]['temperature'] == 0.8                  # two extra samples, at a higher temperature
    shaky = ScriptedBackend(['It happened in 1815.', 'It was in 1914 in Sarajevo.', 'I think 1066 at Hastings.'])
    s = ChatSession(shaky, check=ConsistencyCheck(n=3))
    out = s.answer('when?')
    assert 'not sure' in out and s.last_agreement < 0.5
    assert s.history[-1]['content'] == 'It happened in 1815.'                                  # the note is not stored as the model's words


def test_the_terminal_chat(ollama, tmp_path):
    host, _ = ollama
    out = io.StringIO()
    typed = 'What is the capital of France?\n/system Be brief\nhi\n/save %s\n/bogus\n/quit\n' % (tmp_path / 'c.txt')
    code = main(['--model', 'llama3.2', '--host', host], stdin=io.StringIO(typed), stdout=out)
    text = out.getvalue()
    assert code == 0 and 'bot> The capital of France is Paris.' in text and 'unknown command' in text and 'system prompt: Be brief' in text
    assert 'you: hi' in (tmp_path / 'c.txt').read_text()
    out = io.StringIO()
    assert main(['--model', 'missing', '--host', host], stdin=io.StringIO('hello\n'), stdout=out) == 0
    assert '[error]' in out.getvalue() and 'ollama pull' in out.getvalue()                # explained, not a traceback
    out = io.StringIO()
    assert main(['--list-models', '--host', host], stdout=out) == 0 and 'qwen2.5:latest' in out.getvalue()
    assert main(['--list-models', '--host', 'http://127.0.0.1:9'], stdout=io.StringIO()) == 2
    assert main(['--backend', 'llamacpp'], stdout=io.StringIO()) == 2                         # needs --model-path


def test_the_web_chat_is_local_and_streams():
    be = ScriptedBackend(['first answer here', 'second answer here'])
    srv = make_server(ChatSession(be), port=0)
    assert srv.server_address[0] == '127.0.0.1'
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = 'http://127.0.0.1:%d' % srv.server_address[1]

    def post(path, payload):
        req = urllib.request.Request(base + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=10)
    try:
        assert 'textarea' in urllib.request.urlopen(base + '/', timeout=10).read().decode()
        assert post('/api/ask', {'message': 'hello'}).read().decode().strip() == 'first answer here'
        assert post('/api/ask', {'message': 'again'}).read().decode().strip() == 'second answer here'
        assert [m['content'] for m in be.calls[1]['messages']][-3:] == ['hello', 'first answer here', 'again']      # one conversation
        assert post('/api/reset', {}).read() == b'ok'
        with pytest.raises(urllib.error.HTTPError) as e:
            post('/api/ask', {'message': '   '})
        assert e.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()
