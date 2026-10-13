"""The same chat in a browser tab, served from this machine only (127.0.0.1) with the standard library.

Open ``http://127.0.0.1:8765`` after ``quantum-mind-chat --web``. Type in the box and press Enter; Shift+Enter makes a new line.
The answer streams in. There is one conversation per server, and nothing leaves your machine except the request to the local
model server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import ChatError

__all__ = ['make_server', 'serve']

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local chat</title><style>
:root{--bg:#fff;--fg:#1a1a1a;--card:#f3f4f6;--me:#dbeafe;--line:#d1d5db}
@media (prefers-color-scheme: dark){:root{--bg:#111418;--fg:#e8eaed;--card:#1c2128;--me:#1f3a5f;--line:#30363d}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;display:flex;
flex-direction:column;height:100vh}
#log{flex:1;overflow:auto;padding:16px;max-width:820px;width:100%;margin:0 auto;box-sizing:border-box}
.m{padding:10px 14px;border-radius:10px;margin:8px 0;white-space:pre-wrap}.u{background:var(--me);margin-left:15%}.b{
background:var(--card);margin-right:15%}
form{display:flex;gap:8px;padding:12px 16px;max-width:820px;width:100%;margin:0 auto;box-sizing:border-box}
textarea{flex:1;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);
color:var(--fg);resize:none}
button{font:inherit;padding:0 18px;border-radius:8px;border:1px solid var(--line);background:var(--card);
color:var(--fg);cursor:pointer}
small{color:#6b7280;text-align:center;padding-bottom:8px}
</style></head><body><div id="log"></div>
<form id="f"><textarea id="q" rows="2" placeholder="Type a question and press Enter" autofocus></textarea><button>Send</button></form>
<small>Runs on this computer. Enter sends, Shift+Enter adds a line. <a href="#" id="reset">Clear conversation</a></small>
<script>
const log=document.getElementById('log'),q=document.getElementById('q');
function add(cls,t){const d=document.createElement('div');d.className='m '+cls;d.textContent=t;log.appendChild(d);
log.scrollTop=log.scrollHeight;return d}
async function send(){const text=q.value.trim();if(!text)return;q.value='';add('u',text);const b=add('b','');
 try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
message:text})});
  const rd=r.body.getReader(),dec=new TextDecoder();for(;;){const {done,value}=await rd.read();if(done)break;
b.textContent+=dec.decode(value);log.scrollTop=log.scrollHeight}
 }catch(e){b.textContent='[error] '+e}}
document.getElementById('f').addEventListener('submit',e=>{e.preventDefault();send()});
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
document.getElementById('reset').addEventListener('click',async e=>{e.preventDefault();await fetch('/api/reset',{
method:'POST'});log.innerHTML=''});
</script></body></html>"""


def make_server(session, port=8765):
    """A threading HTTP server bound to 127.0.0.1 for ``session`` (port 0 picks a free port)."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='text/plain; charset=utf-8'):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._send(200, PAGE, 'text/html; charset=utf-8')
            else:
                self._send(404, 'not found')

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0) or 0)
            if self.path == '/api/reset':
                with lock:
                    session.reset()
                return self._send(200, 'ok')
            if self.path != '/api/ask':
                return self._send(404, 'not found')
            try:
                message = str(json.loads(self.rfile.read(n) or b'{}').get('message', '')).strip()
            except ValueError:
                return self._send(400, 'bad request')
            if not message:
                return self._send(400, 'empty message')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with lock:                                                       # one conversation: one answer at a time
                try:
                    for piece in session.ask(message):
                        self.wfile.write(piece.encode())
                        self.wfile.flush()
                except ChatError as e:
                    self.wfile.write(('[error] %s' % e).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(session, port=8765):
    """Serve the chat page until Ctrl-C."""
    server = make_server(session, port)
    print('chat page on http://127.0.0.1:%d  (this machine only; Ctrl-C stops it)' % server.server_address[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
