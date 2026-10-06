// Runs the website playground's maths in Node.js (no browser): node site/check_playground.js site/static/playground.js '<sequences JSON>'
const fs = require('fs');
const el = () => ({ style: {}, textContent: '', addEventListener() {}, querySelectorAll: () => [], querySelector: () => ({ addEventListener() {} }) });
global.window = global; global.Plotly = { newPlot() {}, restyle() {} };
global.document = { getElementById: (id) => (id === 'playground' ? el() : el()), documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: () => '' });
global.requestAnimationFrame = (f) => 0;
eval(fs.readFileSync(process.argv[2], 'utf8'));
const P = window.__qlcogPlayground;
const seqs = JSON.parse(process.argv[3]); const out = [];
for (const seq of seqs) {
  let s = [[1, 0], [0, 0], [0, 0], [0, 0]];
  for (const [g, q, t] of seq) {
    if (g === 'CX01') s = P.cnotPow(s, 0, 1, t); else if (g === 'CX10') s = P.cnotPow(s, 1, 0, t);
    else { const [n, a] = P.GATES[g]; s = P.apply1(s, P.rot(n, a, t), q); }
  }
  out.push({ psi: s, b0: P.bloch(s, 0), b1: P.bloch(s, 1), c: P.concurrence(s) });
}
console.log(JSON.stringify(out));
