// Two-qubit playground: exact state-vector simulation in the browser, drawn on two Bloch spheres.
// Basis index i = q0 + 2*q1 (Qiskit order). Gates animate as fractional powers U^t, t: 0 -> 1.
(function () {
  const root = document.getElementById('playground');
  if (!root || !window.Plotly) return;
  const css = getComputedStyle(document.documentElement);
  const col = (v) => css.getPropertyValue(v).trim();
  const C = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1]], mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
    conj: (a) => [a[0], -a[1]], abs2: (a) => a[0] * a[0] + a[1] * a[1], s: (x, a) => [x * a[0], x * a[1]],
    exp: (t) => [Math.cos(t), Math.sin(t)],
  };
  let psi = [[1, 0], [0, 0], [0, 0], [0, 0]];
  const trails = [[], []]; const history = [];

  // single-qubit fractional gate: rotation by angle*t about axis n (global phase dropped)
  function rot(n, angle, t) {
    const a = angle * t / 2, c = Math.cos(a), s = Math.sin(a);
    const [x, y, z] = n;                       // U = cos a I - i sin a (n.sigma)
    return [[[c, -s * z], [-s * y, -s * x]], [[s * y, -s * x], [c, s * z]]];
  }
  const R2 = Math.SQRT1_2;
  const GATES = {
    H: [[R2, 0, R2], Math.PI], X: [[1, 0, 0], Math.PI], Y: [[0, 1, 0], Math.PI], Z: [[0, 0, 1], Math.PI],
    S: [[0, 0, 1], Math.PI / 2], T: [[0, 0, 1], Math.PI / 4], 'Rx': [[1, 0, 0], Math.PI / 4],
    'Ry': [[0, 1, 0], Math.PI / 4], 'Rz': [[0, 0, 1], Math.PI / 4], 'Sdg': [[0, 0, 1], -Math.PI / 2],
  };
  function apply1(state, U, q) {
    const out = state.map((a) => a.slice());
    for (let i = 0; i < 4; i++) {
      if (((i >> q) & 1) !== 0) continue;
      const j = i | (1 << q), a = state[i], b = state[j];
      out[i] = C.add(C.mul(U[0][0], a), C.mul(U[0][1], b));
      out[j] = C.add(C.mul(U[1][0], a), C.mul(U[1][1], b));
    }
    return out;
  }
  function cnotPow(state, ctrl, targ, t) {      // |0><0| (x) I + |1><1| (x) X^t,  X^t = ((1+e^{i pi t})/2) I + ((1-e^{i pi t})/2) X
    const e = C.exp(Math.PI * t), p = C.s(0.5, C.add([1, 0], e)), m = C.s(0.5, C.add([1, 0], C.s(-1, e)));
    const out = state.map((a) => a.slice());
    for (let i = 0; i < 4; i++) {
      if (((i >> ctrl) & 1) !== 1 || ((i >> targ) & 1) !== 0) continue;
      const j = i | (1 << targ), a = state[i], b = state[j];
      out[i] = C.add(C.mul(p, a), C.mul(m, b)); out[j] = C.add(C.mul(m, a), C.mul(p, b));
    }
    return out;
  }
  function bloch(state, q) {
    let r00 = 0, r11 = 0, r01 = [0, 0];
    for (let o = 0; o < 2; o++) {               // trace out the other qubit
      const i0 = q === 0 ? 2 * o : o, i1 = q === 0 ? 2 * o + 1 : o + 2;
      r00 += C.abs2(state[i0]); r11 += C.abs2(state[i1]); r01 = C.add(r01, C.mul(state[i0], C.conj(state[i1])));
    }
    return [2 * r01[0], -2 * r01[1], r00 - r11];
  }
  const concurrence = (s) => 2 * Math.sqrt(C.abs2(C.add(C.mul(s[0], s[3]), C.s(-1, C.mul(s[1], s[2])))));

  // ---- drawing
  const ids = ['pg-q0', 'pg-q1'], colours = [col('--like') || '#ff7b72', col('--quantum') || '#79c0ff'];
  function sphereTraces(c) {
    const tr = [];
    const N = 36, M = 18, X = [], Y = [], Z = [];
    for (let i = 0; i <= M; i++) {
      const v = Math.PI * i / M; X.push([]); Y.push([]); Z.push([]);
      for (let j = 0; j <= N; j++) { const w = 2 * Math.PI * j / N; X[i].push(Math.sin(v) * Math.cos(w)); Y[i].push(Math.sin(v) * Math.sin(w)); Z[i].push(Math.cos(v)); }
    }
    tr.push({ type: 'surface', x: X, y: Y, z: Z, opacity: 0.12, showscale: false, hoverinfo: 'skip', colorscale: [[0, col('--quantum')], [1, col('--quantum')]] });
    const ring = (f) => { const a = { x: [], y: [], z: [] }; for (let k = 0; k <= 120; k++) { const w = 2 * Math.PI * k / 120, p = f(w); a.x.push(p[0]); a.y.push(p[1]); a.z.push(p[2]); } return a; };
    for (const f of [(w) => [Math.cos(w), Math.sin(w), 0], (w) => [Math.cos(w), 0, Math.sin(w)], (w) => [0, Math.cos(w), Math.sin(w)]]) {
      tr.push({ type: 'scatter3d', mode: 'lines', ...ring(f), line: { color: col('--line'), width: 2 }, hoverinfo: 'skip', showlegend: false });
    }
    for (const a of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      tr.push({ type: 'scatter3d', mode: 'lines', x: [-a[0], a[0]], y: [-a[1], a[1]], z: [-a[2], a[2]], line: { color: col('--muted'), width: 2, dash: 'dash' }, hoverinfo: 'skip', showlegend: false });
    }
    tr.push({ type: 'scatter3d', mode: 'text', x: [0, 0, 1.25, 0], y: [0, 0, 0, 1.25], z: [1.2, -1.2, 0, 0], text: ['|0⟩', '|1⟩', '|+⟩', '|+i⟩'], textfont: { color: col('--text'), size: 14 }, hoverinfo: 'skip', showlegend: false });
    // dynamic: trail, arrow, tip (indices 8, 9, 10)
    tr.push({ type: 'scatter3d', mode: 'lines', x: [0], y: [0], z: [1], line: { color: c, width: 4 }, opacity: 0.55, hoverinfo: 'skip', showlegend: false });
    tr.push({ type: 'scatter3d', mode: 'lines', x: [0, 0], y: [0, 0], z: [0, 1], line: { color: c, width: 10 }, hoverinfo: 'skip', showlegend: false });
    tr.push({ type: 'scatter3d', mode: 'markers', x: [0], y: [0], z: [1], marker: { color: c, size: 6 }, hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>', showlegend: false });
    return tr;
  }
  const ax = { visible: false, range: [-1.3, 1.3] };
  const layout = { paper_bgcolor: col('--bg'), margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
    scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: 'cube', bgcolor: col('--bg'), camera: { eye: { x: 1.35, y: 1.0, z: 0.7 } } } };
  ids.forEach((id, q) => Plotly.newPlot(id, sphereTraces(colours[q]), layout, { displayModeBar: false, responsive: true }));

  function draw(state) {
    [0, 1].forEach((q) => {
      const r = bloch(state, q); trails[q].push(r);
      const T = trails[q];
      Plotly.restyle(ids[q], { x: [T.map((p) => p[0]), [0, r[0]], [r[0]]], y: [T.map((p) => p[1]), [0, r[1]], [r[1]]], z: [T.map((p) => p[2]), [0, r[2]], [r[2]]] }, [8, 9, 10]);
    });
    const c = concurrence(state), r0 = bloch(state, 0), r1 = bloch(state, 1), L = (r) => Math.hypot(...r);
    document.getElementById('pg-conc').style.width = (100 * c).toFixed(1) + '%';
    document.getElementById('pg-conc-v').textContent = c.toFixed(3);
    const amp = [0, 1, 2, 3].map((i) => {                         // labelled |q0 q1>
      const a = state[i]; return ['|' + (i & 1) + ((i >> 1) & 1) + '⟩', a[0].toFixed(3) + (a[1] >= 0 ? ' + ' : ' - ') + Math.abs(a[1]).toFixed(3) + 'i'];
    });
    document.getElementById('pg-read').textContent =
      'amplitudes (|q0 q1⟩):\n' + amp.map(([k, v]) => '  ' + k + '  ' + v).join('\n') +
      '\nBloch-vector length |r|:  qubit 0 = ' + L(r0).toFixed(3) + ',  qubit 1 = ' + L(r1).toFixed(3);
    document.getElementById('pg-hist').textContent = history.length ? history.join(' · ') : '(no gates yet)';
  }
  let busy = false;
  const DURATION = 600;                         // ms; time-based, so slow devices skip frames instead of lagging
  function animate(step, label, done) {
    if (busy) return; busy = true; const start = psi, t0 = performance.now();
    history.push(label);
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / DURATION), s = step(start, t); draw(s);
      if (t < 1) requestAnimationFrame(tick); else { psi = s; busy = false; if (done) done(); }
    };
    requestAnimationFrame(tick);
  }
  root.querySelectorAll('[data-gate]').forEach((b) => b.addEventListener('click', () => {
    const g = b.dataset.gate, q = +b.dataset.q, [n, ang] = GATES[g];
    animate((s, t) => apply1(s, rot(n, ang, t), q), g + '(q' + q + ')');
  }));
  root.querySelector('[data-cnot="01"]').addEventListener('click', () => animate((s, t) => cnotPow(s, 0, 1, t), 'CNOT(q0→q1)'));
  root.querySelector('[data-cnot="10"]').addEventListener('click', () => animate((s, t) => cnotPow(s, 1, 0, t), 'CNOT(q1→q0)'));
  root.querySelector('[data-reset]').addEventListener('click', () => {
    if (busy) return; psi = [[1, 0], [0, 0], [0, 0], [0, 0]]; trails[0] = []; trails[1] = []; history.length = 0; draw(psi);
  });
  root.querySelector('[data-bell]').addEventListener('click', () => {
    if (busy) return; const h = GATES.H;
    animate((s, t) => apply1(s, rot(h[0], h[1], t), 0), 'H(q0)', () => animate((s, t) => cnotPow(s, 0, 1, t), 'CNOT(q0→q1)'));
  });
  draw(psi);
  window.__quantumMindPlayground = { apply1, cnotPow, rot, bloch, concurrence, GATES };   // exposed for the build-time check
})();
