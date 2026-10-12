"""A tiny vision-language-action model for one motor, trained end to end in NumPy.

Three slots with explicit interfaces, so each can be replaced later:

* **grounding** (vision + language): ``ground(image, text)`` returns a distribution over table cells, "which place
  does the instruction mean". It is a single attention step: the instruction forms a query, every image cell a key.
* **belief / ask gate**: lives in :mod:`quantum_mind.embodied.policies`; it reads that distribution, not just its argmax.
* **action**: ``act(target, theta, omega)`` maps the expected target and the noisy motor state to a command.

``fit`` trains grounding and action together by behaviour cloning (cross-entropy on the intended cell, squared error
on a PD demonstrator's command); gradients are written by hand and checked against finite differences in the tests.
"""
from __future__ import annotations

import numpy as np

from .motor import expert_command
from .scene import WORDS, cell_centres, sample_scene


def _softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


class _Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            params[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def make_dataset(n_scenes, seed, n_cells=16):
    """Scenes plus, for each, one random motor state and the demonstrator's command."""
    rng = np.random.default_rng(seed)
    scenes = [sample_scene(rng, n_cells) for _ in range(n_scenes)]
    return scenes


def _batch(scenes, idx, rng, n_cells):
    xs = cell_centres(n_cells)
    img = np.stack([scenes[i].image for i in idx])
    txt = np.stack([scenes[i].text for i in idx])
    tgt = np.array([scenes[i].target for i in idx])
    theta = rng.uniform(-1, 1, len(idx))
    omega = np.clip(rng.standard_normal(len(idx)), -3, 3)
    u_star = expert_command(xs[tgt], theta, omega)
    return img, txt, tgt, theta + 0.005 * rng.standard_normal(len(idx)), omega + 0.15 * rng.standard_normal(len(idx)), u_star


class TinyVLA:
    """Grounding by attention plus a small action head (about 1.3k parameters at the defaults)."""

    def __init__(self, n_cells=16, d=16, hidden=24, seed=0):
        self.n_cells, self.d, self.hidden = n_cells, d, hidden
        rng = np.random.default_rng(seed)
        self.xs = cell_centres(n_cells)
        self.p = {'Wv': 0.5 * rng.standard_normal((3, d)), 'bv': np.zeros(d),
                  'Wk': rng.standard_normal((d, d)) / np.sqrt(d), 'P': 0.1 * rng.standard_normal((n_cells, d)),
                  'Wq': 0.5 * rng.standard_normal((len(WORDS), d)),
                  'A1': rng.standard_normal((2, hidden)) / np.sqrt(2), 'c1': np.zeros(hidden),
                  'A2': rng.standard_normal((hidden, 1)) / np.sqrt(hidden), 'c2': np.zeros(1)}

    @property
    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    # ---- grounding slot -------------------------------------------------------------------------------------
    def _ground(self, img, txt):
        p = self.p
        hp = img @ p['Wv'] + p['bv']
        h = np.maximum(hp, 0.0)
        k = h @ p['Wk'] + p['P']
        q = txt @ p['Wq']
        s = np.einsum('nd,nwd->nw', q, k) / np.sqrt(self.d)
        return _softmax(s), (hp, h, k, q)

    def ground(self, image, text):
        """Distribution over cells for the instruction in the image. Accepts one scene or a batch."""
        img, txt = np.asarray(image, float), np.asarray(text, float)
        single = img.ndim == 2
        g = self._ground(img[None] if single else img, txt[None] if single else txt)[0]
        return g[0] if single else g

    def expected_target(self, g):
        return np.asarray(g) @ self.xs

    # ---- action slot ----------------------------------------------------------------------------------------
    def _act(self, target, theta, omega):
        p = self.p
        a = np.stack([target - theta, omega], axis=-1)
        hid = np.tanh(a @ p['A1'] + p['c1'])
        u = np.tanh(hid @ p['A2'] + p['c2'])[..., 0]
        return u, (a, hid)

    def act(self, target, theta, omega):
        """Motor command in [-1, 1] for a target position and the measured state."""
        return self._act(np.asarray(target, float), np.asarray(theta, float), np.asarray(omega, float))[0]

    # ---- training -------------------------------------------------------------------------------------------
    def loss_and_grads(self, img, txt, tgt, theta, omega, u_star, lam_g=1.0, lam_a=1.0, l2=1e-4):
        n = len(tgt)
        p = self.p
        g, (hp, h, k, q) = self._ground(img, txt)
        xhat = g @ self.xs
        u, (a, hid) = self._act(xhat, theta, omega)
        ce = -np.log(g[np.arange(n), tgt] + 1e-12).mean()
        mse = ((u - u_star) ** 2).mean()
        loss = lam_g * ce + lam_a * mse + 0.5 * l2 * sum((v ** 2).sum() for kk, v in p.items() if kk[0] in 'WAP')
        gr = {}
        dzo = (2 * lam_a * (u - u_star) / n * (1 - u ** 2))[:, None]
        gr['A2'], gr['c2'] = hid.T @ dzo, dzo.sum(0)
        dz1 = (dzo @ p['A2'].T) * (1 - hid ** 2)
        gr['A1'], gr['c1'] = a.T @ dz1, dz1.sum(0)
        dxhat = (dz1 @ p['A1'].T)[:, 0]
        onehot = np.zeros_like(g)
        onehot[np.arange(n), tgt] = 1.0
        ds = g * dxhat[:, None] * (self.xs[None] - xhat[:, None]) + lam_g * (g - onehot) / n
        ds = ds / np.sqrt(self.d)
        dq = np.einsum('nw,nwd->nd', ds, k)
        dk = ds[:, :, None] * q[:, None, :]
        gr['P'] = dk.sum(0)
        gr['Wk'] = np.einsum('nwv,nwd->vd', h, dk)
        dhp = (dk @ p['Wk'].T) * (hp > 0)
        gr['Wv'] = np.einsum('nwc,nwv->cv', img, dhp)
        gr['bv'] = dhp.sum((0, 1))
        gr['Wq'] = txt.T @ dq
        for kk in p:
            if kk[0] in 'WAP':
                gr[kk] = gr[kk] + l2 * p[kk]
        return loss, gr

    def fit(self, scenes, steps=3000, batch=128, lr=0.01, seed=0, lam_g=1.0, lam_a=1.0):
        rng = np.random.default_rng(seed)
        opt = _Adam(self.p, lr)
        for _ in range(steps):
            idx = rng.integers(0, len(scenes), batch)
            _, gr = self.loss_and_grads(*_batch(scenes, idx, rng, self.n_cells), lam_g=lam_g, lam_a=lam_a)
            opt.step(self.p, gr)
        return self


class BCBaseline:
    """The usual end-to-end behaviour-cloning policy: image, instruction and state in, command out (one hidden layer).

    No grounding bottleneck, no distribution over cells, so it can neither be calibrated per place nor ask.
    """

    def __init__(self, n_cells=16, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        n_in = n_cells * 3 + len(WORDS) + 2
        self.n_cells = n_cells
        self.p = {'W1': rng.standard_normal((n_in, hidden)) / np.sqrt(n_in), 'b1': np.zeros(hidden),
                  'W2': rng.standard_normal((hidden, 1)) / np.sqrt(hidden), 'b2': np.zeros(1)}

    @property
    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def _x(self, img, txt, theta, omega):
        n = len(theta)
        return np.concatenate([img.reshape(n, -1), txt, theta[:, None], omega[:, None]], axis=1)

    def command(self, image, text, theta, omega):
        x = self._x(np.asarray(image, float)[None], np.asarray(text, float)[None], np.array([theta], float), np.array([omega], float))
        return float(np.tanh(np.tanh(x @ self.p['W1'] + self.p['b1']) @ self.p['W2'] + self.p['b2'])[0, 0])

    def fit(self, scenes, steps=3000, batch=128, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        opt = _Adam(self.p, lr)
        p = self.p
        for _ in range(steps):
            idx = rng.integers(0, len(scenes), batch)
            img, txt, _, th, om, ustar = _batch(scenes, idx, rng, self.n_cells)
            x = self._x(img, txt, th, om)
            hid = np.tanh(x @ p['W1'] + p['b1'])
            u = np.tanh(hid @ p['W2'] + p['b2'])[:, 0]
            dzo = (2 * (u - ustar) / batch * (1 - u ** 2))[:, None]
            dz1 = (dzo @ p['W2'].T) * (1 - hid ** 2)
            opt.step(p, {'W2': hid.T @ dzo, 'b2': dzo.sum(0), 'W1': x.T @ dz1, 'b1': dz1.sum(0)})
        return self
