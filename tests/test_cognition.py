"""The quantum-cognition core: its Bayes limit, quantum-probability identities, gradients and what it can and cannot learn."""
import numpy as np
import pytest

pytest.importorskip('jax')
pytest.importorskip('optax')
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from quantum_mind.cognition import (BayesCore, ContextWeightedCore, GRUCore, MarkovContextCore, QuantumCognitionCore,  # noqa: E402
                                    hermitian, update_state)

D, F, K, T = 3, 3, 2, 3


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, T, F)).astype(np.float32), np.eye(K, dtype=np.float32)[rng.integers(0, K, (n, T))])


def _random_params(core, seed=0, rot=1.0):
    p = core.init(jax.random.PRNGKey(seed))
    p['W'] = p['W'] * 4
    p['H'] = rot * jax.random.normal(jax.random.PRNGKey(seed + 1), p['H'].shape)
    p['phi'] = jax.random.uniform(jax.random.PRNGKey(seed + 2), (D,), maxval=6.28)
    return p


def test_zero_rotations_is_exactly_a_bayes_filter():
    core = QuantumCognitionCore(D, F, K)
    p = _random_params(core, rot=0.0)
    X, C = _data(50)
    got = np.asarray(core.probs(p, jnp.asarray(X), jnp.asarray(C), jnp.ones((50, T), bool)))
    logit = np.asarray(p['a']) + np.clip(X @ np.asarray(p['W']) + np.asarray(p['b']), -8, 8).sum(1)
    want = np.exp(logit - logit.max(1, keepdims=True))
    want /= want.sum(1, keepdims=True)
    assert np.allclose(got, want, atol=1e-5)
    assert np.allclose(BayesCore(D, F, K).probs(p, jnp.asarray(X), jnp.asarray(C), jnp.ones((50, T), bool)), want, atol=1e-5)


def test_state_is_a_density_matrix_and_probabilities_sum_to_one():
    core = QuantumCognitionCore(D, F, K)
    p = _random_params(core)
    X, C = _data(8)
    rho = np.asarray(jax.vmap(lambda x, c: core.state(p, x, c, jnp.ones(T, bool)))(jnp.asarray(X), jnp.asarray(C)))
    assert np.allclose(rho, np.conj(np.swapaxes(rho, 1, 2)), atol=1e-5)
    assert np.allclose(np.trace(rho, axis1=1, axis2=2).real, 1, atol=1e-5)
    assert np.linalg.eigvalsh(rho).min() > -1e-5
    assert np.allclose(core.probs(p, jnp.asarray(X), jnp.asarray(C), jnp.ones((8, T), bool)).sum(1), 1, atol=1e-5)


def test_projective_measurements_obey_the_qq_equality():
    """Order effects from non-commuting projectors still satisfy p(A yes,B no)+p(A no,B yes) = p(B yes,A no)+p(B no,A yes)."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        psi = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        rho = jnp.asarray(np.outer(psi, psi.conj()) / np.vdot(psi, psi))
        U = {}
        for q in 'AB':
            U[q] = jax.scipy.linalg.expm(1j * hermitian(jnp.asarray(rng.standard_normal((D, D)))))
        yes, no = jnp.array([1.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0])

        def seq(first, second, a, b):
            return float(jnp.real(jnp.trace(update_state(update_state(rho, a, U[first]), b, U[second]))))
        lhs = seq('A', 'B', yes, no) + seq('A', 'B', no, yes)
        rhs = seq('B', 'A', yes, no) + seq('B', 'A', no, yes)
        assert lhs == pytest.approx(rhs, abs=1e-5)
        assert seq('A', 'B', yes, yes) != pytest.approx(seq('B', 'A', yes, yes), abs=1e-3)      # but the order does matter


def test_gradients_match_finite_differences():
    jax.config.update('jax_enable_x64', True)
    try:
        core = QuantumCognitionCore(D, F, K)
        p = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float64), _random_params(core, rot=0.4))
        X, C = _data(12)
        X, C, m, y = jnp.asarray(X, jnp.float64), jnp.asarray(C, jnp.float64), jnp.ones((12, T), bool), jnp.arange(12) % D
        f = lambda q: core._nll(q, X, C, m, y)                                      # noqa: E731
        g = jax.grad(f)(p)
        for name in ('H', 'W', 'phi'):
            idx = tuple(0 for _ in p[name].shape)
            e = jnp.zeros_like(p[name]).at[idx].set(1e-6)
            fd = (f({**p, name: p[name] + e}) - f({**p, name: p[name] - e})) / 2e-6
            assert float(fd) == pytest.approx(float(g[name][idx]), rel=2e-3, abs=1e-6)
    finally:
        jax.config.update('jax_enable_x64', False)


def test_order_matters_for_the_quantum_core_but_not_for_bayes():
    q, b = QuantumCognitionCore(D, F, K), BayesCore(D, F, K)
    pq = _random_params(q, rot=1.0)
    pb = b.init(jax.random.PRNGKey(0))
    X, C = _data(30)
    perm = [2, 0, 1]
    ones = jnp.ones((30, T), bool)
    a1 = np.asarray(q.probs(pq, jnp.asarray(X), jnp.asarray(C), ones))
    a2 = np.asarray(q.probs(pq, jnp.asarray(X[:, perm]), jnp.asarray(C[:, perm]), ones))
    assert np.abs(a1 - a2).max() > 0.02
    b1 = np.asarray(b.probs(pb, jnp.asarray(X), jnp.asarray(C), ones))
    b2 = np.asarray(b.probs(pb, jnp.asarray(X[:, perm]), jnp.asarray(C[:, perm]), ones))
    assert np.allclose(b1, b2, atol=1e-5)


def test_masking_ignores_padded_steps():
    core = QuantumCognitionCore(D, F, K)
    p = _random_params(core)
    X, C = _data(6)
    mask = np.ones((6, T), bool)
    mask[:, -1] = False
    full = np.asarray(core.probs(p, jnp.asarray(X), jnp.asarray(C), jnp.asarray(mask)))
    X2 = X.copy()
    X2[:, -1] = 99.0                                                                  # garbage in the masked step changes nothing
    again = np.asarray(core.probs(p, jnp.asarray(X2), jnp.asarray(C), jnp.asarray(mask)))
    assert np.allclose(full, again, atol=1e-6)


def _sample(core, p, n, seed):
    rng = np.random.default_rng(seed)
    X, C = _data(n, seed)
    P = np.asarray(core.probs(p, jnp.asarray(X), jnp.asarray(C), jnp.ones((n, T), bool)))
    y = np.array([rng.choice(D, p=q / q.sum()) for q in P])
    return X, C, y, P


def test_it_learns_a_quantum_order_effect_and_does_not_invent_one():
    kw = dict(steps=250, restarts=2, seed=0)
    gen = QuantumCognitionCore(D, F, K)
    p = _random_params(gen, seed=4, rot=1.2)
    p['g'] = jnp.asarray(-3.0)
    Xa, Ca, ya, _ = _sample(gen, p, 1500, 1)
    Xb, Cb, yb, Pb = _sample(gen, p, 3000, 2)
    oracle = -np.mean(np.log(Pb[np.arange(len(yb)), yb] + 1e-7))
    q = QuantumCognitionCore(D, F, K, **kw).fit(Xa, Ca, ya)
    b = BayesCore(D, F, K, **kw).fit(Xa, Ca, ya)
    gq, gb = q.log_loss(Xb, Cb, yb) - oracle, b.log_loss(Xb, Cb, yb) - oracle
    assert gq < 0.5 * gb and gb > 0.05                                   # Bayes cannot represent it; the quantum core mostly can
    ctrl = BayesCore(D, F, K)                                            # truth is a plain product: no order effect to find
    pc = _random_params(ctrl, seed=5)
    Xa, Ca, ya, _ = _sample(ctrl, pc, 1500, 3)
    Xb, Cb, yb, Pb = _sample(ctrl, pc, 3000, 4)
    oracle = -np.mean(np.log(Pb[np.arange(len(yb)), yb] + 1e-7))
    q = QuantumCognitionCore(D, F, K, **kw).fit(Xa, Ca, ya)
    assert q.log_loss(Xb, Cb, yb) - oracle < 0.03                        # no worse than a hair above perfect
    assert q.rotation_size < 0.35


def test_every_model_fits_and_predicts():
    X, C = _data(200)
    y = np.arange(200) % D
    for cls in (BayesCore, ContextWeightedCore, MarkovContextCore, GRUCore, QuantumCognitionCore):
        m = cls(D, F, K, steps=20, restarts=1).fit(X, C, y)
        p = m.predict_proba(X, C)
        assert p.shape == (200, D) and np.allclose(p.sum(1), 1, atol=1e-4) and m.n_params > 0
