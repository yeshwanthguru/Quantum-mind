"""Learning modules: environments, exploration, quantum policy, tensor-train layers, quanvolution."""
import numpy as np
import pytest

from quantum_mind.envs import ClarificationEnv, TrustHandoverEnv, GridWorldEnv, register_envs
from quantum_mind.inspired.exploration import (EpsilonGreedy, Boltzmann, UCB, AmplitudeExploration, TabularAgent,
                                               StateIndexer, run_episodes)
from quantum_mind.inspired.tensor_layers import TTMatrix, factorise, compress_layers, apply_mlp
from quantum_mind.quantum.policy import VariationalPolicy, reinforce
from quantum_mind.quantum.quanvolution import QuanvolutionFilter, RandomConvFilter, extract_patches


def test_environments_follow_the_gymnasium_api():
    gym = pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env
    for env in (ClarificationEnv(), TrustHandoverEnv(), GridWorldEnv()):
        check_env(env, skip_render_check=True)
    ids = register_envs()
    assert 'quantum_mind/Clarification-v0' in ids
    env = gym.make('quantum_mind/TrustHandover-v0')
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_clarification_episode_ends_when_the_robot_acts():
    env = ClarificationEnv()
    env.reset(seed=1)
    n_q = len(env.model.questions)
    _, r, term, _, _ = env.step(n_q)                        # act on the first hypothesis at once
    assert term and r in (env.success_reward, -env.error_cost)


def test_exploration_strategies_solve_the_grid_world():
    for explorer in (EpsilonGreedy(seed=0), Boltzmann(seed=0), UCB(seed=0),
                     AmplitudeExploration(GridWorldEnv().observation_space.n, 4, seed=0)):
        env = GridWorldEnv()
        agent = TabularAgent(env.observation_space.n, env.action_space.n, explorer)
        out = run_episodes(agent, env, 200, max_steps=100)
        assert out['returns'][-20:].mean() > out['returns'][:20].mean()


def test_amplitude_exploration_keeps_a_floor_and_normalises():
    ex = AmplitudeExploration(1, 3, floor=0.05, seed=0)
    q = np.zeros(3)
    for _ in range(200):
        ex.update(0, 0, 1.0, q + np.array([1.0, 0.0, 0.0]))
    p = ex.amp[0] ** 2
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 0 and p.min() >= 0.05 ** 2 - 1e-12
    idx = StateIndexer(4)
    assert idx((0, 1)) == 0 and idx((1, 1)) == 1 and idx((0, 1)) == 0


def test_variational_policy_gradient_matches_finite_differences():
    pi = VariationalPolicy(n_features=2, n_actions=3, layers=2, seed=1, lr=0.0)
    S = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]])
    A = np.array([0, 2, 1])
    adv = np.array([1.0, -0.5, 2.0])

    def loss(w):
        pi.weights = w
        p = pi.probabilities(S)
        return -np.mean(adv * np.log(p[np.arange(3), A]))
    w0 = pi.weights.copy()
    pi.lr = 1.0
    pi.update(S, A, adv)
    analytic = w0 - pi.weights
    pi.weights = w0.copy()
    eps = 1e-6
    numeric = np.array([(loss(w0 + eps * e) - loss(w0 - eps * e)) / (2 * eps) for e in np.eye(len(w0))])
    assert np.allclose(analytic, numeric, atol=1e-6)
    assert np.allclose(pi.probabilities(S).sum(1), 1.0)


def test_reinforce_runs_on_an_environment():
    env = GridWorldEnv()
    pi = VariationalPolicy(n_features=2, n_actions=4, layers=1, seed=0)

    def feats(s):
        return np.array([s % 5, s // 5], float) / 4 * np.pi
    returns = reinforce(pi, env, 4, feats, batch=2)
    assert returns.shape == (4,)


def test_tensor_train_layers():
    assert factorise(64, 3) == (4, 4, 4) and np.prod(factorise(100, 3)) == 100
    rng = np.random.default_rng(0)
    W = rng.normal(size=(32, 48))
    tt = TTMatrix.from_dense(W, (4, 8), (6, 8), max_rank=100)
    X = rng.normal(size=(5, 48))
    assert tt.relative_error(W) < 1e-10 and np.allclose(tt.matvec(X), X @ W.T)
    low = TTMatrix.from_dense(W, (4, 8), (6, 8), max_rank=2)
    assert low.n_params < W.size and low.ranks == [2] and 0 < low.relative_error(W) < 1
    layers = [(rng.normal(size=(64, 64)), np.zeros(64)), (rng.normal(size=(3, 64)), np.zeros(3))]
    comp, report = compress_layers(layers, max_rank=64, d=2)
    assert report['errors'][1] == 0.0 and report['errors'][0] < 1e-10
    X64 = rng.normal(size=(5, 64))
    assert np.allclose(apply_mlp(comp, X64), apply_mlp(layers, X64))
    with pytest.raises(ValueError):
        TTMatrix.from_dense(W, (4, 4), (6, 8))


def test_quanvolution_filter():
    imgs = np.random.default_rng(0).random((2, 6, 6))
    P = extract_patches(imgs)
    assert P.shape == (2, 3, 3, 4) and np.allclose(P[0, 0, 0], imgs[0, :2, :2].ravel())
    f = QuanvolutionFilter(seed=3)
    out = f.transform(imgs)
    assert out.shape == (2, 3, 3, 4) and np.all(np.abs(out) <= 1 + 1e-12)
    # zero layers: encoding only, so <Z> = cos(pi x) for each pixel
    enc = QuanvolutionFilter(layers=0).transform(imgs)
    assert np.allclose(enc, np.cos(np.pi * P))
    assert RandomConvFilter(channels=6).transform(imgs).shape == (2, 3, 3, 6)


def test_mpl_style_and_tutorial_converter(tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib as mpl
    import sys
    import pathlib
    from quantum_mind.viz import use_mpl_style
    t = use_mpl_style('dark')
    assert mpl.rcParams['axes.facecolor'] == t['bg']
    mpl.rcdefaults()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tutorials'))
    from _convert import to_notebook, sources
    assert len(sources()) >= 15
    src = tmp_path / '99_x.md'
    src.write_text('# T\n\ntext\n\n```python\nx = 1\n```\n\nmore\n')
    nb = to_notebook(src)
    assert [c.cell_type for c in nb.cells] == ['markdown', 'code', 'markdown'] and nb.cells[1].source == 'x = 1'
