"""Environments with simulated people and the plotting style helpers."""
import numpy as np
import pytest

from quantum_mind.envs import ClarificationEnv, TrustHandoverEnv, register_envs


def test_environments_follow_the_gymnasium_api():
    gym = pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env
    for env in (ClarificationEnv(), TrustHandoverEnv()):
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
