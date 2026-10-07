"""Reinforcement learning with simulated people
=============================================

Learning, robotics: tabular agents learn to clarify an ambiguous request from a simulated person whose
answers depend on question order, with amplitude (Born-rule) exploration next to epsilon-greedy,
softmax and UCB; then a small variational quantum policy is trained with REINFORCE on CartPole.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_35.png'
import numpy as np
from quantum_mind.envs import ClarificationEnv
from quantum_mind.inspired.exploration import (EpsilonGreedy, Boltzmann, UCB, AmplitudeExploration, TabularAgent,
                                               StateIndexer, run_episodes)
from quantum_mind.quantum.policy import VariationalPolicy, reinforce

# %%
# Clarification with a simulated person
# -------------------------------------
for name, make in (('epsilon-greedy', lambda: EpsilonGreedy(seed=0)), ('softmax', lambda: Boltzmann(seed=0)),
                   ('UCB', lambda: UCB(seed=0)), ('amplitude', lambda: AmplitudeExploration(64, 6, seed=0))):
    agent = TabularAgent(64, 6, make())
    out = run_episodes(agent, ClarificationEnv(), 400, state_fn=StateIndexer(64), max_steps=10)
    print('%-15s mean return over the last 100 episodes %.2f' % (name, out['returns'][-100:].mean()))

# %%
# A variational quantum policy on CartPole
# ----------------------------------------
import gymnasium as gym


def features(obs):
    return np.tanh(np.asarray(obs) * np.array([1.0, 0.5, 4.0, 1.0])) * np.pi / 2


pi = VariationalPolicy(n_features=4, n_actions=2, layers=2, lr=0.1, seed=0)
returns = reinforce(pi, gym.make('CartPole-v1'), 150, features, batch=5)
print('quantum policy, mean episode length: first 25 episodes %.1f, last 25 episodes %.1f'
      % (returns[:25].mean(), returns[-25:].mean()))
print('A linear softmax policy trained the same way learns faster (tutorial 11).')
