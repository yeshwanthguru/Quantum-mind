"""Robotics: learning to navigate a grid with quantum-inspired reinforcement learning (Dong et al.)
versus classical Q-learning, 10 seeds each. The shortest path has 10 steps."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # run without installing
import numpy as np
from qlcog.inspired import GridWorld, QuantumInspiredQLearning, QLearning, train

env = GridWorld(); print('grid %dx%d, walls %s, shortest path %d steps' % (env.size, env.size, sorted(env.walls), env.shortest_path()))
for step_reward in (0.0, -0.01):
    for name, make in (('quantum-inspired QRL', lambda s: QuantumInspiredQLearning(env.n_states, env.n_actions, seed=s)),
                       ('Q-learning (epsilon-greedy)', lambda s: QLearning(env.n_states, env.n_actions, seed=s))):
        runs = np.array([train(make(seed), GridWorld(step_reward=step_reward), 300) for seed in range(10)])
        print('step reward %5.2f  %-28s steps/episode: first 20 %.1f, last 50 %.1f +- %.1f'
              % (step_reward, name, runs[:, :20].mean(), runs[:, -50:].mean(), runs[:, -50:].mean(1).std()))
