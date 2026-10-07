"""Robot navigation with quantum-inspired RL
=========================================

Robotics: learning to navigate a grid with quantum-inspired reinforcement learning (Dong et al.)
versus classical Q-learning, 10 seeds each. The shortest path has 10 steps.
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ex_30.png'
import numpy as np
from quantum_mind.inspired import GridWorld, QuantumInspiredQLearning, QLearning, train

# %%
# The grid and two learners
# -------------------------
# Ten seeds per learner and reward setting; the table reports early and late episode lengths.
env = GridWorld(); print('grid %dx%d, walls %s, shortest path %d steps' % (env.size, env.size, sorted(env.walls), env.shortest_path()))
for step_reward in (0.0, -0.01):
    for name, make in (('quantum-inspired QRL', lambda s: QuantumInspiredQLearning(env.n_states, env.n_actions, seed=s)),
                       ('Q-learning (epsilon-greedy)', lambda s: QLearning(env.n_states, env.n_actions, seed=s))):
        runs = np.array([train(make(seed), GridWorld(step_reward=step_reward), 300) for seed in range(10)])
        print('step reward %5.2f  %-28s steps/episode: first 20 %.1f, last 50 %.1f +- %.1f'
              % (step_reward, name, runs[:, :20].mean(), runs[:, -50:].mean(), runs[:, -50:].mean(1).std()))
