# A variational quantum policy

A **policy** maps a state to action probabilities. A variational quantum policy computes those
probabilities as the Born-rule probabilities of a parameterised circuit: the state's features set
rotation angles (data re-uploading), trainable rotations and entangling gates follow, and the
read-out qubits are measured. Training uses **REINFORCE**: make actions that led to higher return
more likely. Gradients are exact, from the adjoint method of the built-in simulator.

This tutorial trains the policy on CartPole (balance a pole on a cart), compares it with a classical
linear softmax policy trained the same way, and exports the circuit to Qiskit.

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from quantum_mind.viz import use_mpl_style
from quantum_mind.quantum.policy import VariationalPolicy, reinforce
theme = use_mpl_style('dark')

def features(obs):
    """Cart position, velocity, pole angle and angular velocity, squashed to rotation angles."""
    return np.tanh(np.asarray(obs) * np.array([1.0, 0.5, 4.0, 1.0])) * np.pi / 2

pi = VariationalPolicy(n_features=4, n_actions=2, layers=2, lr=0.1, seed=0)
print('qubits:', pi.circuit.n, '  trainable angles:', pi.circuit.n_weights)
print('action probabilities at rest:', pi.probabilities([features(np.zeros(4))]).round(3))
```

## 1. Train

```python
env = gym.make('CartPole-v1')
returns_q = reinforce(pi, env, 250, features, gamma=0.99, batch=5, seed=0)
print('episode length: first 25 episodes %.1f, last 25 episodes %.1f' % (returns_q[:25].mean(), returns_q[-25:].mean()))
```

## 2. A classical baseline with the same training loop

A linear softmax policy with 10 parameters (one weight per feature and action, plus biases), trained
with the same REINFORCE settings.

```python
class LinearSoftmaxPolicy:
    def __init__(self, n_features, n_actions, lr=0.05, seed=0):
        self.W = np.zeros((n_actions, n_features)); self.b = np.zeros(n_actions)
        self.lr, self.rng, self.n_actions = lr, np.random.default_rng(seed), n_actions
    def probabilities(self, S):
        z = np.atleast_2d(S) @ self.W.T + self.b
        e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    def act(self, s):
        return int(self.rng.choice(self.n_actions, p=self.probabilities(s)[0]))
    def update(self, S, A, adv):
        P = self.probabilities(S); G = -P; G[np.arange(len(A)), A] += 1          # d log pi / d logits
        self.W += self.lr * (G * adv[:, None]).T @ np.atleast_2d(S) / len(A)
        self.b += self.lr * (G * adv[:, None]).mean(0)

returns_c = reinforce(LinearSoftmaxPolicy(4, 2, lr=0.05), gym.make('CartPole-v1'), 250, features, gamma=0.99, batch=5, seed=0)
print('linear softmax: first 25 episodes %.1f, last 25 episodes %.1f' % (returns_c[:25].mean(), returns_c[-25:].mean()))
```

```python
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for r, name, c in ((returns_q, 'variational quantum policy (20 angles)', theme['palette'][2]),
                   (returns_c, 'linear softmax policy (10 weights)', theme['palette'][1])):
    ax.plot(r, color=c, alpha=0.25)
    ax.plot(np.arange(19, len(r)), np.convolve(r, np.ones(20) / 20, mode='valid'), color=c, lw=2.2, label=name)
ax.set_xlabel('episode'); ax.set_ylabel('episode length (max 500)'); ax.set_title('CartPole with REINFORCE'); ax.legend()
plt.show()
```

## 3. The circuit

The trained policy for one state, as a Qiskit circuit that can run on Aer or IBM Quantum (needs
`pip install "quantum-mind[qiskit]"`).

```python
try:
    qc = pi.to_qiskit(features([0.0, 0.1, 0.05, -0.2]))
    print(qc.draw(output='text', fold=110))
except ImportError:
    print('install qiskit to export the circuit')
```

**Result.** Both policies learn, but the classical linear policy learns much faster: over the last 25
of 250 episodes it balances the pole for about 378 steps on average, against about 90 for the quantum
policy. The quantum policy also has to be simulated, which costs far more than evaluating a linear
model. It is a teaching and research tool for small tasks, not a controller for a real robot.

References: Williams (1992), *Machine Learning* 8, 229-256; Jerbi et al. (2021), NeurIPS; Pérez-Salinas
et al. (2020), *Quantum* 4, 226 (data re-uploading).
