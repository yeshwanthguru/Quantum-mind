# Quanvolution: quantum filters for images

A convolutional layer slides small filters over an image. A **quanvolutional** layer (Henderson et
al., 2020) replaces each filter with a small quantum circuit: every 2 × 2 patch is encoded into four
qubits (one $R_y(\pi x)$ rotation per pixel), a fixed random circuit is applied, and the Pauli-Z
expectation of each qubit becomes one output channel. A classical classifier is trained on the
result. This tutorial builds the feature maps, looks at them, and compares them fairly with a
classical random filter of the same shape and with the raw pixels.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from quantum_mind.viz import use_mpl_style
from quantum_mind.quantum.quanvolution import QuanvolutionFilter, RandomConvFilter
theme = use_mpl_style('dark')
digits = load_digits()
images, labels = digits.images / 16.0, digits.target
qf = QuanvolutionFilter(size=2, stride=2, layers=1, seed=0)
print('qubits per patch:', qf.n_qubits, '   output shape for one image:', qf.transform(images[:1]).shape)
```

## 1. See the feature maps

```python
maps = qf.transform(images[:3])
fig, axes = plt.subplots(3, 5, figsize=(8.4, 5.2))
for i in range(3):
    axes[i, 0].imshow(images[i], cmap='magma'); axes[i, 0].set_title('digit %d' % labels[i])
    for k in range(4):
        axes[i, k + 1].imshow(maps[i, :, :, k], cmap='twilight', vmin=-1, vmax=1)
        axes[i, k + 1].set_title('qubit %d ⟨Z⟩' % k)
for ax in axes.ravel():
    ax.axis('off')
fig.suptitle('8×8 digits and their four quanvolution channels (4×4 each)'); plt.tight_layout(); plt.show()
```

## 2. The circuit for one patch

```python
try:
    print(qf.to_qiskit(images[0, 2:4, 2:4]).draw(output='text'))     # a patch from the middle of the digit
except ImportError:
    print('install qiskit to export the circuit')
```

## 3. Fair comparison

Logistic regression, 5-fold cross-validation, the same number of features (64) for each input.

```python
n = len(images)
inputs = {'raw pixels': images.reshape(n, -1)}
for s in range(3):
    inputs['quanvolution, circuit %d' % s] = QuanvolutionFilter(seed=s).transform(images).reshape(n, -1)
    inputs['random classical filter %d' % s] = RandomConvFilter(seed=s).transform(images).reshape(n, -1)
scores = {}
for k, F in inputs.items():
    sc = cross_val_score(LogisticRegression(max_iter=3000), F, labels, cv=5)
    scores[k] = sc
    print('%-28s accuracy %.3f ± %.3f' % (k, sc.mean(), sc.std()))
```

```python
fig, ax = plt.subplots(figsize=(7.0, 3.4))
names = list(scores)
cols = [theme['text'] if n.startswith('raw') else theme['palette'][2] if n.startswith('quan') else theme['palette'][1] for n in names]
ax.barh(names[::-1], [scores[k].mean() for k in names][::-1], xerr=[scores[k].std() for k in names][::-1], color=cols[::-1])
ax.set_xlim(0.6, 1.0); ax.set_xlabel('cross-validated accuracy'); ax.set_title('Digits: quantum against classical features')
plt.show()
```

**Result.** On this data the quanvolution features are less accurate than both the random classical
filter and the raw pixels. The filter is included as a common reference point in quantum image
processing and because its circuits export to Qiskit, not because it helps a robot's vision.

References: Henderson et al. (2020), *Quantum Machine Intelligence* 2, 2; LeCun et al. (1998),
*Proceedings of the IEEE* 86, 2278-2324.
