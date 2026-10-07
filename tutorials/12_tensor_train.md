# Compressing a network with tensor trains

Robots have small computers. A dense layer with a 256 × 1024 weight matrix stores 262,144 numbers. The
**tensor-train** (TT, or matrix product operator) format reshapes the matrix into a tensor and
factorises it into a chain of small cores by successive truncated SVDs. The **bond dimension** (TT
rank) sets the trade-off between size and accuracy. This is the structure physicists use for quantum
many-body states, which is why the method is called quantum-inspired.

This tutorial compresses a trained classifier, measures size, accuracy and speed, and states where
the method helps and where it does not.

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from quantum_mind.viz import use_mpl_style
from quantum_mind.inspired.tensor_layers import TTMatrix, compress_layers, apply_mlp
theme = use_mpl_style('dark')
```

## 1. What the format looks like

```python
rng = np.random.default_rng(0)
W = rng.normal(size=(64, 256))
for r in (2, 4, 8, 16, 64):
    tt = TTMatrix.from_dense(W, out_modes=(4, 4, 4), in_modes=(8, 8, 4), max_rank=r)
    print('max rank %2d: cores %s, ranks %s, %5d numbers (%.0fx smaller), relative error %.3f'
          % (r, [c.shape for c in tt.cores], tt.ranks, tt.n_params, W.size / tt.n_params, tt.relative_error(W)))
```

A random matrix has no low-rank structure to exploit, so the error falls slowly. Only at full rank
is the factorisation exact.

## 2. Compress a trained network

Train a small network on the 8 × 8 digits, then compress its first layer at several ranks.

```python
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
net = MLPClassifier(hidden_layer_sizes=(256,), activation='tanh', max_iter=400, random_state=0).fit(Xtr, ytr)
layers = [(W.T, b) for W, b in zip(net.coefs_, net.intercepts_)]          # (out, in) weights
dense_acc = (apply_mlp(layers, Xte).argmax(1) == yte).mean()
print('dense network: %d parameters, test accuracy %.3f' % (sum(W.size + b.size for W, b in layers), dense_acc))

rows = []
for r in (1, 2, 4, 8, 16, 32):
    comp, rep = compress_layers(layers, max_rank=r, d=3, min_size=1000)
    acc = (apply_mlp(comp, Xte).argmax(1) == yte).mean()
    rows.append((r, rep['compressed_params'], rep['ratio'], acc))
    print('rank %2d: %6d parameters (%.1fx smaller), layer error %.2f, test accuracy %.3f'
          % (r, rep['compressed_params'], rep['ratio'], rep['errors'][0], acc))
```

```python
r_, n_, ratio_, acc_ = zip(*rows)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(ratio_, acc_, 'o-', color=theme['palette'][2], lw=2.2)
for r, x, a in zip(r_, ratio_, acc_):
    ax.annotate('rank %d' % r, (x, a), textcoords='offset points', xytext=(6, -12), color=theme['text'])
ax.axhline(dense_acc, color=theme['text'], ls='--', label='dense')
ax.set_xscale('log'); ax.set_xlabel('compression ratio'); ax.set_ylabel('test accuracy')
ax.set_title('Accuracy against compression'); ax.legend()
plt.show()
```

## 3. Speed: measure, do not assume

```python
Wbig = rng.normal(size=(1024, 1024)); Xb = rng.normal(size=(256, 1024))
for r in (4, 16, 64):
    tt = TTMatrix.from_dense(Wbig, (16, 8, 8), (16, 8, 8), max_rank=r)
    t0 = time.perf_counter(); [tt.matvec(Xb) for _ in range(20)]; t_tt = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter(); [Xb @ Wbig.T for _ in range(20)]; t_d = (time.perf_counter() - t0) / 20
    print('1024x1024, rank %2d: %6.0fx fewer numbers, TT %.2f ms, dense %.2f ms' % (r, Wbig.size / tt.n_params, 1e3 * t_tt, 1e3 * t_d))
```

**Result.** Compressing a trained layer after the fact with TT-SVD loses accuracy quickly on this
small network: the layer only becomes accurate again near full rank, where it is no smaller. In
practice TT layers are trained directly in the TT format (Novikov et al., 2015) or fine-tuned after
compression, which this tutorial does not do. TT compression can save memory by large factors when
the weights have low-rank structure. It does not automatically save time: optimised dense matrix products are very fast
on CPUs and GPUs, and the TT product is a chain of small ones. Measure on the target hardware.

References: Oseledets (2011), *SIAM J. Sci. Comput.* 33, 2295-2317; Novikov et al. (2015), NeurIPS.
