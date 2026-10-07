# Deep learning

Deep learning trains **neural networks** with many layers. Each layer applies a linear map followed by
a simple non-linear function; stacked together, layers learn features directly from raw data such as
pixels, audio or joint readings. Training uses **gradient descent**, with gradients computed by
**backpropagation**.

```{mermaid}
flowchart LR
    X["input<br/>image · sound · state"]:::in --> L1["layer 1<br/>W₁x + b₁ → ReLU"]:::op --> L2["layer 2<br/>W₂h + b₂ → ReLU"]:::op --> O["output<br/>softmax probabilities"]:::out
    O --> LOSS["loss<br/>cross-entropy"]:::base
    LOSS -.->|backpropagation: ∂loss/∂W| L2
    LOSS -.-> L1
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3
```

## From the basics

Neuron and layer
: A neuron computes $\sigma(w \cdot x + b)$: a weighted sum passed through an **activation** such as
  ReLU or tanh. A layer is many neurons in parallel, written as one matrix product.

Gradient descent and backpropagation
: Parameters move a small step against the gradient of the loss. Backpropagation computes all
  gradients in one backward pass using the chain rule. Variational quantum circuits are trained the
  same way, with gradients from the **parameter-shift rule** or the **adjoint method**.

Convolutional networks (CNNs)
: Layers that slide a small filter over an image, so the same pattern detector is used everywhere.
  The basis of most computer vision. A **quanvolutional** filter replaces the classical filter with a
  small random quantum circuit.

Recurrent networks and transformers
: Models for sequences. Transformers use **attention** to relate every element of a sequence to every
  other and are the basis of LLMs and VLMs.

Softmax and confidence
: The softmax turns scores into probabilities. These probabilities are often over-confident, which is
  why robots should recalibrate them before trusting them.

Compression
: Robots have small computers, so large networks are pruned, quantised or **factorised**. The
  **tensor-train** (matrix product operator) format stores a weight matrix as a chain of small cores,
  the same structure physicists use for quantum many-body states.

## A small example: a two-layer network in NumPy

```python
import numpy as np
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(16, 4)), np.zeros(16)
W2, b2 = rng.normal(size=(3, 16)), np.zeros(3)

def forward(x):
    h = np.maximum(0, W1 @ x + b1)              # ReLU hidden layer
    z = W2 @ h + b2
    return np.exp(z) / np.exp(z).sum()          # softmax over 3 classes

print(forward(np.array([0.1, -0.3, 0.8, 0.5])).round(3))
```

## A small example: compressing a layer

```python
from quantum_mind.inspired.tensor_layers import TTMatrix
W = rng.normal(size=(64, 256))
tt = TTMatrix.from_dense(W, out_modes=(4, 4, 4), in_modes=(8, 8, 4), max_rank=4)
print(W.size, tt.n_params)                      # 16384 numbers against far fewer
```

## Where Quantum Mind fits

The library does not replace deep networks; it sits next to them. Their outputs (detector scores,
speech hypotheses) are recalibrated and fused, and the quantum and quantum-inspired pillars offer
variational circuits, quanvolution and tensor-train layers, each measured against a classical
baseline. Several of those measurements favour the classical method, and the docs say so.

- [Tutorial: multimodal fusion](../tutorials/08_fusion.ipynb)
- [Tutorial: tensor-train compression](../tutorials/12_tensor_train.ipynb)
- [Tutorial: quanvolution](../tutorials/13_quanvolution.ipynb)

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436-444.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing
  Systems*, 30.
- Novikov, A., Podoprikhin, D., Osokin, A., & Vetrov, D. (2015). Tensorizing neural networks.
  *Advances in Neural Information Processing Systems*, 28.
- Henderson, M., Shakya, S., Pradhan, S., & Cook, T. (2020). Quanvolutional neural networks: Powering
  image recognition with quantum circuits. *Quantum Machine Intelligence*, 2, 2.
