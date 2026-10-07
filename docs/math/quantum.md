# Quantum models

## Circuits, encodings and gradients

An $n$-qubit state $\lvert\psi\rangle \in \mathbb{C}^{2^n}$ evolves under gates; rotations are
$R_\sigma(\theta) = e^{-i\theta\sigma/2}$ for $\sigma \in \{X, Y, Z\}$, and qubit $q$ is bit $q$ of the basis
index (Qiskit ordering). Batches of inputs are simulated at once.

**Angle encoding**: $\prod_j R_y^{(j)}(s\,x_j)\lvert 0\rangle$.
**ZZ feature map** (Havlíček et al.): per repetition, $H$ and $R_z(2x_i)$ on each qubit, then
$R_{zz}(2(\pi - x_i)(\pi - x_j))$ on each pair.
**Hardware-efficient ansatz**: layers of single-qubit rotations with one weight each, then entanglers along
a chain. **Data re-uploading**: encoding and trainable layers alternate, so the output is a trainable
Fourier series in the inputs.

**Parameter-shift rule** for a weight entering a single rotation $e^{-iwG/2}$ with $G^2 = I$:

$$
\frac{\partial f}{\partial w_k} = \frac{f(w + \tfrac\pi2 e_k) - f(w - \tfrac\pi2 e_k)}{2}.
$$

**Adjoint method** (used by the simulator): for $f = \mathcal{L}(\lvert\psi(w)\rangle)$ with
$\lvert\psi\rangle = U_L\cdots U_1\lvert 0\rangle$, propagate $\lvert\lambda\rangle = \partial\mathcal L/\partial\langle\psi\rvert$ backwards; each
gate contributes $\partial f/\partial w_k = 2\,\mathrm{Re}\,\langle\lambda_k\rvert\,\partial_{w_k}U_k\,\lvert\psi_{k-1}\rangle$, so all
gradients cost about as much as three forward simulations.

## Variational classifier and regressor

Class probabilities are the Born-rule marginals of the first $\lceil\log_2 C\rceil$ qubits, restricted to the
$C$ labels and renormalised; training minimises the cross-entropy $-\frac1n\sum_i\log p_w(y_i \mid x_i)$ with
L-BFGS. The regressor predicts $\hat y = \alpha\langle Z_0\rangle + \beta$ (scale and offset from the target
range) and minimises the squared error. (`VariationalClassifier`, `VariationalRegressor`)

## Quantum kernels

$$
k(x, x') = \lvert\langle\phi(x)\vert\phi(x')\rangle\rvert^2 \qquad \text{(fidelity kernel, ZZ feature map)}.
$$

- **Kernel ridge classifier**: one-versus-rest with targets $\pm1$, $\alpha = (K + \lambda I)^{-1}y$,
  $f(x) = \sum_i \alpha_i k(x, x_i)$.
- **Anomaly score**: squared feature-space distance to the mean of the normal data,
  $k(x, x) - \frac2n\sum_i k(x, x_i) + \frac{1}{n^2}\sum_{ij}k(x_i, x_j)$; threshold at a training quantile.
- **Spectral clustering**: leading eigenvectors of $D^{-1/2}KD^{-1/2}$, $D = \mathrm{diag}(K\mathbf 1)$, then
  $k$-means.

The baseline is the RBF kernel $k(x, x') = \exp(-\gamma\lVert x - x'\rVert^2)$.

## QAOA

For a QUBO with Ising form $C(z) = \sum_i h_iz_i + \sum_{i<j}J_{ij}z_iz_j + \mathrm{const}$, $z_i = \pm1$:

$$
\lvert\gamma, \beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l\sum_j X_j}\,e^{-i\gamma_l C}\ \lvert +\rangle^{\otimes n}, \qquad
\min_{\gamma, \beta}\ \langle\gamma, \beta\rvert C\lvert\gamma, \beta\rangle .
$$

The cost layer is built from $R_z$ and $R_{zz}$ gates. Angles are optimised with exact gradients (L-BFGS-B,
then a COBYLA polish), starting from a depth-1 grid search extended layer by layer (INTERP) and random
restarts. Reported: the expectation, the most likely bit string and $P(\text{optimal})$.

## VQE

$$
E(\theta) = \langle\psi(\theta)\rvert H\lvert\psi(\theta)\rangle, \qquad H = \sum_k c_k\,\sigma^{(k)}_1\otimes\cdots\otimes\sigma^{(k)}_n,
$$

minimised over a hardware-efficient ansatz from several starts; $E(\theta) \ge E_0$ by the variational
principle. The baseline is exact diagonalisation.

## Grover search

With $M$ marked items among $N = 2^n$, the oracle $O = I - 2\sum_{x\in\text{marked}}\lvert x\rangle\langle x\rvert$ and the
diffusion $D = 2\lvert s\rangle\langle s\rvert - I$ ($\lvert s\rangle$ uniform). After $k$ iterations

$$
P(\text{marked}) = \sin^2\bigl((2k + 1)\theta\bigr), \quad \sin\theta = \sqrt{M/N}, \qquad
k^\star = \Bigl\lfloor\frac{\pi}{4}\sqrt{N/M}\Bigr\rfloor .
$$

## Quantum Fourier transform and period finding

$$
\mathrm{QFT}\,\lvert j\rangle = \frac{1}{\sqrt N}\sum_{k=0}^{N-1} e^{2\pi i jk/N}\lvert k\rangle .
$$

On a uniform superposition over $\{x_0 + mr\}$ with period $r \mid N$, the QFT output is supported on the
multiples of $N/r$; the period is read from the peaks.

## Amplitude estimation

$$
A\lvert 0\rangle\lvert 0\rangle = \sum_i\sqrt{p_i}\,\lvert i\rangle\bigl(\sqrt{1 - f_i}\lvert 0\rangle + \sqrt{f_i}\lvert 1\rangle\bigr), \qquad
a = \sum_i p_if_i = \mathbb{E}[f] = \sin^2\theta .
$$

With the Grover operator $Q = AS_0A^\dagger S_\chi$, measuring $Q^mA\lvert 0\rangle$ gives 1 with probability
$\sin^2((2m + 1)\theta)$. Maximum-likelihood amplitude estimation combines $h_m$ ones out of $N_m$ shots at
$m = 0, 1, 2, 4, \dots$:

$$
\hat\theta = \arg\max_\theta \sum_m \Bigl[h_m\log\sin^2((2m{+}1)\theta) + (N_m - h_m)\log\cos^2((2m{+}1)\theta)\Bigr], \qquad \hat a = \sin^2\hat\theta .
$$

Its error falls roughly as $1/(\text{calls to } A)$, against $1/\sqrt{\text{samples}}$ for Monte Carlo (the
baseline).

## Quantum walks and centrality

$\lvert\psi(t)\rangle = e^{-iAt}\lvert\psi(0)\rangle$ with $A$ the adjacency matrix. Quantum-walk centrality is
the long-time average of the occupation probabilities from the uniform superposition,

$$
\bar p_v = \lim_{T\to\infty}\frac1T\int_0^T\lvert\langle v\vert\psi(t)\rangle\rvert^2\,dt = \sum_\lambda \lvert\langle v\rvert P_\lambda\lvert\psi(0)\rangle\rvert^2,
$$

summed over the eigenspace projectors $P_\lambda$ of $A$. Baselines: PageRank,
$r = \frac{1 - d}{n}\mathbf 1 + d\,A^T D^{-1} r$, and normalised degree.

References: Schuld & Petruccione (2021); Mitarai et al. (2018) and Schuld et al. (2019) for parameter
shift; Jones & Gacon (2020) for adjoint differentiation; Havlíček et al. (2019); Farhi, Goldstone &
Gutmann (2014); Zhou et al. (2020) for INTERP; Peruzzo et al. (2014); Grover (1996); Nielsen & Chuang
(2010); Suzuki et al. (2020); Farhi & Gutmann (1998); Izaac et al. (2017).
