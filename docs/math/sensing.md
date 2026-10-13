# Quantum sensing

## Quantum limits

A phase read from $N$ independent probes repeated $\nu$ times is limited by the **standard quantum limit**
$\Delta\varphi_{\mathrm{SQL}} = 1/\sqrt{N\nu}$; entangled probes can in principle reach the **Heisenberg limit**
$\Delta\varphi_{\mathrm{H}} = 1/(N\sqrt{\nu})$ (Giovannetti, Lloyd and Maccone, 2004). `phase_sensitivity` returns either.

## Atom interferometer

A light-pulse atom interferometer with pulse separation $T$ and effective wave number $k_{\mathrm{eff}} = 4\pi/\lambda$ reads a
constant acceleration $a$ as the phase $\varphi = k_{\mathrm{eff}}\,a\,T^2$ (Kasevich and Chu, 1992). With fringe contrast $C$ the
single-shot acceleration sensitivity is

$$
\Delta a = \frac{\Delta\varphi}{C\,k_{\mathrm{eff}}\,T^2} ,
$$

so it improves as $1/\sqrt N$ and as $1/T^2$ (`atom_interferometer_sensitivity`).

## Accelerometer model

A sensor $k$ outputs, at its rate $f_k$, $z = a + b_k + n$ with white noise $n \sim \mathcal{N}(0, \sigma_k^2)$ and a random-walk bias
$b_k(t + \mathrm{d}t) = b_k(t) + w$, $w \sim \mathcal{N}(0, \beta_k^2\,\mathrm{d}t)$, initial bias $\mathcal{N}(0, b_{0,k}^2)$ (`Sensor`).

## Dead-reckoning filter

The true acceleration is an Ornstein-Uhlenbeck process, $a(t + \mathrm{d}t) = e^{-\mathrm{d}t/\tau}a(t) + \sqrt{1 - e^{-2\mathrm{d}t/\tau}}\,\sigma_a\,\xi$.
The Kalman state is $(p, v, a, b_1, \dots, b_K)$ with

$$
p \leftarrow p + v\,\mathrm{d}t + \tfrac12 a\,\mathrm{d}t^2, \qquad v \leftarrow v + a\,\mathrm{d}t, \qquad a \leftarrow e^{-\mathrm{d}t/\tau}a, \qquad b_k \leftarrow b_k ,
$$

process noise $\sigma_a^2(1 - e^{-2\mathrm{d}t/\tau})$ on $a$ and $\beta_k^2\,\mathrm{d}t$ on $b_k$. Sensor $k$ updates the filter with
$H = e_a + e_{b_k}$ and $R = \sigma_k^2$ at its own sample times; a position fix updates with $H = e_p$ and its own variance
(Kalman, 1960). Position error grows when a bias cannot be separated from the true acceleration; a sensor with a much smaller bias
makes the other sensor's bias observable, which is why a slow, stable sensor helps a fast, drifting one.

## Scheduling the quantum sensor with a QUBO

The weighted position variance of the fused Kalman filter for a set $S$ of quantum readings is
$J(S)=\sum_t w_t\,P^{S}_{pp}(t)$, with $P^S$ from the covariance recursion in which a reading at slot $i$ is a measurement with variance
$\sigma_q^2/c_i^2$ ($c_i$ is the fringe contrast; slots with $c_i$ below a threshold are removed). This is exact and needs no simulation noise.

Around a schedule $x^0\in\{0,1\}^n$, with $\delta_i=x_i-x^0_i$ and $s_i=1-2x^0_i$ the change of $x_i$ when flipped,

$$ J(x)\approx J_0+\sum_i a_i\,\delta_i+\sum_{i<j} b_{ij}\,\delta_i\delta_j,\qquad
a_i=\frac{J(x^0\oplus e_i)-J_0}{s_i},\quad
b_{ij}=\frac{J(x^0\oplus e_i\oplus e_j)-J_0-(J(x^0\oplus e_i)-J_0)-(J(x^0\oplus e_j)-J_0)}{s_is_j}. $$

The model is exact for every schedule within two flips of $x^0$. Choosing exactly $k$ slots under it is

$$ \min_x\ \sum_i \ell_i x_i+\sum_{i<j} b_{ij}x_ix_j\quad\text{s.t.}\ \sum_i x_i=k,\qquad \ell_i=a_i-\sum_{j\ne i}b_{ij}x^0_j, $$

the cardinality-constrained QUBO of `grasp_selection` (Lucas, 2014 penalty encoding). Its minimiser becomes the next expansion point if
$J$ improves, so the iteration never ends worse than its start. From the empty schedule the same formula is the inclusion-exclusion expansion,
which fails when readings are almost redundant: if one reading does nearly all the work, $\sum_i F_i-\sum_{i<j}(F_i+F_j-F_{ij})$ cancels to
about zero while the true gain is that of one reading.
