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
