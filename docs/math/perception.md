# Perception and multimodal fusion

All cues are turned into likelihoods $L_c(h)$ over the same hypotheses $h = 1, \dots, K$ (the candidate
objects or intents), then fused.

## Adapters

A small floor $\varepsilon$ is added before normalising, $\mathrm{norm}(v)_h = (\max(v_h, 0) + \varepsilon)/\sum_{h'}(\cdot)$, so
no hypothesis gets probability zero.

**Object detector** (`detector_likelihood`). Confidences $s_h \in [0, 1]$ are tempered by a power, logits
by a softmax:

$$
L(h) = \mathrm{norm}\bigl(s_h^{1/T}\bigr) \quad\text{or}\quad L(h) = \mathrm{norm}\bigl(e^{(z_h - \max z)/T}\bigr).
$$

$T > 1$ flattens an over-confident detector.

**Speech n-best list** (`asr_likelihood`). For hypotheses (transcripts) $(t_j, c_j)$ with confidences $c_j$
and keyword sets $W_h$, let $\mathrm{hit}_j(h) = 1$ if a keyword of $h$ occurs in $t_j$. Then

$$
L(h) = \mathrm{norm}\Bigl(\sum_j c_j\,\frac{\mathrm{hit}_j(h)}{\sum_{h'}\mathrm{hit}_j(h')}\Bigr),
$$

and a transcript that mentions no keyword spreads its confidence uniformly.

**Gaze or pointing direction** (`direction_likelihood`). A von Mises-Fisher likelihood with concentration
$\kappa$ around the direction $d$ to each target $t_h$:

$$
L(h) = \mathrm{norm}\Bigl(\exp\bigl(\kappa\,(\cos\angle(d, t_h) - 1)\bigr)\Bigr), \qquad
\cos\angle(d, t_h) = \frac{d^T t_h}{\lVert d\rVert\,\lVert t_h\rVert}.
$$

## Bayesian fusion

Cues independent given the hypothesis:

$$
P(h \mid \text{cues}) \propto \pi(h)\prod_c L_c(h).
$$

The order of the cues does not matter. (`bayes_fusion`)

## Dempster-Shafer fusion

Each cue $c$ with reliability $r_c$ gives singleton masses $m_c(h) = r_c\,L_c(h)/\sum_{h'}L_c(h')$ and
ignorance $m_c(\Omega) = 1 - r_c$. Starting from total ignorance ($m = 0$, $\omega = 1$), Dempster's rule
for singleton masses is applied cue by cue:

$$
m'(h) = \frac{m(h)\,m_c(h) + m(h)\,m_c(\Omega) + \omega\,m_c(h)}{Z}, \qquad
\omega' = \frac{\omega\,m_c(\Omega)}{Z},
$$

with $Z = \sum_h[\cdots] + \omega\,m_c(\Omega) = 1 - \text{conflict}$. The output is the pignistic
probability $m(h) + \omega/K$ together with the remaining ignorance $\omega$. (`dempster_shafer_fusion`)

## Quantum-like fusion (incompatible cues)

The belief is an amplitude vector $\psi = \sqrt{\pi}$. Cue $c$ acts as a Kraus operator

$$
K_c = R(\theta_c)\,\mathrm{diag}\bigl(\sqrt{L_c}\bigr)\,R(\theta_c)^T,
$$

where $R(\theta_c)$ rotates the intent basis by the cue's incompatibility angle $\theta_c$ in the plane of
the cue's two most likely intents. For cues applied in the order $c_1, \dots, c_m$,

$$
P(h \mid c_1, \dots, c_m) = \frac{\lvert (K_{c_m}\cdots K_{c_1}\psi)_h\rvert^2}{\lVert K_{c_m}\cdots K_{c_1}\psi\rVert^2}.
$$

With every $\theta_c = 0$ the operators are diagonal and commute, and this is exactly Bayes' rule; with
$\theta_c \neq 0$ the result depends on the cue order. (`QuantumIntentResolver`; `BayesIntentResolver` is the
$\theta = 0$ baseline.)

**Fitting the angles** (`fit_incompatibility`). From logged trials (cue order $o_i$, chosen intent $y_i$),

$$
\hat\theta = \arg\max_{\theta \in [0, \pi/2]^C} \sum_i \log P_\theta(y_i \mid o_i).
$$

**Comparison** (`compare_fusion`): held-out log loss $-\frac1n\sum_i \log P(y_i \mid o_i)$ and accuracy for the
three rules.

## Bistable perception

The quantum Zeno model and its baselines are on the
[quantum-like families page](quantum_like.md#bistable-perception-quantum-zeno).

References: Fisher (1953) and Mardia & Jupp (2000) for the von Mises-Fisher distribution; Shafer (1976);
Smets & Kennes (1994) for the pignistic transform; Busemeyer & Bruza (2024), chapter 4.
