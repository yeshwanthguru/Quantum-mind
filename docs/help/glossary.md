# Glossary

Terms used across the documentation, in alphabetical order. The [mathematics pages](../math/index.md)
give the equations.

```{glossary}
Adjoint method
  A way to compute the gradient of a circuit's output with respect to every angle in about three
  simulations, by running the circuit forward and a co-state backward. Used by the simulator to
  train circuits with many angles.

AIC, BIC
  Akaike and Bayesian information criteria: $2k - 2\log L$ and $k\log n - 2\log L$. Lower is better;
  both penalise the number of parameters $k$, BIC more strongly for large samples $n$.

Amplitude
  A (possibly complex) number $\psi_i$ whose squared magnitude $\lvert\psi_i\rvert^2$ is a probability. Amplitudes
  add before squaring, which is the source of interference.

Anchoring model
  Classical baseline for order effects: the second answer is pulled towards the first.

Ask or act
  The robot's decision rule: ask when the expected cost of acting, error cost times $(1 - p)$, is larger
  than the cost of asking, or when its models of the person disagree.

Atom interferometer
  A sensor that splits and recombines a cloud of cold atoms with laser pulses; it reads acceleration as a phase
  $\varphi = k_{\mathrm{eff}} a T^2$ and is very sensitive and almost drift-free, but slow.

Baseline
  The classical model a quantum-like model is compared with on the same
  data. Every model in the library has at least one.

Bloch sphere
  The picture of a qubit's state as a point on (pure) or inside (mixed) a unit sphere.

Born rule
  The probability of an outcome is the squared length of the state's projection onto that outcome:
  $p = \lVert P\psi\rVert^2$.

Calibration
  A confidence is calibrated when events reported with confidence $c$ happen with frequency $c$.

Clopper-Pearson interval
  Exact binomial confidence interval for a success rate; its lower end is a safe confidence for
  risky actions.

COBRA
  Context-rotated density-matrix belief filter: the previous sensor frame rotates the basis in which the
  next frame is measured, so what a reading means depends on what came before it.

Coherence
  The off-diagonal part of a density matrix. COBRA reports its l1 value (the sum of the moduli of the
  off-diagonal entries) as a diagnostic.

Concurrence
  A measure of two-qubit entanglement between 0 (product state) and 1 (maximally entangled).

Conformal prediction
  A method that turns any classifier's scores into prediction sets containing the true label with a
  chosen probability, for exchangeable data.

Decoherence, dephasing
  Loss of the off-diagonal terms of a density matrix; a quantum-like belief then behaves more and more
  like a classical one.

Dempster-Shafer theory
  Evidence theory in which mass can be assigned to "any hypothesis" (ignorance) as well as to single
  hypotheses.

Density matrix
  $\rho$, a positive matrix with trace 1 that describes pure and mixed states.

Dephasing
  Loss of coherence: the off-diagonal entries shrink while the diagonal is unchanged.

ECE
  Expected calibration error: the bin-weighted average gap between accuracy and confidence.

Empirical Bayes
  Estimating a prior from data, for example the typical confidence bias of a module from earlier tasks.

Entanglement
  Correlation between qubits that no product of single-qubit states can describe.

Generalised measurement
  An update $\rho \to M\rho M^\dagger/\mathrm{tr}(M\rho M^\dagger)$ with a positive operator $M$; with a diagonal
  $M$ it is Bayes' rule, with a rotated $M$ it reads the evidence in a different basis.

Gymnasium
  The standard Python interface for reinforcement-learning environments (`reset`, `step`).

Incompatible questions
  Questions whose projectors do not commute, $P_AP_B \neq P_BP_A$; the order of asking them then
  matters.

Interference term
  The extra term $2\sqrt{\cdot}\cos\theta$ in a quantum-like probability that makes it differ from the
  classical law of total probability.

Kalman filter
  The optimal linear estimator of a state from noisy measurements; the multi-rate version here fuses sensors that report at
  different rates.

Kraus operator
  A matrix $K$ that updates a state as $\psi \mapsto K\psi/\lVert K\psi\rVert$; used for quantum-like cue fusion.

Lindblad equation
  The equation of motion of an open quantum system: unitary evolution plus dissipation and dephasing.

Lüders rule
  After a "yes", the state becomes its normalised projection $P\psi/\lVert P\psi\rVert$.

MAP estimate
  Maximum a posteriori: the parameters that maximise likelihood times prior.

Model recovery
  Simulating data from each candidate model and checking that model comparison picks the true one.

Order effect
  The answer to a question depends on which question was asked before it.

Partial pooling
  Estimating each person's parameters with a prior learnt from the population, so estimates are shrunk
  towards the population when data are few.

Projector
  A matrix $P$ with $P^2 = P = P^\dagger$; it represents the "yes" answer to a question.

QAOA
  Quantum Approximate Optimisation Algorithm: alternating cost and mixer layers that prepare a state concentrated on good
  solutions of a binary optimisation problem. Here it runs through Qiskit on an exact simulator.

QQ equality
  A parameter-free prediction of every projective model of two questions: the probability of giving
  different answers is the same in both orders.

Quantum Zeno effect
  Frequent observation slows the evolution of a quantum system; used to model bistable perception.

Quantum-like
  A model of human judgement that uses quantum probability; it assumes nothing quantum in the brain.

Qubit
  A two-level quantum system, $\alpha\lvert 0\rangle + \beta\lvert 1\rangle$ with $\lvert\alpha\rvert^2 + \lvert\beta\rvert^2 = 1$.

QUBO
  Quadratic unconstrained binary optimisation: minimise $x^TQx$ over bit strings. Robot allocation, grasp selection and
  scheduling are encoded this way with penalty terms.

REINFORCE
  Policy-gradient method that increases the log-probability of actions that led to better than
  average returns.

Standard quantum limit
  The best phase resolution, $1/\sqrt{N}$ for $N$ independent probes; entangled probes can in principle reach the Heisenberg
  limit $1/N$.

Temperature scaling
  Recalibration of a classifier by dividing its logits by one fitted temperature $T$.

Value of information (VOI)
  How much a question is expected to reduce the cost of the final decision.

Variational circuit
  A parameterised circuit whose angles are trained by a classical optimiser.
```
