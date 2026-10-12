# `quantum_mind.embodied`: QR-VLA, a one-motor quantum-robotics vision-language-action prototype

A deliberately small system to build the fourth strand of the library on: embodied AI. One simulated DC-motor joint must move to
the object that an instruction names on a one-dimensional table. Everything (model, gradients, training) is NumPy.

```
image ──┐
        ├─► grounding slot (attention) ─► belief over cells ─► ask gate ─► belief fusion ─► action slot ─► motor command
text  ──┘                                                                                      ▲
                                                                          noisy encoder: position, velocity
```

| Piece | What it is |
|---|---|
| `OneMotor`, `expert_command` | DC-motor joint with friction, command saturation, domain randomisation and a noisy encoder; saturated PD demonstrator |
| `sample_scene`, `clarification` | 2-3 coloured objects, instructions ("go to the red / leftmost / rightmost / block"), and the person's clarifying answer |
| `TinyVLA` | grounding by one attention step (817 parameters) + action head; hand-written gradients, checked against finite differences |
| `BCBaseline` | the usual end-to-end behaviour-cloning policy (image, text, state to command), the baseline |
| `calibrate_gate`, `vla_plan`, `fuse` | split-conformal "ask when the prediction set has more than one cell", then Bayes fusion of the answer |

Run `python benchmarks/one_motor_vla_benchmark.py` for the closed-loop study (results in `benchmarks/results/one_motor_vla/`).

## What this is and is not

* It is a prototype to scale from: replace the toy image with a camera encoder, the bag of words with a language model, the
  single joint with a robot, and keep the slots.
* It is **not** a new VLA architecture and it is not competitive with large VLA models. Asking for help with conformal prediction
  has prior art in LLM planning (e.g. KnowNo, Ren et al., 2023); the grounding-by-attention and behaviour-cloning ingredients are standard.
* What the library adds, and what is still to be tested here before any novelty is claimed: (1) a belief slot that is
  **quantum-probability** (order effects between an instruction and its clarification) instead of the classical product used now;
  (2) a state slot fed by the **quantum-sensing** estimator of `quantum_mind.sensing`; (3) assignment of several arms with the
  QUBO tools of `quantum_mind.computing`. Each needs the same treatment as the rest of the library: a classical baseline, an
  ablation, a control task and paired tests.
* All data are simulated. No real motor, camera or person was involved.

## Belief slot: what the benchmark found (`benchmarks/results/belief_slot/`)

`BayesSlot`, `TemperedSlot` (classical context weights) and `QuantumSlot` (a COBRA-style rotated measurement per answer context) fuse the
instruction with two clarifying answers. `qiskit_slot.qiskit_posterior` reproduces the quantum slot exactly as a circuit on a simulator.

* **Where the quantum slot wins:** only when the simulated person is built to be non-commuting (log-loss 0.60 against 0.80 for the
  classical context weights). That person uses the slot's own form, so this shows representational capacity, not that people behave so.
* **Where it does not:** with a classical context-noise person the classical context weights are better (0.417 against 0.457), and in the
  closed loop the quantum slot does not beat Bayes or the classical slot (success 0.912, 0.916, 0.925). Two noisy answers were not worth
  their question cost over one.
* Real evidence needs answers from real people (or a robot's real users); that is the open step.

## The three strands in one system: `QuantumRobotVLA`

| Slot | Strand | Quantum component | Classical rival it is measured against |
|---|---|---|---|
| belief | cognition | `QuantumSlot` (rotated measurements per answer; Qiskit circuit reproduces it) | `BayesSlot`, `TemperedSlot` |
| state | sensing | `kalman + MEMS + quantum` (slow atom-interferometer accelerometer pins the MEMS bias) | encoder, Kalman with encoder only, Kalman with the nominal motor model |
| assignment | computing | `assign_targets` over the QUBO solvers incl. Qiskit QAOA | exact, annealing, greedy |

The grounding model (`TinyVLA`) is classical; the quantum parts are simulated on a classical machine.

### What the state benchmark found (`benchmarks/results/vla_state/`)

The encoder drops out 1 s into a 3 s move. With the plant as modelled, the classical Kalman filter that uses the known command and the nominal
motor wins by a wide margin (success 0.96 against 0.43 for the quantum-accelerometer fusion). When an unmodelled push acts on the joint, the model-based
filter collapses (0.14 and 0.07) and the quantum-accelerometer fusion is best (0.43 and 0.29, MEMS alone 0.14 and 0.13). So the accelerometer
earns its place only where the dynamics are not known, and the quantum one beats the MEMS one only because its bias is near zero. All simulated, with
illustrative sensor values.

## The motor agent: monitor, diagnose, self-heal (`agent.py`, `benchmarks/results/motor_agent/`)

`MotorAgent` supervises the motor every control step. It estimates the state (encoder + MEMS + quantum accelerometer), checks the encoder
against an accelerometer prediction, fits single-fault models of the motor's acceleration over the last second (gain loss, friction, external push),
self-calibrates the healthy plant online, and heals: encoder faults fall back on a model-based state, plant faults re-linearise the command with the
fitted plant, a dead actuator ends in a safe stop. Faults are injected by `SensedMotor(faults=[...])`.

Closed loop, simulated, one fault at 1.5 s of a 4 s two-leg move, 5 seeds x 80 episodes (success = within a quarter cell at the end):

| Fault | Best no-agent policy | Agent, monitor only | Agent (self-heal) |
|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 |
| encoder dropout | 0.58 | 0.44 | 0.85 |
| encoder offset | 0.00 | 0.00 | 0.86 |
| actuator gain loss | 0.77 | 0.77 | 0.85 |
| friction increase | 0.77 | 0.68 | 0.99 |
| external push | 0.00 | 0.00 | 0.52 |

What this does not show, stated plainly:

* **Pushes are still the weak case:** the diagnosis is right in about half of the episodes (a push against the motion looks like friction until the
  joint stops) and success is 0.52.
* **Gain loss is diagnosed in about half of the episodes** (the plain controller tolerates much of it anyway).
* **The quantum accelerometer adds little inside the agent:** the agent without it scores 0.81 and 0.83 on encoder dropout and offset against 0.85 and
  0.86 with it, and 0.55 against 0.52 on pushes (not retuned for that case).
* A first version of the agent raised false alarms on 4% of healthy episodes and cost 4 points of success there; the encoder check now needs three
  consecutive jumps with an acceleration-dependent threshold (0.3% false alarms). The thresholds were tuned on separate seeds but on this fault family.
* The agent knows the nominal plant, handles five fault kinds one at a time, and its thresholds were tuned on this fault family. Nothing here was run on hardware.
