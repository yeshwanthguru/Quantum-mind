# `quantum_mind.embodied`: a one-motor vision-language-action prototype

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
