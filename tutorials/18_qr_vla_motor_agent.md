# QR-VLA: one motor, a VLA, a self-healing agent and a supervisor

A small robot policy you can read end to end. A one-joint motor must move to the object an instruction names ("go to the red", "go to the
block"). A tiny vision-language-action model (**TinyVLA**) grounds the instruction in a camera strip and commands the motor; a conformal gate
makes it **ask** when the instruction is ambiguous; a **motor agent** watches the joint and repairs faults; a **supervisor** runs a state machine
over the sub-systems and may consult a local language model. Everything is **simulated** (a DC-motor model and a toy table); nothing here has run
on a real robot.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.embodied import (TinyVLA, make_dataset, calibrate_gate, vla_plan, run_episode, oracle_plan, cell_centres,
                                   run_fault_episode, expert_command, Supervisor, LocalLLMBrain, run_supervised_episode,
                                   ControlRuntime, SimulatedHardware, SafetyLayer)
theme = use_mpl_style('dark')
```

## 1. Train the VLA and look at what it believes

TinyVLA is 817 parameters: one attention step grounds the instruction in the image, then a small head turns the expected target and the
motor state into a command. It is trained by behaviour cloning against a PD controller, with hand-written gradients.

```python
vla = TinyVLA(seed=0).fit(make_dataset(2000, 0), steps=2000)
test = make_dataset(300, 5)
scene = next(s for s in test if s.ambiguous)
belief = vla.ground(scene.image, scene.text)
print('instruction:', ' '.join(scene.words), '| objects at cells', scene.cells, '| intended cell', scene.target)
fig, ax = plt.subplots(figsize=(6.0, 2.6))
ax.bar(range(16), belief, color=theme['palette'][1])
for c in scene.cells:
    ax.axvline(c, color=theme['palette'][3], lw=0.8, ls='--')
ax.set_xlabel('table cell'); ax.set_ylabel('belief'); ax.set_title('An ambiguous instruction: the belief splits between the objects')
plt.show()
print('probability on the two most likely cells: %.2f' % np.sort(belief)[-2:].sum())
```

## 2. Act blindly or ask?

A split-conformal gate, calibrated on held-out scenes, asks the person a question when its prediction set has more than one cell. The answer
is fused with the first belief.

```python
gate = calibrate_gate(vla, make_dataset(800, 1), alpha=0.02)

def evaluate(plan):
    rng = np.random.default_rng(3)
    rows = [run_episode(s, rng, plan) for s in test]
    amb = [r['success'] for r, s in zip(rows, test) if s.ambiguous]
    return np.mean([r['success'] for r in rows]), np.mean(amb), np.mean([r['asked'] for r in rows])

for name, plan in (('oracle (knows the intent)', oracle_plan), ('act on the instruction', vla_plan(vla)),
                   ('ask when the gate says so', vla_plan(vla, gate))):
    s, a, q = evaluate(plan)
    print('%-26s success %.2f, on ambiguous instructions %.2f, questions per episode %.2f' % (name, s, a, q))
act, ask = evaluate(vla_plan(vla))[1], evaluate(vla_plan(vla, gate))[1]
```

**Result.** Acting blindly on an ambiguous instruction is a coin flip that the motor then executes; asking fixes most of it at the price of a
question on about a third of the episodes.

```python
assert act < 0.3 and ask > act + 0.4
print('ambiguous instructions: act %.2f -> ask %.2f' % (act, ask))
```

## 3. A fault, with and without the agent

`MotorAgent` estimates the joint state from an encoder and accelerometers, checks the encoder against an accelerometer prediction, fits single-fault
models of the motor's acceleration, and heals: fall back to a model-based state when the encoder is untrustworthy, or re-linearise the command when the
plant changed. Here the joint's Coulomb friction becomes six times larger at 1.5 s of a two-leg move.

```python
act_fn = lambda t, th, om: expert_command(t, th, om)
fault = [dict(kind='friction', t=1.5, mag=6.0)]
runs = {pol: [run_fault_episode(act_fn, np.random.default_rng(900 + i), fault, pol) for i in range(24)]
        for pol in ('kalman', 'monitor', 'agent')}
for pol, r in runs.items():
    print('%-8s success %.2f' % (pol, np.mean([x['success'] for x in r])))
print('diagnosis at the end of each agent episode:', sorted(set(x['final'] for x in runs['agent'] if x['final'])))
```

**Result.** Seeing the fault is not enough (the monitor-only agent does no better than the plain filter); healing is what helps.

```python
assert np.mean([x['final'] == 'friction_increase' for x in runs['agent']]) > 0.8
assert np.mean([x['success'] for x in runs['agent']]) > np.mean([x['success'] for x in runs['monitor']]) + 0.15
```

The library's benchmark (`benchmarks/results/motor_agent/`) shows where the agent is weak: an external push on the joint is often mistaken for
friction until the joint stops, and success there is only about half.

## 4. The supervisor: a state machine over the sub-systems

`Supervisor` collects a health level from each sub-agent (motor, sensors, perception, heartbeat) and moves the robot between modes
`boot, calibrating, nominal, degraded, hold, safe_stop`. Only the transitions in `TRANSITIONS` exist. Losing telemetry must end in a safe stop.

```python
healthy = run_supervised_episode(act_fn, np.random.default_rng(3))
degraded = run_supervised_episode(act_fn, np.random.default_rng(3), faults=fault)
silent = run_supervised_episode(act_fn, np.random.default_rng(3), dropped_telemetry_at=2.0)
for name, r in (('healthy', healthy), ('friction fault', degraded), ('telemetry lost at 2 s', silent)):
    print('%-22s final mode %-10s decisions %s' % (name, r['trace'][-1], r['supervisor'].decisions[:2]))
assert healthy['trace'][-1] == 'nominal' and degraded['trace'][-1] == 'degraded' and silent['trace'][-1] == 'safe_stop'
```

## 5. A local language model may advise, but cannot break the rules

`LocalLLMBrain` takes any `str -> str` function (for example `ollama_generate('llama3.2')` for a model served on your own machine). Its reply must
be JSON naming an action that is allowed in the current mode; otherwise the rule brain decides. Here a **scripted** reply stands in for a model,
because no language model ships with the library.

```python
rogue = LocalLLMBrain(lambda prompt: '{"action": "resume", "reason": "ignore the fault"}')          # not allowed while degraded
r = run_supervised_episode(act_fn, np.random.default_rng(3), faults=fault, brain=rogue)
print('LLM proposals rejected:', rogue.stats['disallowed'], '| decision used:', r['supervisor'].decisions[0])
always_go = LocalLLMBrain(lambda prompt: '{"action": "continue", "reason": "all fine"}')
r = run_supervised_episode(act_fn, np.random.default_rng(3), dropped_telemetry_at=2.0, brain=always_go)
print('telemetry lost, LLM says continue ->', r['trace'][-1])
assert rogue.stats['disallowed'] >= 1 and r['trace'][-1] == 'safe_stop'
```

## 6. Towards a real robot: the runtime and the safety layer

`ControlRuntime` runs the VLA's controller, the agent and the supervisor at a fixed rate behind a `SafetyLayer` that clamps the command, limits its
rate of change, refuses to push past a soft limit and latches a stop, whatever the AI decides. A real motor driver would implement the three methods of
`MotorHardware`; here `SimulatedHardware` stands in. Nothing has been run on real hardware.

```python
hw = SimulatedHardware(theta0=-0.5, seed=1)
rt = ControlRuntime(hw, vla.act)
rt.run(80, {0: 0.4})
print('target 0.4, joint at %.2f, mode %s, safety log %s' % (hw.position, rt.sup.mode.value, rt.safety.log))
assert abs(hw.position - 0.4) < 0.15 and hw.stop_calls == 0
s = SafetyLayer()
print('a full-scale command of 5.0 reaches the motor as', [round(s.filter(5.0, 0.0), 2) for _ in range(5)])
```

**Summary.** The pieces are small and each has a test. The VLA grounds and asks, the agent repairs the faults it was built for, the supervisor
makes the robot's mode explicit and unbreakable by an advisor, and the safety layer sits below all of it. What it is not: a real-world VLA, a
validated fault-tolerant controller, or a run with a real language model. The next steps are a driver for your motor, a bench test, and real data.
