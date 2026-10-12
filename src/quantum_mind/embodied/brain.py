"""The main brain: a supervisor that keeps a state machine over the robot's sub-agents and may consult a local language model.

Structure
---------
* **Sub-agents** (:class:`MotorSubAgent`, :class:`SensorSubAgent`, :class:`PerceptionSubAgent`, :class:`Heartbeat`) each watch
  one sub-system and report a :class:`Status` with a health level (``OK``, ``WARN``, ``FAULT``, ``HEALING``, ``FAILED``). The
  motor sub-agent wraps :class:`~quantum_mind.embodied.agent.MotorAgent`, which does its own local self-healing.
* **State machine** (:class:`StateMachine`): the robot is in one :class:`Mode` (``BOOT``, ``CALIBRATING``, ``NOMINAL``,
  ``DEGRADED``, ``HOLD``, ``SAFE_STOP``). Only the transitions in :data:`TRANSITIONS` exist; anything else is refused and
  logged.
* **Brain** (:class:`RuleBrain` or :class:`LocalLLMBrain`): when a sub-system reports something other than OK, the brain picks
  one *action* from the set allowed in the current mode. A local LLM can be the brain, but it only proposes: its reply must be
  JSON with an allowed action, and hard rules run first and cannot be overridden (any ``FAILED`` sub-agent means ``SAFE_STOP``;
  ``SAFE_STOP`` is left only by a human ``reset``). An unparsable or disallowed reply falls back to the rule brain and is
  counted.
* **Supervisor** (:class:`Supervisor`): runs the sub-agents every step, fires state-machine events, asks the brain, applies the
  action to the command, and keeps an incident log.

The LLM is optional and runs locally (:func:`ollama_generate` talks to an Ollama server on this machine; any ``str -> str``
callable works). No language model ships with the library and none was run during development: the LLM path is tested with
scripted replies only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class Health(Enum):
    """Health level a sub-agent reports, from fine (``OK``) to out of service (``FAILED``)."""

    OK = 0
    WARN = 1
    HEALING = 2
    FAULT = 3
    FAILED = 4


class Mode(Enum):
    """Operating mode of the robot, the states of the supervisor's state machine."""

    BOOT = 'boot'
    CALIBRATING = 'calibrating'
    NOMINAL = 'nominal'
    DEGRADED = 'degraded'
    HOLD = 'hold'
    SAFE_STOP = 'safe_stop'


#: (mode, event) -> next mode. Any pair not listed is an illegal transition.
TRANSITIONS = {
    (Mode.BOOT, 'start'): Mode.CALIBRATING,
    (Mode.CALIBRATING, 'calibrated'): Mode.NOMINAL,
    (Mode.NOMINAL, 'fault'): Mode.DEGRADED,
    (Mode.DEGRADED, 'recovered'): Mode.NOMINAL,
    (Mode.NOMINAL, 'ambiguity'): Mode.HOLD,
    (Mode.DEGRADED, 'ambiguity'): Mode.HOLD,
    (Mode.HOLD, 'clarified'): Mode.NOMINAL,
    (Mode.HOLD, 'fault'): Mode.DEGRADED,
    (Mode.BOOT, 'failure'): Mode.SAFE_STOP,
    (Mode.CALIBRATING, 'failure'): Mode.SAFE_STOP,
    (Mode.NOMINAL, 'failure'): Mode.SAFE_STOP,
    (Mode.DEGRADED, 'failure'): Mode.SAFE_STOP,
    (Mode.HOLD, 'failure'): Mode.SAFE_STOP,
    (Mode.SAFE_STOP, 'reset'): Mode.BOOT,
}

#: Actions the brain may choose in each mode.
ALLOWED = {
    Mode.BOOT: {'continue'}, Mode.CALIBRATING: {'continue'},
    Mode.NOMINAL: {'continue', 'slow_down', 'safe_stop'},
    Mode.DEGRADED: {'continue', 'slow_down', 'hold', 'safe_stop'},
    Mode.HOLD: {'hold', 'resume', 'safe_stop'},
    Mode.SAFE_STOP: {'stay'},
}


@dataclass
class Status:
    name: str
    health: Health
    message: str = ''
    data: dict = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    reason: str = ''
    source: str = 'rules'


class StateMachine:
    def __init__(self):
        self.mode, self.log, self.refused = Mode.BOOT, [], []

    def fire(self, event, t=0.0):
        nxt = TRANSITIONS.get((self.mode, event))
        if nxt is None:
            self.refused.append((round(t, 2), self.mode.value, event))
            return False
        self.log.append((round(t, 2), self.mode.value, event, nxt.value))
        self.mode = nxt
        return True


# ------------------------------------------------------------------------------------------------------ sub-agents
class MotorSubAgent:
    """Reports the motor agent's view: healing a fault, failed (safe stop) or fine."""
    name = 'motor'

    def update(self, ctx):
        a = ctx['agent']
        plant = sorted(a.flags & {'actuator_degraded', 'friction_increase', 'external_push'})
        if a.safe_stop:
            return Status(self.name, Health.FAILED, 'actuator unrecoverable', {'flags': sorted(a.flags)})
        if plant or not a.enc_ok:
            kind = a.label if plant and a.label else (a.final_diagnosis or 'fault')
            return Status(self.name, Health.HEALING if a.heal else Health.FAULT, 'diagnosed %s' % kind, {'flags': sorted(a.flags)})
        return Status(self.name, Health.OK, 'nominal')


class SensorSubAgent:
    """Encoder, MEMS and quantum accelerometer health."""
    name = 'sensors'

    def __init__(self, quantum_timeout=3.0, bias_limit=0.15):
        self.quantum_timeout, self.bias_limit, self.last_q = quantum_timeout, bias_limit, 0.0

    def update(self, ctx):
        a, t = ctx['agent'], ctx['t']
        if ctx.get('a_q') is not None:
            self.last_q = t
        if not a.enc_ok:
            return Status(self.name, Health.FAULT, 'encoder unusable', {'encoder': False})
        bias = abs(a.est.x[2]) * a.s
        if bias > self.bias_limit:
            return Status(self.name, Health.WARN, 'MEMS bias %.2f m/s^2 is large' % bias, {'bias': float(bias)})
        if a.quantum and t - self.last_q > self.quantum_timeout:
            return Status(self.name, Health.WARN, 'no quantum accelerometer sample for %.1f s' % (t - self.last_q))
        return Status(self.name, Health.OK, 'nominal')


class PerceptionSubAgent:
    """Grounding confidence: WARN while the instruction is ambiguous (the gate would ask)."""
    name = 'perception'

    def update(self, ctx):
        if ctx.get('ambiguous'):
            return Status(self.name, Health.WARN, 'instruction ambiguous: asking the person', {})
        return Status(self.name, Health.OK, 'nominal')


class Heartbeat:
    """Watchdog: FAILED if no telemetry arrived within ``timeout`` seconds."""
    name = 'heartbeat'

    def __init__(self, timeout=0.5):
        self.timeout, self.last = timeout, 0.0

    def update(self, ctx):
        if ctx.get('fresh', True):
            self.last = ctx['t']
        if ctx['t'] - self.last > self.timeout:
            return Status(self.name, Health.FAILED, 'telemetry lost for %.2f s' % (ctx['t'] - self.last))
        return Status(self.name, Health.OK, 'alive')


# ------------------------------------------------------------------------------------------------------ brains
class RuleBrain:
    """Deterministic policy; also the fallback when the LLM misbehaves."""
    source = 'rules'

    def decide(self, mode, statuses, history=()):
        if mode == Mode.SAFE_STOP:
            return Decision('stay', 'safe stop is left only by a human reset')
        worst = max((s.health.value for s in statuses), default=0)
        if mode == Mode.HOLD:
            waiting = any(s.name == 'perception' and s.health != Health.OK for s in statuses)
            return Decision('hold', 'waiting for the clarification') if waiting else Decision('resume', 'the instruction is clear again')
        if worst >= Health.FAULT.value:
            return Decision('hold', 'a sub-system fault is not being healed')
        if worst == Health.HEALING.value:
            return Decision('slow_down', 'a fault is being healed; move at reduced command')
        return Decision('continue', 'all sub-systems nominal' if worst == 0 else 'warnings only')


def _extract_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object in the reply')
    return json.loads(m.group(0))


class LocalLLMBrain:
    """Ask a local language model to choose the action. ``generate(prompt) -> str`` is any local text generator.

    The prompt carries the mode, every status and the allowed actions; the reply must contain ``{"action": ..., "reason": ...}``.
    A reply that is not JSON, or names an action that is not allowed in this mode, is discarded and the rule brain decides
    (counted in ``stats``).
    """
    source = 'llm'

    def __init__(self, generate, fallback=None):
        self.generate, self.fallback = generate, fallback or RuleBrain()
        self.stats = dict(calls=0, accepted=0, invalid_json=0, disallowed=0, errors=0)
        self.last_prompt = self.last_reply = None

    def prompt(self, mode, statuses, history):
        body = dict(mode=mode.value, allowed_actions=sorted(ALLOWED[mode]),
                    sub_systems=[dict(name=s.name, health=s.health.name, message=s.message) for s in statuses],
                    recent_events=list(history)[-5:])
        return ('You supervise a robot motor. Reply with one JSON object {"action": <one of allowed_actions>, '
                '"reason": <short text>} and nothing else.\n' + json.dumps(body))

    def decide(self, mode, statuses, history=()):
        self.stats['calls'] += 1
        self.last_prompt = self.prompt(mode, statuses, history)
        try:
            self.last_reply = self.generate(self.last_prompt)
            out = _extract_json(self.last_reply)
        except Exception:
            self.stats['errors' if self.last_reply is None else 'invalid_json'] += 1
            return self.fallback.decide(mode, statuses, history)
        action = str(out.get('action', ''))
        if action not in ALLOWED[mode]:
            self.stats['disallowed'] += 1
            return self.fallback.decide(mode, statuses, history)
        self.stats['accepted'] += 1
        return Decision(action, str(out.get('reason', '')), 'llm')


def ollama_generate(model='llama3.2', host='http://localhost:11434', timeout=30.0):
    """A ``generate`` function for a local Ollama server (needs Ollama running on this machine; not exercised in the test suite)."""
    import urllib.request

    def generate(prompt):
        payload = json.dumps(dict(model=model, prompt=prompt, stream=False, format='json')).encode()
        req = urllib.request.Request(host + '/api/generate', payload, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:                  # noqa: S310 - local server
            return json.loads(r.read())['response']
    return generate


# ------------------------------------------------------------------------------------------------------ supervisor
class Supervisor:
    """The main brain's loop: sub-agent statuses -> state-machine events -> brain decision -> command limits."""

    def __init__(self, subagents=None, brain=None, calibrate_steps=20, recover_steps=40, dt=0.05, slow_factor=0.5):
        self.subs = subagents or [MotorSubAgent(), SensorSubAgent(), Heartbeat()]
        self.brain, self.sm = brain or RuleBrain(), StateMachine()
        self.cal_steps, self.rec_steps, self.dt, self.slow = calibrate_steps, recover_steps, dt, slow_factor
        self.reset()

    def reset(self):
        self.sm = StateMachine()
        self.k, self.ok_run, self.action, self.statuses = 0, 0, 'continue', []
        self.incidents, self.decisions, self._last_key = [], [], None
        self.sm.fire('start', 0.0)

    @property
    def mode(self):
        return self.sm.mode

    def human_reset(self):
        """The only way out of SAFE_STOP (an operator has inspected the machine)."""
        if self.sm.fire('reset', self.k * self.dt):
            self.reset()

    def observe(self, ctx):
        """Run one supervision step with the telemetry context; returns the current mode."""
        self.k += 1
        t = ctx['t']
        self.statuses = [s.update(ctx) for s in self.subs]
        worst = max(s.health.value for s in self.statuses)
        if self.mode == Mode.SAFE_STOP:
            self.action = 'stay'
            return self.mode
        if worst >= Health.FAILED.value:                                         # hard rule: not negotiable, the LLM is not asked
            self.sm.fire('failure', t)
            self.action = 'safe_stop'
            self._note(t, 'safe_stop', 'hard rule: ' + '; '.join(s.message for s in self.statuses if s.health == Health.FAILED), 'rules')
            return self.mode
        perception_warn = any(s.name == 'perception' and s.health != Health.OK for s in self.statuses)
        fault = any(s.health in (Health.HEALING, Health.FAULT) for s in self.statuses)
        self.ok_run = self.ok_run + 1 if worst == 0 else 0
        if self.mode == Mode.CALIBRATING and self.k >= self.cal_steps and not fault:
            self.sm.fire('calibrated', t)
        elif self.mode == Mode.NOMINAL and fault:
            self.sm.fire('fault', t)
        elif self.mode == Mode.HOLD and not perception_warn:
            self.sm.fire('clarified', t)
            if fault:
                self.sm.fire('fault', t)
        elif self.mode == Mode.DEGRADED and self.ok_run >= self.rec_steps:
            self.sm.fire('recovered', t)
        if perception_warn and self.mode in (Mode.NOMINAL, Mode.DEGRADED):
            self.sm.fire('ambiguity', t)
        key = (self.mode, tuple((s.name, s.health) for s in self.statuses))
        if worst > 0 or self.mode == Mode.HOLD:                                  # consult the brain only when something is not OK
            if key != self._last_key:
                d = self.brain.decide(self.mode, self.statuses, [e[1:] for e in self.incidents])
                self._note(t, d.action, d.reason, d.source)
                self.action = d.action
                if d.action == 'safe_stop':
                    self.sm.fire('failure', t)
                elif d.action == 'resume' and self.mode == Mode.HOLD:
                    self.sm.fire('clarified', t)
        else:
            self.action = 'continue'
        self._last_key = key
        return self.mode

    def _note(self, t, action, reason, source):
        self.decisions.append((round(t, 2), action, source))
        self.incidents.append((round(t, 2), action, '[%s] %s' % (source, reason)))

    def limit(self, u):
        """Apply the current action to a motor command."""
        if self.mode in (Mode.SAFE_STOP, Mode.HOLD) or self.action in ('hold', 'safe_stop', 'stay'):
            return 0.0
        if self.action == 'slow_down':
            return float(u) * self.slow
        return float(u)

    def report(self):
        lines = ['mode: %s' % self.mode.value]
        lines += ['  %-10s %-8s %s' % (s.name, s.health.name, s.message) for s in self.statuses]
        lines += ['incident at %.2fs: %s (%s)' % i for i in self.incidents[-5:]]
        return '\n'.join(lines)


def run_supervised_episode(act, rng, faults=(), brain=None, steps=100, switch_at=30, tol=0.0625, heal=True, dropped_telemetry_at=None):
    """Two-leg move under the main brain. Returns the final error, the mode trace and the supervisor.

    ``dropped_telemetry_at`` (seconds) stops telemetry from arriving from then on, which the heartbeat must catch."""
    from .agent import MotorAgent
    from .state import SensedMotor
    sm = SensedMotor(faults=faults)
    agent = MotorAgent(heal=heal)
    sup = Supervisor(brain=brain)
    a, b = rng.uniform(-0.9, 0.9, 2)
    z0 = sm.reset(rng.uniform(-1, 1), rng)
    agent.reset(z0)
    th, om, trace = z0, 0.0, []
    for k in range(steps):
        tgt = a if k < switch_at else b
        u = sup.limit(agent.control(float(act(tgt, th, om))))
        meas = sm.step(u)
        fresh = dropped_telemetry_at is None or sm.t < dropped_telemetry_at
        if fresh:
            th, om = agent.step(meas, u)
        sup.observe(dict(t=sm.t, agent=agent, a_q=meas['a_q'], fresh=fresh))
        trace.append(sup.mode.value)
    err = abs(sm.m.theta - b)
    return dict(error=err, success=err < tol, trace=trace, supervisor=sup)
